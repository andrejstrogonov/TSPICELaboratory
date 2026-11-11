"""Experimental U-Net surrogates and reproducible synthetic physics examples."""
"""TODO: Добавить Graph Networks  измерить все на метриках"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

MODEL_SPECS = {
    "poisson": {
        "label": "Электростатика · уравнение Пуассона",
        "in_channels": 3,
        "out_channels": 1,
        "output_names": ("potential_V",),
    },
    "magnetostatic": {
        "label": "Магнитостатика · потенциал Az",
        "in_channels": 3,
        "out_channels": 1,
        "output_names": ("vector_potential_Az_normalized",),
    },
    "carriers": {
        "label": "Носители · равновесные n/p",
        "in_channels": 4,
        "out_channels": 2,
        "output_names": ("log10_electron_density_norm", "log10_hole_density_norm"),
    },
}


class BaseUNet(nn.Module):
    """Small 2D U-Net used by the three independently trained surrogate models."""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.enc1 = self._block(in_channels, 16)
        self.enc2 = self._block(16, 32)
        self.pool = nn.MaxPool2d(2)
        self.bottleneck = self._block(32, 64)
        self.up2 = nn.ConvTranspose2d(64, 32, 2, stride=2)
        self.dec2 = self._block(64, 32)
        self.up1 = nn.ConvTranspose2d(32, 16, 2, stride=2)
        self.dec1 = self._block(32, 16)
        self.final_conv = nn.Conv2d(16, out_channels, 1)

    @staticmethod
    def _block(in_channels: int, out_channels: int) -> nn.Sequential:
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        if inputs.shape[-2] % 4 or inputs.shape[-1] % 4:
            raise ValueError("U-Net input height and width must be multiples of 4.")
        enc1 = self.enc1(inputs)
        enc2 = self.enc2(self.pool(enc1))
        bottleneck = self.bottleneck(self.pool(enc2))
        dec2 = self.dec2(torch.cat((self.up2(bottleneck), enc2), dim=1))
        dec1 = self.dec1(torch.cat((self.up1(dec2), enc1), dim=1))
        return self.final_conv(dec1)


def _harmonic_mean(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    return 2 * left * right / np.maximum(left + right, 1e-12)


def _solve_poisson(epsilon: np.ndarray, boundary: np.ndarray, source: np.ndarray,
                   *, variable_epsilon: bool) -> np.ndarray:
    height, width = epsilon.shape
    potential = boundary.copy()
    fixed = np.zeros_like(potential, dtype=bool)
    fixed[[0, -1], :] = True
    fixed[:, [0, -1]] = True
    dx2 = 1.0 / max(width - 1, 1) ** 2
    dy2 = 1.0 / max(height - 1, 1) ** 2
    for _ in range(500):
        center = epsilon[1:-1, 1:-1] if variable_epsilon else 1.0
        west = (
            _harmonic_mean(center, epsilon[1:-1, :-2])
            if variable_epsilon else 1.0
        )
        east = (
            _harmonic_mean(center, epsilon[1:-1, 2:])
            if variable_epsilon else 1.0
        )
        north = (
            _harmonic_mean(center, epsilon[:-2, 1:-1])
            if variable_epsilon else 1.0
        )
        south = (
            _harmonic_mean(center, epsilon[2:, 1:-1])
            if variable_epsilon else 1.0
        )
        weights = (west + east) * dx2 + (north + south) * dy2
        updated = (
            (west * potential[1:-1, :-2] + east * potential[1:-1, 2:]) * dx2
            + (north * potential[:-2, 1:-1] + south * potential[2:, 1:-1]) * dy2
            + source[1:-1, 1:-1] * dx2 * dy2
        ) / np.maximum(weights, 1e-12)
        if float(np.max(np.abs(updated - potential[1:-1, 1:-1]))) < 1e-5:
            potential[1:-1, 1:-1] = updated
            break
        potential[1:-1, 1:-1] = updated
        potential[fixed] = boundary[fixed]
    return potential


def create_test_example(
    model_name: str, *, size: int = 32, seed: int = 0
) -> tuple[np.ndarray, np.ndarray]:
    """Create one deterministic, dimensionless / normalized teaching example."""
    if model_name not in MODEL_SPECS:
        raise ValueError(f"Unknown physics model: {model_name}")
    if size < 16 or size % 4:
        raise ValueError("Grid size must be a multiple of 4 and at least 16.")
    rng = np.random.default_rng(seed)
    coordinates = np.linspace(0.0, 1.0, size, dtype=np.float32)
    x, y = np.meshgrid(coordinates, coordinates)

    if model_name == "poisson":
        epsilon = rng.choice((1.0, 3.9, 11.7), size=(size, size)).astype(np.float32)
        if rng.random() < 0.8:
            epsilon[:, size // 3 : 2 * size // 3] = rng.choice((3.9, 11.7))
        boundary = np.zeros((size, size), dtype=np.float32)
        boundary[0, :] = rng.uniform(-1, 1)
        boundary[-1, :] = rng.uniform(-1, 1)
        boundary[:, 0] = rng.uniform(-1, 1)
        boundary[:, -1] = rng.uniform(-1, 1)
        boundary_mask = np.zeros_like(boundary)
        boundary_mask[[0, -1], :] = 1
        boundary_mask[:, [0, -1]] = 1
        source = np.zeros((size, size), dtype=np.float32)
        target = _solve_poisson(epsilon, boundary, source, variable_epsilon=True)
        inputs = np.stack((epsilon / 11.7, boundary, boundary_mask))
        return inputs.astype(np.float32), target[None].astype(np.float32)

    if model_name == "magnetostatic":
        conductor = np.zeros((size, size), dtype=np.float32)
        x0, x1 = sorted(rng.integers(size // 4, 3 * size // 4, size=2))
        y0, y1 = sorted(rng.integers(size // 4, 3 * size // 4, size=2))
        conductor[y0:max(y0 + 1, y1), x0:max(x0 + 1, x1)] = 1.0
        current_density = conductor * rng.choice((-1.0, 1.0))
        boundary = np.zeros((size, size), dtype=np.float32)
        boundary[[0, -1], :] = rng.uniform(-0.25, 0.25, size=(2, size))
        boundary[:, [0, -1]] = rng.uniform(-0.25, 0.25, size=(size, 2))
        target = _solve_poisson(
            np.ones_like(conductor), boundary, current_density, variable_epsilon=False
        )
        scale = max(float(np.max(np.abs(target))), 1e-6)
        inputs = np.stack((conductor, boundary, current_density))
        return inputs.astype(np.float32), (target / scale)[None].astype(np.float32)

    donors = rng.uniform(0, 1, size=(size, size)).astype(np.float32)
    acceptors = rng.uniform(0, 1, size=(size, size)).astype(np.float32)
    donors[donors < 0.65] *= 0.05
    acceptors[acceptors < 0.65] *= 0.05
    potential = (
        rng.uniform(-0.15, 0.15)
        + rng.uniform(-0.1, 0.1) * (x - 0.5)
        + rng.uniform(-0.1, 0.1) * (y - 0.5)
    ).astype(np.float32)
    semiconductor = np.zeros((size, size), dtype=np.float32)
    y0, y1 = sorted(rng.integers(size // 4, 3 * size // 4, size=2))
    semiconductor[y0:max(y0 + 1, y1), 1:-1] = 1.0
    net_doping = (donors - acceptors) * 5.0
    ni = 1e-3
    equilibrium_n = 0.5 * (net_doping + np.sqrt(net_doping**2 + 4 * ni**2))
    equilibrium_p = ni**2 / np.maximum(equilibrium_n, 1e-8)
    voltage_thermal = 0.02585
    electrons = equilibrium_n * np.exp(np.clip(potential / voltage_thermal, -8, 8))
    holes = equilibrium_p * np.exp(np.clip(-potential / voltage_thermal, -8, 8))
    electrons = np.where(semiconductor > 0, electrons, 1e-8)
    holes = np.where(semiconductor > 0, holes, 1e-8)
    target = np.stack(
        (
            np.log10(electrons.clip(1e-8, 1e8)) / 8,
            np.log10(holes.clip(1e-8, 1e8)) / 8,
        )
    )
    inputs = np.stack((donors, acceptors, potential, semiconductor))
    return inputs.astype(np.float32), target.astype(np.float32)


def generate_test_examples(
    directory: str | Path, *, count: int = 3, size: int = 32, seed: int = 2026
) -> list[Path]:
    """Save reproducible input/target pairs for all three model families."""
    if count < 1:
        raise ValueError("Example count must be at least one.")
    output_dir = Path(directory)
    output_dir.mkdir(parents=True, exist_ok=True)
    saved = []
    for model_name, spec in MODEL_SPECS.items():
        inputs, targets = zip(
            *(
                create_test_example(model_name, size=size, seed=seed + index)
                for index in range(count)
            )
        )
        path = output_dir / f"{model_name}_examples.npz"
        np.savez_compressed(
            path,
            inputs=np.stack(inputs),
            targets=np.stack(targets),
            model_name=np.asarray(model_name),
            output_names=np.asarray(spec["output_names"]),
        )
        saved.append(path)
    return saved


def _resolve_device(device_name: str) -> torch.device:
    if device_name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested, but no CUDA-enabled PyTorch GPU is available.")
    if device_name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device_name not in ("cpu", "cuda"):
        raise ValueError("Device must be 'auto', 'cpu', or 'cuda'.")
    return torch.device(device_name)


def _configure_nvidia(device: torch.device) -> bool:
    use_amp = device.type == "cuda"
    if use_amp:
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cuda.matmul.allow_tf32 = True
    return use_amp


def train_model(
    model_name: str,
    *,
    samples: int = 64,
    epochs: int = 20,
    batch_size: int = 8,
    size: int = 32,
    seed: int = 2026,
    checkpoint_path: str | Path,
    device_name: str = "auto",
) -> dict[str, object]:
    if model_name not in MODEL_SPECS:
        raise ValueError(f"Unknown physics model: {model_name}")
    if samples < 2 or epochs < 1 or batch_size < 1:
        raise ValueError("Training requires >=2 samples, >=1 epoch, and >=1 batch item.")
    device = _resolve_device(device_name)
    use_amp = _configure_nvidia(device)
    torch.manual_seed(seed)
    examples = [
        create_test_example(model_name, size=size, seed=seed + index)
        for index in range(samples)
    ]
    inputs = torch.from_numpy(np.stack([item[0] for item in examples]))
    targets = torch.from_numpy(np.stack([item[1] for item in examples]))
    dataset = TensorDataset(inputs, targets)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        pin_memory=use_amp,
        num_workers=0,
    )
    spec = MODEL_SPECS[model_name]
    memory_format = torch.channels_last if use_amp else torch.contiguous_format
    model = BaseUNet(spec["in_channels"], spec["out_channels"]).to(
        device=device, memory_format=memory_format
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    final_loss = math.inf
    for _ in range(epochs):
        model.train()
        losses = []
        for batch_inputs, batch_targets in loader:
            batch_inputs = batch_inputs.to(
                device, non_blocking=use_amp, memory_format=memory_format
            )
            batch_targets = batch_targets.to(device, non_blocking=use_amp)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(
                device_type=device.type, dtype=torch.float16, enabled=use_amp
            ):
                prediction = model(batch_inputs)
                loss = nn.functional.mse_loss(prediction.float(), batch_targets.float())
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            losses.append(float(loss.detach().cpu()))
        final_loss = float(np.mean(losses))

    checkpoint = Path(checkpoint_path)
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_name": model_name,
            "in_channels": spec["in_channels"],
            "out_channels": spec["out_channels"],
            "state_dict": model.cpu().state_dict(),
            "train_loss": final_loss,
            "training_samples": samples,
            "epochs": epochs,
            "grid_size": size,
        },
        checkpoint,
    )
    return {
        "checkpoint": str(checkpoint),
        "train_loss": final_loss,
        "device": (
            f"CUDA: {torch.cuda.get_device_name(device)}" if use_amp else "CPU"
        ),
    }


def predict(
    model_name: str,
    checkpoint_path: str | Path,
    inputs: np.ndarray,
    *,
    device_name: str = "auto",
) -> tuple[np.ndarray, str]:
    if model_name not in MODEL_SPECS:
        raise ValueError(f"Unknown physics model: {model_name}")
    if inputs.ndim != 4 or inputs.shape[1] != MODEL_SPECS[model_name]["in_channels"]:
        raise ValueError("Input must have NCHW shape and the selected model's channel count.")
    device = _resolve_device(device_name)
    use_amp = _configure_nvidia(device)
    checkpoint = torch.load(
        checkpoint_path, map_location="cpu", weights_only=True
    )
    if checkpoint.get("model_name") != model_name:
        raise ValueError("Checkpoint does not match the selected physics model.")
    spec = MODEL_SPECS[model_name]
    if (
        checkpoint.get("in_channels") != spec["in_channels"]
        or checkpoint.get("out_channels") != spec["out_channels"]
    ):
        raise ValueError("Checkpoint channel metadata is incompatible.")
    memory_format = torch.channels_last if use_amp else torch.contiguous_format
    model = BaseUNet(spec["in_channels"], spec["out_channels"]).to(
        device=device, memory_format=memory_format
    )
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    tensor = torch.from_numpy(np.asarray(inputs, dtype=np.float32)).to(
        device, memory_format=memory_format
    )
    with torch.inference_mode(), torch.autocast(
        device_type=device.type, dtype=torch.float16, enabled=use_amp
    ):
        result = model(tensor).float().cpu().numpy()
    backend = f"CUDA: {torch.cuda.get_device_name(device)}" if use_amp else "CPU"
    return result, backend


def export_onnx(
    model_name: str, checkpoint_path: str | Path, output_path: str | Path
) -> Path:
    if model_name not in MODEL_SPECS:
        raise ValueError(f"Unknown physics model: {model_name}")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if checkpoint.get("model_name") != model_name:
        raise ValueError("Checkpoint does not match the selected physics model.")
    spec = MODEL_SPECS[model_name]
    model = BaseUNet(spec["in_channels"], spec["out_channels"])
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        model,
        torch.zeros(1, spec["in_channels"], 32, 32),
        path,
        input_names=["input_features"],
        output_names=["output_fields"],
        dynamic_axes={
            "input_features": {0: "batch", 2: "height", 3: "width"},
            "output_fields": {0: "batch", 2: "height", 3: "width"},
        },
        opset_version=17,
    )
    import onnx

    onnx.checker.check_model(onnx.load(path))
    return path


def _main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    examples = commands.add_parser("examples", help="generate synthetic test examples")
    examples.add_argument("--directory", default="data/neural_examples")
    examples.add_argument("--count", type=int, default=3)
    examples.add_argument("--size", type=int, default=32)

    training = commands.add_parser("train", help="train one independent physics model")
    training.add_argument("model", choices=(*MODEL_SPECS, "all"))
    training.add_argument("--samples", type=int, default=64)
    training.add_argument("--epochs", type=int, default=20)
    training.add_argument("--batch-size", type=int, default=8)
    training.add_argument("--size", type=int, default=32)
    training.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    training.add_argument("--directory", default="data/neural_models")

    inference = commands.add_parser("infer", help="run a trained model on a test example")
    inference.add_argument("model", choices=tuple(MODEL_SPECS))
    inference.add_argument("--checkpoint", required=True)
    inference.add_argument("--output", default="data/neural_results/prediction.npz")
    inference.add_argument("--size", type=int, default=32)
    inference.add_argument("--seed", type=int, default=99)
    inference.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")

    onnx_command = commands.add_parser("export-onnx", help="export a trained checkpoint")
    onnx_command.add_argument("model", choices=tuple(MODEL_SPECS))
    onnx_command.add_argument("--checkpoint", required=True)
    onnx_command.add_argument("--output", required=True)
    args = parser.parse_args()

    if args.command == "examples":
        for path in generate_test_examples(
            args.directory, count=args.count, size=args.size
        ):
            print(path)
    elif args.command == "train":
        models = tuple(MODEL_SPECS) if args.model == "all" else (args.model,)
        for model_name in models:
            result = train_model(
                model_name,
                samples=args.samples,
                epochs=args.epochs,
                batch_size=args.batch_size,
                size=args.size,
                checkpoint_path=Path(args.directory) / f"{model_name}.pt",
                device_name=args.device,
            )
            print(json.dumps({"model": model_name, **result}, ensure_ascii=False))
    elif args.command == "infer":
        inputs, _ = create_test_example(args.model, size=args.size, seed=args.seed)
        prediction, backend = predict(
            args.model, args.checkpoint, inputs[None], device_name=args.device
        )
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            output,
            inputs=inputs[None],
            prediction=prediction,
            output_names=np.asarray(MODEL_SPECS[args.model]["output_names"]),
        )
        print(f"{output} · {backend} · shape={prediction.shape}")
    else:
        print(export_onnx(args.model, args.checkpoint, args.output))


if __name__ == "__main__":
    _main()
