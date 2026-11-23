"""
A-7 · Нейросетевой суррогат U-Net (3 независимые физические сети:
poisson, magnetostatic, carriers) — синтетика -> обучение -> инференс ->
warm start для физического решателя -> экспорт в ONNX (opset 17).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

ModelName = Literal["poisson", "magnetostatic", "carriers"]


@dataclass(frozen=True)
class ModelSpec:
    name: ModelName
    in_channels: int
    out_channels: int
    input_names: tuple[str, ...]
    output_names: tuple[str, ...]
    description: str


MODEL_SPECS: dict[ModelName, ModelSpec] = {
    "poisson": ModelSpec(
        name="poisson",
        in_channels=3,
        out_channels=1,
        input_names=("eps_norm", "psi_boundary", "boundary_mask"),
        output_names=("potential_V",),
        description="Электростатический потенциал с переменным диэлектриком",
    ),
    "magnetostatic": ModelSpec(
        name="magnetostatic",
        in_channels=3,
        out_channels=1,
        input_names=("conductor", "psi_boundary", "J_z"),
        output_names=("Az_norm",),
        description="Магнитный векторный потенциал Az (скалярное приближение)",
    ),
    "carriers": ModelSpec(
        name="carriers",
        in_channels=4,
        out_channels=2,
        input_names=("N_D_log", "N_A_log", "psi", "semiconductor_mask"),
        output_names=("log10_n_div8", "log10_p_div8"),
        description="Концентрации электронов и дырок (log10 / 8)",
    ),
}


def get_model_spec(model_name: str) -> ModelSpec:
    if model_name not in MODEL_SPECS:
        raise ValueError(
            f"Неизвестная модель '{model_name}'. Доступны: {list(MODEL_SPECS.keys())}"
        )
    return MODEL_SPECS[model_name]


# =====================================================================
# Архитектура BaseUNet: enc1(16) -> enc2(32) -> bottleneck(64) -> up2/dec2 -> up1/dec1 -> 1x1
# =====================================================================


class ConvBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.SiLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.SiLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class BaseUNet(nn.Module):
    """
    Компактная 2D U-Net для регулярных сеток произвольного размера, кратного 4.
    Вход:  [B, in_channels, H, W]
    Выход: [B, out_channels, H, W]
    """

    def __init__(self, in_channels: int, out_channels: int, base_channels: int = 16) -> None:
        super().__init__()
        c1 = base_channels        # 16
        c2 = base_channels * 2    # 32
        c3 = base_channels * 4    # 64

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.base_channels = base_channels

        self.enc1 = ConvBlock(in_channels, c1)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.enc2 = ConvBlock(c1, c2)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.bottleneck = ConvBlock(c2, c3)

        self.up2 = nn.ConvTranspose2d(c3, c2, kernel_size=2, stride=2)
        self.dec2 = ConvBlock(c2 + c2, c2)

        self.up1 = nn.ConvTranspose2d(c2, c1, kernel_size=2, stride=2)
        self.dec1 = ConvBlock(c1 + c1, c1)

        self.head = nn.Conv2d(c1, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool1(e1))
        b = self.bottleneck(self.pool2(e2))

        d2 = self.up2(b)
        if d2.shape[-2:] != e2.shape[-2:]:
            d2 = F.interpolate(d2, size=e2.shape[-2:], mode="bilinear", align_corners=False)
        d2 = self.dec2(torch.cat([d2, e2], dim=1))

        d1 = self.up1(d2)
        if d1.shape[-2:] != e1.shape[-2:]:
            d1 = F.interpolate(d1, size=e1.shape[-2:], mode="bilinear", align_corners=False)
        d1 = self.dec1(torch.cat([d1, e1], dim=1))

        return self.head(d1)


def build_unet_for_spec(model_name: ModelName, base_channels: int = 16) -> BaseUNet:
    spec = get_model_spec(model_name)
    return BaseUNet(
        in_channels=spec.in_channels,
        out_channels=spec.out_channels,
        base_channels=base_channels,
    )


# =====================================================================
# 1 · Генерация детерминированных синтетических примеров
# =====================================================================


def _solve_2d_variable_eps_fd(
    eps_rel: np.ndarray,
    psi_bc: np.ndarray,
    bc_mask: np.ndarray,
    rhs: np.ndarray | None = None,
    iters: int = 350,
) -> np.ndarray:
    """
    Быстрый решатель конечных разностей (SOR / Gauss-Seidel) с переменным eps
    для генерации синтетических меток (не является TCAD/измерениями).
    """
    psi = psi_bc.copy().astype(np.float64)
    if rhs is None:
        rhs = np.zeros_like(psi, dtype=np.float64)

    eps_e = 0.5 * (eps_rel[1:-1, 1:-1] + eps_rel[1:-1, 2:])
    eps_w = 0.5 * (eps_rel[1:-1, 1:-1] + eps_rel[1:-1, :-2])
    eps_n = 0.5 * (eps_rel[1:-1, 1:-1] + eps_rel[:-2, 1:-1])
    eps_s = 0.5 * (eps_rel[1:-1, 1:-1] + eps_rel[2:, 1:-1])
    denom = np.maximum(eps_e + eps_w + eps_n + eps_s, 1e-8)

    free = bc_mask[1:-1, 1:-1] < 0.5
    omega = 1.65

    for _ in range(iters):
        psi_new = (
            eps_e * psi[1:-1, 2:]
            + eps_w * psi[1:-1, :-2]
            + eps_n * psi[:-2, 1:-1]
            + eps_s * psi[2:, 1:-1]
            + rhs[1:-1, 1:-1]
        ) / denom
        updated = (1.0 - omega) * psi[1:-1, 1:-1] + omega * psi_new
        psi[1:-1, 1:-1] = np.where(free, updated, psi_bc[1:-1, 1:-1])

    return psi.astype(np.float32)


def create_test_example(
    model_name: ModelName,
    size: int = 64,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Генерирует одну детерминированную синтетическую пару (input [C_in, H, W], target [C_out, H, W]).
    """
    if size < 16 or size % 4 != 0:
        raise ValueError("Размер сетки size должен быть >= 16 и кратен 4.")

    get_model_spec(model_name)
    rng = np.random.default_rng(seed)

    if model_name == "poisson":
        eps_rel = np.full((size, size), 11.7, dtype=np.float32)  # Si
        ox_depth = int(rng.integers(max(3, size // 10), max(5, size // 4)))
        eps_rel[:ox_depth, :] = float(rng.choice([3.9, 9.0, 22.0]))  # SiO2 / Al2O3 / HfO2

        bc_mask = np.zeros((size, size), dtype=np.float32)
        psi_bc = np.zeros((size, size), dtype=np.float32)

        # Нижняя граница — подложка (0 В)
        bc_mask[-1, :] = 1.0
        psi_bc[-1, :] = 0.0

        # Боковые области у поверхности — исток (0 В) и сток (V_d)
        sd_width = max(3, size // 5)
        v_drain = float(rng.uniform(0.05, 1.2))
        bc_mask[: ox_depth + 2, :sd_width] = 1.0
        psi_bc[: ox_depth + 2, :sd_width] = 0.0
        bc_mask[: ox_depth + 2, -sd_width:] = 1.0
        psi_bc[: ox_depth + 2, -sd_width:] = v_drain

        # Верхний затвор по центру
        v_gate = float(rng.uniform(-0.5, 1.5))
        g_start = sd_width + 1
        g_end = size - sd_width - 1
        bc_mask[0, g_start:g_end] = 1.0
        psi_bc[0, g_start:g_end] = v_gate

        target_psi = _solve_2d_variable_eps_fd(eps_rel, psi_bc, bc_mask)
        inp = np.stack([eps_rel / 11.7, psi_bc, bc_mask], axis=0)
        tgt = target_psi[None, ...]
        return inp.astype(np.float32), tgt.astype(np.float32)

    if model_name == "magnetostatic":
        conductor = np.zeros((size, size), dtype=np.float32)
        jz = np.zeros((size, size), dtype=np.float32)
        bc_mask = np.zeros((size, size), dtype=np.float32)
        psi_bc = np.zeros((size, size), dtype=np.float32)

        # Нулевой векторный потенциал на внешней границе области
        bc_mask[0, :] = 1.0
        bc_mask[-1, :] = 1.0
        bc_mask[:, 0] = 1.0
        bc_mask[:, -1] = 1.0

        # Шина / канал с плотностью тока J_z
        cx = int(rng.integers(size // 4, 3 * size // 4))
        cz = int(rng.integers(size // 4, 3 * size // 4))
        rx = int(rng.integers(2, max(4, size // 6)))
        rz = int(rng.integers(2, max(4, size // 6)))
        conductor[max(1, cz - rz) : min(size - 1, cz + rz), max(1, cx - rx) : min(size - 1, cx + rx)] = 1.0
        jz_val = float(rng.uniform(-1.0, 1.0))
        jz = conductor * jz_val

        # Скалярное Poisson-приближение для Az
        mu_rel = np.ones((size, size), dtype=np.float32)
        az = _solve_2d_variable_eps_fd(mu_rel, psi_bc, bc_mask, rhs=jz * 0.05, iters=300)
        max_abs = max(float(np.max(np.abs(az))), 1e-6)
        az_norm = az / max_abs

        inp = np.stack([conductor, psi_bc, jz], axis=0)
        tgt = az_norm[None, ...]
        return inp.astype(np.float32), tgt.astype(np.float32)

    # model_name == "carriers"
    semiconductor = np.ones((size, size), dtype=np.float32)
    ox_depth = int(rng.integers(max(2, size // 12), max(4, size // 5)))
    semiconductor[:ox_depth, :] = 0.0

    # Логарифмическое сжатие легирования (log10(N) / 20)
    nd_exp = np.full((size, size), 10.0, dtype=np.float32)
    na_exp = np.full((size, size), float(rng.uniform(15.0, 18.0)), dtype=np.float32)

    sd_width = max(3, size // 5)
    sd_depth = max(ox_depth + 4, size // 3)
    nd_exp[ox_depth:sd_depth, :sd_width] = float(rng.uniform(19.0, 20.5))
    nd_exp[ox_depth:sd_depth, -sd_width:] = float(rng.uniform(19.0, 20.5))
    na_exp[ox_depth:sd_depth, :sd_width] = 10.0
    na_exp[ox_depth:sd_depth, -sd_width:] = 10.0

    nd_exp *= semiconductor
    na_exp *= semiconductor

    # Потенциал psi в области
    zz, xx = np.meshgrid(
        np.linspace(1.0, 0.0, size, dtype=np.float32),
        np.linspace(0.0, 1.0, size, dtype=np.float32),
        indexing="ij",
    )
    v_gate = float(rng.uniform(-0.3, 1.2))
    v_drain = float(rng.uniform(0.0, 1.0))
    psi = (v_gate * (zz ** 2) + v_drain * xx * 0.4) * semiconductor

    # Локальное равновесие + закон действующих масс (ni ~ 1e10 см^-3 при 300 K)
    vt = 0.02585
    ni = 1.0e10
    nd_val = np.where(semiconductor > 0.5, 10.0 ** nd_exp, 0.0)
    na_val = np.where(semiconductor > 0.5, 10.0 ** na_exp, 0.0)
    net_doping = nd_val - na_val

    # Квазиравновесная оценка n и p с ограничением экспоненты
    arg = np.clip(psi / vt, -25.0, 30.0)
    n_eq = np.where(
        semiconductor > 0.5,
        np.maximum(net_doping, ni) * np.exp(arg * 0.35),
        1.0e8,
    )
    n_clip = np.clip(n_eq, 1.0e8, 1.0e21)
    p_clip = np.where(
        semiconductor > 0.5,
        np.clip((ni ** 2) / n_clip, 1.0e8, 1.0e21),
        1.0e8,
    )

    log_n_norm = np.where(semiconductor > 0.5, np.log10(n_clip) / 8.0, 1.0).astype(np.float32)
    log_p_norm = np.where(semiconductor > 0.5, np.log10(p_clip) / 8.0, 1.0).astype(np.float32)

    inp = np.stack([nd_exp / 20.0, na_exp / 20.0, psi, semiconductor], axis=0)
    tgt = np.stack([log_n_norm, log_p_norm], axis=0)
    return inp.astype(np.float32), tgt.astype(np.float32)


def generate_test_examples(
    model_name: ModelName,
    num_samples: int = 128,
    grid_size: int = 64,
    base_seed: int = 42,
    save_path: str | Path | None = None,
) -> dict[str, Any]:
    """
    Генерирует набор синтетических примеров и при необходимости сохраняет в *.npz.
    """
    spec = get_model_spec(model_name)
    inputs = np.empty((num_samples, spec.in_channels, grid_size, grid_size), dtype=np.float32)
    targets = np.empty((num_samples, spec.out_channels, grid_size, grid_size), dtype=np.float32)

    for idx in range(num_samples):
        inp, tgt = create_test_example(model_name, size=grid_size, seed=base_seed + idx)
        inputs[idx] = inp
        targets[idx] = tgt

    dataset = {
        "model_name": model_name,
        "inputs": inputs,
        "targets": targets,
        "output_names": np.array(spec.output_names),
    }

    if save_path is not None:
        out_file = Path(save_path)
        out_file.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(out_file, **dataset)

    return dataset


# =====================================================================
# 2 · Настройка устройства и обучение (train_model)
# =====================================================================


def _resolve_device(device_pref: Literal["auto", "cpu", "cuda"] = "auto") -> torch.device:
    if device_pref == "cpu":
        return torch.device("cpu")
    if device_pref == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("Запрошен CUDA, но GPU недоступен.")
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _configure_nvidia(device: torch.device) -> bool:
    """
    Настраивает cuDNN benchmark, TF32 и возвращает флаг доступности AMP FP16.
    """
    if device.type != "cuda":
        return False
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    return True


def _device_label(device: torch.device) -> str:
    if device.type == "cuda":
        return f"CUDA: {torch.cuda.get_device_name(device)}"
    return "CPU"


def train_model(
    model_name: ModelName,
    inputs: np.ndarray | torch.Tensor,
    targets: np.ndarray | torch.Tensor,
    epochs: int = 20,
    batch_size: int = 16,
    lr: float = 1e-3,
    device_pref: Literal["auto", "cpu", "cuda"] = "auto",
    checkpoint_path: str | Path | None = None,
) -> tuple[BaseUNet, dict[str, Any]]:
    """
    Обучает U-Net для указанной физической сети с AdamW и AMP FP16 (на CUDA).
    """
    spec = get_model_spec(model_name)
    device = _resolve_device(device_pref)
    use_amp = _configure_nvidia(device)

    x_tensor = torch.as_tensor(inputs, dtype=torch.float32)
    y_tensor = torch.as_tensor(targets, dtype=torch.float32)

    if x_tensor.ndim != 4 or x_tensor.shape[1] != spec.in_channels:
        raise ValueError(
            f"Ожидался вход [N, {spec.in_channels}, H, W], получено {tuple(x_tensor.shape)}"
        )
    if y_tensor.ndim != 4 or y_tensor.shape[1] != spec.out_channels:
        raise ValueError(
            f"Ожидался выход [N, {spec.out_channels}, H, W], получено {tuple(y_tensor.shape)}"
        )

    model = build_unet_for_spec(model_name).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    loader = DataLoader(
        TensorDataset(x_tensor, y_tensor),
        batch_size=batch_size,
        shuffle=True,
    )

    final_loss = 0.0
    model.train()
    for _epoch in range(epochs):
        epoch_loss = 0.0
        total_items = 0
        for bx, by in loader:
            bx = bx.to(device, non_blocking=True)
            by = by.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=use_amp):
                pred = model(bx)
                loss = F.mse_loss(pred.float(), by.float())

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            bs = bx.size(0)
            epoch_loss += float(loss.item()) * bs
            total_items += bs

        final_loss = epoch_loss / max(total_items, 1)

    checkpoint: dict[str, Any] = {
        "model_name": model_name,
        "in_channels": spec.in_channels,
        "out_channels": spec.out_channels,
        "base_channels": model.base_channels,
        "state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
        "train_loss": final_loss,
        "training_samples": int(x_tensor.shape[0]),
        "epochs": epochs,
        "grid_size": int(x_tensor.shape[-1]),
    }

    if checkpoint_path is not None:
        ckpt_file = Path(checkpoint_path)
        ckpt_file.parent.mkdir(parents=True, exist_ok=True)
        torch.save(checkpoint, ckpt_file)

    return model, checkpoint


# =====================================================================
# 3 · Инференс (predict)
# =====================================================================


def load_model_from_checkpoint(
    checkpoint: dict[str, Any],
    device_pref: Literal["auto", "cpu", "cuda"] = "auto",
) -> tuple[BaseUNet, torch.device]:
    model_name = checkpoint["model_name"]
    spec = get_model_spec(model_name)

    if int(checkpoint["in_channels"]) != spec.in_channels or int(checkpoint["out_channels"]) != spec.out_channels:
        raise ValueError(
            f"Несовпадение каналов checkpoint для '{model_name}': "
            f"ожидалось ({spec.in_channels}->{spec.out_channels}), "
            f"получено ({checkpoint['in_channels']}->{checkpoint['out_channels']})."
        )

    device = _resolve_device(device_pref)
    model = BaseUNet(
        in_channels=spec.in_channels,
        out_channels=spec.out_channels,
        base_channels=int(checkpoint.get("base_channels", 16)),
    ).to(device)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    return model, device


def predict(
    model: BaseUNet,
    checkpoint: dict[str, Any],
    input_tensor: np.ndarray | torch.Tensor,
    device_pref: Literal["auto", "cpu", "cuda"] = "auto",
) -> tuple[np.ndarray, str]:
    """
    Выполняет инференс NCHW с проверкой соответствия модели и checkpoint.
    Возвращает (field_array [B, C_out, H, W], backend_label).
    """
    model_name = checkpoint.get("model_name")
    spec = get_model_spec(str(model_name))

    if (
        model.in_channels != spec.in_channels
        or model.out_channels != spec.out_channels
        or int(checkpoint.get("in_channels", -1)) != spec.in_channels
        or int(checkpoint.get("out_channels", -1)) != spec.out_channels
    ):
        raise ValueError(f"Несоответствие модели и checkpoint для '{model_name}'.")

    x = torch.as_tensor(input_tensor, dtype=torch.float32)
    if x.ndim != 4 or x.shape[1] != spec.in_channels:
        raise ValueError(
            f"Ожидался тензор NCHW с {spec.in_channels} каналами, получено {tuple(x.shape)}."
        )

    device = _resolve_device(device_pref)
    use_amp = _configure_nvidia(device)
    model = model.to(device).eval()

    with torch.inference_mode():
        x_dev = x.to(device)
        with torch.amp.autocast("cuda", enabled=use_amp):
            out = model(x_dev)
        out_np = out.float().cpu().numpy()

    return out_np, _device_label(device)


# =====================================================================
# 4 · Мост в физический решатель (predict_field_warm_start)
# =====================================================================


def predict_field_warm_start(
    model: BaseUNet,
    checkpoint: dict[str, Any],
    mesh_axes: tuple[np.ndarray, np.ndarray, np.ndarray],
    device_params: dict[str, float],
    Vg: float,
    Vd: float,
    psi_2d_hint: np.ndarray | None = None,
    device_pref: Literal["auto", "cpu", "cuda"] = "auto",
) -> tuple[np.ndarray, str]:
    """
    Формирует начальное приближение (C_out, nx, ny, nz) для физического решателя:
    1) Строит 2D-срез [C_in, nx, nz] по контактам (gate сверху, drain справа) и параметрам прибора;
    2) Запускает U-Net и денормализует выход (для carriers: 10^(y*8), clip 1e8..1e21);
    3) Экструдирует срез вдоль оси Y (однородность по ширине канала).
    """
    x_axis, y_axis, z_axis = mesh_axes
    nx, ny, nz = len(x_axis), len(y_axis), len(z_axis)
    model_name = checkpoint["model_name"]
    get_model_spec(model_name)

    tox_nm = float(device_params.get("tox_nm", 2.0))
    t_si_nm = float(device_params.get("body_thickness_nm", 20.0))
    eps_ox = float(device_params.get("eps_ox", 3.9))
    eps_si = float(device_params.get("eps_si", 11.7))
    nd_cm3 = float(device_params.get("nd_cm3", 1.0e20))
    na_cm3 = float(device_params.get("na_cm3", 1.0e17))

    z_min, z_max = float(z_axis[0]), float(z_axis[-1])
    z_span = max(z_max - z_min, 1e-12)
    z_norm = (z_axis - z_min) / z_span  # 0 — верх (затвор), 1 — низ (подложка)
    ox_frac = np.clip(tox_nm / max(tox_nm + t_si_nm, 1e-6), 0.05, 0.45)

    sd_cells = max(2, nx // 5)
    ox_cells = max(1, int(round(ox_frac * nz)))

    if model_name == "poisson":
        eps_slice = np.full((nx, nz), eps_si, dtype=np.float32)
        eps_slice[:, :ox_cells] = eps_ox

        psi_bc = np.zeros((nx, nz), dtype=np.float32)
        bc_mask = np.zeros((nx, nz), dtype=np.float32)

        # Подложка снизу (z = max) -> 0 В
        bc_mask[:, -1] = 1.0
        psi_bc[:, -1] = 0.0

        # Исток слева (x = 0..sd_cells) -> 0 В
        bc_mask[:sd_cells, : ox_cells + 2] = 1.0
        psi_bc[:sd_cells, : ox_cells + 2] = 0.0

        # Сток справа (x = nx-sd_cells..nx) -> Vd
        bc_mask[-sd_cells:, : ox_cells + 2] = 1.0
        psi_bc[-sd_cells:, : ox_cells + 2] = float(Vd)

        # Затвор сверху по центру (z = 0) -> Vg
        bc_mask[sd_cells:-sd_cells, 0] = 1.0
        psi_bc[sd_cells:-sd_cells, 0] = float(Vg)

        slice_in = np.stack([eps_slice / 11.7, psi_bc, bc_mask], axis=0)[None, ...]
        pred_2d, backend = predict(model, checkpoint, slice_in, device_pref=device_pref)
        # pred_2d: [1, 1, nx, nz] -> [1, nx, ny, nz]
        field_3d = np.repeat(pred_2d[0, :, :, None, :], ny, axis=2)
        return field_3d.astype(np.float64), backend

    if model_name == "carriers":
        semi_mask = np.ones((nx, nz), dtype=np.float32)
        semi_mask[:, :ox_cells] = 0.0

        nd_slice = np.full((nx, nz), np.log10(1.0e10), dtype=np.float32)
        na_slice = np.full((nx, nz), np.log10(max(na_cm3, 1.0e10)), dtype=np.float32)

        sd_depth = min(nz, ox_cells + max(3, nz // 3))
        nd_slice[:sd_cells, ox_cells:sd_depth] = np.log10(max(nd_cm3, 1.0e10))
        nd_slice[-sd_cells:, ox_cells:sd_depth] = np.log10(max(nd_cm3, 1.0e10))
        na_slice[:sd_cells, ox_cells:sd_depth] = np.log10(1.0e10)
        na_slice[-sd_cells:, ox_cells:sd_depth] = np.log10(1.0e10)

        nd_slice *= semi_mask
        na_slice *= semi_mask

        if psi_2d_hint is not None and psi_2d_hint.shape == (nx, nz):
            psi_slice = psi_2d_hint.astype(np.float32)
        else:
            x_lin = np.linspace(0.0, 1.0, nx, dtype=np.float32)[:, None]
            z_lin = (1.0 - z_norm.astype(np.float32))[None, :]
            psi_slice = (float(Vg) * (z_lin ** 2) + float(Vd) * x_lin * 0.4) * semi_mask

        slice_in = np.stack(
            [nd_slice / 20.0, na_slice / 20.0, psi_slice, semi_mask],
            axis=0,
        )[None, ...]

        pred_2d, backend = predict(model, checkpoint, slice_in, device_pref=device_pref)
        # Денормализация концентраций: 10^(y * 8), ограничение [1e8, 1e21] см^-3
        carriers_2d = np.clip(np.power(10.0, pred_2d[0] * 8.0), 1.0e8, 1.0e21)
        field_3d = np.repeat(carriers_2d[:, :, None, :], ny, axis=2)
        return field_3d.astype(np.float64), backend

    # magnetostatic
    conductor = np.zeros((nx, nz), dtype=np.float32)
    conductor[sd_cells:-sd_cells, ox_cells : max(ox_cells + 2, nz // 2)] = 1.0
    jz = conductor * float(device_params.get("jz_norm", 0.5))
    psi_bc = np.zeros((nx, nz), dtype=np.float32)

    slice_in = np.stack([conductor, psi_bc, jz], axis=0)[None, ...]
    pred_2d, backend = predict(model, checkpoint, slice_in, device_pref=device_pref)
    field_3d = np.repeat(pred_2d[0, :, :, None, :], ny, axis=2)
    return field_3d.astype(np.float64), backend


# =====================================================================
# 5 · Экспорт в ONNX (opset 17) для Rust ort::Session
# =====================================================================


def export_onnx(
    model: BaseUNet,
    checkpoint: dict[str, Any],
    onnx_path: str | Path,
    sample_hw: tuple[int, int] = (64, 64),
    opset_version: int = 17,
) -> Path:
    """
    Экспортирует обученную U-Net в формат ONNX (opset 17) с динамическими осями
    batch / height / width и именем входа 'input_features' для Rust ort::Session.
    """
    model_name = checkpoint["model_name"]
    spec = get_model_spec(model_name)

    model_cpu = model.to("cpu").eval()
    dummy_input = torch.randn(1, spec.in_channels, sample_hw[0], sample_hw[1], dtype=torch.float32)

    out_path = Path(onnx_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    torch.onnx.export(
        model_cpu,
        dummy_input,
        str(out_path),
        export_params=True,
        opset_version=opset_version,
        do_constant_folding=True,
        input_names=["input_features"],
        output_names=["predicted_field"],
        dynamic_axes={
            "input_features": {0: "batch", 2: "height", 3: "width"},
            "predicted_field": {0: "batch", 2: "height", 3: "width"},
        },
    )

    try:
        import onnx

        onnx_model = onnx.load(str(out_path))
        onnx.checker.check_model(onnx_model)
    except ImportError:
        pass

    return out_path


# =====================================================================
# Пример самопроверки пайплайна
# =====================================================================

if __name__ == "__main__":
    data = generate_test_examples("carriers", num_samples=16, grid_size=32, base_seed=7)
    net, ckpt = train_model(
        "carriers",
        data["inputs"],
        data["targets"],
        epochs=2,
        batch_size=8,
        device_pref="auto",
    )

    axes = (
        np.linspace(0.0, 50e-9, 32),
        np.linspace(0.0, 20e-9, 8),
        np.linspace(0.0, 25e-9, 32),
    )
    warm_fields, used_backend = predict_field_warm_start(
        net,
        ckpt,
        mesh_axes=axes,
        device_params={"tox_nm": 1.5, "body_thickness_nm": 15.0, "nd_cm3": 1e20, "na_cm3": 1e17},
        Vg=0.8,
        Vd=0.5,
    )
    print(f"Warm start shape: {warm_fields.shape}, backend: {used_backend}")
