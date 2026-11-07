import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from src.neural_physics import (
    MODEL_SPECS,
    BaseUNet,
    create_test_example,
    generate_test_examples,
    predict,
    train_model,
)


class NeuralPhysicsTests(unittest.TestCase):
    def test_all_three_physics_examples_have_expected_shapes_and_finite_values(self):
        for model_name, spec in MODEL_SPECS.items():
            with self.subTest(model=model_name):
                inputs, targets = create_test_example(
                    model_name, size=32, seed=7
                )
                self.assertEqual(inputs.shape, (spec["in_channels"], 32, 32))
                self.assertEqual(targets.shape, (spec["out_channels"], 32, 32))
                self.assertTrue(np.isfinite(inputs).all())
                self.assertTrue(np.isfinite(targets).all())
                model = BaseUNet(spec["in_channels"], spec["out_channels"])
                with torch.inference_mode():
                    output = model(torch.from_numpy(inputs[None]))
                self.assertEqual(
                    tuple(output.shape), (1, spec["out_channels"], 32, 32)
                )

    def test_synthetic_examples_are_saved_for_each_physics_model(self):
        with tempfile.TemporaryDirectory() as directory:
            paths = generate_test_examples(directory, count=2, seed=11)
            self.assertEqual(len(paths), 3)
            for path in paths:
                with self.subTest(path=path.name):
                    with np.load(path) as data:
                        self.assertEqual(data["inputs"].shape[0], 2)
                        self.assertEqual(data["targets"].shape[0], 2)

    def test_training_checkpoint_runs_inference_and_checks_model_identity(self):
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "poisson.pt"
            train_result = train_model(
                "poisson",
                samples=2,
                epochs=1,
                batch_size=2,
                size=16,
                checkpoint_path=checkpoint,
                device_name="cpu",
            )
            self.assertTrue(checkpoint.is_file())
            self.assertTrue(np.isfinite(train_result["train_loss"]))
            inputs, _ = create_test_example("poisson", size=16, seed=2)
            prediction, backend = predict(
                "poisson", checkpoint, inputs[None], device_name="cpu"
            )
            self.assertEqual(prediction.shape, (1, 1, 16, 16))
            self.assertEqual(backend, "CPU")
            with self.assertRaisesRegex(ValueError, "does not match"):
                predict(
                    "magnetostatic", checkpoint, np.zeros((1, 3, 16, 16), np.float32),
                    device_name="cpu",
                )


if __name__ == "__main__":
    unittest.main()
