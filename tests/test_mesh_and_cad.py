import struct
import tempfile
import unittest
from pathlib import Path

import numpy as np

from src.cad import load_cad_device_model, load_stl
from src.model import DeviceParameters, export_analysis_report
from src.simulation import _adaptive_axis


class MeshAndCadTests(unittest.TestCase):
    def test_adaptive_axis_refines_around_geometry_features(self):
        axis = _adaptive_axis(1.0, 17, (0.25, 0.5, 0.75))

        self.assertEqual(len(axis), 17)
        self.assertEqual(axis[0], 0)
        self.assertEqual(axis[-1], 1)
        self.assertTrue(np.all(np.diff(axis) > 0))
        self.assertLess(np.min(np.diff(axis)), np.diff(axis)[0] * 0.7)

    def test_loads_bundled_cad_stl_in_millimetres(self):
        path = Path("data/examples/example_transistor_channel.stl")

        vertices, faces = load_stl(path)

        self.assertEqual(vertices.shape, (36, 3))
        self.assertEqual(faces.shape, (12, 3))
        np.testing.assert_allclose(np.ptp(vertices, axis=0), [1e6, 2e5, 1e5])

    def test_loads_binary_stl_and_applies_unit_scale(self):
        header = b"example binary STL".ljust(80, b" ")
        triangle = struct.pack(
            "<12fH",
            0, 0, 1,
            0, 0, 0,
            1, 0, 0,
            0, 1, 1,
            0,
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model.stl"
            path.write_bytes(header + struct.pack("<I", 1) + triangle)

            vertices, faces = load_stl(path, unit_to_nm=1_000)

        self.assertEqual(faces.shape, (1, 3))
        np.testing.assert_allclose(vertices, [[0, 0, 0], [1_000, 0, 0], [0, 1_000, 1_000]])

    def test_rejects_malformed_stl_and_invalid_scale(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bad.stl"
            path.write_text("not an STL", encoding="ascii")

            with self.assertRaisesRegex(ValueError, "truncated"):
                load_stl(path)
            with self.assertRaisesRegex(ValueError, "positive and finite"):
                load_stl(path, unit_to_nm=0)

    def test_loads_exported_device_model_json(self):
        device = DeviceParameters(
            name="cad_nmos",
            epsilon_r=11.7,
            electron_mobility=1400,
            hole_mobility=450,
            bandgap=1.12,
            doping=1e16,
            tox_nm=5,
            width_um=1,
            length_um=0.18,
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "device.json"
            path.write_text(
                export_analysis_report(device, {"VTH0": 0.4}),
                encoding="utf-8",
            )

            model = load_cad_device_model(path)

        self.assertEqual(model.name, "cad_nmos")
        self.assertEqual(model.device_type, "planar_mosfet")
        self.assertEqual(model.length_um, 0.18)
        self.assertEqual(model.tox_nm, 5)

    def test_rejects_incomplete_device_model_json(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "invalid.json"
            path.write_text('{"device_type": "hemt"}', encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "material.*geometry"):
                load_cad_device_model(path)


if __name__ == "__main__":
    unittest.main()
