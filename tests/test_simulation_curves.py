import math
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

import numpy as np

from src.model import DeviceParameters, extract_curve_metrics, load_measurement
from src import simulation
from src.simulation import (
    solve_pde_characteristics,
    solve_quantum_charge_distribution,
)


class SimulationCurveTests(unittest.TestCase):
    def setUp(self):
        self.device = DeviceParameters(
            name="curve_test",
            epsilon_r=11.7,
            electron_mobility=1400,
            hole_mobility=450,
            bandgap=1.12,
            doping=1e16,
            tox_nm=5,
            width_um=1,
            length_um=0.18,
        )

    def _load_log(self, text):
        temporary = tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", suffix=".log", delete=False
        )
        with temporary:
            temporary.write(text)
        path = Path(temporary.name)
        self.addCleanup(path.unlink, missing_ok=True)
        return load_measurement(path)

    def test_imports_whitespace_silvaco_idvg_and_calculates_metrics(self):
        measurement = self._load_log(
            "Vg (V) Id (A)\n"
            "0 1e-12\n"
            "0.2 1e-10\n"
            "0.4 1e-8\n"
            "0.6 1e-6\n"
            "0.8 1e-4\n"
        )
        self.assertEqual(measurement.kind, "idvg")
        metrics = extract_curve_metrics(measurement, self.device)
        self.assertGreater(metrics["ION_A"], metrics["IOFF_A"])
        self.assertGreater(metrics["GM_MAX_S"], 0)
        self.assertIn("VTH0", metrics)
        self.assertIn("VTH_CONSTANT_CURRENT_V", metrics)

    def test_gate_leakage_and_cv_component_extraction(self):
        gate_leakage = self._load_log(
            "Vg Ig\n-1 1e-12\n0 2e-12\n1 1e-9\n"
        )
        self.assertEqual(gate_leakage.kind, "igvg")
        self.assertAlmostEqual(
            extract_curve_metrics(gate_leakage, self.device)["IG_MAX_A"], 1e-9
        )

        cv = self._load_log(
            "Vg Cgg Cgs Cgd\n0 1e-12 2e-13 1e-13\n"
            "0.5 2e-12 3e-13 2e-13\n1 3e-12 4e-13 3e-13\n"
        )
        self.assertEqual(cv.kind, "cv")
        metrics = extract_curve_metrics(cv, self.device)
        self.assertIn("CGG", metrics)
        self.assertAlmostEqual(metrics["CGS_MEAN_F"], 3e-13)
        self.assertAlmostEqual(metrics["CGD_MEAN_F"], 2e-13)

    def test_gummel_and_bjt_output_metrics(self):
        gummel = self._load_log(
            "Vbe Ic Ib\n0.5 1e-6 1e-8\n0.6 1e-4 1e-6\n0.7 1e-2 1e-4\n"
        )
        self.assertEqual(gummel.kind, "gummel")
        self.assertAlmostEqual(extract_curve_metrics(gummel, self.device)["BETA_MAX"], 100)

        output = self._load_log(
            "Vce Ic\n0 0\n0.1 1e-3\n0.2 2e-3\n0.4 2.2e-3\n0.8 2.3e-3\n"
        )
        self.assertEqual(output.kind, "bjt_output")
        self.assertGreater(extract_curve_metrics(output, self.device)["IC_MAX_A"], 0)

    def test_ac_frequencies_and_gate_charge(self):
        ac = self._load_log(
            "Frequency H21_dB U_dB\n"
            "1e6 20 20\n1e7 10 10\n1e8 0 0\n1e9 -10 -10\n"
        )
        self.assertEqual(ac.kind, "ac")
        metrics = extract_curve_metrics(ac, self.device)
        self.assertAlmostEqual(metrics["FT_HZ"], 1e8)
        self.assertAlmostEqual(metrics["FMAX_HZ"], 1e8)

        charge = self._load_log(
            "time Ig Vg\n0 1e-3 0\n1e-9 1e-3 0.5\n2e-9 1e-3 1\n"
        )
        self.assertEqual(charge.kind, "gate_charge")
        self.assertAlmostEqual(extract_curve_metrics(charge, self.device)["QG_C"], 2e-12)

    def test_transient_switching_times(self):
        transient = self._load_log(
            "time Vg Id\n"
            "0 0 0\n1e-9 0.5 0.1\n2e-9 1 1\n"
            "3e-9 1 1\n4e-9 0.5 0.1\n5e-9 0 0\n"
        )
        self.assertEqual(transient.kind, "transient")
        metrics = extract_curve_metrics(transient, self.device)
        self.assertGreater(metrics["TON_10_90_S"], 0)
        self.assertGreater(metrics["TOFF_90_10_S"], 0)

    def test_bundled_silvaco_style_curve_examples(self):
        examples = Path("data/examples")
        for filename, expected_kind in (
            ("silvaco_idvg.log", "idvg"),
            ("silvaco_igvg.log", "igvg"),
            ("silvaco_cv.log", "cv"),
            ("bjt_gummel.log", "gummel"),
            ("ac_gains.log", "ac"),
            ("transient_switching.log", "transient"),
            ("gate_charge.log", "gate_charge"),
        ):
            with self.subTest(filename=filename):
                measurement = load_measurement(examples / filename)
                self.assertEqual(measurement.kind, expected_kind)
                self.assertTrue(extract_curve_metrics(measurement, self.device))

    def test_pde_solver_generates_physical_units_and_supported_curve_families(self):
        result = solve_pde_characteristics(
            self.device,
            use_gpu=False,
        )
        self.assertIn("CPU", result.backend)
        self.assertEqual(len(result.curves), 7)
        transfer_curves = [curve for curve in result.curves if curve.kind == "idvg"]
        output_curves = [curve for curve in result.curves if curve.kind == "idvd"]
        cv_curves = [curve for curve in result.curves if curve.kind == "cv"]
        self.assertEqual(len(transfer_curves), 2)
        self.assertEqual(len(output_curves), 4)
        self.assertEqual(len(cv_curves), 1)

        for curve in result.curves:
            self.assertTrue(all(map(math.isfinite, curve.x + curve.y)))
            self.assertTrue(all(value >= 0 for value in curve.y))
            if curve.kind in ("idvg", "idvd"):
                self.assertTrue(
                    all(left <= right for left, right in zip(curve.y, curve.y[1:]))
                )
        self.assertEqual(len(transfer_curves[0].x), 31)
        self.assertLess(max(transfer_curves[1].y), 1e-2)
        self.assertLess(max(cv_curves[0].y), 1e-12)
        self.assertGreater(max(cv_curves[0].y), 0)
        oxide_capacitance = (
            8.8541878128e-12
            * self.device.epsilon_r
            / (self.device.tox_nm * 1e-9)
            * self.device.width_um
            * self.device.length_um
            * 1e-12
        )
        self.assertGreater(max(cv_curves[0].y), oxide_capacitance * 0.1)
        self.assertLess(max(cv_curves[0].y), oxide_capacitance * 1.1)

    def test_pde_solver_does_not_generate_unsupported_device_curves(self):
        hemt = replace(
            self.device,
            device_type="hemt",
            geometry={
                "barrier_thickness_nm": 20,
                "sheet_density_cm2": 1e13,
                "channel_mobility_cm2_vs": 1500,
                "access_resistance_ohm_mm": 0.5,
                "gate_recess_nm": 0,
            },
        )
        with self.assertRaisesRegex(ValueError, "HEMT and TFET"):
            solve_pde_characteristics(hemt, use_gpu=False)

    def test_pde_solver_converges_for_supported_ui_channel_geometries(self):
        devices = (
            replace(
                self.device,
                device_type="fd_soi",
                geometry={
                    "body_thickness_nm": 7,
                    "buried_oxide_nm": 25,
                    "back_gate_voltage": 0,
                },
            ),
            replace(
                self.device,
                device_type="finfet",
                geometry={
                    "fin_width_nm": 8,
                    "fin_height_nm": 40,
                    "fin_count": 2,
                    "fin_pitch_nm": 30,
                },
            ),
            replace(
                self.device,
                device_type="gaa_nanosheet",
                geometry={
                    "sheet_width_nm": 30,
                    "sheet_thickness_nm": 8,
                    "sheet_count": 3,
                    "sheet_pitch_nm": 30,
                },
            ),
            replace(
                self.device,
                device_type="gaa_nanowire",
                geometry={"wire_diameter_nm": 10, "wire_count": 3},
            ),
        )
        for device in devices:
            with self.subTest(device_type=device.device_type):
                result = solve_pde_characteristics(device, use_gpu=False)
                self.assertEqual(len(result.curves), 7)
                self.assertIn("CPU", result.backend)

    def test_self_consistent_quantum_charge_solver_returns_2d_and_3d_fields(self):
        charge_2d = solve_quantum_charge_distribution(
            self.device,
            gate_voltage=0.8,
            drain_voltage=0.1,
            dimension="2d",
            grid_shape=(9, 5, 9),
            use_gpu=False,
        )
        self.assertEqual(charge_2d.values.shape, (9, 2, 9))
        self.assertIn("Schrödinger", charge_2d.description)
        self.assertGreater(charge_2d.metrics["MAX_ELECTRON_DENSITY_CM3"], 0)
        self.assertGreater(charge_2d.metrics["QUANTUM_ITERATIONS"], 1)

        finfet = replace(
            self.device,
            device_type="finfet",
            geometry={
                "fin_width_nm": 8,
                "fin_height_nm": 40,
                "fin_count": 1,
                "fin_pitch_nm": 30,
            },
        )
        charge_3d = solve_quantum_charge_distribution(
            finfet,
            gate_voltage=0.8,
            drain_voltage=0.1,
            dimension="3d",
            grid_shape=(9, 5, 9),
            use_gpu=False,
        )
        self.assertEqual(charge_3d.values.shape, (9, 5, 9))
        self.assertTrue(all(map(math.isfinite, charge_3d.values.flat)))
        self.assertGreater(
            float(charge_3d.values[:, 0, :].mean()),
            float(charge_3d.values[:, 2, :].mean()),
        )
        self.assertIn("CPU", charge_3d.backend)

    def test_quantum_charge_solver_rejects_non_mos_models(self):
        hemt = replace(
            self.device,
            device_type="hemt",
            geometry={
                "barrier_thickness_nm": 20,
                "sheet_density_cm2": 1e13,
                "channel_mobility_cm2_vs": 1500,
                "access_resistance_ohm_mm": 0.5,
                "gate_recess_nm": 0,
            },
        )
        with self.assertRaisesRegex(ValueError, "TFET/HEMT"):
            solve_quantum_charge_distribution(
                hemt,
                gate_voltage=0.8,
                drain_voltage=0.1,
                use_gpu=False,
            )

    def test_quantum_poisson_matrix_is_symmetric_and_torch_matches_numpy(self):
        try:
            import torch
        except ImportError:
            self.skipTest("PyTorch is optional")

        nx, ny, nz = 9, 5, 9
        length_m = self.device.length_um * 1e-6
        width_m = 8e-9
        depth_m = 8e-9
        dx, dy, dz = (
            length_m / (nx - 1),
            width_m / (ny - 1),
            depth_m / (nz - 1),
        )
        epsilon = simulation.EPSILON_0 * self.device.epsilon_r
        cox = epsilon / (self.device.tox_nm * 1e-9)
        matrix, _, _ = simulation._assemble_quantum_poisson_matrix(
            nx, ny, nz, dx, dy, dz, epsilon, cox, 0.8, 0.1, True, False
        )
        self.assertEqual((matrix - matrix.T).nnz, 0)

        doping = self.device.doping * 1e6
        mass = 0.26 * simulation.ELECTRON_MASS
        kT = simulation.BOLTZMANN * self.device.temperature_k
        nc = 2 * (
            2 * math.pi * mass * kT / (2 * math.pi * simulation.PLANCK_REDUCED) ** 2
        ) ** 1.5
        eta = math.log(doping / nc)
        common = (
            0.8,
            0.1,
            nx,
            ny,
            nz,
            (dx, dy, dz),
            epsilon,
            cox,
            doping,
            mass,
            kT,
            eta,
            length_m,
            True,
            False,
        )
        numpy_result = simulation._quantum_poisson_numpy(*common)
        torch_result = simulation._quantum_poisson_torch(
            torch, *common, device_type="cpu"
        )
        potential_error = np.max(np.abs(numpy_result[0] - torch_result[0]))
        density_error = np.max(np.abs(numpy_result[1] - torch_result[1]))
        self.assertLess(potential_error / np.max(np.abs(numpy_result[0])), 5e-5)
        self.assertLess(density_error / np.max(np.abs(numpy_result[1])), 2e-5)


if __name__ == "__main__":
    unittest.main()
