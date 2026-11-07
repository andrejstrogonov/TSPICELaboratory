import csv
import math
import tempfile
import unittest
from pathlib import Path

from src.model import (
    BOLTZMANN,
    ELECTRON_CHARGE,
    DeviceParameters,
    export_spice_model,
    extract_cv,
    extract_idvg,
    extract_output_metrics,
    extract_transfer_metrics,
    load_measurement,
)
from src.simulation import (
    MAX_MESH_NODES,
    _validate_grid_shape,
    estimate_mesh_memory_bytes,
    estimate_tetrahedron_count,
    solve_quasistatic_electric_field,
    solve_steady_thermal,
)


class ModelTests(unittest.TestCase):
    def setUp(self):
        self.device = DeviceParameters(
            name="test_nmos",
            epsilon_r=11.7,
            electron_mobility=1400,
            hole_mobility=450,
            bandgap=1.12,
            doping=1e16,
            tox_nm=5,
            width_um=1,
            length_um=0.18,
        )

    def _write_csv(self, headers, rows, delimiter=","):
        temporary = tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", newline="", suffix=".csv", delete=False
        )
        with temporary:
            writer = csv.writer(temporary, delimiter=delimiter)
            writer.writerow(headers)
            writer.writerows(rows)
        self.addCleanup(Path(temporary.name).unlink, missing_ok=True)
        return temporary.name

    def test_loads_idvg_and_drain_bias(self):
        path = self._write_csv(
            ["Vg (V)", "Vd (V)", "Id (A)"],
            [[0, 0.05, 1e-12], [0.2, 0.05, 2e-10], [0.4, 0.05, 1e-8]],
            delimiter=";",
        )
        measurement = load_measurement(path)
        self.assertEqual(measurement.kind, "idvg")
        self.assertEqual(measurement.drain_voltage, [0.05, 0.05, 0.05])
        self.assertEqual(len(measurement.x), 3)

    def test_detects_idvd_with_fixed_gate_bias(self):
        path = self._write_csv(
            ["Vg (V)", "Vd (V)", "Id (A)"],
            [[0.8, 0.0, 0], [0.8, 0.05, 0.001], [0.8, 0.1, 0.0018],
             [0.8, 0.2, 0.0025], [0.8, 0.4, 0.0027]],
        )
        measurement = load_measurement(path)
        self.assertEqual(measurement.kind, "idvd")
        self.assertEqual(measurement.gate_bias, 0.8)
        metrics = extract_output_metrics(measurement, self.device)
        self.assertEqual(metrics["VGS_BIAS"], 0.8)
        self.assertGreater(metrics["RON_OHM"], 0)
        self.assertGreater(metrics["IDS_MAX"], 0)

    def test_loads_cv_data_and_rejects_bad_rows(self):
        path = self._write_csv(["Vg", "Cg"], [[0, 1e-12], [0.5, 2e-12], [1, 3e-12]])
        self.assertEqual(load_measurement(path).kind, "cv")
        invalid = self._write_csv(["Vg", "Id"], [[0, 1e-12], [0.5, "bad"], [1, 3e-12]])
        with self.assertRaisesRegex(ValueError, "line 3"):
            load_measurement(invalid)

    def test_fits_idvg_parameters(self):
        import numpy as np

        cox = self.device.oxide_capacitance_per_area
        width = self.device.width_um * 1e-6
        length = self.device.length_um * 1e-6
        ut = BOLTZMANN * self.device.temperature_k / ELECTRON_CHARGE
        expected_vth = 0.42
        expected_mobility = 320 * 1e-4
        expected_factor = 1.45
        vg = np.linspace(0, 1.2, 50)
        vd = np.full_like(vg, 0.05)
        slope = expected_factor * ut
        softplus = np.logaddexp(0, (vg - expected_vth) / (2 * slope))
        current = (
            expected_mobility * cox * width / length * 2 * slope**2
            * softplus**2 * (1 - np.exp(-vd / ut))
        )
        from src.model import Measurement

        fit = extract_idvg(
            Measurement("idvg", vg.tolist(), current.tolist(), vd.tolist()), self.device
        )
        self.assertAlmostEqual(fit["VTH0"], expected_vth, delta=0.03)
        self.assertAlmostEqual(fit["U0"], 320, delta=20)
        self.assertAlmostEqual(fit["NFACTOR"], expected_factor, delta=0.2)

    def test_cv_summary_and_spice_export(self):
        from src.model import Measurement

        extracted = extract_cv(
            Measurement("cv", [-0.5, 0, 0.5, 1], [1e-12, 2e-12, 5e-12, 6e-12]),
            self.device,
        )
        self.assertEqual(extracted["CGG"], 5.5e-12)
        self.assertGreaterEqual(extracted["CGSO"], 0)
        card = export_spice_model(self.device, {"VTH0": 0.4, "U0": 320, "NFACTOR": 1.5})
        self.assertIn(".model test_nmos nmos level=54", card)
        self.assertIn("+ TOXE=5e-09", card)

    def test_device_parameters_require_positive_values(self):
        with self.assertRaisesRegex(ValueError, "positive"):
            self.device.__class__(
                **{**self.device.__dict__, "tox_nm": 0}
            ).validate()

    def test_geometry_widths_and_family_specific_transfer_metrics(self):
        from src.model import Measurement

        structures = [
            ("fd_soi", {"body_thickness_nm": 7, "buried_oxide_nm": 25, "back_gate_voltage": 0}, 1.0),
            ("finfet", {"fin_width_nm": 8, "fin_height_nm": 40, "fin_count": 2, "fin_pitch_nm": 30}, 0.176),
            ("gaa_nanosheet", {"sheet_width_nm": 30, "sheet_thickness_nm": 8, "sheet_count": 3, "sheet_pitch_nm": 30}, 0.228),
            ("gaa_nanowire", {"wire_diameter_nm": 10, "wire_count": 3}, math.pi * 0.03),
        ]
        for device_type, geometry, expected_width in structures:
            with self.subTest(device_type=device_type):
                device = DeviceParameters(
                    **{**self.device.__dict__, "device_type": device_type, "geometry": geometry}
                )
                self.assertAlmostEqual(device.effective_width_um, expected_width)
        tfet = DeviceParameters(
            **{
                **self.device.__dict__,
                "device_type": "tfet",
                "geometry": {
                    "tunnel_length_nm": 20,
                    "channel_thickness_nm": 8,
                    "source_doping_cm3": 1e20,
                    "drain_doping_cm3": 1e18,
                },
            }
        )
        metrics = extract_transfer_metrics(
            Measurement("idvg", [0, 0.1, 0.2, 0.3, 0.4, 0.5], [1e-12, 1e-11, 1e-10, 1e-9, 1e-7, 1e-5]),
            tfet,
        )
        self.assertGreater(metrics["ION_IOFF"], 1e5)
        self.assertGreater(metrics["SS_MV_DEC"], 0)

    def test_hemt_metrics_include_sheet_resistance_and_report(self):
        from src.model import Measurement, export_analysis_report

        hemt = DeviceParameters(
            **{
                **self.device.__dict__,
                "device_type": "hemt",
                "geometry": {
                    "barrier_thickness_nm": 20,
                    "sheet_density_cm2": 1e13,
                    "channel_mobility_cm2_vs": 1500,
                    "access_resistance_ohm_mm": 0.5,
                    "gate_recess_nm": 0,
                },
            }
        )
        transfer = Measurement(
            "idvg",
            [-2, -1.5, -1, -0.5, 0],
            [1e-9, 1e-6, 1e-4, 2e-4, 3e-4],
        )
        metrics = extract_transfer_metrics(transfer, hemt)
        self.assertGreater(metrics["RSH_OHM_SQ"], 0)
        self.assertGreater(metrics["VTH_CONSTANT_CURRENT"], -2)
        self.assertLess(metrics["VTH_CONSTANT_CURRENT"], -1.5)
        output = extract_output_metrics(
            Measurement("idvd", [0.02, 0.05, 0.1, 0.5, 1], [1e-3, 2e-3, 2.5e-3, 3e-3, 3.1e-3]),
            hemt,
        )
        self.assertGreater(output["RACCESS_OHM"], 0)
        self.assertIn('"device_type": "hemt"', export_analysis_report(hemt, metrics))

    def test_bsim4_export_is_restricted_to_planar_bulk(self):
        finfet = DeviceParameters(
            **{
                **self.device.__dict__,
                "device_type": "finfet",
                "geometry": {
                    "fin_width_nm": 8,
                    "fin_height_nm": 40,
                    "fin_count": 2,
                    "fin_pitch_nm": 30,
                },
            }
        )
        with self.assertRaisesRegex(ValueError, "only for planar"):
            export_spice_model(finfet, {"VTH0": 0.4})

    def test_bundled_family_examples_run_through_the_matching_analysis(self):
        examples = Path("data/examples")
        structures = {
            "planar_mosfet": ("idvg_vd_005.csv", {}),
            "fd_soi": (
                "fd_soi_idvg.csv",
                {"body_thickness_nm": 7, "buried_oxide_nm": 25, "back_gate_voltage": 0},
            ),
            "finfet": (
                "finfet_idvg.csv",
                {"fin_width_nm": 8, "fin_height_nm": 40, "fin_count": 2, "fin_pitch_nm": 30},
            ),
            "gaa_nanosheet": (
                "gaa_nanosheet_idvg.csv",
                {"sheet_width_nm": 30, "sheet_thickness_nm": 8, "sheet_count": 3, "sheet_pitch_nm": 30},
            ),
            "gaa_nanowire": (
                "gaa_nanowire_idvg.csv",
                {"wire_diameter_nm": 10, "wire_count": 3},
            ),
            "tfet": (
                "tfet_idvg.csv",
                {
                    "tunnel_length_nm": 20,
                    "channel_thickness_nm": 8,
                    "source_doping_cm3": 1e20,
                    "drain_doping_cm3": 1e18,
                },
            ),
            "hemt": (
                "hemt_idvg.csv",
                {
                    "barrier_thickness_nm": 20,
                    "sheet_density_cm2": 1e13,
                    "channel_mobility_cm2_vs": 1500,
                    "access_resistance_ohm_mm": 0.5,
                    "gate_recess_nm": 0,
                },
            ),
        }
        for device_type, (filename, geometry) in structures.items():
            with self.subTest(device_type=device_type):
                device = DeviceParameters(
                    **{
                        **self.device.__dict__,
                        "device_type": device_type,
                        "geometry": geometry,
                    }
                )
                measurement = load_measurement(examples / filename)
                if device_type in ("tfet", "hemt"):
                    extracted = extract_transfer_metrics(measurement, device)
                else:
                    extracted = extract_idvg(measurement, device)
                self.assertIn("ION" if device_type in ("tfet", "hemt") else "VTH0", extracted)
        output = load_measurement(examples / "idvd_vg_080.csv")
        self.assertEqual(output.kind, "idvd")
        self.assertEqual(output.gate_bias, 0.8)
        self.assertIn("RON_OHM", extract_output_metrics(output, self.device))
        cv = load_measurement(examples / "cv_sweep.csv")
        self.assertIn("CGG", extract_cv(cv, self.device))

    def test_3d_thermal_and_quasistatic_field_solvers(self):
        thermal = solve_steady_thermal(
            self.device,
            drain_voltage=0.8,
            drain_current=1e-4,
            ambient_temperature_k=300,
            thermal_conductivity_w_mk=130,
            substrate_thickness_um=0.5,
            grid_shape=(9, 7, 7),
        )
        self.assertEqual(thermal.values.shape, (9, 7, 7))
        self.assertGreater(thermal.metrics["TMAX_K"], 300)
        self.assertGreater(thermal.metrics["POWER_W"], 0)

        electric, magnetic = solve_quasistatic_electric_field(
            self.device,
            gate_voltage=0.8,
            drain_voltage=0.8,
            drain_current=1e-4,
            grid_shape=(9, 7, 7),
        )
        self.assertEqual(electric.values.shape, (9, 7, 7))
        self.assertGreater(electric.metrics["EMAX_V_M"], 0)
        self.assertGreater(electric.metrics["CGEO_F"], 0)
        self.assertTrue((electric.values >= 0).all())
        self.assertGreater(magnetic.metrics["BMAX_EST_T"], 0)
        self.assertEqual(magnetic.values.shape, (9, 7, 7))
        self.assertIn("Biot", magnetic.value_name)
        self.assertIn("SciPy", thermal.backend)
        self.assertIn("SciPy", electric.backend)

    def test_3d_solvers_accept_each_supported_device_family(self):
        families = {
            "fd_soi": {"body_thickness_nm": 7, "buried_oxide_nm": 25, "back_gate_voltage": 0},
            "finfet": {"fin_width_nm": 8, "fin_height_nm": 40, "fin_count": 2, "fin_pitch_nm": 30},
            "gaa_nanosheet": {"sheet_width_nm": 30, "sheet_thickness_nm": 8, "sheet_count": 3, "sheet_pitch_nm": 30},
            "gaa_nanowire": {"wire_diameter_nm": 10, "wire_count": 3},
            "tfet": {"tunnel_length_nm": 20, "channel_thickness_nm": 8, "source_doping_cm3": 1e20, "drain_doping_cm3": 1e18},
            "hemt": {"barrier_thickness_nm": 20, "sheet_density_cm2": 1e13, "channel_mobility_cm2_vs": 1500, "access_resistance_ohm_mm": 0.5, "gate_recess_nm": 0},
        }
        for device_type, geometry in families.items():
            with self.subTest(device_type=device_type):
                device = DeviceParameters(
                    **{
                        **self.device.__dict__,
                        "device_type": device_type,
                        "geometry": geometry,
                    }
                )
                thermal = solve_steady_thermal(
                    device,
                    drain_voltage=0.5,
                    drain_current=1e-4,
                    ambient_temperature_k=300,
                    thermal_conductivity_w_mk=100,
                    substrate_thickness_um=0.5,
                    grid_shape=(5, 5, 5),
                )
                electric, magnetic = solve_quasistatic_electric_field(
                    device,
                    gate_voltage=0.5,
                    drain_voltage=0.5,
                    drain_current=1e-4,
                    grid_shape=(5, 5, 5),
                )
                self.assertGreater(thermal.metrics["TMAX_K"], 300)
                self.assertGreater(electric.metrics["EMAX_V_M"], 0)
                self.assertGreater(magnetic.metrics["BMAX_EST_T"], 0)

    def test_mesh_node_element_and_memory_limits(self):
        rough = (9, 7, 7)
        optimal = (17, 13, 11)
        precise = (33, 25, 21)
        self.assertEqual(estimate_tetrahedron_count(rough), 1728)
        self.assertEqual(estimate_tetrahedron_count(optimal), 11520)
        self.assertEqual(estimate_tetrahedron_count(precise), 92160)
        self.assertEqual(estimate_mesh_memory_bytes((300, 100, 100)), 7_500_000_000)
        self.assertEqual(MAX_MESH_NODES, 3_000_000)
        with self.assertRaisesRegex(ValueError, "supported maximum"):
            _validate_grid_shape((300_001, 4, 4))


if __name__ == "__main__":
    unittest.main()
