import math
import os
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
from PyQt6.QtWidgets import QApplication

from src.main import TransistorWorkbench
from src.cad import load_cad_device_model, load_stl
from src.extractors import generate_memory_verilog_a
from src.model import (
    DeviceParameters,
    Measurement,
    extract_curve_metrics,
    load_measurement,
)
from src.simulation import solve_memory_characteristics


class MemoryDeviceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.application = QApplication.instance() or QApplication([])

    def setUp(self):
        self.base_device = DeviceParameters(
            name="memory_test",
            epsilon_r=11.7,
            electron_mobility=1400,
            hole_mobility=450,
            bandgap=1.12,
            doping=1e16,
            tox_nm=5,
            width_um=1,
            length_um=0.18,
        )

    def _memory_device(self, family):
        geometry = {
            "active_layer_nm": 10,
            "substrate_oxide_nm": 100,
            "low_resistance_ohm": 100,
            "high_resistance_ohm": 100000,
            "set_voltage_v": 0.3,
            "reset_voltage_v": -0.3,
            "activation_energy_ev": 0.6,
            "initial_state": 0.5,
        }
        if family == "memtransistor":
            geometry["state_relaxation_ms"] = 100
            geometry["gate_coupling"] = 2
            geometry["ion_mobility_cm2_vs"] = 1e-12
        else:
            geometry["ion_mobility_cm2_vs"] = 1e-10
        return replace(
            self.base_device,
            device_type=family,
            geometry=geometry,
        )

    def test_memristor_generates_hysteretic_iv_and_resistance_curves(self):
        result = solve_memory_characteristics(self._memory_device("memristor"))

        self.assertEqual(len(result.curves), 6)
        iv_curves = [curve for curve in result.curves if curve.kind == "memristor_iv"]
        resistance_curves = [
            curve for curve in result.curves if curve.kind == "memristor_resistance"
        ]
        self.assertEqual(len(iv_curves), 3)
        self.assertEqual(len(resistance_curves), 3)
        for curve in result.curves:
            self.assertTrue(all(math.isfinite(value) for value in curve.x + curve.y))
        loop = iv_curves[0]
        self.assertEqual(loop.x[0], loop.x[-1])
        self.assertGreater(abs(loop.y[150] - loop.y[250]), 0)
        self.assertTrue(
            all(0 <= state <= 1 for state in loop.columns["memory_state"])
        )
        metrics = extract_curve_metrics(
            Measurement(
                kind=loop.kind,
                x=loop.x,
                y=loop.y,
                label=loop.label,
                columns=loop.columns,
            ),
            self._memory_device("memristor"),
        )
        self.assertGreater(metrics["MEMRISTOR_HYSTERESIS_AREA_A_V"], 0)

    def test_memtransistor_generates_rate_dependent_memory_transfer_curves(self):
        result = solve_memory_characteristics(self._memory_device("memtransistor"))

        self.assertEqual(len(result.curves), 3)
        self.assertTrue(all(curve.kind == "memory_idvg" for curve in result.curves))
        currents = [curve.y for curve in result.curves]
        self.assertGreater(abs(currents[0][150] - currents[0][250]), 0)
        self.assertFalse(np.array_equal(currents[0], currents[-1]))
        for curve in result.curves:
            self.assertTrue(all(value >= 0 for value in curve.y))
            self.assertTrue(
                all(0 <= state <= 1 for state in curve.columns["memory_state"])
            )
        curve = result.curves[0]
        metrics = extract_curve_metrics(
            Measurement(
                kind=curve.kind,
                x=curve.x,
                y=curve.y,
                label=curve.label,
                columns=curve.columns,
            ),
            self._memory_device("memtransistor"),
        )
        self.assertGreater(metrics["MEMTRANSISTOR_HYSTERESIS_AREA_A_V"], 0)

    def test_memristive_devices_are_available_and_plot_generated_curves_in_ui(self):
        window = TransistorWorkbench()
        self.addCleanup(window.close)

        for family, expected_curve_count in (("memristor", 6), ("memtransistor", 3)):
            with self.subTest(family=family):
                window.device_type.setCurrentIndex(window.device_type.findData(family))
                device = window._device_parameters()
                self.assertFalse(window.pde_gpu_check.isEnabled())
                self.assertFalse(window.charge_button.isEnabled())
                self.assertTrue(window.pde_button.isEnabled())
                self.assertTrue(window.solve_button.isEnabled())
                window.pde_button.click()
                worker = window.characteristics_worker
                self.assertTrue(worker.wait(5000))
                self.application.processEvents()
                self.assertEqual(len(window.measurements), expected_curve_count)
                self.assertEqual(
                    window.characteristic_selector.currentData(),
                    "memristor_iv" if family == "memristor" else "memory_idvg",
                )
                self.assertGreater(len(window.axes.lines), 0)
                selected = window.measurements[0]
                metrics = extract_curve_metrics(selected, device)
                hysteresis_key = (
                    "MEMRISTOR_HYSTERESIS_AREA_A_V"
                    if family == "memristor"
                    else "MEMTRANSISTOR_HYSTERESIS_AREA_A_V"
                )
                self.assertGreater(metrics[hysteresis_key], 0)
                window._clear_measurements()

    def test_3d_field_simulation_runs_from_ui_for_both_memory_devices(self):
        window = TransistorWorkbench()
        self.addCleanup(window.close)
        window.gpu_check.setChecked(False)

        for family in ("memristor", "memtransistor"):
            with self.subTest(family=family):
                window.device_type.setCurrentIndex(window.device_type.findData(family))
                window._run_3d_simulations()
                worker = window.simulation_worker
                self.assertTrue(worker.wait(10000))
                self.application.processEvents()
                self.assertEqual(
                    set(window.simulation_solutions),
                    {"thermal", "electric", "magnetic"},
                )
                self.assertGreater(
                    window.simulation_solutions["thermal"].metrics["TMAX_K"], 300
                )
                self.assertIn(
                    "слоистой диэлектрической",
                    window.simulation_solutions["electric"].description,
                )
                self.assertGreater(
                    window.simulation_solutions["magnetic"].metrics["BMAX_EST_T"],
                    0,
                )
                window.simulation_solutions.clear()

    def test_bundled_curves_models_and_drawings_load_and_calculate(self):
        examples = Path("data/examples/memory")
        memristor = load_cad_device_model(examples / "memristor_model.json")
        memtransistor = load_cad_device_model(examples / "memtransistor_model.json")
        self.assertEqual(memristor.device_type, "memristor")
        self.assertEqual(memtransistor.device_type, "memtransistor")

        for drawing in ("memristor_stack.stl", "memtransistor_stack.stl"):
            vertices, faces = load_stl(examples / drawing)
            self.assertGreater(len(vertices), 0)
            self.assertGreater(len(faces), 0)

        csv_files = sorted(examples.glob("*.csv"))
        self.assertEqual(len(csv_files), 9)
        measurements = [load_measurement(path) for path in csv_files]
        self.assertEqual(
            sum(measurement.kind == "memristor_iv" for measurement in measurements),
            3,
        )
        self.assertEqual(
            sum(
                measurement.kind == "memristor_resistance"
                for measurement in measurements
            ),
            3,
        )
        self.assertEqual(
            sum(measurement.kind == "memory_idvg" for measurement in measurements),
            3,
        )
        for measurement in measurements:
            model = memtransistor if measurement.kind == "memory_idvg" else memristor
            self.assertTrue(extract_curve_metrics(measurement, model))
        generated = solve_memory_characteristics(
            memristor,
            voltage_min_v=-1.0,
            voltage_max_v=1.0,
            points_per_sweep=201,
        )
        generated_iv = next(
            curve for curve in generated.curves if curve.kind == "memristor_iv"
        )
        imported_iv = load_measurement(examples / "memristor_iv_10vs.csv")
        np.testing.assert_allclose(imported_iv.x, generated_iv.x, rtol=0, atol=1e-14)
        np.testing.assert_allclose(imported_iv.y, generated_iv.y, rtol=1e-12, atol=0)

        window = TransistorWorkbench()
        self.addCleanup(window.close)
        for family, kinds in (
            ("memristor", {"memristor_iv", "memristor_resistance"}),
            ("memtransistor", {"memory_idvg"}),
        ):
            window.device_type.setCurrentIndex(window.device_type.findData(family))
            matching_files = [
                str(path)
                for path, measurement in zip(csv_files, measurements)
                if measurement.kind in kinds
            ]
            window._load_curve_paths(matching_files)
            window._fit_all()
            self.assertEqual(len(window.analysis_results), len(matching_files))
            self.assertTrue(
                all(error is None for _metrics, error in window.analysis_results.values())
            )
            window._clear_measurements()

        for family, drawing, model in (
            ("memristor", "memristor_stack.stl", memristor),
            ("memtransistor", "memtransistor_stack.stl", memtransistor),
        ):
            with self.subTest(imported_family=family):
                vertices, faces = load_stl(examples / drawing)
                window.cad_geometry = {
                    "vertices": vertices,
                    "faces": faces,
                    "path": str(examples / drawing),
                }
                window._apply_cad_device_model(model)
                window.device_type.setCurrentIndex(
                    window.device_type.findData("cad_custom")
                )
                window.geometry_redraw_timer.stop()
                self.assertEqual(window._device_parameters().device_type, family)
                window.pde_button.click()
                self.assertTrue(window.characteristics_worker.wait(5000))
                self.application.processEvents()
                self.assertGreater(len(window.measurements), 0)
                window._clear_measurements()

    def test_verilog_a_models_include_state_equations_and_parameters(self):
        memristor = generate_memory_verilog_a(self._memory_device("memristor"))
        self.assertIn("module memory_test(p, n);", memristor)
        self.assertIn("x = idt(dxdt, x_init);", memristor)
        self.assertIn("parameter real Ron = 100;", memristor)
        self.assertIn("parameter real Roff = 100000;", memristor)
        self.assertIn("parameter real Ea = 0.6;", memristor)
        self.assertIn("I(p, n) <+ vmem/rmem;", memristor)

        memtransistor = generate_memory_verilog_a(
            self._memory_device("memtransistor")
        )
        self.assertIn("module memory_test(d, g, s);", memtransistor)
        self.assertIn("parameter real gate_coupling = 2;", memtransistor)
        self.assertIn("if (vgs >= Vset)", memtransistor)
        self.assertIn("else if (vgs <= Vreset)", memtransistor)
        self.assertIn("I(d, s) <+ vds*(gate_factor/rmem", memtransistor)

    def test_verilog_a_export_saves_generated_model(self):
        window = TransistorWorkbench()
        self.addCleanup(window.close)
        window.device_type.setCurrentIndex(
            window.device_type.findData("memristor")
        )
        with tempfile.TemporaryDirectory() as directory:
            path = str(Path(directory) / "device.va")
            with patch(
                "src.main.QFileDialog.getSaveFileName", return_value=(path, "")
            ):
                window._export_model()
            self.assertIn("module", Path(path).read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
