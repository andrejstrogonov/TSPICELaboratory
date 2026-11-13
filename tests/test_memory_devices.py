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
from src.model import (
    DEVICE_GEOMETRY_FIELDS,
    DEVICE_TYPES,
    DeviceParameters,
    Measurement,
    extract_curve_metrics,
    generate_memory_verilog_a,
    generate_neuristor_verilog_a,
    generate_sol_gel_memristor_verilog_a,
    load_measurement,
)
from src.simulation import solve_memory_characteristics, solve_neuristor_characteristics


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
                self.assertTrue(window.generate_memory_button.isEnabled())
                window._generate_memory_curves()
                self.assertEqual(len(window.measurements), expected_curve_count)
                self.assertGreater(len(window.axes.lines), 0)
                selected = window.measurements[0]
                metrics = extract_curve_metrics(selected, device)
                hysteresis_key = (
                    "MEMRISTOR_HYSTERESIS_AREA_A_V"
                    if family == "memristor"
                    else "MEMTRANSISTOR_HYSTERESIS_AREA_A_V"
                )
                self.assertGreater(metrics[hysteresis_key], 0)
                window.measurements.clear()
                window.measurement_list.clear()

    def test_memory_curves_are_disabled_for_non_memory_devices(self):
        window = TransistorWorkbench()
        self.addCleanup(window.close)

        window.device_type.setCurrentIndex(
            window.device_type.findData("planar_mosfet")
        )
        self.assertFalse(window.generate_memory_button.isEnabled())

    def test_bundled_curves_models_and_drawings_load_and_calculate(self):
        examples = Path("data/examples/memory")
        memristor = load_cad_device_model(examples / "memristor_model.json")
        memtransistor = load_cad_device_model(examples / "memtransistor_model.json")
        neuristor = load_cad_device_model(examples / "neuristor_model.json")
        self.assertEqual(memristor.device_type, "memristor")
        self.assertEqual(memtransistor.device_type, "memtransistor")
        self.assertEqual(neuristor.device_type, "neuristor")

        for drawing in ("memristor_stack.stl", "memtransistor_stack.stl"):
            vertices, faces = load_stl(examples / drawing)
            self.assertGreater(len(vertices), 0)
            self.assertGreater(len(faces), 0)

        csv_files = sorted(examples.glob("*.csv"))
        self.assertEqual(len(csv_files), 12)
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
        self.assertEqual(
            sum(measurement.kind == "neuristor_spikes" for measurement in measurements),
            3,
        )
        for measurement in measurements:
            if measurement.kind == "memory_idvg":
                model = memtransistor
            elif measurement.kind == "neuristor_spikes":
                model = neuristor
            else:
                model = memristor
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
            ("neuristor", {"neuristor_spikes"}),
        ):
            window.device_type.setCurrentIndex(window.device_type.findData(family))
            matching_files = [
                str(path)
                for path, measurement in zip(csv_files, measurements)
                if measurement.kind in kinds
            ]
            window._load_curve_paths(matching_files)
            self.assertEqual(len(window.measurements), len(matching_files))
            for row in range(len(matching_files)):
                window.measurement_list.setCurrentRow(row)
                window._fit_selected()
                self.assertTrue(window.extracted)
            window.measurements.clear()
            window.measurement_list.clear()

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
                self.assertEqual(window._device_parameters().device_type, family)
                window._generate_memory_curves()
                self.assertGreater(len(window.measurements), 0)
                window.measurements.clear()
                window.measurement_list.clear()

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

    def test_sol_gel_memristor_model_uses_thresholded_vacancy_drift(self):
        model = generate_sol_gel_memristor_verilog_a(
            self._memory_device("memristor")
        )

        self.assertIn("module memory_test(p, n);", model)
        self.assertIn("parameter real D = 1e-08;", model)
        self.assertIn("gmem = (1.0 - xlim)/Roff + xlim/Ron;", model)
        self.assertIn("rate = mu_ion/(D*D);", model)
        self.assertIn("if (vmem >= Vset)", model)
        self.assertIn("else if (vmem <= Vreset)", model)
        self.assertIn("I(p, n) <+ vmem*gmem;", model)
        self.assertNotIn("limexp", model)
        with self.assertRaisesRegex(ValueError, "requires a memristor"):
            generate_sol_gel_memristor_verilog_a(
                self._memory_device("memtransistor")
            )

    def test_sol_gel_model_is_selectable_in_ui_and_exported(self):
        window = TransistorWorkbench()
        self.addCleanup(window.close)
        self.assertTrue(window.memory_model_method.isHidden())
        window.device_type.setCurrentIndex(
            window.device_type.findData("memristor")
        )
        self.assertFalse(window.memory_model_method.isHidden())
        self.assertEqual(window.memory_model_method.findData("sol_gel"), 1)
        window.memory_model_method.setCurrentIndex(
            window.memory_model_method.findData("sol_gel")
        )
        with tempfile.TemporaryDirectory() as directory:
            path = str(Path(directory) / "sol_gel.va")
            with patch(
                "src.main.QFileDialog.getSaveFileName", return_value=(path, "")
            ):
                window._export_model()
            self.assertIn(
                "rate = mu_ion/(D*D);",
                Path(path).read_text(encoding="utf-8"),
            )

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


class NeuristorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.application = QApplication.instance() or QApplication([])

    def _neuristor(self, **overrides):
        geometry = {
            "active_layer_nm": 50,
            "substrate_oxide_nm": 300,
            "line_segments": 12,
            "cold_resistance_ohm": 20000,
            "hot_resistance_ohm": 200,
            "threshold_voltage_v": 1.2,
            "hold_voltage_v": 0.6,
            "load_resistance_ohm": 5000,
            "line_capacitance_pf": 2,
            "thermal_time_ns": 50,
            "refractory_time_ns": 500,
            "activation_energy_ev": 0.4,
        }
        geometry.update(overrides)
        return DeviceParameters(
            name="neuristor_test",
            epsilon_r=11.7,
            electron_mobility=1400,
            hole_mobility=450,
            bandgap=1.12,
            doping=1e16,
            tox_nm=5,
            width_um=1,
            length_um=0.18,
            device_type="neuristor",
            geometry=geometry,
        )

    def test_neuristor_is_registered_as_a_device_type(self):
        self.assertIn("neuristor", DEVICE_TYPES)
        self.assertIn("cold_resistance_ohm", DEVICE_GEOMETRY_FIELDS["neuristor"])
        self._neuristor().validate()

    def test_neuristor_geometry_validation_rejects_nonphysical_values(self):
        invalid = (
            ({"hold_voltage_v": 1.5}, "Hold voltage must be smaller"),
            ({"hot_resistance_ohm": 30000}, "Hot resistance must be smaller"),
            ({"line_segments": 2.5}, "Line segments must be an integer"),
            ({"refractory_time_ns": 10}, "Refractory time must be longer"),
        )
        for overrides, message in invalid:
            with self.subTest(overrides=overrides):
                with self.assertRaisesRegex(ValueError, message):
                    self._neuristor(**overrides).validate()

    def test_neuristor_emits_spikes_above_threshold_only(self):
        device = self._neuristor()
        subthreshold = solve_neuristor_characteristics(
            device, supply_min_v=0.5, supply_max_v=0.6, supplies=2
        )
        for curve in subthreshold.curves:
            self.assertEqual(sum(curve.columns["memory_state"]), 0)

        suprathreshold = solve_neuristor_characteristics(
            device, supply_min_v=2.0, supply_max_v=3.0, supplies=2
        )
        for curve in suprathreshold.curves:
            self.assertGreater(sum(curve.columns["memory_state"]), 0)

    def test_neuristor_spike_peak_is_clipped_at_the_threshold(self):
        device = self._neuristor()
        result = solve_neuristor_characteristics(
            device, supply_min_v=3.0, supply_max_v=9.0, supplies=3
        )
        threshold = device.geometry["threshold_voltage_v"]
        for curve in result.curves:
            peak = max(curve.columns["voltage"])
            self.assertLess(peak, threshold * 1.1)
            self.assertTrue(
                all(math.isfinite(value) for value in curve.columns["voltage"])
            )

    def test_neuristor_spike_rate_increases_with_supply(self):
        device = self._neuristor()
        result = solve_neuristor_characteristics(
            device,
            supply_min_v=2.0,
            supply_max_v=6.0,
            supplies=4,
            duration_ns=5000,
            points=10000,
        )
        periods = []
        for curve in result.curves:
            state = np.asarray(curve.columns["memory_state"])
            starts = np.flatnonzero(np.diff(state) == 1)
            self.assertGreaterEqual(len(starts), 2)
            step = curve.x[1] - curve.x[0]
            periods.append(float(np.mean(np.diff(starts))) * step)
        self.assertTrue(
            all(earlier > later for earlier, later in zip(periods, periods[1:])),
            f"Spike period must shorten as the supply rises: {periods}",
        )
        self.assertGreater(
            result.diagnostics["SPIKE_RATE_MAX_HZ"],
            result.diagnostics["SPIKE_RATE_HZ"],
        )

    def test_bundled_example_traces_reproduce_the_generator(self):
        examples = Path("data/examples/memory")
        device = load_cad_device_model(examples / "neuristor_model.json")
        generated = solve_neuristor_characteristics(
            device,
            supply_min_v=2.0,
            supply_max_v=5.0,
            supplies=3,
            duration_ns=5000,
            points=2001,
        )
        for curve in generated.curves:
            supply = curve.columns["supply_voltage"][0]
            path = examples / f"neuristor_spikes_{supply:g}vs.csv"
            self.assertTrue(path.exists(), f"missing example trace {path.name}")
            imported = load_measurement(path)
            self.assertEqual(imported.kind, "neuristor_spikes")
            np.testing.assert_allclose(imported.x, curve.x, rtol=0, atol=1e-18)
            np.testing.assert_allclose(imported.y, curve.y, rtol=1e-9, atol=1e-12)
            self.assertGreater(sum(imported.columns["memory_state"]), 0)

    def test_subthreshold_supply_produces_no_spikes(self):
        device = self._neuristor()
        critical = device.geometry["threshold_voltage_v"] * (
            device.geometry["cold_resistance_ohm"]
            + device.geometry["load_resistance_ohm"]
        ) / device.geometry["cold_resistance_ohm"]
        result = solve_neuristor_characteristics(
            device, supply_min_v=0.5, supply_max_v=critical * 0.9, supplies=2
        )
        self.assertAlmostEqual(result.diagnostics["CRITICAL_SUPPLY_V"], critical)
        self.assertEqual(result.diagnostics["SPIKE_RATE_MAX_HZ"], 0.0)
        for curve in result.curves:
            self.assertEqual(sum(curve.columns["memory_state"]), 0)

    def test_neuristor_verilog_a_renders_threshold_parameters(self):
        text = generate_neuristor_verilog_a(self._neuristor())
        self.assertIn("module neuristor_test", text)
        self.assertIn("Vt = 1.2", text)
        self.assertIn("Vh = 0.6", text)
        self.assertIn("Rhot = 200", text)
        self.assertIn("t_ref", text)
        with self.assertRaises(ValueError):
            generate_neuristor_verilog_a(
                DeviceParameters(
                    name="mos",
                    epsilon_r=11.7,
                    electron_mobility=1400,
                    hole_mobility=450,
                    bandgap=1.12,
                    doping=1e16,
                    tox_nm=5,
                    width_um=1,
                    length_um=0.18,
                )
            )

    def test_neuristor_curves_and_export_work_from_the_ui(self):
        window = TransistorWorkbench()
        self.addCleanup(window.close)
        window.device_type.setCurrentIndex(window.device_type.findData("neuristor"))
        window.model_name.setText("neuristor_test")
        self.assertTrue(window.generate_memory_button.isEnabled())
        self.assertIn("Verilog-A", window.export_button.text())

        window._generate_memory_curves()
        self.assertGreater(len(window.measurements), 0)
        self.assertEqual(
            {item.kind for item in window.measurements}, {"neuristor_spikes"}
        )
        self.assertGreater(len(window.axes.lines), 0)
        self.assertIn("THRESHOLD_V", window.result_text.toPlainText())

        with tempfile.TemporaryDirectory() as directory:
            path = str(Path(directory) / "neuristor.va")
            with patch(
                "src.main.QFileDialog.getSaveFileName", return_value=(path, "")
            ):
                window._export_model()
            text = Path(path).read_text(encoding="utf-8")
        self.assertIn("module neuristor_test", text)
        self.assertIn("Vt = 1.2", text)


if __name__ == "__main__":
    unittest.main()
