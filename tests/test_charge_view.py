import os
import unittest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt6.QtWidgets import QApplication

from src.main import TransistorWorkbench
from src.simulation import solve_quantum_charge_distribution


class ChargeViewTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.application = QApplication.instance() or QApplication([])

    def test_default_2d_charge_field_renders_without_voxel_renderer(self):
        window = TransistorWorkbench()
        self.addCleanup(window.close)
        solution = solve_quantum_charge_distribution(
            window._device_parameters(),
            gate_voltage=window.simulation_fields["gate_voltage"].value(),
            drain_voltage=window.simulation_fields["drain_voltage"].value(),
            dimension="2d",
            grid_shape=(17, 7, 17),
            use_gpu=False,
        )
        window.simulation_solutions["carrier_charge"] = solution
        window.view_mode.setCurrentIndex(
            window.view_mode.findData("carrier_charge")
        )
        window.simulation_canvas.draw()

        self.assertIsNotNone(window.simulation_colorbar)
        self.assertGreaterEqual(len(window.simulation_axes.collections), 5)
        self.assertEqual(solution.values.shape, (17, 2, 17))


if __name__ == "__main__":
    unittest.main()
