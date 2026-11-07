import unittest

import numpy as np

from src.extractors import (
    CADTransistorExtractor,
    FDSOIExtractor,
    FinFETExtractor,
    GAANanosheetExtractor,
    GAANanowireExtractor,
    HEMTExtractor,
    PlanarMOSFETExtractor,
    TFETExtractor,
    create_extractor,
)
from src.model import DeviceParameters


class CompactExtractorTests(unittest.TestCase):
    def setUp(self):
        self.vg = np.linspace(0.0, 1.0, 21)
        self.vd_linear = 0.05
        self.vd_saturation = 0.8
        self.id_linear = 1e-12 + 2e-4 * np.maximum(self.vg - 0.35, 0) ** 2
        self.id_saturation = 1e-12 + 1e-4 * np.maximum(self.vg - 0.35, 0) ** 2
        self.cv_vg = np.linspace(-1, 1, 21)
        self.cgg = 0.4e-12 + 3.6e-12 / (1 + np.exp(-5 * self.cv_vg))

    def _device(self, family, geometry=None):
        return DeviceParameters(
            name="compact_test",
            epsilon_r=11.7,
            electron_mobility=1400,
            hole_mobility=450,
            bandgap=1.12,
            doping=1e16,
            tox_nm=5,
            width_um=1,
            length_um=0.18,
            device_type=family,
            geometry=geometry or {},
        )

    def _extract(self, extractor):
        return extractor.extract(
            v_graphs={
                "linear": self.vg.tolist(),
                "saturation": self.vg.tolist(),
            },
            c_graphs={"vg": self.cv_vg.tolist(), "cgg": self.cgg.tolist()},
            id_graphs={
                "linear": {
                    "vg": self.vg.tolist(),
                    "id": self.id_linear.tolist(),
                    "vd": self.vd_linear,
                },
                "saturation": {
                    "vg": self.vg.tolist(),
                    "id": self.id_saturation.tolist(),
                    "vd": self.vd_saturation,
                },
            },
        )

    def test_factory_uses_qcombobox_display_names(self):
        cases = (
            ("Планарный MOSFET (bulk)", "planar_mosfet", PlanarMOSFETExtractor),
            ("FD-SOI MOSFET", "fd_soi", FDSOIExtractor),
            ("FinFET", "finfet", FinFETExtractor),
            ("GAA nanosheet", "gaa_nanosheet", GAANanosheetExtractor),
            ("GAA nanowire", "gaa_nanowire", GAANanowireExtractor),
            ("TFET", "tfet", TFETExtractor),
            ("HEMT", "hemt", HEMTExtractor),
        )
        for display, family, expected in cases:
            with self.subTest(display=display):
                device = self._device(family)
                self.assertIsInstance(create_extractor(device, display), expected)

    def test_planar_and_fd_soi_emit_level_54_and_fit_cv_parameters(self):
        planar = create_extractor(self._device("planar_mosfet"))
        card = self._extract(planar)
        self.assertIn("level=54", card)
        self.assertIn("VTH0=", card)
        self.assertIn("U0=", card)
        self.assertIn("TOXE=", card)
        self.assertTrue(np.isfinite(planar.metrics["VFB_EST_V"]))
        self.assertGreater(planar.metrics["TOX_M"], 0)
        self.assertIn("CV_MIDPOINT_V", planar.metrics)

        fd_soi = create_extractor(
            self._device(
                "fd_soi",
                {
                    "body_thickness_nm": 7,
                    "buried_oxide_nm": 25,
                    "back_gate_voltage": 0,
                },
            ),
            "FD-SOI MOSFET",
        )
        fd_card = self._extract(fd_soi)
        self.assertIn("level=54", fd_card)
        self.assertIn("TBOX=25nm", fd_card)
        self.assertIn("BOX_COUPLING_EST", fd_soi.metrics)
        self.assertIn("VTH0_ZERO_BACK_GATE_EST_V", fd_soi.metrics)

    def test_cmg_families_emit_level_72_and_geometry(self):
        fin = create_extractor(
            self._device(
                "finfet",
                {
                    "fin_width_nm": 8,
                    "fin_height_nm": 40,
                    "fin_count": 2,
                    "fin_pitch_nm": 30,
                },
            )
        )
        fin_card = self._extract(fin)
        self.assertIn("level=72", fin_card)
        self.assertIn("HFIN=", fin_card)
        self.assertIn("NFIN=2", fin_card)

        sheet = create_extractor(
            self._device(
                "gaa_nanosheet",
                {
                    "sheet_width_nm": 30,
                    "sheet_thickness_nm": 8,
                    "sheet_count": 3,
                    "sheet_pitch_nm": 30,
                },
            )
        )
        self.assertIn("level=72", self._extract(sheet))
        self.assertGreater(sheet.metrics["GATE_PERIMETER_M"], 0)

        wire = create_extractor(
            self._device(
                "gaa_nanowire",
                {"wire_diameter_nm": 10, "wire_count": 3},
            )
        )
        self.assertIn("D=", self._extract(wire))
        self.assertAlmostEqual(wire.metrics["WIRE_RADIUS_M"], 5e-9)

    def test_tfet_hemt_and_cad_emit_behavioral_subcircuits(self):
        tfet = create_extractor(
            self._device(
                "tfet",
                {
                    "tunnel_length_nm": 20,
                    "channel_thickness_nm": 8,
                    "source_doping_cm3": 1e20,
                    "drain_doping_cm3": 1e18,
                },
            )
        )
        tfet_card = self._extract(tfet)
        self.assertIn(".subckt compact_test D G S", tfet_card)
        self.assertIn("POLY(2)", tfet_card)
        self.assertTrue(np.isfinite(tfet.metrics["SS_MV_DEC"]))

        hemt = create_extractor(
            self._device(
                "hemt",
                {
                    "barrier_thickness_nm": 20,
                    "sheet_density_cm2": 1e13,
                    "channel_mobility_cm2_vs": 1500,
                    "access_resistance_ohm_mm": 0.5,
                    "gate_recess_nm": 2,
                },
            )
        )
        hemt_card = self._extract(hemt)
        self.assertIn(".subckt compact_test D G S", hemt_card)
        self.assertGreater(hemt.metrics["N2DEG_M2"], 0)

        cad = create_extractor(self._device("planar_mosfet"), "CAD-транзистор")
        self.assertIsInstance(cad, CADTransistorExtractor)
        cad_card = self._extract(cad)
        self.assertIn(".subckt compact_test D G S", cad_card)
        self.assertIn("POLY(2)", cad_card)
        self.assertIn("POLY_5", cad.metrics)


if __name__ == "__main__":
    unittest.main()
