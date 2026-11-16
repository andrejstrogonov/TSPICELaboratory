"""Compact-model extraction backends for the desktop transistor workbench.

The fitted parameters are data-derived estimates. Only the BSIM cards use
standard model names; HEMT, TFET, and CAD exports are behavioral subcircuits.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Mapping, Sequence

try:
    from .model import (
        BOLTZMANN,
        DEVICE_TYPES,
        ELECTRON_CHARGE,
        EPSILON_0,
        DeviceParameters,
        Measurement,
        extract_idvg,
    )
except ImportError:
    from model import (
        BOLTZMANN,
        DEVICE_TYPES,
        ELECTRON_CHARGE,
        EPSILON_0,
        DeviceParameters,
        Measurement,
        extract_idvg,
    )


TRANSISTOR_TYPE_ALIASES = {
    "планарный mosfet (bulk)": "planar_mosfet",
    "planar_mosfet": "planar_mosfet",
    "fd-soi mosfet": "fd_soi",
    "fd_soi": "fd_soi",
    "finfet": "finfet",
    "gaa nanosheet": "gaa_nanosheet",
    "gaa_nanosheet": "gaa_nanosheet",
    "gaa nanowire": "gaa_nanowire",
    "gaa_nanowire": "gaa_nanowire",
    "tfet": "tfet",
    "hemt": "hemt",
    "мемристор": "memristor",
    "memristor": "memristor",
    "мемтранзистор": "memtransistor",
    "memtransistor": "memtransistor",
    "cad-транзистор": "cad_custom",
    "cad-транзистор (custom)": "cad_custom",
    "cad_custom": "cad_custom",
}

@dataclass(frozen=True)
class TransferSweep:
    gate_voltage: Sequence[float]
    drain_current: Sequence[float]
    drain_voltage: float | Sequence[float] | None = None


class BaseExtractor:
    """Shared data validation, fitting, metric extraction, and model rendering."""

    family = ""
    model_description = ""

    def __init__(
        self,
        device: DeviceParameters,
        transistor_type: str | None = None,
    ) -> None:
        self.device = device
        self.transistor_type = transistor_type or device.device_type
        self.family = _normalize_family(self.transistor_type)
        if self.family not in DEVICE_TYPES and self.family != "cad_custom":
            raise ValueError(f"Unsupported transistor type: {self.transistor_type!r}.")
        if self.family != "cad_custom" and device.device_type != self.family:
            raise ValueError(
                f"Selected transistor type {self.transistor_type!r} does not match "
                f"device parameters {device.device_type!r}."
            )
        self.metrics: dict[str, float] = {}
        self.model_notes: list[str] = []
        self._fit_sweeps: list[TransferSweep] = []

    def extract(
        self,
        v_graphs: Mapping[str, Sequence[float]] | None = None,
        c_graphs: Mapping[str, Sequence[float]] | None = None,
        id_graphs: Mapping[str, object] | None = None,
    ) -> str:
        """Fit curves and return a card.

        ``v_graphs`` maps sweep names to Vg arrays; ``id_graphs`` maps those
        names to dictionaries containing ``vg``, ``id`` and optional ``vd``.
        ``c_graphs`` contains paired ``vg`` and ``cgg`` arrays.
        """
        self.metrics = self.extract_metrics(v_graphs, c_graphs, id_graphs)
        card = self.render_spice(self.metrics)
        notes = "\n".join(f"* NOTE: {note}" for note in self.model_notes)
        return f"{card}\n{notes}\n"

    def extract_metrics(
        self,
        v_graphs: Mapping[str, Sequence[float]] | None = None,
        c_graphs: Mapping[str, Sequence[float]] | None = None,
        id_graphs: Mapping[str, object] | None = None,
    ) -> dict[str, float]:
        v_graphs = v_graphs or {}
        c_graphs = c_graphs or {}
        id_graphs = id_graphs or {}
        sweeps = _read_sweeps(v_graphs, id_graphs)
        if not sweeps:
            raise ValueError("Provide at least one Id-Vg transfer sweep.")
        self.device.validate()
        self._fit_sweeps = list(sweeps)

        linear = _select_sweep(sweeps, "linear")
        saturation = _select_sweep(sweeps, "saturation")
        preferred = linear or saturation or sweeps[0]
        vg, ids = _validated_transfer(preferred)
        metrics = _fit_common_transfer(self.device, vg, ids, preferred.drain_voltage)

        cgg_data = _read_cv(c_graphs)
        if cgg_data is not None:
            cv, cgg = cgg_data
            metrics.update(
                _fit_capacitance(self.device, cv, cgg, metrics["VTH0"])
            )
        else:
            metrics["TOX_M"] = self.device.tox_nm * 1e-9
            metrics["EOT_M"] = self.device.tox_nm * 1e-9
            metrics["VFB_EST_V"] = _flat_band_voltage(
                self.device, metrics["VTH0"], metrics["TOX_M"]
            )

        if linear is not None:
            linear_vg, linear_id = _validated_transfer(linear)
            vd = _representative_drain_voltage(linear.drain_voltage, linear_vg)
            gm, _threshold = _maximum_gm_threshold(linear_vg, linear_id)
            if vd > 0 and math.isfinite(gm):
                width_m = self.device.effective_width_um * 1e-6
                length_m = self.device.length_um * 1e-6
                cox = EPSILON_0 * 3.9 / metrics["EOT_M"]
                mobility_m2 = gm * length_m / (cox * width_m * vd)
                if math.isfinite(mobility_m2) and mobility_m2 > 0:
                    metrics["U0"] = mobility_m2 * 1e4
                    metrics["U0_LINEAR_CM2_VS"] = metrics["U0"]

        metrics.update(self._family_metrics(vg, ids, cgg_data))
        self.model_notes = self._notes()
        return metrics

    def _family_metrics(self, vg, ids, cgg_data) -> dict[str, float]:
        return {}

    def _notes(self) -> list[str]:
        return [
            "Parameters are estimates fitted to supplied curves; validate against "
            "the target simulator and measured device data."
        ]

    def render_spice(self, metrics: Mapping[str, float]) -> str:
        raise NotImplementedError

    def _render_behavioral_polynomial(self, metrics: Mapping[str, float]) -> str:
        raise NotImplementedError


class PlanarMOSFETExtractor(BaseExtractor):
    family = "planar_mosfet"
    model_description = "BSIM4 LEVEL=54"

    def render_spice(self, metrics: Mapping[str, float]) -> str:
        supported = ("VTH0", "U0", "NFACTOR", "CGSO", "CGDO")
        values = {key: metrics[key] for key in supported if key in metrics}
        values["TOXE"] = metrics["EOT_M"]
        lines = [
            f".model {_model_name(self.device.name)} nmos level=54 version=4.8.1"
        ]
        lines.extend(f"+ {key}={value:.8g}" for key, value in values.items())
        lines.append("* VFB_EST is reported in the extraction metrics, not a BSIM4 card parameter.")
        return "\n".join(lines) + "\n"


class FDSOIExtractor(PlanarMOSFETExtractor):
    family = "fd_soi"
    model_description = "BSIM4 LEVEL=54 (requested approximation for FD-SOI)"

    def _family_metrics(self, vg, ids, cgg_data) -> dict[str, float]:
        geometry = self.device.geometry
        if cgg_data is None:
            tox_m = self.device.tox_nm * 1e-9
        else:
            area_m2 = self.device.effective_width_um * self.device.length_um * 1e-12
            tox_m = EPSILON_0 * self.device.epsilon_r * area_m2 / max(cgg_data[1])
        tox_nm = tox_m * 1e9
        tbox_nm = geometry["buried_oxide_nm"]
        back_bias = geometry["back_gate_voltage"]
        coupling = min(tox_nm / tbox_nm, 1.0)
        shift = -coupling * back_bias
        _gm, measured_vth = _maximum_gm_threshold(vg, ids)
        return {
            "BOX_COUPLING_EST": coupling,
            "VTH_BACK_GATE_SHIFT_EST_V": shift,
            "VTH0_ZERO_BACK_GATE_EST_V": measured_vth - shift,
        }

    def render_spice(self, metrics: Mapping[str, float]) -> str:
        card = super().render_spice(metrics)
        box_nm = self.device.geometry["buried_oxide_nm"]
        body_nm = self.device.geometry["body_thickness_nm"]
        back_bias = self.device.geometry["back_gate_voltage"]
        return (
            card
            + f"* FD-SOI geometry: TSI={body_nm:.8g}nm TBOX={box_nm:.8g}nm "
            f"VBACK={back_bias:.8g}V\n"
            + f"* Simplified BOX coupling={metrics['BOX_COUPLING_EST']:.8g}; "
            f"VTH shift estimate={metrics['VTH_BACK_GATE_SHIFT_EST_V']:.8g}V.\n"
            + "* LEVEL=54 is not a calibrated FD-SOI model; BOX effect is only a first-order estimate.\n"
        )

    def _notes(self) -> list[str]:
        return super()._notes() + [
            "BSIM4 is a bulk MOSFET model, not the dedicated BSIM-SOI/BSIM-IMG "
            "formulation; BOX/back-gate values are reported but not solved by this card."
        ]


class FinFETExtractor(BaseExtractor):
    family = "finfet"
    model_description = "BSIM-CMG LEVEL=72"

    def _family_metrics(self, vg, ids, cgg_data) -> dict[str, float]:
        geometry = self.device.geometry
        return {
            "HFIN_M": geometry["fin_height_nm"] * 1e-9,
            "TFIN_M": geometry["fin_width_nm"] * 1e-9,
            "NFIN": geometry["fin_count"],
        }

    def render_spice(self, metrics: Mapping[str, float]) -> str:
        return _render_cmg_card(self.device, metrics, "finfet")


class GAANanosheetExtractor(BaseExtractor):
    family = "gaa_nanosheet"
    model_description = "BSIM-CMG LEVEL=72"

    def _family_metrics(self, vg, ids, cgg_data) -> dict[str, float]:
        geometry = self.device.geometry
        return {
            "HFIN_M": geometry["sheet_width_nm"] * 1e-9,
            "TFIN_M": geometry["sheet_thickness_nm"] * 1e-9,
            "NFIN": geometry["sheet_count"],
            "GATE_PERIMETER_M": 2
            * (geometry["sheet_width_nm"] + geometry["sheet_thickness_nm"])
            * geometry["sheet_count"]
            * 1e-9,
        }

    def render_spice(self, metrics: Mapping[str, float]) -> str:
        return _render_cmg_card(self.device, metrics, "gaa_nanosheet")


class GAANanowireExtractor(BaseExtractor):
    family = "gaa_nanowire"
    model_description = "BSIM-CMG LEVEL=72"

    def _family_metrics(self, vg, ids, cgg_data) -> dict[str, float]:
        diameter = self.device.geometry["wire_diameter_nm"] * 1e-9
        return {
            "WIRE_RADIUS_M": diameter / 2,
            "NFIN": self.device.geometry["wire_count"],
            "GATE_PERIMETER_M": (
                math.pi * diameter * self.device.geometry["wire_count"]
            ),
        }

    def render_spice(self, metrics: Mapping[str, float]) -> str:
        return _render_cmg_card(self.device, metrics, "gaa_nanowire")


class TFETExtractor(BaseExtractor):
    family = "tfet"
    model_description = "behavioral BTBT-fit subcircuit"

    def _family_metrics(self, vg, ids, cgg_data) -> dict[str, float]:
        import numpy as np

        current = abs(ids)
        positive = current[current > 0]
        if positive.size < 3:
            raise ValueError("TFET extraction requires at least three positive Id points.")
        cutoff = np_quantile(positive, 0.75)
        mask = (current > 0) & (current <= cutoff)
        if np.count_nonzero(mask) < 3:
            mask = current > 0
        slope = _fit_log_slope(vg[mask], current[mask])
        swing = abs(1 / slope) * 1000 if slope else math.inf
        if not math.isfinite(swing):
            raise ValueError("Could not estimate TFET subthreshold swing.")
        return {
            "VOFF_V": _maximum_gm_threshold(vg, current)[1],
            "SS_MV_DEC": swing,
            "ION_A": float(current.max()),
            "IOFF_A": float(positive.min()),
            **_polynomial_metrics(self._fit_sweeps),
        }

    def render_spice(self, metrics: Mapping[str, float]) -> str:
        return self._render_behavioral_polynomial(metrics)

    def _render_behavioral_polynomial(self, metrics: Mapping[str, float]) -> str:
        coefficients, bounds = _polynomial_from_metrics(metrics)
        return _subckt_card(
            self.device.name,
            "D G S",
            coefficients,
            f"TFET behavioral Id(Vg,Vd) polynomial; fitted Voff={metrics['VOFF_V']:.8g}V, "
            f"SS={metrics['SS_MV_DEC']:.8g}mV/dec; fit range Vg=[{bounds[0]:.8g},{bounds[1]:.8g}]V.",
        )

    def _notes(self) -> list[str]:
        return super()._notes() + [
            "The exported subcircuit is a polynomial behavioral fit, not a physical "
            "band-to-band tunneling model; it does not extrapolate reliably outside "
            "the fitted voltage range."
        ]


class HEMTExtractor(BaseExtractor):
    family = "hemt"
    model_description = "behavioral HEMT fit (not ASM-HEMT/MVSG)"

    def _family_metrics(self, vg, ids, cgg_data) -> dict[str, float]:
        metrics: dict[str, float] = {
            "VOFF_V": _maximum_gm_threshold(vg, abs(ids))[1],
            "GATE_RECESS_M": self.device.geometry["gate_recess_nm"] * 1e-9,
            "BARRIER_THICKNESS_M": self.device.geometry["barrier_thickness_nm"] * 1e-9,
        }
        if cgg_data is not None:
            plateau = float(np_quantile(cgg_data[1], 0.9))
            area_m2 = (
                self.device.effective_width_um
                * self.device.length_um
                * 1e-12
            )
            metrics["C2DEG_F"] = plateau
            metrics["N2DEG_M2"] = plateau / (ELECTRON_CHARGE * area_m2)
        else:
            metrics["N2DEG_M2"] = (
                self.device.geometry["sheet_density_cm2"] * 1e4
            )
        return {**metrics, **_polynomial_metrics(self._fit_sweeps)}

    def render_spice(self, metrics: Mapping[str, float]) -> str:
        return self._render_behavioral_polynomial(metrics)

    def _render_behavioral_polynomial(self, metrics: Mapping[str, float]) -> str:
        coefficients, bounds = _polynomial_from_metrics(metrics)
        return _subckt_card(
            self.device.name,
            "D G S",
            coefficients,
            f"HEMT behavioral Id(Vg,Vd) fit; Voff={metrics['VOFF_V']:.8g}V, "
            f"N2DEG={metrics['N2DEG_M2']:.8g}m^-2, "
            f"D={metrics['BARRIER_THICKNESS_M']:.8g}m, "
            f"Vg fit range=[{bounds[0]:.8g},{bounds[1]:.8g}]V.",
        )

    def _notes(self) -> list[str]:
        return super()._notes() + [
            "ASM-HEMT and MVSG are simulator-specific compact models, generally "
            "requiring their dedicated model implementation or Verilog-A source. "
            "This export is a measured-curve polynomial, not either physical model."
        ]


class CADTransistorExtractor(BaseExtractor):
    family = "cad_custom"
    model_description = "polynomial behavioral subcircuit"

    def _family_metrics(self, vg, ids, cgg_data) -> dict[str, float]:
        return {
            **_polynomial_metrics(self._fit_sweeps),
        }

    def render_spice(self, metrics: Mapping[str, float]) -> str:
        coefficients, bounds = _polynomial_from_metrics(metrics)
        return _subckt_card(
            self.device.name,
            "D G S",
            coefficients,
            f"CAD transistor polynomial behavioral model; valid over "
            f"Vg=[{bounds[0]:.8g},{bounds[1]:.8g}]V.",
        )

    def _notes(self) -> list[str]:
        return super()._notes() + [
            "The six-coefficient polynomial is a behavioral approximation over "
            "the measured Id-Vg/Vd range, not a geometry-derived compact model."
        ]


_EXTRACTOR_CLASSES = {
    "planar_mosfet": PlanarMOSFETExtractor,
    "fd_soi": FDSOIExtractor,
    "finfet": FinFETExtractor,
    "gaa_nanosheet": GAANanosheetExtractor,
    "gaa_nanowire": GAANanowireExtractor,
    "tfet": TFETExtractor,
    "hemt": HEMTExtractor,
    "cad_custom": CADTransistorExtractor,
}


def create_extractor(
    device: DeviceParameters, transistor_type: str | None = None
) -> BaseExtractor:
    """Choose an extractor from an internal family id or the QComboBox label."""
    family = _normalize_family(transistor_type or device.device_type)
    if family == "cad_custom":
        return CADTransistorExtractor(device, family)
    extractor = _EXTRACTOR_CLASSES.get(family)
    if extractor is None:
        raise ValueError(f"Unsupported transistor type: {transistor_type!r}.")
    return extractor(device, family)


def generate_memory_verilog_a(device: DeviceParameters) -> str:
    """Render a parameterized Verilog-A compact model for a memory device."""
    device.validate()
    if device.device_type not in ("memristor", "memtransistor"):
        raise ValueError("Verilog-A memory model requires a memristor or memtransistor.")

    geometry = device.geometry
    name = _model_name(device.name)
    common_parameters = (
        f"    parameter real Ron = {geometry['low_resistance_ohm']:.12g};",
        f"    parameter real Roff = {geometry['high_resistance_ohm']:.12g};",
        f"    parameter real Vset = {geometry['set_voltage_v']:.12g};",
        f"    parameter real Vreset = {geometry['reset_voltage_v']:.12g};",
        f"    parameter real mu_ion = {geometry['ion_mobility_cm2_vs'] * 1e-4:.12g}; // m^2/(V*s)",
        f"    parameter real D = {geometry['active_layer_nm'] * 1e-9:.12g}; // m",
        f"    parameter real Ea = {geometry['activation_energy_ev']:.12g}; // eV",
        "    parameter real Tnom = 300.0; // K",
        f"    parameter real x_init = {geometry['initial_state']:.12g};",
        "    parameter real kB_eV = 8.617333262145e-5; // eV/K",
        "    parameter integer window_order = 1;",
    )
    preamble = (
        "`include \"constants.vams\"",
        "`include \"disciplines.vams\"",
        "",
    )
    if device.device_type == "memristor":
        body = (
            f"module {name}(p, n);",
            "    inout p, n;",
            "    electrical p, n;",
            *common_parameters,
            "    real x, xlim, vmem, rmem, dxdt, mu_eff, window;",
            "",
            "    analog begin",
            "        vmem = V(p, n);",
            "        x = idt(dxdt, x_init);",
            "        xlim = min(max(x, 0.0), 1.0);",
            "        rmem = Ron*xlim + Roff*(1.0 - xlim);",
            "        mu_eff = mu_ion*limexp(-Ea/kB_eV*(1.0/$temperature - 1.0/Tnom));",
            "        dxdt = 0.0;",
            "        window = 0.0;",
            "        if (vmem >= Vset) begin",
            "            window = 1.0 - pow(xlim, 2*window_order);",
            "            dxdt = mu_eff*Ron/(D*D)*(vmem/rmem)*window;",
            "        end else if (vmem <= Vreset) begin",
            "            window = 1.0 - pow(1.0 - xlim, 2*window_order);",
            "            dxdt = mu_eff*Ron/(D*D)*(vmem/rmem)*window;",
            "        end",
            "        if ((x <= 0.0) && (dxdt < 0.0)) dxdt = 0.0;",
            "        if ((x >= 1.0) && (dxdt > 0.0)) dxdt = 0.0;",
            "        I(p, n) <+ vmem/rmem;",
            "    end",
            "endmodule",
        )
    else:
        body = (
            f"module {name}(d, g, s);",
            "    inout d, g, s;",
            "    electrical d, g, s;",
            *common_parameters,
            f"    parameter real Vth0 = {0.35:.12g}; // V",
            f"    parameter real Vscale = {0.1:.12g}; // V",
            f"    parameter real dVth = {0.25:.12g}; // V, state-dependent barrier shift",
            f"    parameter real gate_coupling = {geometry['gate_coupling']:.12g};",
            f"    parameter real tau_ret = {geometry['state_relaxation_ms'] * 1e-3:.12g}; // s",
            "    parameter real Goff = 1.0/Roff;",
            "    real x, xlim, vds, vgs, rmem, gate_factor, dxdt, mu_eff, window;",
            "",
            "    analog begin",
            "        vds = V(d, s);",
            "        vgs = V(g, s);",
            "        x = idt(dxdt, x_init);",
            "        xlim = min(max(x, 0.0), 1.0);",
            "        rmem = Ron*xlim + Roff*(1.0 - xlim);",
            "        gate_factor = 1.0/(1.0 + limexp(-(vgs - (Vth0 - dVth*xlim))/Vscale));",
            "        mu_eff = mu_ion*limexp(-Ea/kB_eV*(1.0/$temperature - 1.0/Tnom));",
            "        dxdt = -xlim/tau_ret;",
            "        window = 0.0;",
            "        if (vgs >= Vset) begin",
            "            window = 1.0 - pow(xlim, 2*window_order);",
            "            dxdt = mu_eff/(D*D)*vgs*window*(1.0 + gate_coupling*gate_factor);",
            "        end else if (vgs <= Vreset) begin",
            "            window = 1.0 - pow(1.0 - xlim, 2*window_order);",
            "            dxdt = mu_eff/(D*D)*vgs*window*(1.0 + gate_coupling*gate_factor);",
            "        end",
            "        if ((x <= 0.0) && (dxdt < 0.0)) dxdt = 0.0;",
            "        if ((x >= 1.0) && (dxdt > 0.0)) dxdt = 0.0;",
            "        I(d, s) <+ vds*(gate_factor/rmem + (1.0 - gate_factor)*Goff);",
            "    end",
            "endmodule",
        )
    return "\n".join((*preamble, *body, ""))


def generate_sol_gel_memristor_verilog_a(device: DeviceParameters) -> str:
    """Render a conservative threshold model for a sol-gel oxide memristor."""
    device.validate()
    if device.device_type != "memristor":
        raise ValueError("The sol-gel Verilog-A model requires a memristor.")

    geometry = device.geometry
    name = _model_name(device.name)
    body = (
        f"module {name}(p, n);",
        "    inout p, n;",
        "    electrical p, n;",
        f"    parameter real Ron = {geometry['low_resistance_ohm']:.12g};",
        f"    parameter real Roff = {geometry['high_resistance_ohm']:.12g};",
        f"    parameter real Vset = {geometry['set_voltage_v']:.12g};",
        f"    parameter real Vreset = {geometry['reset_voltage_v']:.12g};",
        f"    parameter real mu_ion = {geometry['ion_mobility_cm2_vs'] * 1e-4:.12g};",
        f"    parameter real D = {geometry['active_layer_nm'] * 1e-9:.12g};",
        f"    parameter real x_init = {geometry['initial_state']:.12g};",
        "    real x, xlim, vmem, gmem, dxdt, rate;",
        "",
        "    analog begin",
        "        vmem = V(p, n);",
        "        x = idt(dxdt, x_init);",
        "        xlim = x;",
        "        if (xlim < 0.0) xlim = 0.0;",
        "        if (xlim > 1.0) xlim = 1.0;",
        "        gmem = (1.0 - xlim)/Roff + xlim/Ron;",
        "        rate = mu_ion/(D*D);",
        "        dxdt = 0.0;",
        "        if (vmem >= Vset)",
        "            dxdt = rate*(vmem - Vset)*(1.0 - xlim);",
        "        else if (vmem <= Vreset)",
        "            dxdt = rate*(vmem - Vreset)*xlim;",
        "        if ((x <= 0.0) && (dxdt < 0.0)) dxdt = 0.0;",
        "        if ((x >= 1.0) && (dxdt > 0.0)) dxdt = 0.0;",
        "        I(p, n) <+ vmem*gmem;",
        "    end",
        "endmodule",
    )
    preamble = (
        "`include \"constants.vams\"",
        "`include \"disciplines.vams\"",
        "// Sol-gel metal-oxide model; x is the normalized vacancy-rich path fraction.",
        "// Positive terminal voltage drives SET; negative voltage drives RESET.",
        "",
    )
    return "\n".join((*preamble, *body, ""))


def _normalize_family(transistor_type: str) -> str:
    return TRANSISTOR_TYPE_ALIASES.get(transistor_type.strip().casefold(), "")


def _read_sweeps(
    v_graphs: Mapping[str, Sequence[float]], id_graphs: Mapping[str, object]
) -> list[TransferSweep]:
    result: list[TransferSweep] = []
    if "vg" in id_graphs and "id" in id_graphs:
        result.append(
            TransferSweep(
                id_graphs["vg"],
                id_graphs["id"],
                id_graphs.get("vd"),
            )
        )
        return result
    for key, raw in id_graphs.items():
        mode = (
            "linear"
            if str(key).startswith("linear")
            else "saturation"
            if str(key).startswith("saturation")
            else ""
        )
        if not mode:
            continue
        if isinstance(raw, Mapping):
            vg = raw.get(
                "vg",
                v_graphs.get(str(key), v_graphs.get(mode, v_graphs.get("vg"))),
            )
            ids = raw.get("id", raw.get("ids"))
            vd = raw.get("vd")
        elif isinstance(raw, (tuple, list)) and len(raw) in (2, 3):
            vg = v_graphs.get(str(key), v_graphs.get(mode, v_graphs.get("vg")))
            ids = raw[0] if len(raw) == 2 and vg is None else raw[1] if len(raw) == 2 else raw[1]
            if len(raw) == 2 and vg is None:
                vg = raw[0]
            vd = raw[2] if len(raw) == 3 else None
        else:
            raise ValueError(f"{mode} Id data must be a mapping or (Vg, Id[, Vd]).")
        if vg is None or ids is None:
            raise ValueError(f"{mode} sweep must provide both gate voltage and drain current.")
        result.append(TransferSweep(vg, ids, vd))
    return result


def _select_sweep(sweeps: Sequence[TransferSweep], mode: str) -> TransferSweep | None:
    for sweep in sweeps:
        vd = _representative_drain_voltage(sweep.drain_voltage, sweep.gate_voltage)
        if (mode == "linear" and 0 < vd <= 0.2) or (mode == "saturation" and vd > 0.2):
            return sweep
    return None


def _validated_transfer(sweep: TransferSweep):
    import numpy as np

    vg = np.asarray(sweep.gate_voltage, dtype=np.float64)
    ids = np.asarray(sweep.drain_current, dtype=np.float64)
    if vg.ndim != 1 or ids.ndim != 1 or vg.size != ids.size or vg.size < 5:
        raise ValueError("Id-Vg sweeps require at least five paired one-dimensional points.")
    if not np.all(np.isfinite(vg)) or not np.all(np.isfinite(ids)):
        raise ValueError("Transfer curves must contain only finite values.")
    if np.any(ids < 0):
        raise ValueError("Drain-current values must be non-negative.")
    order = np.argsort(vg)
    vg, ids = vg[order], ids[order]
    if np.unique(vg).size < 5 or np.any(np.diff(vg) <= 0):
        raise ValueError("Id-Vg sweeps must contain at least five distinct gate voltages.")
    return vg, ids


def _representative_drain_voltage(vd, vg) -> float:
    import numpy as np

    if vd is None:
        return min(0.05, max(float(np.ptp(vg)) * 0.05, 1e-3))
    values = np.asarray(vd, dtype=float)
    if not np.all(np.isfinite(values)) or np.any(values < 0):
        raise ValueError("Drain voltage must be finite and non-negative.")
    return float(np.median(values))


def _fit_common_transfer(device, vg, ids, vd) -> dict[str, float]:
    import numpy as np

    positive = ids[ids > 0]
    if positive.size < 3:
        raise ValueError("Id-Vg extraction requires at least three positive current points.")
    if len(vg) >= 5:
        drain_values = np.full_like(vg, _representative_drain_voltage(vd, vg))
        measurement = Measurement(
            kind="idvg",
            x=vg.tolist(),
            y=ids.tolist(),
            drain_voltage=drain_values.tolist(),
        )
        try:
            fitted = extract_idvg(measurement, device)
        except (ValueError, RuntimeError):
            fitted = {}
    else:
        fitted = {}
    gm, vth = _maximum_gm_threshold(vg, ids)
    fitted["VTH0"] = vth
    fitted["VTH_GM_V"] = vth
    fitted["GM_MAX_S"] = gm
    fitted["ION_A"] = float(np.max(ids))
    fitted["IOFF_A"] = float(np.min(positive))
    fitted["ION_IOFF"] = float(np.max(ids) / np.min(positive))
    return fitted


def _maximum_gm_threshold(vg, ids) -> tuple[float, float]:
    import numpy as np

    gm = np.gradient(ids, vg)
    finite = np.isfinite(gm) & (gm > 0)
    if not np.any(finite):
        raise ValueError("Could not find a positive transconductance in Id-Vg data.")
    indices = np.flatnonzero(finite)
    index = int(indices[np.argmax(gm[finite])])
    return float(gm[index]), float(vg[index] - ids[index] / gm[index])


def _read_cv(c_graphs: Mapping[str, Sequence[float]]):
    import numpy as np

    vg = c_graphs.get("vg", c_graphs.get("Vg"))
    cgg = c_graphs.get("cgg", c_graphs.get("Cgg"))
    if vg is None and cgg is None:
        return None
    if vg is None or cgg is None:
        raise ValueError("C-V input must contain both Vg and Cgg arrays.")
    vg = np.asarray(vg, dtype=float)
    cgg = np.asarray(cgg, dtype=float)
    if vg.ndim != 1 or cgg.ndim != 1 or vg.size != cgg.size or vg.size < 3:
        raise ValueError("C-V data require at least three paired points.")
    if not np.all(np.isfinite(vg)) or not np.all(np.isfinite(cgg)) or np.any(cgg <= 0):
        raise ValueError("C-V voltages must be finite and capacitances positive.")
    order = np.argsort(vg)
    return vg[order], cgg[order]


def _fit_capacitance(
    device: DeviceParameters, vg, cgg, threshold_voltage: float
) -> dict[str, float]:
    import numpy as np
    from scipy.optimize import least_squares

    area = device.effective_width_um * device.length_um * 1e-12
    if area <= 0:
        raise ValueError("Device gate area must be positive.")
    observed = float(np.max(cgg))
    oxide_estimate = EPSILON_0 * device.epsilon_r * area / observed
    voltage_span = max(float(np.ptp(vg)), 1e-3)
    capacitance_scale = max(observed, 1e-30)

    def cv_model(parameters):
        baseline, amplitude, midpoint, slope = parameters
        argument = np.clip((vg - midpoint) / slope, -60, 60)
        return baseline + amplitude / (1 + np.exp(-argument))

    fit = least_squares(
        lambda parameters: (cv_model(parameters) - cgg) / capacitance_scale,
        x0=[
            float(np.min(cgg)),
            max(observed - float(np.min(cgg)), observed * 1e-3),
            float(np.median(vg)),
            voltage_span / 10,
        ],
        bounds=(
            [0, 0, float(np.min(vg)) - voltage_span, 1e-6],
            [
                max(observed * 10, 1e-30),
                max(observed * 10, 1e-30),
                float(np.max(vg)) + voltage_span,
                voltage_span * 100,
            ],
        ),
        max_nfev=5000,
    )
    tox = float(oxide_estimate)
    eot = tox * 3.9 / device.epsilon_r
    phi_f, gamma = _mos_body_effect(device, tox)
    return {
        "CGG_MAX_F": observed,
        "TOX_M": tox,
        "EOT_M": eot,
        "VFB_EST_V": threshold_voltage - 2 * phi_f - gamma * math.sqrt(2 * phi_f),
        "VFB_FORMULA_OFFSET_V": 2 * phi_f + gamma * math.sqrt(2 * phi_f),
        "CV_MIDPOINT_V": float(fit.x[2]),
        "CV_TRANSITION_SLOPE_V": float(fit.x[3]),
    }


def _mos_body_effect(device: DeviceParameters, tox_m: float) -> tuple[float, float]:
    cox = EPSILON_0 * device.epsilon_r / tox_m
    phi_f = BOLTZMANN * device.temperature_k / ELECTRON_CHARGE * math.log(
        max(device.doping, 1.0) / 1.45e10
    )
    phi_f = max(phi_f, 0.0)
    silicon_permittivity = 11.7 * EPSILON_0
    gamma = math.sqrt(
        2 * ELECTRON_CHARGE * silicon_permittivity * device.doping * 1e6
    ) / cox
    return phi_f, gamma


def _flat_band_voltage(
    device: DeviceParameters, threshold_voltage: float, tox_m: float
) -> float:
    phi_f, gamma = _mos_body_effect(device, tox_m)
    return threshold_voltage - 2 * phi_f - gamma * math.sqrt(2 * phi_f)


def _fit_log_slope(vg, ids) -> float:
    import numpy as np
    from scipy.optimize import least_squares

    log_ids = np.log10(ids)
    center = float(np.mean(vg))
    result = least_squares(
        lambda parameters: parameters[0] + parameters[1] * (vg - center) - log_ids,
        x0=[
            float(np.mean(log_ids)),
            float(np.polyfit(vg - center, log_ids, 1)[0]),
        ],
    )
    return float(result.x[1])


def _fit_iv_polynomial(sweeps: Sequence[TransferSweep]):
    import numpy as np
    from scipy.optimize import least_squares

    gate_values = []
    drain_values = []
    current_values = []
    for sweep in sweeps:
        vg, ids = _validated_transfer(sweep)
        if sweep.drain_voltage is None:
            drain = np.full_like(vg, _representative_drain_voltage(None, vg))
        else:
            drain_values_for_sweep = np.asarray(sweep.drain_voltage, dtype=float)
            if drain_values_for_sweep.ndim == 0:
                drain = np.full_like(vg, float(drain_values_for_sweep))
            elif drain_values_for_sweep.size == vg.size:
                drain = drain_values_for_sweep[
                    np.argsort(np.asarray(sweep.gate_voltage))
                ]
            else:
                raise ValueError("Vd array must match the corresponding Id-Vg sweep.")
        gate_values.extend(vg.tolist())
        drain_values.extend(drain.tolist())
        current_values.extend(ids.tolist())

    vg_all = np.asarray(gate_values, dtype=float)
    vd_all = np.asarray(drain_values, dtype=float)
    ids_all = np.asarray(current_values, dtype=float)
    design = np.column_stack(
        (
            np.ones_like(vg_all),
            vg_all,
            vd_all,
            vg_all**2,
            vg_all * vd_all,
            vd_all**2,
        )
    )
    current_scale = max(float(np.max(np.abs(ids_all))), 1e-30)
    initial, *_ = np.linalg.lstsq(design, ids_all, rcond=None)
    fit = least_squares(
        lambda coefficients: (design @ coefficients - ids_all) / current_scale,
        x0=initial,
        max_nfev=5000,
    )
    if not fit.success and fit.status <= 0:
        raise RuntimeError(f"Behavioral polynomial fitting failed: {fit.message}")
    return (
        [float(value) for value in fit.x],
        (float(np.min(vg_all)), float(np.max(vg_all))),
    )


def _polynomial_metrics(sweeps: Sequence[TransferSweep]) -> dict[str, float]:
    coefficients, bounds = _fit_iv_polynomial(sweeps)
    return {
        **{f"POLY_{index}": value for index, value in enumerate(coefficients)},
        "VG_MIN_V": bounds[0],
        "VG_MAX_V": bounds[1],
    }


def _polynomial_from_metrics(metrics: Mapping[str, float]):
    coefficients = [metrics[f"POLY_{index}"] for index in range(6)]
    return coefficients, (metrics["VG_MIN_V"], metrics["VG_MAX_V"])


def _render_cmg_card(device, metrics, family):
    geometry = device.geometry
    if family == "finfet":
        dimensions = {
            "HFIN": metrics["HFIN_M"],
            "TFIN": metrics["TFIN_M"],
            "NFIN": metrics["NFIN"],
        }
    elif family == "gaa_nanosheet":
        dimensions = {
            "HFIN": metrics["HFIN_M"],
            "TFIN": metrics["TFIN_M"],
            "NFIN": metrics["NFIN"],
        }
    else:
        dimensions = {
            "D": 2 * metrics["WIRE_RADIUS_M"],
            "NFIN": metrics["NFIN"],
        }
    params = {
        "VTH0": metrics["VTH0"],
        "U0": metrics["U0"],
        "EOT": metrics["EOT_M"],
        **dimensions,
    }
    lines = [
        f".model {_model_name(device.name)} nmos level=72",
        "* BSIM-CMG geometry units are metres; verify parameters with simulator version.",
    ]
    lines.extend(f"+ {name}={value:.8g}" for name, value in params.items())
    return "\n".join(lines) + "\n"


def _subckt_card(name: str, pins: str, coefficients, comment: str) -> str:
    values = " ".join(f"{coefficient:.10g}" for coefficient in coefficients)
    safe_name = _model_name(name)
    return (
        f"* {comment}\n"
        f".subckt {safe_name} {pins}\n"
        f"GDS D S POLY(2) G S D S {values}\n"
        f".ends {safe_name}\n"
    )


def _model_name(name: str) -> str:
    sanitized = re.sub(r"[^A-Za-z0-9_.$]", "_", name)
    if not sanitized or not sanitized[0].isalpha():
        sanitized = f"Q_{sanitized}"
    return sanitized


def np_quantile(values, quantile: float) -> float:
    import numpy as np

    return float(np.quantile(values, quantile))
