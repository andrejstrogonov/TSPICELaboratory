"""CSV handling, compact-model fitting, and SPICE export."""

from __future__ import annotations

import csv
import math
import re
from dataclasses import dataclass, field
from pathlib import Path

EPSILON_0 = 8.8541878128e-12
BOLTZMANN = 1.380649e-23
ELECTRON_CHARGE = 1.602176634e-19
DEVICE_TYPES = (
    "planar_mosfet",
    "fd_soi",
    "finfet",
    "gaa_nanosheet",
    "gaa_nanowire",
    "tfet",
    "hemt",
    "memristor",
    "memtransistor",
)
DEVICE_GEOMETRY_FIELDS = {
    "planar_mosfet": (),
    "fd_soi": ("body_thickness_nm", "buried_oxide_nm", "back_gate_voltage"),
    "finfet": ("fin_width_nm", "fin_height_nm", "fin_count", "fin_pitch_nm"),
    "gaa_nanosheet": (
        "sheet_width_nm",
        "sheet_thickness_nm",
        "sheet_count",
        "sheet_pitch_nm",
    ),
    "gaa_nanowire": ("wire_diameter_nm", "wire_count"),
    "tfet": (
        "tunnel_length_nm",
        "channel_thickness_nm",
        "source_doping_cm3",
        "drain_doping_cm3",
    ),
    "hemt": (
        "barrier_thickness_nm",
        "sheet_density_cm2",
        "channel_mobility_cm2_vs",
        "access_resistance_ohm_mm",
        "gate_recess_nm",
    ),
    "memristor": (
        "active_layer_nm",
        "substrate_oxide_nm",
        "ion_mobility_cm2_vs",
        "low_resistance_ohm",
        "high_resistance_ohm",
        "set_voltage_v",
        "reset_voltage_v",
        "activation_energy_ev",
        "initial_state",
    ),
    "memtransistor": (
        "active_layer_nm",
        "substrate_oxide_nm",
        "ion_mobility_cm2_vs",
        "low_resistance_ohm",
        "high_resistance_ohm",
        "state_relaxation_ms",
        "set_voltage_v",
        "reset_voltage_v",
        "activation_energy_ev",
        "gate_coupling",
        "initial_state",
    ),
}


@dataclass(frozen=True)
class Measurement:
    kind: str
    x: list[float]
    y: list[float]
    drain_voltage: list[float] | None = None
    label: str = ""
    gate_bias: float | None = None
    columns: dict[str, list[float]] = field(default_factory=dict)


@dataclass(frozen=True)
class DeviceParameters:
    name: str
    epsilon_r: float
    electron_mobility: float
    hole_mobility: float
    bandgap: float
    doping: float
    tox_nm: float
    width_um: float
    length_um: float
    temperature_k: float = 300.0
    device_type: str = "planar_mosfet"
    geometry: dict[str, float] = field(default_factory=dict)
    barrier_material: str = "AlGaN/GaN"
    junction_material: str = "Si/Si"

    def validate(self) -> None:
        if self.device_type not in DEVICE_TYPES:
            raise ValueError(f"Unsupported transistor type: {self.device_type}.")
        positive = {
            "relative permittivity": self.epsilon_r,
            "electron mobility": self.electron_mobility,
            "hole mobility": self.hole_mobility,
            "bandgap": self.bandgap,
            "doping": self.doping,
            "oxide thickness": self.tox_nm,
            "width": self.width_um,
            "length": self.length_um,
            "temperature": self.temperature_k,
        }
        missing = set(DEVICE_GEOMETRY_FIELDS[self.device_type]) - self.geometry.keys()
        if missing:
            raise ValueError(
                f"Missing {self.device_type} parameters: {', '.join(sorted(missing))}."
            )
        nonnegative = {
            "access_resistance_ohm_mm",
            "gate_recess_nm",
            "gate_coupling",
            "initial_state",
        }
        signed = {"back_gate_voltage", "reset_voltage_v"}
        positive.update(
            (key, value)
            for key, value in self.geometry.items()
            if key not in {*signed, *nonnegative}
        )
        for key in nonnegative & self.geometry.keys():
            value = self.geometry[key]
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"{key.replace('_', ' ').capitalize()} must be non-negative.")
        for key in ("fin_count", "sheet_count", "wire_count"):
            if key in self.geometry and not float(self.geometry[key]).is_integer():
                raise ValueError(f"{key.replace('_', ' ').capitalize()} must be an integer.")
        if (
            self.device_type == "finfet"
            and self.geometry["fin_pitch_nm"] < self.geometry["fin_width_nm"]
        ):
            raise ValueError("Fin pitch must not be smaller than fin width.")
        if (
            self.device_type == "gaa_nanosheet"
            and self.geometry["sheet_pitch_nm"] < self.geometry["sheet_thickness_nm"]
        ):
            raise ValueError("Nanosheet pitch must not be smaller than sheet thickness.")
        for key in signed & self.geometry.keys():
            if not math.isfinite(self.geometry[key]):
                raise ValueError(f"{key.replace('_', ' ').capitalize()} must be finite.")
        for label, value in positive.items():
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{label.replace('_', ' ').capitalize()} must be a positive finite number.")
        if not self.name.strip() or any(ch.isspace() for ch in self.name):
            raise ValueError("Model name must be non-empty and contain no spaces.")
        if self.device_type == "hemt" and not self.barrier_material.strip():
            raise ValueError("HEMT barrier material must not be empty.")
        if self.device_type == "tfet" and not self.junction_material.strip():
            raise ValueError("TFET junction material must not be empty.")
        if self.device_type in ("memristor", "memtransistor"):
            if self.geometry["low_resistance_ohm"] >= self.geometry["high_resistance_ohm"]:
                raise ValueError("Low resistance must be smaller than high resistance.")
            if self.geometry["set_voltage_v"] <= 0:
                raise ValueError("SET threshold voltage must be positive.")
            if self.geometry["reset_voltage_v"] >= 0:
                raise ValueError("RESET threshold voltage must be negative.")
            if self.geometry["initial_state"] > 1:
                raise ValueError("Initial memory state must be between zero and one.")

    @property
    def effective_width_um(self) -> float:
        """Return the total conducting perimeter used by the first-order fit."""
        self.validate()
        if self.device_type == "finfet":
            perimeter_nm = self.geometry["fin_count"] * (
                2 * self.geometry["fin_height_nm"] + self.geometry["fin_width_nm"]
            )
            return perimeter_nm * 1e-3
        if self.device_type == "gaa_nanosheet":
            perimeter_nm = self.geometry["sheet_count"] * (
                2 * self.geometry["sheet_width_nm"]
                + 2 * self.geometry["sheet_thickness_nm"]
            )
            return perimeter_nm * 1e-3
        if self.device_type == "gaa_nanowire":
            perimeter_nm = (
                math.pi
                * self.geometry["wire_diameter_nm"]
                * self.geometry["wire_count"]
            )
            return perimeter_nm * 1e-3
        return self.width_um

    @property
    def oxide_capacitance_per_area(self) -> float:
        self.validate()
        return EPSILON_0 * self.epsilon_r / (self.tox_nm * 1e-9)


def load_measurement(path: str | Path) -> Measurement:
    """Load a headed CSV/whitespace Silvaco log curve."""
    file_path = Path(path)
    text = file_path.read_text(encoding="utf-8-sig")
    lines = [
        line.strip()
        for line in text.splitlines()
        if line.strip() and not line.lstrip().startswith(("#", "*", "//"))
    ]
    if len(lines) < 2:
        raise ValueError("Measurement file must contain a header and data rows.")
    delimiter = _measurement_delimiter(lines[0])
    if delimiter:
        parsed_rows = list(csv.reader(lines, delimiter=delimiter))
    else:
        header_tokens = re.findall(r"\S+(?:\s*\([^)]*\))?", lines[0])
        parsed_rows = [header_tokens] + [
            re.split(r"\s+", line) for line in lines[1:]
        ]
    headers = [header.strip() for header in parsed_rows[0]]
    if len(headers) < 2:
        raise ValueError("Measurement file must have at least two columns.")
    normalized_headers = [_canonical_measurement_column(header) for header in headers]
    if any(not header for header in normalized_headers):
        raise ValueError("Measurement file has an empty or unsupported column name.")
    if len(set(normalized_headers)) != len(normalized_headers):
        raise ValueError("Measurement file contains duplicate equivalent columns.")
    columns = {header: [] for header in normalized_headers}
    for line_number, row in enumerate(parsed_rows[1:], start=2):
        if len(row) != len(headers):
            continue
        try:
            values = [float(value) for value in row]
        except ValueError:
            if row and row[0].strip() == headers[0]:
                continue
            raise ValueError(f"Invalid numeric value on line {line_number}.")
        if not all(math.isfinite(value) for value in values):
            raise ValueError(f"Non-finite measurement on line {line_number}.")
        for key, value in zip(normalized_headers, values):
            columns[key].append(value)

    columns = {key: values for key, values in columns.items() if values}
    aliases = {
        "voltage": ("voltage", "v"),
        "current": ("current", "i"),
        "gate_voltage": ("vg", "vgs"),
        "drain_voltage": ("vd", "vds"),
        "drain_current": ("id", "ids"),
        "gate_current": ("ig", "igs"),
        "capacitance": ("cgg", "cg"),
        "frequency": ("frequency", "freq", "f"),
        "h21": ("h21", "current_gain", "abs_h21"),
        "mason_u": ("mason_u", "unilateral_gain", "u_gain", "u"),
        "vbe": ("vbe",),
        "vce": ("vce",),
        "collector_current": ("ic",),
        "base_current": ("ib",),
        "time": ("time", "t"),
        "resistance": ("resistance", "r"),
        "memory_state": ("memory_state", "state", "x"),
        "sweep_rate": ("sweep_rate", "rate"),
        "collector_capacitance": ("cjc",),
        "emitter_capacitance": ("cje",),
        "gate_source_capacitance": ("cgs",),
        "gate_drain_capacitance": ("cgd",),
        "gate_bulk_capacitance": ("cgb",),
        "voltage_gain": ("voltage_gain", "av", "s21"),
        "z21": ("z21",),
        "y21": ("y21",),
    }
    for canonical, names in aliases.items():
        if canonical not in columns:
            source = next((name for name in names if name in columns), None)
            if source:
                columns[canonical] = columns[source]
    gate = columns.get("gate_voltage")
    drain = columns.get("drain_voltage")
    current = columns.get("drain_current")
    gate_current = columns.get("gate_current")
    capacitance = columns.get("capacitance")
    frequency = columns.get("frequency")
    vbe, vce = columns.get("vbe"), columns.get("vce")
    collector = columns.get("collector_current")
    base = columns.get("base_current")
    time_values = columns.get("time")
    gate_bias = None
    if vbe and collector and base:
        kind, x_values, y_values = "gummel", vbe, collector
    elif vce and collector:
        kind, x_values, y_values = "bjt_output", vce, collector
    elif columns.get("voltage") and columns.get("current") and columns.get("memory_state"):
        kind, x_values, y_values = (
            "memristor_iv",
            columns["voltage"],
            columns["current"],
        )
    elif (
        time_values
        and columns.get("resistance")
        and columns.get("memory_state")
    ):
        kind, x_values, y_values = (
            "memristor_resistance",
            time_values,
            columns["resistance"],
        )
    elif gate and current and columns.get("memory_state"):
        kind, x_values, y_values = "memory_idvg", gate, current
    elif time_values and gate_current:
        kind, x_values, y_values = "gate_charge", time_values, gate_current
    elif time_values and current and gate:
        kind, x_values, y_values = "transient", time_values, current
    elif gate and gate_current:
        kind, x_values, y_values = "igvg", gate, gate_current
    elif frequency and any(
        key in columns
        for key in (
            "h21",
            "h21_db",
            "mason_u",
            "mason_u_db",
            "s11",
            "s11_db",
            "s12",
            "s12_db",
            "s21",
            "s21_db",
            "s22",
            "s22_db",
            "y11",
            "y12",
            "z21",
            "y21",
            "y22",
            "z11",
            "z12",
            "z22",
        )
    ):
        kind, x_values, y_values = "ac", frequency, next(
            columns[key]
            for key in (
                "h21",
                "h21_db",
                "mason_u",
                "mason_u_db",
                "s11",
                "s11_db",
                "s12",
                "s12_db",
                "s21",
                "s21_db",
                "s22",
                "s22_db",
                "y11",
                "y12",
                "z21",
                "y21",
                "y22",
                "z11",
                "z12",
                "z22",
            )
            if key in columns
        )
    elif current and gate and drain and len(set(gate)) == 1 and len(set(drain)) > 1:
        kind, x_values, y_values = "idvd", drain, current
        gate_bias = gate[0]
    elif gate and capacitance:
        kind, x_values, y_values = "cv", gate, capacitance
    elif current and gate:
        kind, x_values, y_values = "idvg", gate, current
    elif current and drain:
        kind, x_values, y_values = "idvd", drain, current
        if gate and len(set(gate)) == 1:
            gate_bias = gate[0]
    else:
        raise ValueError(
            "Unsupported columns. Supply a headed Id-Vg/Id-Vd/C-V, Ig-Vg, "
            "Gummel, BJT output, AC gain, transient, or gate-current-vs-time dataset."
        )

    if len(x_values) < 3:
        raise ValueError("At least three measurement points are required.")
    if len(set(x_values)) < 3:
        raise ValueError("At least three distinct x-axis values are required.")
    drain_values = columns.get("drain_voltage") if kind == "idvg" else None
    return Measurement(
        kind=kind,
        x=list(x_values),
        y=list(y_values),
        drain_voltage=drain_values,
        label=file_path.name,
        gate_bias=gate_bias,
        columns=columns,
    )


def _measurement_delimiter(header: str) -> str | None:
    try:
        return csv.Sniffer().sniff(header, delimiters=",;\t").delimiter
    except csv.Error:
        return None


def _canonical_measurement_column(header: str) -> str:
    normalized = _normalize_header(header)
    aliases = {
        "vgs": "gate_voltage",
        "gatevoltage": "gate_voltage",
        "vg": "gate_voltage",
        "voltage": "voltage",
        "v": "voltage",
        "vds": "drain_voltage",
        "drainvoltage": "drain_voltage",
        "vd": "drain_voltage",
        "ids": "drain_current",
        "draincurrent": "drain_current",
        "id": "drain_current",
        "current": "current",
        "i": "current",
        "igs": "gate_current",
        "gatecurrent": "gate_current",
        "ig": "gate_current",
        "cgg": "capacitance",
        "cg": "capacitance",
        "capacitance": "capacitance",
        "frequency": "frequency",
        "freq": "frequency",
        "f": "frequency",
        "currentgain": "h21",
        "absh21": "h21",
        "h21": "h21",
        "h21db": "h21_db",
        "unilateralgain": "mason_u",
        "masongain": "mason_u",
        "udb": "mason_u_db",
        "ugain": "mason_u",
        "u": "mason_u",
        "vbe": "vbe",
        "vce": "vce",
        "ic": "collector_current",
        "collectorcurrent": "collector_current",
        "ib": "base_current",
        "basecurrent": "base_current",
        "time": "time",
        "t": "time",
        "resistance": "resistance",
        "r": "resistance",
        "memorystate": "memory_state",
        "state": "memory_state",
        "x": "memory_state",
        "sweeprate": "sweep_rate",
        "rate": "sweep_rate",
        "cgs": "gate_source_capacitance",
        "cgd": "gate_drain_capacitance",
        "cgb": "gate_bulk_capacitance",
        "cjc": "collector_capacitance",
        "cje": "emitter_capacitance",
        "voltagegain": "voltage_gain",
        "av": "voltage_gain",
        "s11": "s11",
        "s12": "s12",
        "s21": "s21",
        "s22": "s22",
        "s11mag": "s11",
        "s12mag": "s12",
        "s21mag": "s21",
        "s22mag": "s22",
        "s11db": "s11_db",
        "s12db": "s12_db",
        "s21db": "s21_db",
        "s22db": "s22_db",
        "y11": "y11",
        "y12": "y12",
        "y21": "y21",
        "y22": "y22",
        "z21": "z21",
        "z11": "z11",
        "z12": "z12",
        "z22": "z22",
    }
    return aliases.get(normalized, normalized)


def extract_idvg(measurement: Measurement, device: DeviceParameters) -> dict[str, float]:
    """Fit threshold voltage, effective mobility, and subthreshold factor."""
    if measurement.kind != "idvg":
        raise ValueError("Id-Vg extraction requires a gate-voltage/current dataset.")
    if len(measurement.x) < 5:
        raise ValueError("At least five Id-Vg points are required for fitting.")
    device.validate()
    try:
        import numpy as np
        from scipy.optimize import least_squares
    except ImportError as error:
        raise RuntimeError("Install NumPy and SciPy to fit measurement data.") from error

    vg = np.asarray(measurement.x, dtype=float)
    measured = np.asarray(measurement.y, dtype=float)
    vd = np.asarray(
        measurement.drain_voltage
        if measurement.drain_voltage
        else [0.05] * len(measurement.x),
        dtype=float,
    )
    if np.any(measured < 0):
        raise ValueError("Drain current values must be non-negative.")
    ut = BOLTZMANN * device.temperature_k / ELECTRON_CHARGE
    cox = device.oxide_capacitance_per_area
    width = device.effective_width_um * 1e-6
    length = device.length_um * 1e-6
    current_floor = max(float(np.max(measured)) * 1e-12, 1e-18)

    def predicted(params):
        threshold, log_mobility, factor = params
        mobility = np.exp(log_mobility)
        slope = max(factor, 0.5) * ut
        argument = np.clip((vg - threshold) / (2 * slope), -60, 60)
        softplus = np.logaddexp(0.0, argument)
        channel = (mobility * cox * width / length) * 2 * slope**2 * softplus**2
        drain_factor = 1 - np.exp(-np.maximum(vd, 0) / ut)
        return np.maximum(channel * drain_factor, current_floor)

    initial_threshold = float(np.median(vg))
    initial_mobility = device.electron_mobility * 1e-4

    def residual(params):
        return np.log10(predicted(params)) - np.log10(np.maximum(measured, current_floor))

    result = least_squares(
        residual,
        x0=[initial_threshold, math.log(initial_mobility), 1.5],
        bounds=(
            [float(np.min(vg)) - 2, math.log(1e-6), 0.5],
            [float(np.max(vg)) + 2, math.log(10), 10],
        ),
        max_nfev=5000,
    )
    if not result.success and result.status <= 0:
        raise RuntimeError(f"Id-Vg fitting failed: {result.message}")
    threshold, log_mobility, factor = result.x
    return {
        "VTH0": float(threshold),
        "U0": float(np.exp(log_mobility) * 1e4),
        "NFACTOR": float(factor),
        "RMSE_LOG10": float(np.sqrt(np.mean(residual(result.x) ** 2))),
    }


def extract_cv(measurement: Measurement, device: DeviceParameters) -> dict[str, float]:
    """Summarize measured gate capacitance and estimate overlap capacitance."""
    if measurement.kind != "cv":
        raise ValueError("C-V extraction requires a gate-voltage/capacitance dataset.")
    device.validate()
    positive = [value for value in measurement.y if value > 0]
    if not positive:
        raise ValueError("C-V data must contain positive capacitance values.")
    sorted_values = sorted(positive)
    high_bias = sorted_values[max(0, int(len(sorted_values) * 0.75) - 1) :]
    cgg = sum(high_bias) / len(high_bias)
    effective_width = device.effective_width_um
    intrinsic = (
        device.oxide_capacitance_per_area * effective_width * device.length_um * 1e-12
    )
    overlap_per_width = max(cgg - intrinsic, 0.0) / (2 * effective_width * 1e-6)
    return {
        "CGG": cgg,
        "CGSO": overlap_per_width,
        "CGDO": overlap_per_width,
        "CINTRINSIC": intrinsic,
    }


def extract_transfer_metrics(
    measurement: Measurement, device: DeviceParameters
) -> dict[str, float]:
    """Extract data-derived transfer-curve metrics for TFET and HEMT devices."""
    if measurement.kind != "idvg":
        raise ValueError("Transfer metrics require a gate-voltage/current dataset.")
    if len(measurement.x) < 5:
        raise ValueError("At least five Id-Vg points are required.")
    device.validate()
    try:
        import numpy as np
    except ImportError as error:
        raise RuntimeError("Install NumPy to analyze transfer measurements.") from error

    vg = np.asarray(measurement.x, dtype=float)
    ids = np.asarray(measurement.y, dtype=float)
    if np.any(ids < 0):
        raise ValueError("Drain current values must be non-negative.")
    order = np.argsort(vg)
    vg, ids = vg[order], ids[order]
    positive = ids[ids > 0]
    if positive.size < 3:
        raise ValueError("At least three positive current measurements are required.")

    current_floor = max(float(np.max(positive)) * 1e-15, 1e-30)
    log_ids = np.log10(np.maximum(ids, current_floor))
    threshold_current = 1e-7 * device.effective_width_um / device.length_um
    threshold = _interpolate_log_current(
        vg, np.maximum(ids, current_floor), threshold_current
    )
    log_steps = np.diff(log_ids)
    voltage_steps = np.diff(vg)
    slopes = np.divide(
        voltage_steps,
        log_steps,
        out=np.full_like(log_steps, np.inf),
        where=(log_steps != 0) & (voltage_steps > 0),
    )
    # Avoid an on-current derivative being reported as the subthreshold swing.
    subthreshold = (ids[1:] <= np.max(positive) * 1e-2) & (
        ids[:-1] <= np.max(positive) * 1e-2
    )
    valid = subthreshold & np.isfinite(slopes) & (slopes > 0)
    if not np.any(valid):
        valid = np.isfinite(slopes) & (slopes > 0)
    if not np.any(valid):
        raise ValueError("Could not estimate subthreshold swing from this Id-Vg sweep.")
    gm_slopes = np.divide(
        np.diff(ids),
        voltage_steps,
        out=np.full_like(voltage_steps, np.nan),
        where=voltage_steps > 0,
    )
    finite_gm = gm_slopes[np.isfinite(gm_slopes)]
    if finite_gm.size == 0:
        raise ValueError("Could not estimate transconductance from this Id-Vg sweep.")
    result = {
        "VTH_CONSTANT_CURRENT": threshold,
        "ION": float(np.max(ids)),
        "IOFF": float(np.min(positive)),
        "ION_IOFF": float(np.max(ids) / np.min(positive)),
        "SS_MV_DEC": float(np.min(slopes[valid]) * 1000),
        "GM_MAX": float(np.max(finite_gm)),
    }
    if device.device_type == "hemt":
        sheet_density = device.geometry["sheet_density_cm2"] * 1e4
        mobility = device.geometry["channel_mobility_cm2_vs"] * 1e-4
        result["RSH_OHM_SQ"] = 1 / (ELECTRON_CHARGE * sheet_density * mobility)
    return result


def extract_output_metrics(
    measurement: Measurement, device: DeviceParameters
) -> dict[str, float]:
    """Summarize an output curve with on-resistance and output conductance."""
    if measurement.kind != "idvd":
        raise ValueError("Output metrics require a drain-voltage/current dataset.")
    if len(measurement.x) < 3:
        raise ValueError("At least three Id-Vd points are required.")
    device.validate()
    try:
        import numpy as np
    except ImportError as error:
        raise RuntimeError("Install NumPy to analyze output measurements.") from error

    vd = np.asarray(measurement.x, dtype=float)
    ids = np.asarray(measurement.y, dtype=float)
    if np.any(ids < 0):
        raise ValueError("Drain current values must be non-negative.")
    order = np.argsort(vd)
    vd, ids = vd[order], ids[order]
    positive = ids > 0
    if np.count_nonzero(positive) < 2:
        raise ValueError("At least two positive current measurements are required.")
    low_bias = (vd > 0) & (vd <= np.max(vd) * 0.2) & positive
    if np.count_nonzero(low_bias) < 2:
        low_bias = positive
    high_bias = positive & (vd >= np.max(vd) * 0.8)
    if np.count_nonzero(high_bias) < 2:
        high_bias = positive
    if len(np.unique(vd[low_bias])) < 2 or len(np.unique(vd[high_bias])) < 2:
        raise ValueError("Id-Vd data must contain at least two distinct low- and high-bias points.")
    ron_slope = float(np.polyfit(vd[low_bias], ids[low_bias], 1)[0])
    output_conductance = float(np.polyfit(vd[high_bias], ids[high_bias], 1)[0])
    if ron_slope <= 0:
        raise ValueError("The low-bias Id-Vd slope must be positive to estimate RON.")
    result = {
        "IDS_MAX": float(np.max(ids)),
        "VDS_AT_MAX": float(vd[np.argmax(ids)]),
        "RON_OHM": 1 / ron_slope,
        "GDS_S": output_conductance,
    }
    if measurement.gate_bias is not None:
        result["VGS_BIAS"] = measurement.gate_bias
    if device.device_type == "hemt":
        width_mm = device.width_um * 1e-3
        result["RACCESS_OHM"] = (
            device.geometry["access_resistance_ohm_mm"] / width_mm
        )
    return result


def extract_curve_metrics(
    measurement: Measurement, device: DeviceParameters
) -> dict[str, float]:
    """Extract measurable metrics from imported device-simulation curves."""
    import numpy as np

    device.validate()
    result: dict[str, float] = {}
    x = np.asarray(measurement.x, dtype=np.float64)
    y = np.asarray(measurement.y, dtype=np.float64)
    order = np.argsort(x)
    x, y = x[order], y[order]
    columns = {
        key: np.asarray(values, dtype=np.float64)
        for key, values in measurement.columns.items()
    }

    if measurement.kind == "idvg":
        if len(x) < 5:
            raise ValueError("Id-Vg analysis requires at least five data points.")
        if device.device_type in ("tfet", "hemt"):
            result.update(extract_transfer_metrics(measurement, device))
        else:
            result.update(extract_idvg(measurement, device))
            result.update(_extract_current_sweep_metrics(x, y, device))
    elif measurement.kind == "idvd":
        result.update(extract_output_metrics(measurement, device))
        knee = _breakdown_knee_voltage(x, y)
        if knee is not None:
            result["VBR_KNEE_V"] = knee
    elif measurement.kind == "cv":
        result.update(extract_cv(measurement, device))
        for key, metric in (
            ("gate_source_capacitance", "CGS_MEAN_F"),
            ("gate_drain_capacitance", "CGD_MEAN_F"),
            ("gate_bulk_capacitance", "CGB_MEAN_F"),
            ("collector_capacitance", "CJC_MEAN_F"),
            ("emitter_capacitance", "CJE_MEAN_F"),
        ):
            if key in columns:
                result[metric] = float(np.mean(np.abs(columns[key])))
    elif measurement.kind == "igvg":
        leakage = np.abs(y)
        positive = leakage[leakage > 0]
        if not positive.size:
            raise ValueError("Ig-Vg curve must contain a non-zero gate current.")
        result.update(
            {
                "IG_MAX_A": float(np.max(leakage)),
                "IG_AT_MAX_GATE_A": float(leakage[np.argmax(x)]),
                "IG_MIN_NONZERO_A": float(np.min(positive)),
                "IG_MAX_MIN_RATIO": float(np.max(positive) / np.min(positive)),
            }
        )
    elif measurement.kind == "gummel":
        ic = np.abs(columns["collector_current"])
        ib = np.abs(columns["base_current"])
        beta = np.divide(ic, ib, out=np.zeros_like(ic), where=ib > 0)
        if not np.any(beta > 0):
            raise ValueError("Gummel data must include positive collector/base current.")
        result.update(
            {
                "IC_MAX_A": float(np.max(ic)),
                "IB_MAX_A": float(np.max(ib)),
                "BETA_MAX": float(np.max(beta)),
                "BETA_AT_MAX_IC": float(beta[np.argmax(ic)]),
            }
        )
        for key, metric in (
            ("collector_capacitance", "CJC_MEAN_F"),
            ("emitter_capacitance", "CJE_MEAN_F"),
        ):
            if key in columns:
                result[metric] = float(np.mean(np.abs(columns[key])))
    elif measurement.kind == "bjt_output":
        ic = np.abs(y)
        result["IC_MAX_A"] = float(np.max(ic))
        if len(x) >= 3 and np.ptp(x) > 0:
            conductance = np.gradient(y[order], x[order])
            result["GCE_MAX_S"] = float(np.max(np.abs(conductance)))
            knee = _breakdown_knee_voltage(x, y)
            if knee is not None:
                result["VBR_KNEE_V"] = knee
        for key, metric in (
            ("collector_capacitance", "CJC_MEAN_F"),
            ("emitter_capacitance", "CJE_MEAN_F"),
        ):
            if key in columns:
                result[metric] = float(np.mean(np.abs(columns[key])))
    elif measurement.kind == "ac":
        frequency = np.asarray(columns["frequency"], dtype=np.float64)
        if np.any(frequency <= 0):
            raise ValueError("AC frequencies must be positive.")
        gains = {
            "h21": np.asarray(columns["h21"], dtype=np.float64)
            if "h21" in columns
            else 10 ** (np.asarray(columns["h21_db"]) / 20)
            if "h21_db" in columns
            else None,
            "mason_u": np.asarray(columns["mason_u"], dtype=np.float64)
            if "mason_u" in columns
            else 10 ** (np.asarray(columns["mason_u_db"]) / 10)
            if "mason_u_db" in columns
            else None,
        }
        for key, metric in (("h21", "FT_HZ"), ("mason_u", "FMAX_HZ")):
            if gains[key] is not None:
                crossing = _unity_gain_frequency(frequency, gains[key])
                if crossing is not None:
                    result[metric] = crossing
        if gains["h21"] is not None:
            result["H21_MAX"] = float(np.max(np.abs(gains["h21"])))
        if gains["mason_u"] is not None:
            result["MASON_U_MAX"] = float(np.max(np.abs(gains["mason_u"])))
        for parameter in (
            "s11",
            "s12",
            "s21",
            "s22",
            "y11",
            "y12",
            "y21",
            "y22",
            "z11",
            "z12",
            "z21",
            "z22",
        ):
            db_key = f"{parameter}_db"
            if parameter in columns:
                result[f"{parameter.upper()}_MAX"] = float(
                    np.max(np.abs(columns[parameter]))
                )
            elif db_key in columns:
                linear = 10 ** (columns[db_key] / 20)
                result[f"{parameter.upper()}_MAX"] = float(np.max(np.abs(linear)))
    elif measurement.kind == "transient":
        time = np.asarray(columns["time"], dtype=np.float64)
        if np.any(np.diff(time) <= 0):
            raise ValueError("Transient time values must be strictly increasing.")
        result.update(_extract_transient_metrics(time, y))
        if "gate_voltage" in columns:
            result.update(
                {
                    "VG_MIN_V": float(np.min(columns["gate_voltage"])),
                    "VG_MAX_V": float(np.max(columns["gate_voltage"])),
                }
            )
    elif measurement.kind == "gate_charge":
        time = np.asarray(columns["time"], dtype=np.float64)
        if np.any(np.diff(time) <= 0):
            raise ValueError("Gate-current time values must be strictly increasing.")
        ig = np.asarray(columns["gate_current"], dtype=np.float64)
        result["QG_C"] = float(abs(np.sum((ig[1:] + ig[:-1]) * np.diff(time) * 0.5)))
        if "gate_voltage" in columns:
            result["VG_SWING_V"] = float(
                np.max(columns["gate_voltage"]) - np.min(columns["gate_voltage"])
            )
    elif measurement.kind == "memristor_iv":
        voltage = np.asarray(columns.get("voltage", measurement.x), dtype=np.float64)
        current = np.asarray(columns.get("current", measurement.y), dtype=np.float64)
        resistance = np.asarray(columns["resistance"], dtype=np.float64)
        if voltage.size < 3 or voltage.size != current.size:
            raise ValueError("Memristor I-V data requires at least three paired points.")
        area = abs(
            float(
                np.sum(
                    voltage * np.roll(current, -1)
                    - current * np.roll(voltage, -1)
                )
            )
        ) * 0.5
        result.update(
            {
                "MEMRISTOR_R_ON_OHM": float(np.min(resistance)),
                "MEMRISTOR_R_OFF_OHM": float(np.max(resistance)),
                "MEMRISTOR_ON_OFF_RATIO": float(np.max(resistance) / np.min(resistance)),
                "MEMRISTOR_HYSTERESIS_AREA_A_V": area,
                "MEMRISTOR_I_MAX_A": float(np.max(np.abs(current))),
            }
        )
    elif measurement.kind == "memristor_resistance":
        resistance = np.asarray(columns.get("resistance", y), dtype=np.float64)
        result.update(
            {
                "MEMRISTOR_R_MIN_OHM": float(np.min(resistance)),
                "MEMRISTOR_R_MAX_OHM": float(np.max(resistance)),
            }
        )
        if "memory_state" in columns:
            result["MEMORY_STATE_SWING"] = float(
                np.ptp(np.asarray(columns["memory_state"], dtype=np.float64))
            )
    elif measurement.kind == "memory_idvg":
        voltage = np.asarray(columns.get("gate_voltage", measurement.x), dtype=np.float64)
        current = np.abs(
            np.asarray(columns.get("drain_current", measurement.y), dtype=np.float64)
        )
        positive = current[current > 0]
        if not positive.size:
            raise ValueError("Memtransistor Id-Vg data must contain positive current.")
        area = abs(
            float(
                np.sum(
                    voltage * np.roll(current, -1)
                    - current * np.roll(voltage, -1)
                )
            )
        ) * 0.5
        result.update(
            {
                "MEMTRANSISTOR_ION_A": float(np.max(current)),
                "MEMTRANSISTOR_IOFF_A": float(np.min(positive)),
                "MEMTRANSISTOR_ON_OFF_RATIO": float(np.max(current) / np.min(positive)),
                "MEMTRANSISTOR_HYSTERESIS_AREA_A_V": area,
            }
        )
        if "memory_state" in columns:
            result["MEMORY_STATE_SWING"] = float(
                np.ptp(np.asarray(columns["memory_state"], dtype=np.float64))
            )
    else:
        raise ValueError(f"Unsupported measurement curve: {measurement.kind}.")

    return result


def _extract_current_sweep_metrics(
    gate_voltage, drain_current, device: DeviceParameters
) -> dict[str, float]:
    import numpy as np

    current = np.abs(drain_current)
    positive = current[current > 0]
    if not positive.size:
        raise ValueError("Id-Vg data must contain a non-zero current.")
    log_current = np.log10(np.maximum(current, np.max(positive) * 1e-15))
    voltage_step = np.diff(gate_voltage)
    gm = np.divide(
        np.diff(current),
        voltage_step,
        out=np.full_like(voltage_step, np.nan),
        where=voltage_step > 0,
    )
    log_slope = np.divide(
        voltage_step,
        np.diff(log_current),
        out=np.full_like(voltage_step, np.inf),
        where=np.diff(log_current) > 0,
    )
    subthreshold = (
        (current[:-1] <= np.max(positive) * 1e-2)
        & (current[1:] <= np.max(positive) * 1e-2)
        & np.isfinite(log_slope)
        & (log_slope > 0)
    )
    result = {
        "ION_A": float(np.max(current)),
        "IOFF_A": float(np.min(positive)),
        "ION_IOFF": float(np.max(current) / np.min(positive)),
        "GM_MAX_S": float(np.nanmax(gm)),
    }
    if np.any(subthreshold):
        result["SS_MV_DEC"] = float(np.min(log_slope[subthreshold]) * 1000)
    threshold_current = 1e-7 * device.effective_width_um / device.length_um
    if np.min(positive) <= threshold_current <= np.max(positive):
        result["VTH_CONSTANT_CURRENT_V"] = _interpolate_log_current(
            gate_voltage, np.maximum(current, np.max(positive) * 1e-15), threshold_current
        )
    return result


def _breakdown_knee_voltage(drain_voltage, drain_current) -> float | None:
    import numpy as np

    voltage = np.asarray(drain_voltage, dtype=np.float64)
    current = np.abs(np.asarray(drain_current, dtype=np.float64))
    positive = current > 0
    if np.count_nonzero(positive) < 3:
        return None
    voltage, current = voltage[positive], current[positive]
    order = np.argsort(voltage)
    voltage, current = voltage[order], current[order]
    if len(np.unique(voltage)) < 3:
        return None
    log_current = np.log10(current)
    slope = np.gradient(log_current, voltage)
    index = int(np.argmax(slope))
    if slope[index] <= 0:
        return None
    return float(voltage[index])


def _unity_gain_frequency(frequency, gain) -> float | None:
    import numpy as np

    frequency = np.asarray(frequency, dtype=np.float64)
    gain = np.abs(np.asarray(gain, dtype=np.float64))
    valid = (frequency > 0) & (gain > 0)
    frequency, gain = frequency[valid], gain[valid]
    order = np.argsort(frequency)
    frequency, gain = frequency[order], gain[order]
    if len(frequency) < 2:
        return None
    crossings = np.flatnonzero((gain[:-1] - 1) * (gain[1:] - 1) <= 0)
    if not crossings.size:
        return None
    index = int(crossings[0])
    if gain[index] == 1:
        return float(frequency[index])
    fraction = (0 - math.log10(gain[index])) / (
        math.log10(gain[index + 1]) - math.log10(gain[index])
    )
    return float(
        10
        ** (
            math.log10(frequency[index])
            + fraction * (math.log10(frequency[index + 1]) - math.log10(frequency[index]))
        )
    )


def _extract_transient_metrics(time, response) -> dict[str, float]:
    import numpy as np

    response = np.asarray(response, dtype=np.float64)
    time = np.asarray(time, dtype=np.float64)
    low, high = float(np.min(response)), float(np.max(response))
    swing = high - low
    if swing <= 0:
        raise ValueError("Transient response must vary over time.")
    rising_10 = _first_threshold_time(time, response, low + 0.1 * swing, rising=True)
    rising_90 = _first_threshold_time(time, response, low + 0.9 * swing, rising=True)
    falling_90 = _first_threshold_time(time, response, low + 0.9 * swing, rising=False)
    falling_10 = _first_threshold_time(time, response, low + 0.1 * swing, rising=False)
    result: dict[str, float] = {}
    if rising_10 is not None and rising_90 is not None and rising_90 >= rising_10:
        result["TON_10_90_S"] = rising_90 - rising_10
    if falling_90 is not None and falling_10 is not None and falling_10 >= falling_90:
        result["TOFF_90_10_S"] = falling_10 - falling_90
    if not result:
        raise ValueError("Could not identify rising or falling edges in transient data.")
    return result


def _first_threshold_time(time, response, threshold, *, rising: bool) -> float | None:
    import numpy as np

    if rising:
        crossings = np.flatnonzero(
            (response[:-1] < threshold) & (response[1:] >= threshold)
        )
    else:
        crossings = np.flatnonzero(
            (response[:-1] > threshold) & (response[1:] <= threshold)
        )
    if not crossings.size:
        return None
    index = int(crossings[0])
    fraction = (threshold - response[index]) / (
        response[index + 1] - response[index]
    )
    return float(time[index] + fraction * (time[index + 1] - time[index]))


def _interpolate_log_current(
    gate_voltage, drain_current, threshold_current: float
) -> float:
    import numpy as np

    if threshold_current < np.min(drain_current) or threshold_current > np.max(drain_current):
        raise ValueError(
            "The Id-Vg sweep does not cross the selected constant-current threshold."
        )
    log_current = np.log10(drain_current)
    log_target = math.log10(threshold_current)
    crossings = np.flatnonzero(
        ((log_current[:-1] <= log_target) & (log_current[1:] >= log_target))
        | ((log_current[:-1] >= log_target) & (log_current[1:] <= log_target))
    )
    for index in crossings:
        if log_current[index] == log_target:
            return float(gate_voltage[index])
        if log_current[index + 1] == log_target:
            return float(gate_voltage[index + 1])
        fraction = (log_target - log_current[index]) / (
            log_current[index + 1] - log_current[index]
        )
        return float(
            gate_voltage[index]
            + fraction * (gate_voltage[index + 1] - gate_voltage[index])
        )
    raise ValueError("Could not interpolate the constant-current threshold.")


def export_analysis_report(
    device: DeviceParameters, extracted: dict[str, float]
) -> str:
    """Serialize device inputs and extracted values without claiming a SPICE model."""
    import json

    device.validate()
    return json.dumps(
        {
            "device_type": device.device_type,
            "name": device.name,
            "material": {
                "epsilon_r": device.epsilon_r,
                "electron_mobility_cm2_vs": device.electron_mobility,
                "hole_mobility_cm2_vs": device.hole_mobility,
                "bandgap_ev": device.bandgap,
                "doping_cm3": device.doping,
                "temperature_k": device.temperature_k,
            },
            "geometry": {
                "width_um": device.width_um,
                "length_um": device.length_um,
                "oxide_thickness_nm": device.tox_nm,
                "effective_width_um": device.effective_width_um,
                **device.geometry,
            },
            "barrier_material": (
                device.barrier_material if device.device_type == "hemt" else None
            ),
            "junction_material": (
                device.junction_material if device.device_type == "tfet" else None
            ),
            "extracted": extracted,
            "note": "Analysis results only; not a calibrated compact model.",
        },
        indent=2,
    )


def export_spice_model(parameters: DeviceParameters, extracted: dict[str, float]) -> str:
    """Render the fitted values as a BSIM4 .model card."""
    parameters.validate()
    if parameters.device_type != "planar_mosfet":
        raise ValueError(
            "BSIM4 export is enabled only for planar bulk MOSFETs. "
            "Export the analysis report for other transistor structures."
        )
    supported = ("VTH0", "U0", "NFACTOR", "CGSO", "CGDO")
    values = {key: extracted[key] for key in supported if key in extracted}
    if not values:
        raise ValueError("No extracted parameters are available for SPICE export.")
    values["TOXE"] = parameters.tox_nm * 1e-9
    lines = [f".model {parameters.name} nmos level=54 version=4.8.1"]
    lines.extend(f"+ {key}={value:.8g}" for key, value in values.items())
    return "\n".join(lines) + "\n"


def _normalize_header(value: str) -> str:
    without_units = re.sub(r"\s*[\[(].*?[\])]", "", value)
    return "".join(character for character in without_units.strip().lower() if character.isalnum())


def _find_header(headers: dict[str, str], *names: str) -> str | None:
    for name in names:
        normalized = _normalize_header(name)
        if normalized in headers:
            return headers[normalized]
    return None
