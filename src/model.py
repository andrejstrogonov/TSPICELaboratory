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
}


@dataclass(frozen=True)
class Measurement:
    kind: str
    x: list[float]
    y: list[float]
    drain_voltage: list[float] | None = None
    label: str = ""
    gate_bias: float | None = None


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
        nonnegative = {"access_resistance_ohm_mm", "gate_recess_nm"}
        positive.update(
            (key, value)
            for key, value in self.geometry.items()
            if key not in {"back_gate_voltage", *nonnegative}
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
        if not math.isfinite(self.geometry.get("back_gate_voltage", 0.0)):
            raise ValueError("Back-gate voltage must be a finite number.")
        for label, value in positive.items():
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{label.replace('_', ' ').capitalize()} must be a positive finite number.")
        if not self.name.strip() or any(ch.isspace() for ch in self.name):
            raise ValueError("Model name must be non-empty and contain no spaces.")
        if self.device_type == "hemt" and not self.barrier_material.strip():
            raise ValueError("HEMT barrier material must not be empty.")
        if self.device_type == "tfet" and not self.junction_material.strip():
            raise ValueError("TFET junction material must not be empty.")

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
    """Load Id-Vg, Id-Vd, or C-V data from a CSV with a header row."""
    file_path = Path(path)
    with file_path.open("r", encoding="utf-8-sig", newline="") as stream:
        sample = stream.read(4096)
        stream.seek(0)
        try:
            dialect = csv.Sniffer().sniff(sample, delimiters=",;\t")
        except csv.Error:
            dialect = csv.excel
        rows = csv.DictReader(stream, dialect=dialect)
        if not rows.fieldnames:
            raise ValueError("CSV must have a header row.")
        headers = {_normalize_header(header): header for header in rows.fieldnames}
        gate = _find_header(headers, "vg", "vgs", "gate_voltage")
        drain = _find_header(headers, "vd", "vds", "drain_voltage")
        current = _find_header(headers, "id", "ids", "drain_current")
        capacitance = _find_header(headers, "cg", "cgg", "capacitance")
        records = list(rows)
        x_column = gate or drain
        gate_bias = None
        if current and gate and drain:
            gate_values = []
            drain_values = []
            for row in records:
                try:
                    gate_value = float(row[gate])
                    drain_value = float(row[drain])
                except (TypeError, ValueError, KeyError) as error:
                    raise ValueError("Invalid gate or drain voltage in CSV.") from error
                if not math.isfinite(gate_value) or not math.isfinite(drain_value):
                    raise ValueError("Non-finite gate or drain voltage in CSV.")
                gate_values.append(gate_value)
                drain_values.append(drain_value)
            if len(set(gate_values)) == 1 and len(set(drain_values)) > 1:
                kind = "idvd"
                x_header = drain
                gate_bias = gate_values[0]
                drain_header = None
            else:
                kind = "idvg"
                x_header = gate
                drain_header = drain
            y_header = current
        elif current and x_column:
            kind = "idvg" if gate else "idvd"
            x_header = x_column
            y_header = current
            drain_header = None
        elif gate and capacitance:
            kind = "cv"
            x_header = gate
            y_header = capacitance
            drain_header = None
        else:
            raise ValueError(
                "Unsupported CSV columns. Use Vg, Id (and optional Vd), "
                "Vd, Id, or Vg, Cg."
            )

        x_values: list[float] = []
        y_values: list[float] = []
        drain_values: list[float] = []
        for line_number, row in enumerate(records, start=2):
            try:
                x_value = float(row[x_header])
                y_value = float(row[y_header])
                vd_value = float(row[drain_header]) if drain_header else None
            except (TypeError, ValueError, KeyError) as error:
                raise ValueError(f"Invalid numeric value on CSV line {line_number}.") from error
            if not math.isfinite(x_value) or not math.isfinite(y_value):
                raise ValueError(f"Non-finite measurement on CSV line {line_number}.")
            x_values.append(x_value)
            y_values.append(y_value)
            if vd_value is not None:
                if not math.isfinite(vd_value):
                    raise ValueError(f"Non-finite drain voltage on CSV line {line_number}.")
                drain_values.append(vd_value)

    if len(x_values) < 3:
        raise ValueError("At least three measurement points are required.")
    if len(set(x_values)) < 3:
        raise ValueError("At least three distinct voltage points are required.")
    if kind != "cv" and any(value < 0 for value in y_values):
        raise ValueError("Drain current values must be non-negative.")
    return Measurement(
        kind=kind,
        x=x_values,
        y=y_values,
        drain_voltage=drain_values if drain_header else None,
        label=file_path.name,
        gate_bias=gate_bias,
    )


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
