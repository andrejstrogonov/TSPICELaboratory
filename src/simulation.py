"""Behavioral curve generation for memristors and memtransistors."""

from __future__ import annotations

import math
from dataclasses import dataclass

BOLTZMANN = 1.380649e-23
ELECTRON_CHARGE = 1.602176634e-19


@dataclass(frozen=True)
class GeneratedCurve:
    kind: str
    x: list[float]
    y: list[float]
    label: str
    columns: dict[str, list[float]]


@dataclass(frozen=True)
class PDECharacteristicResult:
    curves: list[GeneratedCurve]
    backend: str
    diagnostics: dict[str, float]
    model_description: str


def solve_memory_characteristics(
    device,
    *,
    voltage_min_v: float = -1.0,
    voltage_max_v: float = 1.0,
    drain_voltage_v: float = 0.1,
    points_per_sweep: int = 201,
) -> PDECharacteristicResult:
    """Generate illustrative state-model curves for memristors and memtransistors."""
    import numpy as np

    device.validate()
    if device.device_type not in ("memristor", "memtransistor"):
        raise ValueError("Memory characteristics require a memristor or memtransistor.")
    if (
        not math.isfinite(voltage_min_v)
        or not math.isfinite(voltage_max_v)
        or voltage_min_v >= voltage_max_v
    ):
        raise ValueError("Memory sweep bounds must be finite and increasing.")
    if not math.isfinite(drain_voltage_v) or drain_voltage_v <= 0:
        raise ValueError("Memtransistor drain voltage must be positive and finite.")
    if points_per_sweep < 11:
        raise ValueError("Memory sweep requires at least 11 points per direction.")

    parameters = device.geometry
    ron = parameters["low_resistance_ohm"]
    roff = parameters["high_resistance_ohm"]
    if ron >= roff:
        raise ValueError("Low resistance must be smaller than high resistance.")
    active_layer_m = parameters["active_layer_nm"] * 1e-9
    mobility_temperature_factor = math.exp(
        float(
            np.clip(
                -parameters["activation_energy_ev"]
                / 8.617333262145e-5
                * (1.0 / device.temperature_k - 1.0 / 300.0),
                -700,
                700,
            )
        )
    )
    sweep_up = np.linspace(voltage_min_v, voltage_max_v, points_per_sweep)
    sweep_down = np.linspace(voltage_max_v, voltage_min_v, points_per_sweep)[1:]
    sweep_voltage = np.concatenate((sweep_up, sweep_down))
    curves: list[GeneratedCurve] = []

    if device.device_type == "memristor":
        rates_v_s = (10.0, 100.0, 1_000.0)
        ion_mobility_m2_vs = parameters["ion_mobility_cm2_vs"] * 1e-4
        drift_scale = (
            ion_mobility_m2_vs
            * ron
            / active_layer_m**2
            * mobility_temperature_factor
        )
        for rate in rates_v_s:
            time = np.arange(sweep_voltage.size, dtype=np.float64) * (
                (voltage_max_v - voltage_min_v) / (points_per_sweep - 1) / rate
            )
            state = np.empty_like(sweep_voltage)
            resistance = np.empty_like(sweep_voltage)
            current = np.empty_like(sweep_voltage)
            state[0] = parameters["initial_state"]
            for index, voltage in enumerate(sweep_voltage):
                x = float(np.clip(state[index], 0.0, 1.0))
                resistance[index] = roff + (ron - roff) * x
                current[index] = voltage / resistance[index]
                if index + 1 < sweep_voltage.size:
                    dt = time[index + 1] - time[index]
                    average_current = (
                        current[index] + sweep_voltage[index + 1] / resistance[index]
                    ) * 0.5
                    if voltage >= parameters["set_voltage_v"]:
                        window = 1.0 - x**2
                        dxdt = drift_scale * average_current * window
                    elif voltage <= parameters["reset_voltage_v"]:
                        window = 1.0 - (1.0 - x) ** 2
                        dxdt = drift_scale * average_current * window
                    else:
                        dxdt = 0.0
                    state[index + 1] = np.clip(x + dxdt * dt, 0.0, 1.0)
            suffix = f"{rate:g} V/s"
            curves.append(
                GeneratedCurve(
                    kind="memristor_iv",
                    x=sweep_voltage.tolist(),
                    y=current.tolist(),
                    label=f"Мемристор I–V, {suffix}",
                    columns={
                        "voltage": sweep_voltage.tolist(),
                        "current": current.tolist(),
                        "time": time.tolist(),
                        "resistance": resistance.tolist(),
                        "memory_state": state.tolist(),
                        "sweep_rate": [rate] * sweep_voltage.size,
                    },
                )
            )
            curves.append(
                GeneratedCurve(
                    kind="memristor_resistance",
                    x=time.tolist(),
                    y=resistance.tolist(),
                    label=f"Мемристор R–время, {suffix}",
                    columns={
                        "time": time.tolist(),
                        "resistance": resistance.tolist(),
                        "voltage": sweep_voltage.tolist(),
                        "memory_state": state.tolist(),
                        "sweep_rate": [rate] * sweep_voltage.size,
                    },
                )
            )
        description = (
            "Иллюстративная модель дрейфа ионов с ограниченным состоянием x∈[0,1], "
            "линейной смесью R_on/R_off и оконной функцией. Не калиброванный "
            "материал-специфичный SPICE/TCAD-расчёт."
        )
    else:
        rates_v_s = (0.25, 1.0, 4.0)
        oxide_capacitance = device.oxide_capacitance_per_area
        width_m = device.effective_width_um * 1e-6
        length_m = device.length_um * 1e-6
        mobility_m2_vs = device.electron_mobility * 1e-4
        beta = mobility_m2_vs * oxide_capacitance * width_m / length_m
        threshold = 0.35
        retention_time_s = parameters["state_relaxation_ms"] * 1e-3
        memory_gain = 1.0 + 0.15 * math.log10(roff / ron)
        drift_scale = (
            parameters["ion_mobility_cm2_vs"]
            * 1e-4
            / active_layer_m**2
            * mobility_temperature_factor
        )
        for rate in rates_v_s:
            time = np.arange(sweep_voltage.size, dtype=np.float64) * (
                (voltage_max_v - voltage_min_v) / (points_per_sweep - 1) / rate
            )
            state = np.empty_like(sweep_voltage)
            current = np.empty_like(sweep_voltage)
            state[0] = parameters["initial_state"]
            for index, gate in enumerate(sweep_voltage):
                x = float(np.clip(state[index], 0.0, 1.0))
                shifted_threshold = threshold - 0.25 * (x - 0.5)
                overdrive = max(float(gate) - shifted_threshold, 0.0)
                gate_factor = 1.0 / (
                    1.0 + math.exp(float(np.clip(-overdrive / 0.1, -60, 60)))
                )
                memory_factor = 1.0 + (memory_gain - 1.0) * x
                current[index] = (
                    1e-12 + beta * overdrive**2 * drain_voltage_v * memory_factor
                )
                if index + 1 < sweep_voltage.size:
                    dt = time[index + 1] - time[index]
                    if gate >= parameters["set_voltage_v"]:
                        window = 1.0 - x**2
                        dxdt = (
                            drift_scale
                            * gate
                            * window
                            * (1.0 + parameters["gate_coupling"] * gate_factor)
                        )
                    elif gate <= parameters["reset_voltage_v"]:
                        window = 1.0 - (1.0 - x) ** 2
                        dxdt = (
                            drift_scale
                            * gate
                            * window
                            * (1.0 + parameters["gate_coupling"] * gate_factor)
                        )
                    else:
                        dxdt = -x / retention_time_s
                    state[index + 1] = np.clip(x + dxdt * dt, 0.0, 1.0)
            curves.append(
                GeneratedCurve(
                    kind="memory_idvg",
                    x=sweep_voltage.tolist(),
                    y=current.tolist(),
                    label=f"Мемтранзистор Id–Vg, Vd={drain_voltage_v:g} V, {rate:g} V/s",
                    columns={
                        "gate_voltage": sweep_voltage.tolist(),
                        "drain_current": current.tolist(),
                        "drain_voltage": [drain_voltage_v] * sweep_voltage.size,
                        "time": time.tolist(),
                        "memory_state": state.tolist(),
                        "sweep_rate": [rate] * sweep_voltage.size,
                    },
                )
            )
        description = (
            "Иллюстративная модель мемтранзистора с релаксацией состояния, "
            "зависящим от затвора сдвигом порога и поведенческой моделью канального тока. "
            "Не калиброванный материал-специфичный SPICE/TCAD-расчёт."
        )

    return PDECharacteristicResult(
        curves=curves,
        backend="CPU (state-based behavioral model)",
        diagnostics={
            "SWEEP_MIN_V": float(voltage_min_v),
            "SWEEP_MAX_V": float(voltage_max_v),
            "DRAIN_V": float(drain_voltage_v),
            "POINTS_PER_DIRECTION": float(points_per_sweep),
            "R_ON_OHM": float(ron),
            "R_OFF_OHM": float(roff),
        },
        model_description=description,
    )
