"""Behavioral curve generation for memristors, memtransistors, and neuristors."""

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


def solve_neuristor_characteristics(
    device,
    *,
    supply_min_v: float = 0.5,
    supply_max_v: float = 6.0,
    supplies: int = 6,
    duration_ns: float = 2000.0,
    points: int = 4000,
) -> PDECharacteristicResult:
    """Simulate a Mott threshold-switch neuristor driving a capacitive line.

    Each segment is a volatile Mott switch (VO2 IMT) in series with a load
    resistor; the line capacitance is charged through the load and discharged
    through the switched (hot) device. The device is volatile by construction:
    it holds only while the terminal voltage exceeds the hold voltage and
    self-resets once the capacitor discharges, which is what makes the line
    excitable rather than a non-volatile memory element.
    """
    import numpy as np

    device.validate()
    if device.device_type != "neuristor":
        raise ValueError("Neuristor characteristics require a neuristor device.")
    if (
        not math.isfinite(supply_min_v)
        or not math.isfinite(supply_max_v)
        or supply_min_v <= 0
        or supply_min_v >= supply_max_v
    ):
        raise ValueError("Supply bounds must be positive, finite, and increasing.")
    if supplies < 2:
        raise ValueError("At least two supply voltages are required.")
    if not math.isfinite(duration_ns) or duration_ns <= 0:
        raise ValueError("Simulation duration must be positive and finite.")
    if points < 100:
        raise ValueError("Neuristor simulation requires at least 100 time points.")
    parameters = device.geometry
    cold = parameters["cold_resistance_ohm"]
    hot = parameters["hot_resistance_ohm"]
    threshold = parameters["threshold_voltage_v"]
    hold = parameters["hold_voltage_v"]
    load = parameters["load_resistance_ohm"]
    capacitance_f = parameters["line_capacitance_pf"] * 1e-12
    thermal_s = parameters["thermal_time_ns"] * 1e-9
    refractory_s = parameters["refractory_time_ns"] * 1e-9
    segments = int(parameters["line_segments"])
    # The cold device and the load form a divider, so the node settles at
    # Vdd*Rcold/(Rcold+Rload). Below this supply the node never reaches V_t and
    # the line stays subthreshold, emitting no spikes.
    critical_supply = threshold * (cold + load) / cold

    time = np.linspace(0.0, duration_ns * 1e-9, points)
    step = time[1] - time[0]
    supply_values = np.linspace(supply_min_v, supply_max_v, supplies)
    curves: list[GeneratedCurve] = []
    spike_rates: list[float] = []
    for supply in supply_values:
        voltage = np.empty_like(time)
        current = np.empty_like(time)
        resistance = np.empty_like(time)
        state = np.empty_like(time)
        voltage[0] = 0.0
        switched = False
        refractory_until = -math.inf
        spike_times: list[float] = []
        for index in range(time.size):
            node = float(voltage[index])
            if switched:
                switched = node >= hold
                if not switched:
                    refractory_until = float(time[index]) + refractory_s
            elif node >= threshold and float(time[index]) >= refractory_until:
                switched = True
                spike_times.append(float(time[index]))
            observed = hot if switched else cold
            resistance[index] = observed
            state[index] = 1.0 if switched else 0.0
            current[index] = node / observed
            if index + 1 < time.size:
                if switched:
                    # The switched filament shunts the line: the node collapses
                    # through the hot path instead of tracking the supply, which
                    # clips the spike at the threshold and releases the switch.
                    next_node = node * math.exp(-step / (hot * capacitance_f))
                else:
                    if float(time[index]) < refractory_until:
                        # Refractory: released heat keeps the filament inhibited,
                        # so the line cannot charge back towards the supply and
                        # the node stays below the threshold.
                        next_node = node * math.exp(-step / refractory_s)
                    else:
                        # Cold branch: the line charges through the load and the
                        # cold device towards the divider equilibrium
                        # Vdd*Rcold/(Rcold+Rload). Below the critical supply this
                        # equilibrium stays under V_t and no spike can start.
                        equilibrium = supply * cold / (cold + load)
                        next_node = equilibrium + (node - equilibrium) * math.exp(
                            -step / ((load + cold) * capacitance_f)
                        )
                voltage[index + 1] = max(next_node, 0.0)
        mean_interval = (
            float(np.mean(np.diff(spike_times))) if len(spike_times) > 1 else 0.0
        )
        spike_rates.append(len(spike_times) / (duration_ns * 1e-9))
        curves.append(
            GeneratedCurve(
                kind="neuristor_spikes",
                x=time.tolist(),
                y=voltage.tolist(),
                label=f"Нейристор V(t), Vdd={supply:g} В",
                columns={
                    "time": time.tolist(),
                    "voltage": voltage.tolist(),
                    "current": current.tolist(),
                    "resistance": resistance.tolist(),
                    "memory_state": state.tolist(),
                    "supply_voltage": [float(supply)] * time.size,
                    "spike_period": [mean_interval] * time.size,
                    "propagation_delay": [mean_interval / segments] * time.size,
                },
            )
        )

    firing = np.asarray(spike_rates, dtype=np.float64)
    active = firing[firing > 0]
    description = (
        "Пороговая модель нейристора на обратимом переходе металл-изолятор (Mott/VO₂): "
        "холодное и горячее состояние с гистерезисом V_t > V_h, отрицательное "
        "дифференциальное сопротивление, запорный элемент и ёмкость линии. "
        "Нейристор энергонезависимо не хранит состояние: после снятия "
        "напряжения он возвращается в холодное состояние, что и обеспечивает "
        "генерацию спайков и передачу возбуждения по линии."
    )
    return PDECharacteristicResult(
        curves=curves,
        backend="CPU (threshold-switch behavioral model)",
        diagnostics={
            "THRESHOLD_V": float(threshold),
            "HOLD_V": float(hold),
            "CRITICAL_SUPPLY_V": float(critical_supply),
            "R_COLD_OHM": float(cold),
            "R_HOT_OHM": float(hot),
            "ON_OFF_RATIO": float(cold / hot),
            "LINE_SEGMENTS": float(segments),
            "REFRACTORY_NS": float(parameters["refractory_time_ns"]),
            "SPIKE_RATE_HZ": float(np.mean(active)) if active.size else 0.0,
            "SPIKE_RATE_MAX_HZ": float(np.max(firing)) if firing.size else 0.0,
        },
        model_description=description,
    )
