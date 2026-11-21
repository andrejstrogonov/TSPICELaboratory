//! A-4 · Алгоритм генерации характеристик памяти (state ODE)
//! Источник: src/simulation.py::solve_memory_characteristics
//! Поддерживаемые семейства: memristor, memtransistor (CPU-only behavioral model).

use std::collections::HashMap;
use std::f64::consts::PI;

/// Постоянная Больцмана [эВ/К]
const KB_EV_PER_K: f64 = 8.617_333_262_145e-5;
/// Диэлектрическая постоянная вакуума [Ф/м]
const EPS0_F_PER_M: f64 = 8.854_187_812_8e-12;
/// Опорная комнатная температура [К]
const T_REF_K: f64 = 300.0;

/// Ошибки валидации и расчёта state-моделей памяти
#[derive(Debug, Clone, PartialEq)]
pub enum MemoryStateError {
    UnsupportedDeviceType(String),
    InvalidResistanceRange { r_on: f64, r_off: f64 },
    InvalidSweepConfig(String),
    InvalidParameter(String),
}

impl std::fmt::Display for MemoryStateError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::UnsupportedDeviceType(dev) => {
                write!(f, "Семейство '{}' не поддерживается state-моделью памяти", dev)
            }
            Self::InvalidResistanceRange { r_on, r_off } => {
                write!(f, "ValueError: R_on ({}) >= R_off ({})", r_on, r_off)
            }
            Self::InvalidSweepConfig(msg) => write!(f, "Некорректная конфигурация развёртки: {}", msg),
            Self::InvalidParameter(msg) => write!(f, "Некорректный параметр прибора: {}", msg),
        }
    }
}

impl std::error::Error for MemoryStateError {}

/// Тип элемента памяти
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MemoryDeviceType {
    Memristor,
    Memtransistor,
}

impl MemoryDeviceType {
    pub fn from_str_id(value: &str) -> Result<Self, MemoryStateError> {
        match value {
            "memristor" => Ok(Self::Memristor),
            "memtransistor" => Ok(Self::Memtransistor),
            other => Err(MemoryStateError::UnsupportedDeviceType(other.to_string())),
        }
    }

    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Memristor => "memristor",
            Self::Memtransistor => "memtransistor",
        }
    }
}

/// Входные физические и геометрические параметры прибора
#[derive(Debug, Clone)]
pub struct MemoryDeviceParameters {
    pub device_type: MemoryDeviceType,
    /// Температура [К]
    pub temperature_k: f64,
    /// Толщина активного слоя D [нм]
    pub active_layer_nm: f64,
    /// Толщина подзатворного оксида Tox [нм] (для мемтранзистора)
    pub tox_nm: f64,
    /// Относительная диэлектрическая проницаемость оксида
    pub epsilon_r: f64,
    /// Подвижность электронов [см^2/(В*с)] (для мемтранзистора)
    pub electron_mobility_cm2_vs: f64,
    /// Ширина канала W [мкм]
    pub width_um: f64,
    /// Длина канала L [мкм]
    pub length_um: f64,
    /// Подвижность ионов / кислородных вакансий mu_v [см^2/(В*с)]
    pub ion_mobility_cm2_vs: f64,
    /// Сопротивление в открытом состоянии R_on (LRS) [Ом]
    pub low_resistance_ohm: f64,
    /// Сопротивление в закрытом состоянии R_off (HRS) [Ом]
    pub high_resistance_ohm: f64,
    /// Пороговое напряжение переключения SET [В] (> 0)
    pub set_voltage_v: f64,
    /// Пороговое напряжение переключения RESET [В] (< 0)
    pub reset_voltage_v: f64,
    /// Энергия активации ионного дрейфа Ea [эВ]
    pub activation_energy_ev: f64,
    /// Начальное внутреннее состояние x0 in [0, 1]
    pub initial_state: f64,
    /// Постоянная времени релаксации состояния tau_ret [мс] (для мемтранзистора)
    pub state_relaxation_ms: f64,
    /// Коэффициент связи затвора gate_coupling (для мемтранзистора)
    pub gate_coupling: f64,
}

impl Default for MemoryDeviceParameters {
    fn default() -> Self {
        Self {
            device_type: MemoryDeviceType::Memristor,
            temperature_k: 300.0,
            active_layer_nm: 10.0,
            tox_nm: 5.0,
            epsilon_r: 3.9,
            electron_mobility_cm2_vs: 200.0,
            width_um: 1.0,
            length_um: 1.0,
            ion_mobility_cm2_vs: 1e-10,
            low_resistance_ohm: 100.0,
            high_resistance_ohm: 1e5,
            set_voltage_v: 0.3,
            reset_voltage_v: -0.3,
            activation_energy_ev: 0.6,
            initial_state: 0.5,
            state_relaxation_ms: 100.0,
            gate_coupling: 2.0,
        }
    }
}

impl MemoryDeviceParameters {
    /// Проверка предусловий модели перед интегрированием
    pub fn validate(&self) -> Result<(), MemoryStateError> {
        if !(self.low_resistance_ohm > 0.0) || !(self.high_resistance_ohm > 0.0) {
            return Err(MemoryStateError::InvalidParameter(
                "R_on и R_off должны быть строго положительными".into(),
            ));
        }
        if self.low_resistance_ohm >= self.high_resistance_ohm {
            return Err(MemoryStateError::InvalidResistanceRange {
                r_on: self.low_resistance_ohm,
                r_off: self.high_resistance_ohm,
            });
        }
        if self.active_layer_nm <= 0.0 {
            return Err(MemoryStateError::InvalidParameter(
                "active_layer_nm должен быть > 0".into(),
            ));
        }
        if self.temperature_k <= 0.0 {
            return Err(MemoryStateError::InvalidParameter(
                "temperature_k должен быть > 0".into(),
            ));
        }
        if !(0.0..=1.0).contains(&self.initial_state) {
            return Err(MemoryStateError::InvalidParameter(
                "initial_state должен находиться в диапазоне [0, 1]".into(),
            ));
        }
        if self.set_voltage_v <= 0.0 {
            return Err(MemoryStateError::InvalidParameter(
                "set_voltage_v должен быть > 0".into(),
            ));
        }
        if self.reset_voltage_v >= 0.0 {
            return Err(MemoryStateError::InvalidParameter(
                "reset_voltage_v должен быть < 0".into(),
            ));
        }
        if self.device_type == MemoryDeviceType::Memtransistor {
            if self.tox_nm <= 0.0 || self.width_um <= 0.0 || self.length_um <= 0.0 {
                return Err(MemoryStateError::InvalidParameter(
                    "tox_nm, width_um и length_um должны быть > 0 для мемтранзистора".into(),
                ));
            }
            if self.state_relaxation_ms <= 0.0 {
                return Err(MemoryStateError::InvalidParameter(
                    "state_relaxation_ms должен быть > 0".into(),
                ));
            }
        }
        Ok(())
    }
}

/// Конфигурация треугольного цикла возбуждения V(t)
#[derive(Debug, Clone)]
pub struct MemorySweepConfig {
    pub v_min: f64,
    pub v_max: f64,
    pub points_per_sweep: usize,
    pub vd_bias: f64,
    pub sweep_rates_v_s: Option<Vec<f64>>,
}

impl Default for MemorySweepConfig {
    fn default() -> Self {
        Self {
            v_min: -1.0,
            v_max: 1.0,
            points_per_sweep: 201,
            vd_bias: 0.1,
            sweep_rates_v_s: None,
        }
    }
}

impl MemorySweepConfig {
    pub fn validate(&self) -> Result<(), MemoryStateError> {
        if self.points_per_sweep < 3 {
            return Err(MemoryStateError::InvalidSweepConfig(
                "points_per_sweep должен быть >= 3".into(),
            ));
        }
        if self.v_max <= self.v_min {
            return Err(MemoryStateError::InvalidSweepConfig(
                "v_max должен быть строго больше v_min".into(),
            ));
        }
        if let Some(rates) = &self.sweep_rates_v_s {
            if rates.is_empty() || rates.iter().any(|&r| r <= 0.0) {
                return Err(MemoryStateError::InvalidSweepConfig(
                    "Все скорости развёртки sweep_rates_v_s должны быть > 0".into(),
                ));
            }
        }
        Ok(())
    }
}

/// Сгенерированная кривая для графика GUI и хранения в БД
#[derive(Debug, Clone)]
pub struct GeneratedCurve {
    pub spec_id: String,
    pub label: String,
    pub x_field: String,
    pub y_field: String,
    pub sweep_rate_v_s: f64,
    pub x: Vec<f64>,
    pub y: Vec<f64>,
}

/// Метрики памяти по конкретной скорости развёртки
#[derive(Debug, Clone)]
pub struct MemoryRateMetrics {
    pub sweep_rate_v_s: f64,
    pub r_min_ohm: f64,
    pub r_max_ohm: f64,
    pub on_off_ratio: f64,
    pub hysteresis_area_a_v: f64,
    pub i_max_a: f64,
    pub memory_state_swing: f64,
    pub retention_time_s: Option<f64>,
}

/// Траектория эволюции состояния по точкам развёртки
#[derive(Debug, Clone)]
pub struct MemoryTrajectory {
    pub sweep_rate_v_s: f64,
    pub time_s: Vec<f64>,
    pub voltage_v: Vec<f64>,
    pub current_a: Vec<f64>,
    pub resistance_ohm: Vec<f64>,
    pub state_x: Vec<f64>,
}

/// Итоговый результат расчёта state-модели памяти
#[derive(Debug, Clone)]
pub struct MemoryCharacteristicResult {
    pub backend: String,
    pub curves: Vec<GeneratedCurve>,
    pub trajectories: Vec<MemoryTrajectory>,
    pub metrics_by_rate: Vec<MemoryRateMetrics>,
    pub summary_metrics: HashMap<String, f64>,
    pub model_description: String,
}

/// Формирование равномерной сетки linspace(start, stop, num)
fn linspace(start: f64, stop: f64, num: usize) -> Vec<f64> {
    if num == 0 {
        return Vec::new();
    }
    if num == 1 {
        return vec![start];
    }
    let step = (stop - start) / ((num - 1) as f64);
    (0..num).map(|i| start + step * (i as f64)).collect()
}

/// Формирование замкнутой петли возбуждения V(t): sweep_up + sweep_down[1..]
fn build_closed_voltage_loop(v_min: f64, v_max: f64, points_per_sweep: usize) -> Vec<f64> {
    let sweep_up = linspace(v_min, v_max, points_per_sweep);
    let sweep_down = linspace(v_max, v_min, points_per_sweep);

    let mut loop_v = Vec::with_capacity(2 * points_per_sweep - 1);
    loop_v.extend_from_slice(&sweep_up);
    if sweep_down.len() > 1 {
        loop_v.extend_from_slice(&sweep_down[1..]);
    }
    loop_v
}

/// Вычисление площади петли гистерезиса по формуле шнурования (трапеций) |∮ I dV|
fn compute_hysteresis_loop_area(voltage: &[f64], current: &[f64]) -> f64 {
    if voltage.len() < 2 || voltage.len() != current.len() {
        return 0.0;
    }
    let mut area = 0.0;
    for i in 0..(voltage.len() - 1) {
        let dv = voltage[i + 1] - voltage[i];
        let i_avg = 0.5 * (current[i] + current[i + 1]);
        area += i_avg * dv;
    }
    area.abs()
}

/// Основная функция расчёта характеристик памяти (алгоритм A-4)
pub fn solve_memory_characteristics(
    params: &MemoryDeviceParameters,
    config: &MemorySweepConfig,
) -> Result<MemoryCharacteristicResult, MemoryStateError> {
    params.validate()?;
    config.validate()?;

    // Температурный фактор Аррениуса: mu_T = exp(-Ea / kB * (1/T - 1/300))
    let mu_t = (-params.activation_energy_ev / KB_EV_PER_K
        * (1.0 / params.temperature_k - 1.0 / T_REF_K))
        .exp();

    // Толщина активного слоя в метрах и подвижность в м^2/(В*с)
    let active_layer_m = params.active_layer_nm * 1e-9;
    let mu_v_m2_vs = params.ion_mobility_cm2_vs * 1e-4;

    // Замкнутый профиль возбуждения V(t)
    let voltage_loop = build_closed_voltage_loop(
        config.v_min,
        config.v_max,
        config.points_per_sweep,
    );
    let dv_step = (config.v_max - config.v_min) / ((config.points_per_sweep - 1) as f64);

    let default_rates = match params.device_type {
        MemoryDeviceType::Memristor => vec![10.0, 100.0, 1000.0],
        MemoryDeviceType::Memtransistor => vec![0.25, 1.0, 4.0],
    };
    let rates = config.sweep_rates_v_s.clone().unwrap_or(default_rates);

    let mut curves = Vec::new();
    let mut trajectories = Vec::new();
    let mut metrics_by_rate = Vec::new();

    match params.device_type {
        MemoryDeviceType::Memristor => {
            // drift_scale = (mu_v * R_on / D^2) * mu_T
            let drift_scale =
                (mu_v_m2_vs * params.low_resistance_ohm / (active_layer_m * active_layer_m)) * mu_t;

            for &rate in &rates {
                let dt = dv_step / rate;
                let n_pts = voltage_loop.len();

                let mut time_s = Vec::with_capacity(n_pts);
                let mut current_a = Vec::with_capacity(n_pts);
                let mut resistance_ohm = Vec::with_capacity(n_pts);
                let mut state_x = Vec::with_capacity(n_pts);

                let mut x = params.initial_state.clamp(0.0, 1.0);
                let mut t = 0.0;

                for idx in 0..n_pts {
                    let v = voltage_loop[idx];
                    let r = params.high_resistance_ohm
                        + (params.low_resistance_ohm - params.high_resistance_ohm) * x;
                    let i_curr = v / r;

                    time_s.push(t);
                    current_a.push(i_curr);
                    resistance_ohm.push(r);
                    state_x.push(x);

                    if idx + 1 < n_pts {
                        let v_next = voltage_loop[idx + 1];
                        let i_next_est = v_next / r;
                        let i_avg = 0.5 * (i_curr + i_next_est);

                        // Оконная функция Biolek и пороговое переключение
                        let dx_dt = if v >= params.set_voltage_v {
                            let window = 1.0 - x * x;
                            drift_scale * i_avg * window
                        } else if v <= params.reset_voltage_v {
                            let one_minus_x = 1.0 - x;
                            let window = 1.0 - one_minus_x * one_minus_x;
                            drift_scale * i_avg * window
                        } else {
                            0.0
                        };

                        x = (x + dx_dt * dt).clamp(0.0, 1.0);
                        t += dt;
                    }
                }

                let r_min = resistance_ohm.iter().copied().fold(f64::INFINITY, f64::min);
                let r_max = resistance_ohm.iter().copied().fold(f64::NEG_INFINITY, f64::max);
                let i_max = current_a.iter().map(|v| v.abs()).fold(0.0, f64::max);
                let x_min = state_x.iter().copied().fold(f64::INFINITY, f64::min);
                let x_max = state_x.iter().copied().fold(f64::NEG_INFINITY, f64::max);
                let area = compute_hysteresis_loop_area(&voltage_loop, &current_a);

                metrics_by_rate.push(MemoryRateMetrics {
                    sweep_rate_v_s: rate,
                    r_min_ohm: r_min,
                    r_max_ohm: r_max,
                    on_off_ratio: if r_min > 0.0 { r_max / r_min } else { 0.0 },
                    hysteresis_area_a_v: area,
                    i_max_a: i_max,
                    memory_state_swing: x_max - x_min,
                    retention_time_s: None,
                });

                curves.push(GeneratedCurve {
                    spec_id: "memristor_iv".to_string(),
                    label: format!("I–V гистерезис ({} В/с)", rate),
                    x_field: "voltage_v".to_string(),
                    y_field: "current_a".to_string(),
                    sweep_rate_v_s: rate,
                    x: voltage_loop.clone(),
                    y: current_a.clone(),
                });

                curves.push(GeneratedCurve {
                    spec_id: "memristor_resistance".to_string(),
                    label: format!("R(t) эволюция ({} В/с)", rate),
                    x_field: "time_s".to_string(),
                    y_field: "resistance_ohm".to_string(),
                    sweep_rate_v_s: rate,
                    x: time_s.clone(),
                    y: resistance_ohm.clone(),
                });

                trajectories.push(MemoryTrajectory {
                    sweep_rate_v_s: rate,
                    time_s,
                    voltage_v: voltage_loop.clone(),
                    current_a,
                    resistance_ohm,
                    state_x,
                });
            }
        }
        MemoryDeviceType::Memtransistor => {
            // Параметры канала транзисторной части
            let tox_m = params.tox_nm * 1e-9;
            let c_ox = EPS0_F_PER_M * params.epsilon_r / tox_m;
            let mu_n_m2_vs = params.electron_mobility_cm2_vs * 1e-4;
            let beta = mu_n_m2_vs * c_ox * (params.width_um / params.length_um);
            let memory_gain = 1.0 + 0.15 * (params.high_resistance_ohm / params.low_resistance_ohm).log10();
            let tau_ret_s = params.state_relaxation_ms * 1e-3;
            let drift_scale = (mu_v_m2_vs / (active_layer_m * active_layer_m)) * mu_t;

            for &rate in &rates {
                let dt = dv_step / rate;
                let n_pts = voltage_loop.len();

                let mut time_s = Vec::with_capacity(n_pts);
                let mut current_a = Vec::with_capacity(n_pts);
                let mut resistance_ohm = Vec::with_capacity(n_pts);
                let mut state_x = Vec::with_capacity(n_pts);

                let mut x = params.initial_state.clamp(0.0, 1.0);
                let mut t = 0.0;

                for idx in 0..n_pts {
                    let vg = voltage_loop[idx];
                    // Сдвиг порогового напряжения состоянием V_T(x)
                    let v_th_x = 0.35 - 0.25 * (x - 0.5);
                    let v_ov = (vg - v_th_x).max(0.0);

                    let id = 1e-12
                        + beta * v_ov * v_ov * config.vd_bias * (1.0 + (memory_gain - 1.0) * x);
                    let r_channel = if id > 0.0 {
                        config.vd_bias.abs() / id
                    } else {
                        params.high_resistance_ohm
                    };

                    time_s.push(t);
                    current_a.push(id);
                    resistance_ohm.push(r_channel);
                    state_x.push(x);

                    if idx + 1 < n_pts {
                        // Модуляция скорости дрейфа полем затвора
                        let f_gate = (0.5 * PI * vg.abs()).tanh();
                        let gate_factor = 1.0 + params.gate_coupling * f_gate;

                        let dx_dt = if vg >= params.set_voltage_v {
                            let window = 1.0 - x * x;
                            drift_scale * vg * window * gate_factor
                        } else if vg <= params.reset_voltage_v {
                            let one_minus_x = 1.0 - x;
                            let window = 1.0 - one_minus_x * one_minus_x;
                            drift_scale * vg * window * gate_factor
                        } else {
                            // Релаксация состояния в режиме удержания
                            -x / tau_ret_s
                        };

                        x = (x + dx_dt * dt).clamp(0.0, 1.0);
                        t += dt;
                    }
                }

                let r_min = resistance_ohm.iter().copied().fold(f64::INFINITY, f64::min);
                let r_max = resistance_ohm.iter().copied().fold(f64::NEG_INFINITY, f64::max);
                let i_min = current_a.iter().copied().fold(f64::INFINITY, f64::min);
                let i_max = current_a.iter().copied().fold(f64::NEG_INFINITY, f64::max);
                let x_min = state_x.iter().copied().fold(f64::INFINITY, f64::min);
                let x_max = state_x.iter().copied().fold(f64::NEG_INFINITY, f64::max);
                let area = compute_hysteresis_loop_area(&voltage_loop, &current_a);

                metrics_by_rate.push(MemoryRateMetrics {
                    sweep_rate_v_s: rate,
                    r_min_ohm: r_min,
                    r_max_ohm: r_max,
                    on_off_ratio: if i_min > 0.0 { i_max / i_min } else { 0.0 },
                    hysteresis_area_a_v: area,
                    i_max_a: i_max,
                    memory_state_swing: x_max - x_min,
                    retention_time_s: Some(tau_ret_s),
                });

                curves.push(GeneratedCurve {
                    spec_id: "memory_idvg".to_string(),
                    label: format!("Id–Vg память ({} В/с)", rate),
                    x_field: "gate_voltage_v".to_string(),
                    y_field: "drain_current_a".to_string(),
                    sweep_rate_v_s: rate,
                    x: voltage_loop.clone(),
                    y: current_a.clone(),
                });

                curves.push(GeneratedCurve {
                    spec_id: "memristor_resistance".to_string(),
                    label: format!("R_ch(t) ({} В/с)", rate),
                    x_field: "time_s".to_string(),
                    y_field: "resistance_ohm".to_string(),
                    sweep_rate_v_s: rate,
                    x: time_s.clone(),
                    y: resistance_ohm.clone(),
                });

                trajectories.push(MemoryTrajectory {
                    sweep_rate_v_s: rate,
                    time_s,
                    voltage_v: voltage_loop.clone(),
                    current_a,
                    resistance_ohm,
                    state_x,
                });
            }
        }
    }

    // Агрегированные метрики по номинальной (первой) скорости развёртки
    let mut summary_metrics = HashMap::new();
    if let Some(first) = metrics_by_rate.first() {
        match params.device_type {
            MemoryDeviceType::Memristor => {
                summary_metrics.insert("MEMRISTOR_R_ON_OHM".into(), first.r_min_ohm);
                summary_metrics.insert("MEMRISTOR_R_OFF_OHM".into(), first.r_max_ohm);
                summary_metrics.insert("MEMRISTOR_ON_OFF_RATIO".into(), first.on_off_ratio);
                summary_metrics.insert("MEMRISTOR_HYSTERESIS_AREA_A_V".into(), first.hysteresis_area_a_v);
                summary_metrics.insert("MEMRISTOR_I_MAX_A".into(), first.i_max_a);
                summary_metrics.insert("MEMORY_STATE_SWING".into(), first.memory_state_swing);
            }
            MemoryDeviceType::Memtransistor => {
                let i_min = trajectories
                    .first()
                    .and_then(|tr| tr.current_a.iter().copied().reduce(f64::min))
                    .unwrap_or(0.0);
                summary_metrics.insert("MEMTRANSISTOR_ION_A".into(), first.i_max_a);
                summary_metrics.insert("MEMTRANSISTOR_IOFF_A".into(), i_min);
                summary_metrics.insert("MEMTRANSISTOR_ON_OFF_RATIO".into(), first.on_off_ratio);
                summary_metrics.insert("MEMTRANSISTOR_HYSTERESIS_AREA_A_V".into(), first.hysteresis_area_a_v);
                summary_metrics.insert("MEMORY_STATE_SWING".into(), first.memory_state_swing);
                if let Some(tau) = first.retention_time_s {
                    summary_metrics.insert("RETENTION_TIME_S".into(), tau);
                }
            }
        }
    }

    let model_description = format!(
        "Behavioral CPU state-based ODE model ({}): windowed Biolek ion drift с температурным фактором Аррениуса.          Внимание: кривые являются демонстрационной поведенческой аппроксимацией, а не калиброванной материало-специфичной или TCAD-моделью.",
        params.device_type.as_str()
    );

    Ok(MemoryCharacteristicResult {
        backend: "CPU (state-based behavioral model)".to_string(),
        curves,
        trajectories,
        metrics_by_rate,
        summary_metrics,
        model_description,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_memristor_hysteresis_loop() {
        let params = MemoryDeviceParameters::default();
        let config = MemorySweepConfig::default();
        let res = solve_memory_characteristics(&params, &config).expect("Расчёт должен завершиться");

        assert_eq!(res.curves.len(), 6); // 3 скорости * (IV + R(t))
        assert!(res.summary_metrics["MEMRISTOR_HYSTERESIS_AREA_A_V"] > 0.0);
        assert!(res.summary_metrics["MEMORY_STATE_SWING"] > 0.0);
    }

    #[test]
    fn test_memtransistor_relaxation_and_shift() {
        let mut params = MemoryDeviceParameters::default();
        params.device_type = MemoryDeviceType::Memtransistor;
        let config = MemorySweepConfig::default();
        let res = solve_memory_characteristics(&params, &config).expect("Расчёт должен завершиться");

        assert_eq!(res.curves.len(), 6);
        assert!(res.summary_metrics["MEMTRANSISTOR_ON_OFF_RATIO"] > 1.0);
        assert_eq!(res.summary_metrics["RETENTION_TIME_S"], 0.1);
    }

    #[test]
    fn test_invalid_resistance_order_rejected() {
        let mut params = MemoryDeviceParameters::default();
        params.low_resistance_ohm = 1e5;
        params.high_resistance_ohm = 100.0;
        let err = solve_memory_characteristics(&params, &MemorySweepConfig::default()).unwrap_err();
        assert!(matches!(err, MemoryStateError::InvalidResistanceRange { .. }));
    }
}
