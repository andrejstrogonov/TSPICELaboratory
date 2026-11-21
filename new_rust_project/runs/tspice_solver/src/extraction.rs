//! Алгоритм A-5 · Пошаговый алгоритм экстракции BSIM4 (FR-EXT-01..04) и curve metrics.
//! Источник: ALGO_EXTRACTION (src/extractors.py, src/model.py).

use std::collections::BTreeMap;
use std::fmt;

const EPSILON_0: f64 = 8.854_187_812_8e-12; // Ф/м
const K_BOLTZMANN: f64 = 1.380_649e-23;    // Дж/К
const Q_ELECTRON: f64 = 1.602_176_634e-19; // Кл

/// Семейство полупроводникового прибора.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DeviceType {
    PlanarMosfet,
    FdSoi,
    Finfet,
    GaaNanosheet,
    GaaNanowire,
    Tfet,
    Hemt,
    CadCustom,
}

/// Целевой SPICE-симулятор для подстановки LEVEL и синтаксиса карты.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TargetSimulator {
    Ngspice,    // LEVEL = 54 для BSIM4
    MicroCap12, // LEVEL = 14 для BSIM4
    TannerTSpice,
    Xyce,
}

/// Шаг 0: Стратегия экстрактора по семейству прибора.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ExtractorStrategy {
    PlanarBsim4,         // LEVEL 54 / 14
    FdSoiBsim4Box,       // BSIM4 + геом. поправка BOX
    MultigateCmgLevel72, // FinFET / GAA -> LEVEL 72
    BehavioralSubckt,    // TFET / HEMT / CAD -> полиномиальная .subckt
}

/// Физические и геометрические параметры транзистора.
#[derive(Debug, Clone)]
pub struct DeviceParameters {
    pub name: String,
    pub device_type: DeviceType,
    pub width_um: f64,
    pub length_um: f64,
    pub tox_nm: f64,
    pub epsilon_r: f64,
    pub electron_mobility_cm2_vs: f64,
    pub doping_cm3: f64,
    pub temperature_k: f64,
    /// Эффективная ширина (для FinFET/GAA учитывает периметр рёбер/листов).
    pub w_eff_um: Option<f64>,
    /// Толщина скрытого оксида BOX (нм) для FD-SOI.
    pub buried_oxide_nm: Option<f64>,
}

impl DeviceParameters {
    pub fn effective_width_um(&self) -> f64 {
        self.w_eff_um.unwrap_or(self.width_um)
    }

    pub fn cox_f_m2(&self) -> f64 {
        let tox_m = self.tox_nm * 1e-9;
        EPSILON_0 * self.epsilon_r / tox_m
    }

    pub fn thermal_voltage_v(&self) -> f64 {
        K_BOLTZMANN * self.temperature_k / Q_ELECTRON
    }
}

/// Передаточная или выходная ВАХ (V -> I).
#[derive(Debug, Clone)]
pub struct IvSweep {
    pub v: Vec<f64>,
    pub i: Vec<f64>,
    /// Фиксированное смещение второго электрода (Vd для Id-Vg, Vg для Id-Vd).
    pub bias_v: f64,
    /// Длина канала экземпляра для группового фиттинга (FR-EXT-02).
    pub instance_length_um: Option<f64>,
    /// Ширина канала экземпляра для группового фиттинга (FR-EXT-02).
    pub instance_width_um: Option<f64>,
}

/// Вольт-фарадная характеристика (Vg -> Cgg).
#[derive(Debug, Clone)]
pub struct CvSweep {
    pub vg: Vec<f64>,
    pub cgg_f: Vec<f64>,
}

/// ВЧ-свип (f -> |h21|, Mason U).
#[derive(Debug, Clone)]
pub struct AcSweep {
    pub freq_hz: Vec<f64>,
    pub h21_mag: Vec<f64>,
    pub mason_u: Vec<f64>,
}

/// Переходная характеристика (t -> Vout, Ig).
#[derive(Debug, Clone)]
pub struct TransientSweep {
    pub time_s: Vec<f64>,
    pub v_out: Vec<f64>,
    pub ig_a: Vec<f64>,
}

/// Входной набор графиков для экстракции.
#[derive(Debug, Clone, Default)]
pub struct ExtractionInput {
    pub id_vg_graphs: Vec<IvSweep>,
    pub id_vd_graphs: Vec<IvSweep>,
    pub cv_graphs: Vec<CvSweep>,
    pub breakdown_graph: Option<IvSweep>,
    pub ac_graph: Option<AcSweep>,
    pub transient_graph: Option<TransientSweep>,
    /// Принудительный режим вывода .model с номером LEVEL (даже для не-planar).
    pub force_level_mode: bool,
}

/// Стартовые приближения (FR-EXT-01).
#[derive(Debug, Clone)]
pub struct StartGuesses {
    pub vth0_start: f64,
    pub u0_start_m2_vs: f64,
    pub gm_max_s: f64,
    pub vg_at_gm_max: f64,
}

/// Аудит извлечённого параметра с фиксацией выхода на границу (FR-EXT-04).
#[derive(Debug, Clone)]
pub struct ExtractedParameterRecord {
    pub name: String,
    pub value: f64,
    pub step_no: u8,
    pub lower_bound: f64,
    pub upper_bound: f64,
    pub bound_hit: bool,
}

/// Непараметрические метрики кривых (Шаг 5).
#[derive(Debug, Clone, Default)]
pub struct CurveMetrics {
    pub ss_mv_dec: Option<f64>,
    pub ion_a: Option<f64>,
    pub ioff_a: Option<f64>,
    pub ion_ioff_ratio: Option<f64>,
    pub vth_constant_current_v: Option<f64>,
    pub gm_max_s: Option<f64>,
    pub ron_ohm: Option<f64>,
    pub gds_s: Option<f64>,
    pub vbr_knee_v: Option<f64>,
    pub cgg_mean_f: Option<f64>,
    pub cgso_f_m: Option<f64>,
    pub cgdo_f_m: Option<f64>,
    pub ft_hz: Option<f64>,
    pub fmax_hz: Option<f64>,
    pub ton_10_90_s: Option<f64>,
    pub toff_90_10_s: Option<f64>,
    pub qg_c: Option<f64>,
}

/// Сформированный гладкий ёмкостной профиль (Шаг 6, FR-EXT-03).
#[derive(Debug, Clone)]
pub struct SmoothCapacitanceModel {
    pub expression_family: String, // "C1/C2-continuous tanh/EKV"
    pub c_min_f: f64,
    pub c_max_f: f64,
    pub v_flatband_v: f64,
    pub transition_slope_v: f64,
    pub poly2_eliminated: bool,
}

/// Итоговый артефакт экстракции (Шаг 7 -> spice_models).
#[derive(Debug, Clone)]
pub struct SpiceModelArtifact {
    pub strategy: ExtractorStrategy,
    pub model_type: String,
    pub level: Option<u16>,
    pub target_simulator: TargetSimulator,
    pub model_text: String,
    pub fitness_score: f64,
    pub rmse_log10_subthreshold: f64,
    pub mape_above_threshold_pct: f64,
    pub parameters: Vec<ExtractedParameterRecord>,
    pub curve_metrics: CurveMetrics,
    pub smooth_capacitance: Option<SmoothCapacitanceModel>,
    pub compatibility_disclaimer: Option<String>,
}

/// Ошибки алгоритма экстракции.
#[derive(Debug, Clone)]
pub enum ExtractionError {
    InvalidGeometry(String),
    MissingTransferCurve(String),
    InsufficientPoints(String),
    NonPositiveSlope(String),
}

impl fmt::Display for ExtractionError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidGeometry(m) => write!(f, "Некорректная геометрия прибора: {}", m),
            Self::MissingTransferCurve(m) => write!(f, "Отсутствует передаточная ВАХ Id-Vg: {}", m),
            Self::InsufficientPoints(m) => write!(f, "Недостаточно точек кривой: {}", m),
            Self::NonPositiveSlope(m) => write!(f, "Не удалось определить положительную крутизну gm: {}", m),
        }
    }
}

impl std::error::Error for ExtractionError {}

// ============================================================================
// Шаг 0 · Маршрутизация по семейству
// ============================================================================

pub fn route_extractor(device_type: DeviceType) -> ExtractorStrategy {
    match device_type {
        DeviceType::PlanarMosfet => ExtractorStrategy::PlanarBsim4,
        DeviceType::FdSoi => ExtractorStrategy::FdSoiBsim4Box,
        DeviceType::Finfet | DeviceType::GaaNanosheet | DeviceType::GaaNanowire => {
            ExtractorStrategy::MultigateCmgLevel72
        }
        DeviceType::Tfet | DeviceType::Hemt | DeviceType::CadCustom => {
            ExtractorStrategy::BehavioralSubckt
        }
    }
}

// ============================================================================
// Численное дифференцирование (центральные разности на неравномерной сетке)
// ============================================================================

pub fn numerical_derivative(x: &[f64], y: &[f64]) -> Vec<f64> {
    let n = x.len().min(y.len());
    if n == 0 {
        return Vec::new();
    }
    if n == 1 {
        return vec![0.0];
    }
    let mut dy_dx = vec![0.0; n];
    for i in 0..n {
        if i == 0 {
            let dx = x[1] - x[0];
            dy_dx[0] = if dx.abs() > 1e-18 { (y[1] - y[0]) / dx } else { 0.0 };
        } else if i == n - 1 {
            let dx = x[n - 1] - x[n - 2];
            dy_dx[n - 1] = if dx.abs() > 1e-18 {
                (y[n - 1] - y[n - 2]) / dx
            } else {
                0.0
            };
        } else {
            let h1 = x[i] - x[i - 1];
            let h2 = x[i + 1] - x[i];
            if h1.abs() > 1e-18 && h2.abs() > 1e-18 {
                // Взвешенная центральная разность 2-го порядка для неравномерной сетки
                let w_prev = -h2 / (h1 * (h1 + h2));
                let w_curr = (h2 - h1) / (h1 * h2);
                let w_next = h1 / (h2 * (h1 + h2));
                dy_dx[i] = w_prev * y[i - 1] + w_curr * y[i] + w_next * y[i + 1];
            } else {
                let dx = x[i + 1] - x[i - 1];
                dy_dx[i] = if dx.abs() > 1e-18 {
                    (y[i + 1] - y[i - 1]) / dx
                } else {
                    0.0
                };
            }
        }
    }
    dy_dx
}

// ============================================================================
// Шаг 1 · Стартовые приближения (FR-EXT-01)
// ============================================================================

pub fn compute_start_guesses(
    device: &DeviceParameters,
    id_vg: &IvSweep,
) -> Result<StartGuesses, ExtractionError> {
    if id_vg.v.len() < 3 || id_vg.v.len() != id_vg.i.len() {
        return Err(ExtractionError::InsufficientPoints(
            "Для расчёта стартовых приближений требуется минимум 3 точки Id-Vg".into(),
        ));
    }

    let gm = numerical_derivative(&id_vg.v, &id_vg.i);
    let mut best_idx = 0usize;
    let mut gm_max = f64::NEG_INFINITY;

    for (idx, &g) in gm.iter().enumerate() {
        if g.is_finite() && g > gm_max {
            gm_max = g;
            best_idx = idx;
        }
    }

    if !gm_max.is_finite() || gm_max <= 0.0 {
        return Err(ExtractionError::NonPositiveSlope(
            "Максимум крутизны gm <= 0 на передаточной ВАХ".into(),
        ));
    }

    let vg_peak = id_vg.v[best_idx];
    let id_peak = id_vg.i[best_idx].abs();
    let vds = id_vg.bias_v.abs().max(1e-3);

    // Линейная экстраполяция из точки максимума крутизны: VTH0_start = Vg - Id / gm - Vds/2
    let vth0_start = vg_peak - (id_peak / gm_max) - 0.5 * vds;

    let w_m = device.effective_width_um() * 1e-6;
    let l_m = device.length_um * 1e-6;
    let cox = device.cox_f_m2();

    // U0 = gm * L / (W * Cox * Vds) [м²/(В·с)]
    let u0_raw = (gm_max * l_m) / (w_m * cox * vds);
    let u0_start_m2_vs = u0_raw.clamp(1e-5, 1.0); // 0.1 .. 10000 см²/(В·с)

    Ok(StartGuesses {
        vth0_start,
        u0_start_m2_vs,
        gm_max_s: gm_max,
        vg_at_gm_max: vg_peak,
    })
}

// ============================================================================
// Шаг 2 · Ошибка по режимам (NFR-REL-01)
// ============================================================================

#[derive(Debug, Clone, Default)]
pub struct RegimeResiduals {
    pub subthreshold_log10_errors: Vec<f64>,
    pub above_threshold_pct_errors: Vec<f64>,
    pub gm_rel_errors: Vec<f64>,
    pub gds_rel_errors: Vec<f64>,
}

impl RegimeResiduals {
    pub fn rmse_log10(&self) -> f64 {
        if self.subthreshold_log10_errors.is_empty() {
            return 0.0;
        }
        let sum_sq: f64 = self.subthreshold_log10_errors.iter().map(|e| e * e).sum();
        (sum_sq / self.subthreshold_log10_errors.len() as f64).sqrt()
    }

    pub fn mape_above_pct(&self) -> f64 {
        if self.above_threshold_pct_errors.is_empty() {
            return 0.0;
        }
        let sum_abs: f64 = self.above_threshold_pct_errors.iter().map(|e| e.abs()).sum();
        sum_abs / self.above_threshold_pct_errors.len() as f64
    }

    /// Комбинированный вектор невязок для Trust-Region Reflective (TRF) least_squares.
    pub fn combined_vector(&self) -> Vec<f64> {
        let mut out = Vec::with_capacity(
            self.subthreshold_log10_errors.len()
                + self.above_threshold_pct_errors.len()
                + self.gm_rel_errors.len()
                + self.gds_rel_errors.len(),
        );
        // Нормируем процентную ошибку (делим на 100), чтобы масштабы с log10 были сопоставимы
        out.extend(self.subthreshold_log10_errors.iter().copied());
        out.extend(self.above_threshold_pct_errors.iter().map(|p| p / 100.0));
        out.extend(self.gm_rel_errors.iter().map(|g| g * 0.5));
        out.extend(self.gds_rel_errors.iter().map(|g| g * 0.5));
        out
    }
}

// ============================================================================
// Компактная физическая модель BSIM4-уровня для расчёта невязок фиттинга
// ============================================================================

#[derive(Debug, Clone)]
pub struct Bsim4ParameterSet {
    // Шаг 1: Длинный канал
    pub vth0: f64,
    pub k1: f64,
    pub k2: f64,
    // Шаг 2: Короткий канал
    pub dvt0: f64,
    pub dvt1: f64,
    // Шаг 3: Подвижность
    pub u0: f64, // м²/(В·с)
    pub ua: f64,
    pub ub: f64,
    pub uc: f64,
    // Шаг 4: Сопротивление / насыщение скорости
    pub rdsw: f64, // Ом·мкм
    pub vsat: f64, // м/с
    // Шаг 5: Выходные характеристики
    pub pclm: f64,
    pub pdibl: f64,
    // Шаг 6: Затворные ёмкости
    pub cgso: f64, // Ф/м
    pub cgdo: f64, // Ф/м
}

impl Bsim4ParameterSet {
    pub fn from_start_guesses(guesses: &StartGuesses) -> Self {
        Self {
            vth0: guesses.vth0_start,
            k1: 0.5,
            k2: 0.0,
            dvt0: 2.2,
            dvt1: 0.53,
            u0: guesses.u0_start_m2_vs,
            ua: 1e-9,
            ub: 1e-18,
            uc: -0.0465,
            rdsw: 200.0,
            vsat: 1.0e5,
            pclm: 1.3,
            pdibl: 0.08,
            cgso: 2.0e-10,
            cgdo: 2.0e-10,
        }
    }

    pub fn to_vec(&self) -> [f64; 15] {
        [
            self.vth0, self.k1, self.k2, self.dvt0, self.dvt1, self.u0, self.ua, self.ub, self.uc,
            self.rdsw, self.vsat, self.pclm, self.pdibl, self.cgso, self.cgdo,
        ]
    }

    pub fn from_vec(v: &[f64; 15]) -> Self {
        Self {
            vth0: v[0],
            k1: v[1],
            k2: v[2],
            dvt0: v[3],
            dvt1: v[4],
            u0: v[5],
            ua: v[6],
            ub: v[7],
            uc: v[8],
            rdsw: v[9],
            vsat: v[10],
            pclm: v[11],
            pdibl: v[12],
            cgso: v[13],
            cgdo: v[14],
        }
    }

    pub fn bounds() -> [(f64, f64); 15] {
        [
            (-2.0, 2.5),      // 0: VTH0
            (0.01, 2.5),      // 1: K1
            (-0.5, 0.5),      // 2: K2
            (0.0, 10.0),      // 3: DVT0
            (0.05, 5.0),      // 4: DVT1
            (1e-4, 0.5),      // 5: U0 [м²/(В·с)]
            (-1e-8, 5e-8),    // 6: UA
            (-1e-17, 1e-16),  // 7: UB
            (-0.5, 0.5),      // 8: UC
            (0.0, 5000.0),    // 9: RDSW [Ом·мкм]
            (1e4, 5e5),       // 10: VSAT [м/с]
            (0.01, 10.0),     // 11: PCLM
            (0.0, 2.0),       // 12: PDIBL
            (0.0, 1e-8),      // 13: CGSO [Ф/м]
            (0.0, 1e-8),      // 14: CGDO [Ф/м]
        ]
    }

    pub fn names() -> [&'static str; 15] {
        [
            "VTH0", "K1", "K2", "DVT0", "DVT1", "U0", "UA", "UB", "UC", "RDSW", "VSAT", "PCLM",
            "PDIBL", "CGSO", "CGDO",
        ]
    }

    pub fn step_numbers() -> [u8; 15] {
        [1, 1, 1, 2, 2, 3, 3, 3, 3, 4, 4, 5, 5, 6, 6]
    }

    /// Оценка тока стока Id (гладкая C²-модель плавного перехода подпорог -> сильная инверсия).
    pub fn eval_id(
        &self,
        device: &DeviceParameters,
        vg: f64,
        vd: f64,
        w_um: f64,
        l_um: f64,
        nfactor: f64,
    ) -> f64 {
        let vt = device.thermal_voltage_v();
        let cox = device.cox_f_m2();
        let tox_m = device.tox_nm * 1e-9;
        let w_m = w_um * 1e-6;
        let l_m = l_um * 1e-6;

        // Поправка короткого канала и DIBL
        let sce_rolloff = self.dvt0 * (-self.dvt1 * l_um / 0.1_f64.max(tox_m * 1e6)).exp();
        let dibl_shift = self.pdibl * vd * (0.1 / l_um.max(0.01));
        let box_shift = match device.buried_oxide_nm {
            Some(t_box) if device.device_type == DeviceType::FdSoi => {
                -0.02 * (25.0 / t_box.max(1.0))
            }
            _ => 0.0,
        };

        let vth_eff = self.vth0 + 0.05 * self.k1 - 0.01 * self.k2 - sce_rolloff - dibl_shift + box_shift;

        // Гладкое эффективное перенапряжение (softplus / log(1 + exp)) для C²-гладкости
        let n_vt = nfactor.max(1.0) * vt;
        let arg = ((vg - vth_eff) / (2.0 * n_vt)).clamp(-40.0, 40.0);
        let v_gst_eff = 2.0 * n_vt * (1.0 + arg.exp()).ln();

        // Деградация подвижности вертикальным полем
        let e_eff = (v_gst_eff + 2.0 * vth_eff.abs()) / (6.0 * tox_m.max(1e-10));
        let mob_denom = (1.0 + self.ua * e_eff + self.ub * e_eff * e_eff).max(0.2);
        let mu_eff = self.u0 / mob_denom;

        // Насыщение скорости носителей
        let v_dsat = (v_gst_eff * self.vsat * l_m)
            / (v_gst_eff + self.vsat * l_m / mu_eff.max(1e-6)).max(1e-9);
        let m_smooth = 4.0;
        let vd_eff = vd
            / (1.0 + (vd / v_dsat.max(1e-4)).powf(m_smooth)).powf(1.0 / m_smooth);

        // Модуляция длины канала (CLM)
        let clm_factor = 1.0 + self.pclm * (vd - vd_eff).max(0.0) / l_um.max(0.01);

        let id_intrinsic = mu_eff
            * cox
            * (w_m / l_m)
            * (v_gst_eff * vd_eff - 0.5 * vd_eff * vd_eff)
            * clm_factor;

        // Учёт последовательного сопротивления исток/сток RDSW [Ом·мкм]
        let r_series = self.rdsw / w_um.max(1e-3);
        let id_with_rs = id_intrinsic / (1.0 + r_series * id_intrinsic / vd.max(1e-4));

        id_with_rs.max(1e-15)
    }
}

// ============================================================================
// Шаги 3 и 4 · Пошаговый TRF-фиттинг BSIM4 (FR-EXT-02, FR-EXT-04)
// ============================================================================

fn evaluate_residuals(
    params: &Bsim4ParameterSet,
    device: &DeviceParameters,
    input: &ExtractionInput,
    nfactor: f64,
) -> RegimeResiduals {
    let mut res = RegimeResiduals::default();

    for sweep in &input.id_vg_graphs {
        let w_um = sweep.instance_width_um.unwrap_or_else(|| device.effective_width_um());
        let l_um = sweep.instance_length_um.unwrap_or(device.length_um);
        let vd = sweep.bias_v;

        let mut id_sim_vec = Vec::with_capacity(sweep.v.len());
        for (&vg, &id_meas) in sweep.v.iter().zip(sweep.i.iter()) {
            let id_meas_abs = id_meas.abs().max(1e-15);
            let id_sim = params.eval_id(device, vg, vd, w_um, l_um, nfactor);
            id_sim_vec.push(id_sim);

            if vg < params.vth0 {
                // Подпороговый режим: log10(Id_model) - log10(Id_data)
                res.subthreshold_log10_errors
                    .push(id_sim.log10() - id_meas_abs.log10());
            } else {
                // Надпороговый режим: (Id_model - Id_data) / Id_data * 100 %
                res.above_threshold_pct_errors
                    .push((id_sim - id_meas_abs) / id_meas_abs * 100.0);
            }
        }

        // Накопление невязок крутизны gm
        let gm_meas = numerical_derivative(&sweep.v, &sweep.i);
        let gm_sim = numerical_derivative(&sweep.v, &id_sim_vec);
        for (&gm_m, &gm_s) in gm_meas.iter().zip(gm_sim.iter()) {
            if gm_m.abs() > 1e-9 {
                res.gm_rel_errors.push((gm_s - gm_m) / gm_m.abs());
            }
        }
    }

    for sweep in &input.id_vd_graphs {
        let w_um = sweep.instance_width_um.unwrap_or_else(|| device.effective_width_um());
        let l_um = sweep.instance_length_um.unwrap_or(device.length_um);
        let vg = sweep.bias_v;

        let mut id_sim_vec = Vec::with_capacity(sweep.v.len());
        for (&vd, &id_meas) in sweep.v.iter().zip(sweep.i.iter()) {
            let id_meas_abs = id_meas.abs().max(1e-15);
            let id_sim = params.eval_id(device, vg, vd, w_um, l_um, nfactor);
            id_sim_vec.push(id_sim);

            if vg < params.vth0 {
                res.subthreshold_log10_errors
                    .push(id_sim.log10() - id_meas_abs.log10());
            } else {
                res.above_threshold_pct_errors
                    .push((id_sim - id_meas_abs) / id_meas_abs * 100.0);
            }
        }

        // Накопление невязок выходной проводимости gds
        let gds_meas = numerical_derivative(&sweep.v, &sweep.i);
        let gds_sim = numerical_derivative(&sweep.v, &id_sim_vec);
        for (&gds_m, &gds_s) in gds_meas.iter().zip(gds_sim.iter()) {
            if gds_m.abs() > 1e-9 {
                res.gds_rel_errors.push((gds_s - gds_m) / gds_m.abs());
            }
        }
    }

    res
}

/// Оптимизация выбранного подмножества параметров методом Левенберга–Марквардта с проекцией на границы (TRF-подобный).
fn optimize_parameter_subset(
    x: &mut [f64; 15],
    active_indices: &[usize],
    bounds: &[(f64, f64); 15],
    device: &DeviceParameters,
    input: &ExtractionInput,
    nfactor: f64,
    max_iters: usize,
) {
    if active_indices.is_empty() {
        return;
    }

    let k = active_indices.len();
    let mut lambda = 1e-2_f64;

    let mut current_res = evaluate_residuals(&Bsim4ParameterSet::from_vec(x), device, input, nfactor)
        .combined_vector();
    if current_res.is_empty() {
        return;
    }
    let m = current_res.len();
    let mut current_cost: f64 = current_res.iter().map(|r| r * r).sum();

    for _iter in 0..max_iters {
        // Численный Якобиан m x k по активным параметрам
        let mut jac = vec![vec![0.0_f64; k]; m];
        for (col, &param_idx) in active_indices.iter().enumerate() {
            let (lo, hi) = bounds[param_idx];
            let span = (hi - lo).max(1e-12);
            let eps = (x[param_idx].abs() * 1e-5).max(span * 1e-6);

            let mut x_plus = *x;
            x_plus[param_idx] = (x_plus[param_idx] + eps).min(hi);
            let actual_step = x_plus[param_idx] - x[param_idx];
            if actual_step.abs() < 1e-18 {
                continue;
            }

            let res_plus =
                evaluate_residuals(&Bsim4ParameterSet::from_vec(&x_plus), device, input, nfactor)
                    .combined_vector();
            for row in 0..m {
                jac[row][col] = (res_plus[row] - current_res[row]) / actual_step;
            }
        }

        // Формирование J^T J + lambda * diag(J^T J) и -J^T r
        let mut jtj = vec![vec![0.0_f64; k]; k];
        let mut jtr = vec![0.0_f64; k];
        for row in 0..m {
            for i in 0..k {
                jtr[i] -= jac[row][i] * current_res[row];
                for j in 0..k {
                    jtj[i][j] += jac[row][i] * jac[row][j];
                }
            }
        }
        for i in 0..k {
            jtj[i][i] += lambda * jtj[i][i].max(1e-8);
        }

        let Some(delta) = solve_dense_linear(&jtj, &jtr) else {
            lambda *= 10.0;
            continue;
        };

        let mut x_trial = *x;
        let mut max_rel_step = 0.0_f64;
        for (col, &param_idx) in active_indices.iter().enumerate() {
            let (lo, hi) = bounds[param_idx];
            let next_val = (x_trial[param_idx] + delta[col]).clamp(lo, hi);
            let rel = (next_val - x_trial[param_idx]).abs() / x_trial[param_idx].abs().max(1e-6);
            if rel > max_rel_step {
                max_rel_step = rel;
            }
            x_trial[param_idx] = next_val;
        }

        let trial_res =
            evaluate_residuals(&Bsim4ParameterSet::from_vec(&x_trial), device, input, nfactor)
                .combined_vector();
        let trial_cost: f64 = trial_res.iter().map(|r| r * r).sum();

        if trial_cost < current_cost {
            *x = x_trial;
            current_res = trial_res;
            current_cost = trial_cost;
            lambda = (lambda * 0.3).max(1e-7);
            if max_rel_step < 1e-6 {
                break;
            }
        } else {
            lambda *= 5.0;
            if lambda > 1e8 {
                break;
            }
        }
    }
}

fn solve_dense_linear(a: &[Vec<f64>], b: &[f64]) -> Option<Vec<f64>> {
    let n = b.len();
    let mut aug = vec![vec![0.0; n + 1]; n];
    for i in 0..n {
        for j in 0..n {
            aug[i][j] = a[i][j];
        }
        aug[i][n] = b[i];
    }

    for col in 0..n {
        let mut pivot = col;
        for row in (col + 1)..n {
            if aug[row][col].abs() > aug[pivot][col].abs() {
                pivot = row;
            }
        }
        if aug[pivot][col].abs() < 1e-18 {
            return None;
        }
        aug.swap(col, pivot);

        let diag = aug[col][col];
        for j in col..=n {
            aug[col][j] /= diag;
        }
        for row in 0..n {
            if row != col {
                let factor = aug[row][col];
                for j in col..=n {
                    aug[row][j] -= factor * aug[col][j];
                }
            }
        }
    }

    Some((0..n).map(|i| aug[i][n]).collect())
}

// ============================================================================
// Шаг 5 · Curve metrics (независимо от фита)
// ============================================================================

pub fn extract_curve_metrics(device: &DeviceParameters, input: &ExtractionInput) -> CurveMetrics {
    let mut metrics = CurveMetrics::default();

    // 1. Метрики передаточной характеристики Id-Vg
    if let Some(id_vg) = input.id_vg_graphs.first() {
        if id_vg.v.len() >= 3 && id_vg.v.len() == id_vg.i.len() {
            let mut ion = 0.0_f64;
            let mut ioff = f64::INFINITY;
            for &curr in &id_vg.i {
                let abs_i = curr.abs();
                if abs_i > ion {
                    ion = abs_i;
                }
                if abs_i > 0.0 && abs_i < ioff {
                    ioff = abs_i;
                }
            }
            metrics.ion_a = Some(ion);
            if ioff.is_finite() {
                metrics.ioff_a = Some(ioff);
                metrics.ion_ioff_ratio = Some(ion / ioff);
            }

            // GM_MAX_S
            let gm = numerical_derivative(&id_vg.v, &id_vg.i);
            let gm_max = gm.iter().copied().fold(f64::NEG_INFINITY, f64::max);
            if gm_max.is_finite() {
                metrics.gm_max_s = Some(gm_max);
            }

            // SS = 1000 * min |dVg / dlog10(Id)| [мВ/дек]
            let log10_i: Vec<f64> = id_vg.i.iter().map(|i| i.abs().max(1e-18).log10()).collect();
            let dlog_dv = numerical_derivative(&id_vg.v, &log10_i);
            let mut ss_min = f64::INFINITY;
            for &slope in &dlog_dv {
                if slope > 1e-6 {
                    let ss_candidate = 1000.0 / slope;
                    if ss_candidate < ss_min {
                        ss_min = ss_candidate;
                    }
                }
            }
            if ss_min.is_finite() {
                metrics.ss_mv_dec = Some(ss_min);
            }

            // VTH_CONSTANT_CURRENT: Vg при Id = (W / L) * 1e-7 А
            let target_id = (device.effective_width_um() / device.length_um.max(1e-4)) * 1e-7;
            metrics.vth_constant_current_v = interpolate_x_for_y(&id_vg.v, &id_vg.i, target_id);
        }
    }

    // 2. Метрики выходной характеристики Id-Vd (Ron при Vd->0, gds в насыщении)
    if let Some(id_vd) = input.id_vd_graphs.last() {
        if id_vd.v.len() >= 3 && id_vd.v.len() == id_vd.i.len() {
            let gds_vec = numerical_derivative(&id_vd.v, &id_vd.i);

            // Ron = dVd / dId в первой ненулевой точке (Vd -> 0)
            if let Some(&g_lin) = gds_vec.first() {
                if g_lin.abs() > 1e-15 {
                    metrics.ron_ohm = Some(1.0 / g_lin.abs());
                }
            }

            // gds в области насыщения (последняя четверть свипа Vd)
            let sat_start = (gds_vec.len() * 3) / 4;
            let sat_slice = &gds_vec[sat_start..];
            if !sat_slice.is_empty() {
                let gds_sat = sat_slice.iter().sum::<f64>() / sat_slice.len() as f64;
                metrics.gds_s = Some(gds_sat);
            }
        }
    }

    // 3. Напряжение пробоя VBR_KNEE = Vd в точке max dlog(Id)/dVd
    if let Some(br) = &input.breakdown_graph {
        if br.v.len() >= 3 && br.v.len() == br.i.len() {
            let log_i: Vec<f64> = br.i.iter().map(|i| i.abs().max(1e-18).ln()).collect();
            let dlog_dvd = numerical_derivative(&br.v, &log_i);
            let mut best_idx = 0usize;
            let mut max_slope = f64::NEG_INFINITY;
            for (idx, &s) in dlog_dvd.iter().enumerate() {
                if s.is_finite() && s > max_slope {
                    max_slope = s;
                    best_idx = idx;
                }
            }
            if max_slope.is_finite() {
                metrics.vbr_knee_v = Some(br.v[best_idx]);
            }
        }
    }

    // 4. Затворные ёмкости Cgg, Cgso, Cgdo
    let w_m = device.effective_width_um() * 1e-6;
    if let Some(cv) = input.cv_graphs.first() {
        if !cv.cgg_f.is_empty() {
            let c_mean = cv.cgg_f.iter().sum::<f64>() / cv.cgg_f.len() as f64;
            let c_min = cv.cgg_f.iter().copied().fold(f64::INFINITY, f64::min);
            metrics.cgg_mean_f = Some(c_mean);
            let overlap_per_side = (0.5 * c_min.max(0.0)) / w_m.max(1e-9);
            metrics.cgso_f_m = Some(overlap_per_side);
            metrics.cgdo_f_m = Some(overlap_per_side);
        }
    } else {
        // Геометрическая оценка Cgg = Cox * W * L
        let c_geom = device.cox_f_m2() * w_m * (device.length_um * 1e-6);
        metrics.cgg_mean_f = Some(c_geom);
        metrics.cgso_f_m = Some(0.1 * c_geom / w_m.max(1e-9));
        metrics.cgdo_f_m = Some(0.1 * c_geom / w_m.max(1e-9));
    }

    // 5. ВЧ-метрики fT и fmax (|h21| = 1 и Mason U = 1)
    if let Some(ac) = &input.ac_graph {
        metrics.ft_hz = find_unity_crossing_freq(&ac.freq_hz, &ac.h21_mag);
        metrics.fmax_hz = find_unity_crossing_freq(&ac.freq_hz, &ac.mason_u);
    }

    // 6. Переходные характеристики TON / TOFF 10-90% и заряд затвора Qg = \int Ig dt
    if let Some(tr) = &input.transient_graph {
        if tr.time_s.len() >= 3 {
            metrics.ton_10_90_s = compute_transition_10_90(&tr.time_s, &tr.v_out, true);
            metrics.toff_90_10_s = compute_transition_10_90(&tr.time_s, &tr.v_out, false);
            if tr.ig_a.len() == tr.time_s.len() {
                let mut qg = 0.0_f64;
                for i in 1..tr.time_s.len() {
                    let dt = (tr.time_s[i] - tr.time_s[i - 1]).max(0.0);
                    qg += 0.5 * (tr.ig_a[i - 1].abs() + tr.ig_a[i].abs()) * dt;
                }
                metrics.qg_c = Some(qg);
            }
        }
    }

    metrics
}

fn interpolate_x_for_y(x: &[f64], y: &[f64], target_y: f64) -> Option<f64> {
    for i in 1..x.len().min(y.len()) {
        let y0 = y[i - 1].abs();
        let y1 = y[i].abs();
        if (y0 <= target_y && target_y <= y1) || (y1 <= target_y && target_y <= y0) {
            let dy = y1 - y0;
            if dy.abs() < 1e-24 {
                return Some(x[i]);
            }
            let alpha = (target_y - y0) / dy;
            return Some(x[i - 1] + alpha * (x[i] - x[i - 1]));
        }
    }
    None
}

fn find_unity_crossing_freq(freq: &[f64], gain: &[f64]) -> Option<f64> {
    for i in 1..freq.len().min(gain.len()) {
        let g0 = gain[i - 1];
        let g1 = gain[i];
        if (g0 >= 1.0 && g1 <= 1.0) || (g0 <= 1.0 && g1 >= 1.0) {
            let dg = g1 - g0;
            if dg.abs() < 1e-15 {
                return Some(freq[i]);
            }
            let alpha = (1.0 - g0) / dg;
            return Some(freq[i - 1] + alpha * (freq[i] - freq[i - 1]));
        }
    }
    None
}

fn compute_transition_10_90(time: &[f64], signal: &[f64], rising: bool) -> Option<f64> {
    let n = time.len().min(signal.len());
    if n < 3 {
        return None;
    }
    let v_min = signal.iter().copied().fold(f64::INFINITY, f64::min);
    let v_max = signal.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let swing = v_max - v_min;
    if swing <= 1e-12 {
        return None;
    }
    let level_10 = v_min + 0.1 * swing;
    let level_90 = v_min + 0.9 * swing;

    let t10 = interpolate_x_for_y(time, signal, level_10)?;
    let t90 = interpolate_x_for_y(time, signal, level_90)?;
    if rising && t90 >= t10 {
        Some(t90 - t10)
    } else if !rising && t10 >= t90 {
        Some(t10 - t90)
    } else {
        Some((t90 - t10).abs())
    }
}

// ============================================================================
// Шаг 6 · Гладкость ёмкостных выражений (FR-EXT-03, устранение POLY(2))
// ============================================================================

pub fn build_smooth_capacitance_model(
    device: &DeviceParameters,
    cv_graphs: &[CvSweep],
    vth0: f64,
) -> SmoothCapacitanceModel {
    let w_m = device.effective_width_um() * 1e-6;
    let l_m = device.length_um * 1e-6;
    let c_ox_total = device.cox_f_m2() * w_m * l_m;

    let (c_min, c_max, v_fb) = if let Some(cv) = cv_graphs.first() {
        if !cv.cgg_f.is_empty() {
            let min_c = cv.cgg_f.iter().copied().fold(f64::INFINITY, f64::min).max(0.0);
            let max_c = cv.cgg_f.iter().copied().fold(f64::NEG_INFINITY, f64::max).max(min_c);
            let mid_c = 0.5 * (min_c + max_c);
            let v_mid = interpolate_x_for_y(&cv.vg, &cv.cgg_f, mid_c).unwrap_or(vth0);
            (min_c, max_c, v_mid)
        } else {
            (0.15 * c_ox_total, c_ox_total, vth0)
        }
    } else {
        (0.15 * c_ox_total, c_ox_total, vth0)
    };

    SmoothCapacitanceModel {
        expression_family: "C1/C2-continuous tanh/EKV (without POLY(2))".into(),
        c_min_f: c_min,
        c_max_f: c_max,
        v_flatband_v: v_fb,
        transition_slope_v: (2.0 * device.thermal_voltage_v()).max(0.025),
        poly2_eliminated: true,
    }
}

// ============================================================================
// Шаг 7 · Главный конвейер экстракции и генерация SPICE-карты
// ============================================================================

pub fn extract_compact_model(
    device: &DeviceParameters,
    input: &ExtractionInput,
    target_simulator: TargetSimulator,
) -> Result<SpiceModelArtifact, ExtractionError> {
    if device.width_um <= 0.0 || device.length_um <= 0.0 || device.tox_nm <= 0.0 {
        return Err(ExtractionError::InvalidGeometry(
            "W, L и Tox должны быть строго положительными.".into(),
        ));
    }

    let strategy = route_extractor(device.device_type);

    // Шаг 5: Независимый расчёт curve metrics
    let curve_metrics = extract_curve_metrics(device, input);

    // Шаг 1: Стартовые приближения по первой передаточной ВАХ (если доступна)
    let guesses = if let Some(first_id_vg) = input.id_vg_graphs.first() {
        compute_start_guesses(device, first_id_vg)?
    } else if strategy == ExtractorStrategy::BehavioralSubckt {
        StartGuesses {
            vth0_start: 0.4,
            u0_start_m2_vs: (device.electron_mobility_cm2_vs * 1e-4).max(1e-4),
            gm_max_s: curve_metrics.gm_max_s.unwrap_or(1e-4),
            vg_at_gm_max: 0.8,
        }
    } else {
        return Err(ExtractionError::MissingTransferCurve(
            "Для экстракции компактной модели требуется хотя бы одна кривая Id-Vg.".into(),
        ));
    };

    let nfactor = curve_metrics
        .ss_mv_dec
        .map(|ss| {
            let ideal_ss = 1000.0 * device.thermal_voltage_v() * 10.0_f64.ln();
            (ss / ideal_ss).clamp(1.0, 5.0)
        })
        .unwrap_or(1.2);

    // Шаг 3 & 4: Пошаговая последовательность BSIM4 (FR-EXT-04)
    let mut bsim = Bsim4ParameterSet::from_start_guesses(&guesses);
    if let Some(cgso) = curve_metrics.cgso_f_m {
        bsim.cgso = cgso;
    }
    if let Some(cgdo) = curve_metrics.cgdo_f_m {
        bsim.cgdo = cgdo;
    }

    let mut x = bsim.to_vec();
    let bounds = Bsim4ParameterSet::bounds();

    // Последовательность из 6 шагов с фиксацией найденных параметров
    let step_groups: [&[usize]; 6] = [
        &[0, 1, 2],       // 1. Длинный канал: VTH0, K1, K2
        &[3, 4],          // 2. Короткий канал: DVT0, DVT1
        &[5, 6, 7, 8],    // 3. Подвижность: U0, UA, UB, UC
        &[9, 10],         // 4. Сопротивление / скорость: RDSW, VSAT
        &[11, 12],        // 5. Выходные: PCLM, PDIBL
        &[13, 14],        // 6. Затворные ёмкости: CGSO, CGDO
    ];

    for group in step_groups {
        optimize_parameter_subset(&mut x, group, &bounds, device, input, nfactor, 25);
    }

    let fitted = Bsim4ParameterSet::from_vec(&x);
    let final_residuals = evaluate_residuals(&fitted, device, input, nfactor);
    let rmse_log10 = final_residuals.rmse_log10();
    let mape_above = final_residuals.mape_above_pct();
    let fitness_score = 1.0 / (1.0 + rmse_log10 + 0.01 * mape_above);

    // Аудит параметров с фиксацией bound_hit
    let names = Bsim4ParameterSet::names();
    let steps = Bsim4ParameterSet::step_numbers();
    let mut parameter_records = Vec::with_capacity(15);
    for i in 0..15 {
        let (lo, hi) = bounds[i];
        let tol = (hi - lo).max(1e-12) * 1e-5;
        let hit = (x[i] - lo).abs() <= tol || (hi - x[i]).abs() <= tol;
        parameter_records.push(ExtractedParameterRecord {
            name: names[i].to_string(),
            value: x[i],
            step_no: steps[i],
            lower_bound: lo,
            upper_bound: hi,
            bound_hit: hit,
        });
    }

    // Шаг 6: Гладкость ёмкостных выражений (FR-EXT-03)
    let smooth_cap = build_smooth_capacitance_model(device, &input.cv_graphs, fitted.vth0);

    // Шаг 7: Генерация итоговой карты / JSON-отчёта
    let emit_model_card =
        device.device_type == DeviceType::PlanarMosfet || input.force_level_mode;

    if emit_model_card {
        let level = match strategy {
            ExtractorStrategy::MultigateCmgLevel72 => 72,
            _ => match target_simulator {
                TargetSimulator::MicroCap12 => 14,
                _ => 54,
            },
        };

        let mut lines = Vec::new();
        lines.push(format!(
            "* TSPICE Laboratory extracted compact model for {} ({:?})",
            device.name, device.device_type
        ));
        lines.push(format!(".MODEL {} NMOS (", device.name));
        lines.push(format!("+ LEVEL   = {}", level));
        lines.push(format!("+ VERSION = 4.8.2"));
        lines.push(format!("+ TNOM    = {:.2}", device.temperature_k - 273.15));
        lines.push(format!("+ TOXE    = {:.6e}", device.tox_nm * 1e-9));
        lines.push(format!("+ VTH0    = {:.6}", fitted.vth0));
        lines.push(format!("+ K1      = {:.6}", fitted.k1));
        lines.push(format!("+ K2      = {:.6}", fitted.k2));
        lines.push(format!("+ DVT0    = {:.6}", fitted.dvt0));
        lines.push(format!("+ DVT1    = {:.6}", fitted.dvt1));
        lines.push(format!("+ U0      = {:.6}", fitted.u0 * 1e4)); // перевод в см²/(В·с)
        lines.push(format!("+ UA      = {:.6e}", fitted.ua));
        lines.push(format!("+ UB      = {:.6e}", fitted.ub));
        lines.push(format!("+ UC      = {:.6}", fitted.uc));
        lines.push(format!("+ RDSW    = {:.6}", fitted.rdsw));
        lines.push(format!("+ VSAT    = {:.6e}", fitted.vsat));
        lines.push(format!("+ PCLM    = {:.6}", fitted.pclm));
        lines.push(format!("+ PDIBLC1 = {:.6}", fitted.pdibl));
        lines.push(format!("+ CGSO    = {:.6e}", fitted.cgso));
        lines.push(format!("+ CGDO    = {:.6e}", fitted.cgdo));
        lines.push(format!("+ NFACTOR = {:.6}", nfactor));
        lines.push(")".to_string());

        let disclaimer = if device.device_type != DeviceType::PlanarMosfet {
            Some(
                "Запрошенная совместимость LEVEL; не заменяет специализированную физическую модель семейства."
                    .to_string(),
            )
        } else {
            None
        };

        Ok(SpiceModelArtifact {
            strategy,
            model_type: if level == 72 {
                "BSIM-CMG".into()
            } else {
                "BSIM4".into()
            },
            level: Some(level),
            target_simulator,
            model_text: lines.join("\n"),
            fitness_score,
            rmse_log10_subthreshold: rmse_log10,
            mape_above_threshold_pct: mape_above,
            parameters: parameter_records,
            curve_metrics,
            smooth_capacitance: Some(smooth_cap),
            compatibility_disclaimer: disclaimer,
        })
    } else {
        // Для не-planar семейств без LEVEL-режима: JSON-отчёт параметров + curve metrics без заявления совместимости
        let mut param_map = BTreeMap::new();
        for rec in &parameter_records {
            param_map.insert(rec.name.clone(), rec.value);
        }

        let json_report = format!(
            "{{\n  \"device\": \"{}\",\n  \"family\": \"{:?}\",\n  \"compatibility_claim\": false,\n  \"metrics\": {{\n    \"VTH_CONSTANT_CURRENT_V\": {:?},\n    \"ION_A\": {:?},\n    \"IOFF_A\": {:?},\n    \"SS_MV_DEC\": {:?},\n    \"GM_MAX_S\": {:?},\n    \"RON_OHM\": {:?},\n    \"GDS_S\": {:?}\n  }},\n  \"smooth_capacitance\": {{\n    \"family\": \"{}\",\n    \"c_min_f\": {:.6e},\n    \"c_max_f\": {:.6e},\n    \"poly2_eliminated\": {}\n  }}\n}}",
            device.name,
            device.device_type,
            curve_metrics.vth_constant_current_v,
            curve_metrics.ion_a,
            curve_metrics.ioff_a,
            curve_metrics.ss_mv_dec,
            curve_metrics.gm_max_s,
            curve_metrics.ron_ohm,
            curve_metrics.gds_s,
            smooth_cap.expression_family,
            smooth_cap.c_min_f,
            smooth_cap.c_max_f,
            smooth_cap.poly2_eliminated
        );

        Ok(SpiceModelArtifact {
            strategy,
            model_type: "JSON_CURVE_REPORT".into(),
            level: None,
            target_simulator,
            model_text: json_report,
            fitness_score,
            rmse_log10_subthreshold: rmse_log10,
            mape_above_threshold_pct: mape_above,
            parameters: parameter_records,
            curve_metrics,
            smooth_capacitance: Some(smooth_cap),
            compatibility_disclaimer: Some(
                "Отчёт параметров и curve metrics сформирован без заявления о BSIM-совместимости (FR-EXP-01)."
                    .to_string(),
            ),
        })
    }
}
