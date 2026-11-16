//! A-6 · Двухэтапная верификация карты модели и loopback-тестирование в EDA (FR-EXT-05/06/07)
//!
//! Реализует полный конвейер верификации:
//! 1. Предварительный фиттинг (BSIM4 шаги 1..6) -> карта-кандидат.
//! 2. Генерация и компиляция Verilog-A в OSDI через OpenVAF (для memristor / memtransistor).
//! 3. Loopback-тест в ngspice: эталонная карта -> ВАХ -> повторная экстракция -> сверка параметров.
//! 4. Сквозной прогон нетлиста в Micro-Cap 12 с подменой LEVEL (54 -> 14).
//! 5. Валидация фазового портрета петли гистерезиса (площадь, форма «восьмёрки», СКО <= 0.5%).

use std::collections::HashMap;
use std::fmt;
use std::time::{Duration, Instant};

/// Максимально допустимое время полного цикла верификации (NFR-PERF-04: <= 5 мин).
pub const DEFAULT_VERIFICATION_BUDGET_S: f64 = 300.0;

/// Максимально допустимое отклонение площади петли от референса в процентах (НФТ.4.2).
pub const MAX_HYSTERESIS_DEVIATION_PCT: f64 = 0.5;

/// Целевой EDA-симулятор.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EdaSimulator {
    Ngspice,
    MicroCap,
    Tanner,
    Xyce,
}

impl EdaSimulator {
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Ngspice => "ngspice",
            Self::MicroCap => "microcap",
            Self::Tanner => "tanner",
            Self::Xyce => "xyce",
        }
    }
}

/// Режим верификации (соответствует столбцу `mode` таблицы `verification_runs`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VerificationMode {
    Loopback,
    Hysteresis,
    Netlist,
}

impl VerificationMode {
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Loopback => "loopback",
            Self::Hysteresis => "hysteresis",
            Self::Netlist => "netlist",
        }
    }
}

/// Вариант Verilog-A модели для элементов памяти (D-7).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MemoryVerilogAFlavor {
    DriftStrukov,
    ThresholdMetalOxide,
}

/// Семейство прибора для выбора ветки верификации.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DeviceKind {
    PlanarMosfet,
    FdSoi,
    Finfet,
    GaaNanosheet,
    GaaNanowire,
    Tfet,
    Hemt,
    Memristor,
    Memtransistor,
    CadCustom,
}

impl DeviceKind {
    pub fn requires_verilog_a(&self) -> bool {
        matches!(self, Self::Memristor | Self::Memtransistor)
    }
}

/// Типизированные ошибки этапа верификации.
#[derive(Debug, Clone, PartialEq)]
pub enum VerificationError {
    InvalidInput(String),
    EdaToolUnavailable { simulator: String, reason: String },
    OpenVafCompilationFailed(String),
    TimeoutExceeded { elapsed_s: f64, budget_s: f64 },
    DegenerateHysteresisLoop { loop_area_a_v: f64, reason: String },
}

impl fmt::Display for VerificationError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidInput(msg) => write!(f, "Invalid verification input: {}", msg),
            Self::EdaToolUnavailable { simulator, reason } => {
                write!(f, "EDA tool '{}' unavailable: {}", simulator, reason)
            }
            Self::OpenVafCompilationFailed(msg) => {
                write!(f, "OpenVAF compilation failed: {}", msg)
            }
            Self::TimeoutExceeded { elapsed_s, budget_s } => {
                write!(
                    f,
                    "Verification budget exceeded: {:.2}s > {:.2}s (NFR-PERF-04)",
                    elapsed_s, budget_s
                )
            }
            Self::DegenerateHysteresisLoop { loop_area_a_v, reason } => {
                write!(
                    f,
                    "Degenerate hysteresis loop (area={:.4e} A*V): {}",
                    loop_area_a_v, reason
                )
            }
        }
    }
}

impl std::error::Error for VerificationError {}

/// Набор точек вольт-амперной характеристики (V -> I).
#[derive(Debug, Clone)]
pub struct IvCurve {
    pub name: String,
    pub voltage_v: Vec<f64>,
    pub current_a: Vec<f64>,
}

/// Параметры элемента памяти для генерации Verilog-A (D-7 / D-8).
#[derive(Debug, Clone)]
pub struct MemoryVerilogAParams {
    pub ron_ohm: f64,
    pub roff_ohm: f64,
    pub vset_v: f64,
    pub vreset_v: f64,
    pub mu_ion_cm2_vs: f64,
    pub active_layer_nm: f64,
    pub activation_energy_ev: f64,
    pub tnom_k: f64,
    pub x_init: f64,
    pub window_order: u32,
    pub gate_coupling: f64,
    pub relaxation_ms: f64,
}

impl Default for MemoryVerilogAParams {
    fn default() -> Self {
        Self {
            ron_ohm: 100.0,
            roff_ohm: 100_000.0,
            vset_v: 0.3,
            vreset_v: -0.3,
            mu_ion_cm2_vs: 1e-10,
            active_layer_nm: 10.0,
            activation_energy_ev: 0.6,
            tnom_k: 300.0,
            x_init: 0.5,
            window_order: 1,
            gate_coupling: 2.0,
            relaxation_ms: 100.0,
        }
    }
}

/// Кандидат компактной модели после предварительного фиттинга (Этап 1).
#[derive(Debug, Clone)]
pub struct CandidateModelCard {
    pub model_name: String,
    pub model_type: String,
    pub level: u32,
    pub parameters: HashMap<String, f64>,
    pub model_text: String,
    pub fitness_score: f64,
}

/// Результат компиляции Verilog-A в OSDI-модуль.
#[derive(Debug, Clone)]
pub struct OsdICompilationArtifact {
    pub verilog_a_source: String,
    pub osdi_binary_path: String,
    pub module_name: String,
}

/// Результат валидации петли гистерезиса.
#[derive(Debug, Clone)]
pub struct HysteresisValidationReport {
    pub loop_area_a_v: f64,
    pub is_figure_eight: bool,
    pub degenerate: bool,
    pub deviation_from_reference_pct: f64,
    pub passed: bool,
    pub diagnostic_message: String,
}

/// Итоговая запись для таблицы `verification_runs` и `spice_models`.
#[derive(Debug, Clone)]
pub struct VerificationRunRecord {
    pub task_id: u64,
    pub simulator: EdaSimulator,
    pub mode: VerificationMode,
    pub verified_model_text: String,
    pub final_parameters: HashMap<String, f64>,
    pub loop_area_a_v: Option<f64>,
    pub degenerate: bool,
    pub max_rel_error: f64,
    pub passed: bool,
    pub elapsed_s: f64,
    pub log_lines: Vec<String>,
}

// ============================================================================
// Абстракции внешних EDA-инструментов (Dependency Inversion для тестов и CLI)
// ============================================================================

/// Этап 1: Предварительный фиттинг в Python/Rust (BSIM4 шаги 1..6).
pub trait PreliminaryFitter {
    fn fit_candidate(
        &self,
        device_kind: DeviceKind,
        measured_curves: &[IvCurve],
    ) -> Result<CandidateModelCard, VerificationError>;

    fn extract_from_reference_iv(
        &self,
        device_kind: DeviceKind,
        reference_curves: &[IvCurve],
    ) -> Result<HashMap<String, f64>, VerificationError>;
}

/// Компилятор OpenVAF: Verilog-A -> OSDI с автоматическим дифференцированием (AD).
pub trait OpenVafCompiler {
    fn compile_to_osdi(
        &self,
        module_name: &str,
        verilog_a_code: &str,
    ) -> Result<OsdICompilationArtifact, VerificationError>;
}

/// Этап 2а: Симулятор ngspice для loopback-прогона (FR-EXT-07).
pub trait NgspiceRunner {
    fn run_loopback_iv(
        &self,
        card: &CandidateModelCard,
        osdi: Option<&OsdICompilationArtifact>,
    ) -> Result<Vec<IvCurve>, VerificationError>;
}

/// Этап 2б: Симулятор Micro-Cap 12 для сквозного прогона нетлиста (FR-EXT-06).
pub trait MicroCapRunner {
    fn run_full_netlist(
        &self,
        netlist_text: &str,
    ) -> Result<Vec<IvCurve>, VerificationError>;
}

// ============================================================================
// Детерминированная генерация Verilog-A и подмена LEVEL (D-7, A-6)
// ============================================================================

/// Детерминированно генерирует код Verilog-A для мемристора или мемтранзистора.
pub fn generate_memory_verilog_a(
    is_memtransistor: bool,
    flavor: MemoryVerilogAFlavor,
    params: &MemoryVerilogAParams,
) -> Result<(String, String), VerificationError> {
    if params.ron_ohm <= 0.0 || params.roff_ohm <= params.ron_ohm {
        return Err(VerificationError::InvalidInput(
            "Требуется 0 < Ron < Roff для генерации Verilog-A.".into(),
        ));
    }
    if params.vset_v <= 0.0 || params.vreset_v >= 0.0 {
        return Err(VerificationError::InvalidInput(
            "Требуется Vset > 0 и Vreset < 0.".into(),
        ));
    }
    if !(0.0..=1.0).contains(&params.x_init) {
        return Err(VerificationError::InvalidInput(
            "x_init должен находиться в диапазоне [0, 1].".into(),
        ));
    }

    let module_name = if is_memtransistor {
        "tspice_memtransistor_va"
    } else {
        "tspice_memristor_va"
    };

    let flavor_comment = match flavor {
        MemoryVerilogAFlavor::DriftStrukov => "Windowed linear ion-drift (Strukov / Biolek)",
        MemoryVerilogAFlavor::ThresholdMetalOxide => "Threshold-activated metal-oxide switching",
    };

    let code = if !is_memtransistor {
        format!(
            r#"`include "constants.vams"
`include "disciplines.vams"

// {flavor_comment}
module {module_name}(p, n);
    inout p, n;
    electrical p, n, state_x;

    parameter real Ron = {ron:.6e};
    parameter real Roff = {roff:.6e};
    parameter real Vset = {vset:.6};
    parameter real Vreset = {vreset:.6};
    parameter real mu_ion = {mu_ion:.6e};
    parameter real D = {d_nm:.6e} * 1e-9;
    parameter real Ea = {ea:.6};
    parameter real Tnom = {tnom:.6};
    parameter real x_init = {x_init:.6};
    parameter integer window_order = {w_ord};

    real v_pn, i_pn, x_val, r_inst, win, drift_k, temp_factor;

    analog begin
        @(initial_step) begin
            V(state_x) <+ x_init;
        end

        v_pn = V(p, n);
        x_val = limexp(0.0) * V(state_x);
        if (x_val < 0.0) x_val = 0.0;
        if (x_val > 1.0) x_val = 1.0;

        temp_factor = exp(-(Ea / 8.617333e-5) * ((1.0 / $temperature) - (1.0 / Tnom)));
        drift_k = (mu_ion * 1e-4 * Ron / (D * D)) * temp_factor;

        r_inst = Roff + (Ron - Roff) * x_val;
        i_pn = v_pn / r_inst;
        I(p, n) <+ i_pn;

        if (v_pn >= Vset) begin
            win = 1.0 - pow(x_val, 2 * window_order);
            I(state_x) <+ ddt(V(state_x)) - drift_k * i_pn * win;
        end else if (v_pn <= Vreset) begin
            win = 1.0 - pow(1.0 - x_val, 2 * window_order);
            I(state_x) <+ ddt(V(state_x)) - drift_k * i_pn * win;
        end else begin
            I(state_x) <+ ddt(V(state_x));
        end
    end
endmodule
"#,
            flavor_comment = flavor_comment,
            module_name = module_name,
            ron = params.ron_ohm,
            roff = params.roff_ohm,
            vset = params.vset_v,
            vreset = params.vreset_v,
            mu_ion = params.mu_ion_cm2_vs,
            d_nm = params.active_layer_nm,
            ea = params.activation_energy_ev,
            tnom = params.tnom_k,
            x_init = params.x_init,
            w_ord = params.window_order.max(1),
        )
    } else {
        format!(
            r#"`include "constants.vams"
`include "disciplines.vams"

// {flavor_comment} (3-terminal Memtransistor)
module {module_name}(d, g, s);
    inout d, g, s;
    electrical d, g, s, state_x;

    parameter real Ron = {ron:.6e};
    parameter real Roff = {roff:.6e};
    parameter real Vset = {vset:.6};
    parameter real Vreset = {vreset:.6};
    parameter real mu_ion = {mu_ion:.6e};
    parameter real D = {d_nm:.6e} * 1e-9;
    parameter real Ea = {ea:.6};
    parameter real Tnom = {tnom:.6};
    parameter real x_init = {x_init:.6};
    parameter real gate_coupling = {coupling:.6};
    parameter real tau_ret = {tau_ms:.6e} * 1e-3;
    parameter real beta0 = 1.5e-4;

    real v_gs, v_ds, x_val, v_th, v_ov, mem_gain, id_val, win, drift_k;

    analog begin
        @(initial_step) begin
            V(state_x) <+ x_init;
        end

        v_gs = V(g, s);
        v_ds = V(d, s);
        x_val = V(state_x);
        if (x_val < 0.0) x_val = 0.0;
        if (x_val > 1.0) x_val = 1.0;

        v_th = 0.35 - 0.25 * (x_val - 0.5);
        v_ov = (v_gs > v_th) ? (v_gs - v_th) : 0.0;
        mem_gain = 1.0 + 0.15 * ln(Roff / Ron) / ln(10.0);
        id_val = 1e-12 + beta0 * v_ov * v_ov * v_ds * (1.0 + (mem_gain - 1.0) * x_val);
        I(d, s) <+ id_val;

        drift_k = (mu_ion * 1e-4 * Ron / (D * D))
                  * exp(-(Ea / 8.617333e-5) * ((1.0 / $temperature) - (1.0 / Tnom)));

        if (v_gs >= Vset) begin
            win = 1.0 - x_val * x_val;
            I(state_x) <+ ddt(V(state_x)) - drift_k * v_gs * win * (1.0 + gate_coupling);
        end else if (v_gs <= Vreset) begin
            win = 1.0 - (1.0 - x_val) * (1.0 - x_val);
            I(state_x) <+ ddt(V(state_x)) - drift_k * v_gs * win * (1.0 + gate_coupling);
        end else begin
            I(state_x) <+ ddt(V(state_x)) + x_val / tau_ret;
        end
    end
endmodule
"#,
            flavor_comment = flavor_comment,
            module_name = module_name,
            ron = params.ron_ohm,
            roff = params.roff_ohm,
            vset = params.vset_v,
            vreset = params.vreset_v,
            mu_ion = params.mu_ion_cm2_vs,
            d_nm = params.active_layer_nm,
            ea = params.activation_energy_ev,
            tnom = params.tnom_k,
            x_init = params.x_init,
            coupling = params.gate_coupling,
            tau_ms = params.relaxation_ms.max(1e-3),
        )
    };

    Ok((module_name.to_string(), code))
}

/// Подменяет LEVEL=54 (ngspice BSIM4) на LEVEL=14 для сквозного прогона в Micro-Cap 12 (A-6).
pub fn adapt_model_card_for_microcap(model_text: &str) -> String {
    model_text
        .replace("LEVEL=54", "LEVEL=14")
        .replace("LEVEL = 54", "LEVEL = 14")
        .replace("level=54", "level=14")
        .replace("level = 54", "level = 14")
}

// ============================================================================
// Валидация фазового портрета гистерезиса (A-6, VAL)
// ============================================================================

/// Вычисляет площадь замкнутой петли гистерезиса `∮ V dI` по формуле шнурования
/// и проверяет отсутствие вырождения в прямую линию и СКО от эталона <= 0.5%.
pub fn validate_hysteresis_portrait(
    curve: &IvCurve,
    reference_area_a_v: Option<f64>,
    max_deviation_pct: f64,
) -> HysteresisValidationReport {
    let n = curve.voltage_v.len().min(curve.current_a.len());
    if n < 6 {
        return HysteresisValidationReport {
            loop_area_a_v: 0.0,
            is_figure_eight: false,
            degenerate: true,
            deviation_from_reference_pct: 100.0,
            passed: false,
            diagnostic_message: "Недостаточно точек для анализа фазового портрета (< 6).".into(),
        };
    }

    // Площадь замкнутого контура по формуле Гаусса (шнурования)
    let mut signed_area2 = 0.0_f64;
    let mut v_max = 0.0_f64;
    let mut i_max = 0.0_f64;

    for idx in 0..n {
        let next = (idx + 1) % n;
        let v0 = curve.voltage_v[idx];
        let i0 = curve.current_a[idx];
        let v1 = curve.voltage_v[next];
        let i1 = curve.current_a[next];

        signed_area2 += v0 * i1 - v1 * i0;
        v_max = v_max.max(v0.abs());
        i_max = i_max.max(i0.abs());
    }

    let loop_area = 0.5 * signed_area2.abs();
    let bounding_box_area = (2.0 * v_max) * (2.0 * i_max);

    // Относительный коэффициент раскрытия петли: если < 1e-4, петля выродилась в прямую линию
    let fill_ratio = if bounding_box_area > 0.0 {
        loop_area / bounding_box_area
    } else {
        0.0
    };

    // Проверка «восьмёрки» (pinched hysteresis): около V ≈ 0 ток также проходит через 0
    let mut zero_crossing_min_i = f64::INFINITY;
    for (&v, &i) in curve.voltage_v.iter().zip(curve.current_a.iter()) {
        if v_max > 0.0 && v.abs() <= 0.05 * v_max {
            zero_crossing_min_i = zero_crossing_min_i.min(i.abs());
        }
    }
    let is_figure_eight = i_max > 0.0 && zero_crossing_min_i <= 0.1 * i_max;

    let degenerate = fill_ratio < 1e-4 || loop_area <= 0.0;

    let deviation_pct = match reference_area_a_v {
        Some(ref_area) if ref_area.abs() > 1e-30 => {
            ((loop_area - ref_area.abs()).abs() / ref_area.abs()) * 100.0
        }
        _ => 0.0,
    };

    let within_tolerance = deviation_pct <= max_deviation_pct;
    let passed = !degenerate && within_tolerance;

    let diagnostic_message = if degenerate {
        format!(
            "ОТБРАКОВКА: петля гистерезиса выродилась в прямую линию (fill_ratio={:.2e}).",
            fill_ratio
        )
    } else if !within_tolerance {
        format!(
            "ОТБРАКОВКА: отклонение площади петли от референса {:.3}% > {:.3}%.",
            deviation_pct, max_deviation_pct
        )
    } else {
        format!(
            "Петля валидна: S={:.4e} A*V, pinched_zero={}, отклонение={:.3}%.",
            loop_area, is_figure_eight, deviation_pct
        )
    };

    HysteresisValidationReport {
        loop_area_a_v: loop_area,
        is_figure_eight,
        degenerate,
        deviation_from_reference_pct: deviation_pct,
        passed,
        diagnostic_message,
    }
}

// ============================================================================
// Оркестратор двухэтапной верификации (A-6)
// ============================================================================

/// Запрос на запуск сессии верификации.
#[derive(Debug, Clone)]
pub struct VerificationRequest {
    pub task_id: u64,
    pub device_kind: DeviceKind,
    pub measured_curves: Vec<IvCurve>,
    pub memory_va_params: Option<MemoryVerilogAParams>,
    pub verilog_a_flavor: MemoryVerilogAFlavor,
    pub reference_loop_area_a_v: Option<f64>,
    pub max_loopback_rel_error: f64,
    pub budget_s: f64,
}

/// Оркестратор полного цикла двухэтапной верификации (A-6).
pub struct TwoStageVerificationOrchestrator<'a> {
    pub fitter: &'a dyn PreliminaryFitter,
    pub openvaf: &'a dyn OpenVafCompiler,
    pub ngspice: &'a dyn NgspiceRunner,
    pub microcap: &'a dyn MicroCapRunner,
}

impl<'a> TwoStageVerificationOrchestrator<'a> {
    pub fn run(&self, req: &VerificationRequest) -> Result<VerificationRunRecord, VerificationError> {
        let start_time = Instant::now();
        let budget = if req.budget_s > 0.0 {
            req.budget_s
        } else {
            DEFAULT_VERIFICATION_BUDGET_S
        };

        let mut logs = Vec::new();
        logs.push(format!("[A-6] Старт сессии верификации для task_id={}", req.task_id));

        // --------------------------------------------------------------------
        // Этап 1: Предварительный фит (BSIM4 шаги 1..6)
        // --------------------------------------------------------------------
        let candidate = self
            .fitter
            .fit_candidate(req.device_kind, &req.measured_curves)?;
        logs.push(format!(
            "[Этап 1] Получена карта-кандидат '{}' (LEVEL={}, fitness={:.4})",
            candidate.model_name, candidate.level, candidate.fitness_score
        ));
        Self::check_budget(&start_time, budget)?;

        // --------------------------------------------------------------------
        // Опционально: Генерация и компиляция Verilog-A -> OSDI (FR-EXT-05)
        // --------------------------------------------------------------------
        let osdi_artifact = if req.device_kind.requires_verilog_a() {
            let va_params = req.memory_va_params.clone().unwrap_or_default();
            let (module_name, va_code) = generate_memory_verilog_a(
                req.device_kind == DeviceKind::Memtransistor,
                req.verilog_a_flavor,
                &va_params,
            )?;
            let artifact = self.openvaf.compile_to_osdi(&module_name, &va_code)?;
            logs.push(format!(
                "[OpenVAF] Скомпилирован OSDI-модуль '{}' -> {}",
                artifact.module_name, artifact.osdi_binary_path
            ));
            Some(artifact)
        } else {
            None
        };
        Self::check_budget(&start_time, budget)?;

        // --------------------------------------------------------------------
        // Этап 2а: Loopback-тест в ngspice (FR-EXT-07)
        // --------------------------------------------------------------------
        let reference_iv = self
            .ngspice
            .run_loopback_iv(&candidate, osdi_artifact.as_ref())?;
        let recovered_params = self
            .fitter
            .extract_from_reference_iv(req.device_kind, &reference_iv)?;

        let max_loopback_err = compute_max_relative_error(&candidate.parameters, &recovered_params);
        logs.push(format!(
            "[Этап 2а · Loopback] Максимальная относительная ошибка восстановления параметров: {:.4}",
            max_loopback_err
        ));
        Self::check_budget(&start_time, budget)?;

        // --------------------------------------------------------------------
        // Этап 2б: Сквозной прогон нетлиста в Micro-Cap 12 (LEVEL 54 -> 14, FR-EXT-06)
        // --------------------------------------------------------------------
        let microcap_netlist = adapt_model_card_for_microcap(&candidate.model_text);
        let final_curves = self.microcap.run_full_netlist(&microcap_netlist)?;
        logs.push(format!(
            "[Этап 2б · Micro-Cap 12] Сквозной прогон нетлиста завершён (кривых: {})",
            final_curves.len()
        ));
        Self::check_budget(&start_time, budget)?;

        // Финальные параметры извлекаются ТОЛЬКО из данных сквозного прогона (FR-EXT-06)
        let final_parameters = self
            .fitter
            .extract_from_reference_iv(req.device_kind, &final_curves)?;

        // --------------------------------------------------------------------
        // Проверка фазового портрета (для памяти или кривых с гистерезисом)
        // --------------------------------------------------------------------
        let mut loop_area_opt = None;
        let mut degenerate = false;
        let mut hysteresis_passed = true;

        if req.device_kind.requires_verilog_a() {
            let primary_curve = final_curves.first().ok_or_else(|| {
                VerificationError::InvalidInput("Micro-Cap не вернул кривых для анализа петли.".into())
            })?;
            let report = validate_hysteresis_portrait(
                primary_curve,
                req.reference_loop_area_a_v,
                MAX_HYSTERESIS_DEVIATION_PCT,
            );
            logs.push(format!("[Валидатор гистерезиса] {}", report.diagnostic_message));

            loop_area_opt = Some(report.loop_area_a_v);
            degenerate = report.degenerate;
            hysteresis_passed = report.passed;
        }

        let loopback_ok = max_loopback_err <= req.max_loopback_rel_error;
        let overall_passed = loopback_ok && hysteresis_passed && !degenerate;
        let elapsed_s = start_time.elapsed().as_secs_f64();

        logs.push(format!(
            "[Итог A-6] passed={}, elapsed={:.2}s (лимит {:.0}s)",
            overall_passed, elapsed_s, budget
        ));

        Ok(VerificationRunRecord {
            task_id: req.task_id,
            simulator: EdaSimulator::MicroCap,
            mode: if req.device_kind.requires_verilog_a() {
                VerificationMode::Hysteresis
            } else {
                VerificationMode::Loopback
            },
            verified_model_text: microcap_netlist,
            final_parameters,
            loop_area_a_v: loop_area_opt,
            degenerate,
            max_rel_error: max_loopback_err,
            passed: overall_passed,
            elapsed_s,
            log_lines: logs,
        })
    }

    fn check_budget(start: &Instant, budget_s: f64) -> Result<(), VerificationError> {
        let elapsed_s = start.elapsed().as_secs_f64();
        if elapsed_s > budget_s {
            return Err(VerificationError::TimeoutExceeded {
                elapsed_s,
                budget_s,
            });
        }
        Ok(())
    }
}

/// Вычисляет максимальную относительную ошибку между эталонными и восстановленными параметрами.
pub fn compute_max_relative_error(
    expected: &HashMap<String, f64>,
    actual: &HashMap<String, f64>,
) -> f64 {
    let mut max_err = 0.0_f64;
    for (key, &exp_val) in expected {
        if let Some(&act_val) = actual.get(key) {
            let denom = exp_val.abs().max(1e-12);
            let rel_err = (act_val - exp_val).abs() / denom;
            if rel_err.is_finite() {
                max_err = max_err.max(rel_err);
            }
        }
    }
    max_err
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_degenerate_straight_line_rejected() {
        // Прямая линия I = 0.01 * V (вырожденная петля)
        let v: Vec<f64> = (-10..=10).map(|k| k as f64 * 0.1).collect();
        let i: Vec<f64> = v.iter().map(|&u| 0.01 * u).collect();
        let curve = IvCurve {
            name: "degenerate".into(),
            voltage_v: v,
            current_a: i,
        };

        let report = validate_hysteresis_portrait(&curve, None, 0.5);
        assert!(report.degenerate);
        assert!(!report.passed);
    }

    #[test]
    fn test_valid_pinched_hysteresis_loop() {
        // Замкнутая петля «восьмёрка»: I(t) = 1e-3 * sin(t) + 0.3e-3 * sin(2t), V(t) = sin(t)
        let steps = 200;
        let mut v = Vec::with_capacity(steps);
        let mut i = Vec::with_capacity(steps);
        for k in 0..steps {
            let theta = 2.0 * std::f64::consts::PI * (k as f64) / (steps as f64);
            v.push(theta.sin());
            i.push(1e-3 * theta.sin() + 3e-4 * (2.0 * theta).sin());
        }
        let curve = IvCurve {
            name: "memristor_iv".into(),
            voltage_v: v,
            current_a: i,
        };

        let report = validate_hysteresis_portrait(&curve, None, 0.5);
        assert!(!report.degenerate);
        assert!(report.is_figure_eight);
        assert!(report.passed);
        assert!(report.loop_area_a_v > 0.0);
    }

    #[test]
    fn test_level_54_to_14_substitution() {
        let ngspice_card = ".MODEL nmos_core NMOS LEVEL=54 VERSION=4.8.2 VTH0=0.42";
        let mc_card = adapt_model_card_for_microcap(ngspice_card);
        assert!(mc_card.contains("LEVEL=14"));
        assert!(!mc_card.contains("LEVEL=54"));
    }
}
