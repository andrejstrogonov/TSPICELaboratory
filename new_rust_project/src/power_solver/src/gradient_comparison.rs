//! A-10 · Стратегии получения якобиана и производных (g_m, g_ds, dI/dV).
//!
//! Модуль реализует пять стратегий вычисления производных и потребителя
//! Ньютона–Рафсона для глобальной матрицы DAE согласно диаграмме A-10:
//! - `NumericalDifference` — центральная конечная разность (только диагностика);
//! - `AnalyticCModel` — точные аналитические производные C² (BSIM4);
//! - `AutomaticDifferentiation` — forward-mode AD с дуальными числами и ddt()/idt();
//! - `SymbolicAndBytecode` — однократный парсинг в байт-код с soft-clipping (НФТ.3.1/3.2);
//! - `NeuralAutodiff` — backprop-якобиан с обязательным gate-контролем монотонности
//!   и C²-непрерывности против эффекта Рунге (НФТ.5.2).

use std::fmt;

/// Напряжения на выводах четырёхполюсника [Vg, Vd, Vs, Vb].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TerminalVoltages {
    pub vg: f64,
    pub vd: f64,
    pub vs: f64,
    pub vb: f64,
}

impl TerminalVoltages {
    pub fn new(vg: f64, vd: f64, vs: f64, vb: f64) -> Self {
        Self { vg, vd, vs, vb }
    }

    #[inline]
    pub fn vgs(&self) -> f64 {
        self.vg - self.vs
    }

    #[inline]
    pub fn vds(&self) -> f64 {
        self.vd - self.vs
    }
}

/// Строка локального якобиана стокового тока и заряда затвора по узловым напряжениям.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DeviceJacobian {
    /// Ток стока Id [А].
    pub id: f64,
    /// Крутизна g_m = dId / dVg [См].
    pub gm: f64,
    /// Выходная проводимость g_ds = dId / dVd [См].
    pub gds: f64,
    /// Подложечная крутизна g_mb = dId / dVb [См].
    pub gmb: f64,
    /// Заряд затвора Qg [Кл].
    pub qg: f64,
    /// Собственная ёмкость затвора C_gg = dQg / dVg [Ф].
    pub cgg: f64,
    /// Прямая передаточная ёмкость C_gd = -dQg / dVd [Ф].
    pub cgd: f64,
}

impl DeviceJacobian {
    pub fn is_finite(&self) -> bool {
        self.id.is_finite()
            && self.gm.is_finite()
            && self.gds.is_finite()
            && self.gmb.is_finite()
            && self.qg.is_finite()
            && self.cgg.is_finite()
            && self.cgd.is_finite()
    }
}

/// Назначение вызова производных в вычислительном пайплайне.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EvaluationPurpose {
    /// Диагностика и расчёт curve metrics (SS, gm_max, gds).
    DiagnosticMetrics,
    /// Схемная линеаризация в глобальной DAE-матрице Ньютона–Рафсона.
    CircuitNewtonLinearization,
}

/// Тип стратегии вычисления производных.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum JacobianStrategyKind {
    NumericalDifference,
    AnalyticCModel,
    AutomaticDifferentiation,
    SymbolicAndBytecode,
    NeuralAutodiff,
}

/// Ошибки вычисления якобиана и проверки гладкости.
#[derive(Debug, Clone, PartialEq)]
pub enum JacobianError {
    /// Попытка использовать численные разности для схемной линеаризации Ньютона.
    NumericalDiffForbiddenForNewton,
    /// Сработал gate проверки монотонности или C²-непрерывности (эффект Рунге, НФТ.5.2).
    SmoothnessGateRejected { reason: String, max_oscillation_score: f64 },
    /// Ошибка разбора поведенческого выражения при setup.
    BytecodeCompileError(String),
    /// Вырожденный или нечисловой якобиан (singular matrix / timestep too small).
    SingularOrNonFiniteJacobian(String),
    /// Расходимость итераций Ньютона–Рафсона.
    NewtonNotConverged { iterations: usize, residual: f64 },
}

impl fmt::Display for JacobianError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::NumericalDiffForbiddenForNewton => write!(
                f,
                "NumericalDifference разрешён только для диагностики, но запрещён для схемной линеаризации Ньютона"
            ),
            Self::SmoothnessGateRejected {
                reason,
                max_oscillation_score,
            } => write!(
                f,
                "NeuralAutodiff отклонён gate-проверкой (НФТ.5.2): {} (score={:.4e})",
                reason, max_oscillation_score
            ),
            Self::BytecodeCompileError(msg) => write!(f, "Ошибка компиляции байт-кода B-source: {}", msg),
            Self::SingularOrNonFiniteJacobian(msg) => {
                write!(f, "Вырожденный или нечисловой якобиан (timestep too small): {}", msg)
            }
            Self::NewtonNotConverged { iterations, residual } => write!(
                f,
                "Ньютон–Рафсон не сошёлся за {} итераций (невязка {:.3e})",
                iterations, residual
            ),
        }
    }
}

impl std::error::Error for JacobianError {}

/// Единый интерфейс поставщика якобиана и производных модели.
pub trait JacobianProvider {
    fn kind(&self) -> JacobianStrategyKind;

    /// Вычисляет токи, заряды и их производные в рабочей точке.
    fn evaluate(
        &self,
        v: TerminalVoltages,
        dt: f64,
        purpose: EvaluationPurpose,
    ) -> Result<DeviceJacobian, JacobianError>;
}

// ============================================================================
// 1. NumericalDifference (central diff) — только для диагностики
// ============================================================================

/// Центральная конечная разность: dI/dV ≈ (I(V+h) - I(V-h)) / (2h).
/// Используется исключительно в `extract_curve_metrics` и `compute_curve_metrics`.
pub struct NumericalDifference<F>
where
    F: Fn(TerminalVoltages) -> (f64, f64),
{
    pub step_v: f64,
    pub black_box_eval: F,
}

impl<F> NumericalDifference<F>
where
    F: Fn(TerminalVoltages) -> (f64, f64),
{
    pub fn new(step_v: f64, black_box_eval: F) -> Self {
        Self {
            step_v: step_v.abs().max(1e-7),
            black_box_eval,
        }
    }
}

impl<F> JacobianProvider for NumericalDifference<F>
where
    F: Fn(TerminalVoltages) -> (f64, f64),
{
    fn kind(&self) -> JacobianStrategyKind {
        JacobianStrategyKind::NumericalDifference
    }

    fn evaluate(
        &self,
        v: TerminalVoltages,
        _dt: f64,
        purpose: EvaluationPurpose,
    ) -> Result<DeviceJacobian, JacobianError> {
        if purpose == EvaluationPurpose::CircuitNewtonLinearization {
            return Err(JacobianError::NumericalDiffForbiddenForNewton);
        }

        let h = self.step_v;
        let inv_2h = 0.5 / h;
        let (id0, qg0) = (self.black_box_eval)(v);

        let (id_gp, qg_gp) = (self.black_box_eval)(TerminalVoltages { vg: v.vg + h, ..v });
        let (id_gm, qg_gm) = (self.black_box_eval)(TerminalVoltages { vg: v.vg - h, ..v });

        let (id_dp, qg_dp) = (self.black_box_eval)(TerminalVoltages { vd: v.vd + h, ..v });
        let (id_dm, qg_dm) = (self.black_box_eval)(TerminalVoltages { vd: v.vd - h, ..v });

        let (id_bp, _) = (self.black_box_eval)(TerminalVoltages { vb: v.vb + h, ..v });
        let (id_bm, _) = (self.black_box_eval)(TerminalVoltages { vb: v.vb - h, ..v });

        let jac = DeviceJacobian {
            id: id0,
            gm: (id_gp - id_gm) * inv_2h,
            gds: (id_dp - id_dm) * inv_2h,
            gmb: (id_bp - id_bm) * inv_2h,
            qg: qg0,
            cgg: (qg_gp - qg_gm) * inv_2h,
            cgd: -(qg_dp - qg_dm) * inv_2h,
        };

        if !jac.is_finite() {
            return Err(JacobianError::SingularOrNonFiniteJacobian(
                "NaN/Inf при численном дифференцировании".to_string(),
            ));
        }
        Ok(jac)
    }
}

// ============================================================================
// 2. AnalyticC_Model (Built-in C / BSIM4) — C²-непрерывные аналитические формулы
// ============================================================================

/// Аналитическая компактная модель с C²-гладкой сшивкой подпорогового и надпорогового режимов.
#[derive(Debug, Clone)]
pub struct AnalyticCModel {
    pub vth0: f64,
    pub beta: f64,
    pub lambda: f64,
    pub gamma_body: f64,
    pub phi_f: f64,
    pub subthreshold_slope_v: f64,
    pub cox_total: f64,
}

impl Default for AnalyticCModel {
    fn default() -> Self {
        Self {
            vth0: 0.35,
            beta: 2.5e-3,
            lambda: 0.04,
            gamma_body: 0.25,
            phi_f: 0.35,
            subthreshold_slope_v: 0.035,
            cox_total: 1.5e-14,
        }
    }
}

impl AnalyticCModel {
    /// C²-гладкая функция мягкого максимума (softplus): s * ln(1 + exp(x / s)).
    #[inline]
    fn smooth_vov(x: f64, s: f64) -> (f64, f64) {
        let z = (x / s).clamp(-40.0, 40.0);
        let exp_z = z.exp();
        let vov = if z > 20.0 { x } else { s * exp_z.ln_1p() };
        let dvov_dx = exp_z / (1.0 + exp_z); // сигмоида (гладкая производная)
        (vov, dvov_dx)
    }
}

impl JacobianProvider for AnalyticCModel {
    fn kind(&self) -> JacobianStrategyKind {
        JacobianStrategyKind::AnalyticCModel
    }

    fn evaluate(
        &self,
        v: TerminalVoltages,
        _dt: f64,
        _purpose: EvaluationPurpose,
    ) -> Result<DeviceJacobian, JacobianError> {
        let vsb = (v.vs - v.vb).max(-0.5);
        let root_term = (2.0 * self.phi_f + vsb).max(0.05).sqrt();
        let root_0 = (2.0 * self.phi_f).sqrt();
        let vth = self.vth0 + self.gamma_body * (root_term - root_0);
        let dvth_dvb = -self.gamma_body / (2.0 * root_term);

        let vgs = v.vgs();
        let vds = v.vds();
        let (vov, dvov_dvg) = Self::smooth_vov(vgs - vth, self.subthreshold_slope_v);

        // Гладкое насыщение канала через tanh: Vds_eff = Vdsat * tanh(Vds / Vdsat)
        let vdsat = (vov + 0.05).max(0.05);
        let ratio = (vds / vdsat).clamp(-20.0, 20.0);
        let th = ratio.tanh();
        let sech2 = 1.0 - th * th;
        let vds_eff = vdsat * th;

        let dvds_eff_dvd = sech2;
        let dvds_eff_dvov = th - ratio * sech2;

        let clm = 1.0 + self.lambda * vds;
        let core = vov * vds_eff - 0.5 * vds_eff * vds_eff;
        let id = self.beta * core * clm;

        let did_dvov = self.beta * clm * (vds_eff + (vov - vds_eff) * dvds_eff_dvov);
        let gm = did_dvov * dvov_dvg;
        let gds = self.beta * (clm * (vov - vds_eff) * dvds_eff_dvd + self.lambda * core);
        let gmb = -gm * dvth_dvb;

        // C²-непрерывный заряд затвора через tanh
        let arg = ((vgs - vth - 0.5 * vds_eff) / 0.1).clamp(-20.0, 20.0);
        let t_q = arg.tanh();
        let dt_q = 1.0 - t_q * t_q;
        let qg = 0.5 * self.cox_total * (vgs - vth + 0.1 * t_q);
        let cgg = 0.5 * self.cox_total * (1.0 + dt_q);
        let cgd = 0.25 * self.cox_total * dt_q * dvds_eff_dvd;

        let jac = DeviceJacobian {
            id,
            gm,
            gds,
            gmb,
            qg,
            cgg,
            cgd,
        };

        if !jac.is_finite() {
            return Err(JacobianError::SingularOrNonFiniteJacobian(
                "Нечисловое значение в аналитической модели".to_string(),
            ));
        }
        Ok(jac)
    }
}

// ============================================================================
// 3. AutomaticDifferentiation (OpenVAF / AD) — дуальные числа + ddt()/idt()
// ============================================================================

/// Дуальное число `val + eps * der` для forward-mode автоматического дифференцирования.
#[derive(Debug, Clone, Copy)]
pub struct Dual {
    pub val: f64,
    pub der: f64,
}

impl Dual {
    #[inline]
    pub fn var(val: f64) -> Self {
        Self { val, der: 1.0 }
    }

    #[inline]
    pub fn constant(val: f64) -> Self {
        Self { val, der: 0.0 }
    }

    #[inline]
    pub fn add(self, rhs: Self) -> Self {
        Self {
            val: self.val + rhs.val,
            der: self.der + rhs.der,
        }
    }

    #[inline]
    pub fn sub(self, rhs: Self) -> Self {
        Self {
            val: self.val - rhs.val,
            der: self.der - rhs.der,
        }
    }

    #[inline]
    pub fn mul(self, rhs: Self) -> Self {
        Self {
            val: self.val * rhs.val,
            der: self.der * rhs.val + self.val * rhs.der,
        }
    }

    #[inline]
    pub fn div(self, rhs: Self) -> Self {
        Self {
            val: self.val / rhs.val,
            der: (self.der * rhs.val - self.val * rhs.der) / (rhs.val * rhs.val),
        }
    }

    #[inline]
    pub fn tanh(self) -> Self {
        let c = self.val.clamp(-20.0, 20.0);
        let t = c.tanh();
        Self {
            val: t,
            der: (1.0 - t * t) * self.der,
        }
    }

    #[inline]
    pub fn exp(self) -> Self {
        let e = self.val.clamp(-40.0, 40.0).exp();
        Self {
            val: e,
            der: e * self.der,
        }
    }
}

/// Verilog-A AD-модель для мемристора / мемтранзистора с внутренним состоянием `w(t)`.
#[derive(Debug, Clone)]
pub struct AutomaticDifferentiationModel {
    pub ron: f64,
    pub roff: f64,
    pub mobility_v: f64,
    pub thickness_m: f64,
    pub w_prev: f64,
    pub q_prev: f64,
    pub c_par: f64,
}

impl AutomaticDifferentiationModel {
    fn eval_branch_dual(&self, vg: Dual, vd: Dual, vs: Dual, dt: f64) -> (Dual, Dual) {
        let v_dev = vd.sub(vs);
        let v_gate_mod = Dual::constant(1.0).add(vg.mul(Dual::constant(0.25)).tanh().mul(Dual::constant(0.3)));

        // Неявный шаг интегрирования внутреннего состояния w(t) через idt()/ddt()
        let ron = Dual::constant(self.ron);
        let roff = Dual::constant(self.roff);
        let w0 = Dual::constant(self.w_prev);
        let r_approx = roff.add(ron.sub(roff).mul(w0));
        let i_est = v_dev.div(r_approx);

        let k_drift = Dual::constant(self.mobility_v * self.ron / (self.thickness_m * self.thickness_m));
        let window = Dual::constant((1.0 - (2.0 * self.w_prev - 1.0).powi(2)).max(0.05));
        let dw_dt = k_drift.mul(i_est).mul(window).mul(v_gate_mod);

        let w_new_raw = w0.add(dw_dt.mul(Dual::constant(dt.max(1e-12))));
        // Гладкий ограничитель [0, 1] через сигмоиду
        let sig = Dual::constant(1.0).div(
            Dual::constant(1.0).add(Dual::constant(6.0).mul(Dual::constant(0.5).sub(w_new_raw)).exp()),
        );

        let r_eff = roff.add(ron.sub(roff).mul(sig));
        let i_static = v_dev.div(r_eff);

        // Динамический ток смещения ddt(Q) = (Q(t) - Q(t-dt)) / dt
        let qg = Dual::constant(self.c_par).mul(vg.sub(vs));
        let i_disp = qg.sub(Dual::constant(self.q_prev)).div(Dual::constant(dt.max(1e-12)));
        let i_total = i_static.add(i_disp);

        (i_total, qg)
    }
}

impl JacobianProvider for AutomaticDifferentiationModel {
    fn kind(&self) -> JacobianStrategyKind {
        JacobianStrategyKind::AutomaticDifferentiation
    }

    fn evaluate(
        &self,
        v: TerminalVoltages,
        dt: f64,
        _purpose: EvaluationPurpose,
    ) -> Result<DeviceJacobian, JacobianError> {
        let (id_vg, qg_vg) = self.eval_branch_dual(
            Dual::var(v.vg),
            Dual::constant(v.vd),
            Dual::constant(v.vs),
            dt,
        );
        let (id_vd, qg_vd) = self.eval_branch_dual(
            Dual::constant(v.vg),
            Dual::var(v.vd),
            Dual::constant(v.vs),
            dt,
        );

        let jac = DeviceJacobian {
            id: id_vd.val,
            gm: id_vg.der,
            gds: id_vd.der,
            gmb: 0.0,
            qg: qg_vg.val,
            cgg: qg_vg.der,
            cgd: -qg_vd.der,
        };

        if !jac.is_finite() {
            return Err(JacobianError::SingularOrNonFiniteJacobian(
                "Ошибка AD в Verilog-A модели".to_string(),
            ));
        }
        Ok(jac)
    }
}

// ============================================================================
// 4. SymbolicAndBytecode (behavioral B-source) — НФТ.3.1 & НФТ.3.2
// ============================================================================

/// Инструкции стекового байт-кода, скомпилированного однократно при setup (НФТ.3.2).
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum BytecodeOp {
    PushConst(f64),
    LoadVgs,
    LoadVds,
    Add,
    Sub,
    Mul,
    Tanh,
    Softplus,
}

/// Скомпилированный поведенческий B-source для TFET / HEMT / CAD-структур.
#[derive(Debug, Clone)]
pub struct SymbolicAndBytecodeModel {
    ops: Vec<BytecodeOp>,
    clip_limit: f64,
    cox_eff: f64,
}

impl SymbolicAndBytecodeModel {
    /// Однократная компиляция символьного описания при setup.
    pub fn compile(ops: Vec<BytecodeOp>, clip_limit: f64, cox_eff: f64) -> Result<Self, JacobianError> {
        if ops.is_empty() {
            return Err(JacobianError::BytecodeCompileError(
                "Пустой поток инструкций B-source".to_string(),
            ));
        }
        Ok(Self {
            ops,
            clip_limit: clip_limit.abs().max(1.0),
            cox_eff,
        })
    }

    /// Предустановленный байт-код для поведенческой ВАХ TFET/HEMT: I0 * softplus(k1*Vgs - k2) * tanh(k3*Vds).
    pub fn compile_behavioral_fet(i_scale: f64, k_vg: f64, v_shift: f64, k_vd: f64, cox_eff: f64) -> Self {
        Self {
            ops: vec![
                BytecodeOp::PushConst(i_scale),
                BytecodeOp::LoadVgs,
                BytecodeOp::PushConst(k_vg),
                BytecodeOp::Mul,
                BytecodeOp::PushConst(v_shift),
                BytecodeOp::Sub,
                BytecodeOp::Softplus,
                BytecodeOp::Mul,
                BytecodeOp::LoadVds,
                BytecodeOp::PushConst(k_vd),
                BytecodeOp::Mul,
                BytecodeOp::Tanh,
                BytecodeOp::Mul,
            ],
            clip_limit: 50.0,
            cox_eff,
        }
    }

    /// Soft-clipping (НФТ.3.1): ограничивает аргумент в [-L, L] гладко без NaN/Inf.
    #[inline]
    fn soft_clip(&self, x: Dual) -> Dual {
        let l = self.clip_limit;
        let scaled = Dual {
            val: x.val / l,
            der: x.der / l,
        };
        let t = scaled.tanh();
        Dual {
            val: t.val * l,
            der: t.der * l,
        }
    }

    fn execute_dual(&self, vgs: Dual, vds: Dual) -> Result<Dual, JacobianError> {
        let mut stack: Vec<Dual> = Vec::with_capacity(self.ops.len());
        for &op in &self.ops {
            match op {
                BytecodeOp::PushConst(c) => stack.push(Dual::constant(c)),
                BytecodeOp::LoadVgs => stack.push(self.soft_clip(vgs)),
                BytecodeOp::LoadVds => stack.push(self.soft_clip(vds)),
                BytecodeOp::Add => {
                    let b = stack.pop().ok_or_else(|| JacobianError::BytecodeCompileError("Stack underflow".into()))?;
                    let a = stack.pop().ok_or_else(|| JacobianError::BytecodeCompileError("Stack underflow".into()))?;
                    stack.push(self.soft_clip(a.add(b)));
                }
                BytecodeOp::Sub => {
                    let b = stack.pop().ok_or_else(|| JacobianError::BytecodeCompileError("Stack underflow".into()))?;
                    let a = stack.pop().ok_or_else(|| JacobianError::BytecodeCompileError("Stack underflow".into()))?;
                    stack.push(self.soft_clip(a.sub(b)));
                }
                BytecodeOp::Mul => {
                    let b = stack.pop().ok_or_else(|| JacobianError::BytecodeCompileError("Stack underflow".into()))?;
                    let a = stack.pop().ok_or_else(|| JacobianError::BytecodeCompileError("Stack underflow".into()))?;
                    stack.push(self.soft_clip(a.mul(b)));
                }
                BytecodeOp::Tanh => {
                    let a = stack.pop().ok_or_else(|| JacobianError::BytecodeCompileError("Stack underflow".into()))?;
                    stack.push(a.tanh());
                }
                BytecodeOp::Softplus => {
                    let a = stack.pop().ok_or_else(|| JacobianError::BytecodeCompileError("Stack underflow".into()))?;
                    let c = a.val.clamp(-30.0, 30.0);
                    let e = c.exp();
                    let val = if c > 20.0 { c } else { e.ln_1p() };
                    let der = (e / (1.0 + e)) * a.der;
                    stack.push(Dual { val, der });
                }
            }
        }
        stack
            .pop()
            .ok_or_else(|| JacobianError::BytecodeCompileError("Пустой результат стека".into()))
    }
}

impl JacobianProvider for SymbolicAndBytecodeModel {
    fn kind(&self) -> JacobianStrategyKind {
        JacobianStrategyKind::SymbolicAndBytecode
    }

    fn evaluate(
        &self,
        v: TerminalVoltages,
        _dt: f64,
        _purpose: EvaluationPurpose,
    ) -> Result<DeviceJacobian, JacobianError> {
        let id_vg = self.execute_dual(Dual::var(v.vgs()), Dual::constant(v.vds()))?;
        let id_vd = self.execute_dual(Dual::constant(v.vgs()), Dual::var(v.vds()))?;

        let jac = DeviceJacobian {
            id: id_vg.val,
            gm: id_vg.der,
            gds: id_vd.der,
            gmb: 0.0,
            qg: self.cox_eff * v.vgs(),
            cgg: self.cox_eff,
            cgd: 0.0,
        };

        if !jac.is_finite() {
            return Err(JacobianError::SingularOrNonFiniteJacobian(
                "Нечисловой результат байт-кода B-source".to_string(),
            ));
        }
        Ok(jac)
    }
}

// ============================================================================
// 5. NeuralAutodiff (GNN / U-Net backprop) с gate-проверкой (НФТ.5.2)
// ============================================================================

/// Результат gate-проверки монотонности и C²-гладкости нейросетевого суррогата.
#[derive(Debug, Clone, PartialEq)]
pub struct SmoothnessGateReport {
    pub passed: bool,
    pub monotonic_violations: usize,
    pub max_second_derivative_jump: f64,
}

/// Нейросетевой backprop-поставщик якобиана с обязательным gate против эффекта Рунге.
pub struct NeuralAutodiffSurrogate<F>
where
    F: Fn(TerminalVoltages) -> DeviceJacobian,
{
    pub backprop_fn: F,
    pub max_c2_jump_threshold: f64,
    pub gate_verified: bool,
}

impl<F> NeuralAutodiffSurrogate<F>
where
    F: Fn(TerminalVoltages) -> DeviceJacobian,
{
    pub fn new(backprop_fn: F, max_c2_jump_threshold: f64) -> Self {
        Self {
            backprop_fn,
            max_c2_jump_threshold,
            gate_verified: false,
        }
    }

    /// Проверяет сетку смещений на монотонность (gm >= 0, gds >= 0) и отсутствие
    /// высокочастотных осцилляций второй производной (эффект Рунге, НФТ.5.2).
    pub fn run_smoothness_gate(&mut self, vg_grid: &[f64], vd_fixed: f64) -> Result<SmoothnessGateReport, JacobianError> {
        if vg_grid.len() < 4 {
            return Err(JacobianError::SmoothnessGateRejected {
                reason: "Недостаточно точек для проверки C²-гладкости".to_string(),
                max_oscillation_score: f64::INFINITY,
            });
        }

        let mut samples = Vec::with_capacity(vg_grid.len());
        for &vg in vg_grid {
            let j = (self.backprop_fn)(TerminalVoltages::new(vg, vd_fixed, 0.0, 0.0));
            if !j.is_finite() {
                return Err(JacobianError::SmoothnessGateRejected {
                    reason: "Обнаружен NaN/Inf на выходе суррогата".to_string(),
                    max_oscillation_score: f64::INFINITY,
                });
            }
            samples.push((vg, j));
        }

        let mut monotonic_violations = 0usize;
        let mut d2_values = Vec::with_capacity(samples.len() - 1);

        for i in 0..samples.len() {
            if samples[i].1.gm < -1e-12 || samples[i].1.gds < -1e-12 {
                monotonic_violations += 1;
            }
            if i + 1 < samples.len() {
                let dv = (samples[i + 1].0 - samples[i].0).abs().max(1e-9);
                let d2 = (samples[i + 1].1.gm - samples[i].1.gm) / dv;
                d2_values.push(d2);
            }
        }

        let mut max_c2_jump = 0.0_f64;
        for w in d2_values.windows(2) {
            let jump = (w[1] - w[0]).abs();
            if jump > max_c2_jump {
                max_c2_jump = jump;
            }
        }

        let passed = monotonic_violations == 0 && max_c2_jump <= self.max_c2_jump_threshold;
        self.gate_verified = passed;

        if !passed {
            return Err(JacobianError::SmoothnessGateRejected {
                reason: format!(
                    "Нарушений монотонности: {}, скачок d²Id/dVg²: {:.4e} > {:.4e}",
                    monotonic_violations, max_c2_jump, self.max_c2_jump_threshold
                ),
                max_oscillation_score: max_c2_jump,
            });
        }

        Ok(SmoothnessGateReport {
            passed,
            monotonic_violations,
            max_second_derivative_jump: max_c2_jump,
        })
    }
}

impl<F> JacobianProvider for NeuralAutodiffSurrogate<F>
where
    F: Fn(TerminalVoltages) -> DeviceJacobian,
{
    fn kind(&self) -> JacobianStrategyKind {
        JacobianStrategyKind::NeuralAutodiff
    }

    fn evaluate(
        &self,
        v: TerminalVoltages,
        _dt: f64,
        purpose: EvaluationPurpose,
    ) -> Result<DeviceJacobian, JacobianError> {
        if purpose == EvaluationPurpose::CircuitNewtonLinearization && !self.gate_verified {
            return Err(JacobianError::SmoothnessGateRejected {
                reason: "Использование NeuralAutodiff в шаге Ньютона без пройденного gate запрещено".to_string(),
                max_oscillation_score: f64::INFINITY,
            });
        }

        let jac = (self.backprop_fn)(v);
        if !jac.is_finite() {
            return Err(JacobianError::SingularOrNonFiniteJacobian(
                "Нечисловой якобиан нейросетевого суррогата".to_string(),
            ));
        }
        Ok(jac)
    }
}

// ============================================================================
// 6. Consumer: Newton–Raphson (global DAE matrix solver)
// ============================================================================

/// Результат решения нелинейного узла методом Ньютона–Рафсона.
#[derive(Debug, Clone, PartialEq)]
pub struct NewtonSolveResult {
    pub vd_solved: f64,
    pub id_solved: f64,
    pub iterations: usize,
    pub final_residual: f64,
    pub strategy_used: JacobianStrategyKind,
}

/// Потребитель якобиана: решает уравнение KCL для каскада с нагрузочным резистором R_load
/// и питанием V_dd: `f(Vd) = Id(Vg, Vd) - (Vdd - Vd) / R_load = 0`.
pub struct NewtonRaphsonDaeConsumer {
    pub max_iterations: usize,
    pub tolerance_a: f64,
    pub min_conductance: f64,
}

impl Default for NewtonRaphsonDaeConsumer {
    fn default() -> Self {
        Self {
            max_iterations: 50,
            tolerance_a: 1e-11,
            min_conductance: 1e-15,
        }
    }
}

impl NewtonRaphsonDaeConsumer {
    pub fn solve_load_line<P: JacobianProvider>(
        &self,
        provider: &P,
        vg: f64,
        vdd: f64,
        r_load: f64,
        dt: f64,
    ) -> Result<NewtonSolveResult, JacobianError> {
        let mut vd = 0.5 * vdd;
        let g_load = 1.0 / r_load.max(1e-6);

        for iter in 0..self.max_iterations {
            let v = TerminalVoltages::new(vg, vd, 0.0, 0.0);
            let jac = provider.evaluate(v, dt, EvaluationPurpose::CircuitNewtonLinearization)?;

            let residual = jac.id - (vdd - vd) * g_load;
            if residual.abs() <= self.tolerance_a {
                return Ok(NewtonSolveResult {
                    vd_solved: vd,
                    id_solved: jac.id,
                    iterations: iter + 1,
                    final_residual: residual.abs(),
                    strategy_used: provider.kind(),
                });
            }

            // Глобальная матрица Якоби DAE: J_dae = g_ds + G_load
            let j_dae = jac.gds + g_load;
            if !j_dae.is_finite() || j_dae.abs() < self.min_conductance {
                return Err(JacobianError::SingularOrNonFiniteJacobian(format!(
                    "J_dae={:.3e} на итерации {} (потеря гладкости модели)",
                    j_dae, iter
                )));
            }

            let step = (residual / j_dae).clamp(-0.25, 0.25);
            vd = (vd - step).clamp(-0.2, vdd + 0.2);
        }

        let final_jac = provider.evaluate(
            TerminalVoltages::new(vg, vd, 0.0, 0.0),
            dt,
            EvaluationPurpose::CircuitNewtonLinearization,
        )?;
        let final_res = (final_jac.id - (vdd - vd) * g_load).abs();

        Err(JacobianError::NewtonNotConverged {
            iterations: self.max_iterations,
            residual: final_res,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn numerical_diff_allowed_for_metrics_but_blocked_for_newton() {
        let analytic = AnalyticCModel::default();
        let num = NumericalDifference::new(1e-5, |v| {
            let j = analytic
                .evaluate(v, 1e-9, EvaluationPurpose::DiagnosticMetrics)
                .unwrap();
            (j.id, j.qg)
        });

        let v = TerminalVoltages::new(0.8, 0.5, 0.0, 0.0);
        let diag = num
            .evaluate(v, 1e-9, EvaluationPurpose::DiagnosticMetrics)
            .unwrap();
        let exact = analytic
            .evaluate(v, 1e-9, EvaluationPurpose::DiagnosticMetrics)
            .unwrap();

        assert!((diag.gm - exact.gm).abs() / exact.gm < 1e-6);
        assert_eq!(
            num.evaluate(v, 1e-9, EvaluationPurpose::CircuitNewtonLinearization),
            Err(JacobianError::NumericalDiffForbiddenForNewton)
        );
    }

    #[test]
    fn newton_converges_with_analytic_ad_and_bytecode() {
        let solver = NewtonRaphsonDaeConsumer::default();

        let analytic = AnalyticCModel::default();
        let res_ana = solver.solve_load_line(&analytic, 0.9, 1.2, 1_000.0, 1e-9).unwrap();
        assert!(res_ana.final_residual <= 1e-11);

        let ad = AutomaticDifferentiationModel {
            ron: 100.0,
            roff: 10_000.0,
            mobility_v: 1e-14,
            thickness_m: 10e-9,
            w_prev: 0.3,
            q_prev: 0.0,
            c_par: 1e-15,
        };
        let res_ad = solver.solve_load_line(&ad, 0.8, 1.0, 2_000.0, 1e-6).unwrap();
        assert!(res_ad.final_residual <= 1e-11);

        let sym = SymbolicAndBytecodeModel::compile_behavioral_fet(1e-3, 4.0, 1.2, 3.0, 1e-14);
        let res_sym = solver.solve_load_line(&sym, 0.8, 1.2, 1_500.0, 1e-9).unwrap();
        assert!(res_sym.final_residual <= 1e-11);
    }

    #[test]
    fn neural_gate_rejects_runge_oscillations() {
        let mut oscillatory_nn = NeuralAutodiffSurrogate::new(
            |v| {
                // Имитация эффекта Рунге: осциллирующая производная sin(40 * Vg)
                let osc = (40.0 * v.vg).sin();
                DeviceJacobian {
                    id: 1e-3 * v.vg.max(0.0) + 1e-4 * osc,
                    gm: 1e-3 + 4e-3 * (40.0 * v.vg).cos(),
                    gds: 1e-4,
                    gmb: 0.0,
                    qg: 1e-14 * v.vg,
                    cgg: 1e-14,
                    cgd: 0.0,
                }
            },
            0.05,
        );

        let grid: Vec<f64> = (0..25).map(|i| i as f64 * 0.05).collect();
        let gate_res = oscillatory_nn.run_smoothness_gate(&grid, 0.5);
        assert!(matches!(gate_res, Err(JacobianError::SmoothnessGateRejected { .. })));

        let solver = NewtonRaphsonDaeConsumer::default();
        assert!(matches!(
            solver.solve_load_line(&oscillatory_nn, 0.8, 1.2, 1_000.0, 1e-9),
            Err(JacobianError::SmoothnessGateRejected { .. })
        ));
    }
}
