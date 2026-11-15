//! Алгоритм A-3: Reduced-order PDE (1D Poisson + gradual-channel)
//! Генерация характеристик Id–Vg, Id–Vd и Cgg–Vg для MOS-семейств приборов.

use std::collections::BTreeMap;

/// Поддерживаемые семейства приборов для reduced-order PDE
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MosDeviceFamily {
    PlanarMosfet,
    FdSoi,
    Finfet,
    GaaNanosheet,
    GaaNanowire,
}

/// Прочие семейства, для которых reduced-order PDE явно запрещён
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DeviceType {
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

impl DeviceType {
    pub fn as_mos_family(self) -> Result<MosDeviceFamily, ReducedPdeError> {
        match self {
            DeviceType::PlanarMosfet => Ok(MosDeviceFamily::PlanarMosfet),
            DeviceType::FdSoi => Ok(MosDeviceFamily::FdSoi),
            DeviceType::Finfet => Ok(MosDeviceFamily::Finfet),
            DeviceType::GaaNanosheet => Ok(MosDeviceFamily::GaaNanosheet),
            DeviceType::GaaNanowire => Ok(MosDeviceFamily::GaaNanowire),
            _ => Err(ReducedPdeError::UnsupportedDevice(
                "need different carrier/transport equations".to_string(),
            )),
        }
    }
}

/// Входные физико-геометрические параметры прибора
#[derive(Debug, Clone)]
pub struct DeviceParameters {
    pub device_type: DeviceType,
    /// Относительная диэлектрическая проницаемость диэлектрика (например, 3.9 для SiO2)
    pub epsilon_ox: f64,
    /// Относительная диэлектрическая проницаемость канала (например, 11.7 для Si)
    pub epsilon_semi: f64,
    /// Подвижность электронов, см²/(В·с)
    pub electron_mobility_cm2_vs: f64,
    /// Ширина запрещённой зоны, эВ
    pub bandgap_ev: f64,
    /// Концентрация акцепторов в теле (p-тип), см⁻³
    pub doping_cm3: f64,
    /// Эквивалентная толщина подзатворного оксида (EOT), нм
    pub tox_nm: f64,
    /// Толщина тела / канала для 1D сетки Пуассона, нм
    pub body_thickness_nm: f64,
    /// Эффективная ширина канала W_eff, мкм
    pub effective_width_um: f64,
    /// Длина затвора L, мкм
    pub length_um: f64,
    /// Температура, К
    pub temperature_k: f64,
    /// Напряжение плоских зон V_fb, В
    pub v_fb: f64,
}

impl DeviceParameters {
    pub fn validate(&self) -> Result<(), ReducedPdeError> {
        if self.tox_nm <= 0.0
            || self.body_thickness_nm <= 0.0
            || self.effective_width_um <= 0.0
            || self.length_um <= 0.0
            || self.electron_mobility_cm2_vs <= 0.0
            || self.doping_cm3 <= 0.0
            || self.temperature_k <= 0.0
            || self.epsilon_ox <= 0.0
            || self.epsilon_semi <= 0.0
        {
            return Err(ReducedPdeError::InvalidGridOrBounds(
                "Все физические и геометрические размеры должны быть строго положительными"
                    .to_string(),
            ));
        }
        Ok(())
    }
}

/// Конфигурация сеток и диапазонов развёртки (A-3)
#[derive(Debug, Clone)]
pub struct ReducedPdeConfig {
    pub vg_min_v: f64,
    pub vg_max_v: f64,
    pub vd_linear_v: f64,
    pub vd_sat_v: f64,
    pub vd_min_v: f64,
    pub vd_max_v: f64,
    pub gate_points: usize,
    pub drain_points: usize,
    pub channel_points: usize,
    pub poisson_nodes: usize,
    pub output_curve_count: usize,
    pub use_gpu: bool,
}

impl Default for ReducedPdeConfig {
    fn default() -> Self {
        Self {
            vg_min_v: -0.5,
            vg_max_v: 1.2,
            vd_linear_v: 0.05,
            vd_sat_v: 0.8,
            vd_min_v: 0.0,
            vd_max_v: 0.8,
            gate_points: 31,
            drain_points: 25,
            channel_points: 11,
            poisson_nodes: 61,
            output_curve_count: 4,
            use_gpu: false,
        }
    }
}

impl ReducedPdeConfig {
    pub fn validate(&self) -> Result<(), ReducedPdeError> {
        if self.gate_points < 5 {
            return Err(ReducedPdeError::InvalidGridOrBounds(
                "gate_points must be >= 5".to_string(),
            ));
        }
        if self.drain_points < 3 || self.channel_points < 3 {
            return Err(ReducedPdeError::InvalidGridOrBounds(
                "drain_points and channel_points must be >= 3".to_string(),
            ));
        }
        if self.poisson_nodes < 11 {
            return Err(ReducedPdeError::InvalidGridOrBounds(
                "poisson_nodes must be >= 11".to_string(),
            ));
        }
        if self.vg_max_v <= self.vg_min_v || self.vd_max_v <= self.vd_min_v {
            return Err(ReducedPdeError::InvalidGridOrBounds(
                "Некорректные границы развёртки Vg или Vd".to_string(),
            ));
        }
        Ok(())
    }
}

/// Абстракция батчевого 1D решателя Пуассона (CPU / GPU FP64)
pub trait BatchPoisson1DSolver {
    fn backend_name(&self) -> &'static str;

    /// Решает батч 1D нелинейных уравнений Пуассона для пар (vg_eff, v_ch).
    /// Возвращает кортеж векторов (psi_s [В], q_inv [Кл/м²], q_g [Кл/м²]).
    fn solve_batch(
        &self,
        device: &DeviceParameters,
        vg_eff_batch: &[f64],
        v_ch_batch: &[f64],
        poisson_nodes: usize,
    ) -> Result<(Vec<f64>, Vec<f64>, Vec<f64>), ReducedPdeError>;
}

/// Реализация батчевого 1D решателя Пуассона на CPU (метод Ньютона + алгоритм Томаса)
#[derive(Debug, Default, Clone, Copy)]
pub struct CpuBatchPoissonSolver;

impl BatchPoisson1DSolver for CpuBatchPoissonSolver {
    fn backend_name(&self) -> &'static str {
        "cpu_numpy_thomas_fp64"
    }

    fn solve_batch(
        &self,
        device: &DeviceParameters,
        vg_eff_batch: &[f64],
        v_ch_batch: &[f64],
        poisson_nodes: usize,
    ) -> Result<(Vec<f64>, Vec<f64>, Vec<f64>), ReducedPdeError> {
        let batch_len = vg_eff_batch.len();
        if batch_len != v_ch_batch.len() {
            return Err(ReducedPdeError::SolverFailure(
                "Длины батчей vg_eff и v_ch не совпадают".to_string(),
            ));
        }

        let mut psi_s_out = Vec::with_capacity(batch_len);
        let mut q_inv_out = Vec::with_capacity(batch_len);
        let mut q_g_out = Vec::with_capacity(batch_len);

        for i in 0..batch_len {
            let (psi_s, q_inv, q_g) = solve_single_1d_poisson_newton(
                device,
                vg_eff_batch[i],
                v_ch_batch[i],
                poisson_nodes,
            )?;
            psi_s_out.push(psi_s);
            q_inv_out.push(q_inv);
            q_g_out.push(q_g);
        }

        Ok((psi_s_out, q_inv_out, q_g_out))
    }
}

/// Решает 1D нелинейное уравнение Пуассона по толщине тела методом Ньютона + прогонка Томаса.
/// Статистика носителей — Больцман с ограничением показателя экспоненты в диапазоне [-30.0, 50.0].
fn solve_single_1d_poisson_newton(
    device: &DeviceParameters,
    vg_eff: f64,
    v_ch: f64,
    nodes: usize,
) -> Result<(f64, f64, f64), ReducedPdeError> {
    const Q_E: f64 = 1.602_176_634e-19;
    const K_B: f64 = 1.380_649e-23;
    const EPS_0: f64 = 8.854_187_812_8e-12;

    let vt = (K_B * device.temperature_k) / Q_E;
    let eps_semi = EPS_0 * device.epsilon_semi;
    let eps_ox = EPS_0 * device.epsilon_ox;
    let tox_m = device.tox_nm * 1e-9;
    let t_body_m = device.body_thickness_nm * 1e-9;
    let dz = t_body_m / ((nodes - 1) as f64);

    // Ёмкость оксида на единицу площади [Ф/м²]
    let c_ox = eps_ox / tox_m;

    // Концентрация легирования в системе СИ [м⁻³]
    let na_m3 = device.doping_cm3 * 1e6;
    // Собственная концентрация носителей ni(T) [м⁻³]
    let nc_nv_eff = 2.8e25_f64 * (device.temperature_k / 300.0).powf(1.5);
    let ni_m3 = nc_nv_eff * (-device.bandgap_ev / (2.0 * vt)).exp();
    let ni_m3 = ni_m3.max(1.0);

    // Потенциал Ферми объёма phi_F
    let phi_f = vt * (na_m3 / ni_m3).max(1.0).ln();
    let n0_m3 = (ni_m3 * ni_m3) / na_m3;
    let p0_m3 = na_m3;

    // Начальное приближение потенциала по узлам z ∈ [0 .. t_body]
    let mut psi = vec![0.0_f64; nodes];

    let max_newton_iters = 60;
    let tol = 1e-9;

    // Трёхдиагональные массивы для метода прогонки Томаса
    let mut a = vec![0.0_f64; nodes];
    let mut b = vec![0.0_f64; nodes];
    let mut c = vec![0.0_f64; nodes];
    let mut rhs = vec![0.0_f64; nodes];

    for _iter in 0..max_newton_iters {
        // Узел 0 (z = 0): граница оксид / полупроводник (условие непрерывности индукции)
        // eps_semi * (psi[0] - psi[1]) / dz + c_ox * (psi[0] - (vg_eff - device.v_fb)) = 0.5 * dz * rho[0]
        {
            let exp_n = ((psi[0] - v_ch) / vt).clamp(-30.0, 50.0);
            let exp_p = (-psi[0] / vt).clamp(-30.0, 50.0);

            let n_val = n0_m3 * exp_n.exp();
            let p_val = p0_m3 * exp_p.exp();
            let rho = Q_E * (p_val - n_val - na_m3);
            let drho_dpsi = -Q_E * (p_val + n_val) / vt;

            let f0 = eps_semi * (psi[0] - psi[1]) / dz
                + c_ox * (psi[0] - (vg_eff - device.v_fb))
                - 0.5 * dz * rho;

            a[0] = 0.0;
            b[0] = eps_semi / dz + c_ox - 0.5 * dz * drho_dpsi;
            c[0] = -eps_semi / dz;
            rhs[0] = -f0;
        }

        // Внутренние узлы 1 .. nodes - 2
        let inv_dz2 = 1.0 / (dz * dz);
        for j in 1..(nodes - 1) {
            let exp_n = ((psi[j] - v_ch) / vt).clamp(-30.0, 50.0);
            let exp_p = (-psi[j] / vt).clamp(-30.0, 50.0);

            let n_val = n0_m3 * exp_n.exp();
            let p_val = p0_m3 * exp_p.exp();
            let rho = Q_E * (p_val - n_val - na_m3);
            let drho_dpsi = -Q_E * (p_val + n_val) / vt;

            let fj = eps_semi * (psi[j - 1] - 2.0 * psi[j] + psi[j + 1]) * inv_dz2 + rho;

            a[j] = eps_semi * inv_dz2;
            b[j] = -2.0 * eps_semi * inv_dz2 + drho_dpsi;
            c[j] = eps_semi * inv_dz2;
            rhs[j] = -fj;
        }

        // Узел nodes - 1: граница нейтрального тела (Дирихле: psi = 0)
        {
            let last = nodes - 1;
            a[last] = 0.0;
            b[last] = 1.0;
            c[last] = 0.0;
            rhs[last] = -psi[last];
        }

        let delta = thomas_tridiagonal_solve(&a, &b, &c, &rhs)?;

        let mut max_step = 0.0_f64;
        for j in 0..nodes {
            // Демпфирование шага Ньютона для устойчивости при сильной инверсии
            let damped = delta[j].clamp(-5.0 * vt, 5.0 * vt);
            psi[j] += damped;
            max_step = max_step.max(damped.abs());
        }

        if max_step < tol {
            break;
        }
    }

    // Поверхностный потенциал psi_s = psi[0]
    let psi_s = psi[0];

    // Интегрирование инверсионного заряда Q_inv [Кл/м²] методом трапеций по толщине канала
    let mut q_inv = 0.0_f64;
    for j in 0..(nodes - 1) {
        let exp_n0 = ((psi[j] - v_ch) / vt).clamp(-30.0, 50.0);
        let exp_n1 = ((psi[j + 1] - v_ch) / vt).clamp(-30.0, 50.0);
        let n_j = n0_m3 * exp_n0.exp();
        let n_next = n0_m3 * exp_n1.exp();
        // Учитываем избыточный электронный заряд сверх равновесного n0
        let excess_0 = (n_j - n0_m3).max(0.0);
        let excess_1 = (n_next - n0_m3).max(0.0);
        q_inv += 0.5 * Q_E * (excess_0 + excess_1) * dz;
    }

    // Заряд затвора Q_g = C_ox * (V_g_eff - V_fb - psi_s) [Кл/м²]
    let q_g = c_ox * (vg_eff - device.v_fb - psi_s);
    let _ = phi_f;

    Ok((psi_s, q_inv, q_g))
}

/// Трёхдиагональная прогонка (алгоритм Томаса) для системы a[i]*x[i-1] + b[i]*x[i] + c[i]*x[i+1] = d[i]
pub fn thomas_tridiagonal_solve(
    a: &[f64],
    b: &[f64],
    c: &[f64],
    d: &[f64],
) -> Result<Vec<f64>, ReducedPdeError> {
    let n = b.len();
    if n == 0 || a.len() != n || c.len() != n || d.len() != n {
        return Err(ReducedPdeError::SolverFailure(
            "Некорректные размерности трёхдиагональной матрицы".to_string(),
        ));
    }

    let mut c_prime = vec![0.0_f64; n];
    let mut d_prime = vec![0.0_f64; n];
    let mut x = vec![0.0_f64; n];

    if b[0].abs() < 1e-30 {
        return Err(ReducedPdeError::SolverFailure(
            "Нулевой ведущий элемент в алгоритме Томаса (i=0)".to_string(),
        ));
    }

    c_prime[0] = c[0] / b[0];
    d_prime[0] = d[0] / b[0];

    for i in 1..n {
        let denom = b[i] - a[i] * c_prime[i - 1];
        if denom.abs() < 1e-30 {
            return Err(ReducedPdeError::SolverFailure(
                "Вырожденная трёхдиагональная матрица в алгоритме Томаса".to_string(),
            ));
        }
        c_prime[i] = if i < n - 1 { c[i] / denom } else { 0.0 };
        d_prime[i] = (d[i] - a[i] * d_prime[i - 1]) / denom;
    }

    x[n - 1] = d_prime[n - 1];
    for i in (0..(n - 1)).rev() {
        x[i] = d_prime[i] - c_prime[i] * x[i + 1];
    }

    Ok(x)
}

/// Сгенерированная ВАХ / ВФХ кривая
#[derive(Debug, Clone)]
pub struct GeneratedCurve {
    pub spec_id: String,
    pub label: String,
    pub x_name: String,
    pub y_name: String,
    pub bias_label: String,
    pub x: Vec<f64>,
    pub y: Vec<f64>,
}

/// Итоговый результат reduced-order PDE моделирования (A-3)
#[derive(Debug, Clone)]
pub struct PdeCharacteristicResult {
    pub curves: Vec<GeneratedCurve>,
    pub backend: String,
    pub diagnostics: BTreeMap<String, f64>,
    pub model_description: String,
}

#[derive(Debug, Clone)]
pub enum ReducedPdeError {
    UnsupportedDevice(String),
    InvalidGridOrBounds(String),
    SolverFailure(String),
}

impl std::fmt::Display for ReducedPdeError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ReducedPdeError::UnsupportedDevice(msg) => write!(f, "Unsupported device: {msg}"),
            ReducedPdeError::InvalidGridOrBounds(msg) => write!(f, "Invalid config: {msg}"),
            ReducedPdeError::SolverFailure(msg) => write!(f, "Solver failure: {msg}"),
        }
    }
}

impl std::error::Error for ReducedPdeError {}

fn linspace(start: f64, end: f64, count: usize) -> Vec<f64> {
    if count <= 1 {
        return vec![start];
    }
    let step = (end - start) / ((count - 1) as f64);
    (0..count).map(|i| start + (i as f64) * step).collect()
}

/// Главная функция алгоритма A-3: батчевый 1D Poisson + модель плавного канала (GCA)
pub fn solve_pde_characteristics(
    device: &DeviceParameters,
    config: &ReducedPdeConfig,
    solver: &dyn BatchPoisson1DSolver,
) -> Result<PdeCharacteristicResult, ReducedPdeError> {
    // 1. Валидация семейства и сеток
    let _family = device.device_type.as_mos_family()?;
    device.validate()?;
    config.validate()?;

    let vg_grid = linspace(config.vg_min_v, config.vg_max_v, config.gate_points);
    let vd_grid = linspace(config.vd_min_v, config.vd_max_v, config.drain_points);
    let xi_grid = linspace(0.0, 1.0, config.channel_points);

    // 4 значения Vg для выходных характеристик Id–Vd: linspace(0.65 * delta, 1.0 * delta, 4)
    let vg_span = config.vg_max_v - config.vg_min_v;
    let vg_out_levels: Vec<f64> = linspace(0.65, 1.0, config.output_curve_count)
        .into_iter()
        .map(|frac| config.vg_min_v + frac * vg_span)
        .collect();

    // 2. Формирование единого батча запросов (Vg_eff, V_ch)
    let mut vg_eff_batch = Vec::new();
    let mut v_ch_batch = Vec::new();

    // Блок A: transfer_linear (Vg, Vd = vd_linear_v)
    for &vg in &vg_grid {
        for &xi in &xi_grid {
            let v_ch = config.vd_linear_v * xi;
            vg_eff_batch.push(vg - v_ch);
            v_ch_batch.push(v_ch);
        }
    }

    // Блок B: transfer_saturation (Vg, Vd = vd_sat_v)
    for &vg in &vg_grid {
        for &xi in &xi_grid {
            let v_ch = config.vd_sat_v * xi;
            vg_eff_batch.push(vg - v_ch);
            v_ch_batch.push(v_ch);
        }
    }

    // Блок C: cv (Vg, Vd = 0.0)
    for &vg in &vg_grid {
        vg_eff_batch.push(vg);
        v_ch_batch.push(0.0);
    }

    // Блок D: output_0..3 (Vd-развёртка для каждого уровня Vg)
    for &vg_level in &vg_out_levels {
        for &vd in &vd_grid {
            for &xi in &xi_grid {
                let v_ch = vd * xi;
                vg_eff_batch.push(vg_level - v_ch);
                v_ch_batch.push(v_ch);
            }
        }
    }

    // 3. Единый батчевый вызов 1D решателя Пуассона
    let (_psi_s, q_inv_batch, q_g_batch) = solver.solve_batch(
        device,
        &vg_eff_batch,
        &v_ch_batch,
        config.poisson_nodes,
    )?;

    // 4. Интегрирование токов по плавному каналу (gradual channel approximation)
    let mu_m2_vs = device.electron_mobility_cm2_vs * 1e-4;
    let w_m = device.effective_width_um * 1e-6;
    let l_m = device.length_um * 1e-6;
    let prefactor = (mu_m2_vs * w_m) / l_m;
    let n_ch = config.channel_points;

    let integrate_channel_current = |q_slice: &[f64], vd: f64| -> f64 {
        if vd.abs() < 1e-15 {
            return 0.0;
        }
        let dv_step = vd / ((n_ch - 1) as f64);
        let mut integral = 0.0_f64;
        for k in 0..(n_ch - 1) {
            integral += 0.5 * (q_slice[k] + q_slice[k + 1]) * dv_step;
        }
        prefactor * integral
    };

    let mut cursor = 0usize;

    // Извлекаем I_linear(Vg)
    let mut id_linear = Vec::with_capacity(config.gate_points);
    for _ in 0..config.gate_points {
        let slice = &q_inv_batch[cursor..cursor + n_ch];
        id_linear.push(integrate_channel_current(slice, config.vd_linear_v));
        cursor += n_ch;
    }

    // Извлекаем I_sat(Vg)
    let mut id_sat = Vec::with_capacity(config.gate_points);
    for _ in 0..config.gate_points {
        let slice = &q_inv_batch[cursor..cursor + n_ch];
        id_sat.push(integrate_channel_current(slice, config.vd_sat_v));
        cursor += n_ch;
    }

    // Извлекаем Q_g(Vg) при Vd = 0 и вычисляем C_gg(Vg) = max(dQ_g/dVg * W_eff * L * 1e-12, 0)
    let qg_cv = &q_g_batch[cursor..cursor + config.gate_points];
    cursor += config.gate_points;

    let area_m2 = device.effective_width_um * device.length_um * 1e-12;
    let mut cgg_vg = vec![0.0_f64; config.gate_points];
    for i in 0..config.gate_points {
        let dqg_dvg = if i == 0 {
            (qg_cv[1] - qg_cv[0]) / (vg_grid[1] - vg_grid[0])
        } else if i == config.gate_points - 1 {
            (qg_cv[i] - qg_cv[i - 1]) / (vg_grid[i] - vg_grid[i - 1])
        } else {
            (qg_cv[i + 1] - qg_cv[i - 1]) / (vg_grid[i + 1] - vg_grid[i - 1])
        };
        cgg_vg[i] = (dqg_dvg * area_m2).max(0.0);
    }

    // Извлекаем выходные кривые Id–Vd
    let mut output_curves = Vec::with_capacity(vg_out_levels.len());
    for (idx, &vg_val) in vg_out_levels.iter().enumerate() {
        let mut id_out = Vec::with_capacity(config.drain_points);
        for &vd in &vd_grid {
            let slice = &q_inv_batch[cursor..cursor + n_ch];
            id_out.push(integrate_channel_current(slice, vd));
            cursor += n_ch;
        }
        output_curves.push((idx, vg_val, id_out));
    }

    // 5. Упаковка GeneratedCurve[] и диагностики
    let mut curves = Vec::with_capacity(3 + output_curves.len());
    curves.push(GeneratedCurve {
        spec_id: "idvg".to_string(),
        label: format!("Id–Vg (linear, Vd = {:.2} V)", config.vd_linear_v),
        x_name: "Vg".to_string(),
        y_name: "Id".to_string(),
        bias_label: format!("Vd={:.3}V", config.vd_linear_v),
        x: vg_grid.clone(),
        y: id_linear,
    });
    curves.push(GeneratedCurve {
        spec_id: "idvg".to_string(),
        label: format!("Id–Vg (saturation, Vd = {:.2} V)", config.vd_sat_v),
        x_name: "Vg".to_string(),
        y_name: "Id".to_string(),
        bias_label: format!("Vd={:.3}V", config.vd_sat_v),
        x: vg_grid.clone(),
        y: id_sat,
    });
    curves.push(GeneratedCurve {
        spec_id: "cv_cgg".to_string(),
        label: "Cgg–Vg (quasi-static, Vd = 0 V)".to_string(),
        x_name: "Vg".to_string(),
        y_name: "Cgg".to_string(),
        bias_label: "Vd=0.000V".to_string(),
        x: vg_grid,
        y: cgg_vg,
    });

    for (idx, vg_val, id_out) in output_curves {
        curves.push(GeneratedCurve {
            spec_id: "idvd".to_string(),
            label: format!("Id–Vd #{idx} (Vg = {vg_val:.2} V)"),
            x_name: "Vd".to_string(),
            y_name: "Id".to_string(),
            bias_label: format!("Vg={vg_val:.3}V"),
            x: vd_grid.clone(),
            y: id_out,
        });
    }

    let mut diagnostics = BTreeMap::new();
    diagnostics.insert("VG_MIN_V".to_string(), config.vg_min_v);
    diagnostics.insert("VG_MAX_V".to_string(), config.vg_max_v);
    diagnostics.insert("VDS_LINEAR_V".to_string(), config.vd_linear_v);
    diagnostics.insert("VDS_SATURATION_V".to_string(), config.vd_sat_v);
    diagnostics.insert("POISSON_NODES".to_string(), config.poisson_nodes as f64);
    diagnostics.insert("GATE_POINTS".to_string(), config.gate_points as f64);
    diagnostics.insert("DRAIN_POINTS".to_string(), config.drain_points as f64);
    diagnostics.insert("MOBILITY_M2_VS".to_string(), mu_m2_vs);
    diagnostics.insert("FLAT_BAND_ASSUMPTION_V".to_string(), device.v_fb);

    let model_description = "1D nonlinear Poisson along body thickness (Boltzmann carriers,         clamped exponent [-30, 50], neutral body boundary) coupled with gradual-channel         approximation (GCA) and quasi-static Cgg = dQg/dVg."
        .to_string();

    Ok(PdeCharacteristicResult {
        curves,
        backend: solver.backend_name().to_string(),
        diagnostics,
        model_description,
    })
}
