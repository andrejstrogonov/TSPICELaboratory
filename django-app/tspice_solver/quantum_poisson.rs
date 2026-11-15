//! Самосогласованный решатель Quantum-Poisson (2D x-z / 3D x-y-z)
//! Реализация алгоритма A-2 для расчёта карты концентрации носителей n(x, y, z) [см⁻³].

use std::f64::consts::PI;

/// Физические константы (СИ)
const Q_E: f64 = 1.602_176_634e-19;      // Заряд электрона [Кл]
const HBAR: f64 = 1.054_571_817e-34;     // Приведённая постоянная Планка [Дж·с]
const M0: f64 = 9.109_383_7015e-31;      // Масса свободного электрона [кг]
const KB: f64 = 1.380_649e-23;           // Постоянная Больцмана [Дж/К]
const EPS0: f64 = 8.854_187_8128e-12;    // Электрическая постоянная [Ф/м]

/// Семейство полупроводникового прибора
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DeviceFamily {
    PlanarMosfet,
    FdSoi,
    FinFet,
    GaaNanosheet,
    GaaNanowire,
    Tfet,
    Hemt,
    Memristor,
    Memtransistor,
    CadCustom,
}

impl DeviceFamily {
    /// Возвращает true для многозатворных структур с дополнительными поверхностями затвора
    pub fn is_multigate(self) -> bool {
        matches!(
            self,
            DeviceFamily::FinFet | DeviceFamily::GaaNanosheet | DeviceFamily::GaaNanowire
        )
    }

    /// Проверяет поддержку алгоритма Quantum-Poisson для данного семейства
    pub fn supports_quantum_poisson(self) -> bool {
        matches!(
            self,
            DeviceFamily::PlanarMosfet
                | DeviceFamily::FdSoi
                | DeviceFamily::FinFet
                | DeviceFamily::GaaNanosheet
                | DeviceFamily::GaaNanowire
                | DeviceFamily::CadCustom
        )
    }
}

/// Режим размерности сетки Quantum-Poisson
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum QuantumGridMode {
    /// Двумерное сечение (x вдоль канала, z по глубине конфайнмента)
    Mode2D { nx: usize, nz: usize },
    /// Трёхмерная область (x вдоль канала, y по ширине, z по глубине)
    Mode3D { nx: usize, ny: usize, nz: usize },
}

impl QuantumGridMode {
    pub fn shape(&self) -> (usize, usize, usize) {
        match *self {
            QuantumGridMode::Mode2D { nx, nz } => (nx, 1, nz),
            QuantumGridMode::Mode3D { nx, ny, nz } => (nx, ny, nz),
        }
    }

    pub fn is_3d(&self) -> bool {
        matches!(self, QuantumGridMode::Mode3D { .. })
    }
}

/// Вычислительный бэкенд
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ComputeBackend {
    CudaFp64,
    CpuFallback,
}

/// Входные параметры прибора для расчёта
#[derive(Debug, Clone)]
pub struct QuantumDeviceParams {
    pub family: DeviceFamily,
    pub length_um: f64,
    pub width_um: f64,
    pub body_thickness_nm: f64,
    pub tox_nm: f64,
    pub epsilon_r_body: f64,
    pub epsilon_r_ox: f64,
    /// Эффективная масса конфайнмента m*_z / m0
    pub m_eff_conf: f64,
    /// Эффективная масса плотности состояний m*_dos / m0
    pub m_eff_dos: f64,
    /// Фактор вырождения долин (например, 2 для нижних долин (100) Si)
    pub valley_degeneracy: f64,
    /// Концентрация доноров N_D [см⁻³]
    pub nd_cm3: f64,
    /// Концентрация акцепторов N_A [см⁻³]
    pub na_cm3: f64,
    /// Собственная концентрация носителей n_i [см⁻³]
    pub ni_cm3: f64,
    /// Температура [К]
    pub temperature_k: f64,
    /// Работа выхода / плоские зоны V_fb [В]
    pub flatband_voltage_v: f64,
}

impl Default for QuantumDeviceParams {
    fn default() -> Self {
        Self {
            family: DeviceFamily::FinFet,
            length_um: 0.020,
            width_um: 0.010,
            body_thickness_nm: 8.0,
            tox_nm: 1.0,
            epsilon_r_body: 11.7,
            epsilon_r_ox: 3.9,
            m_eff_conf: 0.916,
            m_eff_dos: 0.19,
            valley_degeneracy: 2.0,
            nd_cm3: 1e15,
            na_cm3: 1e17,
            ni_cm3: 1.0e10,
            temperature_k: 300.0,
            flatband_voltage_v: 0.0,
        }
    }
}

/// Напряжения на контактах и параметры запуска
#[derive(Debug, Clone)]
pub struct QuantumPoissonConfig {
    pub mode: QuantumGridMode,
    pub v_gate: f64,
    pub v_drain: f64,
    pub v_source: f64,
    pub use_gpu: bool,
    pub max_gummel_iters: usize,
    pub gummel_tol_v: f64,
    pub cg_max_iters: usize,
    pub cg_tol: f64,
    pub damping: f64,
    pub num_subbands: usize,
}

impl Default for QuantumPoissonConfig {
    fn default() -> Self {
        Self {
            mode: QuantumGridMode::Mode2D { nx: 32, nz: 24 },
            v_gate: 0.8,
            v_drain: 0.05,
            v_source: 0.0,
            use_gpu: true,
            max_gummel_iters: 60,
            gummel_tol_v: 1e-6,
            cg_max_iters: 10_000,
            cg_tol: 1e-10,
            damping: 0.35,
            num_subbands: 6,
        }
    }
}

/// Ортогональные срезы 2D/3D поля для визуализации в GUI
#[derive(Debug, Clone)]
pub struct FieldSlices {
    /// Срез X-Z (вдоль канала и по толщине) при среднем Y
    pub xz_slice: Vec<Vec<f64>>,
    /// Срез X-Y (в плоскости канала) при среднем Z (только для 3D)
    pub xy_slice: Option<Vec<Vec<f64>>>,
    /// Срез Y-Z (поперечное сечение канала) при среднем X (только для 3D)
    pub yz_slice: Option<Vec<Vec<f64>>>,
}

/// Запись об одной итерации Ньютона/Гуммеля
#[derive(Debug, Clone)]
pub struct GummelIterationRecord {
    pub iter: usize,
    pub delta_psi_rel: f64,
    pub cg_iters: usize,
    pub cg_residual: f64,
}

/// Итоговое решение задачи Quantum-Poisson
#[derive(Debug, Clone)]
pub struct QuantumFieldSolution {
    /// Концентрация электронов n(x, y, z) в [см⁻³]
    pub values: Vec<f64>,
    /// Потенциал ψ(x, y, z) в [В]
    pub potential_v: Vec<f64>,
    /// Размерность тензора (nx, ny, nz)
    pub shape: (usize, usize, usize),
    /// Фактически использованный бэкенд
    pub backend: ComputeBackend,
    /// Причина переключения на CPU при изоляции CUDA OOM (NFR-REL-03)
    pub fallback_reason: Option<String>,
    /// Сошёлся ли внешний цикл за лимит итераций
    pub converged: bool,
    /// Журнал сходимости
    pub convergence_history: Vec<GummelIterationRecord>,
    /// Подготовленные сечения для отрисовки
    pub slices: FieldSlices,
}

/// Ошибки решателя Quantum-Poisson
#[derive(Debug, Clone)]
pub enum QuantumSolverError {
    UnsupportedDeviceFamily(DeviceFamily),
    InvalidGridDimensions(String),
    InvalidPhysicalParameter(String),
    CudaOutOfMemory { required_bytes: usize, available_bytes: usize },
    LinearSolverDiverged { residual: f64 },
}

impl std::fmt::Display for QuantumSolverError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            QuantumSolverError::UnsupportedDeviceFamily(fam) => {
                write!(f, "Семейство {:?} не поддерживает квантовый расчёт носителей", fam)
            }
            QuantumSolverError::InvalidGridDimensions(msg) => {
                write!(f, "Некорректные размеры сетки: {}", msg)
            }
            QuantumSolverError::InvalidPhysicalParameter(msg) => {
                write!(f, "Некорректный физический параметр: {}", msg)
            }
            QuantumSolverError::CudaOutOfMemory {
                required_bytes,
                available_bytes,
            } => write!(
                f,
                "CUDA OutOfMemory: требуется {} байт, доступно {} байт",
                required_bytes, available_bytes
            ),
            QuantumSolverError::LinearSolverDiverged { residual } => {
                write!(f, "Линейный решатель CG не сошёлся (невязка {:.3e})", residual)
            }
        }
    }
}

impl std::error::Error for QuantumSolverError {}

/// Интерфейс GPU-ускорителя (PyTorch FP64 matrix-free stencil CG)
pub trait QuantumGpuAccelerator {
    /// Возвращает свободную VRAM в байтах или None, если CUDA недоступна
    fn available_vram_bytes(&self) -> Option<usize>;

    /// Принудительная очистка кэша видеопамяти при OOM (NFR-REL-03)
    fn empty_vram_cache(&self);

    /// Решает линейную систему 5/7-точечного шаблона Пуассона на GPU в FP64
    fn solve_poisson_stencil_fp64(
        &self,
        grid: &PoissonStencilSystem,
        rhs: &[f64],
        initial_psi: &[f64],
        tol: f64,
        max_iters: usize,
    ) -> Result<(Vec<f64>, usize, f64), QuantumSolverError>;
}

/// Сетка, граничные условия и коэффициенты 5/7-точечного шаблона
#[derive(Debug, Clone)]
pub struct PoissonStencilSystem {
    pub nx: usize,
    pub ny: usize,
    pub nz: usize,
    pub dx: f64,
    pub dy: f64,
    pub dz: f64,
    /// Относительная диэлектрическая проницаемость ε(x, y, z) в узлах
    pub eps_r: Vec<f64>,
    /// Маска узлов Дирихле (true = фиксированный потенциал на контакте/нейтральном теле)
    pub dirichlet_mask: Vec<bool>,
    /// Значения Дирихле ψ_D
    pub dirichlet_values: Vec<f64>,
    /// Маска узлов Robin на границе затвора/оксида
    pub robin_mask: Vec<bool>,
    /// Коэффициент Robin: C_ox / (ε_0 * ε_body * dz)
    pub robin_alpha: f64,
    /// Эффективный потенциал затвора V_g - V_fb
    pub v_gate_eff: f64,
}

impl PoissonStencilSystem {
    #[inline]
    pub fn idx(&self, ix: usize, iy: usize, iz: usize) -> usize {
        (ix * self.ny + iy) * self.nz + iz
    }

    pub fn total_nodes(&self) -> usize {
        self.nx * self.ny * self.nz
    }

    /// Оценка объёма памяти для matrix-free CG в FP64
    pub fn estimate_memory_bytes(&self) -> usize {
        // 8 векторов состояния/направлений/ε в FP64 + буферы
        self.total_nodes() * 8 * std::mem::size_of::<f64>() * 2
    }

    /// Построение сетки, контактов Дирихле, доп. границ FinFET/GAA и условия Robin
    pub fn build(params: &QuantumDeviceParams, cfg: &QuantumPoissonConfig) -> Self {
        let (nx, ny, nz) = cfg.mode.shape();
        let total = nx * ny * nz;

        let lx = params.length_um * 1e-6;
        let ly = params.width_um * 1e-6;
        let lz = params.body_thickness_nm * 1e-9;

        let dx = lx / (nx.saturating_sub(1).max(1) as f64);
        let dy = if ny > 1 {
            ly / ((ny - 1) as f64)
        } else {
            ly.max(1e-9)
        };
        let dz = lz / (nz.saturating_sub(1).max(1) as f64);

        let mut eps_r = vec![params.epsilon_r_body; total];
        let mut dirichlet_mask = vec![false; total];
        let mut dirichlet_values = vec![0.0; total];
        let mut robin_mask = vec![false; total];

        // Встроенный потенциал источника/стока относительно нейтрального тела
        let vt = KB * params.temperature_k / Q_E;
        let doping_net = (params.na_cm3 - params.nd_cm3).abs().max(params.ni_cm3);
        let phi_bi = vt * (doping_net / params.ni_cm3).ln();
        let v_src_tot = params.v_source + phi_bi;
        let v_drn_tot = cfg.v_drain + phi_bi;
        let v_gate_eff = cfg.v_gate - params.flatband_voltage_v;

        let tox_m = params.tox_nm * 1e-9;
        let c_ox = EPS0 * params.epsilon_r_ox / tox_m;
        let robin_alpha = c_ox / (EPS0 * params.epsilon_r_body * dz);

        let is_multigate = params.family.is_multigate();
        let is_gaa = matches!(
            params.family,
            DeviceFamily::GaaNanosheet | DeviceFamily::GaaNanowire
        );

        // Область затвора занимает центральную часть канала по X (от 20% до 80%)
        let gate_x_start = nx / 5;
        let gate_x_end = (4 * nx) / 5;

        for ix in 0..nx {
            let frac_x = ix as f64 / (nx.saturating_sub(1).max(1) as f64);
            let psi_lin = v_src_tot + frac_x * (v_drn_tot - v_src_tot);

            for iy in 0..ny {
                for iz in 0..nz {
                    let id = (ix * ny + iy) * nz + iz;
                    dirichlet_values[id] = psi_lin;

                    // Контакты истока и стока (Дирихле)
                    if ix == 0 {
                        dirichlet_mask[id] = true;
                        dirichlet_values[id] = v_src_tot;
                        continue;
                    }
                    if ix == nx - 1 {
                        dirichlet_mask[id] = true;
                        dirichlet_values[id] = v_drn_tot;
                        continue;
                    }

                    // Разбиение слоёв по глубине (тело / BOX для FD-SOI)
                    if params.family == DeviceFamily::FdSoi && iz > (3 * nz) / 4 {
                        eps_r[id] = params.epsilon_r_ox;
                    }

                    let under_gate = ix >= gate_x_start && ix <= gate_x_end;

                    // Верхний затвор (iz == 0): Robin-условие C_ox * (V_g - V) на границе оксида
                    if iz == 0 && under_gate {
                        robin_mask[id] = true;
                    }

                    // Дополнительные граничные поверхности затвора для FinFET и GAA
                    if is_multigate && under_gate {
                        // 2 боковые стенки ребра/листа/провода (iy == 0 и iy == ny - 1)
                        if ny > 1 && (iy == 0 || iy == ny - 1) {
                            robin_mask[id] = true;
                        }
                        // Нижняя поверхность затвора для GAA (круговой охват листа/провода)
                        if is_gaa && iz == nz - 1 {
                            robin_mask[id] = true;
                        }
                    }

                    // Нейтральное тело как граничное условие для ψ на нижней границе
                    if iz == nz - 1 && !robin_mask[id] {
                        dirichlet_mask[id] = true;
                        dirichlet_values[id] = 0.0;
                    }
                }
            }
        }

        Self {
            nx,
            ny,
            nz,
            dx,
            dy,
            dz,
            eps_r,
            dirichlet_mask,
            dirichlet_values,
            robin_mask,
            robin_alpha,
            v_gate_eff,
        }
    }

    /// Matrix-free применение 5/7-точечного оператора -∇·(ε_r ∇ψ) + Robin-члена
    pub fn apply_stencil(&self, x: &[f64], out: &mut [f64]) {
        let inv_dx2 = 1.0 / (self.dx * self.dx);
        let inv_dy2 = if self.ny > 1 {
            1.0 / (self.dy * self.dy)
        } else {
            0.0
        };
        let inv_dz2 = 1.0 / (self.dz * self.dz);

        for ix in 0..self.nx {
            for iy in 0..self.ny {
                for iz in 0..self.nz {
                    let id = self.idx(ix, iy, iz);
                    if self.dirichlet_mask[id] {
                        out[id] = x[id];
                        continue;
                    }

                    let eps_c = self.eps_r[id];
                    let xc = x[id];
                    let mut diag = 0.0;
                    let mut off = 0.0;

                    // Ось X
                    if ix > 0 {
                        let id_m = self.idx(ix - 1, iy, iz);
                        let eps_m = 0.5 * (eps_c + self.eps_r[id_m]);
                        diag += eps_m * inv_dx2;
                        if !self.dirichlet_mask[id_m] {
                            off -= eps_m * inv_dx2 * x[id_m];
                        }
                    }
                    if ix + 1 < self.nx {
                        let id_p = self.idx(ix + 1, iy, iz);
                        let eps_p = 0.5 * (eps_c + self.eps_r[id_p]);
                        diag += eps_p * inv_dx2;
                        if !self.dirichlet_mask[id_p] {
                            off -= eps_p * inv_dx2 * x[id_p];
                        }
                    }

                    // Ось Y (только для 3D)
                    if self.ny > 1 {
                        if iy > 0 {
                            let id_m = self.idx(ix, iy - 1, iz);
                            let eps_m = 0.5 * (eps_c + self.eps_r[id_m]);
                            diag += eps_m * inv_dy2;
                            if !self.dirichlet_mask[id_m] {
                                off -= eps_m * inv_dy2 * x[id_m];
                            }
                        }
                        if iy + 1 < self.ny {
                            let id_p = self.idx(ix, iy + 1, iz);
                            let eps_p = 0.5 * (eps_c + self.eps_r[id_p]);
                            diag += eps_p * inv_dy2;
                            if !self.dirichlet_mask[id_p] {
                                off -= eps_p * inv_dy2 * x[id_p];
                            }
                        }
                    }

                    // Ось Z
                    if iz > 0 {
                        let id_m = self.idx(ix, iy, iz - 1);
                        let eps_m = 0.5 * (eps_c + self.eps_r[id_m]);
                        diag += eps_m * inv_dz2;
                        if !self.dirichlet_mask[id_m] {
                            off -= eps_m * inv_dz2 * x[id_m];
                        }
                    }
                    if iz + 1 < self.nz {
                        let id_p = self.idx(ix, iy, iz + 1);
                        let eps_p = 0.5 * (eps_c + self.eps_r[id_p]);
                        diag += eps_p * inv_dz2;
                        if !self.dirichlet_mask[id_p] {
                            off -= eps_p * inv_dz2 * x[id_p];
                        }
                    }

                    // Условие Robin на границе затвора: ε_r * ∂ψ/∂n + α * ψ = α * V_g_eff
                    if self.robin_mask[id] {
                        diag += eps_c * self.robin_alpha / self.dz;
                    }

                    out[id] = diag * xc + off;
                }
            }
        }
    }

    /// Сборка правой части с учётом заряда ρ и неоднородных условий Дирихле/Robin
    pub fn build_rhs(&self, rho_c_m3: &[f64]) -> Vec<f64> {
        let total = self.total_nodes();
        let mut rhs = vec![0.0; total];

        let inv_dx2 = 1.0 / (self.dx * self.dx);
        let inv_dy2 = if self.ny > 1 {
            1.0 / (self.dy * self.dy)
        } else {
            0.0
        };
        let inv_dz2 = 1.0 / (self.dz * self.dz);

        for ix in 0..self.nx {
            for iy in 0..self.ny {
                for iz in 0..self.nz {
                    let id = self.idx(ix, iy, iz);
                    if self.dirichlet_mask[id] {
                        rhs[id] = self.dirichlet_values[id];
                        continue;
                    }

                    let eps_c = self.eps_r[id];
                    // Уравнение -∇·(ε_r ∇ψ) = ρ / ε_0
                    let mut b = rho_c_m3[id] / EPS0;

                    if self.robin_mask[id] {
                        b += eps_c * self.robin_alpha * self.v_gate_eff / self.dz;
                    }

                    // Перенос известных соседей Дирихле в правую часть
                    if ix > 0 {
                        let id_m = self.idx(ix - 1, iy, iz);
                        if self.dirichlet_mask[id_m] {
                            let eps_m = 0.5 * (eps_c + self.eps_r[id_m]);
                            b += eps_m * inv_dx2 * self.dirichlet_values[id_m];
                        }
                    }
                    if ix + 1 < self.nx {
                        let id_p = self.idx(ix + 1, iy, iz);
                        if self.dirichlet_mask[id_p] {
                            let eps_p = 0.5 * (eps_c + self.eps_r[id_p]);
                            b += eps_p * inv_dx2 * self.dirichlet_values[id_p];
                        }
                    }
                    if self.ny > 1 {
                        if iy > 0 {
                            let id_m = self.idx(ix, iy - 1, iz);
                            if self.dirichlet_mask[id_m] {
                                let eps_m = 0.5 * (eps_c + self.eps_r[id_m]);
                                b += eps_m * inv_dy2 * self.dirichlet_values[id_m];
                            }
                        }
                        if iy + 1 < self.ny {
                            let id_p = self.idx(ix, iy + 1, iz);
                            if self.dirichlet_mask[id_p] {
                                let eps_p = 0.5 * (eps_c + self.eps_r[id_p]);
                                b += eps_p * inv_dy2 * self.dirichlet_values[id_p];
                            }
                        }
                    }
                    if iz > 0 {
                        let id_m = self.idx(ix, iy, iz - 1);
                        if self.dirichlet_mask[id_m] {
                            let eps_m = 0.5 * (eps_c + self.eps_r[id_m]);
                            b += eps_m * inv_dz2 * self.dirichlet_values[id_m];
                        }
                    }
                    if iz + 1 < self.nz {
                        let id_p = self.idx(ix, iy, iz + 1);
                        if self.dirichlet_mask[id_p] {
                            let eps_p = 0.5 * (eps_c + self.eps_r[id_p]);
                            b += eps_p * inv_dz2 * self.dirichlet_values[id_p];
                        }
                    }

                    rhs[id] = b;
                }
            }
        }

        rhs
    }
}

/// CPU fallback: метод сопряжённых градиентов (CG) для matrix-free 5/7-точечного шаблона
fn solve_poisson_cg_cpu(
    system: &PoissonStencilSystem,
    rhs: &[f64],
    initial_psi: &[f64],
    tol: f64,
    max_iters: usize,
) -> Result<(Vec<f64>, usize, f64), QuantumSolverError> {
    let n = system.total_nodes();
    let mut x = initial_psi.to_vec();
    for i in 0..n {
        if system.dirichlet_mask[i] {
            x[i] = system.dirichlet_values[i];
        }
    }

    let mut ax = vec![0.0; n];
    system.apply_stencil(&x, &mut ax);

    let mut r = vec![0.0; n];
    let mut p = vec![0.0; n];
    let mut rs_old = 0.0;
    let mut rhs_norm_sq = 0.0;

    for i in 0..n {
        if system.dirichlet_mask[i] {
            r[i] = 0.0;
            p[i] = 0.0;
        } else {
            r[i] = rhs[i] - ax[i];
            p[i] = r[i];
            rs_old += r[i] * r[i];
            rhs_norm_sq += rhs[i] * rhs[i];
        }
    }

    let target_sq = (tol * tol) * rhs_norm_sq.max(1e-30);
    if rs_old <= target_sq {
        return Ok((x, 0, rs_old.sqrt()));
    }

    let mut ap = vec![0.0; n];
    let mut final_res = rs_old.sqrt();

    for iter in 1..=max_iters {
        system.apply_stencil(&p, &mut ap);

        let mut p_ap = 0.0;
        for i in 0..n {
            if !system.dirichlet_mask[i] {
                p_ap += p[i] * ap[i];
            }
        }

        if p_ap.abs() < 1e-30 {
            break;
        }

        let alpha = rs_old / p_ap;
        let mut rs_new = 0.0;

        for i in 0..n {
            if !system.dirichlet_mask[i] {
                x[i] += alpha * p[i];
                r[i] -= alpha * ap[i];
                rs_new += r[i] * r[i];
            }
        }

        final_res = rs_new.sqrt();
        if rs_new <= target_sq {
            return Ok((x, iter, final_res));
        }

        let beta = rs_new / rs_old;
        for i in 0..n {
            if !system.dirichlet_mask[i] {
                p[i] = r[i] + beta * p[i];
            }
        }
        rs_old = rs_new;
    }

    if final_res.is_nan() || final_res.is_infinite() {
        return Err(QuantumSolverError::LinearSolverDiverged { residual: final_res });
    }

    Ok((x, max_iters, final_res))
}

/// Решение 1D уравнения Шрёдингера по направлению конфайнмента Z (метод Якоби для симметричной 3-диагональной матрицы)
/// Возвращает отсортированные (E_i [Дж], нормированные χ_i(z) [м⁻¹/²]).
fn solve_1d_schrodinger_z(
    potential_v: &[f64],
    dz: f64,
    m_eff_conf: f64,
    num_subbands: usize,
) -> Vec<(f64, Vec<f64>)> {
    let nz = potential_v.len();
    if nz < 3 {
        return Vec::new();
    }

    // Внутренние узлы с жёсткими стенками конфайнмента χ(0) = χ(L_z) = 0
    let m_inner = nz - 2;
    let m_kg = m_eff_conf * M0;
    let t0 = (HBAR * HBAR) / (2.0 * m_kg * dz * dz);

    let mut a = vec![vec![0.0; m_inner]; m_inner];
    let mut v_eig = vec![vec![0.0; m_inner]; m_inner];

    for i in 0..m_inner {
        v_eig[i][i] = 1.0;
        // Потенциальная энергия электрона U(z) = -q * ψ(z)
        let u_pot = -Q_E * potential_v[i + 1];
        a[i][i] = 2.0 * t0 + u_pot;
        if i + 1 < m_inner {
            a[i][i + 1] = -t0;
            a[i + 1][i] = -t0;
        }
    }

    // Итерации вращений Якоби (для типичной размерности nz ~ 16..64 сходится за ~15-25 проходов)
    let max_sweeps = 60;
    for _sweep in 0..max_sweeps {
        let mut max_off = 0.0;
        for p in 0..m_inner {
            for q in (p + 1)..m_inner {
                let apq = a[p][q].abs();
                if apq > max_off {
                    max_off = apq;
                }
            }
        }
        if max_off < 1e-26 {
            break;
        }

        for p in 0..m_inner {
            for q in (p + 1)..m_inner {
                let apq = a[p][q];
                if apq.abs() < 1e-28 {
                    continue;
                }
                let app = a[p][p];
                let aqq = a[q][q];
                let tau = (aqq - app) / (2.0 * apq);
                let t = if tau >= 0.0 {
                    1.0 / (tau + (1.0 + tau * tau).sqrt())
                } else {
                    -1.0 / (-tau + (1.0 + tau * tau).sqrt())
                };
                let c = 1.0 / (1.0 + t * t).sqrt();
                let s = t * c;

                a[p][p] = app - t * apq;
                a[q][q] = aqq + t * apq;
                a[p][q] = 0.0;
                a[q][p] = 0.0;

                for k in 0..m_inner {
                    if k != p && k != q {
                        let akp = a[k][p];
                        let akq = a[k][q];
                        a[k][p] = c * akp - s * akq;
                        a[p][k] = a[k][p];
                        a[k][q] = s * akp + c * akq;
                        a[q][k] = a[k][q];
                    }
                    let vkp = v_eig[k][p];
                    let vkq = v_eig[k][q];
                    v_eig[k][p] = c * vkp - s * vkq;
                    v_eig[k][q] = s * vkp + c * vkq;
                }
            }
        }
    }

    let mut modes: Vec<(f64, Vec<f64>)> = Vec::with_capacity(m_inner);
    for j in 0..m_inner {
        let energy = a[j][j];
        let mut psi_wave = vec![0.0; nz];
        let mut norm_sq = 0.0;
        for i in 0..m_inner {
            psi_wave[i + 1] = v_eig[i][j];
            norm_sq += v_eig[i][j] * v_eig[i][j] * dz;
        }
        let inv_norm = if norm_sq > 0.0 {
            1.0 / norm_sq.sqrt()
        } else {
            0.0
        };
        for val in &mut psi_wave {
            *val *= inv_norm;
        }
        modes.push((energy, psi_wave));
    }

    modes.sort_by(|lhs, rhs| lhs.0.partial_cmp(&rhs.0).unwrap_or(std::cmp::Ordering::Equal));
    modes.truncate(num_subbands.min(m_inner));
    modes
}

/// Заселение 2D-подзон по статистике Ферми–Дирака (не Больцмана):
/// N_i = (g_v * m*_dos * k_B * T / (π * ħ²)) * ln(1 + exp((E_F - E_i) / (k_B * T)))
#[inline]
fn subband_sheet_density_fermi_dirac(
    energy_j: f64,
    fermi_level_j: f64,
    temperature_k: f64,
    m_eff_dos: f64,
    valley_degeneracy: f64,
) -> f64 {
    let kbt = KB * temperature_k;
    let dos_2d = valley_degeneracy * (m_eff_dos * M0) / (PI * HBAR * HBAR);
    let eta = (fermi_level_j - energy_j) / kbt;

    // Численно устойчивый log(1 + exp(eta))
    let log1p_exp = if eta > 40.0 {
        eta
    } else if eta < -40.0 {
        eta.exp()
    } else {
        (1.0 + eta.exp()).ln()
    };

    dos_2d * kbt * log1p_exp
}

/// Основная точка входа алгоритма A-2: самосогласованный Quantum-Poisson (2D x-z / 3D x-y-z)
pub fn solve_quantum_charge_distribution(
    params: &QuantumDeviceParams,
    cfg: &QuantumPoissonConfig,
    gpu: Option<&dyn QuantumGpuAccelerator>,
) -> Result<QuantumFieldSolution, QuantumSolverError> {
    if !params.family.supports_quantum_poisson() {
        return Err(QuantumSolverError::UnsupportedDeviceFamily(params.family));
    }

    let (nx, ny, nz) = cfg.mode.shape();
    if nx < 5 || nz < 5 || (cfg.mode.is_3d() && ny < 3) {
        return Err(QuantumSolverError::InvalidGridDimensions(
            "Требуется nx >= 5, nz >= 5 и ny >= 3 для 3D-режима".to_string(),
        ));
    }
    if params.body_thickness_nm <= 0.0 || params.tox_nm <= 0.0 || params.length_um <= 0.0 {
        return Err(QuantumSolverError::InvalidPhysicalParameter(
            "Толщина тела, Tox и длина канала должны быть строго положительными".to_string(),
        ));
    }

    // 1. Построение геометрии, границ Дирихле/Robin и нулевого приближения ψ
    let system = PoissonStencilSystem::build(params, cfg);
    let total = system.total_nodes();

    let mut psi = system.dirichlet_values.clone();
    let mut n_cm3 = vec![params.ni_cm3; total];
    let mut p_cm3 = vec![params.ni_cm3; total];
    let mut rho_c_m3 = vec![0.0; total];

    let vt = KB * params.temperature_k / Q_E;
    let ni_m3 = params.ni_cm3 * 1e6;
    let nd_m3 = params.nd_cm3 * 1e6;
    let na_m3 = params.na_cm3 * 1e6;

    // Выбор бэкенда с учётом доступности CUDA и оценки VRAM
    let required_vram = system.estimate_memory_bytes();
    let mut active_backend = match (cfg.use_gpu, gpu) {
        (true, Some(accel)) => match accel.available_vram_bytes() {
            Some(free_vram) if free_vram >= required_vram => ComputeBackend::CudaFp64,
            _ => ComputeBackend::CpuFallback,
        },
        _ => ComputeBackend::CpuFallback,
    };
    let mut fallback_reason: Option<String> = None;

    let mut history = Vec::with_capacity(cfg.max_gummel_iters);
    let mut converged = false;

    // 2. Внешний итерационный цикл Ньютона / Гуммеля
    for iter in 1..=cfg.max_gummel_iters {
        // Шаг 1 · Квантовая задача: 1D уравнение Шрёдингера по направлению конфайнмента Z
        // для каждого поперечного сечения (ix, iy)
        for ix in 0..nx {
            let frac_x = ix as f64 / ((nx - 1) as f64);
            // Квазиуровень Ферми электронов вдоль канала от истока к стоку
            let v_fn = params.v_source + frac_x * (cfg.v_drain - params.v_source);
            let fermi_level_j = -Q_E * v_fn;

            for iy in 0..ny {
                let mut psi_col = vec![0.0; nz];
                for iz in 0..nz {
                    psi_col[iz] = psi[system.idx(ix, iy, iz)];
                }

                let subbands = solve_1d_schrodinger_z(
                    &psi_col,
                    system.dz,
                    params.m_eff_conf,
                    cfg.num_subbands,
                );

                // Заселение подзон по Fermi-Dirac и вычисление n(z) = Σ_i N_i |χ_i(z)|²
                let mut n_col_m3 = vec![0.0; nz];
                for (e_i, chi_i) in &subbands {
                    let n_sheet = subband_sheet_density_fermi_dirac(
                        *e_i,
                        fermi_level_j,
                        params.temperature_k,
                        params.m_eff_dos,
                        params.valley_degeneracy,
                    );
                    for iz in 0..nz {
                        n_col_m3[iz] += n_sheet * chi_i[iz] * chi_i[iz];
                    }
                }

                // Шаг 2 · Плотность заряда: p по закону действующих масс, ρ = q(p - n + N_D+ - N_A-)
                for iz in 0..nz {
                    let id = system.idx(ix, iy, iz);
                    let n_val_m3 = n_col_m3[iz].max(1.0);
                    // С учётом локального изгиба зон
                    let p_eq_m3 = ni_m3 * (-psi_col[iz] / vt).clamp(-35.0, 35.0).exp();
                    let p_val_m3 = ((ni_m3 * ni_m3) / n_val_m3).min(p_eq_m3).max(1.0);

                    n_cm3[id] = n_val_m3 * 1e-6;
                    p_cm3[id] = p_val_m3 * 1e-6;
                    rho_c_m3[id] = Q_E * (p_val_m3 - n_val_m3 + nd_m3 - na_m3);
                }
            }
        }

        // Шаг 3 · Сборка правой части для 5/7-точечного шаблона Пуассона
        let rhs = system.build_rhs(&rho_c_m3);

        // Шаг 4 · Линейное решение с изоляцией ошибок CUDA OutOfMemory (NFR-REL-03)
        let (psi_solved, cg_iters, cg_res) = if active_backend == ComputeBackend::CudaFp64 {
            let accel = gpu.expect("GPU accelerator must exist when CudaFp64 is active");
            match accel.solve_poisson_stencil_fp64(&system, &rhs, &psi, cfg.cg_tol, cfg.cg_max_iters)
            {
                Ok(res) => res,
                Err(QuantumSolverError::CudaOutOfMemory {
                    required_bytes,
                    available_bytes,
                }) => {
                    accel.empty_vram_cache();
                    active_backend = ComputeBackend::CpuFallback;
                    fallback_reason = Some(format!(
                        "CUDA OutOfMemory на итерации {} (требуется {} МБ, доступно {} МБ); выполнено переключение на CPU.",
                        iter,
                        required_bytes / (1024 * 1024),
                        available_bytes / (1024 * 1024)
                    ));
                    solve_poisson_cg_cpu(&system, &rhs, &psi, cfg.cg_tol, cfg.cg_max_iters)?
                }
                Err(other) => return Err(other),
            }
        } else {
            solve_poisson_cg_cpu(&system, &rhs, &psi, cfg.cg_tol, cfg.cg_max_iters)?
        };

        // Шаг 5 · Демпфирование обновления и контроль относительной невязки Δψ_rel
        let mut max_diff = 0.0;
        let mut max_norm = 1e-12;
        for i in 0..total {
            let psi_next = (1.0 - cfg.damping) * psi[i] + cfg.damping * psi_solved[i];
            let diff = (psi_next - psi[i]).abs();
            if diff > max_diff {
                max_diff = diff;
            }
            if psi_next.abs() > max_norm {
                max_norm = psi_next.abs();
            }
            psi[i] = psi_next;
        }

        let delta_psi_rel = max_diff / max_norm;
        history.push(GummelIterationRecord {
            iter,
            delta_psi_rel,
            cg_iters,
            cg_residual: cg_res,
        });

        if delta_psi_rel <= cfg.gummel_tol_v {
            converged = true;
            break;
        }
    }

    // Подготовка цветных ортогональных сечений для GUI (2D x-z -> 1 карта, 3D -> 3 среза)
    let slices = extract_orthogonal_slices(&n_cm3, nx, ny, nz, cfg.mode.is_3d());

    Ok(QuantumFieldSolution {
        values: n_cm3,
        potential_v: psi,
        shape: (nx, ny, nz),
        backend: active_backend,
        fallback_reason,
        converged,
        convergence_history: history,
        slices,
    })
}

/// Извлечение 2D-среза X-Z или трёх ортогональных сечений (X-Z, X-Y, Y-Z) для 3D-области
fn extract_orthogonal_slices(
    values: &[f64],
    nx: usize,
    ny: usize,
    nz: usize,
    is_3d: bool,
) -> FieldSlices {
    let mid_x = nx / 2;
    let mid_y = ny / 2;
    let mid_z = nz / 2;

    let mut xz_slice = vec![vec![0.0; nz]; nx];
    for ix in 0..nx {
        for iz in 0..nz {
            xz_slice[ix][iz] = values[(ix * ny + mid_y) * nz + iz];
        }
    }

    if !is_3d {
        return FieldSlices {
            xz_slice,
            xy_slice: None,
            yz_slice: None,
        };
    }

    let mut xy_slice = vec![vec![0.0; ny]; nx];
    for ix in 0..nx {
        for iy in 0..ny {
            xy_slice[ix][iy] = values[(ix * ny + iy) * nz + mid_z];
        }
    }

    let mut yz_slice = vec![vec![0.0; nz]; ny];
    for iy in 0..ny {
        for iz in 0..nz {
            yz_slice[iy][iz] = values[(mid_x * ny + iy) * nz + iz];
        }
    }

    FieldSlices {
        xz_slice,
        xy_slice: Some(xy_slice),
        yz_slice: Some(yz_slice),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_quantum_poisson_2d_finfet_converges() {
        let params = QuantumDeviceParams {
            family: DeviceFamily::FinFet,
            ..Default::default()
        };
        let cfg = QuantumPoissonConfig {
            mode: QuantumGridMode::Mode2D { nx: 16, nz: 14 },
            v_gate: 0.9,
            v_drain: 0.05,
            use_gpu: false,
            max_gummel_iters: 40,
            gummel_tol_v: 1e-4,
            ..Default::default()
        };

        let sol = solve_quantum_charge_distribution(&params, &cfg, None).expect("solver failed");
        assert_eq!(sol.shape, (16, 1, 14));
        assert_eq!(sol.backend, ComputeBackend::CpuFallback);
        assert!(sol.slices.xy_slice.is_none());
        assert!(sol.values.iter().all(|&n| n.is_finite() && n > 0.0));
    }

    #[test]
    fn test_unsupported_family_rejected() {
        let params = QuantumDeviceParams {
            family: DeviceFamily::Memristor,
            ..Default::default()
        };
        let cfg = QuantumPoissonConfig::default();
        let err = solve_quantum_charge_distribution(&params, &cfg, None).unwrap_err();
        assert!(matches!(
            err,
            QuantumSolverError::UnsupportedDeviceFamily(DeviceFamily::Memristor)
        ));
    }
}
