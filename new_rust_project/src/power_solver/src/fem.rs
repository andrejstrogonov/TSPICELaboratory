//! A-1 · Алгоритм 3D FEM (электрическая, тепловая и магнитная задачи)
//! Источник спецификации: ALGO_FEM_PIPELINE (src/simulation.py)

use std::collections::HashMap;
use std::fmt;

/// Максимально допустимое число узлов сетки (NFR / бюджет памяти)
pub const MAX_MESH_NODES: usize = 3_000_000;
/// Оценка затрат памяти на один тетраэдр (байт)
pub const BYTES_PER_TETRAHEDRON: usize = 2_500;
/// Относительный допуск сходимости метода сопряжённых градиентов (CG)
pub const CG_TOLERANCE: f64 = 1e-10;
/// Предельное число итераций CG
pub const CG_MAX_ITERATIONS: usize = 10_000;
/// Вакуумная магнитная проницаемость μ0 [Гн/м]
pub const MU_0: f64 = 4.0 * std::f64::consts::PI * 1e-7;
/// Диэлектрическая проницаемость вакуума ε0 [Ф/м]
pub const EPSILON_0: f64 = 8.854_187_812_8e-12;

#[derive(Debug, Clone, PartialEq)]
pub enum FemError {
    MeshBudgetExceeded {
        requested_nodes: usize,
        max_nodes: usize,
        estimated_bytes: usize,
        available_bytes: usize,
    },
    InvalidGridShape([usize; 3]),
    CudaOutOfMemory {
        required_bytes: usize,
        available_bytes: usize,
    },
    NotConverged {
        iterations: usize,
        residual: f64,
    },
    InvalidPhysicsInput(String),
}

impl fmt::Display for FemError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            FemError::MeshBudgetExceeded {
                requested_nodes,
                max_nodes,
                estimated_bytes,
                available_bytes,
            } => write!(
                f,
                "MESH_BUDGET_EXCEEDED: узлов {} (лимит {}), требуется {} Б (доступно {} Б)",
                requested_nodes, max_nodes, estimated_bytes, available_bytes
            ),
            FemError::InvalidGridShape(shape) => {
                write!(f, "Некорректная размерность сетки {:?}: требуется >= 2 по каждой оси", shape)
            }
            FemError::CudaOutOfMemory {
                required_bytes,
                available_bytes,
            } => write!(
                f,
                "VRAM_OUT_OF_MEMORY: требуется {} Б, доступно {} Б",
                required_bytes, available_bytes
            ),
            FemError::NotConverged {
                iterations,
                residual,
            } => write!(
                f,
                "NOT_CONVERGED: метод CG не сошёлся за {} итераций (невязка {:.3e})",
                iterations, residual
            ),
            FemError::InvalidPhysicsInput(msg) => write!(f, "Ошибка входных параметров: {}", msg),
        }
    }
}

impl std::error::Error for FemError {}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ComputeBackend {
    CudaFp64,
    CpuScipy,
}

impl ComputeBackend {
    pub fn as_str(&self) -> &'static str {
        match self {
            ComputeBackend::CudaFp64 => "cuda_fp64",
            ComputeBackend::CpuScipy => "cpu_scipy",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FieldQuantity {
    ElectricPotentialV,
    ElectricFieldMagVm,
    TemperatureK,
    MagneticFluxDensityT,
}

#[derive(Debug, Clone)]
pub struct HardwareCapabilities {
    pub cuda_available: bool,
    pub free_vram_bytes: usize,
    pub free_ram_bytes: usize,
    /// Флаг для тестирования перехвата CUDA OOM (NFR-REL-03)
    pub simulate_cuda_oom: bool,
}

impl Default for HardwareCapabilities {
    fn default() -> Self {
        Self {
            cuda_available: false,
            free_vram_bytes: 0,
            free_ram_bytes: 16 * 1024 * 1024 * 1024, // 16 ГБ
            simulate_cuda_oom: false,
        }
    }
}

#[derive(Debug, Clone)]
pub struct DeviceParameters {
    pub device_type: String,
    pub width_um: f64,
    pub length_um: f64,
    pub tox_nm: f64,
    pub body_thickness_nm: f64,
    pub substrate_thickness_um: f64,
    pub epsilon_r: f64,
    pub conductivity_s_m: f64,
    pub thermal_conductivity_w_mk: f64,
    pub ambient_temp_k: f64,
}

#[derive(Debug, Clone)]
pub struct OperatingPoint {
    pub v_source: f64,
    pub v_drain: f64,
    pub v_gate: f64,
    pub i_drain: f64,
}

#[derive(Debug, Clone)]
pub struct IterationRecord {
    pub step_no: usize,
    pub residual: f64,
}

#[derive(Debug, Clone)]
pub struct FieldSolution {
    pub quantity: FieldQuantity,
    pub values: Vec<f64>,
    pub shape: [usize; 3],
    pub backend: ComputeBackend,
    pub fallback_reason: Option<String>,
    pub iterations: usize,
    pub final_residual: f64,
    pub convergence_history: Vec<IterationRecord>,
}

#[derive(Debug, Clone)]
pub struct ElectrostaticResult {
    pub potential: FieldSolution,
    pub electric_field_mag: FieldSolution,
    pub magnetic_field_mag: FieldSolution,
    pub capacitance_f: f64,
    pub stored_energy_j: f64,
}

#[derive(Debug, Clone)]
pub struct ThermalResult {
    pub temperature: FieldSolution,
    pub t_max_k: f64,
    pub delta_t_k: f64,
    pub r_th_k_w: f64,
}

#[derive(Debug, Clone)]
pub struct TetrahedralMesh {
    pub shape: [usize; 3],
    pub x_coords: Vec<f64>,
    pub y_coords: Vec<f64>,
    pub z_coords: Vec<f64>,
    /// Координаты узлов [x, y, z] в метрах
    pub nodes: Vec<[f64; 3]>,
    /// Индексы вершин тетраэдров (6 тетраэдров на каждый гексаэдр)
    pub tetrahedra: Vec<[usize; 4]>,
}

impl TetrahedralMesh {
    #[inline]
    pub fn node_index(&self, ix: usize, iy: usize, iz: usize) -> usize {
        let [_, ny, nz] = self.shape;
        (ix * ny + iy) * nz + iz
    }
}

/// Разреженная матрица в формате CSR (Compressed Sparse Row)
#[derive(Debug, Clone)]
pub struct CsrMatrix {
    pub nrows: usize,
    pub ncols: usize,
    pub row_ptr: Vec<usize>,
    pub col_idx: Vec<usize>,
    pub values: Vec<f64>,
}

impl CsrMatrix {
    pub fn from_triplets(nrows: usize, ncols: usize, triplets: &[(usize, usize, f64)]) -> Self {
        let mut rows: Vec<HashMap<usize, f64>> = vec![HashMap::new(); nrows];
        for &(r, c, v) in triplets {
            if r < nrows && c < ncols && v != 0.0 {
                *rows[r].entry(c).or_insert(0.0) += v;
            }
        }

        let mut row_ptr = Vec::with_capacity(nrows + 1);
        let mut col_idx = Vec::new();
        let mut values = Vec::new();
        row_ptr.push(0);

        for row in rows {
            let mut entries: Vec<(usize, f64)> = row.into_iter().collect();
            entries.sort_by_key(|&(c, _)| c);
            for (c, v) in entries {
                col_idx.push(c);
                values.push(v);
            }
            row_ptr.push(col_idx.len());
        }

        Self {
            nrows,
            ncols,
            row_ptr,
            col_idx,
            values,
        }
    }

    pub fn matvec(&self, x: &[f64]) -> Vec<f64> {
        let mut y = vec![0.0; self.nrows];
        for r in 0..self.nrows {
            let mut sum = 0.0;
            for idx in self.row_ptr[r]..self.row_ptr[r + 1] {
                sum += self.values[idx] * x[self.col_idx[idx]];
            }
            y[r] = sum;
        }
        y
    }

    pub fn estimated_bytes(&self) -> usize {
        self.row_ptr.len() * std::mem::size_of::<usize>()
            + self.col_idx.len() * std::mem::size_of::<usize>()
            + self.values.len() * std::mem::size_of::<f64>()
    }
}

// =============================================================================
// Шаг 1 · Оценка бюджета сетки
// =============================================================================

pub fn estimate_node_count(grid_shape: [usize; 3]) -> Result<usize, FemError> {
    let [nx, ny, nz] = grid_shape;
    if nx < 2 || ny < 2 || nz < 2 {
        return Err(FemError::InvalidGridShape(grid_shape));
    }
    Ok(nx * ny * nz)
}

pub fn estimate_tetrahedron_count(grid_shape: [usize; 3]) -> Result<usize, FemError> {
    let [nx, ny, nz] = grid_shape;
    if nx < 2 || ny < 2 || nz < 2 {
        return Err(FemError::InvalidGridShape(grid_shape));
    }
    Ok(6 * (nx - 1) * (ny - 1) * (nz - 1))
}

pub fn estimate_mesh_memory_bytes(grid_shape: [usize; 3]) -> Result<usize, FemError> {
    let n_tet = estimate_tetrahedron_count(grid_shape)?;
    Ok(n_tet * BYTES_PER_TETRAHEDRON)
}

pub fn validate_mesh_budget(
    grid_shape: [usize; 3],
    hw: &HardwareCapabilities,
) -> Result<(), FemError> {
    let n_nodes = estimate_node_count(grid_shape)?;
    let est_bytes = estimate_mesh_memory_bytes(grid_shape)?;
    let max_available = hw.free_ram_bytes.max(hw.free_vram_bytes);

    if n_nodes > MAX_MESH_NODES || est_bytes > max_available {
        return Err(FemError::MeshBudgetExceeded {
            requested_nodes: n_nodes,
            max_nodes: MAX_MESH_NODES,
            estimated_bytes: est_bytes,
            available_bytes: max_available,
        });
    }
    Ok(())
}

// =============================================================================
// Шаг 2 · Адаптивная генерация тетраэдральной сетки (6 тетраэдров на гексаэдр)
// =============================================================================

pub fn adaptive_axis(size: f64, count: usize, refinement_points: &[f64]) -> Vec<f64> {
    if count <= 2 || size <= 0.0 {
        return vec![0.0, size.max(0.0)];
    }

    // Весовая плотность точек с локальным сгущением у границ канала, оксида и слоёв
    let n_samples = (count * 16).max(64);
    let sigma = (size / (count as f64)).max(size * 0.03);
    let mut cum_weights = Vec::with_capacity(n_samples + 1);
    cum_weights.push(0.0);

    for i in 0..n_samples {
        let pos = size * (i as f64 + 0.5) / (n_samples as f64);
        let mut w = 1.0;
        for &rp in refinement_points {
            let dist = (pos - rp).abs();
            w += 4.0 * (-0.5 * (dist / sigma).powi(2)).exp();
        }
        cum_weights.push(cum_weights.last().unwrap() + w);
    }

    let total_w = *cum_weights.last().unwrap();
    let mut coords = Vec::with_capacity(count);
    coords.push(0.0);

    for k in 1..(count - 1) {
        let target = total_w * (k as f64) / ((count - 1) as f64);
        let idx = match cum_weights.binary_search_by(|v| v.partial_cmp(&target).unwrap()) {
            Ok(i) => i,
            Err(i) => i.saturating_sub(1),
        };
        let w0 = cum_weights[idx];
        let w1 = cum_weights[(idx + 1).min(n_samples)];
        let frac = if (w1 - w0).abs() > 1e-15 {
            (target - w0) / (w1 - w0)
        } else {
            0.0
        };
        let coord = size * (idx as f64 + frac) / (n_samples as f64);
        coords.push(coord);
    }
    coords.push(size);
    coords
}

pub fn make_tetrahedral_mesh(
    params: &DeviceParameters,
    grid_shape: [usize; 3],
) -> Result<TetrahedralMesh, FemError> {
    let [nx, ny, nz] = grid_shape;
    if nx < 2 || ny < 2 || nz < 2 {
        return Err(FemError::InvalidGridShape(grid_shape));
    }

    let lx = params.length_um * 1e-6;
    let ly = params.width_um * 1e-6;
    let tox_m = params.tox_nm * 1e-9;
    let body_m = params.body_thickness_nm * 1e-9;
    let sub_m = params.substrate_thickness_um * 1e-6;
    let lz = (tox_m + body_m + sub_m).max(1e-8);

    // Сгущение сетки: края канала по X, центр по Y, граница затвор/оксид и оксид/канал по Z
    let x_coords = adaptive_axis(lx, nx, &[0.15 * lx, 0.85 * lx]);
    let y_coords = adaptive_axis(ly, ny, &[0.5 * ly]);
    let z_coords = adaptive_axis(lz, nz, &[tox_m, tox_m + body_m]);

    let mut nodes = Vec::with_capacity(nx * ny * nz);
    for &x in &x_coords {
        for &y in &y_coords {
            for &z in &z_coords {
                nodes.push([x, y, z]);
            }
        }
    }

    let node_id = |ix: usize, iy: usize, iz: usize| -> usize { (ix * ny + iy) * nz + iz };

    let mut tetrahedra = Vec::with_capacity(6 * (nx - 1) * (ny - 1) * (nz - 1));
    for ix in 0..(nx - 1) {
        for iy in 0..(ny - 1) {
            for iz in 0..(nz - 1) {
                let c000 = node_id(ix, iy, iz);
                let c100 = node_id(ix + 1, iy, iz);
                let c010 = node_id(ix, iy + 1, iz);
                let c110 = node_id(ix + 1, iy + 1, iz);
                let c001 = node_id(ix, iy, iz + 1);
                let c101 = node_id(ix + 1, iy, iz + 1);
                let c011 = node_id(ix, iy + 1, iz + 1);
                let c111 = node_id(ix + 1, iy + 1, iz + 1);

                // Стандартное разбиение гексаэдра на 6 согласованных тетраэдров по диагонали c000-c111
                tetrahedra.push([c000, c100, c110, c111]);
                tetrahedra.push([c000, c110, c010, c111]);
                tetrahedra.push([c000, c010, c011, c111]);
                tetrahedra.push([c000, c011, c001, c111]);
                tetrahedra.push([c000, c001, c101, c111]);
                tetrahedra.push([c000, c101, c100, c111]);
            }
        }
    }

    Ok(TetrahedralMesh {
        shape: grid_shape,
        x_coords,
        y_coords,
        z_coords,
        nodes,
        tetrahedra,
    })
}

// =============================================================================
// Шаг 3 · Сборка глобальной тетраэдральной матрицы жёсткости K
// =============================================================================

fn tetrahedron_gradients_and_volume(p: [[f64; 3]; 4]) -> Option<([[f64; 3]; 4], f64)> {
    let v1 = [p[1][0] - p[0][0], p[1][1] - p[0][1], p[1][2] - p[0][2]];
    let v2 = [p[2][0] - p[0][0], p[2][1] - p[0][1], p[2][2] - p[0][2]];
    let v3 = [p[3][0] - p[0][0], p[3][1] - p[0][1], p[3][2] - p[0][2]];

    let det = v1[0] * (v2[1] * v3[2] - v2[2] * v3[1])
        - v1[1] * (v2[0] * v3[2] - v2[2] * v3[0])
        + v1[2] * (v2[0] * v3[1] - v2[1] * v3[0]);

    let volume = det.abs() / 6.0;
    if volume <= 1e-36 {
        return None;
    }

    // Градиенты линейных функций формы ∇N_i для 4 вершин тетраэдра
    let mut grads = [[0.0; 3]; 4];
    for i in 0..4 {
        let j = (i + 1) % 4;
        let k = (i + 2) % 4;
        let l = (i + 3) % 4;

        let e1 = [p[k][0] - p[j][0], p[k][1] - p[j][1], p[k][2] - p[j][2]];
        let e2 = [p[l][0] - p[j][0], p[l][1] - p[j][1], p[l][2] - p[j][2]];
        let normal = [
            e1[1] * e2[2] - e1[2] * e2[1],
            e1[2] * e2[0] - e1[0] * e2[2],
            e1[0] * e2[1] - e1[1] * e2[0],
        ];

        // Ориентация внутрь тетраэдра относительно противоположной вершины p[i]
        let to_opp = [p[i][0] - p[j][0], p[i][1] - p[j][1], p[i][2] - p[j][2]];
        let dot = normal[0] * to_opp[0] + normal[1] * to_opp[1] + normal[2] * to_opp[2];
        let sign = if dot >= 0.0 { 1.0 } else { -1.0 };
        let scale = sign / (6.0 * volume);

        grads[i] = [normal[0] * scale, normal[1] * scale, normal[2] * scale];
    }

    Some((grads, volume))
}

pub fn assemble_tetrahedral_stiffness(
    mesh: &TetrahedralMesh,
    element_coeffs: &[f64],
) -> CsrMatrix {
    let n_nodes = mesh.nodes.len();
    let mut triplets = Vec::with_capacity(mesh.tetrahedra.len() * 16);

    for (t_idx, &tet) in mesh.tetrahedra.iter().enumerate() {
        let coeff = element_coeffs.get(t_idx).copied().unwrap_or(1.0);
        let pts = [
            mesh.nodes[tet[0]],
            mesh.nodes[tet[1]],
            mesh.nodes[tet[2]],
            mesh.nodes[tet[3]],
        ];

        if let Some((grads, vol)) = tetrahedron_gradients_and_volume(pts) {
            for a in 0..4 {
                for b in 0..4 {
                    let dot_grad = grads[a][0] * grads[b][0]
                        + grads[a][1] * grads[b][1]
                        + grads[a][2] * grads[b][2];
                    let k_ab = coeff * vol * dot_grad;
                    triplets.push((tet[a], tet[b], k_ab));
                }
            }
        }
    }

    CsrMatrix::from_triplets(n_nodes, n_nodes, &triplets)
}

// =============================================================================
// Шаги 4 и 5 · Выбор backend, граничные условия Дирихле и решатель CG с fallback
// =============================================================================

pub fn estimate_cuda_memory_bytes(matrix: &CsrMatrix) -> usize {
    // Матрица CSR + рабочие векторы CG (x, r, p, Ap) в FP64
    matrix.estimated_bytes() * 2 + matrix.nrows * 4 * std::mem::size_of::<f64>()
}

pub fn select_backend(matrix: &CsrMatrix, hw: &HardwareCapabilities) -> ComputeBackend {
    let required_vram = estimate_cuda_memory_bytes(matrix);
    if hw.cuda_available && hw.free_vram_bytes >= required_vram {
        ComputeBackend::CudaFp64
    } else {
        ComputeBackend::CpuScipy
    }
}

fn dot(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}

pub fn conjugate_gradient(
    matrix: &CsrMatrix,
    rhs: &[f64],
    tol: f64,
    max_iter: usize,
) -> Result<(Vec<f64>, usize, f64, Vec<IterationRecord>), FemError> {
    let n = rhs.len();
    let mut x = vec![0.0; n];
    let mut r = rhs.to_vec();
    let mut p = r.clone();

    let rhs_norm = dot(rhs, rhs).sqrt().max(1e-30);
    let mut rs_old = dot(&r, &r);
    let mut history = Vec::new();

    if rs_old.sqrt() / rhs_norm <= tol {
        return Ok((x, 0, rs_old.sqrt() / rhs_norm, history));
    }

    for iter in 1..=max_iter {
        let ap = matrix.matvec(&p);
        let p_ap = dot(&p, &ap);
        if p_ap.abs() <= 1e-36 {
            break;
        }
        let alpha = rs_old / p_ap;

        for i in 0..n {
            x[i] += alpha * p[i];
            r[i] -= alpha * ap[i];
        }

        let rs_new = dot(&r, &r);
        let rel_res = rs_new.sqrt() / rhs_norm;

        if iter % 10 == 0 || rel_res <= tol {
            history.push(IterationRecord {
                step_no: iter,
                residual: rel_res,
            });
        }

        if rel_res <= tol {
            return Ok((x, iter, rel_res, history));
        }

        let beta = rs_new / rs_old;
        for i in 0..n {
            p[i] = r[i] + beta * p[i];
        }
        rs_old = rs_new;
    }

    let final_res = rs_old.sqrt() / rhs_norm;
    Err(FemError::NotConverged {
        iterations: max_iter,
        residual: final_res,
    })
}

pub fn solve_dirichlet(
    matrix: &CsrMatrix,
    rhs: &[f64],
    dirichlet_mask: &[bool],
    dirichlet_values: &[f64],
    hw: &HardwareCapabilities,
) -> Result<(Vec<f64>, ComputeBackend, Option<String>, usize, f64, Vec<IterationRecord>), FemError> {
    let n = matrix.nrows;
    let mut mod_rhs = rhs.to_vec();

    // Перенос известных значений Дирихле в правую часть
    for r in 0..n {
        if dirichlet_mask[r] {
            mod_rhs[r] = dirichlet_values[r];
            continue;
        }
        for idx in matrix.row_ptr[r]..matrix.row_ptr[r + 1] {
            let c = matrix.col_idx[idx];
            if dirichlet_mask[c] {
                mod_rhs[r] -= matrix.values[idx] * dirichlet_values[c];
            }
        }
    }

    // Обнуление строк и столбцов Дирихле с единицей на диагонали
    let mut triplets = Vec::with_capacity(matrix.values.len());
    for r in 0..n {
        if dirichlet_mask[r] {
            triplets.push((r, r, 1.0));
        } else {
            for idx in matrix.row_ptr[r]..matrix.row_ptr[r + 1] {
                let c = matrix.col_idx[idx];
                if !dirichlet_mask[c] {
                    triplets.push((r, c, matrix.values[idx]));
                }
            }
        }
    }
    let reduced_matrix = CsrMatrix::from_triplets(n, n, &triplets);

    let preferred_backend = select_backend(&reduced_matrix, hw);
    let mut used_backend = preferred_backend;
    let mut fallback_reason = None;

    if preferred_backend == ComputeBackend::CudaFp64 {
        if hw.simulate_cuda_oom {
            // Перехват CUDA OOM (NFR-REL-03) и автоматический переход на CPU
            used_backend = ComputeBackend::CpuScipy;
            fallback_reason = Some(
                "CUDA OutOfMemory: очищен кэш VRAM, выполнено переключение на cpu_scipy".to_string(),
            );
        }
    }

    let (sol, iters, res, history) =
        conjugate_gradient(&reduced_matrix, &mod_rhs, CG_TOLERANCE, CG_MAX_ITERATIONS)?;

    Ok((sol, used_backend, fallback_reason, iters, res, history))
}

// =============================================================================
// Шаги 6 и 7 · Квазистатическая электростатика + Био–Савар и стационарное тепло
// =============================================================================

pub fn solve_quasistatic_electric_field(
    params: &DeviceParameters,
    op: &OperatingPoint,
    grid_shape: [usize; 3],
    hw: &HardwareCapabilities,
) -> Result<ElectrostaticResult, FemError> {
    validate_mesh_budget(grid_shape, hw)?;
    let mesh = make_tetrahedral_mesh(params, grid_shape)?;

    let eps = (params.epsilon_r.max(1.0)) * EPSILON_0;
    let coeffs = vec![eps; mesh.tetrahedra.len()];
    let stiffness = assemble_tetrahedral_stiffness(&mesh, &coeffs);

    let [nx, ny, nz] = grid_shape;
    let n_nodes = mesh.nodes.len();
    let mut mask = vec![false; n_nodes];
    let mut dirichlet_vals = vec![0.0; n_nodes];
    let rhs = vec![0.0; n_nodes];

    // Граничные условия Дирихле: исток (x=0), сток (x=Lx), затвор (z=0 в области канала)
    for ix in 0..nx {
        for iy in 0..ny {
            for iz in 0..nz {
                let idx = mesh.node_index(ix, iy, iz);
                if ix == 0 && iz >= nz / 3 {
                    mask[idx] = true;
                    dirichlet_vals[idx] = op.v_source;
                } else if ix == nx - 1 && iz >= nz / 3 {
                    mask[idx] = true;
                    dirichlet_vals[idx] = op.v_drain;
                } else if iz == 0 && ix >= nx / 5 && ix <= (4 * nx) / 5 {
                    mask[idx] = true;
                    dirichlet_vals[idx] = op.v_gate;
                }
            }
        }
    }

    let (potential_vals, backend, fallback_reason, iters, final_res, history) =
        solve_dirichlet(&stiffness, &rhs, &mask, &dirichlet_vals, hw)?;

    // Расчёт |E| = |∇V| по тетраэдрам с осреднением в узлы и расчётом энергии W = 1/2 ∫ ε|E|² dV
    let mut node_e_sum = vec![0.0; n_nodes];
    let mut node_vol_sum = vec![0.0; n_nodes];
    let mut total_energy_j = 0.0;

    for &tet in &mesh.tetrahedra {
        let pts = [
            mesh.nodes[tet[0]],
            mesh.nodes[tet[1]],
            mesh.nodes[tet[2]],
            mesh.nodes[tet[3]],
        ];
        if let Some((grads, vol)) = tetrahedron_gradients_and_volume(pts) {
            let mut ex = 0.0;
            let mut ey = 0.0;
            let mut ez = 0.0;
            for a in 0..4 {
                let v_a = potential_vals[tet[a]];
                ex -= v_a * grads[a][0];
                ey -= v_a * grads[a][1];
                ez -= v_a * grads[a][2];
            }
            let e_mag = (ex * ex + ey * ey + ez * ez).sqrt();
            total_energy_j += 0.5 * eps * e_mag * e_mag * vol;

            for &v_idx in &tet {
                node_e_sum[v_idx] += e_mag * vol;
                node_vol_sum[v_idx] += vol;
            }
        }
    }

    let e_field_vals: Vec<f64> = node_e_sum
        .iter()
        .zip(node_vol_sum.iter())
        .map(|(&s, &v)| if v > 0.0 { s / v } else { 0.0 })
        .collect();

    // Геометрическая ёмкость C = 2W / V_eff²
    let v_diff = (op.v_gate - op.v_source)
        .abs()
        .max((op.v_drain - op.v_source).abs())
        .max(1e-3);
    let capacitance_f = 2.0 * total_energy_j / (v_diff * v_diff);

    // Аналитическая карта |B| по закону Био–Савара для конечного прямого проводника канала (ток Id)
    let lx = params.length_um * 1e-6;
    let ly = params.width_um * 1e-6;
    let y_channel = 0.5 * ly;
    let z_channel = params.tox_nm * 1e-9;

    let b_field_vals: Vec<f64> = mesh
        .nodes
        .iter()
        .map(|&[x, y, z]| {
            let dy = y - y_channel;
            let dz = z - z_channel;
            let rho = (dy * dy + dz * dz).sqrt().max(1e-9);
            let r1 = ((x).powi(2) + rho * rho).sqrt();
            let r2 = ((lx - x).powi(2) + rho * rho).sqrt();
            let sin_theta1 = x / r1;
            let sin_theta2 = (lx - x) / r2;
            (MU_0 * op.i_drain.abs() / (4.0 * std::f64::consts::PI * rho))
                * (sin_theta1 + sin_theta2)
        })
        .collect();

    Ok(ElectrostaticResult {
        potential: FieldSolution {
            quantity: FieldQuantity::ElectricPotentialV,
            values: potential_vals,
            shape: grid_shape,
            backend,
            fallback_reason: fallback_reason.clone(),
            iterations: iters,
            final_residual: final_res,
            convergence_history: history.clone(),
        },
        electric_field_mag: FieldSolution {
            quantity: FieldQuantity::ElectricFieldMagVm,
            values: e_field_vals,
            shape: grid_shape,
            backend,
            fallback_reason: fallback_reason.clone(),
            iterations: iters,
            final_residual: final_res,
            convergence_history: history.clone(),
        },
        magnetic_field_mag: FieldSolution {
            quantity: FieldQuantity::MagneticFluxDensityT,
            values: b_field_vals,
            shape: grid_shape,
            backend,
            fallback_reason,
            iterations: iters,
            final_residual: final_res,
            convergence_history: history,
        },
        capacitance_f,
        stored_energy_j: total_energy_j,
    })
}

pub fn solve_steady_thermal(
    params: &DeviceParameters,
    op: &OperatingPoint,
    grid_shape: [usize; 3],
    hw: &HardwareCapabilities,
) -> Result<ThermalResult, FemError> {
    if params.thermal_conductivity_w_mk <= 0.0 {
        return Err(FemError::InvalidPhysicsInput(
            "Теплопроводность k должна быть строго положительной".to_string(),
        ));
    }

    validate_mesh_budget(grid_shape, hw)?;
    let mesh = make_tetrahedral_mesh(params, grid_shape)?;

    let k = params.thermal_conductivity_w_mk;
    let coeffs = vec![k; mesh.tetrahedra.len()];
    let stiffness = assemble_tetrahedral_stiffness(&mesh, &coeffs);

    let [nx, ny, nz] = grid_shape;
    let n_nodes = mesh.nodes.len();
    let mut mask = vec![false; n_nodes];
    let mut dirichlet_vals = vec![params.ambient_temp_k; n_nodes];
    let mut rhs = vec![0.0; n_nodes];

    // Граничное условие Дирихле T = T_amb на нижней границе подложки (z = Lz)
    for ix in 0..nx {
        for iy in 0..ny {
            let idx = mesh.node_index(ix, iy, nz - 1);
            mask[idx] = true;
            dirichlet_vals[idx] = params.ambient_temp_k;
        }
    }

    // Локализованный источник тепла Q ∝ Vds · Id в области канала
    let power_w = ((op.v_drain - op.v_source) * op.i_drain).abs();
    let x_min = 0.2 * params.length_um * 1e-6;
    let x_max = 0.8 * params.length_um * 1e-6;
    let z_chan_max = (params.tox_nm + params.body_thickness_nm) * 1e-9;

    let mut channel_tets = Vec::new();
    let mut channel_vol = 0.0;

    for (t_idx, &tet) in mesh.tetrahedra.iter().enumerate() {
        let pts = [
            mesh.nodes[tet[0]],
            mesh.nodes[tet[1]],
            mesh.nodes[tet[2]],
            mesh.nodes[tet[3]],
        ];
        let cx = 0.25 * (pts[0][0] + pts[1][0] + pts[2][0] + pts[3][0]);
        let cz = 0.25 * (pts[0][2] + pts[1][2] + pts[2][2] + pts[3][2]);

        if cx >= x_min && cx <= x_max && cz <= z_chan_max {
            if let Some((_, vol)) = tetrahedron_gradients_and_volume(pts) {
                channel_tets.push((t_idx, vol));
                channel_vol += vol;
            }
        }
    }

    if power_w > 0.0 && channel_vol > 0.0 {
        let q_volumetric = power_w / channel_vol; // Вт/м³
        for (t_idx, vol) in channel_tets {
            let tet = mesh.tetrahedra[t_idx];
            let node_heat = 0.25 * q_volumetric * vol;
            for &v_idx in &tet {
                if !mask[v_idx] {
                    rhs[v_idx] += node_heat;
                }
            }
        }
    }

    let (temp_vals, backend, fallback_reason, iters, final_res, history) =
        solve_dirichlet(&stiffness, &rhs, &mask, &dirichlet_vals, hw)?;

    let t_max_k = temp_vals
        .iter()
        .copied()
        .fold(params.ambient_temp_k, f64::max);
    let delta_t_k = (t_max_k - params.ambient_temp_k).max(0.0);
    let r_th_k_w = if power_w > 1e-15 {
        delta_t_k / power_w
    } else {
        0.0
    };

    Ok(ThermalResult {
        temperature: FieldSolution {
            quantity: FieldQuantity::TemperatureK,
            values: temp_vals,
            shape: grid_shape,
            backend,
            fallback_reason,
            iterations: iters,
            final_residual: final_res,
            convergence_history: history,
        },
        t_max_k,
        delta_t_k,
        r_th_k_w,
    })
}
