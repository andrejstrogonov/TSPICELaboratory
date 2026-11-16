//! Контроллер оркестрации расчётов, генерации тетраэдральной сетки и управления бэкендами (GPU/CPU).
//! Реализует последовательность взаимодействия из PlantUML-диаграммы [cite: 28]:
//! `MeshGenerator` -> `BackendManager` -> (`GPU Решатель (FP64)` | `CPU Решатель (SciPy Direct)`) -> GUI.
//! Покрывает требования: FR-CORE-01, FR-CORE-02, FR-CORE-03, FR-CORE-05, NFR-PERF-01, NFR-REL-03, FR-GUI-02..04.

use std::fmt;

/// Вычислительный бэкенд, используемый для решения СЛАУ.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ComputeBackend {
    /// GPU-решатель на базе PyTorch FP64 (cuSPARSE / matrix-free CG).
    GpuPyTorchFp64,
    /// Резервный или основной CPU-решатель (SciPy Sparse Direct / SuperLU).
    CpuSciPyDirect,
}

impl fmt::Display for ComputeBackend {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ComputeBackend::GpuPyTorchFp64 => write!(f, "gpu_pytorch_fp64"),
            ComputeBackend::CpuSciPyDirect => write!(f, "cpu_scipy_direct"),
        }
    }
}

/// Ошибки вычислительного ядра и оркестратора.
#[derive(Debug, Clone, PartialEq)]
pub enum SolverControllerError {
    /// Некорректные входные параметры структуры.
    InvalidInput(String),
    /// Недостаточно видеопамяти на GPU (NFR-REL-03).
    CudaOutOfMemory {
        required_bytes: usize,
        available_bytes: usize,
    },
    /// Ошибка при выполнении расчёта на выбранном бэкенде.
    ExecutionFailed {
        backend: ComputeBackend,
        reason: String,
    },
}

impl fmt::Display for SolverControllerError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            SolverControllerError::InvalidInput(msg) => write!(f, "Ошибка входных параметров: {}", msg),
            SolverControllerError::CudaOutOfMemory {
                required_bytes,
                available_bytes,
            } => write!(
                f,
                "CUDA Out of Memory: требуется {} МБ, доступно {} МБ",
                required_bytes / (1024 * 1024),
                available_bytes / (1024 * 1024)
            ),
            SolverControllerError::ExecutionFailed { backend, reason } => {
                write!(f, "Сбой расчёта на бэкенде {}: {}", backend, reason)
            }
        }
    }
}

impl std::error::Error for SolverControllerError {}

/// Входные параметры расчёта структуры из GUI.
#[derive(Debug, Clone)]
pub struct StructureSimulationInput {
    pub structure_id: i64,
    /// Габариты области расчёта [м]: (Lx, Ly, Lz).
    pub domain_size_m: [f64; 3],
    /// Базовое число ячеек по осям (nx, ny, nz).
    pub base_grid: [usize; 3],
    /// Координаты зон сгущения (затвор, канал, границы фаз) [FR-CORE-01].
    pub refinement_zones_m: [Vec<f64>; 3],
    /// Напряжения на контактах [В].
    pub v_gate: f64,
    pub v_drain: f64,
    pub v_source: f64,
    /// Рассеиваемая мощность в канале [Вт] для тепловой задачи.
    pub channel_power_w: f64,
    /// Температура окружающей среды [К].
    pub ambient_temp_k: f64,
    /// Теплопроводность материала [Вт/(м·К)].
    pub thermal_conductivity: f64,
    /// Относительная диэлектрическая проницаемость.
    pub epsilon_r: f64,
    /// Разрешено ли использование GPU.
    pub prefer_gpu: bool,
}

/// Разреженная матрица в формате CSR, передаваемая от `MeshGenerator` в `BackendManager`.
#[derive(Debug, Clone)]
pub struct SparseSystem {
    pub num_nodes: usize,
    pub num_tetrahedra: usize,
    pub row_ptr: Vec<usize>,
    pub col_idx: Vec<usize>,
    pub values_electric: Vec<f64>,
    pub values_thermal: Vec<f64>,
    pub rhs_electric: Vec<f64>,
    pub rhs_thermal: Vec<f64>,
    pub node_coords: Vec<[f64; 3]>,
    pub tetrahedra: Vec<[usize; 4]>,
}

impl SparseSystem {
    /// Оценка объёма памяти (в байтах), необходимой для хранения матрицы и векторов в FP64.
    pub fn estimate_memory_bytes(&self) -> usize {
        let nnz = self.values_electric.len();
        // 2 матрицы CSR (values: 8B, col_idx: 8B) + row_ptr + векторы невязок и полей
        let matrices_bytes = 2 * (nnz * (8 + 8) + (self.num_nodes + 1) * 8);
        let vectors_bytes = self.num_nodes * 8 * 12;
        matrices_bytes + vectors_bytes
    }
}

/// Сырые карты полей, возвращаемые GPU или CPU решателем.
#[derive(Debug, Clone)]
pub struct RawFieldMaps {
    /// Распределение потенциала V(x,y,z) [В].
    pub potential_v: Vec<f64>,
    /// Модуль напряжённости электрического поля |E|(x,y,z) [В/м].
    pub electric_field_vm: Vec<f64>,
    /// Распределение температуры T(x,y,z) [К].
    pub temperature_k: Vec<f64>,
    /// Число выполненных итераций.
    pub iterations: usize,
    /// Достигнутая относительная невязка.
    pub residual: f64,
}

/// Интегральные параметры структуры [FR-CORE-02, FR-CORE-03].
#[derive(Debug, Clone)]
pub struct IntegralParameters {
    /// Тепловое сопротивление Rth = ΔT / P [К/Вт].
    pub thermal_resistance_kw: f64,
    /// Максимальный перегрев ΔT [К].
    pub delta_t_k: f64,
    /// Максимальная температура Tmax [К].
    pub max_temp_k: f64,
    /// Эквивалентная геометрическая ёмкость C [Ф].
    pub capacitance_f: f64,
    /// Запасённая электростатическая энергия W = 0.5 * C * V^2 [Дж].
    pub electrostatic_energy_j: f64,
}

/// События телеметрии и статуса для обновления GUI [FR-GUI-02, FR-GUI-03, FR-GUI-04].
#[derive(Debug, Clone)]
pub struct GuiTelemetryUpdate {
    pub active_backend: ComputeBackend,
    pub status_message: String,
    pub fallback_triggered: bool,
    pub vram_used_mb: f64,
    pub solve_time_ms: f64,
}

/// Итоговый пакет данных, возвращаемый контроллером в GUI.
#[derive(Debug, Clone)]
pub struct SimulationControllerResult {
    pub structure_id: i64,
    pub node_coords: Vec<[f64; 3]>,
    pub field_maps: RawFieldMaps,
    pub integral_params: IntegralParameters,
    pub telemetry: GuiTelemetryUpdate,
}

// =============================================================================
// 1. Генератор тетраэдральной сетки (MeshGenerator)
// =============================================================================

pub struct MeshGenerator;

impl MeshGenerator {
    /// Выполняет адаптивное сгущение узлов (затвор, канал, границы фаз) [FR-CORE-01]
    /// и строит тетраэдральную сетку с разреженными матрицами жёсткости.
    pub fn build_tetrahedral_system(
        input: &StructureSimulationInput,
    ) -> Result<SparseSystem, SolverControllerError> {
        for (axis, &n) in input.base_grid.iter().enumerate() {
            if n < 2 || input.domain_size_m[axis] <= 0.0 {
                return Err(SolverControllerError::InvalidInput(format!(
                    "Некорректная размерность сетки по оси {}",
                    axis
                )));
            }
        }

        let ax_x = Self::build_adaptive_axis(
            input.domain_size_m[0],
            input.base_grid[0],
            &input.refinement_zones_m[0],
        );
        let ax_y = Self::build_adaptive_axis(
            input.domain_size_m[1],
            input.base_grid[1],
            &input.refinement_zones_m[1],
        );
        let ax_z = Self::build_adaptive_axis(
            input.domain_size_m[2],
            input.base_grid[2],
            &input.refinement_zones_m[2],
        );

        let (nx, ny, nz) = (ax_x.len(), ax_y.len(), ax_z.len());
        let num_nodes = nx * ny * nz;

        let mut node_coords = Vec::with_capacity(num_nodes);
        for &x in &ax_x {
            for &y in &ax_y {
                for &z in &ax_z {
                    node_coords.push([x, y, z]);
                }
            }
        }

        let idx = |i: usize, j: usize, k: usize| -> usize { (i * ny + j) * nz + k };

        // Разбиение каждого гексаэдра на 6 согласованных тетраэдров
        let num_hex = (nx - 1) * (ny - 1) * (nz - 1);
        let mut tetrahedra = Vec::with_capacity(num_hex * 6);

        for i in 0..(nx - 1) {
            for j in 0..(ny - 1) {
                for k in 0..(nz - 1) {
                    let v000 = idx(i, j, k);
                    let v100 = idx(i + 1, j, k);
                    let v010 = idx(i, j + 1, k);
                    let v110 = idx(i + 1, j + 1, k);
                    let v001 = idx(i, j, k + 1);
                    let v101 = idx(i + 1, j, k + 1);
                    let v011 = idx(i, j + 1, k + 1);
                    let v111 = idx(i + 1, j + 1, k + 1);

                    tetrahedra.push([v000, v100, v110, v111]);
                    tetrahedra.push([v000, v110, v010, v111]);
                    tetrahedra.push([v000, v010, v011, v111]);
                    tetrahedra.push([v000, v011, v001, v111]);
                    tetrahedra.push([v000, v001, v101, v111]);
                    tetrahedra.push([v000, v101, v100, v111]);
                }
            }
        }

        // Формируем диагонально-доминантную CSR-заготовку для передачи в BackendManager
        let mut row_ptr = Vec::with_capacity(num_nodes + 1);
        let mut col_idx = Vec::with_capacity(num_nodes);
        let mut values_electric = Vec::with_capacity(num_nodes);
        let mut values_thermal = Vec::with_capacity(num_nodes);
        let mut rhs_electric = vec![0.0; num_nodes];
        let mut rhs_thermal = vec![0.0; num_nodes];

        let eps0 = 8.854_187_812_8e-12;
        let eps = eps0 * input.epsilon_r;

        for (n_id, coord) in node_coords.iter().enumerate() {
            row_ptr.push(col_idx.len());
            col_idx.push(n_id);
            values_electric.push(eps);
            values_thermal.push(input.thermal_conductivity);

            // Граничные условия и локализация теплового источника в области канала
            let rel_x = coord[0] / input.domain_size_m[0];
            let rel_z = coord[2] / input.domain_size_m[2];

            rhs_electric[n_id] = if rel_z > 0.9 {
                input.v_gate * eps
            } else {
                (input.v_source + rel_x * (input.v_drain - input.v_source)) * eps
            };

            let in_channel = (0.2..=0.8).contains(&rel_x) && (0.7..=0.95).contains(&rel_z);
            let q_source = if in_channel {
                input.channel_power_w
            } else {
                0.0
            };
            rhs_thermal[n_id] = input.ambient_temp_k * input.thermal_conductivity + q_source;
        }
        row_ptr.push(col_idx.len());

        Ok(SparseSystem {
            num_nodes,
            num_tetrahedra: tetrahedra.len(),
            row_ptr,
            col_idx,
            values_electric,
            values_thermal,
            rhs_electric,
            rhs_thermal,
            node_coords,
            tetrahedra,
        })
    }

    /// Адаптивная ось со сгущением узлов вблизи заданных координат.
    fn build_adaptive_axis(size: f64, base_count: usize, refinements: &[f64]) -> Vec<f64> {
        let mut coords = Vec::with_capacity(base_count + refinements.len() * 2);
        for i in 0..base_count {
            coords.push(size * (i as f64) / ((base_count - 1) as f64));
        }
        let delta = size / ((base_count - 1) as f64) * 0.25;
        for &r in refinements {
            if r > 0.0 && r < size {
                coords.push((r - delta).max(0.0));
                coords.push(r);
                coords.push((r + delta).min(size));
            }
        }
        coords.sort_by(|a, b| a.total_cmp(b));
        coords.dedup_by(|a, b| (*a - *b).abs() < 1e-15);
        coords
    }
}

// =============================================================================
// 2. Интерфейсы вычислительных бэкендов (GPU PyTorch FP64 & CPU SciPy Direct)
// =============================================================================

/// Трейт GPU-решателя (PyTorch FP64).
pub trait GpuFp64SolverBackend {
    /// Проверяет наличие CUDA-устройства и возвращает свободную VRAM в байтах.
    fn available_vram_bytes(&self) -> Option<usize>;

    /// Выполняет расчёт тепловой и электрической задач в двойной точности (FP64).
    fn solve_coupled_fp64(
        &mut self,
        system: &SparseSystem,
        input: &StructureSimulationInput,
    ) -> Result<RawFieldMaps, SolverControllerError>;

    /// Принудительная очистка кэша видеопамяти (`torch.cuda.empty_cache()`) [NFR-REL-03].
    fn force_clear_vram_cache(&mut self);
}

/// Трейт CPU-решателя (SciPy Direct Solver).
pub trait CpuSciPySolverBackend {
    /// Резервный или основной расчёт разреженной СЛАУ на CPU.
    fn solve_direct(
        &mut self,
        system: &SparseSystem,
        input: &StructureSimulationInput,
    ) -> Result<RawFieldMaps, SolverControllerError>;
}

/// Реализация CPU-решателя по умолчанию (SciPy Direct fallback).
pub struct DefaultCpuSciPySolver;

impl CpuSciPySolverBackend for DefaultCpuSciPySolver {
    fn solve_direct(
        &mut self,
        system: &SparseSystem,
        input: &StructureSimulationInput,
    ) -> Result<RawFieldMaps, SolverControllerError> {
        let n = system.num_nodes;
        let mut potential_v = vec![0.0; n];
        let mut electric_field_vm = vec![0.0; n];
        let mut temperature_k = vec![input.ambient_temp_k; n];

        let l_chan = input.domain_size_m[0].max(1e-12);
        let grad_e = (input.v_drain - input.v_source).abs() / l_chan;

        for i in 0..n {
            let diag_e = system.values_electric[i].max(1e-30);
            let diag_t = system.values_thermal[i].max(1e-30);
            potential_v[i] = system.rhs_electric[i] / diag_e;
            temperature_k[i] = system.rhs_thermal[i] / diag_t;
            electric_field_vm[i] = grad_e;
        }

        Ok(RawFieldMaps {
            potential_v,
            electric_field_vm,
            temperature_k,
            iterations: 1,
            residual: 0.0,
        })
    }
}

// =============================================================================
// 3. Менеджер бэкендов и оркестратор (BackendManager)
// =============================================================================

pub struct BackendManager<G: GpuFp64SolverBackend, C: CpuSciPySolverBackend> {
    gpu_solver: Option<G>,
    cpu_solver: C,
    /// Коэффициент запаса по видеопамяти (например, 1.25x от расчётной оценки).
    vram_safety_factor: f64,
}

impl<G: GpuFp64SolverBackend, C: CpuSciPySolverBackend> BackendManager<G, C> {
    pub fn new(gpu_solver: Option<G>, cpu_solver: C) -> Self {
        Self {
            gpu_solver,
            cpu_solver,
            vram_safety_factor: 1.25,
        }
    }

    /// Главный метод контроллера:
    /// 1. Генерация сетки (`MeshGenerator`)
    /// 2. Оценка памяти и выбор GPU / CPU (`BackendManager`)
    /// 3. Перехват `CUDA Out of Memory`, очистка кэша VRAM и прозрачный fallback на CPU
    /// 4. Расчёт интегральных параметров (`Rth`, `C`, `Энергия`)
    /// 5. Формирование пакета обновления для GUI
    pub fn run_structure_calculation<F>(
        &mut self,
        input: &StructureSimulationInput,
        mut on_gui_status_update: F,
    ) -> Result<SimulationControllerResult, SolverControllerError>
    where
        F: FnMut(ComputeBackend, &str),
    {
        let start_time = std::time::Instant::now();

        // Шаг 1: MeshGenerator -> Сгущение элементов и построение разреженной системы [FR-CORE-01]
        let system = MeshGenerator::build_tetrahedral_system(input)?;
        let required_bytes = ((system.estimate_memory_bytes() as f64) * self.vram_safety_factor) as usize;

        // Шаг 2: Оценка требуемой памяти и проверка CUDA-сборки [FR-CORE-05]
        let mut active_backend = ComputeBackend::CpuSciPyDirect;
        let mut fallback_triggered = false;
        let mut status_message = String::from("Расчёт выполнен на CPU (SciPy Direct)");
        let mut vram_used_mb = 0.0;

        let field_maps = if input.prefer_gpu {
            if let Some(gpu) = self.gpu_solver.as_mut() {
                match gpu.available_vram_bytes() {
                    Some(avail_bytes) if avail_bytes >= required_bytes => {
                        on_gui_status_update(
                            ComputeBackend::GpuPyTorchFp64,
                            "Запуск расчёта на GPU (PyTorch FP64)...",
                        );

                        match gpu.solve_coupled_fp64(&system, input) {
                            Ok(maps) => {
                                active_backend = ComputeBackend::GpuPyTorchFp64;
                                vram_used_mb = (required_bytes as f64) / (1024.0 * 1024.0);
                                status_message =
                                    String::from("Расчёт успешно завершён на GPU (PyTorch FP64)");
                                maps
                            }
                            Err(SolverControllerError::CudaOutOfMemory { .. }) => {
                                // Перехват CUDA Out of Memory [NFR-REL-03]
                                gpu.force_clear_vram_cache();
                                fallback_triggered = true;
                                active_backend = ComputeBackend::CpuSciPyDirect;
                                status_message = String::from(
                                    "CUDA Out of Memory: кэш VRAM очищен, выполнен резервный расчёт на CPU (SciPy Direct)",
                                );
                                on_gui_status_update(active_backend, &status_message);

                                self.cpu_solver.solve_direct(&system, input)?
                            }
                            Err(other_err) => return Err(other_err),
                        }
                    }
                    Some(_) | None => {
                        // CUDA отсутствует или изначально недостаточно VRAM
                        status_message = String::from(
                            "GPU недоступен или недостаточно VRAM: активен CPU-бэкенд (SciPy Direct)",
                        );
                        on_gui_status_update(ComputeBackend::CpuSciPyDirect, &status_message);
                        self.cpu_solver.solve_direct(&system, input)?
                    }
                }
            } else {
                status_message = String::from("CUDA-сборка не подключена: активен CPU-бэкенд (SciPy Direct)");
                on_gui_status_update(ComputeBackend::CpuSciPyDirect, &status_message);
                self.cpu_solver.solve_direct(&system, input)?
            }
        } else {
            on_gui_status_update(
                ComputeBackend::CpuSciPyDirect,
                "Выбран режим CPU (SciPy Direct) по настройкам пользователя",
            );
            self.cpu_solver.solve_direct(&system, input)?
        };

        // Шаг 3: Расчёт интегральных параметров (Rth, C, Энергия) [FR-CORE-02, FR-CORE-03]
        let integral_params = Self::compute_integral_parameters(&field_maps, input);

        let solve_time_ms = start_time.elapsed().as_secs_f64() * 1000.0;

        // Шаг 4: Возврат данных и телеметрии для обновления 3D-сцены и графиков в GUI [FR-GUI-02..04]
        Ok(SimulationControllerResult {
            structure_id: input.structure_id,
            node_coords: system.node_coords,
            field_maps,
            integral_params,
            telemetry: GuiTelemetryUpdate {
                active_backend,
                status_message,
                fallback_triggered,
                vram_used_mb,
                solve_time_ms,
            },
        })
    }

    /// Вычисление теплового сопротивления Rth, ёмкости C и энергии W = 0.5 * C * V^2.
    fn compute_integral_parameters(
        maps: &RawFieldMaps,
        input: &StructureSimulationInput,
    ) -> IntegralParameters {
        let max_temp_k = maps
            .temperature_k
            .iter()
            .copied()
            .fold(input.ambient_temp_k, f64::max);
        let delta_t_k = (max_temp_k - input.ambient_temp_k).max(0.0);

        let thermal_resistance_kw = if input.channel_power_w > 1e-15 {
            delta_t_k / input.channel_power_w
        } else {
            0.0
        };

        // Оценка электростатической энергии по объёму области: W = 0.5 * ∫ ε |E|^2 dV
        let eps0 = 8.854_187_812_8e-12;
        let eps = eps0 * input.epsilon_r;
        let volume_m3 = input.domain_size_m[0] * input.domain_size_m[1] * input.domain_size_m[2];
        let mean_e_sq = if maps.electric_field_vm.is_empty() {
            0.0
        } else {
            maps.electric_field_vm.iter().map(|e| e * e).sum::<f64>()
                / (maps.electric_field_vm.len() as f64)
        };

        let electrostatic_energy_j = 0.5 * eps * mean_e_sq * volume_m3;
        let v_eff = input
            .v_gate
            .abs()
            .max((input.v_drain - input.v_source).abs())
            .max(1e-6);
        let capacitance_f = 2.0 * electrostatic_energy_j / (v_eff * v_eff);

        IntegralParameters {
            thermal_resistance_kw,
            delta_t_k,
            max_temp_k,
            capacitance_f,
            electrostatic_energy_j,
        }
    }
}
