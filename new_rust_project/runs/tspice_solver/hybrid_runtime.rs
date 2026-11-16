// =============================================================================
// TSPICE Laboratory · Enterprise Computational Core (Rust)
// A-9 · Гибридный runtime: маршрутизация задача -> метод -> backend -> персистенция
// Источники: src/simulation.py (guards), src/extractors.py (create_extractor),
//            src/main.py (Worker-классы), RefactorREADME.md (PyQt6 -> Django -> Rust/ONNX)
// =============================================================================

use std::collections::HashMap;
use std::fmt;
use std::sync::Arc;
use std::time::Instant;

// -----------------------------------------------------------------------------
// 1. Семейства приборов и доменные инварианты (D-10 / A-9)
// -----------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum DeviceFamily {
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

impl DeviceFamily {
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::PlanarMosfet => "planar_mosfet",
            Self::FdSoi => "fd_soi",
            Self::Finfet => "finfet",
            Self::GaaNanosheet => "gaa_nanosheet",
            Self::GaaNanowire => "gaa_nanowire",
            Self::Tfet => "tfet",
            Self::Hemt => "hemt",
            Self::Memristor => "memristor",
            Self::Memtransistor => "memtransistor",
            Self::CadCustom => "cad_custom",
        }
    }

    pub fn is_mos_group(&self) -> bool {
        matches!(
            self,
            Self::PlanarMosfet
                | Self::FdSoi
                | Self::Finfet
                | Self::GaaNanosheet
                | Self::GaaNanowire
        )
    }

    pub fn is_memory_group(&self) -> bool {
        matches!(self, Self::Memristor | Self::Memtransistor)
    }

    pub fn supports_quantum_confinement(&self) -> bool {
        matches!(
            self,
            Self::PlanarMosfet
                | Self::FdSoi
                | Self::Finfet
                | Self::GaaNanosheet
                | Self::GaaNanowire
                | Self::CadCustom
        )
    }
}

#[derive(Debug, Clone)]
pub struct DeviceParameters {
    pub name: String,
    pub device_type: DeviceFamily,
    pub epsilon_r: f64,
    pub electron_mobility: f64,
    pub hole_mobility: f64,
    pub bandgap: f64,
    pub doping: f64,
    pub tox_nm: f64,
    pub width_um: f64,
    pub length_um: f64,
    pub temperature_k: f64,
    pub geometry_fields: HashMap<String, f64>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct DomainValidationError {
    pub field: String,
    pub message: String,
}

impl fmt::Display for DomainValidationError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Validation error in '{}': {}", self.field, self.message)
    }
}

impl std::error::Error for DomainValidationError {}

impl DeviceParameters {
    pub fn validate(&self) -> Result<(), DomainValidationError> {
        if self.name.trim().is_empty() || self.name.contains(' ') {
            return Err(DomainValidationError {
                field: "name".into(),
                message: "name must be non-empty and contain no spaces".into(),
            });
        }
        let positive_checks = [
            ("epsilon_r", self.epsilon_r),
            ("electron_mobility", self.electron_mobility),
            ("hole_mobility", self.hole_mobility),
            ("bandgap", self.bandgap),
            ("doping", self.doping),
            ("tox_nm", self.tox_nm),
            ("width_um", self.width_um),
            ("length_um", self.length_um),
            ("temperature_k", self.temperature_k),
        ];
        for (field, val) in positive_checks {
            if !val.is_finite() || val <= 0.0 {
                return Err(DomainValidationError {
                    field: field.into(),
                    message: format!("{} must be finite and > 0", field),
                });
            }
        }

        match self.device_type {
            DeviceFamily::Finfet => {
                let w = *self.geometry_fields.get("fin_width_nm").unwrap_or(&8.0);
                let p = *self.geometry_fields.get("fin_pitch_nm").unwrap_or(&30.0);
                if p < w {
                    return Err(DomainValidationError {
                        field: "fin_pitch_nm".into(),
                        message: "Fin pitch must not be smaller than fin width".into(),
                    });
                }
            }
            DeviceFamily::GaaNanosheet => {
                let t = *self.geometry_fields.get("sheet_thickness_nm").unwrap_or(&8.0);
                let p = *self.geometry_fields.get("sheet_pitch_nm").unwrap_or(&30.0);
                if p < t {
                    return Err(DomainValidationError {
                        field: "sheet_pitch_nm".into(),
                        message: "Sheet pitch must not be smaller than sheet thickness".into(),
                    });
                }
            }
            DeviceFamily::Memristor | DeviceFamily::Memtransistor => {
                let r_on = *self.geometry_fields.get("low_resistance_ohm").unwrap_or(&100.0);
                let r_off = *self.geometry_fields.get("high_resistance_ohm").unwrap_or(&1e5);
                if r_on >= r_off {
                    return Err(DomainValidationError {
                        field: "low_resistance_ohm".into(),
                        message: "R_on must be strictly less than R_off".into(),
                    });
                }
            }
            _ => {}
        }

        Ok(())
    }
}

// -----------------------------------------------------------------------------
// 2. Классы задач, методы, решатели и воркеры GUI (A-9 / D-10)
// -----------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PhysicalFieldKind {
    PotentialAndCapacitance, // V, C, |E| [+ |B|]
    Thermal,                 // T(x,y,z) с Q = Vds * Id
    CarrierConcentration,    // n(x,y,z), p(x,y,z)
}

#[derive(Debug, Clone)]
pub enum ImportPayload {
    CsvMeasurements { rows: usize },
    StlCadGeometry { triangles: usize, unit: String },
    JsonStructure,
}

#[derive(Debug, Clone)]
pub enum TaskRequest {
    ConfigOrImport {
        params: DeviceParameters,
        payload: ImportPayload,
    },
    PhysicalField {
        params: DeviceParameters,
        kind: PhysicalFieldKind,
        include_magnetic_b: bool,
        grid_shape: [usize; 3],
        prefer_gpu: bool,
    },
    Characteristics {
        params: DeviceParameters,
        surrogate_available: bool,
        surrogate_in_confidence_zone: bool,
        prefer_gpu: bool,
    },
    ModelExtraction {
        params: DeviceParameters,
        curves_count: usize,
        run_two_stage_verification: bool,
        prefer_gpu: bool,
    },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SolverCode {
    S01FemLaplace3d,
    S02FemThermal3d,
    S03QuantumPoisson2d,
    S04QuantumPoisson3d,
    S05ReducedPde1d,
    S06MemStateOde,
    S07BiotSavart,
    S08LeastSquaresBsim4,
    S09LeastSquaresCmg,
    S10BehavioralSubckt,
    S11VerilogAOpenVaf,
    S12UnetSurrogate,
    S13GnnSurrogate,
}

impl SolverCode {
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::S01FemLaplace3d => "S01_FEM_LAPLACE_3D",
            Self::S02FemThermal3d => "S02_FEM_THERMAL_3D",
            Self::S03QuantumPoisson2d => "S03_QUANTUM_POISSON_2D",
            Self::S04QuantumPoisson3d => "S04_QUANTUM_POISSON_3D",
            Self::S05ReducedPde1d => "S05_REDUCED_PDE_1D",
            Self::S06MemStateOde => "S06_MEM_STATE_ODE",
            Self::S07BiotSavart => "S07_BIOT_SAVART",
            Self::S08LeastSquaresBsim4 => "S08_LEAST_SQUARES_BSIM4",
            Self::S09LeastSquaresCmg => "S09_LEAST_SQUARES_CMG",
            Self::S10BehavioralSubckt => "S10_BEHAVIORAL_SUBCKT",
            Self::S11VerilogAOpenVaf => "S11_VERILOG_A_OPENVAF",
            Self::S12UnetSurrogate => "S12_UNET_SURROGATE",
            Self::S13GnnSurrogate => "S13_GNN_SURROGATE",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GuiWorkerType {
    SyncConfigurationWorker,
    SimulationWorker,      // A-1
    ChargeWorker,          // A-2
    CharacteristicsWorker, // A-3 / A-4
    NeuralPhysicsWorker,   // A-7 / A-8
    ExtractionWorker,      // A-5 / A-6
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ComputeBackend {
    CudaPyTorchFp64,
    CpuScipySparse,
    CpuNumpyBatch,
    CpuStateOde,
    ExternalEdaPipeline,
}

impl ComputeBackend {
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::CudaPyTorchFp64 => "cuda_pytorch_fp64",
            Self::CpuScipySparse => "cpu_scipy",
            Self::CpuNumpyBatch => "cpu_numpy",
            Self::CpuStateOde => "cpu_state_ode",
            Self::ExternalEdaPipeline => "external_eda",
        }
    }
}

// -----------------------------------------------------------------------------
// 3. Ошибки маршрутизации и исполнения
// -----------------------------------------------------------------------------

#[derive(Debug, Clone, PartialEq)]
pub enum RuntimeError {
    DomainValidation(DomainValidationError),
    UnsupportedPhysicsForFamily {
        family: DeviceFamily,
        reason: String,
    },
    MeshBudgetExceeded {
        nodes: usize,
        max_nodes: usize,
    },
    CudaOutOfMemory {
        required_bytes: usize,
        available_bytes: usize,
    },
    SolverExecutionFailed(String),
}

impl fmt::Display for RuntimeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::DomainValidation(e) => write!(f, "{}", e),
            Self::UnsupportedPhysicsForFamily { family, reason } => {
                write!(f, "Unsupported operation for '{}': {}", family.as_str(), reason)
            }
            Self::MeshBudgetExceeded { nodes, max_nodes } => {
                write!(f, "Mesh budget exceeded: {} > {}", nodes, max_nodes)
            }
            Self::CudaOutOfMemory {
                required_bytes,
                available_bytes,
            } => write!(
                f,
                "CUDA OutOfMemory: required {} bytes, available {} bytes",
                required_bytes, available_bytes
            ),
            Self::SolverExecutionFailed(msg) => write!(f, "Solver failed: {}", msg),
        }
    }
}

impl std::error::Error for RuntimeError {}

impl From<DomainValidationError> for RuntimeError {
    fn from(value: DomainValidationError) -> Self {
        Self::DomainValidation(value)
    }
}

// -----------------------------------------------------------------------------
// 4. Структуры персистенции и наблюдаемости (D1 + A-9)
// -----------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TaskPersistenceStatus {
    Ok,
    Failed,
}

#[derive(Debug, Clone)]
pub struct IterationRecord {
    pub step_no: usize,
    pub residual: f64,
    pub dt_s: f64,
    pub newton_iters: usize,
}

#[derive(Debug, Clone)]
pub struct NeuralRunRecord {
    pub surrogate_solver: SolverCode,
    pub warm_start_residual: f64,
    pub iters_saved: usize,
}

#[derive(Debug, Clone)]
pub struct PersistedArtifacts {
    pub device_structures_updated: bool,
    pub field_data_written: bool,
    pub curve_metrics_written: bool,
    pub spice_models_written: bool,
    pub verification_passed: Option<bool>,
}

#[derive(Debug, Clone)]
pub struct SolverTaskRecord {
    pub status: TaskPersistenceStatus,
    pub worker: GuiWorkerType,
    pub algorithm_tag: &'static str,
    pub solvers_used: Vec<SolverCode>,
    pub backend_used: ComputeBackend,
    pub fallback_reason: Option<String>,
    pub iterations: usize,
    pub final_residual: f64,
    pub error_code: Option<&'static str>,
    pub error_message: Option<String>,
    pub elapsed_ms: u128,
    pub iteration_log: Vec<IterationRecord>,
    pub neural_run: Option<NeuralRunRecord>,
    pub artifacts: PersistedArtifacts,
}

#[derive(Debug, Clone)]
pub struct ProgressTelemetryEvent {
    pub progress_pct: u8,
    pub stage: String,
    pub backend: ComputeBackend,
    pub vram_used_mb: f64,
}

// -----------------------------------------------------------------------------
// 5. Интерфейс вычислительных ядер и аппаратного окружения
// -----------------------------------------------------------------------------

#[derive(Debug, Clone, Copy)]
pub struct HardwareEnvironment {
    pub cuda_available: bool,
    pub fp64_supported: bool,
    pub free_vram_bytes: usize,
}

pub struct ExecutionOutcome {
    pub iterations: usize,
    pub final_residual: f64,
    pub iteration_log: Vec<IterationRecord>,
    pub neural_run: Option<NeuralRunRecord>,
    pub artifacts: PersistedArtifacts,
}

pub trait ComputeKernelExecutor: Send + Sync {
    fn execute_route(
        &self,
        route: &ExecutionRoutePlan,
        request: &TaskRequest,
        backend: ComputeBackend,
        on_progress: &mut dyn FnMut(ProgressTelemetryEvent),
    ) -> Result<ExecutionOutcome, RuntimeError>;

    fn clear_cuda_cache(&self);
}

// -----------------------------------------------------------------------------
// 6. План маршрутизации (Задача -> Алгоритм -> Решатели -> Бэкенд)
// -----------------------------------------------------------------------------

#[derive(Debug, Clone)]
pub struct ExecutionRoutePlan {
    pub algorithm_tag: &'static str,
    pub worker: GuiWorkerType,
    pub solvers: Vec<SolverCode>,
    pub supports_gpu_fp64: bool,
    pub estimated_memory_bytes: usize,
    pub cpu_fallback_backend: ComputeBackend,
}

pub const MAX_MESH_NODES: usize = 3_000_000;
pub const BYTES_PER_TET_NODE: usize = 2500;

pub fn build_route_plan(request: &TaskRequest) -> Result<ExecutionRoutePlan, RuntimeError> {
    match request {
        TaskRequest::ConfigOrImport { params, payload } => {
            params.validate()?;
            let mem = match payload {
                ImportPayload::CsvMeasurements { rows } => rows.saturating_mul(64),
                ImportPayload::StlCadGeometry { triangles, .. } => triangles.saturating_mul(96),
                ImportPayload::JsonStructure => 4096,
            };
            Ok(ExecutionRoutePlan {
                algorithm_tag: "A-0_CONFIG_IMPORT",
                worker: GuiWorkerType::SyncConfigurationWorker,
                solvers: vec![],
                supports_gpu_fp64: false,
                estimated_memory_bytes: mem,
                cpu_fallback_backend: ComputeBackend::CpuNumpyBatch,
            })
        }

        TaskRequest::PhysicalField {
            params,
            kind,
            include_magnetic_b,
            grid_shape,
            ..
        } => {
            params.validate()?;
            let nodes = grid_shape[0]
                .saturating_mul(grid_shape[1])
                .saturating_mul(grid_shape[2]);
            if nodes > MAX_MESH_NODES {
                return Err(RuntimeError::MeshBudgetExceeded {
                    nodes,
                    max_nodes: MAX_MESH_NODES,
                });
            }
            let est_bytes = nodes.saturating_mul(6).saturating_mul(BYTES_PER_TET_NODE);

            match kind {
                PhysicalFieldKind::PotentialAndCapacitance => {
                    if params.device_type.is_memory_group() {
                        return Err(RuntimeError::UnsupportedPhysicsForFamily {
                            family: params.device_type,
                            reason: "Laplace 3D FEM is not applicable to memory state devices"
                                .into(),
                        });
                    }
                    let mut solvers = vec![SolverCode::S01FemLaplace3d];
                    if *include_magnetic_b {
                        solvers.push(SolverCode::S07BiotSavart);
                    }
                    Ok(ExecutionRoutePlan {
                        algorithm_tag: "A-1_FEM_LAPLACE",
                        worker: GuiWorkerType::SimulationWorker,
                        solvers,
                        supports_gpu_fp64: true,
                        estimated_memory_bytes: est_bytes,
                        cpu_fallback_backend: ComputeBackend::CpuScipySparse,
                    })
                }

                PhysicalFieldKind::Thermal => {
                    if params.device_type.is_memory_group() {
                        return Err(RuntimeError::UnsupportedPhysicsForFamily {
                            family: params.device_type,
                            reason: "Thermal 3D FEM is not applicable to memory state devices"
                                .into(),
                        });
                    }
                    Ok(ExecutionRoutePlan {
                        algorithm_tag: "A-1_FEM_THERMAL",
                        worker: GuiWorkerType::SimulationWorker,
                        solvers: vec![SolverCode::S02FemThermal3d],
                        supports_gpu_fp64: true,
                        estimated_memory_bytes: est_bytes,
                        cpu_fallback_backend: ComputeBackend::CpuScipySparse,
                    })
                }

                PhysicalFieldKind::CarrierConcentration => {
                    if !params.device_type.supports_quantum_confinement() {
                        return Err(RuntimeError::UnsupportedPhysicsForFamily {
                            family: params.device_type,
                            reason: "no carrier solver for this family".into(),
                        });
                    }
                    let solver = if grid_shape[1] <= 1 {
                        SolverCode::S03QuantumPoisson2d
                    } else {
                        SolverCode::S04QuantumPoisson3d
                    };
                    Ok(ExecutionRoutePlan {
                        algorithm_tag: "A-2_QUANTUM_POISSON",
                        worker: GuiWorkerType::ChargeWorker,
                        solvers: vec![solver],
                        supports_gpu_fp64: true,
                        estimated_memory_bytes: est_bytes,
                        cpu_fallback_backend: ComputeBackend::CpuScipySparse,
                    })
                }
            }
        }

        TaskRequest::Characteristics {
            params,
            surrogate_available,
            surrogate_in_confidence_zone,
            ..
        } => {
            params.validate()?;
            if params.device_type.is_mos_group() {
                if *surrogate_available && *surrogate_in_confidence_zone {
                    let surrogate = match params.device_type {
                        DeviceFamily::PlanarMosfet | DeviceFamily::FdSoi => {
                            SolverCode::S12UnetSurrogate
                        }
                        _ => SolverCode::S13GnnSurrogate,
                    };
                    Ok(ExecutionRoutePlan {
                        algorithm_tag: "A-7/A-8+A-3_HYBRID_PDE",
                        worker: GuiWorkerType::NeuralPhysicsWorker,
                        solvers: vec![surrogate, SolverCode::S05ReducedPde1d],
                        supports_gpu_fp64: true,
                        estimated_memory_bytes: 32 * 1024 * 1024,
                        cpu_fallback_backend: ComputeBackend::CpuNumpyBatch,
                    })
                } else {
                    Ok(ExecutionRoutePlan {
                        algorithm_tag: "A-3_REDUCED_PDE",
                        worker: GuiWorkerType::CharacteristicsWorker,
                        solvers: vec![SolverCode::S05ReducedPde1d],
                        supports_gpu_fp64: true,
                        estimated_memory_bytes: 16 * 1024 * 1024,
                        cpu_fallback_backend: ComputeBackend::CpuNumpyBatch,
                    })
                }
            } else if params.device_type.is_memory_group() {
                Ok(ExecutionRoutePlan {
                    algorithm_tag: "A-4_MEMORY_STATE_ODE",
                    worker: GuiWorkerType::CharacteristicsWorker,
                    solvers: vec![SolverCode::S06MemStateOde],
                    supports_gpu_fp64: false, // CPU-only по A-4 / A-9
                    estimated_memory_bytes: 4 * 1024 * 1024,
                    cpu_fallback_backend: ComputeBackend::CpuStateOde,
                })
            } else {
                Err(RuntimeError::UnsupportedPhysicsForFamily {
                    family: params.device_type,
                    reason: "Characteristics generation is unsupported; import measurements and compute curve metrics (A-5)".into(),
                })
            }
        }

        TaskRequest::ModelExtraction {
            params,
            run_two_stage_verification,
            ..
        } => {
            params.validate()?;
            let mut solvers = match params.device_type {
                DeviceFamily::PlanarMosfet | DeviceFamily::FdSoi => {
                    vec![SolverCode::S08LeastSquaresBsim4]
                }
                DeviceFamily::Finfet
                | DeviceFamily::GaaNanosheet
                | DeviceFamily::GaaNanowire => vec![SolverCode::S09LeastSquaresCmg],
                DeviceFamily::Memristor | DeviceFamily::Memtransistor => {
                    vec![SolverCode::S11VerilogAOpenVaf]
                }
                DeviceFamily::Tfet | DeviceFamily::Hemt | DeviceFamily::CadCustom => {
                    vec![SolverCode::S10BehavioralSubckt]
                }
            };
            let tag = if *run_two_stage_verification {
                solvers.push(SolverCode::S11VerilogAOpenVaf);
                "A-5+A-6_EXTRACT_AND_VERIFY"
            } else {
                "A-5_COMPACT_EXTRACTION"
            };
            Ok(ExecutionRoutePlan {
                algorithm_tag: tag,
                worker: GuiWorkerType::ExtractionWorker,
                solvers,
                supports_gpu_fp64: false,
                estimated_memory_bytes: 24 * 1024 * 1024,
                cpu_fallback_backend: ComputeBackend::ExternalEdaPipeline,
            })
        }
    }
}

// -----------------------------------------------------------------------------
// 7. Гибридный диспетчер Runtime (Политика GPU/CPU, fallback, персистенция)
// -----------------------------------------------------------------------------

pub struct HybridRuntimeDispatcher {
    hw: HardwareEnvironment,
    executor: Arc<dyn ComputeKernelExecutor>,
}

impl HybridRuntimeDispatcher {
    pub fn new(hw: HardwareEnvironment, executor: Arc<dyn ComputeKernelExecutor>) -> Self {
        Self { hw, executor }
    }

    fn prefer_gpu_flag(request: &TaskRequest) -> bool {
        match request {
            TaskRequest::ConfigOrImport { .. } => false,
            TaskRequest::PhysicalField { prefer_gpu, .. } => *prefer_gpu,
            TaskRequest::Characteristics { prefer_gpu, .. } => *prefer_gpu,
            TaskRequest::ModelExtraction { prefer_gpu, .. } => *prefer_gpu,
        }
    }

    fn select_initial_backend(
        &self,
        plan: &ExecutionRoutePlan,
        prefer_gpu: bool,
    ) -> (ComputeBackend, Option<String>) {
        if !plan.supports_gpu_fp64 || !prefer_gpu {
            return (plan.cpu_fallback_backend, None);
        }
        if !self.hw.cuda_available || !self.hw.fp64_supported {
            return (
                plan.cpu_fallback_backend,
                Some("CUDA FP64 is unavailable on this host; routed to CPU".into()),
            );
        }
        if plan.estimated_memory_bytes > self.hw.free_vram_bytes {
            return (
                plan.cpu_fallback_backend,
                Some(format!(
                    "Insufficient VRAM ({} MB required > {} MB free); routed to CPU",
                    plan.estimated_memory_bytes / (1024 * 1024),
                    self.hw.free_vram_bytes / (1024 * 1024)
                )),
            );
        }
        (ComputeBackend::CudaPyTorchFp64, None)
    }

    pub fn dispatch<F>(
        &self,
        request: TaskRequest,
        mut on_progress: F,
    ) -> SolverTaskRecord
    where
        F: FnMut(ProgressTelemetryEvent),
    {
        let started = Instant::now();

        let plan = match build_route_plan(&request) {
            Ok(p) => p,
            Err(err) => {
                let code = match &err {
                    RuntimeError::DomainValidation(_) => "DEVICE_INVARIANT_VIOLATED",
                    RuntimeError::UnsupportedPhysicsForFamily { .. } => "UNSUPPORTED_DEVICE_FAMILY",
                    RuntimeError::MeshBudgetExceeded { .. } => "MESH_BUDGET_EXCEEDED",
                    RuntimeError::CudaOutOfMemory { .. } => "VRAM_OUT_OF_MEMORY",
                    RuntimeError::SolverExecutionFailed(_) => "SOLVER_EXECUTION_FAILED",
                };
                return SolverTaskRecord {
                    status: TaskPersistenceStatus::Failed,
                    worker: GuiWorkerType::SyncConfigurationWorker,
                    algorithm_tag: "A-9_ROUTING_REJECTED",
                    solvers_used: vec![],
                    backend_used: ComputeBackend::CpuNumpyBatch,
                    fallback_reason: None,
                    iterations: 0,
                    final_residual: f64::NAN,
                    error_code: Some(code),
                    error_message: Some(err.to_string()),
                    elapsed_ms: started.elapsed().as_millis(),
                    iteration_log: vec![],
                    neural_run: None,
                    artifacts: PersistedArtifacts {
                        device_structures_updated: false,
                        field_data_written: false,
                        curve_metrics_written: false,
                        spice_models_written: false,
                        verification_passed: None,
                    },
                };
            }
        };

        let prefer_gpu = Self::prefer_gpu_flag(&request);
        let (mut active_backend, mut fallback_reason) =
            self.select_initial_backend(&plan, prefer_gpu);

        on_progress(ProgressTelemetryEvent {
            progress_pct: 5,
            stage: format!("Routed {} via {}", plan.algorithm_tag, active_backend.as_str()),
            backend: active_backend,
            vram_used_mb: if active_backend == ComputeBackend::CudaPyTorchFp64 {
                plan.estimated_memory_bytes as f64 / (1024.0 * 1024.0)
            } else {
                0.0
            },
        });

        let exec_result =
            self.executor
                .execute_route(&plan, &request, active_backend, &mut on_progress);

        let outcome = match exec_result {
            Ok(out) => out,
            Err(RuntimeError::CudaOutOfMemory {
                required_bytes,
                available_bytes,
            }) if active_backend == ComputeBackend::CudaPyTorchFp64 => {
                // NFR-REL-03: перехват CUDA OOM -> очистка кэша -> откат на CPU -> уведомление GUI
                self.executor.clear_cuda_cache();
                active_backend = plan.cpu_fallback_backend;
                let reason = format!(
                    "CUDA OutOfMemory during execution ({} > {} bytes); switched to {}",
                    required_bytes,
                    available_bytes,
                    active_backend.as_str()
                );
                fallback_reason = Some(reason.clone());

                on_progress(ProgressTelemetryEvent {
                    progress_pct: 15,
                    stage: reason,
                    backend: active_backend,
                    vram_used_mb: 0.0,
                });

                match self.executor.execute_route(
                    &plan,
                    &request,
                    active_backend,
                    &mut on_progress,
                ) {
                    Ok(out) => out,
                    Err(err) => {
                        return SolverTaskRecord {
                            status: TaskPersistenceStatus::Failed,
                            worker: plan.worker,
                            algorithm_tag: plan.algorithm_tag,
                            solvers_used: plan.solvers,
                            backend_used: active_backend,
                            fallback_reason,
                            iterations: 0,
                            final_residual: f64::NAN,
                            error_code: Some("SOLVER_EXECUTION_FAILED"),
                            error_message: Some(err.to_string()),
                            elapsed_ms: started.elapsed().as_millis(),
                            iteration_log: vec![],
                            neural_run: None,
                            artifacts: PersistedArtifacts {
                                device_structures_updated: false,
                                field_data_written: false,
                                curve_metrics_written: false,
                                spice_models_written: false,
                                verification_passed: None,
                            },
                        };
                    }
                }
            }
            Err(err) => {
                return SolverTaskRecord {
                    status: TaskPersistenceStatus::Failed,
                    worker: plan.worker,
                    algorithm_tag: plan.algorithm_tag,
                    solvers_used: plan.solvers,
                    backend_used: active_backend,
                    fallback_reason,
                    iterations: 0,
                    final_residual: f64::NAN,
                    error_code: Some("SOLVER_EXECUTION_FAILED"),
                    error_message: Some(err.to_string()),
                    elapsed_ms: started.elapsed().as_millis(),
                    iteration_log: vec![],
                    neural_run: None,
                    artifacts: PersistedArtifacts {
                        device_structures_updated: false,
                        field_data_written: false,
                        curve_metrics_written: false,
                        spice_models_written: false,
                        verification_passed: None,
                    },
                };
            }
        };

        on_progress(ProgressTelemetryEvent {
            progress_pct: 100,
            stage: "Completed and persisted".into(),
            backend: active_backend,
            vram_used_mb: 0.0,
        });

        SolverTaskRecord {
            status: TaskPersistenceStatus::Ok,
            worker: plan.worker,
            algorithm_tag: plan.algorithm_tag,
            solvers_used: plan.solvers,
            backend_used: active_backend,
            fallback_reason,
            iterations: outcome.iterations,
            final_residual: outcome.final_residual,
            error_code: None,
            error_message: None,
            elapsed_ms: started.elapsed().as_millis(),
            iteration_log: outcome.iteration_log,
            neural_run: outcome.neural_run,
            artifacts: outcome.artifacts,
        }
    }
}

// -----------------------------------------------------------------------------
// 8. Модульные тесты маршрутизации и отказоустойчивости
// -----------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicBool, Ordering};

    struct MockExecutor {
        simulate_cuda_oom: bool,
        cache_cleared: Arc<AtomicBool>,
    }

    impl ComputeKernelExecutor for MockExecutor {
        fn execute_route(
            &self,
            route: &ExecutionRoutePlan,
            _request: &TaskRequest,
            backend: ComputeBackend,
            on_progress: &mut dyn FnMut(ProgressTelemetryEvent),
        ) -> Result<ExecutionOutcome, RuntimeError> {
            if self.simulate_cuda_oom && backend == ComputeBackend::CudaPyTorchFp64 {
                return Err(RuntimeError::CudaOutOfMemory {
                    required_bytes: 16 * 1024 * 1024 * 1024,
                    available_bytes: 8 * 1024 * 1024 * 1024,
                });
            }
            on_progress(ProgressTelemetryEvent {
                progress_pct: 60,
                stage: "Solving".into(),
                backend,
                vram_used_mb: 128.0,
            });
            let is_hybrid = route.solvers.len() > 1
                && (route.solvers[0] == SolverCode::S12UnetSurrogate
                    || route.solvers[0] == SolverCode::S13GnnSurrogate);
            Ok(ExecutionOutcome {
                iterations: if is_hybrid { 4 } else { 120 },
                final_residual: 1e-11,
                iteration_log: vec![IterationRecord {
                    step_no: 1,
                    residual: 1e-11,
                    dt_s: 0.01,
                    newton_iters: if is_hybrid { 4 } else { 120 },
                }],
                neural_run: if is_hybrid {
                    Some(NeuralRunRecord {
                        surrogate_solver: route.solvers[0],
                        warm_start_residual: 2e-3,
                        iters_saved: 116,
                    })
                } else {
                    None
                },
                artifacts: PersistedArtifacts {
                    device_structures_updated: false,
                    field_data_written: true,
                    curve_metrics_written: true,
                    spice_models_written: false,
                    verification_passed: None,
                },
            })
        }

        fn clear_cuda_cache(&self) {
            self.cache_cleared.store(true, Ordering::SeqCst);
        }
    }

    fn base_params(family: DeviceFamily) -> DeviceParameters {
        DeviceParameters {
            name: "test_dev".into(),
            device_type: family,
            epsilon_r: 3.9,
            electron_mobility: 1400.0,
            hole_mobility: 450.0,
            bandgap: 1.12,
            doping: 1e17,
            tox_nm: 1.2,
            width_um: 1.0,
            length_um: 0.05,
            temperature_k: 300.0,
            geometry_fields: HashMap::new(),
        }
    }

    #[test]
    fn test_hybrid_warm_start_routing() {
        let cleared = Arc::new(AtomicBool::new(false));
        let exec = Arc::new(MockExecutor {
            simulate_cuda_oom: false,
            cache_cleared: cleared,
        });
        let dispatcher = HybridRuntimeDispatcher::new(
            HardwareEnvironment {
                cuda_available: true,
                fp64_supported: true,
                free_vram_bytes: 16 * 1024 * 1024 * 1024,
            },
            exec,
        );

        let record = dispatcher.dispatch(
            TaskRequest::Characteristics {
                params: base_params(DeviceFamily::Finfet),
                surrogate_available: true,
                surrogate_in_confidence_zone: true,
                prefer_gpu: true,
            },
            |_| {},
        );

        assert_eq!(record.status, TaskPersistenceStatus::Ok);
        assert_eq!(
            record.solvers_used,
            vec![SolverCode::S13GnnSurrogate, SolverCode::S05ReducedPde1d]
        );
        assert_eq!(record.iterations, 4);
        assert!(record.neural_run.is_some());
    }

    #[test]
    fn test_cuda_oom_triggers_cpu_fallback() {
        let cleared = Arc::new(AtomicBool::new(false));
        let exec = Arc::new(MockExecutor {
            simulate_cuda_oom: true,
            cache_cleared: cleared.clone(),
        });
        let dispatcher = HybridRuntimeDispatcher::new(
            HardwareEnvironment {
                cuda_available: true,
                fp64_supported: true,
                free_vram_bytes: 24 * 1024 * 1024 * 1024,
            },
            exec,
        );

        let record = dispatcher.dispatch(
            TaskRequest::PhysicalField {
                params: base_params(DeviceFamily::PlanarMosfet),
                kind: PhysicalFieldKind::PotentialAndCapacitance,
                include_magnetic_b: true,
                grid_shape: [20, 20, 20],
                prefer_gpu: true,
            },
            |_| {},
        );

        assert_eq!(record.status, TaskPersistenceStatus::Ok);
        assert_eq!(record.backend_used, ComputeBackend::CpuScipySparse);
        assert!(cleared.load(Ordering::SeqCst));
        assert!(record.fallback_reason.is_some());
    }

    #[test]
    fn test_rejects_carrier_solver_for_tfet() {
        let cleared = Arc::new(AtomicBool::new(false));
        let exec = Arc::new(MockExecutor {
            simulate_cuda_oom: false,
            cache_cleared: cleared,
        });
        let dispatcher = HybridRuntimeDispatcher::new(
            HardwareEnvironment {
                cuda_available: false,
                fp64_supported: false,
                free_vram_bytes: 0,
            },
            exec,
        );

        let record = dispatcher.dispatch(
            TaskRequest::PhysicalField {
                params: base_params(DeviceFamily::Tfet),
                kind: PhysicalFieldKind::CarrierConcentration,
                include_magnetic_b: false,
                grid_shape: [20, 1, 20],
                prefer_gpu: false,
            },
            |_| {},
        );

        assert_eq!(record.status, TaskPersistenceStatus::Failed);
        assert_eq!(record.error_code, Some("UNSUPPORTED_DEVICE_FAMILY"));
    }
}
