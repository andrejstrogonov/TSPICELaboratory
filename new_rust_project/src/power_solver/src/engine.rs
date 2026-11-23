//! Единый слой запуска (facade) поверх вычислительных алгоритмов A-1..A-6.
//! Принимает единый `domain::Device` и маппит его в типы конкретных решателей,
//! возвращая унифицированные результаты. Сокращает дублирование кода GUI/моста.

use crate::domain::{CurveResult, Device, DomainError, FieldResult, SweepConfig};
use std::collections::BTreeMap;
use std::collections::HashMap;

// -----------------------------------------------------------------------------
// Конвертеры `domain::Device` -> параметры конкретных решателей
// -----------------------------------------------------------------------------

impl Device {
    pub(crate) fn to_fem(&self) -> crate::fem::DeviceParameters {
        crate::fem::DeviceParameters {
            device_type: self.kind.as_str().to_string(),
            width_um: self.width_um,
            length_um: self.length_um,
            tox_nm: self.tox_nm,
            body_thickness_nm: self.body_thickness_nm,
            substrate_thickness_um: self.substrate_thickness_um,
            epsilon_r: self.epsilon_r,
            conductivity_s_m: self.conductivity_s_m,
            thermal_conductivity_w_mk: self.thermal_conductivity_w_mk,
            ambient_temp_k: self.ambient_temp_k,
        }
    }

    pub(crate) fn to_quantum(&self, effective_mass_ratio: f64) -> crate::quantum_poisson::QuantumDeviceParams {
        crate::quantum_poisson::QuantumDeviceParams {
            family: quantum_family(self.kind),
            length_um: self.length_um,
            width_um: self.effective_width_um(),
            body_thickness_nm: self.body_thickness_nm,
            tox_nm: self.tox_nm,
            epsilon_r_body: self.epsilon_r,
            epsilon_r_ox: self.epsilon_r_ox,
            m_eff_conf: effective_mass_ratio,
            m_eff_dos: self.m_eff_dos,
            valley_degeneracy: self.valley_degeneracy,
            nd_cm3: self.nd_cm3,
            na_cm3: self.na_cm3,
            ni_cm3: self.ni_cm3,
            temperature_k: self.temperature_k,
            flatband_voltage_v: self.v_fb,
        }
    }

    pub(crate) fn to_reduced_pde(&self) -> crate::reduced_pde::DeviceParameters {
        crate::reduced_pde::DeviceParameters {
            device_type: reduced_pde_family(self.kind).unwrap_or(crate::reduced_pde::DeviceType::PlanarMosfet),
            epsilon_ox: self.epsilon_r_ox,
            epsilon_semi: self.epsilon_r,
            electron_mobility_cm2_vs: self.electron_mobility_cm2_vs,
            bandgap_ev: self.bandgap_ev,
            doping_cm3: self.doping_cm3,
            tox_nm: self.tox_nm,
            body_thickness_nm: self.body_thickness_nm,
            effective_width_um: self.effective_width_um(),
            length_um: self.length_um,
            temperature_k: self.temperature_k,
            v_fb: self.v_fb,
        }
    }

    pub(crate) fn to_memory(&self) -> crate::memory_state::MemoryDeviceParameters {
        crate::memory_state::MemoryDeviceParameters {
            device_type: match self.kind {
                crate::domain::DeviceKind::Memtransistor => crate::memory_state::MemoryDeviceType::Memtransistor,
                _ => crate::memory_state::MemoryDeviceType::Memristor,
            },
            temperature_k: self.temperature_k,
            active_layer_nm: self.active_layer_nm,
            tox_nm: self.tox_nm,
            epsilon_r: self.epsilon_r_ox,
            electron_mobility_cm2_vs: self.electron_mobility_cm2_vs,
            width_um: self.width_um,
            length_um: self.length_um,
            ion_mobility_cm2_vs: self.ion_mobility_cm2_vs,
            low_resistance_ohm: self.low_resistance_ohm,
            high_resistance_ohm: self.high_resistance_ohm,
            set_voltage_v: self.set_voltage_v,
            reset_voltage_v: self.reset_voltage_v,
            activation_energy_ev: self.activation_energy_ev,
            initial_state: self.initial_state,
            state_relaxation_ms: self.state_relaxation_ms,
            gate_coupling: self.gate_coupling,
        }
    }

    pub(crate) fn to_extraction(&self) -> crate::extraction::DeviceParameters {
        crate::extraction::DeviceParameters {
            name: self.name.clone(),
            device_type: extraction_family(self.kind)
                .unwrap_or(crate::extraction::DeviceType::CadCustom),
            width_um: self.width_um,
            length_um: self.length_um,
            tox_nm: self.tox_nm,
            epsilon_r: self.epsilon_r,
            electron_mobility_cm2_vs: self.electron_mobility_cm2_vs,
            doping_cm3: self.doping_cm3,
            temperature_k: self.temperature_k,
            w_eff_um: Some(self.effective_width_um()),
            buried_oxide_nm: self.buried_oxide_nm,
        }
    }
}

fn quantum_family(kind: crate::domain::DeviceKind) -> crate::quantum_poisson::DeviceFamily {
    use crate::quantum_poisson::DeviceFamily as QF;
    use crate::domain::DeviceKind as K;
    match kind {
        K::PlanarMosfet => QF::PlanarMosfet,
        K::FdSoi => QF::FdSoi,
        K::Finfet => QF::FinFet,
        K::GaaNanosheet => QF::GaaNanosheet,
        K::GaaNanowire => QF::GaaNanowire,
        K::Tfet => QF::Tfet,
        K::Hemt => QF::Hemt,
        K::Memristor => QF::Memristor,
        K::Memtransistor => QF::Memtransistor,
        K::CadCustom => QF::CadCustom,
    }
}

fn reduced_pde_family(kind: crate::domain::DeviceKind) -> Option<crate::reduced_pde::DeviceType> {
    use crate::reduced_pde::DeviceType as RT;
    use crate::domain::DeviceKind as K;
    Some(match kind {
        K::PlanarMosfet => RT::PlanarMosfet,
        K::FdSoi => RT::FdSoi,
        K::Finfet => RT::Finfet,
        K::GaaNanosheet => RT::GaaNanosheet,
        K::GaaNanowire => RT::GaaNanowire,
        K::Tfet => RT::Tfet,
        K::Hemt => RT::Hemt,
        K::CadCustom => RT::CadCustom,
        K::Memristor | K::Memtransistor => return None,
    })
}

fn extraction_family(kind: crate::domain::DeviceKind) -> Option<crate::extraction::DeviceType> {
    use crate::extraction::DeviceType as ET;
    use crate::domain::DeviceKind as K;
    match kind {
        K::PlanarMosfet => Some(ET::PlanarMosfet),
        K::FdSoi => Some(ET::FdSoi),
        K::Finfet => Some(ET::Finfet),
        K::GaaNanosheet => Some(ET::GaaNanosheet),
        K::GaaNanowire => Some(ET::GaaNanowire),
        K::Tfet => Some(ET::Tfet),
        K::Hemt => Some(ET::Hemt),
        K::CadCustom => Some(ET::CadCustom),
        K::Memristor | K::Memtransistor => None,
    }
}

// -----------------------------------------------------------------------------
// Высокоуровневые функции запуска решателей
// -----------------------------------------------------------------------------

/// A-2 · Самосогласованный Quantum-Poisson (замена solve_quantum_charge_distribution из Python).
pub fn solve_quantum_charge_distribution(
    device: &Device,
    gate_voltage: f64,
    drain_voltage: f64,
    dimension: &str,
    grid_shape: (usize, usize, usize),
    effective_mass_ratio: f64,
    use_gpu: bool,
) -> Result<FieldResult, DomainError> {
    if !device.kind.supports_quantum() {
        return Err(DomainError::Unsupported(format!(
            "Quantum-Poisson не применим к семейству {}",
            device.kind
        )));
    }
    let (nx, ny, nz) = grid_shape;
    if nx * ny * nz > 300_000 {
        return Err(DomainError::Validation("Слишком крупная сетка Quantum-Poisson (лимит 300k узлов)".into()));
    }
    let mode = if dimension.eq_ignore_ascii_case("3d") {
        crate::quantum_poisson::QuantumGridMode::Mode3D { nx, ny, nz }
    } else {
        crate::quantum_poisson::QuantumGridMode::Mode2D { nx, nz }
    };
    let params = device.to_quantum(effective_mass_ratio);
    let cfg = crate::quantum_poisson::QuantumPoissonConfig {
        mode,
        v_gate: gate_voltage,
        v_drain: drain_voltage,
        v_source: 0.0,
        use_gpu,
        ..Default::default()
    };

    let solution =
        crate::quantum_poisson::solve_quantum_charge_distribution(&params, &cfg, None)
            .map_err(|e| DomainError::Unsupported(e.to_string()))?;

    let (sx, sy, sz) = solution.shape;
    let max_den = solution.values.iter().cloned().fold(0.0, f64::max);
    let mut metrics = HashMap::new();
    metrics.insert("QUANTUM_ITERATIONS".into(), solution.convergence_history.len() as f64);
    metrics.insert("MAX_ELECTRON_DENSITY_CM3".into(), max_den);
    metrics.insert("CONVERGED".into(), if solution.converged { 1.0 } else { 0.0 });

    let backend = match solution.backend {
        crate::quantum_poisson::ComputeBackend::CudaFp64 => "CUDA FP64 (Rust stencil-CG)".into(),
        crate::quantum_poisson::ComputeBackend::CpuFallback => "CPU (Rust Newton-CG)".into(),
    };

    Ok(FieldResult {
        values: solution.values,
        shape: [sx, sy, sz],
        backend,
        metrics,
        description: "Самосогласованный Quantum-Poisson (подзоны Шрёдингера + Fermi–Dirac)".into(),
    })
}

/// A-3 · Reduced-order PDE: характеристики Id–Vg / Id–Vd / Cgg (для MOS-семейств).
pub fn solve_pde_characteristics(
    device: &Device,
    cfg: &SweepConfig,
) -> Result<(Vec<CurveResult>, String, BTreeMap<String, f64>), DomainError> {
    if !device.kind.is_mos() {
        return Err(DomainError::Unsupported(format!(
            "Reduced-order PDE не применим к семейству {}",
            device.kind
        )));
    }
    let params = device.to_reduced_pde();
    let config = crate::reduced_pde::ReducedPdeConfig {
        vg_min_v: cfg.vg_min_v,
        vg_max_v: cfg.vg_max_v,
        vd_linear_v: cfg.vd_linear_v,
        vd_sat_v: cfg.vd_sat_v,
        vd_min_v: cfg.vd_min_v,
        vd_max_v: cfg.vd_max_v,
        gate_points: cfg.gate_points,
        drain_points: cfg.drain_points,
        channel_points: cfg.channel_points,
        poisson_nodes: cfg.poisson_nodes,
        output_curve_count: 4,
        use_gpu: cfg.use_gpu,
    };
    let solver = crate::reduced_pde::CpuBatchPoissonSolver;
    let result = crate::reduced_pde::solve_pde_characteristics(&params, &config, &solver)
        .map_err(|e| DomainError::Unsupported(e.to_string()))?;

    let curves = result
        .curves
        .into_iter()
        .map(|c| CurveResult {
            kind: c.spec_id,
            label: c.label,
            x_name: c.x_name,
            y_name: c.y_name,
            x: c.x,
            y: c.y,
        })
        .collect();

    let mut diagnostics = BTreeMap::new();
    for (k, v) in result.diagnostics {
        diagnostics.insert(k, v);
    }
    Ok((curves, result.backend, diagnostics))
}

/// A-4 · State-ODE характеристики памяти (мемристор / мемтранзистор).
pub fn solve_memory_characteristics(
    device: &Device,
) -> Result<(Vec<CurveResult>, HashMap<String, f64>, String), DomainError> {
    if !device.kind.is_memory() {
        return Err(DomainError::Unsupported(format!(
            "State-ODE модель памяти не применима к семейству {}",
            device.kind
        )));
    }
    let params = device.to_memory();
    let config = crate::memory_state::MemorySweepConfig::default();
    let result = crate::memory_state::solve_memory_characteristics(&params, &config)
        .map_err(|e| DomainError::Unsupported(e.to_string()))?;

    let curves = result
        .curves
        .into_iter()
        .map(|c| CurveResult {
            kind: c.spec_id,
            label: c.label,
            x_name: c.x_field,
            y_name: c.y_field,
            x: c.x,
            y: c.y,
        })
        .collect();

    Ok((curves, result.summary_metrics, result.backend))
}

/// A-1 · Стационарный тепловой FEM-расчёт (FR-EXT-08).
pub fn solve_steady_thermal(
    device: &Device,
    drain_voltage_v: f64,
    drain_current_a: f64,
    grid_shape: [usize; 3],
) -> Result<(FieldResult, HashMap<String, f64>), DomainError> {
    let params = device.to_fem();
    let op = crate::fem::OperatingPoint {
        v_source: 0.0,
        v_drain: drain_voltage_v,
        v_gate: 0.0,
        i_drain: drain_current_a,
    };
    let hw = crate::fem::HardwareCapabilities::default();
    let result = crate::fem::solve_steady_thermal(&params, &op, grid_shape, &hw)
        .map_err(|e| DomainError::Unsupported(e.to_string()))?;

    let mut metrics = HashMap::new();
    metrics.insert("POWER_W".into(), (drain_voltage_v * drain_current_a).abs());
    metrics.insert("TMAX_K".into(), result.t_max_k);
    metrics.insert("DELTA_TMAX_K".into(), result.delta_t_k);
    metrics.insert("RTH_K_W".into(), result.r_th_k_w);

    let backend = match result.temperature.backend {
        crate::fem::ComputeBackend::CudaFp64 => "CUDA FP64".into(),
        crate::fem::ComputeBackend::CpuScipy => "CPU (sprs/CG)".into(),
    };
    Ok((
        FieldResult {
            values: result.temperature.values,
            shape: result.temperature.shape,
            backend,
            metrics: metrics.clone(),
            description: "Стационарное 3D уравнение теплопроводности (P = Vds·Id)".into(),
        },
        metrics,
    ))
}

/// A-1 · Квазистатическое электрическое поле (Лаплас + Био–Савар).
pub fn solve_quasistatic_electric_field(
    device: &Device,
    gate_voltage: f64,
    drain_voltage: f64,
    drain_current_a: f64,
    grid_shape: [usize; 3],
) -> Result<(FieldResult, FieldResult, FieldResult), DomainError> {
    let params = device.to_fem();
    let op = crate::fem::OperatingPoint {
        v_source: 0.0,
        v_drain: drain_voltage,
        v_gate: gate_voltage,
        i_drain: drain_current_a,
    };
    let hw = crate::fem::HardwareCapabilities::default();
    let result = crate::fem::solve_quasistatic_electric_field(&params, &op, grid_shape, &hw)
        .map_err(|e| DomainError::Unsupported(e.to_string()))?;

    let backend = match result.potential.backend {
        crate::fem::ComputeBackend::CudaFp64 => "CUDA FP64".into(),
        crate::fem::ComputeBackend::CpuScipy => "CPU (sprs/CG)".into(),
    };
    let mut metrics = HashMap::new();
    metrics.insert("CGEO_F".into(), result.capacitance_f);
    metrics.insert("ELECTROSTATIC_ENERGY_J".into(), result.stored_energy_j);
    metrics.insert(
        "BMAX_EST_T".into(),
        result
            .magnetic_field_mag
            .values
            .iter()
            .cloned()
            .fold(0.0, f64::max),
    );
    metrics.insert("EMAX_V_M".into(), result.electric_field_mag.values.iter().cloned().fold(0.0, f64::max));

    Ok((
        FieldResult {
            values: result.potential.values,
            shape: result.potential.shape,
            backend: backend.clone(),
            metrics: metrics.clone(),
            description: "Квазистатическое 3D уравнение Лапласа (границы Дирихле)".into(),
        },
        FieldResult {
            values: result.electric_field_mag.values,
            shape: result.electric_field_mag.shape,
            backend: backend.clone(),
            metrics: metrics.clone(),
            description: "Модуль напряжённости электрического поля |E(x,y,z)|".into(),
        },
        FieldResult {
            values: result.magnetic_field_mag.values,
            shape: result.magnetic_field_mag.shape,
            backend,
            metrics,
            description: "Аналитическая карта магнитной индукции по Био–Савару–Лапласу".into(),
        },
    ))
}

/// A-5 · Экстракция компактной модели (BSIM4 / BSIM-CMG / поведенческая) + SPICE/JSON.
pub fn extract_compact_model(
    device: &Device,
    sweeps: &[crate::extraction::IvSweep],
    target: crate::extraction::TargetSimulator,
) -> Result<crate::extraction::SpiceModelArtifact, DomainError> {
    let params = device.to_extraction();
    let input = crate::extraction::ExtractionInput {
        id_vg_graphs: sweeps.to_vec(),
        ..Default::default()
    };
    crate::extraction::extract_compact_model(&params, &input, target)
        .map_err(|e| DomainError::Unsupported(e.to_string()))
}

/// A-6 · Генерация Verilog-A для мемристивных моделей (FR-EXT-06).
pub fn generate_memory_verilog_a(
    device: &Device,
) -> Result<String, DomainError> {
    if !device.kind.is_memory() {
        return Err(DomainError::Unsupported("Verilog-A память доступна только для memristor/memtransistor".into()));
    }
    let is_mt = device.kind == crate::domain::DeviceKind::Memtransistor;
    let params = crate::verification::MemoryVerilogAParams {
        ron_ohm: device.low_resistance_ohm,
        roff_ohm: device.high_resistance_ohm,
        vset_v: device.set_voltage_v,
        vreset_v: device.reset_voltage_v,
        mu_ion_cm2_vs: device.ion_mobility_cm2_vs,
        active_layer_nm: device.active_layer_nm,
        activation_energy_ev: device.activation_energy_ev,
        tnom_k: device.temperature_k,
        x_init: device.initial_state,
        gate_coupling: device.gate_coupling,
        relaxation_ms: device.state_relaxation_ms,
        ..Default::default()
    };
    let (mod_name, source) = crate::verification::generate_memory_verilog_a(
        is_mt,
        crate::verification::MemoryVerilogAFlavor::DriftStrukov,
        &params,
    )
    .map_err(|e| DomainError::Unsupported(e.to_string()))?;

    Ok(format!("// Модуль: {mod_name}\n{source}"))
}

/// Конвертация кривой измерения из `io::Measurement` в sweep экстрактора.
pub fn measurement_to_idvg_sweeps(measurements: &[crate::io::Measurement]) -> Vec<crate::extraction::IvSweep> {
    measurements
        .iter()
        .filter(|m| m.kind == "idvg" || m.kind == "idvd")
        .map(|m| crate::extraction::IvSweep {
            v: m.x.clone(),
            i: m.y.clone(),
            bias_v: m
                .columns
                .get("vd")
                .and_then(|c| c.first())
                .copied()
                .unwrap_or(0.05),
            instance_length_um: None,
            instance_width_um: None,
        })
        .collect()
}
