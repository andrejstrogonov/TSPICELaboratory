//! Единая контентная модель предметной области (A-0 · D-10).
//! Устраняет дублирование типов DeviceParameters/ComputeBackend/DeviceFamily,
//! ранее размазанных по 5 модулям. Спецификация: REQUIREMENTS.md (FR-UI-01,
//! FR-EXT-01/02, FR-IO-01/02) и HOW_TO_RECOMMENDATIONS.md (cad-domain).

use std::collections::HashMap;
use std::fmt;

pub const EPSILON_0: f64 = 8.854_187_812_8e-12;
pub const BOLTZMANN: f64 = 1.380_649e-23;
pub const ELECTRON_CHARGE: f64 = 1.602_176_634e-19;

/// Полные идентификаторы семейств приборов (совместимо с Python `DEVICE_TYPES`).
pub const DEVICE_TYPES: [&str; 10] = [
    "planar_mosfet",
    "fd_soi",
    "finfet",
    "gaa_nanosheet",
    "gaa_nanowire",
    "tfet",
    "hemt",
    "memristor",
    "memtransistor",
    "cad_custom",
];

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
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
    pub const ALL: [DeviceKind; 10] = [
        DeviceKind::PlanarMosfet,
        DeviceKind::FdSoi,
        DeviceKind::Finfet,
        DeviceKind::GaaNanosheet,
        DeviceKind::GaaNanowire,
        DeviceKind::Tfet,
        DeviceKind::Hemt,
        DeviceKind::Memristor,
        DeviceKind::Memtransistor,
        DeviceKind::CadCustom,
    ];

    pub fn from_str_id(value: &str) -> Option<Self> {
        DEVICE_TYPES
            .iter()
            .position(|&id| id == value)
            .and_then(|i| DeviceKind::ALL.get(i).copied())
    }

    pub fn as_str(self) -> &'static str {
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

    /// MOS-группа (поддерживает FEM / reduced-order PDE / Quantum-Poisson).
    pub fn is_mos(self) -> bool {
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

    /// Группа памяти (state-ODE модели).
    pub fn is_memory(self) -> bool {
        matches!(self, Self::Memristor | Self::Memtransistor)
    }

    /// Поддержка самосогласованного Quantum-Poisson (FR-CORE-03, A-2).
    pub fn supports_quantum(self) -> bool {
        self.is_mos()
    }

    /// Поддержка exporting BSIM4 (только Planar Bulk MOSFET — FR-EXP-01).
    pub fn exports_bsim4(self) -> bool {
        matches!(self, Self::PlanarMosfet | Self::FdSoi)
    }
}

impl fmt::Display for DeviceKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

/// Единая структура параметров прибора (объединяет fem/quantum/reduced_pde/
/// memory/extraction версии; поля задаются с учётом семейства).
#[derive(Debug, Clone)]
pub struct Device {
    pub name: String,
    pub kind: DeviceKind,
    pub epsilon_r: f64,
    pub epsilon_r_ox: f64,
    pub electron_mobility_cm2_vs: f64,
    pub hole_mobility_cm2_vs: f64,
    pub bandgap_ev: f64,
    pub doping_cm3: f64,
    pub tox_nm: f64,
    pub width_um: f64,
    pub length_um: f64,
    pub body_thickness_nm: f64,
    pub substrate_thickness_um: f64,
    pub temperature_k: f64,
    pub geometry: HashMap<String, f64>,
    pub barrier_material: String,
    pub junction_material: String,
    pub v_fb: f64,
    // Физика памяти (state-ODE)
    pub low_resistance_ohm: f64,
    pub high_resistance_ohm: f64,
    pub set_voltage_v: f64,
    pub reset_voltage_v: f64,
    pub activation_energy_ev: f64,
    pub initial_state: f64,
    pub state_relaxation_ms: f64,
    pub gate_coupling: f64,
    pub ion_mobility_cm2_vs: f64,
    pub active_layer_nm: f64,
    // Quantum / стресс
    pub m_eff_conf: f64,
    pub m_eff_dos: f64,
    pub valley_degeneracy: f64,
    pub nd_cm3: f64,
    pub na_cm3: f64,
    pub ni_cm3: f64,
    // FEM
    pub thermal_conductivity_w_mk: f64,
    pub ambient_temp_k: f64,
    pub conductivity_s_m: f64,
    pub buried_oxide_nm: Option<f64>,
}

impl Device {
    pub fn new(name: impl Into<String>, kind: DeviceKind) -> Self {
        let geometry = match kind {
            DeviceKind::Finfet => HashMap::from([("fin_width_nm".into(), 8.0), ("fin_height_nm".into(), 40.0)]),
            DeviceKind::GaaNanosheet => HashMap::from([("sheet_thickness_nm".into(), 8.0), ("sheet_width_nm".into(), 40.0)]),
            _ => HashMap::new(),
        };
        Self {
            name: name.into(),
            kind,
            epsilon_r: 11.7,
            epsilon_r_ox: 3.9,
            electron_mobility_cm2_vs: 1000.0,
            hole_mobility_cm2_vs: 400.0,
            bandgap_ev: 1.12,
            doping_cm3: 1e16,
            tox_nm: 1.2,
            width_um: 1.0,
            length_um: 0.1,
            body_thickness_nm: 30.0,
            substrate_thickness_um: 200.0,
            temperature_k: 300.0,
            geometry,
            barrier_material: "AlGaN/GaN".into(),
            junction_material: "Si/Si".into(),
            v_fb: 0.0,
            low_resistance_ohm: 100.0,
            high_resistance_ohm: 100_000.0,
            set_voltage_v: 0.3,
            reset_voltage_v: -0.3,
            activation_energy_ev: 0.6,
            initial_state: 0.5,
            state_relaxation_ms: 100.0,
            gate_coupling: 2.0,
            ion_mobility_cm2_vs: 1e-10,
            active_layer_nm: 10.0,
            m_eff_conf: 0.916,
            m_eff_dos: 0.19,
            valley_degeneracy: 2.0,
            nd_cm3: 1e15,
            na_cm3: doping_default(),
            ni_cm3: 1e10,
            thermal_conductivity_w_mk: 130.0,
            ambient_temp_k: 300.0,
            conductivity_s_m: 1.0,
            buried_oxide_nm: None,
        }
    }

    pub fn validate(&self) -> Result<(), DomainError> {
        if self.name.trim().is_empty() {
            return Err(DomainError::Validation("Имя прибора не должно быть пустым".into()));
        }
        let pos = [
            ("epsilon_r", self.epsilon_r),
            ("electron_mobility_cm2_vs", self.electron_mobility_cm2_vs),
            ("bandgap_ev", self.bandgap_ev),
            ("doping_cm3", self.doping_cm3),
            ("tox_nm", self.tox_nm),
            ("width_um", self.width_um),
            ("length_um", self.length_um),
            ("temperature_k", self.temperature_k),
        ];
        for (field, v) in pos {
            if !(v.is_finite() && v > 0.0) {
                return Err(DomainError::Validation(format!(
                    "Поле {field} должно быть положительным конечным числом (получено {v})"
                )));
            }
        }
        if self.is_memory() {
            if self.high_resistance_ohm <= self.low_resistance_ohm {
                return Err(DomainError::Validation(
                    "R_off должно быть больше R_on (HRS > LRS)".into(),
                ));
            }
            if !(0.0..=1.0).contains(&self.initial_state) {
                return Err(DomainError::Validation(
                    "initial_state должен лежать в [0, 1]".into(),
                ));
            }
            if self.set_voltage_v <= 0.0 || self.reset_voltage_v >= 0.0 {
                return Err(DomainError::Validation(
                    "set_voltage_v > 0 и reset_voltage_v < 0".into(),
                ));
            }
        }
        if let Some(box_nm) = self.buried_oxide_nm {
            if box_nm <= 0.0 {
                return Err(DomainError::Validation("buried_oxide_nm должен быть > 0".into()));
            }
        }
        Ok(())
    }

    /// Эффективная ширина канала с учётом периметра многозатворных структур (model.py).
    pub fn effective_width_um(&self) -> f64 {
        match self.kind {
            DeviceKind::Finfet => {
                let nfin = self.geometry.get("fin_count").copied().unwrap_or(1.0) as usize;
                let hfin = self.geometry.get("fin_height_nm").copied().unwrap_or(40.0);
                let wfin = self.geometry.get("fin_width_nm").copied().unwrap_or(8.0);
                (nfin as f64) * (2.0 * hfin + wfin) * 1e-3
            }
            DeviceKind::GaaNanosheet => {
                let n = self.geometry.get("sheet_count").copied().unwrap_or(1.0) as usize;
                let w = self.geometry.get("sheet_width_nm").copied().unwrap_or(40.0);
                let t = self.geometry.get("sheet_thickness_nm").copied().unwrap_or(8.0);
                (n as f64) * (2.0 * w + 2.0 * t) * 1e-3
            }
            DeviceKind::GaaNanowire => {
                let n = self.geometry.get("wire_count").copied().unwrap_or(1.0) as usize;
                let d = self.geometry.get("wire_diameter_nm").copied().unwrap_or(20.0);
                std::f64::consts::PI * d * (n as f64) * 1e-3
            }
            _ => self.width_um,
        }
    }

    pub fn oxide_capacitance_per_area(&self) -> f64 {
        EPSILON_0 * self.epsilon_r_ox / (self.tox_nm * 1e-9)
    }

    pub fn is_mos(&self) -> bool {
        self.kind.is_mos()
    }

    pub fn is_memory(&self) -> bool {
        self.kind.is_memory()
    }

    pub fn device_type_string(&self) -> String {
        self.kind.as_str().to_string()
    }
}

fn doping_default() -> f64 {
    1e16
}

/// Типизированные доменные ошибки (единый контракт для всех слоёв ядра).
#[derive(Debug, Clone, PartialEq)]
pub enum DomainError {
    Validation(String),
    Unsupported(String),
    Io(String),
}

impl fmt::Display for DomainError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Validation(msg) | Self::Io(msg) | Self::Unsupported(msg) => write!(f, "{msg}"),
        }
    }
}

impl std::error::Error for DomainError {}

impl From<std::io::Error> for DomainError {
    fn from(e: std::io::Error) -> Self {
        Self::Io(e.to_string())
    }
}

/// Унифицированные результаты вычислений (общие для ядра, GUI и Python-моста).
#[derive(Debug, Clone)]
pub struct FieldResult {
    pub values: Vec<f64>,
    pub shape: [usize; 3],
    pub backend: String,
    pub metrics: HashMap<String, f64>,
    pub description: String,
}

#[derive(Debug, Clone)]
pub struct CurveResult {
    pub kind: String,
    pub label: String,
    pub x_name: String,
    pub y_name: String,
    pub x: Vec<f64>,
    pub y: Vec<f64>,
}

/// Параметры развёрток (характеристики и поля).
#[derive(Debug, Clone)]
pub struct SweepConfig {
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
    pub grid_shape: [usize; 3],
    pub use_gpu: bool,
}

impl Default for SweepConfig {
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
            grid_shape: [9, 7, 7],
            use_gpu: false,
        }
    }
}
