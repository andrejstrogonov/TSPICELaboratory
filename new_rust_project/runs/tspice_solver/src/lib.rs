//! Вычислительное ядро TSPICE Laboratory (Rust).
//!
//! Архитектура соответствует документации new_rust_project/documentation
//! (HOW_TO_RECOMMENDATIONS.md, REQUIREMENTS.md):
//! - доменный слой [`domain`] — единая контентная модель прибора (вместо 5 дублей);
//! - алгоритмы A-1..A-6 в модулях `fem`, `quantum_poisson`, `reduced_pde`,
//!   `memory_state`, `extraction`, `verification`, `gradient_comparison`,
//!   `hybrid_runtime`;
//! - единый фасад [`engine`] — точка входа для GUI и Python-моста;
//! - слой [`io`] — STL/CSV/JSON импорт-экспорт (перенос cad.py, model.py).
//!
//! По умолчанию крейт собирается нативно (rlib) для автономного .exe
//! (NFR-PORT-01). PyO3-мост включается фичей `python`.

pub mod domain;
pub mod engine;
pub mod io;

pub mod fem;
pub mod quantum_poisson;
pub mod reduced_pde;
pub mod memory_state;
pub mod extraction;
pub mod verification;
pub mod gradient_comparison;
pub mod hybrid_runtime;

#[cfg(feature = "python")]
pub mod python_bridge;

// Рекспорт ключевых типов домена и фасада для удобства потребителей.
pub use domain::{CurveResult, Device, DeviceKind, DomainError, FieldResult, SweepConfig};
pub use engine::{
    extract_compact_model, generate_memory_verilog_a, measurement_to_idvg_sweeps,
    solve_memory_characteristics, solve_pde_characteristics, solve_quantum_charge_distribution,
    solve_quasistatic_electric_field, solve_steady_thermal,
};
pub use io::{load_cad_device_model, load_measurement, load_stl, Measurement, StlMesh};
