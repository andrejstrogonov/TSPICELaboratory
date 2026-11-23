//! Интеграционные тесты нативного ядра TSPICE Laboratory.
//! Покрывают сквозной конвейер A-0..A-6 без зависимостей от Python.

use tspice_solver::domain::{Device, DeviceKind, SweepConfig};

fn finfet_device() -> Device {
    let mut d = Device::new("fin_01", DeviceKind::Finfet);
    d.doping_cm3 = 1e16;
    d.length_um = 0.03;
    d.width_um = 0.2;
    d.tox_nm = 1.2;
    d.body_thickness_nm = 30.0;
    d.geometry.insert("fin_width_nm".into(), 8.0);
    d.geometry.insert("fin_height_nm".into(), 40.0);
    d.geometry.insert("fin_count".into(), 1.0);
    d.validate().expect("valid finfet");
    d
}

fn planar_device() -> Device {
    let mut d = Device::new("nmos_01", DeviceKind::PlanarMosfet);
    d.length_um = 0.5;
    d.width_um = 2.0;
    d.tox_nm = 2.0;
    d.doping_cm3 = 1e16;
    d.validate().expect("valid planar");
    d
}

#[test]
fn quantum_poisson_finfet_2d() {
    let d = finfet_device();
    let r = tspice_solver::solve_quantum_charge_distribution(
        &d, 1.0, 0.05, "2d", (17, 7, 17), 0.26, false,
    )
    .expect("quantum 2d solve");
    assert_eq!(r.shape[0], 17);
    assert_eq!(r.shape[2], 17);
    assert!(r.metrics.contains_key("QUANTUM_ITERATIONS"));
    assert!(!r.values.is_empty());
    assert!(r.metrics["QUANTUM_ITERATIONS"] > 0.0);
}

#[test]
fn quantum_poisson_supported_families_only() {
    let mut d = Device::new("mem_01", DeviceKind::Memristor);
    d.validate().expect("valid memristor");
    let r = tspice_solver::solve_quantum_charge_distribution(
        &d, 1.0, 0.05, "2d", (17, 7, 17), 0.26, false,
    );
    assert!(r.is_err(), "Quantum-Poisson должен отклонять memory-семейства");
}

#[test]
fn reduced_pde_transfer_curves() {
    let d = planar_device();
    let mut cfg = SweepConfig::default();
    cfg.gate_points = 13;
    cfg.drain_points = 9;
    cfg.channel_points = 5;
    cfg.poisson_nodes = 21;
    let (curves, backend, diagnostics) =
        tspice_solver::solve_pde_characteristics(&d, &cfg).expect("pde solve");
    assert!(!backend.is_empty());
    assert!(!curves.is_empty());
    let transfer: Vec<_> = curves
        .iter()
        .filter(|c| c.kind == "idvg" || c.kind == "idvd")
        .collect();
    assert!(!transfer.is_empty(), "должны быть передаточные кривые");
    for c in transfer {
        assert_eq!(c.x.len(), c.y.len());
        assert!(c.x.len() > 2);
    }
    let _ = diagnostics;
}

#[test]
fn memory_hysteresis_has_loop() {
    let mut d = Device::new("rram_01", DeviceKind::Memristor);
    d.low_resistance_ohm = 100.0;
    d.high_resistance_ohm = 1e5;
    d.set_voltage_v = 0.3;
    d.reset_voltage_v = -0.3;
    d.initial_state = 0.5;
    d.validate().expect("valid memristor");
    let (curves, metrics, backend) =
        tspice_solver::solve_memory_characteristics(&d).expect("memory solve");
    assert!(!backend.is_empty());
    assert!(!curves.is_empty());
    let iv: Vec<_> = curves.iter().filter(|c| c.kind.contains("memristor_iv")).collect();
    assert!(!iv.is_empty());
    assert!(metrics.contains_key("HYSTERESIS_LOOP_AREAS") || metrics.values().any(|&v| v > 0.0));
}

#[test]
fn fem_thermal_4x4x4() {
    let d = planar_device();
    let (field, metrics) = tspice_solver::solve_steady_thermal(&d, 0.5, 1e-4, [5, 5, 5])
        .expect("thermal solve");
    assert_eq!(field.shape, [5, 5, 5]);
    assert!(metrics["TMAX_K"] >= 300.0);
    assert!(metrics["POWER_W"] > 0.0);
}

#[test]
fn fem_electrostatic_returns_potential_b_field_e() {
    let d = planar_device();
    let (potential, efield, bfield) = tspice_solver::solve_quasistatic_electric_field(
        &d, 1.0, 0.05, 1e-4, [5, 5, 5],
    )
    .expect("electric solve");
    assert_eq!(potential.shape, [5, 5, 5]);
    assert_eq!(efield.shape, [5, 5, 5]);
    assert_eq!(bfield.shape, [5, 5, 5]);
    assert!(potential
        .values
        .iter()
        .any(|&v| v > 0.0 || v < 0.0));
}

#[test]
fn stl_ascii_parse() {
    use std::io::Write;
    let dir = std::env::temp_dir();
    let path = dir.join("tspice_test_cube.stl");
    let mut f = std::fs::File::create(&path).expect("create temp");
    f.write_all(
        b"solid cube\n\
          facet normal 0 0 1\n    outer loop\n      vertex 0 0 0\n      vertex 1 0 0\n      vertex 0 1 0\n    endloop\n  endfacet\n\
          facet normal 0 0 1\n    outer loop\n      vertex 1 0 0\n      vertex 1 1 1\n      vertex 0 1 0\n    endloop\n  endfacet\n\
          endsolid cube\n",
    )
    .expect("write stl");
    let mesh = tspice_solver::load_stl(&path, 1_000_000.0).expect("parse ascii stl");
    assert_eq!(mesh.faces.len(), 2);
    assert_eq!(mesh.vertices.len(), 6);
    // масштаб в нм: вершина (1,1,1) по z после ×1e6
    assert!(mesh.vertices.iter().any(|v| v[2] > 1e-3));
    let _ = std::fs::remove_file(&path);
}

#[test]
fn csv_measurement_auto_kind() {
    use std::io::Write;
    let dir = std::env::temp_dir();
    let path = dir.join("tspice_test_idvg.csv");
    let mut f = std::fs::File::create(&path).expect("create temp");
    f.write_all(b"vgs,ids\n0.0,1e-12\n0.2,1e-10\n0.4,1e-8\n0.6,1e-6\n0.8,1e-4\n1.0,1e-3\n")
        .expect("write csv");
    let m = tspice_solver::load_measurement(&path).expect("load csv");
    assert_eq!(m.kind, "idvg");
    assert_eq!(m.x.len(), 6);
    let _ = std::fs::remove_file(&path);
}

#[test]
fn extraction_produces_bsim4_card() {
    let d = planar_device();
    let mut vg = Vec::new();
    let mut ids = Vec::new();
    for i in 0..31 {
        let v = -0.5 + i as f64 * (1.7 / 30.0);
        vg.push(v);
        // Простая сигмоидальная модель для синтетической ВАХ
        let vov = (v - 0.4).max(0.0);
        let slope = 0.06;
        let softplus = slope * (1.0 + (vov / slope).exp()).ln();
        ids.push(2e-4 * softplus.powi(2));
    }
    let sweeps = vec![tspice_solver::extraction::IvSweep {
        v: vg,
        i: ids,
        bias_v: 0.05,
        instance_length_um: Some(0.5),
        instance_width_um: Some(2.0),
    }];
    let artifact = tspice_solver::engine::extract_compact_model(
        &d,
        &sweeps,
        tspice_solver::extraction::TargetSimulator::Ngspice,
    )
    .expect("extraction");
    assert!(artifact.model_text.contains(".MODEL"));
    assert!(artifact.model_text.contains("VTH0"));
    assert_eq!(artifact.level, Some(54));
}

#[test]
fn memory_verilog_a_generation() {
    let mut d = Device::new("rram_va", DeviceKind::Memristor);
    d.low_resistance_ohm = 1e3;
    d.high_resistance_ohm = 1e5;
    d.validate().expect("valid memristor");
    let va = tspice_solver::generate_memory_verilog_a(&d).expect("verilog-a");
    assert!(va.contains("module"));
    assert!(va.contains("ddt"));
}

#[test]
fn device_kind_roundtrip() {
    for kind in DeviceKind::ALL {
        assert_eq!(DeviceKind::from_str_id(kind.as_str()), Some(kind));
    }
}
