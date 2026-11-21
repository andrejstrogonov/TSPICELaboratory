//! PyO3-мост (фича `python`) для интеграции с Python-интерфейсом.
//! Совместим с контрактом old_python_project/tests/test_solver.py:
//!   tspice_solver.PyDevice(device_type=..., doping=..., length_um=...,
//!       width_um=..., tox_nm=..., epsilon_r=..., temperature_k=..., geometry=...)
//!   tspice_solver.solve_quantum_charge_distribution(device, gate_voltage=...,
//!       drain_voltage=..., dimension="3d", grid_shape=(...), effective_mass_ratio=0.26)

use crate::domain::{Device, DeviceKind};
use ndarray::Array3;
use numpy::IntoPyArray;
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyModule};

#[pyclass(name = "PyDevice")]
pub struct PyDevice {
    pub(crate) device: Device,
}

#[pymethods]
impl PyDevice {
    #[new]
    #[pyo3(signature = (
        device_type = "planar_mosfet",
        doping = 1e16,
        length_um = 0.1,
        width_um = 1.0,
        tox_nm = 1.2,
        epsilon_r = 11.7,
        temperature_k = 300.0,
        geometry = None,
        body_thickness_nm = 30.0,
        substrate_thickness_um = 200.0
    ))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        device_type: &str,
        doping: f64,
        length_um: f64,
        width_um: f64,
        tox_nm: f64,
        epsilon_r: f64,
        temperature_k: f64,
        geometry: Option<Bound<'_, PyDict>>,
        body_thickness_nm: f64,
        substrate_thickness_um: f64,
    ) -> PyResult<Self> {
        let kind = DeviceKind::from_str_id(device_type)
            .ok_or_else(|| PyValueError::new_err(format!("Unknown device_type: {device_type}")))?;
        let mut device = Device::new("device", kind);
        device.doping_cm3 = doping;
        device.length_um = length_um;
        device.width_um = width_um;
        device.tox_nm = tox_nm;
        device.epsilon_r = epsilon_r;
        device.temperature_k = temperature_k;
        device.body_thickness_nm = body_thickness_nm;
        device.substrate_thickness_um = substrate_thickness_um;
        if let Some(g) = geometry {
            for (key, value) in g.iter() {
                if let (Ok(k), Ok(v)) = (key.extract::<String>(), value.extract::<f64>()) {
                    device.geometry.insert(k, v);
                }
            }
        }
        device.validate().map_err(|e| PyValueError::new_err(e.to_string()))?;
        Ok(Self { device })
    }

    #[getter]
    fn device_type(&self) -> String {
        self.device.kind.as_str().to_string()
    }
    #[getter]
    fn doping(&self) -> f64 {
        self.device.doping_cm3
    }
    #[getter]
    fn length_um(&self) -> f64 {
        self.device.length_um
    }
    #[getter]
    fn width_um(&self) -> f64 {
        self.device.width_um
    }
    #[getter]
    fn tox_nm(&self) -> f64 {
        self.device.tox_nm
    }
    #[getter]
    fn epsilon_r(&self) -> f64 {
        self.device.epsilon_r
    }
    #[getter]
    fn temperature_k(&self) -> f64 {
        self.device.temperature_k
    }
}

/// А-2 · Ускоренное квантовое ядро (Quantum-Poisson) на Rust.
#[pyfunction]
#[pyo3(signature = (
    device,
    gate_voltage,
    drain_voltage,
    dimension = "2d",
    grid_shape = None,
    effective_mass_ratio = 0.26,
    use_gpu = false
))]
#[allow(clippy::too_many_arguments)]
fn solve_quantum_charge_distribution(
    py: Python<'_>,
    device: &PyDevice,
    gate_voltage: f64,
    drain_voltage: f64,
    dimension: &str,
    grid_shape: Option<(usize, usize, usize)>,
    effective_mass_ratio: f64,
    use_gpu: bool,
) -> PyResult<PyObject> {
    let shape = grid_shape.unwrap_or((17, 7, 17));
    let solution = crate::engine::solve_quantum_charge_distribution(
        &device.device,
        gate_voltage,
        drain_voltage,
        dimension,
        shape,
        effective_mass_ratio,
        use_gpu,
    )
    .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;

    let [nxx, nyy, nzz] = solution.shape;
    let values = match Array3::from_shape_vec((nxx, nyy, nzz), solution.values) {
        Ok(a) => a,
        Err(e) => return Err(PyRuntimeError::new_err(e.to_string())),
    };

    let dict = PyDict::new_bound(py);
    dict.set_item("backend", solution.backend)?;
    dict.set_item("values", values.into_pyarray(py))?;
    dict.set_item("description", solution.description)?;
    let metrics = PyDict::new_bound(py);
    for (k, v) in &solution.metrics {
        metrics.set_item(k, *v)?;
    }
    dict.set_item("metrics", metrics)?;
    Ok(dict.into_any().unbind())
}

/// Тепловой FEM-расчёт (FR-EXT-08).
#[pyfunction]
#[pyo3(signature = (device, drain_voltage, drain_current, grid_shape = None))]
fn solve_steady_thermal(
    py: Python<'_>,
    device: &PyDevice,
    drain_voltage: f64,
    drain_current: f64,
    grid_shape: Option<(usize, usize, usize)>,
) -> PyResult<PyObject> {
    let shape = grid_shape.unwrap_or((9, 7, 7));
    let (field, metrics) = crate::engine::solve_steady_thermal(
        &device.device,
        drain_voltage,
        drain_current,
        [shape.0, shape.1, shape.2],
    )
    .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
    let dict = PyDict::new_bound(py);
    dict.set_item("backend", field.backend)?;
    dict.set_item("description", field.description)?;
    dict.set_item("values", field.values.into_pyarray(py))?;
    let metric_dict = PyDict::new_bound(py);
    for (k, v) in metrics {
        metric_dict.set_item(k, v)?;
    }
    dict.set_item("metrics", metric_dict)?;
    Ok(dict.into_any().unbind())
}

/// Характеристики памяти (state-ODE): мемристор / мемтранзистор.
#[pyfunction]
fn solve_memory_characteristics(py: Python<'_>, device: &PyDevice) -> PyResult<PyObject> {
    use pyo3::types::{PyDict as PD, PyList};
    let (curves, metrics, backend) =
        crate::engine::solve_memory_characteristics(&device.device)
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
    let dict = PD::new_bound(py);
    dict.set_item("backend", backend)?;
    let list = PyList::empty_bound(py);
    for c in &curves {
        let d = PD::new_bound(py);
        d.set_item("label", c.label.clone())?;
        d.set_item("x", c.x.clone())?;
        d.set_item("y", c.y.clone())?;
        list.append(d.into_any())?;
    }
    dict.set_item("curves", list)?;
    let metric_dict = PD::new_bound(py);
    for (k, v) in metrics {
        metric_dict.set_item(k, v)?;
    }
    dict.set_item("metrics", metric_dict)?;
    Ok(dict.into_any().unbind())
}

/// Генерация Verilog-A для устройства памяти (FR-EXT-06).
#[pyfunction]
fn generate_memory_verilog_a(py: Python<'_>, device: &PyDevice) -> PyResult<PyObject> {
    let source = crate::engine::generate_memory_verilog_a(&device.device)
        .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
    let obj = source.into_pyobject(py)?;
    Ok(obj.into_any().unbind())
}

/// Тёплый старт через ONNX Runtime (NFR-PERF-03, гибридный режим A-7/A-8).
/// Вход: 4 канала NCHW (donors, acceptors, potential, semiconductor).
/// Выход: предсказанное распределение (n, p) [2, H, W].
#[cfg(feature = "onnx")]
#[pyfunction]
#[pyo3(signature = (device, onnx_model_path, gate_voltage, drain_voltage, size = 32))]
fn solve_with_neural_feedback(
    py: Python<'_>,
    device: &PyDevice,
    onnx_model_path: &str,
    gate_voltage: f64,
    drain_voltage: f64,
    size: usize,
) -> PyResult<PyObject> {
    use ndarray::Array4;
    use ort::{inputs, Session, Value};

    let session =
        Session::builder()
            .and_then(|b| b.commit_from_file(onnx_model_path))
            .map_err(|e| PyRuntimeError::new_err(format!("ONNX session error: {e}")))?;
    let mut input_tensor = Array4::<f32>::zeros((1, 4, size, size));
    input_tensor
        .slice_mut(ndarray::s![0, 0, .., ..])
        .assign(&ndarray::Array2::<f32>::from_elem(
            (size, size),
            (device.device.doping_cm3 * 1e-15) as f32,
        ));
    input_tensor
        .slice_mut(ndarray::s![0, 2, .., ..])
        .assign(&ndarray::Array2::<f32>::from_elem(
            (size, size),
            gate_voltage as f32,
        ));
    input_tensor
        .slice_mut(ndarray::s![0, 3, .., ..])
        .assign(&ndarray::Array2::<f32>::ones((size, size)));

    let outputs = session
        .run(inputs![
            "input_features" => Value::from_array(input_tensor)
                .map_err(|e| PyRuntimeError::new_err(format!("ONNX input: {e}")))?
        ])
        .map_err(|e| PyRuntimeError::new_err(format!("ONNX run: {e}")))?;
    let electron_prediction = outputs["output_fields"]
        .try_extract_tensor::<f32>()
        .map_err(|e| PyRuntimeError::new_err(format!("ONNX output: {e}")))?
        .to_owned()
        .index_axis(ndarray::Axis(0), 0)
        .index_axis(ndarray::Axis(0), 0)
        .to_owned();

    let dict = PyDict::new_bound(py);
    dict.set_item("backend", "Rust + ONNX Runtime (Warm Start)")?;
    dict.set_item("values", electron_prediction.into_pyarray(py))?;
    Ok(dict.into_any().unbind())
}

#[pymodule]
fn tspice_solver(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyDevice>()?;
    m.add_function(wrap_pyfunction!(solve_quantum_charge_distribution, m)?)?;
    m.add_function(wrap_pyfunction!(solve_steady_thermal, m)?)?;
    m.add_function(wrap_pyfunction!(solve_memory_characteristics, m)?)?;
    m.add_function(wrap_pyfunction!(generate_memory_verilog_a, m)?)?;
    #[cfg(feature = "onnx")]
    m.add_function(wrap_pyfunction!(solve_with_neural_feedback, m)?)?;
    Ok(())
}
