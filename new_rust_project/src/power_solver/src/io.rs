//! Импорт/экспорт данных (A-0 · IO). Перенос из old_python_project:
//! - src/cad.py: load_stl (бинарный/text STL), load_cad_device_model (JSON)
//! - src/model.py: load_measurement (CSV/whitespace), export_analysis_report,
//!   export_spice_model (BSIM4 level=54)
//! Требования: FR-IO-01, FR-IO-02, FR-IO-03, FR-EXT-01.

use crate::domain::{CurveResult, Device, DeviceKind, DomainError};
use std::collections::HashMap;
use std::path::Path;

pub const MAX_STL_TRIANGLES: usize = 2_000_000;
const MAX_STL_BYTES: u64 = 256 * 1024 * 1024;

/// Карта псевдонимов колонок CSV -> каноническое имя (model.py::_canonical_measurement_column).
const COLUMN_ALIASES: &[(&str, &str)] = &[
    ("vgs", "vg"),
    ("vg", "vg"),
    ("gate_voltage", "vg"),
    ("vds", "vd"),
    ("vd", "vd"),
    ("drain_voltage", "vd"),
    ("ids", "id"),
    ("id", "id"),
    ("drain_current", "id"),
    ("idsat", "id"),
    ("cgg", "cgg"),
    ("capacitance", "cgg"),
    ("h21", "h21"),
    ("freq", "freq"),
    ("frequency", "freq"),
    ("time", "time"),
    ("t", "time"),
    ("vout", "vout"),
    ("ig", "ig"),
    ("gm", "gm"),
];

fn canonical_column(raw: &str) -> String {
    let low = raw.trim().to_lowercase();
    for (alias, canon) in COLUMN_ALIASES {
        if low.eq_ignore_ascii_case(alias) {
            return canon.to_string();
        }
    }
    low
}

/// Загруженная кривая измерений (kind: idvg | idvd | cv | igvg | transient | memristor_iv).
#[derive(Debug, Clone)]
pub struct Measurement {
    pub kind: String,
    pub x: Vec<f64>,
    pub y: Vec<f64>,
    pub label: String,
    pub columns: HashMap<String, Vec<f64>>,
}

/// Парсер CSV/whitespace/Silvaco-подобных таблиц с автоопределением размерностей и `kind`.
/// Перенос model.py::load_measurement (упрощённая схема без полного разбора .log).
pub fn load_measurement(path: impl AsRef<Path>) -> Result<Measurement, DomainError> {
    let path = path.as_ref();
    let text = std::fs::read_to_string(path)?;
    let mut lines = text.lines().filter(|l| !l.trim().is_empty());

    // Пропуск заголовочных строк Silvaco (* ...) и индекса строк
    let mut header_stop = 0;
    let raw_lines: Vec<&str> = lines.by_ref().collect();
    let mut rows: Vec<Vec<&str>> = Vec::new();
    for (i, line) in raw_lines.iter().enumerate() {
        let trimmed = line.trim();
        if trimmed.starts_with('*') || trimmed.starts_with('#') {
            header_stop = i + 1;
            continue;
        }
        let parts: Vec<&str> = trimmed.split([',', '\t', ' ', ';']).filter(|s| !s.is_empty()).collect();
        if parts.len() >= 2 && parts.iter().all(|p| p.parse::<f64>().is_ok()) {
            rows.push(parts);
        } else {
            // Последний заголовочный ряд
            header_stop = i + 1;
        }
    }

    // Если заголовок не найден по умолчанию, берём первую строку как заголовок
    let col_names: Vec<String> = if header_stop > 0 && header_stop <= raw_lines.len() {
        let hdr = raw_lines[header_stop - 1].trim();
        let parts: Vec<String> = hdr
            .split([',', '\t', ' ', ';'])
            .filter(|s| !s.is_empty())
            .map(canonical_column)
            .collect();
        if parts.len() >= 2 { parts } else { vec!["vg".into(), "id".into()] }
    } else {
        vec!["vg".into(), "id".into()]
    };

    if rows.is_empty() {
        return Err(DomainError::Io("CSV не содержит строк числовых данных".into()));
    }
    let ncols = rows[0].len();
    let mut columns: HashMap<String, Vec<f64>> = HashMap::new();
    for c in 0..ncols {
        let key = col_names.get(c).cloned().unwrap_or_else(|| format!("col{c}"));
        let col: Vec<f64> = rows
            .iter()
            .filter_map(|r| r.get(c).and_then(|s| s.parse::<f64>().ok()))
            .collect();
        columns.entry(key).or_insert_with(Vec::new).extend(col);
    }

    // Автоопределение kind по набору колонок (FR-IO-01)
    let has = |k: &str| columns.contains_key(k);
    let kind = if has("cgg") && has("vg") {
        "cv".to_string()
    } else if has("id") && has("vg") && has("vd") {
        "idvd".to_string()
    } else if has("id") && has("vg") {
        "idvg".to_string()
    } else if has("ig") && has("vg") {
        "igvg".to_string()
    } else if has("vout") && has("time") {
        "transient".to_string()
    } else {
        return Err(DomainError::Io(
            "Не удалось определить тип характеристики (idvg/idvd/cv/igvg/transient)".into(),
        ));
    };

    let (x, y) = match kind.as_str() {
        "cv" => (columns["vg"].clone(), columns["cgg"].clone()),
        "transient" => (columns["time"].clone(), columns["vout"].clone()),
        _ => (columns["vg"].clone(), columns["id"].clone()),
    };

    Ok(Measurement {
        kind,
        label: path.file_stem().map(|s| s.to_string_lossy().to_string()).unwrap_or_default(),
        x,
        y,
        columns,
    })
}

/// Треугольная сетка STL (вершины в нанометрах).
#[derive(Debug, Clone, Default)]
pub struct StlMesh {
    pub vertices: Vec<[f64; 3]>,
    pub faces: Vec<[usize; 3]>,
}

/// Перенос cad.py::load_stl (бинарный STL по размеру, иначе ASCII-парсер `vertex`).
pub fn load_stl(path: impl AsRef<Path>, unit_to_nm: f64) -> Result<StlMesh, DomainError> {
    let path = path.as_ref();
    if !(unit_to_nm.is_finite() && unit_to_nm > 0.0) {
        return Err(DomainError::Io("STL unit scale must be positive and finite".into()));
    }
    let meta = std::fs::metadata(path)?;
    if meta.len() > MAX_STL_BYTES {
        return Err(DomainError::Io("STL file exceeds the 256 MiB import limit".into()));
    }
    let data = std::fs::read(path)?;
    if data.len() < 84 {
        return Err(DomainError::Io("STL file is truncated or does not contain any triangles".into()));
    }

    let binary_count = u32::from_le_bytes([data[80], data[81], data[82], data[83]]) as usize;
    let expected_binary = 84 + binary_count * 50;
    let triangles: Vec<[[f64; 3]; 3]> = if expected_binary == data.len() {
        if binary_count > MAX_STL_TRIANGLES {
            return Err(DomainError::Io(format!(
                "STL contains more than {MAX_STL_TRIANGLES} triangles"
            )));
        }
        let mut tris = Vec::with_capacity(binary_count);
        for i in 0..binary_count {
            let off = 84 + i * 50;
            let mut verts = [[0.0f64; 3]; 3];
            for v in 0..3 {
                for a in 0..3 {
                    let byte_off = off + 12 + v * 12 + a * 4;
                    let bytes: [u8; 4] = [data[byte_off], data[byte_off + 1], data[byte_off + 2], data[byte_off + 3]];
                    verts[v][a] = f32::from_le_bytes(bytes) as f64;
                }
            }
            tris.push(verts);
        }
        tris
    } else {
        // ASCII STL: ищем строки "vertex x y z"
        let text = String::from_utf8_lossy(&data);
        let mut verts: Vec<[f64; 3]> = Vec::new();
        for line in text.lines() {
            if line.trim_start().to_ascii_lowercase().starts_with("vertex") {
                let parts: Vec<&str> = line.split_whitespace().collect();
                if parts.len() >= 4 {
                    if let (Ok(x), Ok(y), Ok(z)) = (
                        parts[1].parse::<f64>(),
                        parts[2].parse::<f64>(),
                        parts[3].parse::<f64>(),
                    ) {
                        verts.push([x, y, z]);
                    }
                }
            }
        }
        if verts.is_empty() || verts.len() % 3 != 0 {
            return Err(DomainError::Io(
                "STL file has an invalid or empty triangle list".into(),
            ));
        }
        if verts.len() / 3 > MAX_STL_TRIANGLES {
            return Err(DomainError::Io(format!(
                "STL contains more than {MAX_STL_TRIANGLES} triangles"
            )));
        }
        verts.chunks_exact(3).map(|c| [c[0], c[1], c[2]]).collect()
    };

    if triangles.is_empty() {
        return Err(DomainError::Io("STL file has no triangles".into()));
    }
    for tri in &triangles {
        for v in tri {
            if !v.iter().all(|c| c.is_finite()) {
                return Err(DomainError::Io("STL geometry contains non-finite coordinates".into()));
            }
        }
    }

    let mut vertices: Vec<[f64; 3]> = Vec::with_capacity(triangles.len() * 3);
    let mut faces: Vec<[usize; 3]> = Vec::with_capacity(triangles.len());
    for tri in &triangles {
        let base = vertices.len();
        for v in tri {
            vertices.push([v[0] * unit_to_nm, v[1] * unit_to_nm, v[2] * unit_to_nm]);
        }
        faces.push([base, base + 1, base + 2]);
    }

    // Проверка ненулевой протяжённости по всем осям (cad.py)
    let min = [0, 1, 2].map(|a| vertices.iter().map(|v| v[a]).fold(f64::INFINITY, f64::min));
    let max = [0, 1, 2].map(|a| vertices.iter().map(|v| v[a]).fold(f64::NEG_INFINITY, f64::max));
    for a in 0..3 {
        if max[a] - min[a] <= 0.0 {
            return Err(DomainError::Io(
                "STL geometry must have non-zero extent along all axes".into(),
            ));
        }
    }
    Ok(StlMesh { vertices, faces })
}

/// Перенос cad.py::load_cad_device_model — загрузка параметров из JSON-отчёта анализа.
pub fn load_cad_device_model(path: impl AsRef<Path>) -> Result<Device, DomainError> {
    let payload: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(path.as_ref())?)?;
    let obj = payload
        .as_object()
        .ok_or_else(|| DomainError::Io("CAD device model must be a JSON object".into()))?;

    let kind_id = obj
        .get("device_type")
        .and_then(|v| v.as_str())
        .unwrap_or("planar_mosfet");
    let kind = DeviceKind::from_str_id(kind_id).ok_or_else(|| {
        DomainError::Io(format!("Unsupported CAD device model type: {kind_id}"))
    })?;

    let material = obj.get("material").and_then(|v| v.as_object()).ok_or_else(|| {
        DomainError::Io("Model JSON must contain 'material' object".into())
    })?;
    let geometry = obj.get("geometry").and_then(|v| v.as_object()).ok_or_else(|| {
        DomainError::Io("Model JSON must contain 'geometry' object".into())
    })?;

    let num = |o: &serde_json::Map<String, serde_json::Value>, key: &str, default: f64| -> f64 {
        o.get(key).and_then(|v| v.as_f64()).unwrap_or(default)
    };

    let mut device = Device::new(
        obj.get("name").and_then(|v| v.as_str()).unwrap_or("cad_device"),
        kind,
    );
    device.epsilon_r = num(material, "epsilon_r", device.epsilon_r);
    device.electron_mobility_cm2_vs = num(material, "electron_mobility_cm2_vs", device.electron_mobility_cm2_vs);
    device.hole_mobility_cm2_vs = num(material, "hole_mobility_cm2_vs", device.hole_mobility_cm2_vs);
    device.bandgap_ev = num(material, "bandgap_ev", device.bandgap_ev);
    device.doping_cm3 = num(material, "doping_cm3", device.doping_cm3);
    device.temperature_k = num(material, "temperature_k", device.temperature_k);
    device.tox_nm = num(geometry, "oxide_thickness_nm", num(geometry, "tox_nm", device.tox_nm));
    device.width_um = num(geometry, "width_um", device.width_um);
    device.length_um = num(geometry, "length_um", device.length_um);
    device.barrier_material = material
        .get("barrier_material")
        .and_then(|v| v.as_str())
        .unwrap_or("AlGaN/GaN")
        .to_string();
    device.junction_material = material
        .get("junction_material")
        .and_then(|v| v.as_str())
        .unwrap_or("Si/Si")
        .to_string();

    device.validate()?;
    Ok(device)
}

/// Формирование JSON-отчёта анализа (перенос model.py::export_analysis_report).
pub fn export_analysis_report(device: &Device, metrics: &HashMap<String, f64>) -> String {
    let mut geometry_map = device.geometry.clone();
    geometry_map.insert("oxide_thickness_nm".into(), device.tox_nm);
    geometry_map.insert("width_um".into(), device.width_um);
    geometry_map.insert("length_um".into(), device.length_um);

    let report = serde_json::json!({
        "name": device.name,
        "device_type": device.kind.as_str(),
        "material": {
            "epsilon_r": device.epsilon_r,
            "electron_mobility_cm2_vs": device.electron_mobility_cm2_vs,
            "hole_mobility_cm2_vs": device.hole_mobility_cm2_vs,
            "bandgap_ev": device.bandgap_ev,
            "doping_cm3": device.doping_cm3,
            "temperature_k": device.temperature_k,
            "barrier_material": device.barrier_material,
            "junction_material": device.junction_material
        },
        "geometry": geometry_map,
        "metrics": metrics
    });
    serde_json::to_string_pretty(&report).unwrap_or_else(|_| "{}".into())
}

/// Экспорт командного вида кривой для JSON (FR-IO-03).
pub fn curves_to_json(curves: &[CurveResult]) -> String {
    let arr: Vec<_> = curves
        .iter()
        .map(|c| {
            serde_json::json!({
                "kind": c.kind, "label": c.label,
                "x_name": c.x_name, "y_name": c.y_name,
                "x": c.x, "y": c.y
            })
        })
        .collect();
    serde_json::to_string_pretty(&serde_json::Value::Array(arr)).unwrap_or_else(|_| "[]".into())
}
