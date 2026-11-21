//! TSPICE Laboratory Workbench — нативный GUI на egui/eframe.
//! Заменяет PyQt6-интерфейс old_python_project/src/main.py (FR-UI-01..06, FR-EXT-*).
//! Все вычисления выполняются в нативном ядре `tspice_solver` (engine).

#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

mod app;

use app::TspiceWorkbenchApp;

fn main() -> Result<(), eframe::Error> {
    let options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default()
            .with_inner_size(egui::vec2(1280.0, 800.0))
            .with_title("TSPICE Laboratory — 3D TCAD / PDK"),
        ..Default::default()
    };
    eframe::run_native(
        "TSPICE Laboratory",
        options,
        Box::new(|_cc| Ok(Box::new(TspiceWorkbenchApp::default()) as Box<dyn eframe::App>)),
    )
}
