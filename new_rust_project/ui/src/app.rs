//! Основное окно воркбенча TSPICE Laboratory (egui).

use egui::RichText;
use egui_plot::{Legend, Line, Plot, PlotPoints};
use std::collections::BTreeMap;

use tspice_solver::domain::{CurveResult, Device, DeviceKind, SweepConfig};
use tspice_solver::engine;

#[derive(Default)]
enum RunAction {
    #[default]
    Idle,
    Pde,
    Memory,
    Thermal,
    Electric,
    Quantum,
}

pub struct TspiceWorkbenchApp {
    device: Device,
    device_name: String,
    grid: (usize, usize, usize),
    vg: f64,
    vd: f64,
    id_a: f64,
    csv_path: String,
    plots: Vec<CurveResult>,
    metrics: BTreeMap<String, f64>,
    output_text: String,
    status: String,
    error: Option<String>,
    last_backend: String,
}

impl Default for TspiceWorkbenchApp {
    fn default() -> Self {
        Self {
            device: Device::new("nmos01", DeviceKind::PlanarMosfet),
            device_name: "nmos01".into(),
            grid: (17, 7, 17),
            vg: 1.0,
            vd: 0.8,
            id_a: 1e-4,
            csv_path: String::new(),
            plots: Vec::new(),
            metrics: BTreeMap::new(),
            output_text: String::new(),
            status: "Готов к работе. Выберите тип прибора и запустите расчёт.".into(),
            error: None,
            last_backend: "—".into(),
        }
    }
}

impl TspiceWorkbenchApp {
    fn run(&mut self, action: &RunAction) {
        self.error = None;
        let result = match action {
            RunAction::Pde => self.run_pde(),
            RunAction::Memory => self.run_memory(),
            RunAction::Thermal => self.run_thermal(),
            RunAction::Electric => self.run_electric(),
            RunAction::Quantum => self.run_quantum(),
            RunAction::Idle => return,
        };
        match result {
            Ok(()) => self.status = format!("Готово. Бэкенд: {}", self.last_backend),
            Err(e) => {
                self.error = Some(e.to_string());
                self.status = format!("Ошибка: {e}");
            }
        }
    }

    fn validated_device(&mut self) -> Result<Device, String> {
        self.device.name = self.device_name.clone();
        self.device
            .validate()
            .map_err(|e| format!("Валидация прибора: {}", e.to_string()))?;
        Ok(self.device.clone())
    }

    fn run_pde(&mut self) -> Result<(), String> {
        let device = self.validated_device()?;
        if !device.kind.is_mos() {
            return Err("Reduced-order PDE доступен только для MOS-семейств (FR-CORE-03)".into());
        }
        let mut cfg = SweepConfig::default();
        cfg.use_gpu = false;
        let (curves, backend, diag) =
            engine::solve_pde_characteristics(&device, &cfg).map_err(|e| e.to_string())?;
        self.metrics = diag.into_iter().collect();
        self.plots = curves;
        self.last_backend = backend;
        self.output_text.clear();
        Ok(())
    }

    fn run_memory(&mut self) -> Result<(), String> {
        let device = self.validated_device()?;
        let (curves, metrics, backend) =
            engine::solve_memory_characteristics(&device).map_err(|e| e.to_string())?;
        self.metrics = metrics.into_iter().collect();
        self.plots = curves;
        self.last_backend = backend;
        self.output_text.clear();
        Ok(())
    }

    fn run_thermal(&mut self) -> Result<(), String> {
        let device = self.validated_device()?;
        let (field, metrics) = engine::solve_steady_thermal(&device, self.vd, self.id_a, [
            self.grid.0, self.grid.1, self.grid.2,
        ])
        .map_err(|e| e.to_string())?;
        self.metrics = metrics.into_iter().collect();
        self.plots = vec![CurveResult {
            kind: "temperature_z".into(),
            label: "T(z) по оси толщины".into(),
            x_name: "z [точка]".into(),
            y_name: "T [K]".into(),
            x: (0..field.values.len()).map(|i| i as f64).collect(),
            y: field.values,
        }];
        self.last_backend = field.backend;
        self.output_text.clear();
        Ok(())
    }

    fn run_electric(&mut self) -> Result<(), String> {
        let device = self.validated_device()?;
        let (pot, efield, bfield) = engine::solve_quasistatic_electric_field(
            &device,
            self.vg,
            self.vd,
            self.id_a,
            [self.grid.0, self.grid.1, self.grid.2],
        )
        .map_err(|e| e.to_string())?;
        let mut metrics = pot.metrics.clone();
        metrics.insert("EMAX_V_M".into(), efield.values.iter().cloned().fold(0.0, f64::max));
        metrics.insert("BMAX_T".into(), bfield.values.iter().cloned().fold(0.0, f64::max));
        self.metrics = metrics.into_iter().collect();
        self.plots = vec![
            CurveResult {
                kind: "potential".into(),
                label: "Потенциал V(x)".into(),
                x_name: "x".into(),
                y_name: "V [В]".into(),
                x: (0..pot.values.len()).map(|i| i as f64).collect(),
                y: pot.values,
            },
            CurveResult {
                kind: "efield".into(),
                label: "|E(x)|".into(),
                x_name: "x".into(),
                y_name: "|E| [В/м]".into(),
                x: (0..efield.values.len()).map(|i| i as f64).collect(),
                y: efield.values,
            },
        ];
        self.last_backend = pot.backend;
        self.output_text.clear();
        Ok(())
    }

    fn run_quantum(&mut self) -> Result<(), String> {
        let device = self.validated_device()?;
        let field = engine::solve_quantum_charge_distribution(
            &device,
            self.vg,
            self.vd,
            "3d",
            self.grid,
            0.26,
            false,
        )
        .map_err(|e| e.to_string())?;
        self.metrics = field.metrics.into_iter().collect();
        // Срез по середине канала: профиль по z при фиксированных (x,y)
        let (nx, ny, nz) = (field.shape[0], field.shape[1], field.shape[2]);
        let (midx, midy) = ((nx / 2).max(0), (ny / 2).max(0));
        let z_profile: Vec<f64> = (0..nz).map(|k| field.values[(midx * ny + midy) * nz + k]).collect();
        self.plots = vec![CurveResult {
            kind: "carrier_density".into(),
            label: "n(z) — плотность электронов".into(),
            x_name: "z [точка]".into(),
            y_name: "n [см⁻³]".into(),
            x: (0..nz).map(|i| i as f64).collect(),
            y: z_profile,
        }];
        self.last_backend = field.backend;
        self.output_text.clear();
        Ok(())
    }

    fn run_extraction(&mut self) -> Result<(), String> {
        let device = self.validated_device()?;
        if self.plots.is_empty() || !device.is_mos() {
            return Err("Сначала рассчитайте характеристики MOS-прибора (кнопка PDE)".into());
        }
        // Преобразуем первую кривую idvg в набор свипов экстрактора
        let sweeps: Vec<tspice_solver::extraction::IvSweep> = self
            .plots
            .iter()
            .filter(|c| c.kind == "idvg")
            .map(|c| tspice_solver::extraction::IvSweep {
                v: c.x.clone(),
                i: c.y.clone(),
                bias_v: 0.05,
                instance_length_um: Some(device.length_um),
                instance_width_um: Some(device.width_um),
            })
            .collect();
        let artifact = engine::extract_compact_model(
            &device,
            &sweeps,
            tspice_solver::extraction::TargetSimulator::Ngspice,
        )
        .map_err(|e| e.to_string())?;
        self.output_text = format!(
            "ПРИГОДНОСТЬ: {:.3}\nRMSE(log10 Id): {:.4}\nMAPE: {:.2}%\n\n{}",
            artifact.fitness_score,
            artifact.rmse_log10_subthreshold,
            artifact.mape_above_threshold_pct,
            artifact.model_text
        );
        Ok(())
    }

    fn run_verilog_a(&mut self) -> Result<(), String> {
        let device = self.validated_device()?;
        if !device.is_memory() {
            return Err("Verilog-A для памяти доступен только для memristor/memtransistor".into());
        }
        self.output_text = engine::generate_memory_verilog_a(&device).map_err(|e| e.to_string())?;
        Ok(())
    }

    fn import_csv(&mut self) {
        if self.csv_path.trim().is_empty() {
            self.error = Some("Укажите путь к CSV-файлу".into());
            return;
        }
        match tspice_solver::load_measurement(&self.csv_path) {
            Ok(m) => {
                self.status = format!(
                    "Импортировано: {} (kind={}, точек={})",
                    m.label, m.kind, m.x.len()
                );
                self.error = None;
            }
            Err(e) => self.error = Some(e.to_string()),
        }
    }

    fn export_json(&mut self) {
        let report = tspice_solver::io::curves_to_json(&self.plots);
        self.output_text = report;
        self.status = "JSON-отчёт кривых сформирован".into();
    }

    fn device_header_ui(&mut self, ui: &mut egui::Ui) {
        ui.heading("Параметры прибора");
        let mut kind = self.device.kind;
        egui::ComboBox::from_label("Тип устройства")
            .selected_text(kind.as_str())
            .show_ui(ui, |ui| {
                for k in DeviceKind::ALL {
                    ui.selectable_value(&mut kind, k, k.as_str());
                }
            });
        if kind != self.device.kind {
            self.device = Device::new(self.device_name.clone(), kind);
        }
        ui.text_edit_singleline(&mut self.device_name);
        egui::Grid::new("device_grid")
            .num_columns(2)
            .spacing(egui::vec2(8.0, 4.0))
            .show(ui, |ui| {
                num_field(ui, "epsilon_r", &mut self.device.epsilon_r);
                num_field(ui, "epsilon_r_ox", &mut self.device.epsilon_r_ox);
                num_field(ui, "μn [см²/В·с]", &mut self.device.electron_mobility_cm2_vs);
                num_field(ui, "μp [см²/В·с]", &mut self.device.hole_mobility_cm2_vs);
                num_field(ui, "Eg [эВ]", &mut self.device.bandgap_ev);
                num_field(ui, "N [см⁻³]", &mut self.device.doping_cm3);
                num_field(ui, "Tox [нм]", &mut self.device.tox_nm);
                num_field(ui, "W [мкм]", &mut self.device.width_um);
                num_field(ui, "L [мкм]", &mut self.device.length_um);
                num_field(ui, "Tbody [нм]", &mut self.device.body_thickness_nm);
                num_field(ui, "Tsub [мкм]", &mut self.device.substrate_thickness_um);
                num_field(ui, "T [K]", &mut self.device.temperature_k);
                ui.end_row();
            });
    }

    fn run_buttons_ui(&mut self, ui: &mut egui::Ui) {
        ui.heading("Запуск расчётов");
        ui.horizontal_wrapped(|ui| {
            if ui.button("Характеристики (PDE)").clicked() {
                self.run(&RunAction::Pde);
            }
            if ui.button("Память (state-ODE)").clicked() {
                self.run(&RunAction::Memory);
            }
            if ui.button("Тепло (FEM)").clicked() {
                self.run(&RunAction::Thermal);
            }
            if ui.button("Электрика (FEM)").clicked() {
                self.run(&RunAction::Electric);
            }
            if ui.button("Quantum-Poisson").clicked() {
                self.run(&RunAction::Quantum);
            }
        });
        egui::Grid::new("op_grid").num_columns(2).spacing(egui::vec2(8.0, 4.0)).show(ui, |ui| {
            num_field(ui, "Vg [В]", &mut self.vg);
            num_field(ui, "Vd [В]", &mut self.vd);
            num_field(ui, "Id [А]", &mut self.id_a);
            ui.end_row();
        });
        ui.horizontal_wrapped(|ui| {
            if ui.button("Экстракция модели").clicked() {
                self.run_extraction().unwrap_or_else(|e| self.error = Some(e));
            }
            if ui.button("Verilog-A (память)").clicked() {
                self.run_verilog_a().unwrap_or_else(|e| self.error = Some(e));
            }
            if ui.button("Экспорт JSON кривых").clicked() {
                self.export_json();
            }
        });

        ui.separator();
        ui.heading("Данные измерений");
        ui.add(egui::TextEdit::singleline(&mut self.csv_path).hint_text("путь к CSV"));
        if ui.button("Импорт CSV").clicked() {
            self.import_csv();
        }
    }
}

fn num_field(ui: &mut egui::Ui, label: &str, value: &mut f64) {
    ui.label(label);
    ui.add(egui::DragValue::new(value).speed(0.01).max_decimals(6));
    ui.end_row();
}

impl eframe::App for TspiceWorkbenchApp {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        egui::TopBottomPanel::bottom("status").show(ctx, |ui| {
            ui.horizontal(|ui| {
                ui.label(RichText::new(&self.status).monospace());
                if let Some(err) = &self.error {
                    ui.label(RichText::new(format!("⚠ {err}")).color(egui::Color32::RED));
                }
            });
        });

        egui::SidePanel::left("device_panel").resizable(true).default_width(330.0_f32).show(ctx, |ui| {
            egui::ScrollArea::vertical().show(ui, |ui| {
                self.device_header_ui(ui);
                ui.separator();
                self.run_buttons_ui(ui);
            });
        });

        egui::CentralPanel::default().show(ctx, |ui| {
            ui.heading("Результаты");

            if !self.metrics.is_empty() {
                egui::CollapsingHeader::new("Метрики")
                    .default_open(true)
                    .show(ui, |ui| {
                        egui::Grid::new("metrics").striped(true).show(ui, |ui| {
                            for (k, v) in &self.metrics {
                                ui.label(k);
                                ui.label(format!("{v:.4e}"));
                                ui.end_row();
                            }
                        });
                    });
            }

            if !self.plots.is_empty() {
                let plot = Plot::new("results_plot")
                    .height(360.0_f32)
                    .legend(Legend::default());
                plot.show(ui, |plot_ui| {
                    for curve in &self.plots {
                        let points: Vec<[f64; 2]> = curve
                            .x
                            .iter()
                            .zip(curve.y.iter())
                            .map(|(&x, &y)| [x, y])
                            .collect();
                        plot_ui.line(
                            Line::new(PlotPoints::from(points)).name(curve.label.clone()),
                        );
                    }
                });
            }

            if !self.output_text.is_empty() {
                ui.separator();
                ui.label("Экспорт / модель");
                egui::ScrollArea::vertical()
                    .max_height(240.0_f32)
                    .show(ui, |ui| {
                        ui.add(
                            egui::TextEdit::multiline(&mut self.output_text)
                                .code_editor()
                                .desired_rows(12),
                        );
                    });
            }
        });
    }
}
