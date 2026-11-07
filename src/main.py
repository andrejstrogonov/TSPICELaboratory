"""PyQt desktop interface for transistor characterization and model fitting."""

from __future__ import annotations

import math
import subprocess
import sys
from pathlib import Path

try:
    from PyQt6.QtCore import QTimer
    from PyQt6.QtWidgets import (
        QApplication,
        QComboBox,
        QDoubleSpinBox,
        QFileDialog,
        QFormLayout,
        QGroupBox,
        QHBoxLayout,
        QLabel,
        QLineEdit,
        QListWidget,
        QMainWindow,
        QMessageBox,
        QPushButton,
        QScrollArea,
        QSplitter,
        QTextEdit,
        QVBoxLayout,
        QWidget,
    )
    from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg, NavigationToolbar2QT
    from matplotlib.figure import Figure
except ImportError as error:
    raise SystemExit(
        "The desktop interface requires PyQt6 and Matplotlib. "
        "Install dependencies with: python -m pip install -r requirements.txt"
    ) from error

try:
    from .model import (
        DeviceParameters,
        DEVICE_GEOMETRY_FIELDS,
        DEVICE_TYPES,
        Measurement,
        export_analysis_report,
        export_spice_model,
        extract_cv,
        extract_idvg,
        extract_output_metrics,
        extract_transfer_metrics,
        load_measurement,
    )
except ImportError:
    from model import (
        DeviceParameters,
        DEVICE_GEOMETRY_FIELDS,
        DEVICE_TYPES,
        Measurement,
        export_analysis_report,
        export_spice_model,
        extract_cv,
        extract_idvg,
        extract_output_metrics,
        extract_transfer_metrics,
        load_measurement,
    )


class TransistorWorkbench(QMainWindow):
    DEVICE_LABELS = {
        "planar_mosfet": "Планарный MOSFET (bulk)",
        "fd_soi": "FD-SOI MOSFET",
        "finfet": "FinFET",
        "gaa_nanosheet": "GAA nanosheet",
        "gaa_nanowire": "GAA nanowire",
        "tfet": "TFET",
        "hemt": "HEMT",
    }
    GEOMETRY_FIELDS = {
        "body_thickness_nm": ("Толщина кремниевого слоя", 7, 0.1, 1000, " нм"),
        "buried_oxide_nm": ("Толщина buried oxide (BOX)", 25, 0.1, 1e6, " нм"),
        "back_gate_voltage": ("Напряжение back-gate", 0, -100, 100, " В"),
        "fin_width_nm": ("Ширина fin", 8, 0.1, 1000, " нм"),
        "fin_height_nm": ("Высота fin", 40, 0.1, 1e6, " нм"),
        "fin_count": ("Количество fin", 2, 1, 1e5, ""),
        "fin_pitch_nm": ("Шаг fin", 30, 0.1, 1e6, " нм"),
        "sheet_width_nm": ("Ширина nanosheet", 30, 0.1, 1e6, " нм"),
        "sheet_thickness_nm": ("Толщина nanosheet", 8, 0.1, 1000, " нм"),
        "sheet_count": ("Количество nanosheet", 3, 1, 1e5, ""),
        "sheet_pitch_nm": ("Шаг nanosheet", 30, 0.1, 1e6, " нм"),
        "wire_diameter_nm": ("Диаметр nanowire", 10, 0.1, 1000, " нм"),
        "wire_count": ("Количество nanowire", 3, 1, 1e5, ""),
        "tunnel_length_nm": ("Длина туннельного перехода", 20, 0.1, 1e6, " нм"),
        "channel_thickness_nm": ("Толщина тела/канала", 8, 0.1, 1000, " нм"),
        "source_doping_cm3": ("Легирование истока", 1e20, 1, 1e27, " см-3"),
        "drain_doping_cm3": ("Легирование стока", 1e18, 1, 1e27, " см-3"),
        "barrier_thickness_nm": ("Толщина барьера", 20, 0.1, 1000, " нм"),
        "sheet_density_cm2": ("Плотность 2DEG", 1e13, 1, 1e17, " см-2"),
        "channel_mobility_cm2_vs": ("Подвижность канала", 1500, 0.1, 1e7, " см2/(В·с)"),
        "access_resistance_ohm_mm": ("Сопротивление доступа", 0.5, 0, 1e6, " Ом·мм"),
        "gate_recess_nm": ("Углубление затвора", 0, 0, 1000, " нм"),
    }

    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Расчёт параметров транзистора")
        self.resize(1180, 760)
        self.measurements: list[Measurement] = []
        self.extracted: dict[str, float] = {}
        self.active_dataset_kind: str | None = None
        self._build_ui()
        self._refresh_gpu_status()
        self.gpu_timer = QTimer(self)
        self.gpu_timer.timeout.connect(self._refresh_gpu_status)
        self.gpu_timer.start(5000)

    def _build_ui(self) -> None:
        root = QWidget()
        layout = QVBoxLayout(root)
        self.setCentralWidget(root)
        title = QLabel("Экстракция параметров кастомного транзистора")
        title.setStyleSheet("font-size: 20px; font-weight: 600;")
        layout.addWidget(title)

        splitter = QSplitter()
        layout.addWidget(splitter, 1)
        controls = QWidget()
        controls_layout = QVBoxLayout(controls)

        device_group = QGroupBox("Тип транзистора")
        device_form = QFormLayout(device_group)
        self.device_type = QComboBox()
        for device_type in DEVICE_TYPES:
            self.device_type.addItem(self.DEVICE_LABELS[device_type], device_type)
        self.device_type.currentIndexChanged.connect(self._update_device_fields)
        device_form.addRow("Структура", self.device_type)
        self.device_description = QLabel()
        self.device_description.setWordWrap(True)
        device_form.addRow(self.device_description)
        self.effective_width_label = QLabel()
        device_form.addRow("Эффективная ширина", self.effective_width_label)
        controls_layout.addWidget(device_group)

        material_group = QGroupBox("Материал и геометрия")
        form = QFormLayout(material_group)
        self.model_name = QLineEdit("custom_nmos")
        form.addRow("Имя SPICE-модели", self.model_name)
        self.fields: dict[str, QDoubleSpinBox] = {}
        self.field_labels: dict[str, QLabel] = {}
        entries = (
            ("epsilon_r", "Диэлектрическая проницаемость", 11.7, 0.001, 1000, ""),
            ("electron_mobility", "Подвижность электронов", 1400, 0.001, 1e7, " см2/(В·с)"),
            ("hole_mobility", "Подвижность дырок", 450, 0.001, 1e7, " см2/(В·с)"),
            ("bandgap", "Ширина запрещённой зоны", 1.12, 0.001, 20, " эВ"),
            ("doping", "Концентрация легирования", 1e16, 1, 1e25, " см-3"),
            ("tox_nm", "Толщина оксида", 5, 0.01, 1e6, " нм"),
            ("width_um", "Ширина канала", 1, 0.001, 1e6, " мкм"),
            ("length_um", "Длина канала", 0.18, 0.001, 1e6, " мкм"),
            ("temperature_k", "Температура", 300, 1, 2000, " К"),
        )
        for key, label, default, minimum, maximum, suffix in entries:
            self._add_field(form, key, label, default, minimum, maximum, suffix)
        self.geometry_form = QFormLayout()
        self.barrier_material = QComboBox()
        self.barrier_material.addItems(
            ["AlGaN/GaN", "InAlN/GaN", "AlN/GaN", "InAlGaAs/InP", "Другое"]
        )
        self.geometry_form.addRow("Материал барьера", self.barrier_material)
        self.junction_material = QComboBox()
        self.junction_material.addItems(
            ["Si/Si", "Ge/Si", "Si/Ge", "InAs/GaSb", "InAs/AlSb", "Другое"]
        )
        self.geometry_form.addRow("Материалы туннельного перехода", self.junction_material)
        for key, spec in self.GEOMETRY_FIELDS.items():
            label, default, minimum, maximum, suffix = spec
            self._add_field(
                self.geometry_form, key, label, default, minimum, maximum, suffix
            )
        controls_layout.addWidget(material_group)

        self.device_group = QGroupBox("Параметры выбранной структуры")
        self.device_form_widget = QWidget()
        self.device_form_widget.setLayout(self.geometry_form)
        device_layout = QVBoxLayout(self.device_group)
        device_layout.addWidget(self.device_form_widget)
        controls_layout.addWidget(self.device_group)

        actions = QHBoxLayout()
        import_button = QPushButton("Импорт CSV")
        import_button.clicked.connect(self._import_csv)
        actions.addWidget(import_button)
        fit_button = QPushButton("Рассчитать параметры")
        fit_button.clicked.connect(self._fit_selected)
        actions.addWidget(fit_button)
        controls_layout.addLayout(actions)
        self.export_button = QPushButton("Экспорт результата")
        self.export_button.clicked.connect(self._export_model)
        controls_layout.addWidget(self.export_button)
        image_button = QPushButton("Сохранить график")
        image_button.clicked.connect(self._save_plot)
        controls_layout.addWidget(image_button)

        controls_layout.addWidget(QLabel("Загруженные характеристики"))
        self.measurement_list = QListWidget()
        self.measurement_list.currentRowChanged.connect(self._plot_selected)
        controls_layout.addWidget(self.measurement_list, 1)
        self.gpu_status = QLabel("GPU status: checking...")
        self.gpu_status.setWordWrap(True)
        controls_layout.addWidget(self.gpu_status)
        controls_scroll = QScrollArea()
        controls_scroll.setWidgetResizable(True)
        controls_scroll.setWidget(controls)
        splitter.addWidget(controls_scroll)

        results = QWidget()
        results_layout = QVBoxLayout(results)
        self.figure = Figure()
        self.canvas = FigureCanvasQTAgg(self.figure)
        self.axes = self.figure.add_subplot(111)
        results_layout.addWidget(NavigationToolbar2QT(self.canvas, self))
        results_layout.addWidget(self.canvas, 1)
        self.result_text = QTextEdit()
        self.result_text.setReadOnly(True)
        self.result_text.setPlaceholderText("Выберите транзистор и измерения, затем запустите расчёт.")
        self.result_text.setMaximumHeight(150)
        results_layout.addWidget(self.result_text)
        splitter.addWidget(results)
        splitter.setSizes([350, 830])
        self._update_device_fields()

    def _add_field(
        self,
        form: QFormLayout,
        key: str,
        label: str,
        default: float,
        minimum: float,
        maximum: float,
        suffix: str,
    ) -> None:
        spin = QDoubleSpinBox()
        spin.setRange(minimum, maximum)
        if key in ("fin_count", "sheet_count", "wire_count"):
            spin.setDecimals(0)
            spin.setSingleStep(1)
        else:
            spin.setDecimals(6 if maximum < 100 else 3)
        spin.setValue(default)
        spin.setSuffix(suffix)
        spin.setKeyboardTracking(False)
        label_widget = QLabel(label)
        self.fields[key] = spin
        self.field_labels[key] = label_widget
        form.addRow(label_widget, spin)
        spin.valueChanged.connect(self._update_effective_width_label)

    def _update_device_fields(self, *_args) -> None:
        device_type = self.device_type.currentData()
        visible_geometry = set(DEVICE_GEOMETRY_FIELDS[device_type])
        self.barrier_material.setVisible(device_type == "hemt")
        material_label = self.geometry_form.labelForField(self.barrier_material)
        material_label.setVisible(device_type == "hemt")
        self.junction_material.setVisible(device_type == "tfet")
        junction_label = self.geometry_form.labelForField(self.junction_material)
        junction_label.setVisible(device_type == "tfet")
        for key in self.GEOMETRY_FIELDS:
            visible = key in visible_geometry
            self.fields[key].setVisible(visible)
            self.field_labels[key].setVisible(visible)
        self.fields["width_um"].setVisible(
            device_type not in ("finfet", "gaa_nanosheet", "gaa_nanowire")
        )
        self.field_labels["width_um"].setVisible(
            device_type not in ("finfet", "gaa_nanosheet", "gaa_nanowire")
        )
        descriptions = {
            "planar_mosfet": "Bulk CMOS: Id–Vg оценивает VTH0, U0 и NFACTOR; поддержан экспорт BSIM4.",
            "fd_soi": "Тонкий кремниевый слой на BOX; задайте body/BOX и back-gate bias. Оценка Id–Vg аналогична MOS-приближению.",
            "finfet": "Мультизатворная структура; эффективная ширина канала вычисляется по числу и периметру fin.",
            "gaa_nanosheet": "Gate-all-around nanosheet; эффективная ширина суммируется по периметру всех листов.",
            "gaa_nanowire": "Gate-all-around nanowire; эффективная ширина вычисляется по окружности и числу проводов.",
            "tfet": "Туннельный FET: извлекаются Vth по постоянному току, Ion/Ioff, SS и gm; BSIM4 не применяется.",
            "hemt": "HEMT: параметры барьера и 2DEG; извлекаются Ion/Ioff, SS, gm и оценка сопротивления листа.",
        }
        self.device_description.setText(descriptions[device_type])
        self.extracted = {}
        self.result_text.clear()
        self.export_button.setText(
            "Экспорт BSIM4-модели"
            if device_type == "planar_mosfet"
            else "Экспорт JSON-отчёта"
        )
        self._update_effective_width_label()

    def _update_effective_width_label(self, *_args) -> None:
        if not hasattr(self, "effective_width_label"):
            return
        device_type = self.device_type.currentData()
        if device_type == "finfet":
            width = self.fields["fin_count"].value() * (
                2 * self.fields["fin_height_nm"].value()
                + self.fields["fin_width_nm"].value()
            ) * 1e-3
        elif device_type == "gaa_nanosheet":
            width = self.fields["sheet_count"].value() * (
                2 * self.fields["sheet_width_nm"].value()
                + 2 * self.fields["sheet_thickness_nm"].value()
            ) * 1e-3
        elif device_type == "gaa_nanowire":
            width = (
                math.pi
                * self.fields["wire_diameter_nm"].value()
                * self.fields["wire_count"].value()
                * 1e-3
            )
        else:
            width = self.fields["width_um"].value()
        self.effective_width_label.setText(f"{width:.6g} мкм")

    def _device_parameters(self) -> DeviceParameters:
        device_type = self.device_type.currentData()
        geometry = {
            key: self.fields[key].value()
            for key in DEVICE_GEOMETRY_FIELDS[device_type]
        }
        parameters = DeviceParameters(
            name=self.model_name.text().strip(),
            epsilon_r=self.fields["epsilon_r"].value(),
            electron_mobility=self.fields["electron_mobility"].value(),
            hole_mobility=self.fields["hole_mobility"].value(),
            bandgap=self.fields["bandgap"].value(),
            doping=self.fields["doping"].value(),
            tox_nm=self.fields["tox_nm"].value(),
            width_um=self.fields["width_um"].value(),
            length_um=self.fields["length_um"].value(),
            temperature_k=self.fields["temperature_k"].value(),
            device_type=device_type,
            geometry=geometry,
            barrier_material=self.barrier_material.currentText(),
            junction_material=self.junction_material.currentText(),
        )
        parameters.validate()
        return parameters

    def _import_csv(self) -> None:
        paths, _ = QFileDialog.getOpenFileNames(
            self,
            "Импорт характеристик из CSV",
            "",
            "CSV files (*.csv);;All files (*)",
        )
        for path in paths:
            try:
                measurement = load_measurement(path)
            except (OSError, ValueError) as error:
                QMessageBox.warning(self, "Не удалось импортировать CSV", f"{path}\n\n{error}")
                continue
            self.measurements.append(measurement)
            self.measurement_list.addItem(f"{measurement.label}  [{measurement.kind.upper()}]")
        if paths and self.measurements:
            self.measurement_list.setCurrentRow(len(self.measurements) - 1)

    def _plot_selected(self, index: int) -> None:
        self.axes.clear()
        self.extracted = {}
        self.active_dataset_kind = None
        self.result_text.clear()
        if not 0 <= index < len(self.measurements):
            self.canvas.draw_idle()
            return
        measurement = self.measurements[index]
        self.axes.plot(measurement.x, measurement.y, "o-", markersize=3, label=measurement.label)
        self.axes.set_xlabel(
            "Напряжение затвора (В)"
            if measurement.kind not in ("idvd",)
            else "Напряжение стока (В)"
        )
        self.axes.set_ylabel(
            "Ёмкость (Ф)" if measurement.kind == "cv" else "Ток стока (А)"
        )
        if measurement.kind != "cv":
            self.axes.set_yscale("log")
        self.axes.grid(True, which="both", alpha=0.3)
        self.axes.legend()
        self.figure.tight_layout()
        self.canvas.draw_idle()

    def _fit_selected(self) -> None:
        index = self.measurement_list.currentRow()
        if not 0 <= index < len(self.measurements):
            QMessageBox.information(
                self, "Нет выбранных данных", "Сначала импортируйте и выберите CSV-файл."
            )
            return
        try:
            device = self._device_parameters()
            measurement = self.measurements[index]
            if measurement.kind == "idvd":
                self.extracted = extract_output_metrics(measurement, device)
            elif measurement.kind == "cv":
                self.extracted = extract_cv(measurement, device)
            elif measurement.kind == "idvg" and device.device_type in (
                "tfet",
                "hemt",
            ):
                self.extracted = extract_transfer_metrics(measurement, device)
            elif measurement.kind == "idvg":
                self.extracted = extract_idvg(measurement, device)
            else:
                raise ValueError(
                    "Неизвестный тип измерений."
                )
        except (ValueError, RuntimeError) as error:
            QMessageBox.warning(self, "Ошибка расчёта параметров", str(error))
            return
        self.active_dataset_kind = measurement.kind
        if measurement.kind == "idvg" and device.device_type in ("tfet", "hemt"):
            self.result_text.setPlainText(
                "\n".join(f"{key} = {value:.8g}" for key, value in self.extracted.items())
                + "\n\nМетрики рассчитаны непосредственно по измеренной кривой; "
                "это не компактная SPICE-модель."
            )
            return
        self.result_text.setPlainText(
            "\n".join(f"{key} = {value:.8g}" for key, value in self.extracted.items())
        )

    def _export_model(self) -> None:
        try:
            device = self._device_parameters()
            if not self.extracted:
                raise ValueError("Сначала выберите набор данных и рассчитайте параметры.")
            if device.device_type == "planar_mosfet":
                text = export_spice_model(device, self.extracted)
                extension = "model"
                file_filter = "SPICE-модель (*.model)"
            else:
                text = export_analysis_report(device, self.extracted)
                extension = "json"
                file_filter = "JSON-отчёт (*.json)"
        except ValueError as error:
            QMessageBox.warning(self, "Невозможно экспортировать модель", str(error))
            return
        path, _ = QFileDialog.getSaveFileName(
            self,
            "Экспорт результатов расчёта",
            f"{self.model_name.text()}.{extension}",
            file_filter,
        )
        if not path:
            return
        try:
            Path(path).write_text(text, encoding="utf-8")
        except OSError as error:
            QMessageBox.critical(self, "Ошибка экспорта", str(error))
            return
        file_kind = "SPICE-модель" if extension == "model" else "JSON-отчёт"
        self.statusBar().showMessage(f"{file_kind} сохранён: {path}", 5000)

    def _save_plot(self) -> None:
        path, _ = QFileDialog.getSaveFileName(
            self,
            "Сохранение графика",
            "характеристики.png",
            "Изображение PNG (*.png);;Документ PDF (*.pdf)",
        )
        if path:
            try:
                self.figure.savefig(path, dpi=180, bbox_inches="tight")
            except OSError as error:
                QMessageBox.critical(self, "Не удалось сохранить график", str(error))

    def _refresh_gpu_status(self) -> None:
        try:
            result = subprocess.run(
                [
                    "nvidia-smi",
                    "--query-gpu=name,temperature.gpu,memory.used,memory.total,utilization.gpu",
                    "--format=csv,noheader,nounits",
                ],
                capture_output=True,
                check=True,
                text=True,
                timeout=2,
            )
            self.gpu_status.setText(
                f"GPU доступен (в расчётах не используется): {result.stdout.strip().splitlines()[0]}"
            )
        except (OSError, subprocess.SubprocessError, IndexError):
            self.gpu_status.setText("Сведения о GPU недоступны (nvidia-smi не найден)")


def main() -> int:
    app = QApplication(sys.argv)
    window = TransistorWorkbench()
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())