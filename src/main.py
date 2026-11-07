"""PyQt desktop interface for transistor characterization and model fitting."""

from __future__ import annotations

import math
import subprocess
import sys
from pathlib import Path

try:
    from PyQt6.QtCore import QTimer
    from PyQt6.QtGui import QGuiApplication
    from PyQt6.QtWidgets import (
        QApplication,
        QComboBox,
        QCheckBox,
        QDoubleSpinBox,
        QFileDialog,
        QFormLayout,
        QGroupBox,
        QGridLayout,
        QHBoxLayout,
        QLabel,
        QLineEdit,
        QListWidget,
        QMainWindow,
        QMessageBox,
        QPushButton,
        QScrollArea,
        QSpinBox,
        QSplitter,
        QTabWidget,
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
    from .simulation import solve_quasistatic_electric_field, solve_steady_thermal
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
    from simulation import solve_quasistatic_electric_field, solve_steady_thermal


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
        screen = QGuiApplication.primaryScreen()
        available = screen.availableGeometry() if screen else None
        width = min(1180, available.width() - 32) if available else 1180
        height = min(760, available.height() - 48) if available else 760
        self.setMinimumSize(min(760, width), min(540, height))
        self.resize(width, height)
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
        splitter.setChildrenCollapsible(False)
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
        self.tabs = QTabWidget()
        self.tabs.addTab(results, "Характеристики")
        self.simulation_tab = self._build_simulation_tab()
        self.tabs.addTab(self.simulation_tab, "3D / Тепло / Поле")
        splitter.addWidget(self.tabs)
        splitter.setSizes([350, 830])
        self.geometry_redraw_timer = QTimer(self)
        self.geometry_redraw_timer.setSingleShot(True)
        self.geometry_redraw_timer.timeout.connect(self._draw_3d_view)
        self.animation_timer = QTimer(self)
        self.animation_timer.timeout.connect(self._rotate_3d_view)
        self._update_device_fields()
        self._draw_3d_view()

    def _build_simulation_tab(self) -> QWidget:
        page = QWidget()
        page_layout = QVBoxLayout(page)
        disclaimer = QLabel(
            "3D FEM на линейных тетраэдрах: стационарная теплопроводность и "
            "квазистатическое электрическое поле. CUDA используется при наличии GPU PyTorch; "
            "иначе применяется CPU-решатель."
        )
        disclaimer.setWordWrap(True)
        page_layout.addWidget(disclaimer)

        controls = QGridLayout()
        self.simulation_fields: dict[str, QDoubleSpinBox] = {}
        simulation_entries = (
            ("gate_voltage", "Vg", 0.8, -100, 100, " В"),
            ("drain_voltage", "Vd", 0.8, -100, 100, " В"),
            ("drain_current", "Id", 1e-4, 0, 100, " А"),
            ("ambient_temperature", "Tокр", 300, 1, 2000, " К"),
            ("thermal_conductivity", "Теплопроводность", 130, 0.01, 5000, " Вт/(м·К)"),
            ("substrate_thickness", "Подложка", 0.5, 0.01, 1e5, " мкм"),
        )
        for index, (key, label, default, minimum, maximum, suffix) in enumerate(
            simulation_entries
        ):
            row, pair = divmod(index, 2)
            label_widget = QLabel(label)
            label_widget.setWordWrap(True)
            field = QDoubleSpinBox()
            field.setRange(minimum, maximum)
            field.setDecimals(9 if key == "drain_current" else (6 if maximum < 100 else 3))
            field.setValue(default)
            field.setSuffix(suffix)
            field.setKeyboardTracking(False)
            self.simulation_fields[key] = field
            field.valueChanged.connect(self._invalidate_simulation_results)
            controls.addWidget(label_widget, row, pair * 2)
            controls.addWidget(field, row, pair * 2 + 1)
        controls.setColumnStretch(1, 1)
        controls.setColumnStretch(3, 1)
        mesh_label = QLabel("Узлов по длине сетки")
        self.mesh_resolution = QSpinBox()
        self.mesh_resolution.setRange(5, 17)
        self.mesh_resolution.setSingleStep(2)
        self.mesh_resolution.setValue(9)
        self.mesh_resolution.setToolTip(
            "Повышение разрешения улучшает детализацию, но увеличивает время решения FEM."
        )
        controls.addWidget(mesh_label, 3, 0)
        controls.addWidget(self.mesh_resolution, 3, 1)
        controls_widget = QWidget()
        controls_widget.setLayout(controls)
        controls_scroll = QScrollArea()
        controls_scroll.setWidgetResizable(True)
        controls_scroll.setWidget(controls_widget)
        controls_scroll.setMaximumHeight(150)
        page_layout.addWidget(controls_scroll)
        action_row = QHBoxLayout()
        solve_button = QPushButton("Рассчитать тепло и поле")
        solve_button.clicked.connect(self._run_3d_simulations)
        action_row.addWidget(solve_button)
        self.view_mode = QComboBox()
        self.view_mode.addItem("3D-геометрия", "geometry")
        self.view_mode.addItem("Температурное поле", "thermal")
        self.view_mode.addItem("Электрическое поле", "electric")
        self.view_mode.addItem("Магнитное поле (оценка)", "magnetic")
        self.view_mode.currentIndexChanged.connect(self._draw_3d_view)
        action_row.addWidget(self.view_mode)
        self.animate_check = QCheckBox("Вращать 3D-сцену")
        self.animate_check.toggled.connect(self._toggle_animation)
        action_row.addWidget(self.animate_check)
        self.gpu_check = QCheckBox("Использовать CUDA GPU, если доступен")
        self.gpu_check.setChecked(True)
        action_row.addWidget(self.gpu_check)
        action_row.addStretch(1)
        page_layout.addLayout(action_row)

        self.simulation_figure = Figure()
        self.simulation_canvas = FigureCanvasQTAgg(self.simulation_figure)
        self.simulation_canvas.setMinimumHeight(220)
        self.simulation_axes = self.simulation_figure.add_subplot(111, projection="3d")
        page_layout.addWidget(self.simulation_canvas, 1)
        self.simulation_result = QTextEdit()
        self.simulation_result.setReadOnly(True)
        self.simulation_result.setMaximumHeight(110)
        self.simulation_result.setPlaceholderText("Результаты расчёта появятся здесь.")
        page_layout.addWidget(self.simulation_result)
        self.simulation_solutions = {}
        return page

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
        spin.valueChanged.connect(self._schedule_3d_redraw)

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
        if hasattr(self, "simulation_solutions"):
            self._invalidate_simulation_results()
        self.export_button.setText(
            "Экспорт BSIM4-модели"
            if device_type == "planar_mosfet"
            else "Экспорт JSON-отчёта"
        )
        self._update_effective_width_label()
        if hasattr(self, "geometry_redraw_timer"):
            self._schedule_3d_redraw()

    def _schedule_3d_redraw(self, *_args) -> None:
        if hasattr(self, "geometry_redraw_timer"):
            self._invalidate_simulation_results()
            self.geometry_redraw_timer.start(150)

    def _invalidate_simulation_results(self, *_args) -> None:
        self.simulation_solutions.clear()
        if hasattr(self, "view_mode") and self.view_mode.currentData() != "geometry":
            self.view_mode.setCurrentIndex(0)
        if hasattr(self, "simulation_result"):
            self.simulation_result.clear()

    def _toggle_animation(self, enabled: bool) -> None:
        if enabled:
            self.animation_timer.start(100)
        else:
            self.animation_timer.stop()

    def _rotate_3d_view(self) -> None:
        self.simulation_axes.view_init(
            elev=self.simulation_axes.elev,
            azim=(self.simulation_axes.azim + 1) % 360,
        )
        self.simulation_canvas.draw_idle()

    def _draw_3d_view(self, *_args) -> None:
        axes = self.simulation_axes
        if hasattr(self, "simulation_colorbar"):
            self.simulation_colorbar.remove()
            del self.simulation_colorbar
        axes.clear()
        mode = self.view_mode.currentData()
        if mode in self.simulation_solutions:
            self._draw_solution(axes, self.simulation_solutions[mode])
        else:
            self._draw_device_geometry(axes)
        self.simulation_figure.tight_layout()
        self.simulation_canvas.draw_idle()

    def _draw_device_geometry(self, axes) -> None:
        import numpy as np

        device = self._device_parameters(allow_empty_name=True)
        length = device.length_um * 1000
        width = device.effective_width_um * 1000
        height = max(
            self.fields["tox_nm"].value() * 2,
            self.GEOMETRY_FIELDS.get(
                {
                    "fd_soi": "body_thickness_nm",
                    "finfet": "fin_height_nm",
                    "gaa_nanosheet": "sheet_thickness_nm",
                    "gaa_nanowire": "wire_diameter_nm",
                    "tfet": "channel_thickness_nm",
                    "hemt": "barrier_thickness_nm",
                }.get(device.device_type, ""),
                ("", 5, 0, 0, ""),
            )[1],
        )
        substrate_height = max(height * 1.5, 20)
        gate_height = max(height, 10)
        self._add_box(
            axes,
            -length * 0.1,
            -width * 0.1,
            0,
            length * 1.2,
            width * 1.2,
            substrate_height,
            "#adb5bd",
            0.65,
        )
        channel_z = substrate_height
        if device.device_type == "fd_soi":
            box_thickness = device.geometry["buried_oxide_nm"]
            self._add_box(
                axes, -length * 0.1, -width * 0.1, channel_z,
                length * 1.2, width * 1.2, box_thickness, "#f4a261", 0.72
            )
            channel_z += box_thickness
        device_top = channel_z + max(height * 2, 40)
        if device.device_type == "finfet":
            fin_width = device.geometry["fin_width_nm"]
            fin_height = device.geometry["fin_height_nm"]
            count = int(device.geometry["fin_count"])
            pitch = device.geometry["fin_pitch_nm"]
            for index in self._display_indices(count):
                y0 = (index - (count - 1) / 2) * pitch - fin_width / 2
                self._add_box(
                    axes, 0, y0, channel_z, length, fin_width, fin_height, "#1677b8"
                )
            gate_z = channel_z + fin_height
            device_top = gate_z + gate_height
            self._add_box(
                axes, length * 0.25, -width * 0.1, gate_z, length * 0.5,
                width * 1.2, gate_height, "#e07a5f", 0.42
            )
        elif device.device_type == "gaa_nanosheet":
            sheet_width = device.geometry["sheet_width_nm"]
            thickness = device.geometry["sheet_thickness_nm"]
            count = int(device.geometry["sheet_count"])
            pitch = device.geometry["sheet_pitch_nm"]
            for index in self._display_indices(count):
                z0 = channel_z + index * pitch
                self._add_box(
                    axes, 0, -sheet_width / 2, z0, length, sheet_width, thickness, "#1677b8"
                )
            device_top = channel_z + count * pitch + gate_height
            self._add_box(
                axes, length * 0.25, -width * 0.1,
                channel_z - gate_height * 0.4,
                length * 0.5, width * 1.2, pitch * max(count, 1) + gate_height,
                "#e07a5f", 0.35
            )
        elif device.device_type == "gaa_nanowire":
            diameter = device.geometry["wire_diameter_nm"]
            count = int(device.geometry["wire_count"])
            theta = np.linspace(0, 2 * math.pi, 16)
            along = np.linspace(0, length, 18)
            xx, tt = np.meshgrid(along, theta)
            for index in self._display_indices(count):
                center_y = (index - (count - 1) / 2) * diameter * 1.8
                yy = center_y + diameter / 2 * np.cos(tt)
                zz = channel_z + diameter / 2 + diameter / 2 * np.sin(tt)
                axes.plot_surface(xx, yy, zz, color="#1677b8", linewidth=0, alpha=0.85)
            device_top = channel_z + diameter * 3 + gate_height
            self._add_box(
                axes, length * 0.25, -width * 0.1, channel_z - gate_height * 0.2,
                length * 0.5, width * 1.2, diameter * 3, "#e07a5f", 0.3
            )
        else:
            if device.device_type == "fd_soi":
                channel_thickness = device.geometry["body_thickness_nm"]
            elif device.device_type == "tfet":
                channel_thickness = device.geometry["channel_thickness_nm"]
            elif device.device_type == "hemt":
                channel_thickness = device.geometry["barrier_thickness_nm"]
            else:
                channel_thickness = max(device.tox_nm * 2, 5)
            channel_width = width
            if device.device_type == "tfet":
                self._add_box(
                    axes, 0, -channel_width / 2, channel_z, length * 0.35,
                    channel_width, channel_thickness * 2, "#f4a261"
                )
                self._add_box(
                    axes, length * 0.35, -channel_width / 2, channel_z, length * 0.3,
                    channel_width, channel_thickness, "#1677b8"
                )
                self._add_box(
                    axes, length * 0.65, -channel_width / 2, channel_z,
                    length * 0.35, channel_width, channel_thickness * 2, "#90be6d"
                )
            else:
                self._add_box(
                    axes, 0, -channel_width / 2, channel_z, length, channel_width,
                    channel_thickness, "#1677b8"
                )
                if device.device_type == "hemt":
                    self._add_box(
                        axes, length * 0.15, -channel_width / 2,
                        channel_z + channel_thickness, length * 0.7,
                        channel_width, device.tox_nm, "#e9c46a"
                    )
            oxide_z = channel_z + channel_thickness
            self._add_box(
                axes, length * 0.25, -channel_width * 0.3, oxide_z,
                length * 0.5, channel_width * 0.6,
                max(device.tox_nm, 2), "#f4a261", 0.85
            )
            self._add_box(
                axes, length * 0.25, -channel_width * 0.3,
                oxide_z + max(device.tox_nm, 2),
                length * 0.5, channel_width * 0.6, gate_height, "#e07a5f", 0.8
            )
        axes.set_title(f"3D-схема: {self.DEVICE_LABELS[device.device_type]}")
        axes.set_xlabel("Длина канала (нм)")
        axes.set_ylabel("Ширина / периметр (нм)")
        axes.set_zlabel("Высота (нм)")
        axes.set_xlim(-length * 0.1, length * 1.1)
        axes.set_ylim(-width * 0.7, width * 0.7)
        axes.set_zlim(0, device_top)
        axes.set_box_aspect((length, width, max(height * 3, 60)))

    @staticmethod
    def _display_indices(count: int, maximum: int = 20) -> list[int]:
        if count <= maximum:
            return list(range(count))
        return sorted(
            {round(index * (count - 1) / (maximum - 1)) for index in range(maximum)}
        )

    @staticmethod
    def _add_box(axes, x, y, z, dx, dy, dz, color, alpha=0.9) -> None:
        axes.bar3d(
            x, y, z, max(dx, 0.1), max(dy, 0.1), max(dz, 0.1),
            color=color, alpha=alpha, shade=True, edgecolor="#333333", linewidth=0.25
        )

    def _draw_solution(self, axes, solution) -> None:
        import numpy as np

        x, y, z = np.meshgrid(
            solution.x_m * 1e9,
            solution.y_m * 1e9,
            solution.z_m * 1e9,
            indexing="ij",
        )
        stride = tuple(max(1, count // 9) for count in solution.values.shape)
        values = solution.values[::stride[0], ::stride[1], ::stride[2]]
        scatter = axes.scatter(
            x[::stride[0], ::stride[1], ::stride[2]],
            y[::stride[0], ::stride[1], ::stride[2]],
            z[::stride[0], ::stride[1], ::stride[2]],
            c=values.ravel(),
            cmap="inferno" if solution.value_unit == "K" else "viridis",
            s=35,
        )
        axes.set_title(solution.value_name)
        axes.set_xlabel("x (нм)")
        axes.set_ylabel("y (нм)")
        axes.set_zlabel("z (нм)")
        axes.set_box_aspect(
            (max(np.ptp(x), 1), max(np.ptp(y), 1), max(np.ptp(z), 1))
        )
        self.simulation_colorbar = self.simulation_figure.colorbar(
            scatter, ax=axes, shrink=0.65, pad=0.12, label=solution.value_unit
        )

    def _run_3d_simulations(self) -> None:
        try:
            device = self._device_parameters()
            settings = {
                key: field.value() for key, field in self.simulation_fields.items()
            }
            thermal = solve_steady_thermal(
                device,
                drain_voltage=settings["drain_voltage"],
                drain_current=settings["drain_current"],
                ambient_temperature_k=settings["ambient_temperature"],
                thermal_conductivity_w_mk=settings["thermal_conductivity"],
                substrate_thickness_um=settings["substrate_thickness"],
                use_gpu=self.gpu_check.isChecked(),
                grid_shape=self._simulation_grid_shape(),
            )
            electric, magnetic = solve_quasistatic_electric_field(
                device,
                gate_voltage=settings["gate_voltage"],
                drain_voltage=settings["drain_voltage"],
                drain_current=settings["drain_current"],
                use_gpu=self.gpu_check.isChecked(),
                grid_shape=self._simulation_grid_shape(),
            )
        except (ValueError, RuntimeError) as error:
            QMessageBox.warning(self, "Ошибка моделирования", str(error))
            return
        self.simulation_solutions = {
            "thermal": thermal,
            "electric": electric,
            "magnetic": magnetic,
        }
        self.view_mode.setCurrentIndex(1)
        self.simulation_result.setPlainText(
            f"Тепловой расчёт: {thermal.description}\n"
            f"Решатель: {thermal.backend}\n"
            + "\n".join(f"{key} = {value:.6g}" for key, value in thermal.metrics.items())
            + f"\n\nЭлектрическое поле: {electric.description}\n"
            f"Решатель: {electric.backend}\n"
            + "\n".join(f"{key} = {value:.6g}" for key, value in electric.metrics.items())
            + f"\n\nМагнитное поле: {magnetic.description}\n"
            + "\n".join(f"{key} = {value:.6g}" for key, value in magnetic.metrics.items())
        )

    def _simulation_grid_shape(self) -> tuple[int, int, int]:
        resolution = self.mesh_resolution.value()
        cross_section = max(5, resolution - 2)
        return resolution, cross_section, cross_section

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

    def _device_parameters(self, *, allow_empty_name: bool = False) -> DeviceParameters:
        device_type = self.device_type.currentData()
        geometry = {
            key: self.fields[key].value()
            for key in DEVICE_GEOMETRY_FIELDS[device_type]
        }
        parameters = DeviceParameters(
            name=self.model_name.text().strip() or ("preview" if allow_empty_name else ""),
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
                f"GPU обнаружен: {result.stdout.strip().splitlines()[0]}. "
                "Backend FEM отображается в результатах расчёта."
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