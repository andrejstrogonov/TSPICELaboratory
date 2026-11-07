"""PyQt desktop interface for transistor characterization and model fitting."""

from __future__ import annotations

import math
import subprocess
import sys
from pathlib import Path

try:
    from PyQt6.QtCore import QThread, QTimer, pyqtSignal
    from PyQt6.QtGui import QGuiApplication, QKeySequence, QShortcut
    from PyQt6.QtWidgets import (
        QApplication,
        QAbstractItemView,
        QComboBox,
        QCheckBox,
        QDoubleSpinBox,
        QFileDialog,
        QFormLayout,
        QGroupBox,
        QGridLayout,
        QHBoxLayout,
        QInputDialog,
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
        QTableWidget,
        QTableWidgetItem,
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
        extract_curve_metrics,
        load_measurement,
    )
    from .cad import load_cad_device_model, load_stl
    from .extractors import create_extractor, generate_memory_verilog_a
    from .simulation import (
        MAX_MESH_NODES,
        estimate_mesh_memory_bytes,
        estimate_tetrahedron_count,
        solve_quasistatic_electric_field,
        solve_quantum_charge_distribution,
        solve_pde_characteristics,
        solve_memory_characteristics,
        solve_steady_thermal,
    )
except ImportError:
    from model import (
        DeviceParameters,
        DEVICE_GEOMETRY_FIELDS,
        DEVICE_TYPES,
        Measurement,
        extract_curve_metrics,
        load_measurement,
    )
    from cad import load_cad_device_model, load_stl
    from extractors import create_extractor, generate_memory_verilog_a
    from simulation import (
        MAX_MESH_NODES,
        estimate_mesh_memory_bytes,
        estimate_tetrahedron_count,
        solve_quasistatic_electric_field,
        solve_quantum_charge_distribution,
        solve_pde_characteristics,
        solve_memory_characteristics,
        solve_steady_thermal,
    )


class TransistorWorkbench(QMainWindow):
    CHARACTERISTIC_SPECS = (
        ("idvg", "MOSFET / FET · Id–Vg", "gate_voltage", "drain_current", "log"),
        ("idvd", "MOSFET / FET · Id–Vd", "drain_voltage", "drain_current", "log"),
        ("igvg", "Затвор · Ig–Vg", "gate_voltage", "gate_current", "log"),
        ("cv_cgg", "Ёмкость · Cgg–Vg", "gate_voltage", "capacitance", "linear"),
        ("cv_cgs", "Ёмкость · Cgs–Vg", "gate_voltage", "gate_source_capacitance", "linear"),
        ("cv_cgd", "Ёмкость · Cgd–Vg", "gate_voltage", "gate_drain_capacitance", "linear"),
        ("cv_cgb", "Ёмкость · Cgb–Vg", "gate_voltage", "gate_bulk_capacitance", "linear"),
        ("gummel_ic", "BJT · Ic–Vbe (Gummel)", "vbe", "collector_current", "log"),
        ("gummel_ib", "BJT · Ib–Vbe (Gummel)", "vbe", "base_current", "log"),
        ("gummel_beta", "BJT · β–Ic", "collector_current", "base_current", "linear"),
        ("bjt_output", "BJT · Ic–Vce", "vce", "collector_current", "log"),
        ("bjt_cje", "BJT · Cje–Vbe", "vbe", "emitter_capacitance", "linear"),
        ("bjt_cjc", "BJT · Cjc–Vce", "vce", "collector_capacitance", "linear"),
        ("ac_h21", "AC · |h21|–частота", "frequency", "h21", "logx"),
        ("ac_h21_db", "AC · h21 (dB)–частота", "frequency", "h21_db", "logx"),
        ("ac_mason", "AC · U Mason–частота", "frequency", "mason_u", "logx"),
        ("ac_mason_db", "AC · U Mason (dB)–частота", "frequency", "mason_u_db", "logx"),
        ("ac_s11", "AC · |S11|–частота", "frequency", "s11", "logx"),
        ("ac_s11_db", "AC · S11 (dB)–частота", "frequency", "s11_db", "logx"),
        ("ac_s12", "AC · |S12|–частота", "frequency", "s12", "logx"),
        ("ac_s12_db", "AC · S12 (dB)–частота", "frequency", "s12_db", "logx"),
        ("ac_s21", "AC · |S21|–частота", "frequency", "s21", "logx"),
        ("ac_s21_db", "AC · S21 (dB)–частота", "frequency", "s21_db", "logx"),
        ("ac_s22", "AC · |S22|–частота", "frequency", "s22", "logx"),
        ("ac_s22_db", "AC · S22 (dB)–частота", "frequency", "s22_db", "logx"),
        ("ac_y11", "AC · Y11–частота", "frequency", "y11", "logx"),
        ("ac_y12", "AC · Y12–частота", "frequency", "y12", "logx"),
        ("ac_y21", "AC · Y21–частота", "frequency", "y21", "logx"),
        ("ac_y22", "AC · Y22–частота", "frequency", "y22", "logx"),
        ("ac_z11", "AC · Z11–частота", "frequency", "z11", "logx"),
        ("ac_z12", "AC · Z12–частота", "frequency", "z12", "logx"),
        ("ac_z21", "AC · Z21–частота", "frequency", "z21", "logx"),
        ("ac_z22", "AC · Z22–частота", "frequency", "z22", "logx"),
        ("transient_id", "Переключение · Id–время", "time", "drain_current", "log"),
        ("transient_vg", "Переключение · Vg–время", "time", "gate_voltage", "linear"),
        ("gate_charge_vg", "Заряд затвора · Qg–Vg", "gate_voltage", "gate_current", "linear"),
        ("gate_charge_time", "Заряд затвора · Qg–время", "time", "gate_current", "linear"),
        ("breakdown", "Пробой · Id–Vd (оценочная кривая)", "drain_voltage", "drain_current", "log"),
        ("memristor_iv", "Мемристор · I–V (гистерезис)", "voltage", "current", "linear"),
        ("memristor_resistance", "Мемристор · R–время", "time", "resistance", "log"),
        ("memory_idvg", "Мемтранзистор · Id–Vg (гистерезис)", "gate_voltage", "drain_current", "log"),
    )
    DEVICE_LABELS = {
        "planar_mosfet": "Планарный MOSFET (bulk)",
        "fd_soi": "FD-SOI MOSFET",
        "finfet": "FinFET",
        "gaa_nanosheet": "GAA nanosheet",
        "gaa_nanowire": "GAA nanowire",
        "tfet": "TFET",
        "hemt": "HEMT",
        "memristor": "Мемристор",
        "memtransistor": "Мемтранзистор",
        "cad_custom": "CAD-транзистор",
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
        "active_layer_nm": ("Толщина активного слоя", 10, 0.1, 1000, " нм"),
        "substrate_oxide_nm": ("Подложка SiO₂ / нижний оксид", 100, 0.1, 1e6, " нм"),
        "ion_mobility_cm2_vs": ("Ионная подвижность", 1e-10, 1e-14, 1e-2, " см²/(В·с)"),
        "low_resistance_ohm": ("R_on / LRS", 100, 0.01, 1e12, " Ом"),
        "high_resistance_ohm": ("R_off / HRS", 100000, 0.02, 1e15, " Ом"),
        "state_relaxation_ms": ("Время релаксации памяти", 100, 0.01, 1e9, " мс"),
        "set_voltage_v": ("Порог SET", 0.3, 0.001, 100, " В"),
        "reset_voltage_v": ("Порог RESET", -0.3, -100, -0.001, " В"),
        "activation_energy_ev": ("Энергия активации", 0.6, 0.001, 10, " эВ"),
        "gate_coupling": ("Связь затвор–состояние", 2, 0, 1000, ""),
        "initial_state": ("Начальное состояние x₀", 0.5, 0, 1, ""),
    }

    class SimulationWorker(QThread):
        completed = pyqtSignal(object)
        failed = pyqtSignal(str)

        def __init__(self, device, settings, grid_shape, use_gpu, parent=None):
            super().__init__(parent)
            self.device = device
            self.settings = settings
            self.grid_shape = grid_shape
            self.use_gpu = use_gpu

        def run(self) -> None:
            settings = self.settings
            try:
                thermal = solve_steady_thermal(
                    self.device,
                    drain_voltage=settings["drain_voltage"],
                    drain_current=settings["drain_current"],
                    ambient_temperature_k=settings["ambient_temperature"],
                    thermal_conductivity_w_mk=settings["thermal_conductivity"],
                    substrate_thickness_um=settings["substrate_thickness"],
                    use_gpu=self.use_gpu,
                    grid_shape=self.grid_shape,
                )
                electric, magnetic = solve_quasistatic_electric_field(
                    self.device,
                    gate_voltage=settings["gate_voltage"],
                    drain_voltage=settings["drain_voltage"],
                    drain_current=settings["drain_current"],
                    use_gpu=self.use_gpu,
                    grid_shape=self.grid_shape,
                )
            except Exception as error:
                self.failed.emit(f"{type(error).__name__}: {error}")
                return
            self.completed.emit((thermal, electric, magnetic))

    class CharacteristicsWorker(QThread):
        completed = pyqtSignal(object)
        failed = pyqtSignal(str)

        def __init__(self, device, settings, use_gpu, parent=None):
            super().__init__(parent)
            self.device = device
            self.settings = settings
            self.use_gpu = use_gpu

        def run(self) -> None:
            try:
                if self.device.device_type in ("memristor", "memtransistor"):
                    result = solve_memory_characteristics(
                        self.device,
                        voltage_min_v=self.settings["gate_min_v"],
                        voltage_max_v=self.settings["gate_max_v"],
                        drain_voltage_v=self.settings["drain_saturation_v"],
                    )
                else:
                    result = solve_pde_characteristics(
                        self.device,
                        gate_min_v=self.settings["gate_min_v"],
                        gate_max_v=self.settings["gate_max_v"],
                        drain_linear_v=self.settings["drain_linear_v"],
                        drain_saturation_v=self.settings["drain_saturation_v"],
                        use_gpu=self.use_gpu,
                    )
            except Exception as error:
                self.failed.emit(f"{type(error).__name__}: {error}")
                return
            self.completed.emit(result)

    class ChargeWorker(QThread):
        completed = pyqtSignal(object)
        failed = pyqtSignal(str)

        def __init__(self, device, settings, use_gpu, parent=None):
            super().__init__(parent)
            self.device = device
            self.settings = settings
            self.use_gpu = use_gpu

        def run(self) -> None:
            try:
                solution = solve_quantum_charge_distribution(
                    self.device,
                    gate_voltage=self.settings["gate_voltage"],
                    drain_voltage=self.settings["drain_voltage"],
                    dimension=self.settings["dimension"],
                    grid_shape=self.settings["grid_shape"],
                    effective_mass_ratio=self.settings["effective_mass_ratio"],
                    use_gpu=self.use_gpu,
                )
            except Exception as error:
                self.failed.emit(f"{type(error).__name__}: {error}")
                return
            self.completed.emit(solution)

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
        self.analysis_results: dict[int, tuple[dict[str, float] | None, str | None]] = {}
        self.active_dataset_kind: str | None = None
        self.cad_geometry = None
        self._cad_family = "planar_mosfet"
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
        self.device_type.addItem(self.DEVICE_LABELS["cad_custom"], "cad_custom")
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

        actions = QGridLayout()
        import_button = QPushButton("Импорт кривых")
        import_button.clicked.connect(self._import_csv)
        actions.addWidget(import_button, 0, 0)
        examples_button = QPushButton("Загрузить примеры")
        examples_button.setToolTip("Добавить все встроенные CSV и Silvaco-style LOG-кривые")
        examples_button.clicked.connect(self._load_curve_examples)
        actions.addWidget(examples_button, 0, 1)
        fit_button = QPushButton("Рассчитать характеристики")
        fit_button.clicked.connect(self._fit_selected)
        actions.addWidget(fit_button, 1, 0)
        fit_all_button = QPushButton("Рассчитать все кривые")
        fit_all_button.setToolTip("Рассчитать метрики для каждого загруженного набора данных")
        fit_all_button.clicked.connect(self._fit_all)
        actions.addWidget(fit_all_button, 1, 1)
        controls_layout.addLayout(actions)
        self.pde_group = QGroupBox("PDE sweep · MOS electrostatics + drift-diffusion")
        pde_form = QFormLayout(self.pde_group)
        self.pde_form = pde_form
        self.pde_fields: dict[str, QDoubleSpinBox] = {}
        for key, label, default, minimum, maximum, suffix in (
            ("gate_min_v", "Vg min", -0.5, -100, 100, " В"),
            ("gate_max_v", "Vg max", 1.2, -100, 100, " В"),
            ("drain_linear_v", "Vd linear", 0.05, 0.001, 100, " В"),
            ("drain_saturation_v", "Vd saturation", 0.8, 0.001, 100, " В"),
        ):
            field = QDoubleSpinBox()
            field.setRange(minimum, maximum)
            field.setDecimals(4)
            field.setValue(default)
            field.setSuffix(suffix)
            field.setKeyboardTracking(False)
            self.pde_fields[key] = field
            pde_form.addRow(label, field)
        self.pde_button = QPushButton("Получить характеристики (PDE)")
        self.pde_button.setToolTip(
            "Рассчитать Id–Vg, четыре Id–Vd и Cgg–Vg для поддерживаемых MOS-структур. "
            "CUDA используется, если отмечена и доступна."
        )
        self.pde_button.clicked.connect(self._run_pde_characteristics)
        pde_form.addRow(self.pde_button)
        self.pde_gpu_check = QCheckBox("Использовать CUDA GPU, если доступен")
        self.pde_gpu_check.setChecked(True)
        pde_form.addRow(self.pde_gpu_check)
        self.pde_status = QLabel(
            "Генерируются только Id–Vg, Id–Vd и Cgg–Vg; это reduced-order модель, не полный TCAD."
        )
        self.pde_status.setWordWrap(True)
        pde_form.addRow(self.pde_status)
        controls_layout.addWidget(self.pde_group)
        self.export_button = QPushButton("Экспорт компактной модели")
        self.export_button.clicked.connect(self._export_model)
        controls_layout.addWidget(self.export_button)
        image_button = QPushButton("Экспорт графика (PDF/PNG)")
        image_button.setToolTip(
            "Сохранить выбранную характеристику: PDF для публикаций или PNG 300 dpi"
        )
        image_button.clicked.connect(self._save_plot)
        controls_layout.addWidget(image_button)

        controls_layout.addWidget(QLabel("Загруженные характеристики"))
        self.measurement_list = QListWidget()
        self.measurement_list.setSelectionMode(
            QAbstractItemView.SelectionMode.ExtendedSelection
        )
        self.measurement_list.currentRowChanged.connect(self._plot_selected)
        controls_layout.addWidget(self.measurement_list, 1)
        list_actions = QHBoxLayout()
        remove_button = QPushButton("Удалить выбранные")
        remove_button.clicked.connect(self._remove_selected_measurements)
        list_actions.addWidget(remove_button)
        clear_button = QPushButton("Очистить список")
        clear_button.clicked.connect(self._clear_measurements)
        list_actions.addWidget(clear_button)
        controls_layout.addLayout(list_actions)
        self.remove_measurements_shortcut = QShortcut(
            QKeySequence.StandardKey.Delete, self.measurement_list
        )
        self.remove_measurements_shortcut.activated.connect(
            self._remove_selected_measurements
        )
        self.gpu_status = QLabel("GPU status: checking...")
        self.gpu_status.setWordWrap(True)
        controls_layout.addWidget(self.gpu_status)
        controls_scroll = QScrollArea()
        controls_scroll.setWidgetResizable(True)
        controls_scroll.setWidget(controls)
        splitter.addWidget(controls_scroll)

        results = QWidget()
        results_layout = QVBoxLayout(results)
        characteristic_bar = QHBoxLayout()
        characteristic_bar.addWidget(QLabel("Характеристика:"))
        self.characteristic_selector = QComboBox()
        self.characteristic_selector.setMinimumWidth(280)
        self.characteristic_selector.setStyleSheet(
            "QComboBox { padding: 6px 10px; background: #303030; color: #f0f0f0; "
            "border: 1px solid #555; border-radius: 3px; }"
            "QComboBox QAbstractItemView { background: #252525; color: #f0f0f0; "
            "selection-background-color: #3d78a8; }"
        )
        self.characteristic_selector.currentIndexChanged.connect(
            self._plot_selected_characteristic
        )
        characteristic_bar.addWidget(self.characteristic_selector, 1)
        self.characteristic_status = QLabel(
            "Импортируйте кривые, чтобы выбрать характеристики"
        )
        self.characteristic_status.setStyleSheet("color: #a0a0a0;")
        characteristic_bar.addWidget(self.characteristic_status, 2)
        results_layout.addLayout(characteristic_bar)
        self.figure = Figure()
        self.canvas = FigureCanvasQTAgg(self.figure)
        self.axes = self.figure.add_subplot(111)
        results_layout.addWidget(NavigationToolbar2QT(self.canvas, self))
        results_layout.addWidget(self.canvas, 1)
        self.result_table = QTableWidget(0, 3)
        self.result_table.setHorizontalHeaderLabels(
            ["Набор данных", "Характеристика", "Значение / сообщение"]
        )
        self.result_table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.result_table.setWordWrap(True)
        self.result_table.horizontalHeader().setStretchLastSection(True)
        results_layout.addWidget(self.result_table, 1)
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
        self._refresh_characteristic_options()
        self._draw_3d_view()

    def _build_simulation_tab(self) -> QWidget:
        page = QWidget()
        page_layout = QVBoxLayout(page)
        disclaimer = QLabel(
            "3D FEM на линейных тетраэдрах с локальным сгущением у канала, затвора "
            "и границ материалов. Поле накладывается цветом на поверхность структуры; "
            "наведение показывает интерполированное значение. Для CAD поле переносится "
            "на поверхность по габаритам, но сама STL-поверхность пока не участвует "
            "в сборке FEM-матрицы. Электрическая FEM-задача линейная; кнопка расчёта "
            "заряда запускает отдельный самосогласованный 2D/3D Poisson–Schrödinger "
            "модуль с эффективной массой и конфайнментом. Это reduced-order модель, "
            "не неравновесный drift-diffusion TCAD; CUDA использует FP64, когда доступна."
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
        quality_label = QLabel("Качество сетки")
        self.mesh_quality = QComboBox()
        self.mesh_quality.addItem("Грубо", "rough")
        self.mesh_quality.addItem("Оптимально", "optimal")
        self.mesh_quality.addItem("Точно", "precise")
        self.mesh_quality.addItem("Пользовательская", "custom")
        self.mesh_quality.currentIndexChanged.connect(self._apply_mesh_preset)
        controls.addWidget(quality_label, 3, 0)
        controls.addWidget(self.mesh_quality, 3, 1, 1, 3)
        self.mesh_node_fields: dict[str, QSpinBox] = {}
        for column, (key, label, default) in enumerate(
            (("length", "Узлы L", 9), ("width", "Узлы W", 7), ("height", "Узлы H", 7))
        ):
            controls.addWidget(QLabel(label), 4, column * 2)
            node_count = QSpinBox()
            node_count.setRange(4, MAX_MESH_NODES)
            node_count.setValue(default)
            node_count.setKeyboardTracking(False)
            node_count.valueChanged.connect(self._mesh_dimensions_changed)
            self.mesh_node_fields[key] = node_count
            controls.addWidget(node_count, 4, column * 2 + 1)
        self.mesh_summary = QLabel()
        self.mesh_summary.setWordWrap(True)
        controls.addWidget(self.mesh_summary, 5, 0, 1, 6)
        charge_dimension_label = QLabel("Размерность заряда")
        self.charge_dimension = QComboBox()
        self.charge_dimension.addItem("2D · x–z + квантование по z", "2d")
        self.charge_dimension.addItem("3D · x–y–z + квантование по z", "3d")
        self.charge_dimension.currentIndexChanged.connect(
            self._update_charge_grid_controls
        )
        controls.addWidget(charge_dimension_label, 6, 0, 1, 2)
        controls.addWidget(self.charge_dimension, 6, 2, 1, 4)
        self.charge_grid_fields: dict[str, QSpinBox] = {}
        for column, (key, label, default) in enumerate(
            (("x", "Заряд X", 17), ("y", "Заряд Y", 7), ("z", "Заряд Z", 17))
        ):
            controls.addWidget(QLabel(label), 7, column * 2)
            node_count = QSpinBox()
            node_count.setRange(7 if key != "y" else 5, 64)
            node_count.setValue(default)
            node_count.setKeyboardTracking(False)
            self.charge_grid_fields[key] = node_count
            controls.addWidget(node_count, 7, column * 2 + 1)
        self.effective_mass_field = QDoubleSpinBox()
        self.effective_mass_field.setRange(0.05, 2.0)
        self.effective_mass_field.setSingleStep(0.01)
        self.effective_mass_field.setDecimals(3)
        self.effective_mass_field.setValue(0.26)
        self.effective_mass_field.setSuffix(" m₀")
        controls.addWidget(QLabel("Масса электронов"), 8, 0, 1, 2)
        controls.addWidget(self.effective_mass_field, 8, 2, 1, 2)
        self.charge_model_note = QLabel(
            "Квантовый Poisson: self-consistent поперечные подзоны; "
            "CUDA FP64 при доступности. Не заменяет неравновесный TCAD."
        )
        self.charge_model_note.setWordWrap(True)
        controls.addWidget(self.charge_model_note, 8, 4, 1, 2)
        controls_widget = QWidget()
        controls_widget.setLayout(controls)
        controls_scroll = QScrollArea()
        controls_scroll.setWidgetResizable(True)
        controls_scroll.setWidget(controls_widget)
        controls_scroll.setMaximumHeight(220)
        page_layout.addWidget(controls_scroll)
        action_row = QHBoxLayout()
        self.solve_button = QPushButton("Рассчитать тепло и поле")
        self.solve_button.clicked.connect(self._run_3d_simulations)
        action_row.addWidget(self.solve_button)
        self.charge_button = QPushButton("Рассчитать заряд носителей")
        self.charge_button.setToolTip(
            "Самосогласованный квантовый Poisson расчёт электронной концентрации "
            "в 2D или 3D с поперечным конфайнментом."
        )
        self.charge_button.clicked.connect(self._run_charge_simulation)
        action_row.addWidget(self.charge_button)
        self.import_cad_button = QPushButton("Загрузить CAD STL")
        self.import_cad_button.clicked.connect(self._import_cad_stl)
        self.clear_cad_button = QPushButton("Сбросить CAD")
        self.clear_cad_button.clicked.connect(self._clear_cad_geometry)
        self.view_mode = QComboBox()
        self.view_mode.addItem("3D-геометрия", "geometry")
        self.view_mode.addItem("Температурное поле", "thermal")
        self.view_mode.addItem("Электрическое поле", "electric")
        self.view_mode.addItem("Магнитное поле (оценка)", "magnetic")
        self.view_mode.addItem("Концентрация электронов (заряд)", "carrier_charge")
        self.view_mode.currentIndexChanged.connect(self._draw_3d_view)
        action_row.addWidget(self.view_mode)
        self.animate_check = QCheckBox("Вращать 3D-сцену")
        self.animate_check.toggled.connect(self._toggle_animation)
        action_row.addWidget(self.animate_check)
        self.gpu_check = QCheckBox("Использовать CUDA GPU, если доступен")
        self.gpu_check.setChecked(True)
        action_row.addWidget(self.gpu_check)
        self.gpu_check.toggled.connect(self.pde_gpu_check.setChecked)
        self.pde_gpu_check.toggled.connect(self.gpu_check.setChecked)
        action_row.addStretch(1)
        page_layout.addLayout(action_row)
        cad_row = QHBoxLayout()
        cad_row.addWidget(self.import_cad_button)
        cad_row.addWidget(self.clear_cad_button)
        cad_row.addStretch(1)
        page_layout.addLayout(cad_row)

        self.simulation_figure = Figure()
        self.simulation_canvas = FigureCanvasQTAgg(self.simulation_figure)
        self.simulation_canvas.setMinimumHeight(220)
        self.simulation_axes = self.simulation_figure.add_subplot(111, projection="3d")
        self._hover_samples = None
        self._hover_label = QLabel(
            "Наведите курсор на поверхность структуры, чтобы увидеть значение поля."
        )
        self._hover_label.setWordWrap(True)
        self.simulation_canvas.mpl_connect("motion_notify_event", self._hover_field)
        page_layout.addWidget(self.simulation_canvas, 1)
        page_layout.addWidget(self._hover_label)
        self.simulation_result = QTextEdit()
        self.simulation_result.setReadOnly(True)
        self.simulation_result.setMaximumHeight(110)
        self.simulation_result.setPlaceholderText("Результаты расчёта появятся здесь.")
        page_layout.addWidget(self.simulation_result)
        self.simulation_solutions = {}
        self._update_mesh_summary()
        self._update_charge_grid_controls()
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
            if key == "ion_mobility_cm2_vs":
                spin.setDecimals(14)
        spin.setValue(default)
        spin.setSuffix(suffix)
        spin.setKeyboardTracking(False)
        label_widget = QLabel(label)
        self.fields[key] = spin
        self.field_labels[key] = label_widget
        form.addRow(label_widget, spin)
        spin.valueChanged.connect(self._invalidate_curve_analysis)
        spin.valueChanged.connect(self._update_effective_width_label)
        spin.valueChanged.connect(self._schedule_3d_redraw)

    def _invalidate_curve_analysis(self, *_args) -> None:
        self.analysis_results.clear()
        self.extracted = {}
        self.active_dataset_kind = None
        if hasattr(self, "result_table"):
            self.result_table.setRowCount(0)

    def _update_device_fields(self, *_args) -> None:
        device_type = self.device_type.currentData()
        family = self._cad_family if device_type == "cad_custom" else device_type
        if family in ("memristor", "memtransistor"):
            self.fields["ion_mobility_cm2_vs"].setValue(
                1e-10 if family == "memristor" else 1e-12
            )
        visible_geometry = set(DEVICE_GEOMETRY_FIELDS[family])
        self.barrier_material.setVisible(family == "hemt")
        material_label = self.geometry_form.labelForField(self.barrier_material)
        material_label.setVisible(family == "hemt")
        self.junction_material.setVisible(family == "tfet")
        junction_label = self.geometry_form.labelForField(self.junction_material)
        junction_label.setVisible(family == "tfet")
        for key in self.GEOMETRY_FIELDS:
            visible = key in visible_geometry
            self.fields[key].setVisible(visible)
            self.field_labels[key].setVisible(visible)
        self.fields["width_um"].setVisible(
            family not in ("finfet", "gaa_nanosheet", "gaa_nanowire")
        )
        self.field_labels["width_um"].setVisible(
            family not in ("finfet", "gaa_nanosheet", "gaa_nanowire")
        )
        descriptions = {
            "planar_mosfet": "Bulk CMOS: Id–Vg оценивает VTH0, U0 и NFACTOR; поддержан экспорт BSIM4.",
            "fd_soi": "Тонкий кремниевый слой на BOX; задайте body/BOX и back-gate bias. Оценка Id–Vg аналогична MOS-приближению.",
            "finfet": "Мультизатворная структура; эффективная ширина канала вычисляется по числу и периметру fin.",
            "gaa_nanosheet": "Gate-all-around nanosheet; эффективная ширина суммируется по периметру всех листов.",
            "gaa_nanowire": "Gate-all-around nanowire; эффективная ширина вычисляется по окружности и числу проводов.",
            "tfet": "Туннельный FET: извлекаются Vth по постоянному току, Ion/Ioff, SS и gm; BSIM4 не применяется.",
            "hemt": "HEMT: параметры барьера и 2DEG; извлекаются Ion/Ioff, SS, gm и оценка сопротивления листа.",
            "memristor": (
                "Мемристор: модель переключения проводящего филамента с внутренним "
                "состоянием, порогами SET/RESET, ионной подвижностью и активационной "
                "энергией. 3D-структура включает Si / SiO₂ / активный слой / верхний "
                "электрод; тепловой и электрический FEM доступны. Можно экспортировать Verilog-A."
            ),
            "memtransistor": (
                "Мемтранзистор: трехвыводная модель D/G/S; затвор модулирует канал и "
                "скорость изменения состояния проводящего филамента. 3D-структура "
                "включает нижний затвор Si / SiO₂ / канал MoS₂ / top-gate dielectric. "
                "Тепловой и электрический FEM доступны. Можно экспортировать Verilog-A."
            ),
            "cad_custom": (
                f"Импортированная CAD-геометрия; модель материала/устройства: "
                f"{self.DEVICE_LABELS[family]}. Настройте физические параметры "
                "в полях ниже или загрузите JSON-модель."
            ),
        }
        self.device_description.setText(descriptions.get(device_type, descriptions[family]))
        self._invalidate_curve_analysis()
        if hasattr(self, "simulation_solutions"):
            self._invalidate_simulation_results()
        memory_device = family in ("memristor", "memtransistor")
        self.export_button.setText(
            "Экспорт Verilog-A модели"
            if memory_device
            else "Экспорт компактной модели"
        )
        self.pde_group.setTitle(
            "Параметры модели памяти"
            if memory_device
            else "PDE sweep · MOS electrostatics + drift-diffusion"
        )
        self.pde_form.labelForField(self.pde_fields["gate_min_v"]).setText(
            "V min" if family == "memristor" else "Vg min"
        )
        self.pde_form.labelForField(self.pde_fields["gate_max_v"]).setText(
            "V max" if family == "memristor" else "Vg max"
        )
        for key in ("drain_linear_v", "drain_saturation_v"):
            visible = family not in ("memristor", "memtransistor") or (
                family == "memtransistor" and key == "drain_saturation_v"
            )
            self.pde_fields[key].setVisible(visible)
            self.pde_form.labelForField(self.pde_fields[key]).setVisible(visible)
        if family == "memtransistor":
            self.pde_form.labelForField(
                self.pde_fields["drain_saturation_v"]
            ).setText("Vd")
        else:
            self.pde_form.labelForField(
                self.pde_fields["drain_saturation_v"]
            ).setText("Vd saturation")
        self.charge_button.setEnabled(not memory_device)
        self.charge_button.setToolTip(
            "Квантовый Poisson-решатель носителей пока не моделирует мемристивное состояние."
            if memory_device
            else "Самосогласованный квантовый Poisson расчёт электронной концентрации "
            "в 2D или 3D с поперечным конфайнментом."
        )
        self.export_button.setEnabled(True)
        self.export_button.setToolTip(
            "Сохранить поведенческую Verilog-A модель устройства памяти."
            if memory_device
            else ""
        )
        self.pde_button.setText(
            "Сгенерировать характеристики памяти"
            if memory_device
            else "Получить характеристики (PDE)"
        )
        self.pde_gpu_check.setEnabled(not memory_device)
        if memory_device:
            self.pde_button.setToolTip(
                "Сгенерировать поведенческие характеристики мемристора/мемтранзистора. "
                "Это динамическая модель внутреннего состояния, не PDE/TCAD."
            )
            self.pde_status.setText(
                "Будут добавлены расчетные гистерезисные кривые. "
                "Модель поведенческая; CUDA не используется."
            )
        else:
            self.pde_button.setToolTip(
                "Рассчитать Id–Vg, четыре Id–Vd и Cgg–Vg для поддерживаемых MOS-структур. "
                "CUDA используется, если отмечена и доступна."
            )
            self.pde_status.setText(
                "Генерируются только Id–Vg, Id–Vd и Cgg–Vg; это reduced-order модель, не полный TCAD."
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
        if mode == "carrier_charge" and mode in self.simulation_solutions:
            solution = self.simulation_solutions[mode]
            self._hover_samples = None
            self._prepare_field_colors(solution)
            self._render_field_solution = None
            self._draw_device_geometry(axes)
            self._draw_charge_volume(axes, solution)
        elif mode in self.simulation_solutions:
            self._render_field_solution = self.simulation_solutions[mode]
            self._hover_samples = None
            self._prepare_field_colors(self._render_field_solution)
            if self.cad_geometry is not None:
                self._draw_cad_geometry(axes)
            else:
                self._draw_device_geometry(axes)
            self._finish_field_render(axes, self._render_field_solution)
        else:
            self._render_field_solution = None
            self._hover_samples = None
            self._hover_label.setText(
                "Наведите курсор на поверхность структуры, чтобы увидеть значение поля."
            )
            self._draw_device_geometry(axes)
        self.simulation_figure.tight_layout()
        self.simulation_canvas.draw_idle()

    def _draw_device_geometry(self, axes) -> None:
        import numpy as np

        if self._render_field_solution is not None:
            self._field_overlay_boxes = []
        if self.cad_geometry is not None:
            self._draw_cad_geometry(axes)
            return
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
                    "memristor": "active_layer_nm",
                    "memtransistor": "active_layer_nm",
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
            "#c18c5d" if device.device_type == "memtransistor" else "#adb5bd",
            0.65,
        )
        channel_z = substrate_height
        if device.device_type in ("memristor", "memtransistor"):
            substrate_oxide = device.geometry["substrate_oxide_nm"]
            self._add_box(
                axes,
                -length * 0.1,
                -width * 0.1,
                channel_z,
                length * 1.2,
                width * 1.2,
                substrate_oxide,
                "#d9e8f5",
                0.75,
            )
            channel_z += substrate_oxide
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
        elif device.device_type == "memristor":
            active_thickness = device.geometry["active_layer_nm"]
            electrode_thickness = max(active_thickness, 10)
            self._add_box(
                axes, 0, -width / 2, channel_z, length, width,
                electrode_thickness, "#7d8597"
            )
            active_z = channel_z + electrode_thickness
            self._add_box(
                axes, 0, -width / 2, active_z, length, width,
                active_thickness, "#9b5de5"
            )
            top_z = active_z + active_thickness
            self._add_box(
                axes, 0, -width / 2, top_z, length, width,
                electrode_thickness, "#d4af37"
            )
            device_top = top_z + electrode_thickness
        else:
            if device.device_type == "fd_soi":
                channel_thickness = device.geometry["body_thickness_nm"]
            elif device.device_type == "tfet":
                channel_thickness = device.geometry["channel_thickness_nm"]
            elif device.device_type == "hemt":
                channel_thickness = device.geometry["barrier_thickness_nm"]
            elif device.device_type == "memtransistor":
                channel_thickness = device.geometry["active_layer_nm"]
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
            elif device.device_type == "memtransistor":
                contact_width = max(length * 0.15, 5)
                self._add_box(
                    axes,
                    0,
                    -channel_width / 2,
                    channel_z,
                    length,
                    channel_width,
                    channel_thickness,
                    "#1677b8",
                )
                self._add_box(
                    axes,
                    0,
                    -channel_width / 2,
                    channel_z + channel_thickness,
                    contact_width,
                    channel_width,
                    max(channel_thickness, 5),
                    "#7d8597",
                )
                self._add_box(
                    axes,
                    length - contact_width,
                    -channel_width / 2,
                    channel_z + channel_thickness,
                    contact_width,
                    channel_width,
                    max(channel_thickness, 5),
                    "#7d8597",
                )
                oxide_z = channel_z + channel_thickness
                oxide_thickness = max(device.tox_nm, 2)
                gate_length = length * 0.5
                gate_start = (length - gate_length) / 2
                self._add_box(
                    axes,
                    gate_start,
                    -channel_width * 0.35,
                    oxide_z,
                    gate_length,
                    channel_width * 0.7,
                    oxide_thickness,
                    "#f4a261",
                    0.85,
                )
                self._add_box(
                    axes,
                    gate_start,
                    -channel_width * 0.35,
                    oxide_z + oxide_thickness,
                    gate_length,
                    channel_width * 0.7,
                    gate_height,
                    "#e07a5f",
                    0.8,
                )
                device_top = oxide_z + oxide_thickness + gate_height
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
            if device.device_type != "memtransistor":
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
        axes.set_title(f"3D-схема: {self.DEVICE_LABELS[device.device_type]}", pad=18)
        axes.set_xlim(-length * 0.1, length * 1.1)
        axes.set_ylim(-width * 0.7, width * 0.7)
        axes.set_zlim(0, device_top)
        axes.set_box_aspect((length, width, max(height * 3, 60)))
        axes.set_xlabel("L (нм)", labelpad=8)
        axes.set_ylabel("W / P (нм)", labelpad=8)
        axes.set_zlabel("H (нм)", labelpad=8)
        axes.tick_params(labelsize=8, pad=1)
        axes.set_proj_type("ortho")
        axes.view_init(elev=24, azim=-58)

    def _import_cad_stl(self) -> None:
        import numpy as np

        path, _ = QFileDialog.getOpenFileName(
            self,
            "Импорт транзистора из CAD",
            "",
            "STL-модель (*.stl);;Все файлы (*)",
        )
        if not path:
            return
        units, accepted = QInputDialog.getItem(
            self,
            "Единицы STL",
            "В каких единицах экспортирована STL-геометрия?",
            ["мм", "мкм", "м", "дюймы"],
            0,
            False,
        )
        if not accepted:
            return
        unit_scales = {"мм": 1e6, "мкм": 1e3, "м": 1e9, "дюймы": 25.4e6}
        try:
            vertices, faces = load_stl(path, unit_to_nm=unit_scales[units])
        except (OSError, ValueError) as error:
            QMessageBox.warning(
                self, "Не удалось импортировать STL", f"{path}\n\n{error}"
            )
            return

        model = None
        if QMessageBox.question(
            self,
            "Параметры CAD-транзистора",
            "Загрузить файл параметров модели JSON? Если выбрать «Нет», "
            "параметры можно заполнить вручную в форме материала и геометрии.",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        ) == QMessageBox.StandardButton.Yes:
            model_path, _ = QFileDialog.getOpenFileName(
                self,
                "Загрузить JSON-модель транзистора",
                str(Path(path).parent),
                "JSON-модель (*.json);;Все файлы (*)",
            )
            if not model_path:
                return
            try:
                model = load_cad_device_model(model_path)
            except (OSError, ValueError) as error:
                QMessageBox.warning(
                    self,
                    "Не удалось загрузить модель",
                    f"{model_path}\n\n{error}",
                )
                return

        self.cad_geometry = {
            "vertices": vertices,
            "faces": faces,
            "path": path,
        }
        bounds = np.ptp(vertices, axis=0)
        if model is None:
            self._cad_family = "planar_mosfet"
            self.fields["length_um"].setValue(bounds[0] / 1000)
            self.fields["width_um"].setValue(bounds[1] / 1000)
            self.model_name.setText(Path(path).stem.replace(" ", "_"))
        else:
            self._apply_cad_device_model(model)
        if self.device_type.currentData() != "cad_custom":
            self.device_type.setCurrentIndex(
                self.device_type.findData("cad_custom")
            )
        else:
            self._update_device_fields()
        self._invalidate_simulation_results()
        self.view_mode.setCurrentIndex(0)
        self._draw_3d_view()
        self.statusBar().showMessage(
            f"CAD-транзистор загружен: {Path(path).name}", 7000
        )

    def _apply_cad_device_model(self, model: DeviceParameters) -> None:
        self._cad_family = model.device_type
        self.model_name.setText(model.name)
        values = {
            "epsilon_r": model.epsilon_r,
            "electron_mobility": model.electron_mobility,
            "hole_mobility": model.hole_mobility,
            "bandgap": model.bandgap,
            "doping": model.doping,
            "tox_nm": model.tox_nm,
            "width_um": model.width_um,
            "length_um": model.length_um,
            "temperature_k": model.temperature_k,
            **model.geometry,
        }
        for key, value in values.items():
            if key in self.fields:
                self.fields[key].setValue(value)
        self.barrier_material.setCurrentText(model.barrier_material)
        self.junction_material.setCurrentText(model.junction_material)

    def _clear_cad_geometry(self) -> None:
        self.cad_geometry = None
        if self.device_type.currentData() == "cad_custom":
            self.device_type.setCurrentIndex(
                self.device_type.findData("planar_mosfet")
            )
        else:
            self.view_mode.setCurrentIndex(0)
            self._draw_3d_view()

    def _draw_cad_geometry(self, axes) -> None:
        import numpy as np
        from mpl_toolkits.mplot3d.art3d import Poly3DCollection

        vertices = self.cad_geometry["vertices"]
        faces = self.cad_geometry["faces"]
        low = vertices.min(axis=0)
        high = vertices.max(axis=0)
        center = (low + high) / 2
        extent = high - low
        self._field_display_bounds = np.column_stack((low, high))
        stride = max(1, math.ceil(len(faces) / 40_000))
        visible_faces = faces[::stride]
        triangles = vertices[visible_faces]
        if self._render_field_solution is None:
            facecolors = "#1677b8"
        else:
            sample_points = triangles.mean(axis=1)
            sample_values = self._sample_scene_field(
                sample_points, self._field_display_bounds
            )
            self._append_hover_samples(sample_points, sample_values)
            facecolors = self._field_colormap(
                self._field_normalizer(sample_values)
            )
        axes.add_collection3d(
            Poly3DCollection(
                triangles,
                facecolors=facecolors,
                edgecolor="#263238" if len(triangles) < 10_000 else "none",
                linewidth=0.12 if len(triangles) < 10_000 else 0,
                alpha=0.96,
            )
        )
        axes.set_xlim(center[0] - extent[0] / 2, center[0] + extent[0] / 2)
        axes.set_ylim(center[1] - extent[1] / 2, center[1] + extent[1] / 2)
        axes.set_zlim(center[2] - extent[2] / 2, center[2] + extent[2] / 2)
        axes.set_box_aspect(np.maximum(extent, extent.max() * 1e-6))
        axes.set_title(f"CAD STL: {Path(self.cad_geometry['path']).name}")
        axes.set_xlabel("X (нм)", labelpad=8)
        axes.set_ylabel("Y (нм)", labelpad=8)
        axes.set_zlabel("Z (нм)", labelpad=8)
        axes.tick_params(labelsize=8, pad=1)
        axes.set_proj_type("ortho")
        axes.view_init(elev=24, azim=-58)

    @staticmethod
    def _display_indices(count: int, maximum: int = 20) -> list[int]:
        if count <= maximum:
            return list(range(count))
        return sorted(
            {round(index * (count - 1) / (maximum - 1)) for index in range(maximum)}
        )

    def _add_box(self, axes, x, y, z, dx, dy, dz, color, alpha=0.9) -> None:
        if self._render_field_solution is not None:
            self._field_overlay_boxes.append((x, y, z, dx, dy, dz))
            alpha = min(alpha, 0.16)
        axes.bar3d(
            x, y, z, max(dx, 0.1), max(dy, 0.1), max(dz, 0.1),
            color=color, alpha=alpha, shade=True, edgecolor="#333333", linewidth=0.25
        )

    def _prepare_field_colors(self, solution) -> None:
        import numpy as np
        from matplotlib import colormaps
        from matplotlib.colors import LogNorm
        from matplotlib.colors import PowerNorm
        from scipy.interpolate import RegularGridInterpolator

        values = np.asarray(solution.values, dtype=np.float64)
        finite_values = values[np.isfinite(values)]
        if not finite_values.size:
            raise ValueError("Поле не содержит конечных значений для визуализации.")
        low, high = float(finite_values.min()), float(finite_values.max())
        if low == high:
            high = low + max(abs(low), 1.0) * 1e-12
        if solution.value_unit == "cm^-3":
            positive = finite_values[finite_values > 0]
            if not positive.size:
                raise ValueError("Расчёт не содержит положительной концентрации электронов.")
            low = max(float(positive.min()), high * 1e-8)
            self._field_normalizer = LogNorm(vmin=low, vmax=max(high, low * 1.001))
        else:
            self._field_normalizer = PowerNorm(gamma=0.7, vmin=low, vmax=high)
        cmap_name = (
            "inferno" if solution.value_unit in ("K", "cm^-3") else "turbo"
        )
        self._field_colormap = colormaps[cmap_name]
        self._field_interpolator = RegularGridInterpolator(
            (solution.x_m, solution.y_m, solution.z_m),
            solution.values,
            bounds_error=False,
            fill_value=None,
        )

    def _draw_charge_volume(self, axes, solution) -> None:
        import numpy as np
        from matplotlib.cm import ScalarMappable

        device = self._device_parameters(allow_empty_name=True)
        x_edges = np.asarray(solution.x_m) * 1e9
        y_edges = np.asarray(solution.y_m) * 1e9
        z_edges = np.asarray(solution.z_m) * 1e9
        if self.cad_geometry is not None:
            vertices = self.cad_geometry["vertices"]
            bounds = np.column_stack((vertices.min(axis=0), vertices.max(axis=0)))
            for edges, axis in zip((x_edges, y_edges, z_edges), range(3)):
                span = edges[-1] - edges[0]
                edges[:] = bounds[axis, 0] + (
                    (edges - edges[0]) / span
                ) * (bounds[axis, 1] - bounds[axis, 0])
        else:
            height_key = {
                "fd_soi": "body_thickness_nm",
                "finfet": "fin_height_nm",
                "gaa_nanosheet": "sheet_thickness_nm",
                "gaa_nanowire": "wire_diameter_nm",
                "tfet": "channel_thickness_nm",
                "hemt": "barrier_thickness_nm",
            }.get(device.device_type)
            active_height = (
                self.fields[height_key].value()
                if height_key is not None
                else 5.0
            )
            device_height = max(self.fields["tox_nm"].value() * 2, active_height)
            channel_z = max(device_height * 1.5, 20.0)
            y_edges = np.linspace(
                -device.effective_width_um * 500,
                device.effective_width_um * 500,
                len(y_edges),
            )
            z_edges += channel_z

        values = np.asarray(solution.values, dtype=np.float64)
        cell_density = sum(
            values[ix : values.shape[0] - 1 + ix,
                   iy : values.shape[1] - 1 + iy,
                   iz : values.shape[2] - 1 + iz]
            for ix, iy, iz in (
                (0, 0, 0), (1, 0, 0), (0, 1, 0), (1, 1, 0),
                (0, 0, 1), (1, 0, 1), (0, 1, 1), (1, 1, 1),
            )
        ) / 8.0
        normalized = self._field_normalizer(cell_density.ravel()).reshape(
            cell_density.shape
        )
        facecolors = self._field_colormap(normalized)
        facecolors[..., 3] = 0.78
        x_centers = (x_edges[:-1] + x_edges[1:]) / 2
        y_centers = (y_edges[:-1] + y_edges[1:]) / 2
        z_centers = (z_edges[:-1] + z_edges[1:]) / 2
        xz_x, xz_z = np.meshgrid(x_edges, z_edges, indexing="ij")
        xz_y = np.full_like(xz_x, (y_edges[0] + y_edges[-1]) / 2)
        if cell_density.shape[1] == 1:
            axes.plot_surface(
                xz_x,
                xz_y,
                xz_z,
                facecolors=facecolors[:, 0, :, :],
                shade=False,
                linewidth=0,
            )
        else:
            middle_y = cell_density.shape[1] // 2
            xz_y.fill(y_centers[middle_y])
            axes.plot_surface(
                xz_x,
                xz_y,
                xz_z,
                facecolors=facecolors[:, middle_y, :, :],
                shade=False,
                linewidth=0,
            )
            xy_x, xy_y = np.meshgrid(x_edges, y_edges, indexing="ij")
            xy_z = np.full_like(xy_x, z_centers[len(z_centers) // 2])
            axes.plot_surface(
                xy_x,
                xy_y,
                xy_z,
                facecolors=facecolors[:, :, len(z_centers) // 2, :],
                shade=False,
                linewidth=0,
            )
            yz_y, yz_z = np.meshgrid(y_edges, z_edges, indexing="ij")
            yz_x = np.full_like(yz_y, x_centers[len(x_centers) // 2])
            axes.plot_surface(
                yz_x,
                yz_y,
                yz_z,
                facecolors=facecolors[len(x_centers) // 2, :, :, :],
                shade=False,
                linewidth=0,
            )
        centers = np.column_stack(
            tuple(
                grid.ravel()
                for grid in np.meshgrid(
                    x_centers, y_centers, z_centers, indexing="ij"
                )
            )
        )
        stride = max(1, len(centers) // 12_000)
        self._append_hover_samples(
            centers.reshape(-1, 3)[::stride], cell_density.ravel()[::stride]
        )
        if self.cad_geometry is None:
            axes.set_xlim(0, device.length_um * 1000)
            axes.set_ylim(
                -device.effective_width_um * 500,
                device.effective_width_um * 500,
            )
        axes.set_zlim(min(axes.get_zlim3d()[0], z_edges[0]), z_edges[-1])
        axes.set_title(solution.value_name)
        axes.set_xlabel("L (нм)", labelpad=8)
        axes.set_ylabel("W (нм)", labelpad=8)
        axes.set_zlabel("Глубина (нм)", labelpad=8)
        self._hover_unit = solution.value_unit
        self._hover_label.setText(
            f"Объёмные ячейки показывают рассчитанную электронную концентрацию "
            f"({solution.description}). Наведите курсор для значения."
        )
        self.simulation_colorbar = self.simulation_figure.colorbar(
            ScalarMappable(norm=self._field_normalizer, cmap=self._field_colormap),
            ax=axes,
            shrink=0.65,
            pad=0.12,
            label=solution.value_unit,
        )

    def _update_charge_grid_controls(self, *_args) -> None:
        if not hasattr(self, "charge_grid_fields"):
            return
        three_dimensional = self.charge_dimension.currentData() == "3d"
        self.charge_grid_fields["y"].setEnabled(three_dimensional)

    def _finish_field_render(self, axes, solution) -> None:
        from matplotlib.cm import ScalarMappable

        if self.cad_geometry is None:
            self._draw_field_on_boxes(axes, solution)
        self._hover_unit = solution.value_unit
        self._hover_label.setText(
            "Цвет поверхности показывает поле на транзисторной структуре. "
            "Наведите курсор на поверхность для координат и значения."
        )
        axes.set_title(solution.value_name)
        self.simulation_colorbar = self.simulation_figure.colorbar(
            ScalarMappable(norm=self._field_normalizer, cmap=self._field_colormap),
            ax=axes,
            shrink=0.65,
            pad=0.12,
            label=solution.value_unit,
        )

    def _draw_field_on_boxes(self, axes, solution) -> None:
        import numpy as np

        self._field_display_bounds = np.array(
            [
                axes.get_xlim3d(),
                axes.get_ylim3d(),
                axes.get_zlim3d(),
            ]
        )
        resolution = 8
        for x, y, z, dx, dy, dz in self._field_overlay_boxes:
            u = np.linspace(0, 1, resolution)
            x_grid, y_grid = np.meshgrid(x + u * dx, y + u * dy, indexing="ij")
            x_grid_z, z_grid = np.meshgrid(x + u * dx, z + u * dz, indexing="ij")
            y_grid_z, z_grid_y = np.meshgrid(y + u * dy, z + u * dz, indexing="ij")
            faces = (
                (x_grid, y_grid, np.full_like(x_grid, z)),
                (x_grid, y_grid, np.full_like(x_grid, z + dz)),
                (x_grid_z, np.full_like(x_grid_z, y), z_grid),
                (x_grid_z, np.full_like(x_grid_z, y + dy), z_grid),
                (np.full_like(y_grid_z, x), y_grid_z, z_grid_y),
                (np.full_like(y_grid_z, x + dx), y_grid_z, z_grid_y),
            )
            for x_grid, y_grid, z_grid in faces:
                points = np.column_stack(
                    (x_grid.ravel(), y_grid.ravel(), z_grid.ravel())
                )
                field_values = self._sample_scene_field(
                    points, self._field_display_bounds
                )
                self._append_hover_samples(points, field_values)
                axes.plot_surface(
                    x_grid,
                    y_grid,
                    z_grid,
                    facecolors=self._field_colormap(
                        self._field_normalizer(field_values.reshape(x_grid.shape))
                    ),
                    linewidth=0,
                    antialiased=False,
                    shade=False,
                    alpha=0.98,
                )

    def _sample_scene_field(self, scene_points_nm, display_bounds) -> object:
        import numpy as np

        solution = self._render_field_solution
        bounds = np.asarray(display_bounds)
        points = np.asarray(scene_points_nm, dtype=np.float64)
        normalized = (points - bounds[:, 0]) / (bounds[:, 1] - bounds[:, 0])
        axes = (solution.x_m, solution.y_m, solution.z_m)
        field_points = np.column_stack(
            [
                axis[0] + np.clip(normalized[:, index], 0, 1) * (axis[-1] - axis[0])
                for index, axis in enumerate(axes)
            ]
        )
        return self._field_interpolator(field_points)

    def _append_hover_samples(self, points_nm, field_values) -> None:
        import numpy as np

        points = np.asarray(points_nm)
        values = np.asarray(field_values)
        stride = max(1, len(points) // 12_000)
        selected = points[::stride]
        selected_values = values[::stride]
        if self._hover_samples is None:
            self._hover_samples = (
                selected[:, 0],
                selected[:, 1],
                selected[:, 2],
                selected_values,
            )
        else:
            self._hover_samples = tuple(
                np.concatenate((old, new))
                for old, new in zip(
                    self._hover_samples,
                    (
                        selected[:, 0],
                        selected[:, 1],
                        selected[:, 2],
                        selected_values,
                    ),
                )
            )

    def _hover_field(self, event) -> None:
        import numpy as np
        from mpl_toolkits.mplot3d import proj3d

        if self._hover_samples is None:
            return
        if (
            event.inaxes is not self.simulation_axes
            or event.x is None
            or event.y is None
        ):
            self._hover_label.setText(
                "Наведите курсор на цветную поверхность структуры для "
                "координат и значения поля."
            )
            return
        x, y, z, values = self._hover_samples
        projected = proj3d.proj_transform(x, y, z, self.simulation_axes.get_proj())
        screen = self.simulation_axes.transData.transform(
            np.column_stack((projected[0], projected[1]))
        )
        distances = np.sum((screen - (event.x, event.y)) ** 2, axis=1)
        nearest = int(np.argmin(distances))
        if distances[nearest] > 18**2:
            self._hover_label.setText(
                "Наведите курсор на цветную поверхность структуры для "
                "координат и значения поля."
            )
            return
        self._hover_label.setText(
            f"X={x[nearest]:.6g} нм · Y={y[nearest]:.6g} нм · "
            f"Z={z[nearest]:.6g} нм · "
            f"значение={values[nearest]:.6g} {self._hover_unit} "
            "(интерполяция поля)"
        )

    def _run_3d_simulations(self) -> None:
        try:
            device = self._device_parameters()
            grid_shape = self._simulation_grid_shape()
            settings = {
                key: field.value() for key, field in self.simulation_fields.items()
            }
        except ValueError as error:
            QMessageBox.warning(self, "Ошибка моделирования", str(error))
            return
        if hasattr(self, "simulation_worker") and self.simulation_worker.isRunning():
            return
        self.simulation_solutions.clear()
        self.simulation_result.setPlainText(
            f"Собирается FEM-сетка {grid_shape[0]}×{grid_shape[1]}×{grid_shape[2]}..."
        )
        self.solve_button.setEnabled(False)
        self.simulation_worker = self.SimulationWorker(
            device,
            settings,
            grid_shape,
            self.gpu_check.isChecked(),
            self,
        )
        self.simulation_worker.completed.connect(self._simulation_completed)
        self.simulation_worker.failed.connect(self._simulation_failed)
        self.simulation_worker.start()

    def _run_charge_simulation(self) -> None:
        if (
            hasattr(self, "charge_worker")
            and self.charge_worker.isRunning()
        ):
            return
        try:
            device = self._device_parameters()
            settings = {
                "gate_voltage": self.simulation_fields["gate_voltage"].value(),
                "drain_voltage": self.simulation_fields["drain_voltage"].value(),
                "dimension": self.charge_dimension.currentData(),
                "grid_shape": tuple(
                    self.charge_grid_fields[key].value()
                    for key in ("x", "y", "z")
                ),
                "effective_mass_ratio": self.effective_mass_field.value(),
            }
        except ValueError as error:
            QMessageBox.warning(self, "Ошибка расчёта заряда", str(error))
            return
        self.charge_button.setEnabled(False)
        self.simulation_result.setPlainText(
            f"Решается {settings['dimension'].upper()} квантовая Poisson-задача "
            f"на сетке {settings['grid_shape']}..."
        )
        self.charge_worker = self.ChargeWorker(
            device, settings, self.gpu_check.isChecked(), self
        )
        self.charge_worker.completed.connect(self._charge_simulation_completed)
        self.charge_worker.failed.connect(self._charge_simulation_failed)
        self.charge_worker.start()

    def _charge_simulation_completed(self, solution) -> None:
        self.simulation_solutions["carrier_charge"] = solution
        self.charge_button.setEnabled(True)
        index = self.view_mode.findData("carrier_charge")
        self.view_mode.setCurrentIndex(index)
        self.simulation_result.setPlainText(
            f"{solution.description}\n"
            f"Вычислительный backend: {solution.backend}\n"
            + "\n".join(
                f"{key} = {value:.6g}" for key, value in solution.metrics.items()
            )
        )

    def _charge_simulation_failed(self, message: str) -> None:
        self.charge_button.setEnabled(True)
        self.simulation_result.setPlainText(
            f"Расчёт распределения носителей не выполнен:\n{message}"
        )
        QMessageBox.warning(self, "Ошибка расчёта заряда", message)

    def _run_pde_characteristics(self) -> None:
        if (
            hasattr(self, "characteristics_worker")
            and self.characteristics_worker.isRunning()
        ):
            return
        try:
            device = self._device_parameters()
            settings = {
                key: field.value() for key, field in self.pde_fields.items()
            }
            if settings["gate_min_v"] >= settings["gate_max_v"]:
                raise ValueError("Минимум sweep должен быть меньше максимума.")
        except ValueError as error:
            QMessageBox.warning(self, "Ошибка расчёта характеристик", str(error))
            return

        self.pde_button.setEnabled(False)
        self.pde_status.setText(
            "Генерируются кривые гистерезиса и состояния…"
            if device.device_type in ("memristor", "memtransistor")
            else "Рассчитываются кривые Id–Vg, Id–Vd и Cgg–Vg…"
        )
        self.characteristics_worker = self.CharacteristicsWorker(
            device, settings, self.pde_gpu_check.isChecked(), self
        )
        self.characteristics_worker.completed.connect(
            self._pde_characteristics_completed
        )
        self.characteristics_worker.failed.connect(self._pde_characteristics_failed)
        self.characteristics_worker.start()

    def _pde_characteristics_completed(self, result) -> None:
        first_index = len(self.measurements)
        for curve in result.curves:
            drain_voltage = curve.columns.get("drain_voltage")
            gate_bias = curve.columns.get("gate_voltage")
            self.measurements.append(
                Measurement(
                    kind=curve.kind,
                    x=curve.x,
                    y=curve.y,
                    drain_voltage=drain_voltage,
                    label=curve.label,
                    gate_bias=(
                        float(gate_bias[0])
                        if gate_bias and len(set(gate_bias)) == 1
                        else None
                    ),
                    columns=curve.columns,
                )
            )
            self.measurement_list.addItem(
                f"{curve.label}  [{curve.kind.upper()}]"
            )
        self.pde_button.setEnabled(True)
        self.measurement_list.setCurrentRow(first_index)
        self._refresh_characteristic_options(
            result.curves[0].kind if result.curves else None
        )
        self.pde_status.setText(
            f"Добавлено кривых: {len(result.curves)} · {result.backend}\n"
            f"{result.model_description}"
        )
        self.statusBar().showMessage(
            f"Расчетные кривые добавлены: {len(result.curves)}", 5000
        )

    def _pde_characteristics_failed(self, message: str) -> None:
        self.pde_button.setEnabled(True)
        self.pde_status.setText(f"Расчёт не выполнен: {message}")
        QMessageBox.warning(self, "Ошибка расчёта характеристик", message)

    def _simulation_completed(self, solutions) -> None:
        thermal, electric, magnetic = solutions
        self.simulation_solutions.update({
            "thermal": thermal,
            "electric": electric,
            "magnetic": magnetic,
        })
        self.solve_button.setEnabled(True)
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

    def _simulation_failed(self, message: str) -> None:
        self.solve_button.setEnabled(True)
        self.simulation_result.setPlainText(f"Расчёт не выполнен:\n{message}")
        QMessageBox.warning(self, "Ошибка моделирования", message)

    def _simulation_grid_shape(self) -> tuple[int, int, int]:
        shape = tuple(
            self.mesh_node_fields[key].value()
            for key in ("length", "width", "height")
        )
        if math.prod(shape) > MAX_MESH_NODES:
            raise ValueError(
                f"Сетка содержит {math.prod(shape):,} узлов; допустимо не более "
                f"{MAX_MESH_NODES:,} узлов всего."
            )
        return shape

    def _apply_mesh_preset(self, *_args) -> None:
        if not hasattr(self, "mesh_node_fields"):
            return
        presets = {
            "rough": (9, 7, 7),
            "optimal": (17, 13, 11),
            "precise": (33, 25, 21),
        }
        shape = presets.get(self.mesh_quality.currentData())
        if shape:
            self._updating_mesh_preset = True
            try:
                for key, value in zip(("length", "width", "height"), shape):
                    self.mesh_node_fields[key].setValue(value)
            finally:
                self._updating_mesh_preset = False
        self._update_mesh_summary()

    def _mesh_dimensions_changed(self, *_args) -> None:
        if getattr(self, "_updating_mesh_preset", False):
            return
        if self.mesh_quality.currentData() != "custom":
            self.mesh_quality.setCurrentIndex(
                self.mesh_quality.findData("custom")
            )
        self._update_mesh_summary()

    def _update_mesh_summary(self) -> None:
        if not hasattr(self, "mesh_node_fields"):
            return
        shape = tuple(
            self.mesh_node_fields[key].value()
            for key in ("length", "width", "height")
        )
        nodes = math.prod(shape)
        elements = estimate_tetrahedron_count(shape)
        memory_gb = estimate_mesh_memory_bytes(shape) / 1e9
        if nodes > MAX_MESH_NODES:
            self.mesh_summary.setText(
                f"Сетка: {nodes:,} узлов — превышен общий лимит {MAX_MESH_NODES:,}. "
                f"Тетраэдров: {elements:,}."
            )
            self.mesh_summary.setStyleSheet("color: #b42318;")
            return
        self.mesh_summary.setStyleSheet("")
        self.mesh_summary.setText(
            f"Узлов: {nodes:,} / {MAX_MESH_NODES:,} · "
            f"Тетраэдров: {elements:,} · "
            f"оценка пикового RAM: ≈{memory_gb:.2f} ГБ"
        )

    def _update_effective_width_label(self, *_args) -> None:
        if not hasattr(self, "effective_width_label"):
            return
        device_type = self.device_type.currentData()
        family = self._cad_family if device_type == "cad_custom" else device_type
        if family == "finfet":
            width = self.fields["fin_count"].value() * (
                2 * self.fields["fin_height_nm"].value()
                + self.fields["fin_width_nm"].value()
            ) * 1e-3
        elif family == "gaa_nanosheet":
            width = self.fields["sheet_count"].value() * (
                2 * self.fields["sheet_width_nm"].value()
                + 2 * self.fields["sheet_thickness_nm"].value()
            ) * 1e-3
        elif family == "gaa_nanowire":
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
        selected_type = self.device_type.currentData()
        device_type = (
            self._cad_family if selected_type == "cad_custom" else selected_type
        )
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
            "Импорт кривых характеристик",
            "",
            "Табличные файлы (*.csv *.log *.txt *.dat);;"
            "CSV (*.csv);;Silvaco log (*.log);;Все файлы (*)",
        )
        self._load_curve_paths(paths)

    def _load_curve_examples(self) -> None:
        examples_dir = Path(__file__).resolve().parents[1] / "data" / "examples"
        paths = sorted(
            path
            for suffix in ("*.csv", "*.log")
            for path in examples_dir.rglob(suffix)
        )
        if not paths:
            QMessageBox.information(
                self, "Примеры не найдены", f"В каталоге нет CSV/LOG-примеров:\n{examples_dir}"
            )
            return
        self._load_curve_paths([str(path) for path in paths])

    def _load_curve_paths(self, paths: list[str]) -> None:
        first_new_index = len(self.measurements)
        for path in paths:
            try:
                measurement = load_measurement(path)
            except (OSError, ValueError) as error:
                QMessageBox.warning(self, "Не удалось импортировать кривую", f"{path}\n\n{error}")
                continue
            self.measurements.append(measurement)
            self.measurement_list.addItem(f"{measurement.label}  [{measurement.kind.upper()}]")
        if len(self.measurements) > first_new_index:
            self.measurement_list.setCurrentRow(len(self.measurements) - 1)
            self.statusBar().showMessage(
                f"Загружено кривых: {len(self.measurements) - first_new_index}", 5000
            )

    def _remove_selected_measurements(self) -> None:
        rows = sorted(
            {self.measurement_list.row(item) for item in self.measurement_list.selectedItems()},
            reverse=True,
        )
        if not rows:
            current = self.measurement_list.currentRow()
            if current >= 0:
                rows = [current]
            else:
                self.statusBar().showMessage("Сначала выберите загруженную характеристику", 4000)
                return

        removed = set(rows)
        self.measurements = [
            measurement
            for index, measurement in enumerate(self.measurements)
            if index not in removed
        ]
        self.analysis_results = {
            index - sum(removed_index < index for removed_index in removed): result
            for index, result in self.analysis_results.items()
            if index not in removed
        }
        selected_row = min(rows[-1], len(self.measurements) - 1)
        self._rebuild_measurement_list(selected_row)
        self.statusBar().showMessage(f"Удалено наборов: {len(removed)}", 5000)

    def _clear_measurements(self) -> None:
        count = len(self.measurements)
        if count == 0:
            return
        self.measurements.clear()
        self.analysis_results.clear()
        self.extracted = {}
        self.active_dataset_kind = None
        self._rebuild_measurement_list(-1)
        self.statusBar().showMessage(f"Список очищен, удалено наборов: {count}", 5000)

    def _rebuild_measurement_list(self, current_row: int) -> None:
        self.measurement_list.blockSignals(True)
        self.measurement_list.clear()
        for measurement in self.measurements:
            self.measurement_list.addItem(
                f"{measurement.label}  [{measurement.kind.upper()}]"
            )
        if self.measurements:
            self.measurement_list.setCurrentRow(
                min(max(current_row, 0), len(self.measurements) - 1)
            )
        self.measurement_list.blockSignals(False)
        if self.measurements:
            self._plot_selected(self.measurement_list.currentRow())
        else:
            self.extracted = {}
            self.active_dataset_kind = None
            self._refresh_characteristic_options()
            self._refresh_result_table()

    def _plot_selected(self, index: int) -> None:
        self.extracted = {}
        self.active_dataset_kind = None
        if not 0 <= index < len(self.measurements):
            self._refresh_characteristic_options()
            return
        cached_metrics, _cached_error = self.analysis_results.get(
            index, (None, None)
        )
        if cached_metrics:
            self.extracted = cached_metrics.copy()
            self.active_dataset_kind = self.measurements[index].kind
        measurement = self.measurements[index]
        defaults = {
            "idvg": "idvg",
            "idvd": "idvd",
            "cv": "cv_cgg",
            "igvg": "igvg",
            "gummel": "gummel_ic",
            "bjt_output": "bjt_output",
            "ac": "ac_h21",
            "transient": "transient_id",
            "gate_charge": "gate_charge_time",
            "memristor_iv": "memristor_iv",
            "memristor_resistance": "memristor_resistance",
            "memory_idvg": "memory_idvg",
        }
        self._refresh_characteristic_options(defaults.get(measurement.kind))

    def _series_for_characteristic(self, measurement: Measurement, key: str):
        import numpy as np

        specs = {spec[0]: spec for spec in self.CHARACTERISTIC_SPECS}
        _key, _title, x_column, y_column, _scale = specs[key]
        columns = measurement.columns
        if key == "idvg" and measurement.kind != "idvg":
            return None
        if key in ("idvd", "breakdown") and measurement.kind != "idvd":
            return None
        if key.startswith("cv_") and measurement.kind != "cv":
            return None
        if key.startswith("gummel_") and measurement.kind != "gummel":
            return None
        if key == "bjt_output" and measurement.kind != "bjt_output":
            return None
        if key.startswith("ac_") and measurement.kind != "ac":
            return None
        if key.startswith("transient_") and measurement.kind != "transient":
            return None
        if key.startswith("gate_charge") and measurement.kind != "gate_charge":
            return None
        if key == "memristor_iv" and measurement.kind != "memristor_iv":
            return None
        if key == "memristor_resistance" and measurement.kind != "memristor_resistance":
            return None
        if key == "memory_idvg" and measurement.kind != "memory_idvg":
            return None

        if key == "gummel_beta":
            x = np.asarray(columns["collector_current"], dtype=float)
            collector = np.abs(x)
            base = np.abs(np.asarray(columns["base_current"], dtype=float))
            y = np.divide(collector, base, out=np.zeros_like(collector), where=base > 0)
            x = collector
        elif key.startswith("gate_charge"):
            time = np.asarray(columns["time"], dtype=float)
            current = np.asarray(columns["gate_current"], dtype=float)
            order = np.argsort(time)
            time, current = time[order], current[order]
            y = np.concatenate(
                ([0.0], np.cumsum((current[1:] + current[:-1]) * np.diff(time) * 0.5))
            )
            if key == "gate_charge_vg":
                if "gate_voltage" not in columns:
                    return None
                x = np.asarray(columns["gate_voltage"], dtype=float)[order]
            else:
                x = time
            y_column = "gate_charge"
        elif y_column == "h21" and "h21" not in columns and "h21_db" in columns:
            x = np.asarray(columns[x_column], dtype=float)
            y = 10 ** (np.asarray(columns["h21_db"], dtype=float) / 20)
        elif y_column == "mason_u" and "mason_u" not in columns and "mason_u_db" in columns:
            x = np.asarray(columns[x_column], dtype=float)
            y = 10 ** (np.asarray(columns["mason_u_db"], dtype=float) / 10)
        elif y_column in ("s11", "s12", "s21", "s22") and y_column not in columns:
            db_column = f"{y_column}_db"
            if db_column not in columns:
                return None
            x = np.asarray(columns[x_column], dtype=float)
            y = np.asarray(columns[db_column], dtype=float)
        elif x_column == "gate_voltage" and y_column == "drain_current" and key in ("idvg", "transient_id"):
            x = np.asarray(columns[x_column] if key == "idvg" else columns["time"], dtype=float)
            y = np.asarray(columns[y_column], dtype=float)
        elif x_column == "drain_voltage" and y_column == "drain_current":
            x = np.asarray(columns[x_column], dtype=float)
            y = np.asarray(columns[y_column], dtype=float)
        else:
            if x_column not in columns or y_column not in columns:
                return None
            x = np.asarray(columns[x_column], dtype=float)
            y = np.asarray(columns[y_column], dtype=float)

        if x.size != y.size or x.size < 2:
            return None
        if y_column in ("drain_current", "gate_current", "collector_current", "base_current"):
            y = np.abs(y)
        if key == "transient_vg":
            x = np.asarray(columns["time"], dtype=float)
            y = np.asarray(columns["gate_voltage"], dtype=float)
        if key not in ("memristor_iv", "memory_idvg"):
            order = np.argsort(x)
            x, y = x[order], y[order]
        return x, y

    def _refresh_characteristic_options(self, preferred: str | None = None) -> None:
        current = self.characteristic_selector.currentData()
        options = []
        for key, title, _x_column, _y_column, _scale in self.CHARACTERISTIC_SPECS:
            count = sum(
                self._series_for_characteristic(measurement, key) is not None
                for measurement in self.measurements
            )
            options.append(
                (key, f"{title}  ·  {count} наб." if count else f"{title}  ·  нет данных")
            )

        self.characteristic_selector.blockSignals(True)
        self.characteristic_selector.clear()
        for key, title in options:
            self.characteristic_selector.addItem(title, key)
        selected_key = preferred if any(key == preferred for key, _title in options) else current
        index = self.characteristic_selector.findData(selected_key)
        self.characteristic_selector.setCurrentIndex(index if index >= 0 else 0)
        self.characteristic_selector.blockSignals(False)
        self._plot_selected_characteristic()

    def _plot_selected_characteristic(self, *_args) -> None:
        self.axes.clear()
        key = self.characteristic_selector.currentData()
        specs = {spec[0]: spec for spec in self.CHARACTERISTIC_SPECS}
        spec = specs.get(key)
        if spec is None:
            self.characteristic_status.setText(
                "Импортируйте кривые, чтобы выбрать характеристики"
            )
            self.canvas.draw_idle()
            return
        _key, title, x_column, y_column, scale = spec
        axis_labels = {
            "gate_voltage": "Напряжение затвора (В)",
            "drain_voltage": "Напряжение стока (В)",
            "gate_current": "Ток затвора (А)",
            "drain_current": "Ток стока (А)",
            "capacitance": "Ёмкость Cgg (Ф)",
            "gate_source_capacitance": "Ёмкость Cgs (Ф)",
            "gate_drain_capacitance": "Ёмкость Cgd (Ф)",
            "gate_bulk_capacitance": "Ёмкость Cgb (Ф)",
            "emitter_capacitance": "Ёмкость эмиттерного перехода Cje (Ф)",
            "collector_capacitance": "Ёмкость коллекторного перехода Cjc (Ф)",
            "vbe": "Напряжение база–эмиттер (В)",
            "vce": "Напряжение коллектор–эмиттер (В)",
            "collector_current": "Ток коллектора Ic (А)",
            "base_current": "Ток базы Ib (А)",
            "frequency": "Частота (Гц)",
            "h21": "Коэффициент передачи тока |h21|",
            "h21_db": "Коэффициент передачи тока h21 (дБ)",
            "mason_u": "Коэффициент Mason U",
            "mason_u_db": "Коэффициент Mason U (дБ)",
            "s11": "S11",
            "s11_db": "S11 (дБ)",
            "s12": "S12",
            "s12_db": "S12 (дБ)",
            "s21": "S21",
            "s21_db": "S21 (дБ)",
            "s22": "S22",
            "s22_db": "S22 (дБ)",
            "y11": "Y11 (С)",
            "y12": "Y12 (С)",
            "y21": "Y21 (С)",
            "y22": "Y22 (С)",
            "z11": "Z11 (Ом)",
            "z12": "Z12 (Ом)",
            "z21": "Z21 (Ом)",
            "z22": "Z22 (Ом)",
            "time": "Время (с)",
            "gate_charge": "Заряд затвора Qg (Кл)",
            "voltage": "Напряжение на мемристоре (В)",
            "current": "Ток мемристора (А)",
            "resistance": "Сопротивление (Ом)",
        }
        x_label = axis_labels.get(x_column, x_column)
        if key.startswith("gate_charge"):
            y_label = "Заряд затвора Qg (Кл)"
            x_label = "Напряжение затвора (В)" if key == "gate_charge_vg" else "Время (с)"
        elif key == "gummel_beta":
            y_label = "Коэффициент усиления по току β"
        else:
            y_label = axis_labels.get(y_column, y_column)
        plotted = 0
        for measurement in self.measurements:
            series = self._series_for_characteristic(measurement, key)
            if series is None:
                continue
            x_values, y_values = series
            if scale == "log":
                valid = y_values > 0
                x_values, y_values = x_values[valid], y_values[valid]
            if len(x_values) < 2:
                continue
            self.axes.plot(
                x_values,
                y_values,
                marker="o",
                markersize=2.5,
                linewidth=1.5,
                label=measurement.label,
            )
            plotted += 1
        self.axes.set_title(title, loc="left", fontsize=11, fontweight="bold")
        self.axes.set_xlabel(x_label)
        self.axes.set_ylabel(y_label)
        if scale == "log":
            self.axes.set_yscale("log")
        elif scale == "logx":
            self.axes.set_xscale("log")
        if plotted:
            self.axes.legend()
            self.characteristic_status.setText(
                f"{plotted} набор(а) данных · выберите характеристику из списка"
            )
        else:
            requirements = self._characteristic_requirements(key)
            self.axes.text(
                0.5,
                0.5,
                f"Нет данных для графика\nТребуемые столбцы: {requirements}",
                transform=self.axes.transAxes,
                ha="center",
                va="center",
                color="#777777",
                wrap=True,
            )
            self.characteristic_status.setText(
                f"График доступен · импортируйте данные со столбцами: {requirements}"
            )
        self.axes.grid(True, which="both", alpha=0.3)
        self.figure.tight_layout()
        self.canvas.draw_idle()

    @staticmethod
    def _characteristic_requirements(key: str) -> str:
        requirements = {
            "idvg": "Vg, Id [и Vd]",
            "idvd": "Vd, Id [и Vg для указания смещения]",
            "igvg": "Vg, Ig",
            "cv_cgg": "Vg, Cgg (или Cg)",
            "cv_cgs": "Vg, Cgs",
            "cv_cgd": "Vg, Cgd",
            "cv_cgb": "Vg, Cgb",
            "gummel_ic": "Vbe, Ic, Ib",
            "gummel_ib": "Vbe, Ic, Ib",
            "gummel_beta": "Vbe, Ic, Ib",
            "bjt_output": "Vce, Ic",
            "bjt_cje": "Vbe, Cje",
            "bjt_cjc": "Vce, Cjc",
            "ac_h21": "Frequency, H21 или H21_dB",
            "ac_h21_db": "Frequency, H21_dB",
            "ac_mason": "Frequency, U или Mason_U",
            "ac_mason_db": "Frequency, U_dB",
            "ac_s11": "Frequency, S11",
            "ac_s11_db": "Frequency, S11_dB",
            "ac_s12": "Frequency, S12",
            "ac_s12_db": "Frequency, S12_dB",
            "ac_s21": "Frequency, S21",
            "ac_s21_db": "Frequency, S21_dB",
            "ac_s22": "Frequency, S22",
            "ac_s22_db": "Frequency, S22_dB",
            "ac_y11": "Frequency, Y11",
            "ac_y12": "Frequency, Y12",
            "ac_y21": "Frequency, Y21",
            "ac_y22": "Frequency, Y22",
            "ac_z11": "Frequency, Z11",
            "ac_z12": "Frequency, Z12",
            "ac_z21": "Frequency, Z21",
            "ac_z22": "Frequency, Z22",
            "transient_id": "time, Id",
            "transient_vg": "time, Vg",
            "gate_charge_vg": "time, Ig, Vg",
            "gate_charge_time": "time, Ig",
            "breakdown": "Vd, Id",
        }
        return requirements.get(key, "соответствующие столбцы")

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
            self.extracted = extract_curve_metrics(measurement, device)
        except (ValueError, RuntimeError) as error:
            QMessageBox.warning(self, "Ошибка расчёта параметров", str(error))
            return
        self.active_dataset_kind = measurement.kind
        self.analysis_results[index] = (self.extracted.copy(), None)
        self._refresh_result_table()

    def _fit_all(self) -> None:
        if not self.measurements:
            QMessageBox.information(
                self, "Нет загруженных кривых", "Сначала импортируйте данные или загрузите примеры."
            )
            return
        try:
            device = self._device_parameters()
        except ValueError as error:
            QMessageBox.warning(self, "Ошибка параметров структуры", str(error))
            return

        self.analysis_results.clear()
        for index, measurement in enumerate(self.measurements):
            try:
                metrics = extract_curve_metrics(measurement, device)
            except (ValueError, RuntimeError) as error:
                self.analysis_results[index] = (None, str(error))
            else:
                self.analysis_results[index] = (metrics, None)
        selected = self.measurement_list.currentRow()
        if selected in self.analysis_results:
            metrics, _error = self.analysis_results[selected]
            self.extracted = metrics.copy() if metrics else {}
            self.active_dataset_kind = (
                self.measurements[selected].kind if metrics else None
            )
        self._refresh_result_table()
        successful = sum(bool(metrics) for metrics, _error in self.analysis_results.values())
        failed = len(self.analysis_results) - successful
        self.statusBar().showMessage(
            f"Рассчитано наборов: {successful}; ошибок: {failed}. "
            "Метрики извлечены из загруженных кривых, а не рассчитаны TCAD-моделью.",
            10000,
        )

    def _refresh_result_table(self) -> None:
        metric_names = {
            "VTH0": "Пороговое напряжение Vth (модель), В",
            "VTH_CONSTANT_CURRENT_V": "Пороговое напряжение Vth (постоянный ток), В",
            "U0": "Эффективная подвижность U0, см²/(В·с)",
            "NFACTOR": "Коэффициент подпороговой модели",
            "RMSE_LOG10": "Ошибка подгонки log₁₀(Id)",
            "ION": "Ток открытого состояния Ion, А",
            "ION_A": "Ток открытого состояния Ion, А",
            "IOFF": "Ток утечки Ioff, А",
            "IOFF_A": "Ток утечки Ioff, А",
            "ION_IOFF": "Отношение Ion/Ioff",
            "GM_MAX": "Максимальная крутизна gm, С",
            "GM_MAX_S": "Максимальная крутизна gm, С",
            "SS_MV_DEC": "Подпороговый размах SS, мВ/дек",
            "IDS_MAX": "Максимальный ток стока, А",
            "VDS_AT_MAX": "Напряжение стока при максимальном токе, В",
            "RON_OHM": "Сопротивление во включенном состоянии RDS(on), Ом",
            "GDS_S": "Выходная проводимость gds, С",
            "GCE_MAX_S": "Максимальная выходная проводимость BJT, С",
            "VBR_KNEE_V": "Оценка колена пробоя, В",
            "CGG": "Средняя ёмкость затвора Cgg, Ф",
            "CGSO": "Перекрывающая ёмкость затвор–исток на ширину, Ф/м",
            "CGDO": "Перекрывающая ёмкость затвор–сток на ширину, Ф/м",
            "CINTRINSIC": "Оценка собственной ёмкости канала, Ф",
            "CGS_MEAN_F": "Средняя ёмкость затвор–исток Cgs, Ф",
            "CGD_MEAN_F": "Средняя ёмкость затвор–сток Cgd, Ф",
            "CGB_MEAN_F": "Средняя ёмкость затвор–подложка Cgb, Ф",
            "CJE_MEAN_F": "Средняя ёмкость эмиттерного перехода Cje, Ф",
            "CJC_MEAN_F": "Средняя ёмкость коллекторного перехода Cjc, Ф",
            "IG_MAX_A": "Максимальный ток утечки затвора Ig, А",
            "IG_AT_MAX_GATE_A": "Ток затвора при максимальном Vg, А",
            "IG_MIN_NONZERO_A": "Минимальный ненулевой ток затвора, А",
            "IG_MAX_MIN_RATIO": "Отношение максимального/минимального Ig",
            "IC_MAX_A": "Максимальный ток коллектора Ic, А",
            "IB_MAX_A": "Максимальный ток базы Ib, А",
            "BETA_MAX": "Максимальный коэффициент усиления β",
            "BETA_AT_MAX_IC": "Коэффициент β при максимальном Ic",
            "FT_HZ": "Граничная частота fT, Гц",
            "FMAX_HZ": "Максимальная частота генерации fmax, Гц",
            "H21_MAX": "Максимальный коэффициент передачи тока |h21|",
            "MASON_U_MAX": "Максимальный коэффициент Mason U",
            "S21_MAX": "Максимальный |S21|",
            "TON_10_90_S": "Время включения 10–90%, с",
            "TOFF_90_10_S": "Время выключения 90–10%, с",
            "VG_MIN_V": "Минимальное напряжение затвора, В",
            "VG_MAX_V": "Максимальное напряжение затвора, В",
            "QG_C": "Заряд затвора Qg, Кл",
            "VG_SWING_V": "Размах напряжения затвора, В",
            "VGS_BIAS": "Напряжение затвора смещения, В",
            "RACCESS_OHM": "Сопротивление доступа, Ом",
            "MEMRISTOR_R_ON_OHM": "Мемристор: сопротивление LRS, Ом",
            "MEMRISTOR_R_OFF_OHM": "Мемристор: сопротивление HRS, Ом",
            "MEMRISTOR_R_MIN_OHM": "Минимальное сопротивление, Ом",
            "MEMRISTOR_R_MAX_OHM": "Максимальное сопротивление, Ом",
            "MEMRISTOR_ON_OFF_RATIO": "Отношение HRS/LRS",
            "MEMRISTOR_HYSTERESIS_AREA_A_V": "Площадь петли гистерезиса, А·В",
            "MEMRISTOR_I_MAX_A": "Максимальный ток мемристора, А",
            "MEMTRANSISTOR_ION_A": "Мемтранзистор: Ion, А",
            "MEMTRANSISTOR_IOFF_A": "Мемтранзистор: Ioff, А",
            "MEMTRANSISTOR_ON_OFF_RATIO": "Мемтранзистор: Ion/Ioff",
            "MEMTRANSISTOR_HYSTERESIS_AREA_A_V": "Площадь петли Id–Vg, А·В",
            "MEMORY_STATE_SWING": "Размах состояния памяти",
        }
        rows: list[tuple[str, str, str]] = []
        for index, measurement in enumerate(self.measurements):
            if index not in self.analysis_results:
                continue
            metrics, error = self.analysis_results[index]
            if error is not None:
                rows.append((measurement.label, "Ошибка анализа", error))
            elif metrics:
                for key, value in metrics.items():
                    rows.append(
                        (measurement.label, metric_names.get(key, key), f"{value:.8g}")
                    )
        self.result_table.setRowCount(len(rows))
        for row, values in enumerate(rows):
            for column, value in enumerate(values):
                self.result_table.setItem(row, column, QTableWidgetItem(value))
        self.result_table.resizeColumnsToContents()
        self.result_table.horizontalHeader().setStretchLastSection(True)

    def _export_model(self) -> None:
        try:
            device = self._device_parameters()
            if device.device_type in ("memristor", "memtransistor"):
                text = generate_memory_verilog_a(device)
                path, _ = QFileDialog.getSaveFileName(
                    self,
                    "Экспорт Verilog-A модели памяти",
                    f"{device.name}.va",
                    "Verilog-A модель (*.va);;Все файлы (*)",
                )
                if not path:
                    return
                if Path(path).suffix.lower() != ".va":
                    path = f"{path}.va"
                try:
                    Path(path).write_text(text, encoding="utf-8")
                except OSError as error:
                    QMessageBox.critical(
                        self, "Ошибка экспорта Verilog-A", str(error)
                    )
                    return
                self.statusBar().showMessage(
                    f"Verilog-A модель сохранена: {path}. "
                    "Проверьте компиляцию в целевом симуляторе.",
                    10000,
                )
                return
            v_graphs, c_graphs, id_graphs = self._compact_model_curve_inputs()
            selected_type = self.DEVICE_LABELS[self.device_type.currentData()]
            extractor = create_extractor(device, selected_type)
            text = extractor.extract(v_graphs, c_graphs, id_graphs)
            self.extracted = extractor.metrics.copy()
            extension = "spice"
            file_filter = "SPICE-карта или subckt (*.spice *.cir *.model)"
        except (ValueError, RuntimeError, NotImplementedError) as error:
            QMessageBox.warning(self, "Не удалось извлечь SPICE-модель", str(error))
            return
        path, _ = QFileDialog.getSaveFileName(
            self,
            "Экспорт извлечённой компактной модели",
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
        limitations = (
            " Поведенческая подгонка; это не ASM-HEMT/MVSG."
            if device.device_type == "hemt"
            else " Поведенческая подгонка, не физическая BTBT-модель."
            if device.device_type == "tfet"
            else " Полиномиальная поведенческая модель CAD."
            if self.device_type.currentData() == "cad_custom"
            else ""
        )
        self.statusBar().showMessage(
            f"{extractor.model_description} сохранена: {path}.{limitations}", 10000
        )

    def _compact_model_curve_inputs(self):
        v_graphs: dict[str, list[float]] = {}
        id_graphs: dict[str, object] = {}
        linear_index = 0
        saturation_index = 0
        c_graphs: dict[str, list[float]] = {}
        for measurement in self.measurements:
            columns = measurement.columns
            if measurement.kind == "idvg":
                vg = columns.get("gate_voltage", measurement.x)
                ids = columns.get("drain_current", measurement.y)
                vd = columns.get("drain_voltage", measurement.drain_voltage)
                representative_vd = (
                    float(sum(vd) / len(vd)) if vd else 0.05
                )
                mode = "linear" if representative_vd <= 0.2 else "saturation"
                index = linear_index if mode == "linear" else saturation_index
                key = f"{mode}_{index}"
                id_graphs[key] = {"vg": list(vg), "id": list(ids), "vd": vd}
                v_graphs[key] = list(vg)
                if mode == "linear":
                    linear_index += 1
                else:
                    saturation_index += 1
            elif measurement.kind == "cv" and not c_graphs:
                c_graphs = {
                    "vg": list(columns.get("gate_voltage", measurement.x)),
                    "cgg": list(columns.get("capacitance", measurement.y)),
                }
        return v_graphs, c_graphs, id_graphs

    def _save_plot(self) -> None:
        key = self.characteristic_selector.currentData()
        title = next(
            (title for item_key, title, *_rest in self.CHARACTERISTIC_SPECS if item_key == key),
            "characteristic",
        )
        file_stem = "".join(
            character if character.isalnum() or character in "-_" else "_"
            for character in key or "characteristic"
        ).strip("_")
        path, _ = QFileDialog.getSaveFileName(
            self,
            "Экспорт графика для публикации",
            f"{file_stem or 'characteristic'}.pdf",
            "Изображение PNG (*.png);;Документ PDF (*.pdf)",
        )
        if path:
            try:
                output = Path(path)
                self.figure.savefig(
                    output,
                    dpi=300,
                    bbox_inches="tight",
                    facecolor=self.figure.get_facecolor(),
                )
            except OSError as error:
                QMessageBox.critical(self, "Не удалось сохранить график", str(error))
                return
            self.statusBar().showMessage(
                f"График «{title}» сохранён: {output}", 5000
            )

    def _refresh_gpu_status(self) -> None:
        interpreter = sys.executable
        try:
            import torch
        except ImportError:
            cuda_status = (
                "Расчёт на CUDA недоступен: PyTorch не установлен в "
                f"активном Python ({interpreter})."
            )
        else:
            if torch.cuda.is_available():
                cuda_status = (
                    "CUDA-решатель готов: "
                    f"{torch.cuda.get_device_name(0)} (FP64; "
                    f"PyTorch {torch.__version__}, CUDA {torch.version.cuda})."
                )
            else:
                cuda_status = (
                    f"CUDA-решатель недоступен в активном Python ({interpreter}): "
                    f"PyTorch {torch.__version__}, CUDA runtime "
                    f"{torch.version.cuda or 'CPU-only'}. Создайте/выберите "
                    "окружение проекта и установите зависимости из requirements.txt "
                    "или environment.yml."
                )
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
            device_line = result.stdout.strip().splitlines()[0]
            gpu_name = device_line.split(",")[0].casefold()
            precision_note = (
                " Расчёт FEM использует FP64; доступная VRAM учитывается автоматически."
                if "4070 ti super" in gpu_name or "4090" in gpu_name
                else ""
            )
            self.gpu_status.setText(
                f"GPU обнаружен: {device_line}. "
                f"Backend расчёта указывается в результатах.{precision_note}\n"
                f"{cuda_status}"
            )
        except (OSError, subprocess.SubprocessError, IndexError):
            self.gpu_status.setText(
                "Сведения о GPU недоступны (nvidia-smi не найден).\n"
                f"{cuda_status}"
            )


def main() -> int:
    app = QApplication(sys.argv)
    window = TransistorWorkbench()
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())