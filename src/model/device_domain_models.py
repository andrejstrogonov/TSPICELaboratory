from __future__ import annotations

from django.core.validators import MaxValueValidator, MinValueValidator
from django.db import models
from django.db.models import F, Q

from .catalog_models import (
    CharacteristicSpec,
    DeviceFamilyChoices,
    DeviceLabel,
    GeometryField,
    MetricLabel,
    SolverCatalog,
)
from .operational_models import (
    DeviceStructure,
    ExtractionSession,
    MeasurementSet,
    SpiceModel,
)


# =============================================================================
# D-2 · Базовый контракт параметров, эффективная геометрия и матрица решателей
# =============================================================================

class DeviceParametersCore(models.Model):
    """
    Базовый физико-геометрический контракт структуры (Группа G0, D-1 / D-2).
    Общие параметры, используемые всеми 10 семействами приборов.
    """
    parameter_row_id = models.BigAutoField(primary_key=True)
    structure = models.OneToOneField(
        DeviceStructure,
        on_delete=models.CASCADE,
        related_name="core_parameters",
        db_column="structure_id",
        verbose_name="Экземпляр структуры",
    )
    device_type = models.ForeignKey(
        DeviceLabel,
        on_delete=models.PROTECT,
        related_name="core_parameter_rows",
        db_column="device_type",
        verbose_name="Семейство прибора",
    )
    name = models.CharField(
        max_length=128,
        unique=True,
        verbose_name="Уникальное имя (без пробелов)",
    )
    epsilon_r = models.FloatField(
        default=3.9,
        validators=[MinValueValidator(1e-6)],
        verbose_name="Относительная диэлектрическая проницаемость εr",
    )
    electron_mobility = models.FloatField(
        default=1400.0,
        validators=[MinValueValidator(1e-6)],
        verbose_name="Подвижность электронов μn, см²/(В·с)",
    )
    hole_mobility = models.FloatField(
        default=450.0,
        validators=[MinValueValidator(1e-6)],
        verbose_name="Подвижность дырок μp, см²/(В·с)",
    )
    bandgap = models.FloatField(
        default=1.12,
        validators=[MinValueValidator(1e-6)],
        verbose_name="Ширина запрещённой зоны Eg, эВ",
    )
    doping = models.FloatField(
        default=1e17,
        validators=[MinValueValidator(1e-6)],
        verbose_name="Концентрация легирования, см⁻³",
    )
    tox_nm = models.FloatField(
        default=2.0,
        validators=[MinValueValidator(1e-6)],
        verbose_name="Эквивалентная толщина оксида EOT (Tox), нм",
    )
    width_um = models.FloatField(
        default=1.0,
        validators=[MinValueValidator(1e-6)],
        verbose_name="Ширина канала W, мкм",
    )
    length_um = models.FloatField(
        default=0.1,
        validators=[MinValueValidator(1e-6)],
        verbose_name="Длина канала L, мкм",
    )
    temperature_k = models.FloatField(
        default=300.0,
        validators=[MinValueValidator(1e-3)],
        verbose_name="Температура T, К",
    )
    barrier_material = models.CharField(
        max_length=64,
        default="AlGaN/GaN",
        blank=True,
        verbose_name="Материал барьера (HEMT)",
    )
    junction_material = models.CharField(
        max_length=64,
        default="Si/Si",
        blank=True,
        verbose_name="Материал гетероперехода (TFET)",
    )
    geometry_data = models.JSONField(
        default=dict,
        blank=True,
        verbose_name="Снимок геометрии JSON",
    )

    class Meta:
        db_table = "device_parameters_core"
        verbose_name = "Базовые параметры прибора (G0)"
        verbose_name_plural = "Базовые параметры приборов (G0)"
        constraints = [
            models.CheckConstraint(condition=Q(epsilon_r__gt=0), name="chk_core_eps_gt_0"),
            models.CheckConstraint(condition=Q(electron_mobility__gt=0), name="chk_core_mun_gt_0"),
            models.CheckConstraint(condition=Q(hole_mobility__gt=0), name="chk_core_mup_gt_0"),
            models.CheckConstraint(condition=Q(bandgap__gt=0), name="chk_core_eg_gt_0"),
            models.CheckConstraint(condition=Q(doping__gt=0), name="chk_core_doping_gt_0"),
            models.CheckConstraint(condition=Q(tox_nm__gt=0), name="chk_core_tox_gt_0"),
            models.CheckConstraint(condition=Q(width_um__gt=0), name="chk_core_w_gt_0"),
            models.CheckConstraint(condition=Q(length_um__gt=0), name="chk_core_l_gt_0"),
            models.CheckConstraint(condition=Q(temperature_k__gt=0), name="chk_core_temp_gt_0"),
        ]

    def __str__(self) -> str:
        return f"{self.name} ({self.device_type_id})"


class EffectiveGeometry(models.Model):
    """
    Вычисляемые геометрические величины W_eff и C_ox (D-2),
    участвующие в фите least_squares и итоговых метриках.
    """
    geometry_id = models.BigAutoField(primary_key=True)
    structure = models.OneToOneField(
        DeviceStructure,
        on_delete=models.CASCADE,
        related_name="effective_geometry",
        db_column="structure_id",
        verbose_name="Экземпляр структуры",
    )
    w_eff_um = models.FloatField(verbose_name="Эффективная ширина W_eff, мкм")
    c_ox_f_m2 = models.FloatField(verbose_name="Удельная ёмкость оксида C_ox, Ф/м²")
    area_factor = models.FloatField(default=1.0, verbose_name="Коэффициент площади / мультипликатор")
    formula_ref = models.CharField(
        max_length=128,
        blank=True,
        default="",
        verbose_name="Ссылка на формулу расчёта",
    )

    class Meta:
        db_table = "effective_geometry"
        verbose_name = "Эффективная геометрия"
        verbose_name_plural = "Эффективные геометрии"

    def __str__(self) -> str:
        return f"EffectiveGeometry(structure={self.structure_id}, W_eff={self.w_eff_um} um)"


class SolverRoute(models.Model):
    """
    Маршрутизация и матрица поддержки решателей по семействам (D-2 / D-10 support_matrix).
    """
    route_id = models.BigAutoField(primary_key=True)
    device_type = models.ForeignKey(
        DeviceLabel,
        on_delete=models.CASCADE,
        related_name="solver_routes",
        db_column="device_type",
        verbose_name="Семейство прибора",
    )
    solver = models.ForeignKey(
        SolverCatalog,
        on_delete=models.CASCADE,
        related_name="device_routes",
        db_column="solver_id",
        verbose_name="Решатель",
    )
    supported = models.BooleanField(default=True, verbose_name="Поддерживается физикой/кодом")
    compatibility_level = models.SmallIntegerField(
        null=True,
        blank=True,
        verbose_name="Уровень SPICE-совместимости (например, 54 или 72)",
    )
    limitation_note = models.TextField(
        blank=True,
        default="",
        verbose_name="Примечание об ограничениях / причине отказа",
    )

    class Meta:
        db_table = "solver_route"
        verbose_name = "Маршрут решателя для семейства"
        verbose_name_plural = "Матрица маршрутов решателей (D-10)"
        constraints = [
            models.UniqueConstraint(
                fields=["device_type", "solver"],
                name="uq_solver_route_device_solver",
            ),
        ]

    def __str__(self) -> str:
        mark = "✔" if self.supported else "−"
        return f"{self.device_type_id} × {self.solver_id} [{mark}]"


# =============================================================================
# D-10 · Семейства метрик и привязка к семействам приборов
# =============================================================================

class MetricFamilyCode(models.TextChoices):
    MF1 = "MF1", "MF1 · Подключение (ION, IOFF, VTH)"
    MF2 = "MF2", "MF2 · Передача (GM_MAX, SS, U0, NFACTOR, VTH0)"
    MF3 = "MF3", "MF3 · Выход (RON, GDS, IDS_MAX, VBR_KNEE)"
    MF4 = "MF4", "MF4 · Ёмкости (CGG, CGSO, CGDO, CGS/CGD/CGB)"
    MF5 = "MF5", "MF5 · ВЧ (FT, FMAX, H21_MAX, MASON_U_MAX)"
    MF6 = "MF6", "MF6 · Переключение (TON, TOFF, QG)"
    MF7 = "MF7", "MF7 · BJT (BETA_MAX, IC_MAX, IB_MAX)"
    MF8 = "MF8", "MF8 · Тепло (TMAX, DELTA_T, RTH)"
    MF9 = "MF9", "MF9 · Память (MEMRISTOR_*, MEMTRANSISTOR_*, SWING)"
    MF10 = "MF10", "MF10 · Качество модели (RMSE_LOG10, fitness_score)"


class MetricFamily(models.Model):
    """
    Классификатор групп метрик MF1..MF10 (D-10).
    """
    family_code = models.CharField(
        max_length=8,
        primary_key=True,
        choices=MetricFamilyCode.choices,
        verbose_name="Код группы метрик",
    )
    display_name = models.CharField(max_length=128, verbose_name="Наименование группы")
    description = models.TextField(blank=True, default="", verbose_name="Перечень ключевых метрик")

    class Meta:
        db_table = "metric_families"
        verbose_name = "Группа метрик (MF)"
        verbose_name_plural = "Группы метрик (MF1..MF10)"

    def __str__(self) -> str:
        return f"{self.family_code}: {self.display_name}"


class MetricBinding(models.Model):
    """
    Привязка допустимых групп метрик к семейству прибора (D-10 metric_binding).
    """
    binding_id = models.BigAutoField(primary_key=True)
    device_type = models.ForeignKey(
        DeviceLabel,
        on_delete=models.CASCADE,
        related_name="metric_bindings",
        db_column="device_id",
        verbose_name="Семейство прибора",
    )
    metric_family = models.ForeignKey(
        MetricFamily,
        on_delete=models.CASCADE,
        related_name="device_bindings",
        db_column="family_code",
        verbose_name="Группа метрик",
    )

    class Meta:
        db_table = "metric_binding"
        verbose_name = "Привязка группы метрик"
        verbose_name_plural = "Привязки групп метрик к семействам"
        constraints = [
            models.UniqueConstraint(
                fields=["device_type", "metric_family"],
                name="uq_metric_binding_device_family",
            ),
        ]

    def __str__(self) -> str:
        return f"{self.device_type_id} -> {self.metric_family_id}"


# =============================================================================
# D-3 · FD-SOI MOSFET (типизированный профиль и производные величины)
# =============================================================================

class FdSoiParameters(models.Model):
    """
    Специфичные параметры геометрии FD-SOI MOSFET (D-3).
    """
    structure = models.OneToOneField(
        DeviceStructure,
        on_delete=models.CASCADE,
        primary_key=True,
        related_name="fd_soi_params",
        db_column="structure_id",
    )
    body_thickness_nm = models.FloatField(
        default=7.0,
        validators=[MinValueValidator(0.1), MaxValueValidator(1000.0)],
        verbose_name="Толщина кремниевого тела T_si, нм",
    )
    buried_oxide_nm = models.FloatField(
        default=25.0,
        validators=[MinValueValidator(0.1), MaxValueValidator(1e6)],
        verbose_name="Толщина скрытого оксида BOX, нм",
    )
    back_gate_voltage = models.FloatField(
        default=0.0,
        validators=[MinValueValidator(-100.0), MaxValueValidator(100.0)],
        verbose_name="Напряжение обратного затвора V_bg, В",
    )
    # Производные расчётные величины (fd_soi_derived)
    c_box_f_m2 = models.FloatField(null=True, blank=True, verbose_name="Удельная ёмкость BOX, Ф/м²")
    gamma_back = models.FloatField(null=True, blank=True, verbose_name="Эффективность back-gate ΔVth/ΔVbg")

    class Meta:
        db_table = "fd_soi_parameters"
        verbose_name = "Параметры FD-SOI MOSFET (D-3)"
        verbose_name_plural = "Параметры FD-SOI MOSFET (D-3)"
        constraints = [
            models.CheckConstraint(condition=Q(body_thickness_nm__gte=0.1, body_thickness_nm__lte=1000.0), name="chk_fdsoi_tsi"),
            models.CheckConstraint(condition=Q(buried_oxide_nm__gte=0.1, buried_oxide_nm__lte=1e6), name="chk_fdsoi_tbox"),
            models.CheckConstraint(condition=Q(back_gate_voltage__gte=-100.0, back_gate_voltage__lte=100.0), name="chk_fdsoi_vbg"),
        ]


# =============================================================================
# D-4 · FinFET (multigate, BSIM-CMG / LEVEL=72)
# =============================================================================

class FinfetParameters(models.Model):
    """
    Специфичные параметры геометрии FinFET (D-4) с инвариантом fin_pitch_nm >= fin_width_nm.
    """
    structure = models.OneToOneField(
        DeviceStructure,
        on_delete=models.CASCADE,
        primary_key=True,
        related_name="finfet_params",
        db_column="structure_id",
    )
    fin_width_nm = models.FloatField(
        default=8.0,
        validators=[MinValueValidator(0.1), MaxValueValidator(1000.0)],
        verbose_name="Ширина ребра W_fin, нм",
    )
    fin_height_nm = models.FloatField(
        default=40.0,
        validators=[MinValueValidator(0.1), MaxValueValidator(1e6)],
        verbose_name="Высота ребра H_fin, нм",
    )
    fin_count = models.IntegerField(
        default=2,
        validators=[MinValueValidator(1), MaxValueValidator(100000)],
        verbose_name="Количество рёбер N_fin",
    )
    fin_pitch_nm = models.FloatField(
        default=30.0,
        validators=[MinValueValidator(0.1), MaxValueValidator(1e6)],
        verbose_name="Шаг рёбер P_fin, нм",
    )
    # Производные величины (finfet_derived)
    perimeter_nm = models.FloatField(null=True, blank=True, verbose_name="Эффективный периметр, нм")
    corner_refinement = models.BooleanField(default=True, verbose_name="Сгущение сетки на углах ребра")

    class Meta:
        db_table = "finfet_parameters"
        verbose_name = "Параметры FinFET (D-4)"
        verbose_name_plural = "Параметры FinFET (D-4)"
        constraints = [
            models.CheckConstraint(
                condition=Q(fin_pitch_nm__gte=F("fin_width_nm")),
                name="chk_finfet_pitch_ge_width",
            ),
            models.CheckConstraint(condition=Q(fin_count__gte=1, fin_count__lte=100000), name="chk_finfet_count"),
        ]


# =============================================================================
# D-5 · GAA Nanosheet и GAA Nanowire
# =============================================================================

class GaaNanosheetParameters(models.Model):
    """
    Специфичные параметры GAA Nanosheet (D-5) с инвариантом sheet_pitch_nm >= sheet_thickness_nm.
    """
    structure = models.OneToOneField(
        DeviceStructure,
        on_delete=models.CASCADE,
        primary_key=True,
        related_name="gaa_nanosheet_params",
        db_column="structure_id",
    )
    sheet_width_nm = models.FloatField(
        default=30.0,
        validators=[MinValueValidator(0.1), MaxValueValidator(1e6)],
        verbose_name="Ширина нанолиста W_ns, нм",
    )
    sheet_thickness_nm = models.FloatField(
        default=8.0,
        validators=[MinValueValidator(0.1), MaxValueValidator(1000.0)],
        verbose_name="Толщина нанолиста T_ns, нм",
    )
    sheet_count = models.IntegerField(
        default=3,
        validators=[MinValueValidator(1), MaxValueValidator(100000)],
        verbose_name="Количество нанолистов N_ns",
    )
    sheet_pitch_nm = models.FloatField(
        default=30.0,
        validators=[MinValueValidator(0.1), MaxValueValidator(1e6)],
        verbose_name="Шаг нанолистов P_ns, нм",
    )
    perimeter_nm = models.FloatField(null=True, blank=True, verbose_name="Суммарный периметр, нм")
    quantum_subbands = models.BooleanField(default=True, verbose_name="Учёт квантовых подзон")

    class Meta:
        db_table = "gaa_nanosheet_parameters"
        verbose_name = "Параметры GAA Nanosheet (D-5)"
        verbose_name_plural = "Параметры GAA Nanosheet (D-5)"
        constraints = [
            models.CheckConstraint(
                condition=Q(sheet_pitch_nm__gte=F("sheet_thickness_nm")),
                name="chk_gaa_ns_pitch_ge_thickness",
            ),
        ]


class GaaNanowireParameters(models.Model):
    """
    Специфичные параметры GAA Nanowire (D-5).
    """
    structure = models.OneToOneField(
        DeviceStructure,
        on_delete=models.CASCADE,
        primary_key=True,
        related_name="gaa_nanowire_params",
        db_column="structure_id",
    )
    wire_diameter_nm = models.FloatField(
        default=10.0,
        validators=[MinValueValidator(0.1), MaxValueValidator(1000.0)],
        verbose_name="Диаметр нанопровода D_nw, нм",
    )
    wire_count = models.IntegerField(
        default=3,
        validators=[MinValueValidator(1), MaxValueValidator(100000)],
        verbose_name="Количество нанопроводов N_nw",
    )
    perimeter_nm = models.FloatField(null=True, blank=True, verbose_name="Суммарный периметр π·D·N, нм")
    quantum_subbands = models.BooleanField(default=True, verbose_name="Учёт квантовых подзон")

    class Meta:
        db_table = "gaa_nanowire_parameters"
        verbose_name = "Параметры GAA Nanowire (D-5)"
        verbose_name_plural = "Параметры GAA Nanowire (D-5)"


# =============================================================================
# D-6 · TFET и HEMT (приборы без BSIM4-совместимости)
# =============================================================================

class TfetParameters(models.Model):
    """
    Специфичные параметры туннельного транзистора TFET (D-6).
    """
    structure = models.OneToOneField(
        DeviceStructure,
        on_delete=models.CASCADE,
        primary_key=True,
        related_name="tfet_params",
        db_column="structure_id",
    )
    tunnel_length_nm = models.FloatField(
        default=20.0,
        validators=[MinValueValidator(0.1), MaxValueValidator(1e6)],
        verbose_name="Длина туннельного перехода L_tun, нм",
    )
    channel_thickness_nm = models.FloatField(
        default=8.0,
        validators=[MinValueValidator(0.1), MaxValueValidator(1000.0)],
        verbose_name="Толщина канала T_ch, нм",
    )
    source_doping_cm3 = models.FloatField(
        default=1e20,
        validators=[MinValueValidator(1.0), MaxValueValidator(1e27)],
        verbose_name="Легирование истока N_src, см⁻³",
    )
    drain_doping_cm3 = models.FloatField(
        default=1e18,
        validators=[MinValueValidator(1.0), MaxValueValidator(1e27)],
        verbose_name="Легирование стока N_drn, см⁻³",
    )
    e_peak_v_cm = models.FloatField(
        null=True,
        blank=True,
        verbose_name="Пиковое электрическое поле на гетеропереходе, В/см",
    )

    class Meta:
        db_table = "tfet_parameters"
        verbose_name = "Параметры TFET (D-6)"
        verbose_name_plural = "Параметры TFET (D-6)"


class HemtParameters(models.Model):
    """
    Специфичные параметры HEMT (D-6).
    """
    structure = models.OneToOneField(
        DeviceStructure,
        on_delete=models.CASCADE,
        primary_key=True,
        related_name="hemt_params",
        db_column="structure_id",
    )
    barrier_thickness_nm = models.FloatField(
        default=20.0,
        validators=[MinValueValidator(0.1), MaxValueValidator(1000.0)],
        verbose_name="Толщина барьерного слоя d_bar, нм",
    )
    sheet_density_cm2 = models.FloatField(
        default=1e13,
        validators=[MinValueValidator(1.0), MaxValueValidator(1e17)],
        verbose_name="Слоевая концентрация 2DEG n_s, см⁻²",
    )
    channel_mobility_cm2_vs = models.FloatField(
        default=1500.0,
        validators=[MinValueValidator(0.1), MaxValueValidator(1e7)],
        verbose_name="Подвижность в канале 2DEG μ_ch, см²/(В·с)",
    )
    access_resistance_ohm_mm = models.FloatField(
        default=0.5,
        validators=[MinValueValidator(0.0), MaxValueValidator(1e6)],
        verbose_name="Удельное сопротивление доступа R_acc, Ом·мм",
    )
    gate_recess_nm = models.FloatField(
        default=0.0,
        validators=[MinValueValidator(0.0), MaxValueValidator(1000.0)],
        verbose_name="Заглубление затвора (gate recess), нм",
    )
    r_sheet_ohm_sq = models.FloatField(null=True, blank=True, verbose_name="Слоевое сопротивление R_sheet, Ом/□")
    r_access_ohm = models.FloatField(null=True, blank=True, verbose_name="Полное сопротивление доступа R_access, Ом")

    class Meta:
        db_table = "hemt_parameters"
        verbose_name = "Параметры HEMT (D-6)"
        verbose_name_plural = "Параметры HEMT (D-6)"


# =============================================================================
# D-7 & D-8 · Мемристор, Мемтранзистор, траектории состояния x(t) и Verilog-A
# =============================================================================

class MemristorParameters(models.Model):
    """
    Специфичные параметры мемристора (D-7) с инвариантом R_on < R_off и V_set > 0 > V_reset.
    """
    structure = models.OneToOneField(
        DeviceStructure,
        on_delete=models.CASCADE,
        primary_key=True,
        related_name="memristor_params",
        db_column="structure_id",
    )
    active_layer_nm = models.FloatField(
        default=10.0,
        validators=[MinValueValidator(0.1), MaxValueValidator(1000.0)],
        verbose_name="Толщина активного слоя D, нм",
    )
    substrate_oxide_nm = models.FloatField(
        default=100.0,
        validators=[MinValueValidator(0.1), MaxValueValidator(1e6)],
        verbose_name="Толщина подложечного оксида, нм",
    )
    ion_mobility_cm2_vs = models.FloatField(
        default=1e-10,
        validators=[MinValueValidator(1e-14), MaxValueValidator(0.01)],
        verbose_name="Подвижность ионов/вакансий μ_v, см²/(В·с)",
    )
    low_resistance_ohm = models.FloatField(
        default=100.0,
        validators=[MinValueValidator(0.01), MaxValueValidator(1e12)],
        verbose_name="Низкоомное сопротивление R_on (LRS), Ом",
    )
    high_resistance_ohm = models.FloatField(
        default=1e5,
        validators=[MinValueValidator(0.02), MaxValueValidator(1e15)],
        verbose_name="Высокоомное сопротивление R_off (HRS), Ом",
    )
    set_voltage_v = models.FloatField(
        default=0.3,
        validators=[MinValueValidator(0.001), MaxValueValidator(100.0)],
        verbose_name="Порог переключения SET V_set (> 0), В",
    )
    reset_voltage_v = models.FloatField(
        default=-0.3,
        validators=[MinValueValidator(-100.0), MaxValueValidator(-0.001)],
        verbose_name="Порог сброса RESET V_reset (< 0), В",
    )
    activation_energy_ev = models.FloatField(
        default=0.6,
        validators=[MinValueValidator(0.001), MaxValueValidator(10.0)],
        verbose_name="Энергия активации Ea, эВ",
    )
    initial_state = models.FloatField(
        default=0.5,
        validators=[MinValueValidator(0.0), MaxValueValidator(1.0)],
        verbose_name="Начальное внутреннее состояние x0 ∈ [0, 1]",
    )
    drift_scale = models.FloatField(null=True, blank=True, verbose_name="Масштаб ионного дрейфа")

    class Meta:
        db_table = "memristor_parameters"
        verbose_name = "Параметры мемристора (D-7)"
        verbose_name_plural = "Параметры мемристоров (D-7)"
        constraints = [
            models.CheckConstraint(
                condition=Q(low_resistance_ohm__lt=F("high_resistance_ohm")),
                name="chk_memristor_ron_lt_roff",
            ),
            models.CheckConstraint(condition=Q(set_voltage_v__gt=0), name="chk_memristor_vset_gt_0"),
            models.CheckConstraint(condition=Q(reset_voltage_v__lt=0), name="chk_memristor_vreset_lt_0"),
            models.CheckConstraint(condition=Q(initial_state__gte=0.0, initial_state__lte=1.0), name="chk_memristor_x0"),
        ]


class MemtransistorParameters(models.Model):
    """
    Специфичные параметры мемтранзистора (D-8): поля мемристора + релаксация и связь с затвором.
    """
    structure = models.OneToOneField(
        DeviceStructure,
        on_delete=models.CASCADE,
        primary_key=True,
        related_name="memtransistor_params",
        db_column="structure_id",
    )
    active_layer_nm = models.FloatField(
        default=10.0,
        validators=[MinValueValidator(0.1), MaxValueValidator(1000.0)],
        verbose_name="Толщина активного слоя D, нм",
    )
    substrate_oxide_nm = models.FloatField(
        default=100.0,
        validators=[MinValueValidator(0.1), MaxValueValidator(1e6)],
        verbose_name="Толщина подложечного оксида, нм",
    )
    ion_mobility_cm2_vs = models.FloatField(
        default=1e-10,
        validators=[MinValueValidator(1e-14), MaxValueValidator(0.01)],
        verbose_name="Подвижность ионов μ_v, см²/(В·с)",
    )
    low_resistance_ohm = models.FloatField(
        default=100.0,
        validators=[MinValueValidator(0.01), MaxValueValidator(1e12)],
        verbose_name="Низкоомное сопротивление R_on, Ом",
    )
    high_resistance_ohm = models.FloatField(
        default=1e5,
        validators=[MinValueValidator(0.02), MaxValueValidator(1e15)],
        verbose_name="Высокоомное сопротивление R_off, Ом",
    )
    state_relaxation_ms = models.FloatField(
        default=100.0,
        validators=[MinValueValidator(0.01), MaxValueValidator(1e9)],
        verbose_name="Постоянная времени релаксации τ_ret, мс",
    )
    set_voltage_v = models.FloatField(
        default=0.3,
        validators=[MinValueValidator(0.001), MaxValueValidator(100.0)],
        verbose_name="Порог SET V_set (> 0), В",
    )
    reset_voltage_v = models.FloatField(
        default=-0.3,
        validators=[MinValueValidator(-100.0), MaxValueValidator(-0.001)],
        verbose_name="Порог RESET V_reset (< 0), В",
    )
    activation_energy_ev = models.FloatField(
        default=0.6,
        validators=[MinValueValidator(0.001), MaxValueValidator(10.0)],
        verbose_name="Энергия активации Ea, эВ",
    )
    gate_coupling = models.FloatField(
        default=2.0,
        validators=[MinValueValidator(0.0), MaxValueValidator(1000.0)],
        verbose_name="Коэффициент связи затвора (gate_coupling)",
    )
    initial_state = models.FloatField(
        default=0.5,
        validators=[MinValueValidator(0.0), MaxValueValidator(1.0)],
        verbose_name="Начальное состояние x0 ∈ [0, 1]",
    )

    class Meta:
        db_table = "memtransistor_parameters"
        verbose_name = "Параметры мемтранзистора (D-8)"
        verbose_name_plural = "Параметры мемтранзисторов (D-8)"
        constraints = [
            models.CheckConstraint(
                condition=Q(low_resistance_ohm__lt=F("high_resistance_ohm")),
                name="chk_memtransistor_ron_lt_roff",
            ),
            models.CheckConstraint(condition=Q(set_voltage_v__gt=0), name="chk_memtransistor_vset_gt_0"),
            models.CheckConstraint(condition=Q(reset_voltage_v__lt=0), name="chk_memtransistor_vreset_lt_0"),
            models.CheckConstraint(condition=Q(initial_state__gte=0.0, initial_state__lte=1.0), name="chk_memtransistor_x0"),
        ]


class MemristorStateSample(models.Model):
    """
    Временные отсчёты динамики внутреннего состояния x(t) мемристора / мемтранзистора (D-7 / D-8).
    """
    sample_id = models.BigAutoField(primary_key=True)
    measurement_set = models.ForeignKey(
        MeasurementSet,
        on_delete=models.CASCADE,
        related_name="memory_state_samples",
        db_column="set_id",
        verbose_name="Набор кривых",
    )
    time_s = models.FloatField(verbose_name="Время t, с")
    voltage_v = models.FloatField(verbose_name="Напряжение V (или Vd), В")
    gate_voltage_v = models.FloatField(null=True, blank=True, verbose_name="Напряжение затвора Vg, В")
    current_a = models.FloatField(verbose_name="Ток I (или Id), А")
    resistance_ohm = models.FloatField(null=True, blank=True, verbose_name="Мгновенное сопротивление R(x), Ом")
    memory_state_x = models.FloatField(
        validators=[MinValueValidator(0.0), MaxValueValidator(1.0)],
        verbose_name="Внутреннее состояние x(t) ∈ [0, 1]",
    )
    sweep_rate_v_s = models.FloatField(null=True, blank=True, verbose_name="Скорость развёртки dV/dt, В/с")
    temperature_k = models.FloatField(default=300.0, verbose_name="Температура T, К")

    class Meta:
        db_table = "memristor_state_samples"
        verbose_name = "Отсчёт состояния памяти x(t)"
        verbose_name_plural = "Отсчёты состояния памяти x(t)"
        indexes = [
            models.Index(fields=["measurement_set", "time_s"], name="idx_mem_state_set_time"),
        ]
        constraints = [
            models.CheckConstraint(
                condition=Q(memory_state_x__gte=0.0, memory_state_x__lte=1.0),
                name="chk_mem_sample_state_range",
            ),
        ]


class VerilogAFlavor(models.TextChoices):
    DRIFT_STRUKOV = "drift_strukov", "Модель ионного дрейфа (Струков / Biolek)"
    THRESHOLD_METAL_OXIDE = "threshold_metal_oxide", "Пороговая модель оксида металла (MO)"


class VerilogAModel(models.Model):
    """
    Сгенерированная и скомпилированная через OpenVAF -> OSDI Verilog-A модель (D-7).
    """
    model_id = models.BigAutoField(primary_key=True)
    structure = models.ForeignKey(
        DeviceStructure,
        on_delete=models.CASCADE,
        related_name="verilog_a_models",
        db_column="structure_id",
        verbose_name="Экземпляр структуры",
    )
    spice_model = models.OneToOneField(
        SpiceModel,
        on_delete=models.SET_NULL,
        null=True,
        blank=True,
        related_name="verilog_a_artifact",
        db_column="spice_model_id",
        verbose_name="Связанная запись SPICE-карты",
    )
    flavor = models.CharField(
        max_length=32,
        choices=VerilogAFlavor.choices,
        default=VerilogAFlavor.DRIFT_STRUKOV,
        verbose_name="Тип компактной Verilog-A модели",
    )
    source_code = models.TextField(verbose_name="Исходный код .va")
    parameters = models.JSONField(
        default=dict,
        verbose_name="Подставленные параметры {Ron, Roff, Vset, Vreset, mu_ion, D, Ea, Tnom, x_init, window_order}",
    )
    osdi_binary_path = models.CharField(
        max_length=512,
        blank=True,
        default="",
        verbose_name="Путь к скомпилированному .osdi модулю",
    )
    compiled_by = models.CharField(
        max_length=64,
        default="OpenVAF -> OSDI",
        verbose_name="Инструмент компиляции",
    )
    created_at = models.DateTimeField(auto_now_add=True, verbose_name="Время генерации")

    class Meta:
        db_table = "verilog_a_models"
        verbose_name = "Verilog-A / OSDI модель"
        verbose_name_plural = "Verilog-A / OSDI модели"

    def __str__(self) -> str:
        return f"VerilogA #{self.model_id} ({self.flavor})"


# =============================================================================
# D-9 · CAD-транзистор (импорт STL и геометрия поверхности)
# =============================================================================

class StlFormatChoices(models.TextChoices):
    BINARY_STL = "binary_stl", "Binary STL"
    ASCII_STL = "ascii_stl", "ASCII STL"


class CadUnitChoices(models.TextChoices):
    M = "m", "Метры (m, scale=1)"
    MM = "mm", "Миллиметры (mm, scale=1e-3)"
    NM = "nm", "Нанометры (nm, scale=1e-9)"


class CadImportStatusChoices(models.TextChoices):
    IMPORTED = "imported", "Импортировано"
    REJECTED = "rejected", "Отклонено"


class CadImportJob(models.Model):
    """
    Задача импорта CAD-геометрии из STL-файла (D-9, FR-IMP-02).
    """
    job_id = models.BigAutoField(primary_key=True)
    device_type = models.ForeignKey(
        DeviceLabel,
        on_delete=models.PROTECT,
        default=DeviceFamilyChoices.CAD_CUSTOM,
        related_name="cad_import_jobs",
        db_column="device_id",
        verbose_name="Семейство прибора",
    )
    structure = models.ForeignKey(
        DeviceStructure,
        on_delete=models.CASCADE,
        null=True,
        blank=True,
        related_name="cad_import_jobs",
        db_column="structure_id",
        verbose_name="Созданная структура",
    )
    file_name = models.CharField(max_length=255, verbose_name="Имя исходного STL-файла")
    format = models.CharField(
        max_length=16,
        choices=StlFormatChoices.choices,
        verbose_name="Формат STL (binary / ascii)",
    )
    unit = models.CharField(
        max_length=8,
        choices=CadUnitChoices.choices,
        verbose_name="Единицы измерения координат",
    )
    scale_to_m = models.FloatField(verbose_name="Коэффициент перевода в метры (1, 1e-3, 1e-9)")
    triangle_count = models.IntegerField(
        default=0,
        validators=[MinValueValidator(0)],
        verbose_name="Количество треугольников",
    )
    bounding_box = models.JSONField(
        default=dict,
        verbose_name="Нормированные габариты {x, y, z}",
    )
    status = models.CharField(
        max_length=16,
        choices=CadImportStatusChoices.choices,
        default=CadImportStatusChoices.IMPORTED,
        verbose_name="Статус импорта",
    )
    error = models.TextField(blank=True, default="", verbose_name="Описание ошибки импорта")
    created_at = models.DateTimeField(auto_now_add=True, verbose_name="Время импорта")

    class Meta:
        db_table = "cad_import_jobs"
        verbose_name = "Задача импорта CAD (STL)"
        verbose_name_plural = "Задачи импорта CAD (STL)"

    def __str__(self) -> str:
        return f"CadImportJob #{self.job_id}: {self.file_name} [{self.status}]"


class CadGeometry(models.Model):
    """
    Бинарные буферы вершин и нормалей STL-поверхности для 3D-отображения полей (D-9).
    """
    geometry_id = models.BigAutoField(primary_key=True)
    job = models.OneToOneField(
        CadImportJob,
        on_delete=models.CASCADE,
        related_name="geometry",
        db_column="job_id",
        verbose_name="Задача импорта STL",
    )
    structure = models.OneToOneField(
        DeviceStructure,
        on_delete=models.CASCADE,
        null=True,
        blank=True,
        related_name="cad_geometry",
        db_column="structure_id",
        verbose_name="Экземпляр структуры",
    )
    vertices_blob = models.BinaryField(verbose_name="Массив вершин (BLOB)")
    normals_blob = models.BinaryField(verbose_name="Массив нормалей (BLOB)")
    model_matrix = models.JSONField(
        default=dict,
        verbose_name="Матрица преобразования (масштаб + трансляция)",
    )

    class Meta:
        db_table = "cad_geometry"
        verbose_name = "CAD-геометрия поверхности (STL)"
        verbose_name_plural = "CAD-геометрии поверхностей (STL)"

    def __str__(self) -> str:
        return f"CadGeometry #{self.geometry_id} (job={self.job_id})"
