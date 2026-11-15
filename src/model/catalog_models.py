from django.db import models


class DeviceFamilyChoices(models.TextChoices):
    PLANAR_MOSFET = "planar_mosfet", "Planar MOSFET"
    FD_SOI = "fd_soi", "FD-SOI"
    FINFET = "finfet", "FinFET"
    GAA_NANOSHEET = "gaa_nanosheet", "GAA Nanosheet"
    GAA_NANOWIRE = "gaa_nanowire", "GAA Nanowire"
    TFET = "tfet", "TFET"
    HEMT = "hemt", "HEMT"
    MEMRISTOR = "memristor", "Memristor"
    MEMTRANSISTOR = "memtransistor", "Memtransistor"
    CAD_CUSTOM = "cad_custom", "CAD Custom"


class ScaleTypeChoices(models.TextChoices):
    LINEAR = "linear", "Linear"
    LOG = "log", "Logarithmic (Y)"
    LOGX = "logx", "Logarithmic (X)"


class DeviceLabel(models.Model):
    """
    device_labels — справочник семейств приборов (10 строк).
    """
    device_id = models.CharField(
        primary_key=True,
        max_length=32,
        choices=DeviceFamilyChoices.choices,
    )
    display_name = models.TextField()

    class Meta:
        db_table = "device_labels"
        verbose_name = "Семейство прибора"
        verbose_name_plural = "Семейства приборов"

    def __str__(self) -> str:
        return f"{self.display_name} ({self.device_id})"


class GeometryField(models.Model):
    """
    geometry_fields — реестр параметров геометрии и материала (32 поля).
    """
    field_id = models.CharField(primary_key=True, max_length=64)
    display_name = models.TextField()
    default_value = models.FloatField(null=True, blank=True)
    min_value = models.FloatField(null=True, blank=True)
    max_value = models.FloatField(null=True, blank=True)
    unit = models.CharField(max_length=32, blank=True, default="")

    class Meta:
        db_table = "geometry_fields"
        verbose_name = "Поле геометрии/материала"
        verbose_name_plural = "Поля геометрии/материалов"

    def __str__(self) -> str:
        return f"{self.display_name} [{self.field_id}]"


class AxisLabel(models.Model):
    """
    axis_labels — словарь осей графиков (41 ось).
    """
    label_id = models.CharField(primary_key=True, max_length=64)
    display_name = models.TextField()

    class Meta:
        db_table = "axis_labels"
        verbose_name = "Ось графика"
        verbose_name_plural = "Оси графиков"

    def __str__(self) -> str:
        return self.display_name


class CharacteristicSpec(models.Model):
    """
    characteristic_specs — спецификация характеристик.
    """
    spec_id = models.CharField(primary_key=True, max_length=64)
    display_name = models.TextField()
    x_axis_field = models.ForeignKey(
        AxisLabel,
        on_delete=models.RESTRICT,
        db_column="x_axis_field",
        related_name="x_characteristic_specs",
    )
    y_axis_field = models.ForeignKey(
        AxisLabel,
        on_delete=models.RESTRICT,
        db_column="y_axis_field",
        related_name="y_characteristic_specs",
    )
    scale_type = models.CharField(
        max_length=16,
        choices=ScaleTypeChoices.choices,
        default=ScaleTypeChoices.LINEAR,
    )

    class Meta:
        db_table = "characteristic_specs"
        verbose_name = "Спецификация характеристики"
        verbose_name_plural = "Спецификации характеристик"

    def __str__(self) -> str:
        return f"{self.display_name} ({self.spec_id})"


class CharacteristicRequirement(models.Model):
    """
    characteristic_requirements — 1:1 требования к сигналам.
    """
    spec = models.OneToOneField(
        CharacteristicSpec,
        primary_key=True,
        on_delete=models.CASCADE,
        db_column="spec_id",
        related_name="requirement",
    )
    required_signals = models.TextField(help_text='Например: "Vg, Id [и Vd]"')

    class Meta:
        db_table = "characteristic_requirements"
        verbose_name = "Требование к сигналам характеристики"
        verbose_name_plural = "Требования к сигналам характеристик"

    def __str__(self) -> str:
        return f"{self.spec_id}: {self.required_signals}"


class MetricLabel(models.Model):
    """
    metric_labels — словарь метрик (66 метрик).
    """
    metric_id = models.CharField(primary_key=True, max_length=64)
    display_name = models.TextField()

    class Meta:
        db_table = "metric_labels"
        verbose_name = "Метрика"
        verbose_name_plural = "Словарь метрик"

    def __str__(self) -> str:
        return f"{self.display_name} ({self.metric_id})"


class CurveMetricMap(models.Model):
    """
    curve_metric_map — связь M:N (метрика × характеристика).
    В Django используется суррогатный PK + UniqueConstraint по (spec, metric).
    """
    spec = models.ForeignKey(
        CharacteristicSpec,
        on_delete=models.CASCADE,
        db_column="spec_id",
        related_name="metric_mappings",
    )
    metric = models.ForeignKey(
        MetricLabel,
        on_delete=models.CASCADE,
        db_column="metric_id",
        related_name="spec_mappings",
    )
    formula_ref = models.TextField(blank=True, default="")
    min_points = models.SmallIntegerField(default=2)

    class Meta:
        db_table = "curve_metric_map"
        verbose_name = "Привязка метрики к характеристике"
        verbose_name_plural = "Привязки метрик к характеристикам"
        constraints = [
            models.UniqueConstraint(
                fields=["spec", "metric"],
                name="uq_curve_metric_map_spec_metric",
            )
        ]

    def __str__(self) -> str:
        return f"{self.spec_id} -> {self.metric_id}"


class DeviceFieldMap(models.Model):
    """
    device_field_map — связь M:N (семейство × поле геометрии/материала).
    """
    device = models.ForeignKey(
        DeviceLabel,
        on_delete=models.CASCADE,
        db_column="device_id",
        related_name="field_mappings",
    )
    field = models.ForeignKey(
        GeometryField,
        on_delete=models.CASCADE,
        db_column="field_id",
        related_name="device_mappings",
    )
    is_required = models.BooleanField(default=True)
    sort_order = models.SmallIntegerField(default=0)
    invariant_rule = models.TextField(blank=True, default="")

    class Meta:
        db_table = "device_field_map"
        ordering = ["device_id", "sort_order"]
        verbose_name = "Поле семейства прибора"
        verbose_name_plural = "Поля семейств приборов"
        constraints = [
            models.UniqueConstraint(
                fields=["device", "field"],
                name="uq_device_field_map_device_field",
            )
        ]

    def __str__(self) -> str:
        return f"{self.device_id}.{self.field_id}"


class SolverCatalog(models.Model):
    """
    solver_catalog — каталог методов расчёта.
    """
    solver_id = models.CharField(primary_key=True, max_length=64)
    label = models.TextField()
    class_of_method = models.CharField(max_length=64)
    supported_devices = models.JSONField(
        default=list,
        help_text="Список идентификаторов device_id, поддерживаемых методом",
    )
    backend = models.CharField(max_length=64)
    accuracy_class = models.CharField(max_length=64)

    class Meta:
        db_table = "solver_catalog"
        verbose_name = "Метод расчёта"
        verbose_name_plural = "Каталог методов расчёта"

    def __str__(self) -> str:
        return f"{self.label} ({self.solver_id})"
