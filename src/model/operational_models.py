from django.db import models
from .catalog_models import (
    CharacteristicSpec,
    DeviceLabel,
    GeometryField,
    MetricLabel,
    SolverCatalog,
)


# =====================================================================
# Context: Configuration
# =====================================================================

class DeviceProfile(models.Model):
    """
    device_profiles — конфигурация семейства прибора.
    """
    profile_id = models.BigAutoField(primary_key=True)
    device = models.ForeignKey(
        DeviceLabel,
        on_delete=models.RESTRICT,
        db_column="device_id",
        related_name="profiles",
    )
    name = models.TextField()
    solver_basis = models.CharField(max_length=64)
    export_format = models.CharField(max_length=64)
    thermal_enabled = models.BooleanField(default=False)
    valid_from = models.DateTimeField(auto_now_add=True)

    class Meta:
        db_table = "device_profiles"
        verbose_name = "Профиль конфигурации"
        verbose_name_plural = "Профили конфигураций"

    def __str__(self) -> str:
        return f"{self.name} ({self.device_id})"


class DeviceStructure(models.Model):
    """
    device_structures — экземпляр структуры прибора.
    """
    structure_id = models.BigAutoField(primary_key=True)
    device = models.ForeignKey(
        DeviceLabel,
        on_delete=models.RESTRICT,
        db_column="device_id",
        related_name="structures",
    )
    profile = models.ForeignKey(
        DeviceProfile,
        on_delete=models.RESTRICT,
        db_column="profile_id",
        null=True,
        blank=True,
        related_name="structures",
    )
    name = models.TextField()
    geometry_data = models.JSONField(default=dict)
    stl_ref = models.TextField(null=True, blank=True)
    unit_scale = models.CharField(max_length=32, default="nm")
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        db_table = "device_structures"
        verbose_name = "Экземпляр структуры"
        verbose_name_plural = "Экземпляры структур"

    def __str__(self) -> str:
        return f"{self.name} (#{self.structure_id})"


class Material(models.Model):
    """
    materials — справочник физических свойств материалов слоёв.
    """
    material_id = models.BigAutoField(primary_key=True)
    name = models.TextField(unique=True)
    epsilon_r = models.FloatField()
    mobility_n = models.FloatField(null=True, blank=True)
    mobility_p = models.FloatField(null=True, blank=True)
    bandgap_ev = models.FloatField(null=True, blank=True)
    k_thermal = models.FloatField(null=True, blank=True)

    class Meta:
        db_table = "materials"
        verbose_name = "Материал"
        verbose_name_plural = "Материалы"

    def __str__(self) -> str:
        return self.name


class DeviceParameterValue(models.Model):
    """
    device_parameter_values — EAV-таблица значений полей структуры.
    """
    value_id = models.BigAutoField(primary_key=True)
    structure = models.ForeignKey(
        DeviceStructure,
        on_delete=models.CASCADE,
        db_column="structure_id",
        related_name="parameter_values",
    )
    field = models.ForeignKey(
        GeometryField,
        on_delete=models.RESTRICT,
        db_column="field_id",
        related_name="parameter_values",
    )
    material = models.ForeignKey(
        Material,
        on_delete=models.SET_NULL,
        db_column="material_id",
        null=True,
        blank=True,
        related_name="parameter_values",
    )
    numeric_value = models.FloatField()
    unit = models.CharField(max_length=32, blank=True, default="")

    class Meta:
        db_table = "device_parameter_values"
        verbose_name = "Значение параметра структуры"
        verbose_name_plural = "Значения параметров структур"
        constraints = [
            models.UniqueConstraint(
                fields=["structure", "field"],
                name="uq_device_param_value_structure_field",
            )
        ]

    def __str__(self) -> str:
        return f"{self.structure_id}:{self.field_id}={self.numeric_value}"


# =====================================================================
# Context: Computation
# =====================================================================

class SolverTaskStatus(models.TextChoices):
    QUEUED = "queued", "Queued"
    RUNNING = "running", "Running"
    OK = "ok", "OK"
    FAILED = "failed", "Failed"
    CANCELLED = "cancelled", "Cancelled"


class SolverBackendChoices(models.TextChoices):
    CUDA_FP64 = "cuda_fp64", "CUDA FP64"
    CPU_SCIPY = "cpu_scipy", "CPU SciPy"
    CPU_TORCH = "cpu_torch", "CPU PyTorch"
    STATE_MODEL = "state_model", "State Model"


class SolverTask(models.Model):
    """
    solver_tasks — задача расчёта.
    """
    task_id = models.BigAutoField(primary_key=True)
    structure = models.ForeignKey(
        DeviceStructure,
        on_delete=models.CASCADE,
        db_column="structure_id",
        related_name="solver_tasks",
    )
    solver = models.ForeignKey(
        SolverCatalog,
        on_delete=models.RESTRICT,
        db_column="solver_id",
        related_name="solver_tasks",
    )
    spec = models.ForeignKey(
        CharacteristicSpec,
        on_delete=models.RESTRICT,
        db_column="spec_id",
        null=True,
        blank=True,
        related_name="solver_tasks",
    )
    status = models.CharField(
        max_length=16,
        choices=SolverTaskStatus.choices,
        default=SolverTaskStatus.QUEUED,
        db_index=True,
    )
    backend_used = models.CharField(
        max_length=32,
        choices=SolverBackendChoices.choices,
        null=True,
        blank=True,
    )
    iterations = models.IntegerField(null=True, blank=True)
    residual = models.FloatField(null=True, blank=True)
    error_code = models.CharField(max_length=64, null=True, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        db_table = "solver_tasks"
        verbose_name = "Задача расчёта"
        verbose_name_plural = "Задачи расчёта"

    def __str__(self) -> str:
        return f"Task #{self.task_id} [{self.status}]"


class FieldQuantityChoices(models.TextChoices):
    TEMPERATURE = "T", "Temperature (T)"
    POTENTIAL = "V", "Electrostatic Potential (V)"
    ELECTRIC_FIELD = "|E|", "Electric Field Magnitude (|E|)"
    ELECTRON_DENSITY = "n", "Electron Density (n)"
    HOLE_DENSITY = "p", "Hole Density (p)"
    MAGNETIC_VECTOR_POTENTIAL = "Az", "Magnetic Vector Potential (Az)"


class FieldData(models.Model):
    """
    field_data — карты физических полей задачи расчёта.
    """
    field_data_id = models.BigAutoField(primary_key=True)
    task = models.ForeignKey(
        SolverTask,
        on_delete=models.CASCADE,
        db_column="task_id",
        related_name="field_maps",
    )
    quantity = models.CharField(max_length=16, choices=FieldQuantityChoices.choices)
    coordinate_mesh = models.BinaryField()
    values = models.BinaryField()
    units = models.CharField(max_length=32)
    checksum = models.CharField(max_length=128)

    class Meta:
        db_table = "field_data"
        verbose_name = "Карта поля"
        verbose_name_plural = "Карты полей"
        constraints = [
            models.UniqueConstraint(
                fields=["task", "quantity"],
                name="uq_field_data_task_quantity",
            )
        ]


class SolverIteration(models.Model):
    """
    solver_iterations — журнал сходимости расчёта по шагам.
    """
    iteration_id = models.BigAutoField(primary_key=True)
    task = models.ForeignKey(
        SolverTask,
        on_delete=models.CASCADE,
        db_column="task_id",
        related_name="convergence_log",
    )
    step_no = models.IntegerField()
    residual = models.FloatField()
    dt_s = models.FloatField(null=True, blank=True)
    newton_iters = models.SmallIntegerField(default=0)

    class Meta:
        db_table = "solver_iterations"
        ordering = ["task_id", "step_no"]
        verbose_name = "Итерация решателя"
        verbose_name_plural = "Журнал сходимости решателя"


# =====================================================================
# Context: AI Platform
# =====================================================================

class SurrogateFamilyChoices(models.TextChoices):
    UNET_POISSON = "unet_poisson", "U-Net Poisson"
    UNET_AZ = "unet_az", "U-Net Az"
    UNET_CARRIERS = "unet_carriers", "U-Net Carriers"
    GNN_FIELD = "gnn_field", "GNN Field"
    GNN_IV = "gnn_iv", "GNN I-V"


class SurrogateTopologyChoices(models.TextChoices):
    UNET = "U-Net", "U-Net"
    GCN = "GCN", "GCN"
    GRAPHSAGE = "GraphSAGE", "GraphSAGE"
    GAT = "GAT", "GAT"


class SurrogateModel(models.Model):
    """
    surrogate_models — реестр обученных суррогатных моделей.
    """
    model_id = models.BigAutoField(primary_key=True)
    family = models.CharField(max_length=32, choices=SurrogateFamilyChoices.choices)
    topology = models.CharField(max_length=32, choices=SurrogateTopologyChoices.choices)
    in_channels = models.SmallIntegerField()
    out_channels = models.SmallIntegerField()
    checkpoint_path = models.TextField(blank=True, default="")
    onnx_path = models.TextField(blank=True, default="")
    grid_size = models.SmallIntegerField()
    train_loss = models.FloatField(null=True, blank=True)
    val_rmse_rel = models.FloatField(null=True, blank=True)
    c2_continuity_ok = models.BooleanField(default=False)
    monotonic_iv_ok = models.BooleanField(default=False)

    class Meta:
        db_table = "surrogate_models"
        verbose_name = "Суррогатная модель"
        verbose_name_plural = "Суррогатные модели"

    def __str__(self) -> str:
        return f"Surrogate #{self.model_id} ({self.family} / {self.topology})"


class NeuralRun(models.Model):
    """
    neural_runs — инференс суррогата для тёплого старта задачи расчёта (1:0..1 с SolverTask).
    """
    run_id = models.BigAutoField(primary_key=True)
    task = models.OneToOneField(
        SolverTask,
        on_delete=models.CASCADE,
        db_column="task_id",
        related_name="neural_run",
    )
    model = models.ForeignKey(
        SurrogateModel,
        on_delete=models.RESTRICT,
        db_column="model_id",
        related_name="neural_runs",
    )
    warm_start_residual = models.FloatField()
    iters_saved = models.IntegerField(default=0)
    uncertainty = models.FloatField(null=True, blank=True)
    fallback_to_fem = models.BooleanField(default=False)

    class Meta:
        db_table = "neural_runs"
        verbose_name = "Запуск суррогатного инференса"
        verbose_name_plural = "Запуски суррогатного инференса"


class TrainingRun(models.Model):
    """
    training_runs — журнал обучения суррогатных моделей.
    """
    training_id = models.BigAutoField(primary_key=True)
    model = models.ForeignKey(
        SurrogateModel,
        on_delete=models.CASCADE,
        db_column="model_id",
        related_name="training_runs",
    )
    samples = models.IntegerField()
    epochs = models.IntegerField()
    batch_size = models.SmallIntegerField()
    seed = models.IntegerField(default=42)
    device = models.CharField(max_length=32, default="cuda")
    amp_enabled = models.BooleanField(default=True)
    loss_data = models.FloatField(null=True, blank=True)
    loss_poisson = models.FloatField(null=True, blank=True)
    loss_continuity = models.FloatField(null=True, blank=True)
    duration_s = models.FloatField(null=True, blank=True)

    class Meta:
        db_table = "training_runs"
        verbose_name = "Прогон обучения"
        verbose_name_plural = "Прогоны обучения"


class DatasetSplitChoices(models.TextChoices):
    TRAIN = "train", "Train"
    VAL = "val", "Validation"
    TEST = "test", "Test"


class SampleSourceChoices(models.TextChoices):
    SYNTHETIC = "synthetic", "Synthetic"
    FEM_REFERENCE = "fem_reference", "FEM Reference"
    MEASUREMENT = "measurement", "Measurement"


class TrainingSample(models.Model):
    """
    training_samples — обучающие/валидационные выборки (вход/цель).
    """
    sample_id = models.BigAutoField(primary_key=True)
    training = models.ForeignKey(
        TrainingRun,
        on_delete=models.CASCADE,
        db_column="training_id",
        related_name="training_samples",
    )
    structure = models.ForeignKey(
        DeviceStructure,
        on_delete=models.CASCADE,
        db_column="structure_id",
        related_name="training_samples",
    )
    inputs_blob = models.BinaryField()
    targets_blob = models.BinaryField()
    split = models.CharField(max_length=16, choices=DatasetSplitChoices.choices)
    source = models.CharField(max_length=32, choices=SampleSourceChoices.choices)

    class Meta:
        db_table = "training_samples"
        verbose_name = "Обучающий пример"
        verbose_name_plural = "Обучающие примеры"


# =====================================================================
# Context: Extraction & Modeling
# =====================================================================

class MeasurementSet(models.Model):
    """
    measurement_sets — импортированные наборы измерений.
    """
    set_id = models.BigAutoField(primary_key=True)
    structure = models.ForeignKey(
        DeviceStructure,
        on_delete=models.CASCADE,
        db_column="structure_id",
        related_name="measurement_sets",
    )
    spec = models.ForeignKey(
        CharacteristicSpec,
        on_delete=models.RESTRICT,
        db_column="spec_id",
        related_name="measurement_sets",
    )
    source_path = models.TextField()
    delimiter = models.CharField(max_length=8, default=",")
    points = models.IntegerField(default=0)
    imported_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        db_table = "measurement_sets"
        verbose_name = "Набор измерений"
        verbose_name_plural = "Наборы измерений"


class MeasurementPoint(models.Model):
    """
    measurement_points — точки кривых измерений.
    """
    point_id = models.BigAutoField(primary_key=True)
    measurement_set = models.ForeignKey(
        MeasurementSet,
        on_delete=models.CASCADE,
        db_column="set_id",
        related_name="curve_points",
    )
    x = models.FloatField()
    y = models.FloatField()
    bias = models.FloatField(null=True, blank=True)
    extra = models.JSONField(default=dict, blank=True)

    class Meta:
        db_table = "measurement_points"
        ordering = ["point_id"]
        verbose_name = "Точка измерения"
        verbose_name_plural = "Точки измерений"


class ExtractionAlgorithmChoices(models.TextChoices):
    BSIM4_STEPS = "BSIM4_STEPS", "BSIM4 Steps"
    CMG = "CMG", "BSIM-CMG"
    BEHAVIORAL = "BEHAVIORAL", "Behavioral"


class ExtractionSession(models.Model):
    """
    extraction_sessions — сессия многопараметрической экстракции.
    """
    session_id = models.BigAutoField(primary_key=True)
    task = models.ForeignKey(
        SolverTask,
        on_delete=models.SET_NULL,
        db_column="task_id",
        null=True,
        blank=True,
        related_name="extraction_sessions",
    )
    structure = models.ForeignKey(
        DeviceStructure,
        on_delete=models.CASCADE,
        db_column="structure_id",
        related_name="extraction_sessions",
    )
    algorithm = models.CharField(
        max_length=32,
        choices=ExtractionAlgorithmChoices.choices,
        default=ExtractionAlgorithmChoices.BSIM4_STEPS,
    )
    algorithm_settings = models.JSONField(default=dict)
    pinn_applied = models.BooleanField(default=False)
    iterations_count = models.IntegerField(default=0)
    final_mse_validation = models.FloatField(null=True, blank=True)
    status = models.CharField(
        max_length=16,
        choices=SolverTaskStatus.choices,
        default=SolverTaskStatus.QUEUED,
    )
    started_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        db_table = "extraction_sessions"
        verbose_name = "Сессия экстракции"
        verbose_name_plural = "Сессии экстракции"


class ExtractedParameter(models.Model):
    """
    extracted_parameters — извлечённые параметры компактной карты.
    """
    extracted_id = models.BigAutoField(primary_key=True)
    session = models.ForeignKey(
        ExtractionSession,
        on_delete=models.CASCADE,
        db_column="session_id",
        related_name="extracted_parameters",
    )
    parameter_name = models.CharField(max_length=64)
    extracted_value = models.FloatField()
    step_no = models.SmallIntegerField(default=1)
    bound_hit = models.BooleanField(default=False)

    class Meta:
        db_table = "extracted_parameters"
        ordering = ["session_id", "step_no", "parameter_name"]
        verbose_name = "Извлечённый параметр"
        verbose_name_plural = "Извлечённые параметры"


class SpiceModelTypeChoices(models.TextChoices):
    BSIM4 = "BSIM4", "BSIM4"
    LEVEL72 = "LEVEL72", "BSIM-CMG (LEVEL 72)"
    SUBCKT = "subckt", "Subcircuit (.subckt)"
    VERILOG_A = "verilog_a", "Verilog-A"


class SpiceModel(models.Model):
    """
    spice_models — итоговая SPICE-карта модели (1:0..1 с ExtractionSession).
    """
    model_id = models.BigAutoField(primary_key=True)
    session = models.OneToOneField(
        ExtractionSession,
        on_delete=models.CASCADE,
        db_column="session_id",
        related_name="spice_model",
    )
    model_text = models.TextField()
    model_type = models.CharField(max_length=32, choices=SpiceModelTypeChoices.choices)
    target_simulator = models.CharField(max_length=32)
    fitness_score = models.FloatField(null=True, blank=True)
    emitted_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        db_table = "spice_models"
        verbose_name = "SPICE-карта модели"
        verbose_name_plural = "SPICE-карты моделей"


class CurveMetricValue(models.Model):
    """
    curve_metrics_values — значения непараметрических метрик кривых.
    """
    value_id = models.BigAutoField(primary_key=True)
    measurement_set = models.ForeignKey(
        MeasurementSet,
        on_delete=models.CASCADE,
        db_column="set_id",
        related_name="metric_values",
    )
    metric = models.ForeignKey(
        MetricLabel,
        on_delete=models.RESTRICT,
        db_column="metric_id",
        related_name="curve_values",
    )
    value = models.FloatField()
    unit = models.CharField(max_length=32, blank=True, default="")

    class Meta:
        db_table = "curve_metrics_values"
        verbose_name = "Значение метрики кривой"
        verbose_name_plural = "Значения метрик кривых"
        constraints = [
            models.UniqueConstraint(
                fields=["measurement_set", "metric"],
                name="uq_curve_metric_value_set_metric",
            )
        ]


# =====================================================================
# Context: Verification & Reporting
# =====================================================================

class EdaSimulatorChoices(models.TextChoices):
    NGSPICE = "ngspice", "ngspice"
    MICROCAP = "microcap", "Micro-Cap"
    TANNER = "tanner", "Tanner T-Spice"
    XYCE = "xyce", "Xyce"


class VerificationModeChoices(models.TextChoices):
    LOOPBACK = "loopback", "Loopback"
    HYSTERESIS = "hysteresis", "Hysteresis"
    NETLIST = "netlist", "Netlist"


class VerificationRun(models.Model):
    """
    verification_runs — верификация SPICE-модели в EDA-симуляторе.
    """
    run_id = models.BigAutoField(primary_key=True)
    model = models.ForeignKey(
        SpiceModel,
        on_delete=models.CASCADE,
        db_column="model_id",
        related_name="verification_runs",
    )
    simulator = models.CharField(max_length=32, choices=EdaSimulatorChoices.choices)
    mode = models.CharField(max_length=32, choices=VerificationModeChoices.choices)
    loop_area_a_v = models.FloatField(null=True, blank=True)
    degenerate = models.BooleanField(default=False)
    max_rel_error = models.FloatField(null=True, blank=True)
    passed = models.BooleanField(default=False)
    log_path = models.TextField(blank=True, default="")

    class Meta:
        db_table = "verification_runs"
        verbose_name = "Прогон верификации EDA"
        verbose_name_plural = "Прогоны верификации EDA"


class Report(models.Model):
    """
    reports — сгенерированные LaTeX/PDF отчёты по экстракции и верификации.
    """
    report_id = models.BigAutoField(primary_key=True)
    session = models.ForeignKey(
        ExtractionSession,
        on_delete=models.CASCADE,
        db_column="session_id",
        null=True,
        blank=True,
        related_name="reports",
    )
    verification_run = models.ForeignKey(
        VerificationRun,
        on_delete=models.CASCADE,
        db_column="run_id",
        null=True,
        blank=True,
        related_name="reports",
    )
    latex_source = models.TextField()
    pdf_blob = models.BinaryField(null=True, blank=True)
    generated_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        db_table = "reports"
        verbose_name = "Отчёт"
        verbose_name_plural = "Отчёты"
