# TSPICE Laboratory — Enterprise Architecture

Итоговая архитектурная модель платформы сквозного моделирования полупроводниковых структур
и экстракции компактных моделей. Документ связывает **бизнес-возможности → слои → доменную
модель данных → алгоритмы → требования**.

Все диаграммы — PlantUML, лежат рядом и проверены на синтаксис. Источник истины для схемы БД —
`django-app/src/sql/*.sql`, для доменных классов — `src/model.py`, `src/extractors.py`,
`src/simulation.py`, `src/neural_physics.py`, для вычислительного ядра — `django-app/tspice_solver/lib.rs`.

---

## 1. Enterprise-карта слоёв

| Слой | Ответственность | Реализация | Диаграмма |
|---|---|---|---|
| Presentation | ввод параметров, графики ВАХ/ВФХ, 3D-сцена, телеметрия GPU | PyQt6 (`src/main.py`, `TransistorWorkbench`, 4 × QThread Worker) | `ui/UI.puml`, `ui/UI_STATE.puml` |
| API-контракт | серверный режим, OpenAPI | `specifications/BACKEND.yaml` (`/api/v1`) | `enterprise/L2_CONTAINERS.puml` |
| Application | валидация, очередь задач, ORM, транзакции, права | Django (`django-app/`) | `enterprise/L2_CONTAINERS.puml` |
| Domain (pure) | инварианты структур, метрики, экспорт карт | `src/model.py`, `src/extractors.py` | `enterprise/L4_CLASSES_DOMAIN.puml` |
| Compute | FEM, quantum-Poisson, reduced-order PDE, state-модели, суррогаты | `src/simulation.py`, `src/neural_physics.py` | `enterprise/L3_COMPONENTS.puml` |
| Native core | ONNX-инференс + точная доводка warm start | Rust (`pyo3`, `ort`, `sprs`) | `enterprise/L3_COMPONENTS.puml` |
| Data | справочники + операционные данные + журналы | SQLite (`django-app/src/sql`) | `database/D0`, `D1` |
| Integration | компиляция Verilog-A, прогон нетлистов | OpenVAF → OSDI, ngspice, Micro-Cap 12 | `backend/EXTRACTION_VERIFICATION_CORE.puml` |
| Platform | CUDA FP64, CI/CD, артефакты | PyTorch, Jenkins, Sonatype Nexus | `deploy/DEPLOY.puml` |

Ключевое архитектурное решение (**гибридный принцип**): нейросеть выдаёт **только стартовый
вектор**, физическую точность финального результата всегда определяет детерминированный
решатель (Rust/FEM/SciPy). Поэтому деградация суррогата влияет на *время до сходимости*, но не
на корректность итога.

```
Visitor → Presentation → Application → Domain → { Compute | Data | Integration }
                                                ↘ Neural Surrogate → warm start → Compute
```

---

## 2. Доменная область (глубоко разработанная)

### 2.1. Модель данных: уровни нормализации

| Уровень | Содержимое | Диаграмма |
|---|---|---|
| D0 · Reference Data | `device_labels` (10 семейств), `geometry_fields` (32 поля с min/max/unit), `axis_labels` (41 ось), `characteristic_specs` (39 характеристик), `characteristic_requirements`, `metric_labels` (66 метрик) — реализовано в SQL | `database/D0_REFERENCE_CATALOG.puml` |
| D1 · Operational | профили → структуры → значения (EAV) → задачи → поля → сессии → параметры → карты → верификация → отчёты; контексты Configuration / Computation / AI Platform / Extraction / Verification | `database/D1_CORE_OPERATIONAL.puml` |
| D-v · Per-device | таблицы параметров, вычисляемых величин, характеристик, формул метрик и формата карты по каждому семейству | `devices/DEVICE_*.puml` |

Проектируемые расширения (на D1 помечены синим): `solver_catalog`, `solver_iterations`,
`neural_runs`, `surrogate_models`, `training_runs`, `training_samples`, `verification_runs`,
`reports`, `device_field_map`, `curve_metric_map`. Они закрывают разрыв между тем, что уже
есть в SQL, и тем, что требуется для трассируемости Enterprise-контура (реестр моделей,
журнал сходимости, журнал верификации).

### 2.2. Семейства приборов и их табличные профили

| Семейство | Специфичные поля (из `DEVICE_GEOMETRY_FIELDS`) | Метод | Карта | Диаграмма |
|---|---|---|---|---|
| Planar Bulk MOSFET | — (только W, L, Tox, профиль легирования) | полный compact-model flow, `least_squares` | **BSIM4 `.model`** (LEVEL 54 / 14) | `devices/DEVICE_PLANAR_MOSFET.puml` |
| FD-SOI | `body_thickness_nm`, `buried_oxide_nm`, `back_gate_voltage` | Laplace FEM + ёмкостная связь BOX | BSIM4 (запрошенная совместимость) | `devices/DEVICE_FD_SOI.puml` |
| FinFET | `fin_width_nm`, `fin_height_nm`, `fin_count`, `fin_pitch_nm` | геометрический FEM, `W_eff = N·(2H+W)` | LEVEL 72 | `devices/DEVICE_FINFET.puml` |
| GAA nanosheet / nanowire | `sheet_width/thickness/count/pitch_nm`, `wire_diameter_nm`, `wire_count` | периметральный FEM + тепло в каждом листе | LEVEL 72 | `devices/DEVICE_GAA.puml` |
| TFET | `tunnel_length_nm`, `channel_thickness_nm`, `source/drain_doping_cm3` | пиковое поле на переходе (FEM Лапласа) | поведенческая `.subckt` | `devices/DEVICE_TFET_HEMT.puml` |
| HEMT | `barrier_thickness_nm`, `sheet_density_cm2`, `channel_mobility_cm2_vs`, `access_resistance_ohm_mm`, `gate_recess_nm` | феноменологическая модель + тепловой FEM | поведенческая `.subckt` | `devices/DEVICE_TFET_HEMT.puml` |
| Мемристор | `active_layer_nm`, `substrate_oxide_nm`, `ion_mobility_cm2_vs`, `low/high_resistance_ohm`, `set/reset_voltage_v`, `activation_energy_ev`, `initial_state` | Струков / Biolek, rate-dependent | Verilog-A (OpenVAF) | `devices/DEVICE_MEMRISTOR.puml` |
| Мемтранзистор | те же + `state_relaxation_ms`, `gate_coupling` | связанное ОДУ + модуляция канала по `x(t)` | Verilog-A | `devices/DEVICE_MEMTRANSISTOR.puml` |
| CAD-транзистор | STL (binary/ASCII), единицы м/мм/нм + JSON-параметры | поле переносится на поверхность по габаритам | JSON + поведенческая `.subckt` | `devices/DEVICE_CAD.puml` |

Сводная матрица покрытия «семейство × решатель × метрики» — `devices/D10_DEVICE_MATRIX.puml`.

### 2.3. Инварианты домена (`DeviceParameters.validate()`)

- все физические величины положительны и конечны; `fin_count`, `sheet_count`, `wire_count` — целые;
- `fin_pitch_nm ≥ fin_width_nm`, `sheet_pitch_nm ≥ sheet_thickness_nm`;
- для памяти: `R_on < R_off`, `V_set > 0`, `V_reset < 0`, `initial_state ∈ [0, 1]`;
- имя модели без пробелов; материал барьера/перехода непустой.

### 2.4. Метрики

10 функциональных групп (`metric_labels`, 66 идентификаторов), привязка к семействам —
секция `MF1…MF10` в `devices/D10_DEVICE_MATRIX.puml`:

- подключение: `ION`, `IOFF`, `ION_IOFF`, `VTH_CONSTANT_CURRENT_V`;
- передача: `GM_MAX_S`, `SS_MV_DEC`, `U0`, `NFACTOR`, `VTH0`;
- выход: `RON_OHM`, `GDS_S`, `IDS_MAX`, `VBR_KNEE_V`;
- ёмкости: `CGG`, `CGSO`, `CGDO`, `CGS/CGD/CGB_MEAN_F`;
- ВЧ: `FT_HZ`, `FMAX_HZ`, `H21_MAX`, `MASON_U_MAX`;
- переключение: `TON_10_90_S`, `TOFF_90_10_S`, `QG_C`;
- BJT: `BETA_MAX`, `IC_MAX_A`, `IB_MAX_A`;
- тепло: `TMAX_K`, `DELTA_T_K`, `RTH_K_W`;
- память: `MEMRISTOR_*`, `MEMTRANSISTOR_*`, `MEMORY_STATE_SWING`;
- качество модели: `RMSE_LOG10`, `fitness_score`, `final_mse_validation`.

---

## 3. Алгоритмы

| № | Алгоритм | Ключевые шаги | Диаграмма |
|---|---|---|---|
| A-1 | 3D FEM (электрика/тепло) | бюджет сетки → адаптивное сгущение → сборка → выбор CUDA/CPU → CG (matvec) → карты `|E|`, `Tmax`, `Rth`, `C`; CUDA OOM → CPU | `algorithms/A1_FEM_PIPELINE.puml` |
| A-2 | Quantum-Poisson 2D/3D | Шрёдингер-подзоны → Fermi–Dirac → ρ → шаблон с переменным ε → CG FP64 → контроль Δψ | `algorithms/A2_QUANTUM_POISSON.puml` |
| A-3 | Reduced-order PDE | батч 1D Poisson (Thomas) → интеграл плавного канала → `Cgg = dQg/dVg` | `algorithms/A3_REDUCED_PDE.puml` |
| A-4 | State-модели памяти | окно Biolek / релаксация и сдвиг порога → `R(x)`, `I(t)` → площадь петли | `algorithms/A4_MEMORY_STATE.puml` |
| A-5 | Экстракция BSIM4 | стартовые точки по `gm` → дифференцированная ошибка → `least_squares` → 6 шагов BSIM4 → curve metrics → гладкие ёмкости | `algorithms/A5_EXTRACTION.puml` |
| A-6 | Верификация | предварительный фит → OpenVAF → loopback ngspice → нетлист Micro-Cap → валидатор гистерезиса | `algorithms/A6_VERIFICATION_LOOPBACK.puml` |
| A-7 | U-Net суррогат | 3 независимые сети → обучение (AMP/TF32) → инференс → warm start → ONNX | `algorithms/A7_UNET_SURROGATE.puml` |
| A-8 | **Graph NN суррогат** | меш → граф, MP-слои (GCN/GraphSAGE/GAT), physics-informed loss, active learning, backprop-якобиан, gate-проверки, ONNX | `algorithms/A8_GRAPH_NN.puml` |
| A-9 | Гибридный runtime | маршрутизация «задача → метод → backend», журналирование, асинхронные QThread | `algorithms/A9_HYBRID_RUNTIME.puml` |
| A-10 | Стратегии производных | численные / аналитические / AD / символьные / нейросетевые — и почему только последние три допустимы для Ньютона | `algorithms/A10_GRADIENT_STRATEGIES.puml` |

### 3.1. Graph NN — целевой метод ускорения характеристик

В коде зафиксировано `TODO: Добавить Graph Networks измерить все на метриках`
(`src/neural_physics.py`), зависимости `torch-geometric` и `torchmetrics` уже в
`requirements.txt`. Проектируемый конвейер (A-8):

1. **Кодирование** — узлы = узлы тетраэдрального меша (или параметрические контрольные точки),
   признаки: координаты, материал, ε/μ/N_D/N_A, тип границы, потенциал контакта; рёбра — вершины
   тетраэдров или k-NN; контекст `u = (Vg, Vd, Vb, T, семейство)`.
2. **Сеть** — L слоёв message passing с edge-features, декодеры: поэлементные поля
   (`ψ, n, p, T`) и интегральный readout (`I_d`, `Q_g`, `Q_d`, `Q_s`).
3. **Loss** — `αL_data + βL_Poisson + γL_continuity + δL_BC + ζL_charge` (Тихонов-регуляризация).
4. **Active learning** — неуверенность выше порога ⇒ локальный точный FEM ⇒ on-line fine-tune
   без остановки расчёта (ФТ.5.4).
5. **Инференс** — `O(1)` по размеру сетки, якобиан через backprop (ФТ.5.3), проверки
   монотонности ВАХ и `C²`-непрерывности как **gate** (НФТ.5.2).
6. **Сопряжение** — warm start в Newton/FEM (3–5 итераций вместо 100–200), экспорт ONNX (opset 17),
   метрики в `neural_runs`. При нарушении физической невязки — принудительный прямой FEM.

Почему GNN, а не только U-Net: тетраэдральный меш и CAD-геометрия нерегулярны, а одна графовая
модель покрывает все семейства приборов без интерполяции на регулярную сетку.

---

## 4. Трассировка требований → артефакты

| Требование | Реализующий элемент | Диаграмма |
|---|---|---|
| FR-CORE-01 (сетка, сгущение) | `_make_tetrahedral_mesh`, `_adaptive_axis` | A-1 шаг 2 |
| FR-CORE-02/03 (тепло, Лаплас) | `solve_steady_thermal`, `solve_quasistatic_electric_field` | A-1 шаги 3–6 |
| FR-CORE-04 (Био–Савар) | `solve_quasistatic_electric_field` (аналитическая карта) | A-1 шаг 6, A-9 |
| FR-CORE-05 / NFR-REL-03 (VRAM) | `Backend Selector`, перехват OOM | A-1 шаг 5, `backend/core/FEM_CORE.puml` |
| FR-EXT-01 (стартовые точки) | `extract_idvg` (max gm, экстраполяция) | A-5 шаг 1 |
| FR-EXT-02 (multi-device fit) | `least_squares`, общие параметры + `W_eff` | A-5 шаг 3 |
| FR-EXT-03 (гладкие ёмкости) | tanh/EKV-формы, запрет `POLY(2)` | A-5 шаг 6 |
| FR-EXT-04 (6 шагов BSIM4) | пошаговая последовательность | A-5 шаг 4 |
| FR-EXT-05 (валидация мемристора) | OpenVAF + ngspice, отбраковка прямой линии | A-6, D-7 |
| FR-EXT-06/07 (2 этапа, loopback) | двухэтапный цикл + loopback | A-6 |
| FR-EXP-01 (дифференцированный экспорт) | BSIM4 только для planar, иначе JSON | D-1, D-v |
| NFR-REL-01 (дифференцированная ошибка) | `log10` для подпорога, `%` для надпорога | A-5 шаг 2 |
| NFR-PERF-03 (≥30 % итераций) | warm start суррогата | A-7 шаг 4, A-8 G6 |
| ФТ.5.1–5.4 / НФТ.5.1–5.4 (AI) | GNN/U-Net конвейер | A-8, A-7 |
| NFR-USA-01 (отчёты) | JSON/LaTeX builder | D1 `reports`, A-6 |

---

## 5. Индекс диаграмм

```
architecture/diagrams/
├── enterprise/          L1_SYSTEM_CONTEXT · L2_CONTAINERS · L3_COMPONENTS · L4_CLASSES_DOMAIN
├── database/            D0_REFERENCE_CATALOG · D1_CORE_OPERATIONAL
├── devices/             DEVICE_PLANAR_MOSFET · DEVICE_FD_SOI · DEVICE_FINFET · DEVICE_GAA
│                        DEVICE_TFET_HEMT · DEVICE_MEMRISTOR · DEVICE_MEMTRANSISTOR
│                        DEVICE_CAD · D10_DEVICE_MATRIX · _TEMPLATE_DEVICE_PROFILE
├── algorithms/          A1_FEM_PIPELINE · A2_QUANTUM_POISSON · A3_REDUCED_PDE
│                        A4_MEMORY_STATE · A5_EXTRACTION · A6_VERIFICATION_LOOPBACK
│                        A7_UNET_SURROGATE · A8_GRAPH_NN · A9_HYBRID_RUNTIME
│                        A10_GRADIENT_STRATEGIES
├── backend/             (существующие: схема приложения, FEM-ядро, экстракция, тесты)
├── ui/                  (существующие: UI, UI_STATE, ActivitiesAndroid)
└── deploy/              (существующий: Jenkins ↔ Nexus ↔ Windows GPU)
```

### Ограничения, зафиксированные в модели намеренно

- FEM-электрика — линейное уравнение Лапласа (без подвижного заряда); quantum-Poisson — отдельный
  self-consistent расчёт без уравнений непрерывности и impact ionization.
- CAD STL задаёт отображение поля, а не расчётный объём; DWG/STEP не читаются.
- Мемристор/мемтранзистор — иллюстративные state-модели, не калиброванный материал.
- TFET/HEMT — curve metrics и поведенческие `.subckt`, а не ASM-HEMT/MVSG/BTBT-физика.
- LEVEL=54 для FD-SOI — запрошенная совместимость, не физически корректная BSIM-SOI.
