import numpy as np
import tspice_solver

# Создаем конфигурацию тестового транзистора (например, FinFET)
geom_params = {
    "fin_width_nm": 8.0,
    "fin_height_nm": 40.0,
}

device = tspice_solver.PyDevice(
    device_type="finfet",
    doping=1e16,            # cm^-3
    length_um=0.03,         # 30 nm gate length
    width_um=0.2,
    tox_nm=1.2,
    epsilon_r=11.7,         # Silicon
    temperature_k=300.0,
    geometry=geom_params
)

# Вызываем расчет ускоренного квантового ядра на Rust
print("Запуск симуляции на бэкенде Rust...")
solution = tspice_solver.solve_quantum_charge_distribution(
    device,
    gate_voltage=1.0,
    drain_voltage=0.05,
    dimension="3d",
    grid_shape=(21, 7, 21),
    effective_mass_ratio=0.26
)

# Проверяем возвращенные типы данных (нативные массивы NumPy)
print(f"Успешно выполнен расчет на бэкенде: {solution.backend}")
print(f"Размерность полученной сетки концентрации: {np.shape(solution.values)}")
print(f"Максимальная концентрация электронов: {solution.metrics['QUANTUM_ITERATIONS']} итераций")
print(f"Описание физической модели: {solution.description}")
