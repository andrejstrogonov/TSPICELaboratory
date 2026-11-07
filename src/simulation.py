"""Linear-tetrahedral FEM solvers with optional PyTorch CUDA execution."""

from __future__ import annotations

import math
from dataclasses import dataclass

MU_0 = 4 * math.pi * 1e-7


@dataclass(frozen=True)
class FieldSolution:
    x_m: object
    y_m: object
    z_m: object
    values: object
    metrics: dict[str, float]
    value_name: str
    value_unit: str
    description: str
    backend: str


def solve_steady_thermal(
    device,
    *,
    drain_voltage: float,
    drain_current: float,
    ambient_temperature_k: float,
    thermal_conductivity_w_mk: float,
    substrate_thickness_um: float,
    grid_shape: tuple[int, int, int] = (9, 7, 7),
    use_gpu: bool = True,
) -> FieldSolution:
    """Solve steady heat conduction on a tetrahedral slab with fixed-temperature walls."""
    import numpy as np

    device.validate()
    _validate_operating_point(drain_voltage, drain_current)
    if not math.isfinite(ambient_temperature_k) or ambient_temperature_k <= 0:
        raise ValueError("Ambient temperature must be positive and finite.")
    if not math.isfinite(thermal_conductivity_w_mk) or thermal_conductivity_w_mk <= 0:
        raise ValueError("Thermal conductivity must be positive and finite.")
    if not math.isfinite(substrate_thickness_um) or substrate_thickness_um <= 0:
        raise ValueError("Substrate thickness must be positive and finite.")
    _validate_grid_shape(grid_shape)

    coordinates, tetrahedra, shape, axes = _make_tetrahedral_mesh(
        device, substrate_thickness_um, grid_shape
    )
    x, y, z = axes
    stiffness = _assemble_tetrahedral_stiffness(
        coordinates, tetrahedra, conductivity=thermal_conductivity_w_mk
    )
    boundary = _boundary_nodes(shape)
    free = ~boundary
    source_node = np.ravel_multi_index(
        (shape[0] // 2, shape[1] // 2, max(1, shape[2] - 2)), shape
    )
    if not free[source_node]:
        source_node = int(np.flatnonzero(free)[len(np.flatnonzero(free)) // 2])
    power = drain_voltage * drain_current
    active_coordinates = coordinates[source_node]
    spacing = np.array([axis[1] - axis[0] for axis in axes])
    sigma = np.maximum(spacing * np.array([1.25, 1.25, 0.8]), 1e-12)
    weights = np.exp(
        -0.5 * np.sum(((coordinates - active_coordinates) / sigma) ** 2, axis=1)
    )
    weights[~free] = 0
    weights /= np.sum(weights)
    rhs = power * weights
    temperature_delta, backend = _solve_dirichlet(
        stiffness, rhs, boundary, np.zeros(len(coordinates)), free, use_gpu
    )
    temperature = ambient_temperature_k + temperature_delta.reshape(shape)
    maximum = float(np.max(temperature))
    return FieldSolution(
        x_m=x,
        y_m=y,
        z_m=z,
        values=temperature,
        metrics={
            "POWER_W": power,
            "TMAX_K": maximum,
            "DELTA_TMAX_K": maximum - ambient_temperature_k,
            "RTH_K_W": (maximum - ambient_temperature_k) / power if power else 0.0,
        },
        value_name="Температура",
        value_unit="K",
        description=(
            "Линейный тетраэдральный FEM: стационарная теплопроводность, "
            "постоянная теплопроводность и фиксированная температура внешних граней."
        ),
        backend=backend,
    )


def solve_quasistatic_electric_field(
    device,
    *,
    gate_voltage: float,
    drain_voltage: float,
    drain_current: float,
    grid_shape: tuple[int, int, int] = (9, 7, 7),
    use_gpu: bool = True,
) -> tuple[FieldSolution, FieldSolution]:
    """Solve a 3D electrostatic Laplace problem on a tetrahedral mesh."""
    import numpy as np

    device.validate()
    _validate_operating_point(drain_voltage, drain_current)
    if not math.isfinite(gate_voltage):
        raise ValueError("Gate voltage must be finite.")
    _validate_grid_shape(grid_shape)
    coordinates, tetrahedra, shape, axes = _make_tetrahedral_mesh(
        device, 0.05, grid_shape
    )
    x, y, z = axes
    stiffness = _assemble_tetrahedral_stiffness(
        coordinates, tetrahedra, conductivity=device.epsilon_r
    )
    boundary = _boundary_nodes(shape)
    fixed_values = np.zeros(len(coordinates), dtype=np.float64)
    grid = np.arange(len(coordinates)).reshape(shape)
    fixed_values[grid[-1, :, :].ravel()] = drain_voltage
    gate_start, gate_end = max(1, shape[0] // 4), min(
        shape[0] - 1, (3 * shape[0]) // 4 + 1
    )
    gate_nodes = grid[gate_start:gate_end, :, -1].ravel()
    boundary[gate_nodes] = True
    fixed_values[gate_nodes] = gate_voltage
    free = ~boundary
    rhs = np.zeros(len(coordinates), dtype=np.float64)
    potential, backend = _solve_dirichlet(
        stiffness, rhs, boundary, fixed_values, free, use_gpu
    )
    potential_grid = potential.reshape(shape)
    dx, dy, dz = (axis[1] - axis[0] for axis in axes)
    gradients = np.gradient(potential_grid, dx, dy, dz, edge_order=1)
    field_magnitude = np.sqrt(sum(component**2 for component in gradients))
    peak_field = float(np.max(field_magnitude))
    separation = max(min(device.length_um * 1e-6, device.tox_nm * 1e-9), 1e-12)
    xx, yy, zz = np.meshgrid(x, y, z, indexing="ij")
    dimensions = (x[-1], y[-1], z[-1])
    wire_radius = max(_active_thickness_m(device) / 2, 1e-12)
    radial_distance = np.maximum(
        np.hypot(
            yy - dimensions[1] / 2,
            zz - dimensions[2] + _active_thickness_m(device) / 2,
        ),
        wire_radius,
    )
    left_distance = np.sqrt(xx**2 + radial_distance**2)
    right_distance = np.sqrt((dimensions[0] - xx) ** 2 + radial_distance**2)
    sin_left = np.clip(xx / left_distance, 0, 1)
    sin_right = np.clip((dimensions[0] - xx) / right_distance, 0, 1)
    magnetic_field_grid = (
        MU_0 * drain_current * (sin_left + sin_right) / (4 * math.pi * radial_distance)
    )
    magnetic_peak = float(np.max(magnetic_field_grid))
    capacitance = (
        8.8541878128e-12
        * device.epsilon_r
        * device.effective_width_um
        * 1e-6
        * device.length_um
        * 1e-6
        / (device.tox_nm * 1e-9)
    )
    electric_solution = FieldSolution(
        x_m=x,
        y_m=y,
        z_m=z,
        values=field_magnitude,
        metrics={
            "EMAX_V_M": peak_field,
            "CGEO_F": capacitance,
            "ELECTROSTATIC_ENERGY_J": 0.5 * capacitance * gate_voltage**2,
            "VGS_V": gate_voltage,
            "VDS_V": drain_voltage,
        },
        value_name="Модуль электрического поля",
        value_unit="V/m",
        description=(
            "Линейный тетраэдральный FEM для уравнения Лапласа с постоянной "
            "диэлектрической проницаемостью; подвижный заряд не учитывается. "
            "B оценено аналитически по закону Ампера."
        ),
        backend=backend,
    )
    magnetic_solution = FieldSolution(
        x_m=x,
        y_m=y,
        z_m=z,
        values=magnetic_field_grid,
        metrics={
            "BMAX_EST_T": magnetic_peak,
            "IDS_A": drain_current,
        },
        value_name="Оценка магнитной индукции (Biot–Savart)",
        value_unit="T",
        description=(
            "Аналитическое поле конечного прямого проводника с заданным током; "
            "в FEM-матрицу не включено."
        ),
        backend="Аналитическая оценка Biot–Savart",
    )
    return electric_solution, magnetic_solution


def _make_tetrahedral_mesh(device, substrate_thickness_um, grid_shape):
    import numpy as np

    nx, ny, nz = grid_shape
    active = _active_thickness_m(device)
    dimensions = (
        device.length_um * 1e-6,
        max(device.effective_width_um * 1e-6, active),
        substrate_thickness_um * 1e-6 + active + device.tox_nm * 1e-9,
    )
    axes = tuple(np.linspace(0.0, size, count) for size, count in zip(dimensions, grid_shape))
    xx, yy, zz = np.meshgrid(*axes, indexing="ij")
    coordinates = np.column_stack((xx.ravel(), yy.ravel(), zz.ravel()))
    node_grid = np.arange(nx * ny * nz).reshape(grid_shape)
    tetrahedra = []
    corners = (
        (0, 0, 0),
        (1, 0, 0),
        (1, 1, 0),
        (0, 1, 0),
        (0, 0, 1),
        (1, 0, 1),
        (1, 1, 1),
        (0, 1, 1),
    )
    local_tetrahedra = (
        (0, 1, 2, 6),
        (0, 2, 3, 6),
        (0, 3, 7, 6),
        (0, 7, 4, 6),
        (0, 4, 5, 6),
        (0, 5, 1, 6),
    )
    for i in range(nx - 1):
        for j in range(ny - 1):
            for k in range(nz - 1):
                cube_nodes = [
                    node_grid[i + di, j + dj, k + dk] for di, dj, dk in corners
                ]
                tetrahedra.extend(
                    [cube_nodes[index] for index in tetra]
                    for tetra in local_tetrahedra
                )
    return coordinates, np.asarray(tetrahedra, dtype=np.int64), grid_shape, axes


def _assemble_tetrahedral_stiffness(coordinates, tetrahedra, conductivity):
    import numpy as np
    from scipy.sparse import coo_matrix

    points = coordinates[tetrahedra]
    shape_matrices = np.concatenate(
        (np.ones((*points.shape[:2], 1)), points), axis=2
    )
    determinants = np.linalg.det(shape_matrices)
    volumes = np.abs(determinants) / 6
    if np.any(volumes <= 0):
        raise ValueError("The tetrahedral mesh contains a degenerate element.")
    shape_inverses = np.linalg.inv(shape_matrices)
    gradients = shape_inverses[:, 1:, :].transpose(0, 2, 1)
    local_matrices = conductivity * volumes[:, None, None] * np.einsum(
        "eia,eja->eij", gradients, gradients
    )
    row_indices = np.repeat(tetrahedra, 4, axis=1).ravel()
    column_indices = np.tile(tetrahedra, (1, 4)).ravel()
    nodes_count = len(coordinates)
    return coo_matrix(
        (
            local_matrices.ravel(),
            (row_indices, column_indices),
        ),
        shape=(nodes_count, nodes_count),
    ).tocsr()


def _solve_dirichlet(stiffness, rhs, boundary, boundary_values, free, use_gpu):
    import numpy as np
    from scipy.sparse.linalg import spsolve

    matrix = stiffness[free][:, free].tocsr()
    reduced_rhs = rhs[free] - stiffness[free][:, boundary] @ boundary_values[boundary]
    result = np.array(boundary_values, dtype=np.float64, copy=True)
    backend = "CPU (SciPy sparse direct solver)"
    if use_gpu:
        try:
            import torch

            if torch.cuda.is_available():
                coo = matrix.tocoo()
                indices = torch.as_tensor(
                    np.vstack((coo.row, coo.col)),
                    dtype=torch.long,
                    device="cuda",
                )
                values = torch.as_tensor(coo.data, dtype=torch.float64, device="cuda")
                operator = torch.sparse_coo_tensor(
                    indices, values, coo.shape, device="cuda"
                ).coalesce()
                vector = torch.as_tensor(reduced_rhs, dtype=torch.float64, device="cuda")
                solution = _conjugate_gradient(operator, vector)
                result[free] = solution.cpu().numpy()
                backend = f"GPU (PyTorch CUDA: {torch.cuda.get_device_name(0)})"
                return result, backend
        except ImportError:
            pass
    result[free] = spsolve(matrix, reduced_rhs)
    return result, backend


def _conjugate_gradient(operator, rhs, *, tolerance=1e-10, max_iterations=10000):
    import torch

    solution = torch.zeros_like(rhs)
    residual = rhs - torch.sparse.mm(operator, solution.unsqueeze(1)).squeeze(1)
    indices = operator.indices()
    diagonal_mask = indices[0] == indices[1]
    diagonal = torch.zeros_like(rhs)
    diagonal.index_add_(0, indices[0, diagonal_mask], operator.values()[diagonal_mask])
    if torch.any(diagonal <= 0):
        raise RuntimeError("CUDA FEM matrix has a non-positive diagonal.")
    preconditioned = residual / diagonal
    direction = preconditioned.clone()
    residual_product = torch.dot(residual, preconditioned)
    initial_norm = torch.linalg.vector_norm(residual)
    if initial_norm.item() == 0:
        return solution
    for _ in range(max_iterations):
        product = torch.sparse.mm(operator, direction.unsqueeze(1)).squeeze(1)
        denominator = torch.dot(direction, product)
        if denominator.item() <= 0:
            raise RuntimeError("CUDA FEM matrix is not positive definite.")
        alpha = residual_product / denominator
        solution = solution + alpha * direction
        residual = residual - alpha * product
        if torch.linalg.vector_norm(residual).item() <= tolerance * initial_norm.item():
            return solution
        preconditioned = residual / diagonal
        new_product = torch.dot(residual, preconditioned)
        direction = preconditioned + (new_product / residual_product) * direction
        residual_product = new_product
    raise RuntimeError("CUDA FEM solver did not converge within the iteration limit.")


def _boundary_nodes(shape):
    import numpy as np

    mask = np.zeros(shape, dtype=bool)
    mask[[0, -1], :, :] = True
    mask[:, [0, -1], :] = True
    mask[:, :, [0, -1]] = True
    return mask.ravel()


def _active_thickness_m(device) -> float:
    geometry = device.geometry
    if device.device_type == "finfet":
        return geometry["fin_height_nm"] * 1e-9
    if device.device_type == "gaa_nanosheet":
        return geometry["sheet_thickness_nm"] * 1e-9
    if device.device_type == "gaa_nanowire":
        return geometry["wire_diameter_nm"] * 1e-9
    if device.device_type == "fd_soi":
        return geometry["body_thickness_nm"] * 1e-9
    if device.device_type == "tfet":
        return geometry["channel_thickness_nm"] * 1e-9
    if device.device_type == "hemt":
        return geometry["barrier_thickness_nm"] * 1e-9
    return max(device.tox_nm * 2e-9, 5e-9)


def _validate_operating_point(drain_voltage: float, drain_current: float) -> None:
    if not math.isfinite(drain_voltage):
        raise ValueError("Drain voltage must be finite.")
    if not math.isfinite(drain_current) or drain_current < 0:
        raise ValueError("Drain current must be non-negative and finite.")


def _validate_grid_shape(grid_shape: tuple[int, int, int]) -> None:
    if len(grid_shape) != 3 or any(
        not isinstance(size, int) or size < 4 or size > 35 for size in grid_shape
    ):
        raise ValueError("Each 3D FEM grid dimension must be an integer from 4 to 35.")
