"""Linear-tetrahedral FEM solvers with optional PyTorch CUDA execution."""

from __future__ import annotations

import math
from dataclasses import dataclass

MU_0 = 4 * math.pi * 1e-7
EPSILON_0 = 8.8541878128e-12
ELECTRON_CHARGE = 1.602176634e-19
BOLTZMANN = 1.380649e-23
PLANCK_REDUCED = 1.054571817e-34
ELECTRON_MASS = 9.1093837139e-31
MAX_MESH_NODES = 3_000_000
ESTIMATED_PEAK_BYTES_PER_NODE = 2_500


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


def solve_quantum_charge_distribution(
    device,
    *,
    gate_voltage: float,
    drain_voltage: float,
    dimension: str = "2d",
    grid_shape: tuple[int, int, int] = (17, 7, 17),
    effective_mass_ratio: float = 0.26,
    use_gpu: bool = True,
) -> FieldSolution:
    """Solve a reduced self-consistent quantum Poisson problem for electron density.

    The model resolves x-z electrostatics in 2D or x-y-z in 3D. Quantum
    confinement is represented by transverse Schrödinger subbands along z
    coupled to a 2D Fermi-Dirac subband population. It is not a full
    non-equilibrium drift-diffusion TCAD solver.
    """
    import numpy as np

    device.validate()
    supported = {
        "planar_mosfet",
        "fd_soi",
        "finfet",
        "gaa_nanosheet",
        "gaa_nanowire",
    }
    if device.device_type not in supported:
        raise ValueError(
            "Quantum charge solving currently supports planar MOSFET, FD-SOI, "
            "FinFET, GAA nanosheet, and GAA nanowire only; TFET/HEMT require "
            "different band-structure and transport models."
        )
    if dimension not in ("2d", "3d"):
        raise ValueError("Dimension must be '2d' or '3d'.")
    if len(grid_shape) != 3:
        raise ValueError("Charge grid must be (x, y, z) node counts.")
    nx, ny_requested, nz = map(int, grid_shape)
    ny = 1 if dimension == "2d" else ny_requested
    if nx < 7 or nz < 7 or (dimension == "3d" and ny < 5):
        raise ValueError("Charge grid requires >=7 x/z nodes and >=5 y nodes in 3D.")
    if (
        nx * ny * nz > 30_000
        or nx * ny > 1_000
        or nx * ny * (nz - 2) ** 3 > 12_000_000
    ):
        raise ValueError(
            "Charge grid exceeds the memory/eigensolver work limit; "
            "reduce node counts (maximum 30,000 cells)."
        )
    if not math.isfinite(gate_voltage) or not math.isfinite(drain_voltage):
        raise ValueError("Gate and drain voltages must be finite.")
    if drain_voltage < 0:
        raise ValueError("Drain voltage must be non-negative for this n-channel model.")
    if not math.isfinite(effective_mass_ratio) or not 0.05 <= effective_mass_ratio <= 2:
        raise ValueError("Electron effective mass ratio must be between 0.05 and 2.")
    if device.doping > 1e19:
        raise ValueError(
            "Quantum charge solver uses a non-degenerate bulk reservoir; "
            "doping above 1e19 cm^-3 is outside this model."
        )

    length_m = device.length_um * 1e-6
    if device.device_type == "finfet":
        width_m = device.geometry["fin_width_nm"] * 1e-9
    elif device.device_type == "gaa_nanosheet":
        width_m = device.geometry["sheet_width_nm"] * 1e-9
    elif device.device_type == "gaa_nanowire":
        width_m = device.geometry["wire_diameter_nm"] * 1e-9
    else:
        width_m = device.width_um * 1e-6
    if device.device_type == "fd_soi":
        depth_m = device.geometry["body_thickness_nm"] * 1e-9
    elif device.device_type == "finfet":
        depth_m = device.geometry["fin_width_nm"] * 1e-9
    elif device.device_type == "gaa_nanosheet":
        depth_m = device.geometry["sheet_thickness_nm"] * 1e-9
    elif device.device_type == "gaa_nanowire":
        depth_m = device.geometry["wire_diameter_nm"] * 1e-9
    else:
        depth_m = 80e-9
    depth_m = max(depth_m, 2e-9)
    x_m = np.linspace(0.0, length_m, nx)
    y_m = (
        np.linspace(-width_m / 2, width_m / 2, ny)
        if dimension == "3d"
        else np.array([ -width_m / 2, width_m / 2], dtype=np.float64)
    )
    z_m = np.linspace(0.0, depth_m, nz)
    epsilon = EPSILON_0 * device.epsilon_r
    cox = EPSILON_0 * device.epsilon_r / (device.tox_nm * 1e-9)
    doping_m3 = device.doping * 1e6
    mass = effective_mass_ratio * ELECTRON_MASS
    kT = BOLTZMANN * device.temperature_k
    nc_m3 = 2 * (
        2 * math.pi * mass * kT / (2 * math.pi * PLANCK_REDUCED) ** 2
    ) ** 1.5
    bulk_eta = math.log(doping_m3 / nc_m3)
    spacings = (x_m[1] - x_m[0], width_m / max(ny - 1, 1), z_m[1] - z_m[0])
    gated_sidewalls = device.device_type in (
        "finfet",
        "gaa_nanosheet",
        "gaa_nanowire",
    )
    gated_bottom = device.device_type in ("gaa_nanosheet", "gaa_nanowire")

    backend = "CPU (NumPy)"
    if use_gpu:
        try:
            import torch

            if torch.cuda.is_available():
                required = nx * ny * nz * 8 * 16 + nx * ny * nz * nz * 8 * 3
                available, total = torch.cuda.mem_get_info()
                if required < available * 0.7:
                    try:
                        potential, electron_density, iterations = (
                            _quantum_poisson_torch(
                                torch,
                                gate_voltage,
                                drain_voltage,
                                nx,
                                ny,
                                nz,
                                spacings,
                                epsilon,
                                cox,
                                doping_m3,
                                mass,
                                kT,
                                bulk_eta,
                                length_m,
                                gated_sidewalls,
                                gated_bottom,
                            )
                        )
                        backend = (
                            f"CUDA FP64 (PyTorch: {torch.cuda.get_device_name(0)})"
                        )
                    except RuntimeError as error:
                        torch.cuda.empty_cache()
                        backend = (
                            f"CPU (CUDA failed: {type(error).__name__}; fallback)"
                        )
                else:
                    backend = (
                        f"CPU (CUDA skipped: estimated {required / 1e9:.3g} GB; "
                        f"{available / 1e9:.3g}/{total / 1e9:.3g} GB free)"
                    )
            else:
                backend = "CPU (CUDA unavailable)"
        except ImportError:
            backend = "CPU (PyTorch CUDA not installed)"

    if backend.startswith("CPU"):
        potential, electron_density, iterations = _quantum_poisson_numpy(
            gate_voltage,
            drain_voltage,
            nx,
            ny,
            nz,
            spacings,
            epsilon,
            cox,
            doping_m3,
            mass,
            kT,
            bulk_eta,
            length_m,
            gated_sidewalls,
            gated_bottom,
        )

    if dimension == "2d":
        electron_density = np.repeat(electron_density, 2, axis=1)
    concentration_cm3 = electron_density / 1e6
    metrics = {
        "GATE_V": float(gate_voltage),
        "DRAIN_V": float(drain_voltage),
        "GRID_X": float(nx),
        "GRID_Y": float(ny),
        "GRID_Z": float(nz),
        "MAX_ELECTRON_DENSITY_CM3": float(np.max(concentration_cm3)),
        "MEAN_ELECTRON_DENSITY_CM3": float(np.mean(concentration_cm3)),
        "QUANTUM_ITERATIONS": float(iterations),
        "EFFECTIVE_MASS_RATIO": float(effective_mass_ratio),
    }
    return FieldSolution(
        x_m=x_m,
        y_m=y_m,
        z_m=z_m,
        values=concentration_cm3,
        metrics=metrics,
        value_name=f"Electron concentration · self-consistent {dimension.upper()} quantum Poisson",
        value_unit="cm^-3",
        description=(
            f"{dimension.upper()} nonlinear Poisson + transverse Schrödinger "
            "subbands (effective-mass approximation, Fermi-Dirac 2D occupancy); "
            "uniform material and quasi-equilibrium local quasi-Fermi level. "
            + (
                "Multi-gate sidewalls are included for FinFET/GAA."
                if gated_sidewalls
                else "Single top gate with a neutral bulk boundary."
            )
            + (
                " GAA bottom surface is gate-coupled."
                if gated_bottom
                else ""
            )
        ),
        backend=backend,
    )


def _quantum_poisson_numpy(
    gate_voltage,
    drain_voltage,
    nx,
    ny,
    nz,
    spacings,
    epsilon,
    cox,
    doping_m3,
    mass,
    kT,
    bulk_eta,
    length_m,
    gated_sidewalls,
    gated_bottom,
):
    import numpy as np
    from scipy.sparse.linalg import factorized

    dx, dy, dz = spacings
    matrix, boundary_rhs, charge_mask = _assemble_quantum_poisson_matrix(
        nx,
        ny,
        nz,
        dx,
        dy,
        dz,
        epsilon,
        cox,
        gate_voltage,
        drain_voltage,
        gated_sidewalls,
        gated_bottom,
    )
    solve_linear = factorized(matrix.tocsc())
    lateral = np.linspace(0.0, length_m, nx)[:, None, None]
    kinetic = PLANCK_REDUCED**2 / (2 * mass * dz**2)
    interior = nz - 2
    off_diagonal = -kinetic
    potential = np.zeros((nx, ny, nz), dtype=np.float64)
    potential[-1, :, :] = drain_voltage
    iterations = 0

    for outer in range(200):
        hamiltonian = np.zeros((nx, ny, interior, interior), dtype=np.float64)
        diagonal = 2 * kinetic - ELECTRON_CHARGE * potential[:, :, 1:-1]
        indices = np.arange(interior)
        hamiltonian[:, :, indices, indices] = diagonal
        hamiltonian[:, :, indices[:-1], indices[1:]] = off_diagonal
        hamiltonian[:, :, indices[1:], indices[:-1]] = off_diagonal
        energies, wavefunctions = np.linalg.eigh(hamiltonian)
        fermi_level = (
            kT * bulk_eta
            - ELECTRON_CHARGE * drain_voltage * lateral[:, 0, :] / length_m
        )
        occupancy = np.logaddexp(
            0.0, (fermi_level[..., None] - energies) / kT
        )
        density_of_states = mass * kT / (math.pi * PLANCK_REDUCED**2)
        sheet_density = density_of_states * occupancy
        electron_density = np.empty_like(potential)
        electron_density[:, :, 1:-1] = (
            np.sum(np.square(wavefunctions) * sheet_density[..., None, :], axis=-1)
            / dz
        )
        electron_density[:, :, 0] = electron_density[:, :, 1]
        electron_density[:, :, -1] = electron_density[:, :, -2]
        charge_density = ELECTRON_CHARGE * (doping_m3 - electron_density)
        rhs = boundary_rhs + charge_density.ravel() * charge_mask
        updated = solve_linear(rhs).reshape((nx, ny, nz))
        mixed = 0.12 * updated + 0.88 * potential
        iterations = outer + 1
        outer_change = float(np.max(np.abs(mixed - potential)))
        if outer_change < 2e-6:
            potential = mixed
            break
        potential = mixed
    else:
        raise RuntimeError(
            "Self-consistent quantum Poisson solver did not converge (CPU); "
            f"last potential update was {outer_change:.3g} V."
        )

    return potential, electron_density, iterations


def _assemble_quantum_poisson_matrix(
    nx,
    ny,
    nz,
    dx,
    dy,
    dz,
    epsilon,
    cox,
    gate_voltage,
    drain_voltage,
    gated_sidewalls,
    gated_bottom,
):
    import numpy as np
    from scipy.sparse import lil_matrix

    shape = (nx, ny, nz)
    matrix = lil_matrix((math.prod(shape), math.prod(shape)), dtype=np.float64)
    boundary_rhs = np.zeros(math.prod(shape), dtype=np.float64)
    charge_volume = np.zeros(shape, dtype=np.float64)
    dirichlet = {}
    for j in range(ny):
        for k in range(nz):
            dirichlet[(0, j, k)] = 0.0
            dirichlet[(nx - 1, j, k)] = drain_voltage
    if not gated_bottom:
        for i in range(1, nx - 1):
            for j in range(ny):
                dirichlet[(i, j, nz - 1)] = 0.0

    def flat(index):
        return np.ravel_multi_index(index, shape)

    cell_x = np.full(nx, dx, dtype=np.float64)
    cell_x[[0, -1]] *= 0.5
    cell_y = (
        np.ones(1, dtype=np.float64)
        if ny == 1
        else np.full(ny, dy, dtype=np.float64)
    )
    if ny > 1:
        cell_y[[0, -1]] *= 0.5
    cell_z = np.full(nz, dz, dtype=np.float64)
    cell_z[[0, -1]] *= 0.5

    for i in range(nx):
        for j in range(ny):
            for k in range(nz):
                row = flat((i, j, k))
                if (i, j, k) in dirichlet:
                    matrix[row, row] = 1.0
                    boundary_rhs[row] = dirichlet[(i, j, k)]
                    continue

                volume = cell_x[i] * cell_y[j] * cell_z[k]
                charge_volume[i, j, k] = volume
                diagonal = 0.0
                for axis, spacing in enumerate((dx, dy, dz)):
                    if axis == 1 and ny == 1:
                        continue
                    if axis == 0:
                        area = cell_y[j] * cell_z[k]
                        coordinate = i
                    elif axis == 1:
                        area = cell_x[i] * cell_z[k]
                        coordinate = j
                    else:
                        area = cell_x[i] * cell_y[j]
                        coordinate = k
                    conductance = epsilon * area / spacing
                    for direction in (-1, 1):
                        neighbor = [i, j, k]
                        neighbor[axis] += direction
                        neighbor = tuple(neighbor)
                        if neighbor[axis] < 0 or neighbor[axis] >= shape[axis]:
                            if axis == 1 and gated_sidewalls:
                                gate_conductance = cox * area
                                diagonal += gate_conductance
                                boundary_rhs[row] += gate_conductance * gate_voltage
                            elif axis == 2 and (
                                (coordinate == 0 and direction < 0)
                                or (
                                    gated_bottom
                                    and coordinate == nz - 1
                                    and direction > 0
                                )
                            ):
                                gate_conductance = cox * area
                                diagonal += gate_conductance
                                boundary_rhs[row] += gate_conductance * gate_voltage
                            continue
                        if neighbor in dirichlet:
                            diagonal += conductance
                            boundary_rhs[row] += (
                                conductance * dirichlet[neighbor]
                            )
                        else:
                            diagonal += conductance
                            matrix[row, flat(neighbor)] -= conductance
                matrix[row, row] = diagonal
    return matrix.tocsr(), boundary_rhs, charge_volume.ravel()


def _quantum_poisson_torch(
    torch,
    gate_voltage,
    drain_voltage,
    nx,
    ny,
    nz,
    spacings,
    epsilon,
    cox,
    doping_m3,
    mass,
    kT,
    bulk_eta,
    length_m,
    gated_sidewalls,
    gated_bottom,
    device_type="cuda",
):
    device = torch.device(device_type)

    dx, dy, dz = spacings
    lateral = torch.linspace(0, length_m, nx, dtype=torch.float64, device=device)[
        :, None, None
    ]
    potential = (
        gate_voltage
        * torch.clamp(
            1
            - torch.linspace(0, 1, nz, dtype=torch.float64, device=device)[
                None, None, :
            ],
            min=0,
        )
        + drain_voltage * lateral / length_m
    ).expand(nx, ny, nz).clone()
    poisson_matrix, boundary_rhs, charge_mask = _assemble_quantum_poisson_matrix(
        nx,
        ny,
        nz,
        dx,
        dy,
        dz,
        epsilon,
        cox,
        gate_voltage,
        drain_voltage,
        gated_sidewalls,
        gated_bottom,
    )
    stencil = _torch_poisson_stencil(
        torch, poisson_matrix, device, (nx, ny, nz)
    )
    diagonal = stencil[0]
    boundary_rhs = torch.as_tensor(boundary_rhs, dtype=torch.float64, device=device)
    charge_mask = torch.as_tensor(charge_mask, dtype=torch.float64, device=device)
    kinetic = PLANCK_REDUCED**2 / (2 * mass * dz**2)
    interior = nz - 2
    hamiltonian_base = torch.diag(
        torch.full((interior,), 2 * kinetic, dtype=torch.float64, device=device)
    )
    hamiltonian_base += torch.diag(
        torch.full((interior - 1,), -kinetic, dtype=torch.float64, device=device),
        diagonal=1,
    )
    hamiltonian_base += torch.diag(
        torch.full((interior - 1,), -kinetic, dtype=torch.float64, device=device),
        diagonal=-1,
    )
    iterations = 0
    for outer in range(200):
        hamiltonian = hamiltonian_base + torch.diag_embed(
            -ELECTRON_CHARGE * potential[:, :, 1:-1]
        )
        energies, wavefunctions = torch.linalg.eigh(hamiltonian)
        fermi_level = (
            kT * bulk_eta
            - ELECTRON_CHARGE * drain_voltage * lateral[:, 0, :] / length_m
        )
        occupancy = torch.nn.functional.softplus(
            (fermi_level[..., None] - energies) / kT
        )
        density_of_states = mass * kT / (math.pi * PLANCK_REDUCED**2)
        sheet_density = density_of_states * occupancy
        electron_density = torch.empty_like(potential)
        electron_density[:, :, 1:-1] = (
            torch.sum(wavefunctions.square() * sheet_density[..., None, :], dim=-1)
            / dz
        )
        electron_density[:, :, 0] = electron_density[:, :, 1]
        electron_density[:, :, -1] = electron_density[:, :, -2]
        charge_density = ELECTRON_CHARGE * (doping_m3 - electron_density)
        rhs = boundary_rhs + charge_density.reshape(-1) * charge_mask
        updated = _solve_poisson_cg(
            torch, stencil, rhs, potential.reshape(-1), (nx, ny, nz)
        ).reshape(nx, ny, nz)
        mixed = 0.12 * updated + 0.88 * potential
        iterations = outer + 1
        outer_change = torch.max(torch.abs(mixed - potential)).item()
        if outer_change < 2e-6:
            potential = mixed
            break
        potential = mixed
    else:
        raise RuntimeError(
            "Self-consistent quantum Poisson solver did not converge (CUDA); "
            f"last potential update was {outer_change:.3g} V."
        )
    if device.type == "cuda":
        torch.cuda.synchronize()
    return (
        potential.cpu().numpy(),
        electron_density.cpu().numpy(),
        iterations,
    )


def _torch_poisson_stencil(torch, matrix, device, shape):
    """Move the seven-point finite-volume operator to CUDA without sparse CSR."""
    nx, ny, nz = shape
    offsets = []
    if nx > 1:
        offsets.append(ny * nz)
    if ny > 1:
        offsets.append(nz)
    if nz > 1:
        offsets.append(1)
    return (
        torch.as_tensor(matrix.diagonal(), dtype=torch.float64, device=device),
        *(
            (
                offset,
                torch.as_tensor(
                    matrix.diagonal(offset),
                    dtype=torch.float64,
                    device=device,
                ),
                torch.as_tensor(
                    matrix.diagonal(-offset),
                    dtype=torch.float64,
                    device=device,
                ),
            )
            for offset in offsets
        ),
    )


def _solve_poisson_cg(torch, stencil, rhs, initial, shape):
    solution = initial.clone()
    diagonal = stencil[0]
    residual = rhs - _apply_poisson_stencil(torch, stencil, solution, shape)
    preconditioned = residual / diagonal
    direction = preconditioned.clone()
    residual_product = torch.dot(residual, preconditioned)
    initial_norm = torch.linalg.vector_norm(residual).item()
    if initial_norm == 0:
        return solution
    converged = False
    residual_norm = initial_norm
    for iteration in range(600):
        product = _apply_poisson_stencil(torch, stencil, direction, shape)
        denominator = torch.dot(direction, product)
        alpha = residual_product / torch.clamp(denominator, min=1e-300)
        solution = solution + alpha * direction
        residual = residual - alpha * product
        if iteration % 10 == 9 or iteration == 599:
            residual_norm = torch.linalg.vector_norm(residual).item()
            if residual_norm <= initial_norm * 1e-7:
                converged = True
                break
        preconditioned = residual / diagonal
        next_product = torch.dot(residual, preconditioned)
        beta = next_product / torch.clamp(residual_product, min=1e-300)
        direction = preconditioned + beta * direction
        residual_product = next_product
    if not converged:
        raise RuntimeError(
            "CUDA sparse Poisson CG did not converge; relative residual "
            f"{residual_norm / max(initial_norm, 1e-300):.3g}."
        )
    return solution


def _apply_poisson_stencil(torch, stencil, vector, shape):
    diagonal, *neighbors = stencil
    result = diagonal * vector
    size = vector.numel()
    for offset, upper, lower in neighbors:
        result[: size - offset] += upper * vector[offset:]
        result[offset:] += lower * vector[: size - offset]
    return result


@dataclass(frozen=True)
class GeneratedCurve:
    kind: str
    x: list[float]
    y: list[float]
    label: str
    columns: dict[str, list[float]]


@dataclass(frozen=True)
class PDECharacteristicResult:
    curves: list[GeneratedCurve]
    backend: str
    diagnostics: dict[str, float]
    model_description: str


def solve_pde_characteristics(
    device,
    *,
    drain_linear_v: float = 0.05,
    drain_saturation_v: float = 0.8,
    drain_output_max_v: float = 0.8,
    gate_min_v: float = -0.5,
    gate_max_v: float = 1.2,
    gate_points: int = 31,
    drain_points: int = 25,
    channel_points: int = 17,
    poisson_nodes: int = 61,
    use_gpu: bool = True,
) -> PDECharacteristicResult:
    """Generate quasi-static MOS curves from 1D Poisson + charge-sheet transport.

    The semiconductor cross-section uses Boltzmann carrier statistics (with
    exponents limited to [-30, 50]) and a neutral-body boundary. Channel current
    follows the gradual-channel drift-diffusion integral. This is a reduced
    model, not a 2D/3D TCAD solver.
    """
    import numpy as np

    device.validate()
    supported = {
        "planar_mosfet",
        "fd_soi",
        "finfet",
        "gaa_nanosheet",
        "gaa_nanowire",
    }
    if device.device_type not in supported:
        raise ValueError(
            "PDE characteristic generation currently supports planar MOSFET, "
            "FD-SOI, FinFET, GAA nanosheet, and GAA nanowire only. HEMT and TFET "
            "need different carrier/transport equations; no synthetic curves were generated."
        )
    for name, value in (
        ("linear drain voltage", drain_linear_v),
        ("saturation drain voltage", drain_saturation_v),
        ("output drain voltage", drain_output_max_v),
    ):
        if not math.isfinite(value) or value <= 0:
            raise ValueError(f"{name.capitalize()} must be positive and finite.")
    if not math.isfinite(gate_min_v) or not math.isfinite(gate_max_v) or gate_min_v >= gate_max_v:
        raise ValueError("Gate-voltage sweep bounds must be finite and increasing.")
    if gate_points < 5 or drain_points < 3 or channel_points < 3 or poisson_nodes < 11:
        raise ValueError("PDE sweep requires >=5 Vg, >=3 Vd/channel, and >=11 Poisson nodes.")

    vg_values = np.linspace(gate_min_v, gate_max_v, gate_points)
    drain_values = np.linspace(0.0, drain_output_max_v, drain_points)
    channel_fraction = np.linspace(0.0, 1.0, channel_points)
    output_gate_biases = np.linspace(
        gate_min_v + 0.65 * (gate_max_v - gate_min_v),
        gate_max_v,
        4,
    )
    requests: list[tuple[str, np.ndarray, np.ndarray]] = []

    def add_request(name, gate_bias, drain_bias):
        gate_bias, drain_bias = np.broadcast_arrays(gate_bias, drain_bias)
        local_channel = drain_bias[..., None] * channel_fraction
        effective_gate = gate_bias[..., None] - local_channel
        requests.append(
            (
                name,
                effective_gate.reshape(-1),
                local_channel.reshape(-1),
                effective_gate.shape,
            )
        )

    add_request("transfer_linear", vg_values, np.full_like(vg_values, drain_linear_v))
    add_request(
        "transfer_saturation",
        vg_values,
        np.full_like(vg_values, drain_saturation_v),
    )
    add_request("cv", vg_values, np.zeros_like(vg_values))
    for curve_index, gate_bias in enumerate(output_gate_biases):
        add_request(
            f"output_{curve_index}",
            np.full_like(drain_values, gate_bias),
            drain_values,
        )

    all_effective_gate = np.concatenate([request[1] for request in requests])
    all_channel_voltage = np.concatenate([request[2] for request in requests])
    surface_potential, inversion_charge, gate_charge, backend = _solve_poisson_batch(
        device,
        all_effective_gate,
        all_channel_voltage,
        poisson_nodes=poisson_nodes,
        use_gpu=use_gpu,
    )
    del surface_potential

    request_values = {}
    offset = 0
    for name, gate_bias, channel_voltage, shape in requests:
        size = len(gate_bias)
        request_values[name] = (
            inversion_charge[offset : offset + size].reshape(shape),
            gate_charge[offset : offset + size].reshape(shape),
        )
        offset += size

    mobility = device.electron_mobility * 1e-4
    width_m = device.effective_width_um * 1e-6
    length_m = device.length_um * 1e-6

    def current_from_charge(charge, drain_voltage):
        return (
            mobility
            * width_m
            / length_m
            * np.sum(
                (charge[..., 1:] + charge[..., :-1]) * 0.5,
                axis=-1,
            )
            * (drain_voltage / (channel_points - 1))
        )

    linear_charge, _ = request_values["transfer_linear"]
    saturation_charge, _ = request_values["transfer_saturation"]
    linear_current = current_from_charge(linear_charge, drain_linear_v)
    saturation_current = current_from_charge(saturation_charge, drain_saturation_v)
    _, gate_charge_grid = request_values["cv"]
    gate_charge_density = gate_charge_grid[:, 0]
    gate_area = device.effective_width_um * device.length_um * 1e-12
    cgg = np.gradient(gate_charge_density, vg_values) * gate_area
    cgg = np.maximum(cgg, 0.0)

    curves = [
        GeneratedCurve(
            kind="idvg",
            x=vg_values.tolist(),
            y=linear_current.tolist(),
            label=f"PDE Id–Vg, Vd={drain_linear_v:g} V",
            columns={
                "gate_voltage": vg_values.tolist(),
                "drain_current": linear_current.tolist(),
                "drain_voltage": [drain_linear_v] * gate_points,
            },
        ),
        GeneratedCurve(
            kind="idvg",
            x=vg_values.tolist(),
            y=saturation_current.tolist(),
            label=f"PDE Id–Vg, Vd={drain_saturation_v:g} V",
            columns={
                "gate_voltage": vg_values.tolist(),
                "drain_current": saturation_current.tolist(),
                "drain_voltage": [drain_saturation_v] * gate_points,
            },
        ),
        GeneratedCurve(
            kind="cv",
            x=vg_values.tolist(),
            y=cgg.tolist(),
            label="PDE Cgg–Vg",
            columns={
                "gate_voltage": vg_values.tolist(),
                "capacitance": cgg.tolist(),
            },
        ),
    ]
    for curve_index, gate_bias in enumerate(output_gate_biases):
        charge, _ = request_values[f"output_{curve_index}"]
        current = current_from_charge(charge, drain_values)
        curves.append(
            GeneratedCurve(
                kind="idvd",
                x=drain_values.tolist(),
                y=current.tolist(),
                label=f"PDE Id–Vd, Vg={gate_bias:g} V",
                columns={
                    "drain_voltage": drain_values.tolist(),
                    "drain_current": current.tolist(),
                    "gate_voltage": [float(gate_bias)] * drain_points,
                },
            )
        )

    diagnostics = {
        "VG_MIN_V": float(vg_values[0]),
        "VG_MAX_V": float(vg_values[-1]),
        "VDS_LINEAR_V": drain_linear_v,
        "VDS_SATURATION_V": drain_saturation_v,
        "POISSON_NODES": float(poisson_nodes),
        "GATE_POINTS": float(gate_points),
        "DRAIN_POINTS": float(drain_points),
        "MOBILITY_M2_VS": mobility,
        "FLAT_BAND_ASSUMPTION_V": 0.0,
    }
    return PDECharacteristicResult(
        curves=curves,
        backend=backend,
        diagnostics=diagnostics,
        model_description=(
            "1D nonlinear Poisson (Boltzmann statistics, exponent limited to [-30, 50]) "
            "+ gradual-channel "
            "drift-diffusion; zero flat-band offset, uniform mobility, neutral-body "
            "boundary. No 2D/3D TCAD, impact ionization, or quantum confinement."
        ),
    )


def _solve_poisson_batch(
    device,
    gate_voltage,
    channel_voltage,
    *,
    poisson_nodes: int,
    use_gpu: bool,
):
    import numpy as np

    if device.device_type == "fd_soi":
        active_depth = device.geometry["body_thickness_nm"] * 0.5e-9
    elif device.device_type == "finfet":
        active_depth = device.geometry["fin_width_nm"] * 0.5e-9
    elif device.device_type == "gaa_nanosheet":
        active_depth = device.geometry["sheet_thickness_nm"] * 0.5e-9
    elif device.device_type == "gaa_nanowire":
        active_depth = device.geometry["wire_diameter_nm"] * 0.5e-9
    else:
        active_depth = 80e-9
    depth = max(active_depth, 2e-9)
    dx = depth / (poisson_nodes - 1)
    tox = device.tox_nm * 1e-9
    cox = EPSILON_0 * device.epsilon_r / tox
    epsilon_s = EPSILON_0 * device.epsilon_r
    doping = device.doping * 1e6
    intrinsic_density = 1.45e16
    electron_density_bulk = intrinsic_density**2 / doping
    thermal_voltage = BOLTZMANN * device.temperature_k / ELECTRON_CHARGE

    backend = "CPU (NumPy vectorized damped Newton)"
    if use_gpu:
        try:
            import torch

            if torch.cuda.is_available():
                required = len(gate_voltage) * poisson_nodes * 8 * 12
                available, total = torch.cuda.mem_get_info()
                if required < available * 0.7:
                    try:
                        potential, charge, gate_charge = _poisson_batch_torch(
                            torch,
                            np,
                            gate_voltage,
                            dx,
                            cox,
                            epsilon_s,
                            doping,
                            electron_density_bulk,
                            thermal_voltage,
                            poisson_nodes,
                        )
                        return (
                            potential,
                            charge,
                            gate_charge,
                            f"GPU FP64 (PyTorch CUDA: {torch.cuda.get_device_name(0)})",
                        )
                    except RuntimeError as error:
                        torch.cuda.empty_cache()
                        backend += (
                            f" (CUDA error: {type(error).__name__}; CPU fallback)"
                        )
                else:
                    backend += (
                        f" (CUDA пропущена: оценка {required / 1e9:.3g} ГБ, "
                        f"свободно {available / 1e9:.3g} из {total / 1e9:.3g} ГБ)"
                    )
            else:
                backend += " (CUDA недоступна; CPU fallback)"
        except ImportError:
            backend += " (PyTorch CUDA недоступен)"
    potential, charge, gate_charge = _poisson_batch_numpy(
        np,
        gate_voltage,
        dx,
        cox,
        epsilon_s,
        doping,
        electron_density_bulk,
        thermal_voltage,
        poisson_nodes,
    )
    return potential, charge, gate_charge, backend


def _poisson_batch_numpy(
    np, gate_voltage, dx, cox, epsilon_s, doping, electron_density_bulk, thermal_voltage, node_count
):
    gate = np.asarray(gate_voltage, dtype=np.float64)
    count = gate.size
    unknown_count = node_count - 1
    potential = np.zeros((count, unknown_count), dtype=np.float64)
    potential[:, 0] = np.clip(gate, -0.4, 0.8)
    potential[:, 1:] = potential[:, :1] * np.linspace(
        1 - 1 / unknown_count, 0, unknown_count - 1
    )
    surface_coefficient = epsilon_s / dx
    laplace_coefficient = epsilon_s / dx**2
    lower = np.full(unknown_count - 1, -laplace_coefficient)
    upper = np.full(unknown_count - 1, -laplace_coefficient)
    lower[0] = 0.0
    upper[0] = -surface_coefficient
    converged = False

    for _ in range(100):
        electron = electron_density_bulk * np.exp(
            np.clip(potential / thermal_voltage, -30, 50)
        )
        hole = doping * np.exp(np.clip(-potential / thermal_voltage, -50, 30))
        residual = np.empty_like(potential)
        residual[:, 0] = (
            surface_coefficient * (potential[:, 0] - potential[:, 1])
            - cox * (gate - potential[:, 0])
        )
        complete = np.concatenate(
            (potential, np.zeros((count, 1), dtype=np.float64)), axis=1
        )
        residual[:, 1:] = (
            -laplace_coefficient
            * (potential[:, :-1] - 2 * potential[:, 1:] + complete[:, 2:])
            - ELECTRON_CHARGE * (hole[:, 1:] - electron[:, 1:] - doping)
        )
        diagonal = np.empty_like(potential)
        diagonal[:, 0] = surface_coefficient + cox
        diagonal[:, 1:] = (
            2 * laplace_coefficient
            + ELECTRON_CHARGE
            * (hole[:, 1:] + electron[:, 1:])
            / thermal_voltage
        )
        delta = _tridiagonal_solve_numpy(lower, diagonal, upper, -residual)
        delta = np.clip(delta, -0.2, 0.2)
        potential += delta
        boundary_scale = np.maximum(
            cox * np.maximum(np.abs(gate), thermal_voltage),
            epsilon_s * thermal_voltage / (dx * (node_count - 1)) ** 2,
        )
        interior_scale = max(
            ELECTRON_CHARGE * doping,
            epsilon_s * thermal_voltage / (dx * (node_count - 1)) ** 2,
        )
        normalized_residual = np.abs(residual)
        normalized_residual[:, 0] /= boundary_scale
        normalized_residual[:, 1:] /= interior_scale
        if (
            float(np.max(np.abs(delta))) < 1e-9
            and float(np.max(normalized_residual)) < 1e-8
        ):
            converged = True
            break
    if not converged:
        raise RuntimeError("The nonlinear Poisson solver did not converge (CPU).")

    complete_potential = np.concatenate(
        (potential, np.zeros((count, 1), dtype=np.float64)), axis=1
    )
    electron = electron_density_bulk * np.exp(
        np.clip(complete_potential / thermal_voltage, -30, 50)
    )
    inversion = ELECTRON_CHARGE * np.sum(
        np.maximum(electron - electron_density_bulk, 0.0), axis=1
    ) * dx
    gate_charge = cox * (gate - potential[:, 0])
    return potential[:, 0], inversion, gate_charge


def _tridiagonal_solve_numpy(lower, diagonal, upper, rhs):
    import numpy as np

    count, width = rhs.shape
    modified_upper = np.zeros((count, width), dtype=np.float64)
    modified_rhs = np.zeros((count, width), dtype=np.float64)
    pivot = diagonal[:, 0]
    modified_upper[:, 0] = upper[0] / pivot
    modified_rhs[:, 0] = rhs[:, 0] / pivot
    for index in range(1, width):
        pivot = diagonal[:, index] - lower[index - 1] * modified_upper[:, index - 1]
        if np.any(np.abs(pivot) < 1e-30):
            raise RuntimeError("Poisson Newton matrix has a singular pivot.")
        if index < width - 1:
            modified_upper[:, index] = upper[index] / pivot
        modified_rhs[:, index] = (
            rhs[:, index] - lower[index - 1] * modified_rhs[:, index - 1]
        ) / pivot
    result = np.empty_like(rhs)
    result[:, -1] = modified_rhs[:, -1]
    for index in range(width - 2, -1, -1):
        result[:, index] = (
            modified_rhs[:, index] - modified_upper[:, index] * result[:, index + 1]
        )
    return result


def _poisson_batch_torch(
    torch, np, gate_voltage, dx, cox, epsilon_s, doping, electron_density_bulk, thermal_voltage, node_count
):
    gate = torch.as_tensor(gate_voltage, dtype=torch.float64, device="cuda")
    count = gate.numel()
    unknown_count = node_count - 1
    potential = torch.zeros(
        (count, unknown_count), dtype=torch.float64, device="cuda"
    )
    potential[:, 0] = torch.clamp(gate, -0.4, 0.8)
    potential[:, 1:] = potential[:, :1] * torch.linspace(
        1 - 1 / unknown_count,
        0,
        unknown_count - 1,
        dtype=torch.float64,
        device="cuda",
    )
    surface_coefficient = epsilon_s / dx
    laplace_coefficient = epsilon_s / dx**2
    lower = torch.full(
        (unknown_count - 1,), -laplace_coefficient, dtype=torch.float64, device="cuda"
    )
    upper = lower.clone()
    lower[0] = 0.0
    upper[0] = -surface_coefficient
    converged = False
    for _ in range(100):
        electron = electron_density_bulk * torch.exp(
            torch.clamp(potential / thermal_voltage, -30, 50)
        )
        hole = doping * torch.exp(torch.clamp(-potential / thermal_voltage, -50, 30))
        residual = torch.empty_like(potential)
        residual[:, 0] = (
            surface_coefficient * (potential[:, 0] - potential[:, 1])
            - cox * (gate - potential[:, 0])
        )
        complete = torch.cat(
            (potential, torch.zeros((count, 1), dtype=torch.float64, device="cuda")),
            dim=1,
        )
        residual[:, 1:] = (
            -laplace_coefficient
            * (potential[:, :-1] - 2 * potential[:, 1:] + complete[:, 2:])
            - ELECTRON_CHARGE * (hole[:, 1:] - electron[:, 1:] - doping)
        )
        diagonal = torch.empty_like(potential)
        diagonal[:, 0] = surface_coefficient + cox
        diagonal[:, 1:] = (
            2 * laplace_coefficient
            + ELECTRON_CHARGE
            * (hole[:, 1:] + electron[:, 1:])
            / thermal_voltage
        )
        delta = _tridiagonal_solve_torch(lower, diagonal, upper, -residual)
        delta = torch.clamp(delta, -0.2, 0.2)
        potential = potential + delta
        boundary_scale = torch.maximum(
            cox * torch.maximum(torch.abs(gate), torch.full_like(gate, thermal_voltage)),
            torch.full_like(
                gate,
                epsilon_s * thermal_voltage / (dx * (node_count - 1)) ** 2,
            ),
        )
        interior_scale = max(
            ELECTRON_CHARGE * doping,
            epsilon_s * thermal_voltage / (dx * (node_count - 1)) ** 2,
        )
        normalized_residual = torch.abs(residual)
        normalized_residual[:, 0] /= boundary_scale
        normalized_residual[:, 1:] /= interior_scale
        if (
            torch.max(torch.abs(delta)).item() < 1e-9
            and torch.max(normalized_residual).item() < 1e-8
        ):
            converged = True
            break
    if not converged:
        raise RuntimeError("The nonlinear Poisson solver did not converge (CUDA).")
    complete = torch.cat(
        (potential, torch.zeros((count, 1), dtype=torch.float64, device="cuda")),
        dim=1,
    )
    electron = electron_density_bulk * torch.exp(
        torch.clamp(complete / thermal_voltage, -30, 50)
    )
    inversion = ELECTRON_CHARGE * torch.sum(
        torch.clamp(electron - electron_density_bulk, min=0.0), dim=1
    ) * dx
    gate_charge = cox * (gate - potential[:, 0])
    torch.cuda.synchronize()
    return (
        potential[:, 0].cpu().numpy(),
        inversion.cpu().numpy(),
        gate_charge.cpu().numpy(),
    )


def _tridiagonal_solve_torch(lower, diagonal, upper, rhs):
    import torch

    count, width = rhs.shape
    modified_upper = [torch.zeros_like(rhs[:, 0])]
    modified_rhs = [rhs[:, 0] / diagonal[:, 0]]
    modified_upper[0] = upper[0] / diagonal[:, 0]
    for index in range(1, width):
        pivot = diagonal[:, index] - lower[index - 1] * modified_upper[index - 1]
        if torch.any(torch.abs(pivot) < 1e-30):
            raise RuntimeError("CUDA Poisson Newton matrix has a singular pivot.")
        modified_upper.append(
            upper[index] / pivot if index < width - 1 else torch.zeros_like(pivot)
        )
        modified_rhs.append(
            (rhs[:, index] - lower[index - 1] * modified_rhs[index - 1]) / pivot
        )
    solution = [None] * width
    solution[-1] = modified_rhs[-1]
    for index in range(width - 2, -1, -1):
        solution[index] = (
            modified_rhs[index] - modified_upper[index] * solution[index + 1]
        )
    return torch.stack(solution, dim=1)


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
    source_index = np.unravel_index(source_node, shape)
    spacing = np.array(
        [
            (axis[min(index + 1, len(axis) - 1)] - axis[max(index - 1, 0)]) / 2
            for axis, index in zip(axes, source_index)
        ]
    )
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
    boundary_grid = np.zeros(shape, dtype=bool)
    boundary_grid[0, :, :] = True
    boundary_grid[-1, :, :] = True
    boundary = boundary_grid.ravel()
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
    gradients = np.gradient(potential_grid, x, y, z, edge_order=1)
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
    substrate_m = substrate_thickness_um * 1e-6
    gate_start_m = substrate_m + active
    axes = (
        _adaptive_axis(dimensions[0], nx, (0.25 * dimensions[0], 0.5 * dimensions[0], 0.75 * dimensions[0])),
        _adaptive_axis(dimensions[1], ny, (0.5 * dimensions[1],)),
        _adaptive_axis(
            dimensions[2],
            nz,
            (substrate_m, gate_start_m, gate_start_m + device.tox_nm * 1e-9),
        ),
    )
    xx, yy, zz = np.meshgrid(*axes, indexing="ij")
    coordinates = np.column_stack((xx.ravel(), yy.ravel(), zz.ravel()))
    node_grid = np.arange(nx * ny * nz).reshape(grid_shape)
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
    corner_nodes = np.stack(
        [
            node_grid[
                di : di + nx - 1,
                dj : dj + ny - 1,
                dk : dk + nz - 1,
            ]
            for di, dj, dk in corners
        ],
        axis=-1,
    ).reshape(-1, 8)
    tetrahedra = corner_nodes[:, np.asarray(local_tetrahedra)].reshape(-1, 4)
    return coordinates, tetrahedra, grid_shape, axes


def _adaptive_axis(size, count, refinement_points):
    """Distribute structured-grid nodes more densely around material interfaces."""
    import numpy as np

    sample_count = max(2049, count * 32)
    sample_axis = np.linspace(0.0, size, sample_count)
    spacing = size / (count - 1)
    density = np.ones(sample_count, dtype=np.float64)
    sigma = max(spacing * 1.5, size * 1e-9)
    for point in refinement_points:
        if 0 < point < size:
            density += 5.0 * np.exp(-0.5 * ((sample_axis - point) / sigma) ** 2)
    increments = np.diff(sample_axis)
    cumulative = np.concatenate(
        ([0.0], np.cumsum((density[:-1] + density[1:]) * 0.5 * increments))
    )
    targets = np.linspace(0.0, cumulative[-1], count)
    axis = np.interp(targets, cumulative, sample_axis)
    axis[0], axis[-1] = 0.0, size
    return axis


def _assemble_tetrahedral_stiffness(coordinates, tetrahedra, conductivity):
    import numpy as np
    from scipy.sparse import coo_matrix

    nodes_count = len(coordinates)
    matrix = None
    chunk_size = 20_000
    for start in range(0, len(tetrahedra), chunk_size):
        elements = tetrahedra[start : start + chunk_size]
        points = coordinates[elements]
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
        coo = coo_matrix(
            (
                local_matrices.ravel(),
                (
                    np.repeat(elements, 4, axis=1).ravel(),
                    np.tile(elements, (1, 4)).ravel(),
                ),
            ),
            shape=(nodes_count, nodes_count),
        ).tocsr()
        matrix = coo if matrix is None else matrix + coo
    if matrix is None:
        raise ValueError("The tetrahedral mesh does not contain any elements.")
    return matrix


def _solve_dirichlet(stiffness, rhs, boundary, boundary_values, free, use_gpu):
    import numpy as np
    from scipy.sparse.linalg import LinearOperator, cg, spsolve

    matrix = stiffness[free][:, free].tocsr()
    reduced_rhs = rhs[free] - stiffness[free][:, boundary] @ boundary_values[boundary]
    result = np.array(boundary_values, dtype=np.float64, copy=True)
    backend = "CPU (SciPy sparse direct solver)"
    gpu_note = ""
    if use_gpu:
        try:
            import torch

            if torch.cuda.is_available():
                required_bytes = _estimate_cuda_memory_bytes(matrix)
                available_bytes, total_bytes = torch.cuda.mem_get_info()
                if required_bytes > available_bytes * 0.8:
                    gpu_note = (
                        f" (CUDA пропущена: оценка VRAM "
                        f"{required_bytes / 1e9:.1f} ГБ, свободно "
                        f"{available_bytes / 1e9:.1f} из {total_bytes / 1e9:.1f} ГБ)"
                    )
                else:
                    try:
                        coo = matrix.tocoo()
                        indices = torch.as_tensor(
                            np.vstack((coo.row, coo.col)),
                            dtype=torch.long,
                            device="cuda",
                        )
                        values = torch.as_tensor(
                            coo.data, dtype=torch.float64, device="cuda"
                        )
                        operator = torch.sparse_coo_tensor(
                            indices, values, coo.shape, device="cuda"
                        ).coalesce()
                        vector = torch.as_tensor(
                            reduced_rhs, dtype=torch.float64, device="cuda"
                        )
                        solution = _conjugate_gradient(operator, vector)
                        result[free] = solution.cpu().numpy()
                        backend = (
                            f"GPU FP64 (PyTorch CUDA: "
                            f"{torch.cuda.get_device_name(0)})"
                        )
                        return result, backend
                    except torch.cuda.OutOfMemoryError:
                        torch.cuda.empty_cache()
                        gpu_note = " (CUDA OOM; расчёт переключён на CPU)"
        except ImportError:
            pass
    if matrix.shape[0] <= 50_000:
        result[free] = spsolve(matrix, reduced_rhs)
        backend = "CPU (SciPy sparse direct solver)" + gpu_note
    else:
        diagonal = matrix.diagonal()
        if np.any(diagonal <= 0):
            raise RuntimeError("CPU FEM matrix has a non-positive diagonal.")
        preconditioner = LinearOperator(
            matrix.shape, matvec=lambda vector: vector / diagonal
        )
        free_solution, status = cg(
            matrix,
            reduced_rhs,
            M=preconditioner,
            rtol=1e-8,
            atol=0,
            maxiter=10_000,
        )
        if status != 0:
            raise RuntimeError(
                f"CPU FEM conjugate-gradient solver failed to converge (status {status})."
            )
        result[free] = free_solution
        backend = "CPU (SciPy preconditioned conjugate gradient)" + gpu_note
    return result, backend


def _estimate_cuda_memory_bytes(matrix) -> int:
    """Conservative estimate for COO conversion, coalescing, and CG work vectors."""
    return int(matrix.nnz * 48 + matrix.shape[0] * 8 * 12)


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
        not isinstance(size, int) or size < 4 for size in grid_shape
    ):
        raise ValueError("Each 3D FEM grid dimension must be an integer of at least 4.")
    node_count = math.prod(grid_shape)
    if node_count > MAX_MESH_NODES:
        raise ValueError(
            f"Mesh contains {node_count:,} nodes; the supported maximum is "
            f"{MAX_MESH_NODES:,} total nodes."
        )
    estimate = estimate_mesh_memory_bytes(grid_shape)
    available = _available_memory_bytes()
    if estimate > available * 0.7:
        raise ValueError(
            f"This mesh needs an estimated {estimate / 1e9:.1f} GB peak RAM, "
            f"but only {available / 1e9:.1f} GB is currently available. "
            "Reduce mesh resolution or close other applications."
        )


def estimate_tetrahedron_count(grid_shape: tuple[int, int, int]) -> int:
    return 6 * math.prod(size - 1 for size in grid_shape)


def estimate_mesh_memory_bytes(grid_shape: tuple[int, int, int]) -> int:
    return math.prod(grid_shape) * ESTIMATED_PEAK_BYTES_PER_NODE


def _available_memory_bytes() -> int:
    import os

    if os.name == "nt":
        import ctypes

        class MemoryStatus(ctypes.Structure):
            _fields_ = [
                ("length", ctypes.c_ulong),
                ("memory_load", ctypes.c_ulong),
                ("total_phys", ctypes.c_ulonglong),
                ("avail_phys", ctypes.c_ulonglong),
                ("total_page", ctypes.c_ulonglong),
                ("avail_page", ctypes.c_ulonglong),
                ("total_virtual", ctypes.c_ulonglong),
                ("avail_virtual", ctypes.c_ulonglong),
                ("avail_extended_virtual", ctypes.c_ulonglong),
            ]

        status = MemoryStatus()
        status.length = ctypes.sizeof(MemoryStatus)
        if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
            return int(status.avail_phys)
    else:
        try:
            return int(os.sysconf("SC_AVPHYS_PAGES") * os.sysconf("SC_PAGE_SIZE"))
        except (ValueError, OSError, AttributeError):
            pass
    return 1_000_000_000
