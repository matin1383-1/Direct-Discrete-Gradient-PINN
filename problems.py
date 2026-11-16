"""PDE definitions, manufactured data, geometries, and RBF-FD operator setup."""
from __future__ import annotations

from dataclasses import dataclass, field
from math import pi
from time import perf_counter
from typing import Callable

import numpy as np
import torch

from rbf_fd import OperatorDiagnostics, build_operators, scipy_to_torch_sparse


PROBLEM_NAMES = ("poisson1d", "poisson2d", "burgers", "allen_cahn", "heat")
HIDDEN_LAYERS = {
    "poisson1d": [20, 20, 20],
    "poisson2d": [20, 20, 20],
    "burgers": [32, 32, 32],
    "allen_cahn": [64, 64, 64, 64],
    "heat": [50, 50, 50, 50],
}
LAMBDA_DEFAULTS = {
    "poisson1d": 0.1,
    "poisson2d": 0.01,
    "burgers": 1.0e-7,
    "allen_cahn": 1.0e-5,
    "heat": 0.01,
}
LEARNING_RATES = {
    "poisson1d": {"direct": 1e-3, "adaptive": 3e-4, "fixed": 1e-3},
    "poisson2d": {"direct": 3e-4, "adaptive": 3e-4, "fixed": 1e-3},
    "burgers": {"direct": 3e-4, "adaptive": 3e-4, "fixed": 3e-4},
    "allen_cahn": {"direct": 1e-3, "adaptive": 1e-3, "fixed": 1e-3},
    "heat": {"direct": 3e-4, "adaptive": 1e-3, "fixed": 3e-4},
}
CORRECTION_COUNTS = {"poisson1d": 8, "poisson2d": 40, "burgers": 40, "allen_cahn": 40, "heat": 40}


def poisson1d_exact(x: torch.Tensor) -> torch.Tensor:
    result = x
    for k in (1, 2, 3, 4, 8):
        result = result + torch.sin(k * x) / k
    return result


def poisson1d_b(x: torch.Tensor) -> torch.Tensor:
    return sum((-k * torch.sin(k * x) for k in (1, 2, 3, 4, 8)), torch.zeros_like(x))


def poisson1d_bx(x: torch.Tensor) -> torch.Tensor:
    return sum((-k * k * torch.cos(k * x) for k in (1, 2, 3, 4, 8)), torch.zeros_like(x))


def poisson2d_exact(xy: torch.Tensor) -> torch.Tensor:
    x, y = xy[:, 0:1], xy[:, 1:2]
    return (16.0 * x * y * (1.0 - x) * (1.0 - y)) ** 10


def _scalar_derivative(value: torch.Tensor, coords: torch.Tensor, create_graph: bool = True) -> torch.Tensor:
    return torch.autograd.grad(
        value, coords, torch.ones_like(value), create_graph=create_graph, retain_graph=True
    )[0]


def poisson2d_b(xy: torch.Tensor, *, create_graph: bool = True) -> torch.Tensor:
    """Manufactured Laplacian, differentiated from the stated exact solution."""
    u = poisson2d_exact(xy)
    grad = _scalar_derivative(u, xy, True)
    uxx = _scalar_derivative(grad[:, 0:1], xy, True)[:, 0:1]
    uyy = _scalar_derivative(grad[:, 1:2], xy, True)[:, 1:2]
    result = uxx + uyy
    return result if create_graph else result.detach()


def heat_exact(xyt: torch.Tensor) -> torch.Tensor:
    x, y, t = xyt[:, 0:1], xyt[:, 1:2], xyt[:, 2:3]
    return 1.0 + torch.sin(pi * x) * torch.cos(pi * y) * torch.sin(pi * t)


def heat_forcing(xyt: torch.Tensor) -> torch.Tensor:
    x, y, t = xyt[:, 0:1], xyt[:, 1:2], xyt[:, 2:3]
    return torch.sin(pi * x) * torch.cos(pi * y) * (
        pi * torch.cos(pi * t) + 2.0 * pi**2 * torch.sin(pi * t)
    )


def disk_cloud(n_rings: int = 10, ghost_offsets: tuple[float, ...] = (0.04, 0.08)):
    """Deterministic quasi-uniform polar disk cloud and outward ghost rings."""
    physical = [np.array([[0.0, 0.0]])]
    ring_slices: list[np.ndarray] = []
    cursor = 1
    for ring in range(1, n_rings + 1):
        count = 8 * ring
        theta = 2.0 * np.pi * np.arange(count) / count + (ring % 2) * np.pi / count
        radius = ring / n_rings
        points = np.column_stack([radius * np.cos(theta), radius * np.sin(theta)])
        physical.append(points)
        ring_slices.append(np.arange(cursor, cursor + count, dtype=np.int64))
        cursor += count
    physical_nodes = np.vstack(physical)
    boundary = ring_slices[-1]
    boundary_xy = physical_nodes[boundary]
    ghosts = [boundary_xy * (1.0 + offset) for offset in ghost_offsets]
    support = np.vstack([physical_nodes, *ghosts])
    interior = np.setdiff1d(np.arange(len(physical_nodes)), boundary)
    normals = boundary_xy / np.linalg.norm(boundary_xy, axis=1, keepdims=True)
    return physical_nodes, support, interior, boundary, normals


@dataclass
class Problem:
    name: str
    spatial_dim: int
    support_nodes: np.ndarray
    target_nodes: np.ndarray
    pde_spatial_indices: np.ndarray
    times: np.ndarray | None
    hidden_layers: list[int]
    operators_scipy: dict[str, object]
    diagnostics: dict[str, OperatorDiagnostics]
    operator_seconds: float
    boundary_indices: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=np.int64))
    boundary_normals: np.ndarray | None = None
    torch_operators: dict[str, torch.Tensor] = field(default_factory=dict)
    device: torch.device | None = None
    dtype: torch.dtype | None = None

    @property
    def input_dim(self) -> int:
        return self.spatial_dim + (1 if self.times is not None else 0)

    @property
    def positive_times(self) -> np.ndarray:
        if self.times is None:
            raise ValueError("steady problem has no time levels")
        return self.times[self.times > 0.0]

    @property
    def candidate_count(self) -> int:
        multiplier = 1 if self.times is None else len(self.positive_times)
        return int(len(self.pde_spatial_indices) * multiplier)

    def to(self, device: torch.device | str, dtype: torch.dtype):
        self.device = torch.device(device)
        self.dtype = dtype
        # Forcing caches are device/dtype-specific and must never survive a
        # migration of a reusable Problem instance.
        for cache_name in ("_poisson2d_constants", "_heat_forcing_constants"):
            if hasattr(self, cache_name):
                delattr(self, cache_name)
        self.torch_operators = {
            name: scipy_to_torch_sparse(matrix, device=self.device, dtype=dtype)
            for name, matrix in self.operators_scipy.items()
        }
        return self

    def tensor(self, array, *, requires_grad: bool = False) -> torch.Tensor:
        return torch.tensor(array, device=self.device, dtype=self.dtype, requires_grad=requires_grad)

    def candidate_coordinates(self, candidate_positions: np.ndarray | torch.Tensor) -> torch.Tensor:
        if isinstance(candidate_positions, torch.Tensor):
            positions = candidate_positions.detach().cpu().numpy().astype(np.int64, copy=False)
        else:
            positions = np.asarray(candidate_positions, dtype=np.int64)
        if self.times is None:
            return self.tensor(self.target_nodes[self.pde_spatial_indices[positions]])
        n_space = len(self.pde_spatial_indices)
        time_indices = positions // n_space
        local_space = positions % n_space
        xy = self.target_nodes[self.pde_spatial_indices[local_space]]
        t = self.positive_times[time_indices, None]
        return self.tensor(np.column_stack([xy, t]))


def _operator_names(dim: int):
    if dim == 1:
        return {(1,): "dx", (2,): "dxx", (3,): "dxxx"}
    return {
        (1, 0): "dx",
        (0, 1): "dy",
        (2, 0): "dxx",
        (0, 2): "dyy",
        (3, 0): "dxxx",
        (1, 2): "dxyy",
        (2, 1): "dyxx",
        (0, 3): "dyyy",
    }


def _build_problem_operators(support: np.ndarray, targets: np.ndarray, dim: int, method=None):
    started = perf_counter()
    name_map = _operator_names(dim)
    if method == "gpinn":
        return {}, {}, 0.0
    dt_derivatives = [(1,), (2,)] if dim == 1 else [(1, 0), (0, 1), (2, 0), (0, 2)]
    if method in ("dt3", "dt4", "dt5"):
        order = int(method[-1])
        matrices, diagnostics = build_operators(
            support, dt_derivatives, targets=targets, approximation_order=order
        )
        suffix = "" if order == 3 else f"_p{order}"
        return ({name_map[k] + suffix: v for k, v in matrices.items()},
                {name_map[k] + suffix: v for k, v in diagnostics.items()},
                perf_counter() - started)
    matrices3, diagnostics3 = build_operators(
        support, name_map.keys(), targets=targets, approximation_order=3
    )
    matrices = {name_map[key]: value for key, value in matrices3.items()}
    diagnostics = {name_map[key]: value for key, value in diagnostics3.items()}
    # DT-PINN requires order-matched first and second spatial derivatives.
    # The direct/hybrid methods continue to use the unsuffixed p=3 matrices.
    dt_derivatives = [(1,), (2,)] if dim == 1 else [(1, 0), (0, 1), (2, 0), (0, 2)]
    orders = (4, 5) if method is None else ((4,) if method == "adaptive" else ())
    for order in orders:
        derivatives = dt_derivatives if method is None else [a for a in dt_derivatives if sum(a) == 2]
        order_matrices, order_diagnostics = build_operators(
            support, derivatives, targets=targets, approximation_order=order
        )
        for key, matrix in order_matrices.items():
            name = f"{name_map[key]}_p{order}"
            matrices[name] = matrix
            diagnostics[name] = order_diagnostics[key]
    return matrices, diagnostics, perf_counter() - started


def make_problem(name: str, *, device="cpu", dtype=torch.float32, method=None) -> Problem:
    if method not in (None, "gpinn", "dt3", "dt4", "dt5", "direct", "adaptive", "fixed"):
        raise ValueError(f"unknown method {method!r}")
    if name not in PROBLEM_NAMES:
        raise ValueError(f"unknown problem {name!r}; choose from {PROBLEM_NAMES}")
    if name == "poisson1d":
        support = np.linspace(0.0, np.pi, 129)[:, None]
        targets = support.copy()
        pde_indices = np.arange(1, len(targets) - 1, dtype=np.int64)
        times = None
        boundary = np.array([0, len(targets) - 1], dtype=np.int64)
        normals = None
    elif name == "poisson2d":
        axis = np.linspace(0.0, 1.0, 21)
        xx, yy = np.meshgrid(axis, axis, indexing="ij")
        support = np.column_stack([xx.ravel(), yy.ravel()])
        targets = support.copy()
        mask = np.all((targets > 0.0) & (targets < 1.0), axis=1)
        pde_indices = np.flatnonzero(mask)
        times = None
        boundary = np.flatnonzero(~mask)
        normals = None
    elif name in ("burgers", "allen_cahn"):
        support = np.linspace(-1.0, 1.0, 65)[:, None]
        targets = support.copy()
        pde_indices = np.arange(1, 64, dtype=np.int64)
        times = np.linspace(0.0, 1.0, 25)
        boundary = np.array([0, 64], dtype=np.int64)
        normals = None
    else:
        targets, support, pde_indices, boundary, normals = disk_cloud()
        times = np.linspace(0.0, 1.0, 25)
    spatial_dim = support.shape[1]
    matrices, diagnostics, elapsed = _build_problem_operators(support, targets, spatial_dim, method)
    return Problem(
        name=name,
        spatial_dim=spatial_dim,
        support_nodes=support,
        target_nodes=targets,
        pde_spatial_indices=pde_indices,
        times=times,
        hidden_layers=HIDDEN_LAYERS[name],
        operators_scipy=matrices,
        diagnostics=diagnostics,
        operator_seconds=elapsed,
        boundary_indices=boundary,
        boundary_normals=normals,
    ).to(device, dtype)


def exact_solution(name: str) -> Callable[[torch.Tensor], torch.Tensor] | None:
    return {"poisson1d": poisson1d_exact, "poisson2d": poisson2d_exact, "heat": heat_exact}.get(name)


def evaluation_coordinates(name: str) -> np.ndarray | None:
    """Independent midpoint samples; never used in losses or point selection."""
    if name == "poisson1d":
        return ((np.arange(2048) + 0.5) * np.pi / 2048)[:, None]
    if name == "poisson2d":
        axis = (np.arange(80) + 0.5) / 80
        x, y = np.meshgrid(axis, axis, indexing="ij")
        return np.column_stack([x.ravel(), y.ravel()])
    if name == "heat":
        # Equal-area radial sampling with a uniform number of angles per ring.
        radius = np.sqrt((np.arange(24) + 0.5) / 24)
        theta = 2 * np.pi * (np.arange(64) + 0.5) / 64
        r, angle = np.meshgrid(radius, theta, indexing="ij")
        xy = np.column_stack([(r * np.cos(angle)).ravel(), (r * np.sin(angle)).ravel()])
        return np.vstack([np.column_stack([xy, np.full(len(xy), t)])
                          for t in (np.arange(20) + 0.5) / 20])
    return None
