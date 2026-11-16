"""Sparse polyharmonic-spline RBF-FD differentiation operators.

The implementation intentionally exposes the local augmented systems.  No
finite-difference or RBF package is used.  Coordinates are centered at each
target and divided by the farthest stencil distance before weights are built.
"""
from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from math import factorial
from time import perf_counter
from typing import Iterable, Sequence

import numpy as np
import scipy.sparse as sp
from scipy.spatial import cKDTree


MultiIndex = tuple[int, ...]


@dataclass(frozen=True)
class OperatorDiagnostics:
    derivative: MultiIndex
    approximation_order: int
    polynomial_degree: int
    phs_exponent: int
    stencil_size: int
    min_condition: float
    median_condition: float
    max_condition: float
    polynomial_reproduction_error: float
    nnz: int
    density: float
    construction_seconds: float
    max_backward_error: float
    max_local_polynomial_error: float


def monomial_powers(dim: int, degree: int) -> list[MultiIndex]:
    """Return all d-dimensional monomial powers of total degree <= degree."""
    powers = [a for a in product(range(degree + 1), repeat=dim) if sum(a) <= degree]
    return sorted(powers, key=lambda a: (sum(a), a))


def choose_phs_exponent(polynomial_degree: int) -> int:
    """Deterministic odd PHS rule: smallest odd >= q, clipped to [5, 11]."""
    exponent = polynomial_degree if polynomial_degree % 2 else polynomial_degree + 1
    return int(min(11, max(5, exponent)))


def _poly_matrix(points: np.ndarray, powers: Sequence[MultiIndex]) -> np.ndarray:
    out = np.ones((len(points), len(powers)), dtype=np.float64)
    for j, alpha in enumerate(powers):
        for axis, power in enumerate(alpha):
            if power:
                out[:, j] *= points[:, axis] ** power
    return out


def _poly_rhs_at_origin(powers: Sequence[MultiIndex], derivative: MultiIndex) -> np.ndarray:
    rhs = np.zeros(len(powers), dtype=np.float64)
    for j, alpha in enumerate(powers):
        if tuple(alpha) == tuple(derivative):
            rhs[j] = float(np.prod([factorial(a) for a in derivative]))
    return rhs


def _phs_derivative_at_target(offsets: np.ndarray, m: int, derivative: MultiIndex) -> np.ndarray:
    """Derivative in the target coordinate of ||x-x_j||^m at x=0.

    ``offsets`` are x_j in centered/scaled coordinates, hence y=x-x_j is
    -offsets at the target.  Explicit tensor formulas cover orders zero to
    three, including repeated and mixed derivatives.
    """
    order = sum(derivative)
    if order > 3:
        raise ValueError("PHS derivatives above third order are not implemented")
    axes = [i for i, count in enumerate(derivative) for _ in range(count)]
    y = -np.asarray(offsets, dtype=np.float64)
    r = np.linalg.norm(y, axis=1)
    if order == 0:
        return r**m
    safe = r > 0.0
    out = np.zeros(len(y), dtype=np.float64)
    if order == 1:
        a = axes[0]
        out[safe] = m * r[safe] ** (m - 2) * y[safe, a]
    elif order == 2:
        a, b = axes
        first = (1.0 if a == b else 0.0) * r[safe] ** (m - 2)
        second = (m - 2) * y[safe, a] * y[safe, b] * r[safe] ** (m - 4)
        out[safe] = m * (first + second)
    else:
        a, b, c = axes
        delta_terms = (
            (1.0 if a == b else 0.0) * y[safe, c]
            + (1.0 if a == c else 0.0) * y[safe, b]
            + (1.0 if b == c else 0.0) * y[safe, a]
        )
        out[safe] = m * (m - 2) * (
            delta_terms * r[safe] ** (m - 4)
            + (m - 4)
            * y[safe, a]
            * y[safe, b]
            * y[safe, c]
            * r[safe] ** (m - 6)
        )
    return out


def _exact_monomial_derivative(
    points: np.ndarray, alpha: MultiIndex, derivative: MultiIndex
) -> np.ndarray:
    value = np.ones(len(points), dtype=np.float64)
    for axis, (power, diff) in enumerate(zip(alpha, derivative)):
        if power < diff:
            return np.zeros(len(points), dtype=np.float64)
        value *= factorial(power) / factorial(power - diff)
        if power > diff:
            value *= points[:, axis] ** (power - diff)
    return value


def build_operator(
    support_nodes: np.ndarray,
    derivative: MultiIndex,
    *,
    targets: np.ndarray | None = None,
    approximation_order: int = 3,
    stencil_size: int | None = None,
    condition_limit: float = 1.0e12,
    reproduction_tolerance: float = 1.0e-7,
) -> tuple[sp.csr_matrix, OperatorDiagnostics]:
    """Construct one sparse RBF-FD derivative matrix.

    Rows correspond to ``targets`` and columns to ``support_nodes``.  A raised
    exception identifies invalid geometry instead of silently using a least-
    squares or pseudoinverse fallback.
    """
    started = perf_counter()
    nodes = np.asarray(support_nodes, dtype=np.float64)
    target_points = nodes if targets is None else np.asarray(targets, dtype=np.float64)
    if nodes.ndim != 2 or target_points.ndim != 2 or nodes.shape[1] != target_points.shape[1]:
        raise ValueError("support_nodes and targets must be 2-D arrays with the same dimension")
    if len(nodes) == 0 or len(target_points) == 0 or not np.all(np.isfinite(nodes)) or not np.all(np.isfinite(target_points)):
        raise ValueError("nodes and targets must be nonempty and finite")
    if approximation_order < 1 or int(approximation_order) != approximation_order:
        raise ValueError("approximation_order must be a positive integer")
    if any(v < 0 or int(v) != v for v in derivative):
        raise ValueError("derivative entries must be nonnegative integers")
    if not np.isfinite(condition_limit) or condition_limit <= 1 or not np.isfinite(reproduction_tolerance) or reproduction_tolerance <= 0:
        raise ValueError("invalid numerical acceptance tolerances")
    dim = nodes.shape[1]
    derivative = tuple(int(v) for v in derivative)
    if len(derivative) != dim or sum(derivative) < 1 or sum(derivative) > 3:
        raise ValueError(f"invalid derivative multi-index {derivative} for dimension {dim}")
    if len(np.unique(nodes, axis=0)) != len(nodes):
        raise ValueError("support_nodes contain duplicate coordinates")

    differential_order = sum(derivative)
    polynomial_degree = approximation_order + differential_order - 1
    powers = monomial_powers(dim, polynomial_degree)
    n_poly = len(powers)
    n_stencil = int(stencil_size or (2 * n_poly + 1))
    if n_stencil < n_poly or n_stencil > len(nodes):
        raise ValueError(
            f"stencil size {n_stencil} is incompatible with {n_poly} polynomials and {len(nodes)} nodes"
        )
    phs_exponent = choose_phs_exponent(polynomial_degree)
    tree = cKDTree(nodes)
    # A Cartesian boundary stencil can contain the nominal number of points yet
    # be non-unisolvent for the full total-degree basis.  Query a deterministic
    # nearest-neighbor reserve and enlarge only such stencils below.
    reserve_size = min(len(nodes), max(n_stencil, 4 * n_poly + 1))
    _, neighbor_indices = tree.query(target_points, k=reserve_size)
    if reserve_size == 1:
        neighbor_indices = neighbor_indices[:, None]

    rows: list[int] = []
    cols: list[int] = []
    values: list[float] = []
    conditions = np.empty(len(target_points), dtype=np.float64)
    poly_rhs = _poly_rhs_at_origin(powers, derivative)
    actual_stencil_sizes: list[int] = []
    backward_errors: list[float] = []
    polynomial_errors: list[float] = []

    for row, reserve in enumerate(np.asarray(neighbor_indices, dtype=np.int64)):
        chosen_size = n_stencil
        while True:
            indices = reserve[:chosen_size]
            local = nodes[indices] - target_points[row]
            radius = float(np.max(np.linalg.norm(local, axis=1)))
            if not np.isfinite(radius) or radius <= np.finfo(np.float64).eps:
                raise ValueError(f"degenerate stencil at target row {row}: radius={radius}")
            z = local / radius
            poly = _poly_matrix(z, powers)
            if np.linalg.matrix_rank(poly) == n_poly:
                break
            chosen_size += 1
            if chosen_size > reserve_size:
                raise np.linalg.LinAlgError(
                    f"polynomially singular nearest-neighbor reserve at target row {row}"
                )
        actual_stencil_sizes.append(chosen_size)
        distances = np.linalg.norm(z[:, None, :] - z[None, :, :], axis=2)
        phi = distances**phs_exponent
        zero = np.zeros((n_poly, n_poly), dtype=np.float64)
        augmented = np.block([[phi, poly], [poly.T, zero]])
        condition = float(np.linalg.cond(augmented))
        conditions[row] = condition
        if not np.isfinite(condition) or condition > condition_limit:
            raise np.linalg.LinAlgError(
                f"ill-conditioned stencil at target row {row}: cond={condition:.3e}"
            )
        rbf_rhs = _phs_derivative_at_target(z, phs_exponent, derivative)
        rhs = np.concatenate([rbf_rhs, poly_rhs])
        try:
            solution = np.linalg.solve(augmented, rhs)
        except np.linalg.LinAlgError as exc:
            raise np.linalg.LinAlgError(f"failed stencil solve at target row {row}") from exc
        weights = solution[:chosen_size] / radius**differential_order
        if not np.all(np.isfinite(weights)):
            raise FloatingPointError(f"non-finite differentiation weights at target row {row}")
        denominator = np.linalg.norm(augmented, np.inf) * np.linalg.norm(solution, np.inf) + np.linalg.norm(rhs, np.inf)
        backward_error = np.linalg.norm(augmented @ solution - rhs, np.inf) / max(denominator, np.finfo(float).tiny)
        polynomial_error = np.linalg.norm(poly.T @ solution[:chosen_size] - poly_rhs, np.inf) / max(1.0, np.linalg.norm(poly_rhs, np.inf))
        if not np.isfinite(backward_error) or backward_error > 1e-12 or not np.isfinite(polynomial_error) or polynomial_error > reproduction_tolerance:
            raise np.linalg.LinAlgError(f"inaccurate stencil at row {row}: backward={backward_error:.3e}, polynomial={polynomial_error:.3e}")
        backward_errors.append(float(backward_error))
        polynomial_errors.append(float(polynomial_error))
        rows.extend([row] * chosen_size)
        cols.extend(indices.tolist())
        values.extend(weights.tolist())

    matrix = sp.csr_matrix((values, (rows, cols)), shape=(len(target_points), len(nodes)))
    reproduction_error = polynomial_reproduction_error(
        matrix, nodes, target_points, derivative, polynomial_degree
    )
    if not np.isfinite(reproduction_error) or reproduction_error > reproduction_tolerance:
        raise np.linalg.LinAlgError(f"operator polynomial reproduction failed: {reproduction_error:.3e}")
    elapsed = perf_counter() - started
    diagnostics = OperatorDiagnostics(
        derivative=derivative,
        approximation_order=approximation_order,
        polynomial_degree=polynomial_degree,
        phs_exponent=phs_exponent,
        stencil_size=max(actual_stencil_sizes),
        min_condition=float(np.min(conditions)),
        median_condition=float(np.median(conditions)),
        max_condition=float(np.max(conditions)),
        polynomial_reproduction_error=float(reproduction_error),
        nnz=int(matrix.nnz),
        density=float(matrix.nnz / np.prod(matrix.shape)),
        construction_seconds=elapsed,
        max_backward_error=max(backward_errors),
        max_local_polynomial_error=max(polynomial_errors),
    )
    return matrix, diagnostics


def polynomial_reproduction_error(
    matrix: sp.spmatrix,
    support_nodes: np.ndarray,
    targets: np.ndarray,
    derivative: MultiIndex,
    degree: int,
) -> float:
    """Maximum scaled infinity-norm error over the augmented monomial basis."""
    errors: list[float] = []
    for alpha in monomial_powers(support_nodes.shape[1], degree):
        sampled = _poly_matrix(support_nodes, [alpha])[:, 0]
        predicted = np.asarray(matrix @ sampled).ravel()
        exact = _exact_monomial_derivative(targets, alpha, derivative)
        scale = max(1.0, float(np.max(np.abs(exact))))
        errors.append(float(np.max(np.abs(predicted - exact)) / scale))
    return max(errors, default=0.0)


def build_operators(
    support_nodes: np.ndarray,
    derivatives: Iterable[MultiIndex],
    *,
    targets: np.ndarray | None = None,
    approximation_order: int = 3,
    condition_limit: float = 1.0e12,
) -> tuple[dict[MultiIndex, sp.csr_matrix], dict[MultiIndex, OperatorDiagnostics]]:
    matrices: dict[MultiIndex, sp.csr_matrix] = {}
    diagnostics: dict[MultiIndex, OperatorDiagnostics] = {}
    for derivative in derivatives:
        key = tuple(derivative)
        matrices[key], diagnostics[key] = build_operator(
            support_nodes,
            key,
            targets=targets,
            approximation_order=approximation_order,
            condition_limit=condition_limit,
        )
    return matrices, diagnostics


def scipy_to_torch_sparse(matrix: sp.spmatrix, *, device, dtype):
    """Convert SciPy sparse storage without breaking downstream autograd."""
    import torch

    coo = matrix.tocoo()
    indices = torch.tensor(np.vstack([coo.row, coo.col]), dtype=torch.int64, device=device)
    values = torch.tensor(coo.data, dtype=dtype, device=device)
    return torch.sparse_coo_tensor(indices, values, coo.shape, device=device, dtype=dtype).coalesce()
