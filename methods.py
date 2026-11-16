"""gPINN, DT-PINN and discrete-gradient/hybrid losses and point selectors."""
from __future__ import annotations

from dataclasses import dataclass
from math import pi

import numpy as np
import torch

from problems import heat_forcing, poisson1d_b, poisson1d_bx, poisson2d_b


METHOD_NAMES = ("gpinn", "dt3", "dt4", "dt5", "direct", "adaptive", "fixed")
DT_ORDERS = {"dt3": 3, "dt4": 4, "dt5": 5}


def _grad(value: torch.Tensor, coordinates: torch.Tensor, *, create_graph: bool = True):
    return torch.autograd.grad(
        value,
        coordinates,
        torch.ones_like(value),
        create_graph=create_graph,
        retain_graph=True,
    )[0]


def _spatial_apply(operator: torch.Tensor, values: torch.Tensor) -> torch.Tensor:
    """Apply an (n_target,n_support) sparse operator to time-major values."""
    if values.ndim == 1:
        values = values[None, :]
    return torch.sparse.mm(operator, values.T).T


def _space_time_coordinates(problem, times: np.ndarray) -> torch.Tensor:
    space = problem.tensor(problem.support_nodes)
    t = problem.tensor(times).reshape(-1, 1)
    nt, ns = len(t), len(space)
    space_block = space.repeat(nt, 1)
    time_block = t.repeat_interleave(ns, dim=0)
    return torch.cat([space_block, time_block], dim=1).requires_grad_(True)


def _poisson2d_constants(problem):
    cache_name = "_poisson2d_constants"
    cached = getattr(problem, cache_name, None)
    if cached is not None:
        return cached
    with torch.enable_grad():
        coords = problem.tensor(problem.target_nodes, requires_grad=True)
        b = poisson2d_b(coords, create_graph=True)
        b_grad = _grad(b, coords, create_graph=False)
    cached = (b.detach(), b_grad.detach())
    setattr(problem, cache_name, cached)
    return cached


def _heat_forcing_constants(problem, times: np.ndarray):
    key = tuple(float(v) for v in times)
    cache = getattr(problem, "_heat_forcing_constants", {})
    if key in cache:
        return cache[key]
    spatial = problem.target_nodes[problem.pde_spatial_indices]
    xyt = np.vstack(
        [np.column_stack([spatial, np.full(len(spatial), t)]) for t in times]
    )
    with torch.enable_grad():
        coords = problem.tensor(xyt, requires_grad=True)
        forcing = heat_forcing(coords)
        forcing_grad = _grad(forcing, coords, create_graph=False)
    result = (forcing.detach().reshape(-1), forcing_grad.detach())
    cache[key] = result
    setattr(problem, "_heat_forcing_constants", cache)
    return result


def _ordered_operator(problem, name: str, order: int) -> torch.Tensor:
    key = name if order == 3 else f"{name}_p{order}"
    return problem.torch_operators[key]


def _heat_robin_reference(problem, times: np.ndarray) -> torch.Tensor:
    normals = problem.tensor(problem.boundary_normals)
    boundary_xy = problem.tensor(problem.target_nodes[problem.boundary_indices])
    time_values = problem.tensor(times).reshape(-1, 1)
    x = boundary_xy[:, 0].reshape(1, -1)
    y = boundary_xy[:, 1].reshape(1, -1)
    exact_u = 1.0 + torch.sin(pi * x) * torch.cos(pi * y) * torch.sin(pi * time_values)
    exact_dx = pi * torch.cos(pi * x) * torch.cos(pi * y) * torch.sin(pi * time_values)
    exact_dy = -pi * torch.sin(pi * x) * torch.sin(pi * y) * torch.sin(pi * time_values)
    return exact_u + exact_dx * normals[:, 0] + exact_dy * normals[:, 1]


def _heat_discrete_constraints(problem, u, dx_u, dy_u, times: np.ndarray) -> torch.Tensor:
    n_physical = len(problem.target_nodes)
    initial_loss = torch.mean((u[0, :n_physical] - 1.0) ** 2)
    bidx = torch.as_tensor(problem.boundary_indices, device=problem.device, dtype=torch.long)
    normals = problem.tensor(problem.boundary_normals)
    target_u = u[:, :n_physical]
    robin_left = target_u[:, bidx] + dx_u[:, bidx] * normals[:, 0] + dy_u[:, bidx] * normals[:, 1]
    return initial_loss + torch.mean((robin_left - _heat_robin_reference(problem, times)) ** 2)


def _heat_ad_constraints(problem, model) -> torch.Tensor:
    """Initial and Robin penalties with an AD normal derivative for gPINN."""
    initial_xy = problem.target_nodes
    initial_coords = problem.tensor(
        np.column_stack([initial_xy, np.zeros(len(initial_xy))])
    )
    initial_loss = torch.mean((model(initial_coords) - 1.0) ** 2)

    boundary_xy = problem.target_nodes[problem.boundary_indices]
    boundary_blocks = [
        np.column_stack([boundary_xy, np.full(len(boundary_xy), time)])
        for time in problem.times
    ]
    boundary_coords = problem.tensor(np.vstack(boundary_blocks), requires_grad=True)
    boundary_u = model(boundary_coords)
    boundary_gradient = _grad(boundary_u, boundary_coords, create_graph=True)
    nt, nb = len(problem.times), len(boundary_xy)
    normals = problem.tensor(problem.boundary_normals)
    normal_derivative = (
        boundary_gradient[:, 0].reshape(nt, nb) * normals[:, 0]
        + boundary_gradient[:, 1].reshape(nt, nb) * normals[:, 1]
    )
    robin_left = boundary_u.reshape(nt, nb) + normal_derivative
    return initial_loss + torch.mean(
        (robin_left - _heat_robin_reference(problem, problem.times)) ** 2
    )


@dataclass
class DiscreteTerms:
    residual: torch.Tensor
    gradients: list[torch.Tensor]
    constraint_loss: torch.Tensor
    support_values: torch.Tensor


@dataclass
class LossBreakdown:
    total: torch.Tensor
    discrete_pde: torch.Tensor
    discrete_gradient: torch.Tensor
    ad_residual: torch.Tensor
    ad_gradient: torch.Tensor
    constraints: torch.Tensor

    def scalars(self) -> dict[str, float]:
        return {
            "total_loss": float(self.total.detach()),
            "discrete_pde_loss": float(self.discrete_pde.detach()),
            "discrete_gradient_loss": float(self.discrete_gradient.detach()),
            "ad_residual_loss": float(self.ad_residual.detach()),
            "ad_gradient_loss": float(self.ad_gradient.detach()),
            "constraint_loss": float(self.constraints.detach()),
        }


def compute_discrete_terms(problem, model) -> DiscreteTerms:
    """Evaluate the explicitly discretized residual and analytical gradients."""
    ops = problem.torch_operators
    idx = torch.as_tensor(problem.pde_spatial_indices, device=problem.device, dtype=torch.long)
    zero = torch.zeros((), device=problem.device, dtype=problem.dtype)

    if problem.times is None:
        coordinates = problem.tensor(problem.support_nodes)
        u = model(coordinates).reshape(1, -1)
        if problem.name == "poisson1d":
            target_x = problem.tensor(problem.target_nodes)[:, 0]
            r = _spatial_apply(ops["dxx"], u)[0] - poisson1d_b(target_x)
            gx = _spatial_apply(ops["dxxx"], u)[0] - poisson1d_bx(target_x)
            return DiscreteTerms(r[idx], [gx[idx]], zero, u)
        b, b_grad = _poisson2d_constants(problem)
        lap = _spatial_apply(ops["dxx"], u) + _spatial_apply(ops["dyy"], u)
        gx = _spatial_apply(ops["dxxx"], u) + _spatial_apply(ops["dxyy"], u)
        gy = _spatial_apply(ops["dyxx"], u) + _spatial_apply(ops["dyyy"], u)
        return DiscreteTerms(
            lap[0, idx] - b[idx, 0],
            [gx[0, idx] - b_grad[idx, 0], gy[0, idx] - b_grad[idx, 1]],
            zero,
            u,
        )

    times = problem.positive_times if problem.name != "heat" else problem.times
    coordinates = _space_time_coordinates(problem, times)
    u_flat = model(coordinates)
    first = _grad(u_flat, coordinates, create_graph=True)
    time_axis = problem.spatial_dim
    ut_flat = first[:, time_axis : time_axis + 1]
    utt_flat = _grad(ut_flat, coordinates, create_graph=True)[:, time_axis : time_axis + 1]
    nt, ns = len(times), len(problem.support_nodes)
    u = u_flat.reshape(nt, ns)
    ut = ut_flat.reshape(nt, ns)
    utt = utt_flat.reshape(nt, ns)

    dx_u = _spatial_apply(ops["dx"], u)
    dxx_u = _spatial_apply(ops["dxx"], u)
    dx_ut = _spatial_apply(ops["dx"], ut)
    dxx_ut = _spatial_apply(ops["dxx"], ut)
    dxxx_u = _spatial_apply(ops["dxxx"], u)
    target_u = u[:, : len(problem.target_nodes)]
    target_ut = ut[:, : len(problem.target_nodes)]
    target_utt = utt[:, : len(problem.target_nodes)]

    if problem.name == "burgers":
        nu = 0.01 / pi
        residual = target_ut + target_u * dx_u - nu * dxx_u
        gx = dx_ut + dx_u**2 + target_u * dxx_u - nu * dxxx_u
        gt = target_utt + target_ut * dx_u + target_u * dx_ut - nu * dxx_ut
        return DiscreteTerms(
            residual[:, idx].reshape(-1),
            [gx[:, idx].reshape(-1), gt[:, idx].reshape(-1)],
            zero,
            u,
        )
    if problem.name == "allen_cahn":
        residual = target_ut - 0.001 * dxx_u + 5.0 * (target_u**3 - target_u)
        gx = dx_ut - 0.001 * dxxx_u + 5.0 * (3.0 * target_u**2 - 1.0) * dx_u
        gt = target_utt - 0.001 * dxx_ut + 5.0 * (3.0 * target_u**2 - 1.0) * target_ut
        return DiscreteTerms(
            residual[:, idx].reshape(-1),
            [gx[:, idx].reshape(-1), gt[:, idx].reshape(-1)],
            zero,
            u,
        )

    # Heat: the first row is t=0 and is reserved for the initial penalty.
    dy_u = _spatial_apply(ops["dy"], u)
    dyy_u = _spatial_apply(ops["dyy"], u)
    dy_ut = _spatial_apply(ops["dy"], ut)
    dxxx_u = _spatial_apply(ops["dxxx"], u)
    dxyy_u = _spatial_apply(ops["dxyy"], u)
    dyxx_u = _spatial_apply(ops["dyxx"], u)
    dyyy_u = _spatial_apply(ops["dyyy"], u)
    forcing, forcing_grad = _heat_forcing_constants(problem, problem.positive_times)
    positive = slice(1, None)
    residual = target_ut[positive, :,] - dxx_u[positive, :] - dyy_u[positive, :]
    gx = dx_ut[positive, :] - dxxx_u[positive, :] - dxyy_u[positive, :]
    gy = dy_ut[positive, :] - dyxx_u[positive, :] - dyyy_u[positive, :]
    gt = target_utt[positive, :] - dxx_ut[positive, :] - _spatial_apply(ops["dyy"], ut)[positive, :]
    residual = residual[:, idx].reshape(-1) - forcing
    gx = gx[:, idx].reshape(-1) - forcing_grad[:, 0]
    gy = gy[:, idx].reshape(-1) - forcing_grad[:, 1]
    gt = gt[:, idx].reshape(-1) - forcing_grad[:, 2]

    constraint = _heat_discrete_constraints(problem, u, dx_u, dy_u, times)
    return DiscreteTerms(residual, [gx, gy, gt], constraint, u)


def compute_dt_terms(problem, model, order: int) -> tuple[torch.Tensor, torch.Tensor]:
    """DT-PINN residual with order-matched RBF-FD spatial derivatives.

    DT-PINN has no residual-gradient penalty. Temporal derivatives remain AD,
    and the heat Robin normal derivative uses the same order-matched discrete
    first derivatives as the PDE backbone.
    """
    if order not in (3, 4, 5):
        raise ValueError("DT-PINN order must be 3, 4, or 5")
    idx = torch.as_tensor(problem.pde_spatial_indices, device=problem.device, dtype=torch.long)
    zero = torch.zeros((), device=problem.device, dtype=problem.dtype)
    dxx = _ordered_operator(problem, "dxx", order)

    if problem.times is None:
        u = model(problem.tensor(problem.support_nodes)).reshape(1, -1)
        if problem.name == "poisson1d":
            x = problem.tensor(problem.target_nodes)[:, 0]
            residual = _spatial_apply(dxx, u)[0] - poisson1d_b(x)
            return residual[idx], zero
        dyy = _ordered_operator(problem, "dyy", order)
        b, _ = _poisson2d_constants(problem)
        residual = _spatial_apply(dxx, u) + _spatial_apply(dyy, u)
        return residual[0, idx] - b[idx, 0], zero

    times = problem.positive_times if problem.name != "heat" else problem.times
    coordinates = _space_time_coordinates(problem, times)
    u_flat = model(coordinates)
    ut_flat = _grad(u_flat, coordinates, create_graph=True)[:, problem.spatial_dim : problem.spatial_dim + 1]
    nt, ns = len(times), len(problem.support_nodes)
    u = u_flat.reshape(nt, ns)
    ut = ut_flat.reshape(nt, ns)
    target_u = u[:, : len(problem.target_nodes)]
    target_ut = ut[:, : len(problem.target_nodes)]
    dxx_u = _spatial_apply(dxx, u)

    if problem.name == "burgers":
        dx_u = _spatial_apply(_ordered_operator(problem, "dx", order), u)
        residual = target_ut + target_u * dx_u - (0.01 / pi) * dxx_u
        return residual[:, idx].reshape(-1), zero
    if problem.name == "allen_cahn":
        residual = target_ut - 0.001 * dxx_u + 5.0 * (target_u**3 - target_u)
        return residual[:, idx].reshape(-1), zero

    dyy = _ordered_operator(problem, "dyy", order)
    dx = _ordered_operator(problem, "dx", order)
    dy = _ordered_operator(problem, "dy", order)
    dyy_u = _spatial_apply(dyy, u)
    forcing, _ = _heat_forcing_constants(problem, problem.positive_times)
    residual = target_ut[1:, :] - dxx_u[1:, :] - dyy_u[1:, :]
    residual = residual[:, idx].reshape(-1) - forcing
    constraint = _heat_discrete_constraints(
        problem, u, _spatial_apply(dx, u), _spatial_apply(dy, u), times
    )
    return residual, constraint


def compute_ad_correction(problem, model, candidate_positions) -> tuple[torch.Tensor, list[torch.Tensor]]:
    """Genuine pointwise AD residual and all coordinate derivatives on C."""
    coords = problem.candidate_coordinates(candidate_positions).detach().clone().requires_grad_(True)
    u = model(coords)
    du = _grad(u, coords, create_graph=True)
    if problem.name == "poisson1d":
        ux = du[:, 0:1]
        uxx = _grad(ux, coords, create_graph=True)[:, 0:1]
        residual = uxx - poisson1d_b(coords[:, 0:1])
    elif problem.name == "poisson2d":
        ux, uy = du[:, 0:1], du[:, 1:2]
        uxx = _grad(ux, coords, create_graph=True)[:, 0:1]
        uyy = _grad(uy, coords, create_graph=True)[:, 1:2]
        residual = uxx + uyy - poisson2d_b(coords, create_graph=True)
    elif problem.name == "burgers":
        ux, ut = du[:, 0:1], du[:, 1:2]
        uxx = _grad(ux, coords, create_graph=True)[:, 0:1]
        residual = ut + u * ux - (0.01 / pi) * uxx
    elif problem.name == "allen_cahn":
        ux, ut = du[:, 0:1], du[:, 1:2]
        uxx = _grad(ux, coords, create_graph=True)[:, 0:1]
        residual = ut - 0.001 * uxx + 5.0 * (u**3 - u)
    else:
        ux, uy, ut = du[:, 0:1], du[:, 1:2], du[:, 2:3]
        uxx = _grad(ux, coords, create_graph=True)[:, 0:1]
        uyy = _grad(uy, coords, create_graph=True)[:, 1:2]
        residual = ut - uxx - uyy - heat_forcing(coords)
    residual_gradient = _grad(residual, coords, create_graph=True)
    return residual.reshape(-1), [residual_gradient[:, j] for j in range(problem.input_dim)]


def compute_gpinn_terms(problem, model) -> tuple[torch.Tensor, list[torch.Tensor], torch.Tensor]:
    """Full-collocation AD residual and coordinate-gradient terms for gPINN."""
    candidates = np.arange(problem.candidate_count, dtype=np.int64)
    residual, gradients = compute_ad_correction(problem, model, candidates)
    if problem.name == "heat":
        constraints = _heat_ad_constraints(problem, model)
    else:
        constraints = torch.zeros((), device=problem.device, dtype=problem.dtype)
    return residual, gradients, constraints


def compute_loss(
    problem,
    model,
    *,
    method: str,
    lambda_weight: float,
    beta: float = 1.0,
    correction_positions=None,
) -> LossBreakdown:
    if method not in METHOD_NAMES:
        raise ValueError(f"unknown method {method!r}")
    zero = torch.zeros((), device=problem.device, dtype=problem.dtype)

    if method == "gpinn":
        residual, gradients, constraints = compute_gpinn_terms(problem, model)
        ad_residual = torch.mean(residual**2)
        ad_gradient = sum((torch.mean(value**2) for value in gradients), zero)
        total = ad_residual + lambda_weight * ad_gradient + constraints
        return LossBreakdown(total, zero, zero, ad_residual, ad_gradient, constraints)

    if method in DT_ORDERS:
        residual, constraints = compute_dt_terms(problem, model, DT_ORDERS[method])
        discrete_pde = torch.mean(residual**2)
        total = discrete_pde + constraints
        return LossBreakdown(total, discrete_pde, zero, zero, zero, constraints)

    terms = compute_discrete_terms(problem, model)
    discrete_pde = torch.mean(terms.residual**2)
    discrete_gradient = sum((torch.mean(value**2) for value in terms.gradients), zero)
    ad_residual = torch.zeros_like(discrete_pde)
    ad_gradient = torch.zeros_like(discrete_pde)
    if method in ("adaptive", "fixed"):
        if correction_positions is None:
            raise ValueError(f"{method} requires correction positions")
        correction_residual, correction_gradients = compute_ad_correction(
            problem, model, correction_positions
        )
        ad_residual = torch.mean(correction_residual**2)
        ad_gradient = sum(
            (torch.mean(value**2) for value in correction_gradients), torch.zeros_like(ad_residual)
        )
    total = (
        discrete_pde
        + lambda_weight * discrete_gradient
        + terms.constraint_loss
        + beta * (ad_residual + lambda_weight * ad_gradient)
    )
    return LossBreakdown(
        total, discrete_pde, discrete_gradient, ad_residual, ad_gradient, terms.constraint_loss
    )


def disagreement_indicator(problem, model) -> np.ndarray:
    """Absolute p=3 versus p=4 second-derivative disagreement on candidates."""
    with torch.no_grad():
        if problem.times is None:
            u = model(problem.tensor(problem.support_nodes)).reshape(1, -1)
        else:
            coords = _space_time_coordinates(problem, problem.positive_times)
            u = model(coords).reshape(len(problem.positive_times), len(problem.support_nodes))
        if problem.spatial_dim == 1:
            p3 = _spatial_apply(problem.torch_operators["dxx"], u)
            p4 = _spatial_apply(problem.torch_operators["dxx_p4"], u)
        else:
            p3 = _spatial_apply(problem.torch_operators["dxx"], u) + _spatial_apply(
                problem.torch_operators["dyy"], u
            )
            p4 = _spatial_apply(problem.torch_operators["dxx_p4"], u) + _spatial_apply(
                problem.torch_operators["dyy_p4"], u
            )
        idx = torch.as_tensor(problem.pde_spatial_indices, device=problem.device, dtype=torch.long)
        return torch.abs(p3[:, idx] - p4[:, idx]).reshape(-1).cpu().numpy()


class AdaptiveCorrectionSelector:
    def __init__(self, candidate_count: int, correction_count: int, *, seed: int = 200):
        if not 1 <= correction_count <= candidate_count:
            raise ValueError("correction_count must be in [1, candidate_count]")
        self.candidate_count = int(candidate_count)
        self.correction_count = int(correction_count)
        self.cursor = int(seed % candidate_count)
        self.updates = 0

    def select(self, disagreement: np.ndarray) -> np.ndarray:
        eta = np.asarray(disagreement, dtype=np.float64).reshape(-1)
        if len(eta) != self.candidate_count or not np.all(np.isfinite(eta)):
            raise ValueError("disagreement must contain one finite value per candidate")
        high_count = self.correction_count // 2
        # lexsort makes ties deterministic by ascending candidate index.
        ordered = np.lexsort((np.arange(self.candidate_count), -eta))
        high = ordered[:high_count].tolist()
        selected = set(high)
        coverage: list[int] = []
        scanned = 0
        while len(coverage) < self.correction_count - high_count:
            candidate = (self.cursor + scanned) % self.candidate_count
            scanned += 1
            if candidate not in selected:
                coverage.append(candidate)
                selected.add(candidate)
            if scanned > 2 * self.candidate_count:
                raise RuntimeError("unable to fill rotating coverage selection")
        self.cursor = (self.cursor + self.correction_count - high_count) % self.candidate_count
        self.updates += 1
        result = np.asarray(high + coverage, dtype=np.int64)
        if len(result) != self.correction_count or len(np.unique(result)) != len(result):
            raise RuntimeError("adaptive selection failed uniqueness/count invariant")
        return result


class FixedCorrectionSelector:
    def __init__(self, candidate_count: int, correction_count: int):
        if not 1 <= correction_count <= candidate_count:
            raise ValueError("correction_count must be in [1, candidate_count]")
        self._indices = np.floor(
            (np.arange(correction_count, dtype=np.float64) + 0.5)
            * candidate_count
            / correction_count
        ).astype(np.int64)
        if len(np.unique(self._indices)) != correction_count:
            raise RuntimeError("evenly spaced fixed selection unexpectedly produced duplicates")

    def select(self) -> np.ndarray:
        return self._indices.copy()
