"""Numerical and end-to-end verification for the standalone implementation."""
from __future__ import annotations

import tempfile
import unittest
from math import pi
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import torch

from methods import (
    compute_ad_correction,
    compute_discrete_terms,
    compute_dt_terms,
    compute_gpinn_terms,
    compute_loss,
)
from networks import FullyConnectedTanh, ConstrainedSolution, make_network
from problems import heat_exact, make_problem, poisson1d_exact
from rbf_fd import build_operator
from train import TrainingConfig, run_training


torch.set_num_threads(1)


class TestRBFFD(unittest.TestCase):
    def test_1d_first_second_third_derivatives(self):
        nodes = np.linspace(-1.0, 1.0, 81)[:, None]
        targets = nodes[5:-5]
        sampled = np.sin(2.0 * nodes[:, 0])
        cases = [
            ((1,), 2.0 * np.cos(2.0 * targets[:, 0]), 1e-5),
            ((2,), -4.0 * np.sin(2.0 * targets[:, 0]), 1e-4),
            ((3,), -8.0 * np.cos(2.0 * targets[:, 0]), 2e-3),
        ]
        for derivative, exact, tolerance in cases:
            operator, diagnostics = build_operator(nodes, derivative, targets=targets)
            self.assertLess(np.max(np.abs(operator @ sampled - exact)), tolerance)
            self.assertLess(diagnostics.polynomial_reproduction_error, 1e-7)
            self.assertTrue(np.isfinite(diagnostics.max_condition))

    def test_2d_mixed_third_derivatives(self):
        axis = np.linspace(-1.0, 1.0, 15)
        xx, yy = np.meshgrid(axis, axis, indexing="ij")
        nodes = np.column_stack([xx.ravel(), yy.ravel()])
        targets = nodes[(np.abs(nodes[:, 0]) < 0.72) & (np.abs(nodes[:, 1]) < 0.72)]
        sampled = np.sin(nodes[:, 0]) * np.cos(nodes[:, 1])
        cases = {
            (3, 0): -np.cos(targets[:, 0]) * np.cos(targets[:, 1]),
            (1, 2): -np.cos(targets[:, 0]) * np.cos(targets[:, 1]),
            (2, 1): np.sin(targets[:, 0]) * np.sin(targets[:, 1]),
            (0, 3): np.sin(targets[:, 0]) * np.sin(targets[:, 1]),
        }
        for derivative, exact in cases.items():
            operator, diagnostics = build_operator(nodes, derivative, targets=targets)
            self.assertLess(np.max(np.abs(operator @ sampled - exact)), 3e-3)
            self.assertLess(diagnostics.polynomial_reproduction_error, 1e-7)

    def test_degenerate_cloud_is_rejected(self):
        duplicate_nodes = np.array([[0.0], [0.0], [1.0], [2.0], [3.0], [4.0], [5.0]])
        with self.assertRaises(ValueError):
            build_operator(duplicate_nodes, (1,), stencil_size=5)


class AnalyticBurgersField(torch.nn.Module):
    def forward(self, coordinates):
        return torch.sin(pi * coordinates[:, 0:1]) * torch.exp(-coordinates[:, 1:2])


class TestResiduals(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.poisson = make_problem("poisson1d")
        cls.burgers = make_problem("burgers")

    def test_poisson_exact_discrete_residual_and_gradient(self):
        terms = compute_discrete_terms(self.poisson, poisson1d_exact)
        residual_rms = torch.sqrt(torch.mean(terms.residual**2)).detach().item()
        gradient_rms = torch.sqrt(torch.mean(terms.gradients[0] ** 2)).detach().item()
        self.assertLess(residual_rms, 0.01)
        self.assertLess(gradient_rms, 0.25)

    def test_gpinn_exact_residual_and_gradient(self):
        residual, gradients, constraints = compute_gpinn_terms(self.poisson, poisson1d_exact)
        self.assertLess(torch.max(torch.abs(residual)).detach().item(), 2e-4)
        self.assertLess(torch.max(torch.abs(gradients[0])).detach().item(), 2e-3)
        self.assertEqual(constraints.detach().item(), 0.0)

    def test_dt_orders_have_independent_operators_and_accurate_residuals(self):
        for order in (3, 4, 5):
            key = "dxx" if order == 3 else f"dxx_p{order}"
            self.assertIn(key, self.poisson.torch_operators)
            residual, constraints = compute_dt_terms(self.poisson, poisson1d_exact, order)
            self.assertTrue(torch.all(torch.isfinite(residual)))
            self.assertLess(torch.sqrt(torch.mean(residual**2)).detach().item(), 0.01)
            self.assertEqual(constraints.detach().item(), 0.0)

    def test_burgers_discrete_formulas_against_analytic_reference(self):
        terms = compute_discrete_terms(self.burgers, AnalyticBurgersField())
        nspace = len(self.burgers.pde_spatial_indices)
        x = self.burgers.tensor(
            self.burgers.target_nodes[self.burgers.pde_spatial_indices]
        )[:, 0].repeat(len(self.burgers.positive_times))
        t = self.burgers.tensor(self.burgers.positive_times).repeat_interleave(nspace)
        u = torch.sin(pi * x) * torch.exp(-t)
        ux = pi * torch.cos(pi * x) * torch.exp(-t)
        uxx = -(pi**2) * u
        uxxx = -(pi**3) * torch.cos(pi * x) * torch.exp(-t)
        ut, utt, uxt, uxxt = -u, u, -ux, -uxx
        nu = 0.01 / pi
        expected = [
            ut + u * ux - nu * uxx,
            uxt + ux**2 + u * uxx - nu * uxxx,
            utt + ut * ux + u * uxt - nu * uxxt,
        ]
        actual = [terms.residual, *terms.gradients]
        tolerances = [2e-5, 5e-4, 2e-5]
        for value, reference, tolerance in zip(actual, expected, tolerances):
            rms = torch.sqrt(torch.mean((value - reference) ** 2)).detach().item()
            self.assertLess(rms, tolerance)

    def test_ad_correction_is_genuine_coordinate_ad(self):
        positions = np.array([0, 300, 900, self.burgers.candidate_count - 1])
        residual, gradients = compute_ad_correction(
            self.burgers, AnalyticBurgersField(), positions
        )
        coordinates = self.burgers.candidate_coordinates(positions)
        x, t = coordinates[:, 0], coordinates[:, 1]
        u = torch.sin(pi * x) * torch.exp(-t)
        ux = pi * torch.cos(pi * x) * torch.exp(-t)
        uxx = -(pi**2) * u
        uxxx = -(pi**3) * torch.cos(pi * x) * torch.exp(-t)
        nu = 0.01 / pi
        references = [
            -u + u * ux - nu * uxx,
            -ux + ux**2 + u * uxx - nu * uxxx,
            u - u * ux - u * ux + nu * uxx,
        ]
        for value, reference in zip([residual, *gradients], references):
            self.assertTrue(torch.allclose(value.detach(), reference, atol=2e-5, rtol=2e-5))


class TestConstraints(unittest.TestCase):
    def test_hard_constraints(self):
        cases = [
            ("poisson1d", np.array([[0.0], [pi]])),
            ("poisson2d", np.array([[0.0, 0.3], [1.0, 0.7], [0.4, 0.0], [0.6, 1.0]])),
            ("burgers", np.array([[-1.0, 0.4], [1.0, 0.8], [0.25, 0.0]])),
            ("allen_cahn", np.array([[-1.0, 0.4], [1.0, 0.8], [0.25, 0.0]])),
        ]
        for name, points in cases:
            problem = make_problem(name)
            raw = FullyConnectedTanh(problem.input_dim, [5])
            model = ConstrainedSolution(name, raw)
            coordinates = problem.tensor(points)
            values = model(coordinates).detach().reshape(-1)
            if name == "poisson1d":
                expected = torch.tensor([0.0, pi], dtype=values.dtype)
            elif name == "poisson2d":
                expected = torch.zeros_like(values)
            elif name == "burgers":
                expected = -torch.sin(pi * coordinates[:, 0])
            else:
                expected = coordinates[:, 0] ** 2 * torch.cos(pi * coordinates[:, 0])
            self.assertTrue(torch.allclose(values, expected, atol=2e-6, rtol=2e-6))

class TestDifferentiabilityAndExecution(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.poisson = make_problem("poisson1d")

    def test_all_losses_have_finite_parameter_gradients(self):
        for method in ("gpinn", "dt3", "dt4", "dt5", "direct"):
            model = make_network(self.poisson)
            loss = compute_loss(
                self.poisson,
                model,
                method=method,
                lambda_weight=0.1,
            )
            loss.total.backward()
            self.assertTrue(torch.isfinite(loss.total))
            self.assertTrue(
                all(
                    parameter.grad is not None and torch.all(torch.isfinite(parameter.grad))
                    for parameter in model.parameters()
                )
            )

    def test_short_training_all_methods_steady_and_time_dependent(self):
        burgers = make_problem("burgers")
        with tempfile.TemporaryDirectory() as destination:
            for problem in (self.poisson, burgers):
                for method in ("gpinn", "dt3", "dt4", "dt5", "direct"):
                    config = TrainingConfig(
                        method,
                        problem.name,
                        steps=1,
                        seed=202,
                        output_dir=destination,
                        log_every=1,
                    )
                    _, records, _ = run_training(
                        config, problem=problem, save_outputs=False, verbose=False
                    )
                    self.assertEqual(len(records), 1)
                    self.assertTrue(np.isfinite(records[0]["total_loss"]))

    def test_disk_geometry_robin_and_heat_residual_smoke(self):
        heat = make_problem("heat")
        self.assertGreater(len(heat.support_nodes), len(heat.target_nodes))
        radii = np.linalg.norm(heat.target_nodes[heat.boundary_indices], axis=1)
        np.testing.assert_allclose(radii, 1.0, atol=1e-12)
        terms = compute_discrete_terms(heat, heat_exact)
        self.assertLess(terms.constraint_loss.detach().item(), 1e-5)
        self.assertLess(torch.sqrt(torch.mean(terms.residual**2)).detach().item(), 0.03)
        for gradient in terms.gradients:
            self.assertLess(torch.sqrt(torch.mean(gradient**2)).detach().item(), 0.1)


if __name__ == "__main__":
    unittest.main(verbosity=2)
