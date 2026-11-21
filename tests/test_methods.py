"""Numerical verification for the methods and benchmark used in the paper."""
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
    PAPER_METHOD_NAMES,
    compute_discrete_terms,
    compute_dt_terms,
    compute_gpinn_terms,
    compute_loss,
)
from networks import FullyConnectedTanh, ConstrainedSolution, make_network
from problems import make_problem, poisson1d_exact
from rbf_fd import build_operator
from train import TrainingConfig, run_training


torch.set_num_threads(1)


class TestPaperRBFFD(unittest.TestCase):
    def test_second_and_third_derivatives(self):
        nodes = np.linspace(-1.0, 1.0, 81)[:, None]
        targets = nodes[5:-5]
        sampled = np.sin(2.0 * nodes[:, 0])
        cases = [
            ((2,), -4.0 * np.sin(2.0 * targets[:, 0]), 1e-4),
            ((3,), -8.0 * np.cos(2.0 * targets[:, 0]), 2e-3),
        ]
        for derivative, exact, tolerance in cases:
            operator, diagnostics = build_operator(nodes, derivative, targets=targets)
            self.assertLess(np.max(np.abs(operator @ sampled - exact)), tolerance)
            self.assertLess(diagnostics.polynomial_reproduction_error, 1e-7)
            self.assertTrue(np.isfinite(diagnostics.max_condition))

    def test_duplicate_support_nodes_are_rejected(self):
        nodes = np.array([[0.0], [0.0], [1.0], [2.0], [3.0], [4.0], [5.0]])
        with self.assertRaises(ValueError):
            build_operator(nodes, (2,), stencil_size=5)


class TestPaperResiduals(unittest.TestCase):
    def test_ddg_exact_residual_and_gradient(self):
        problem = make_problem("poisson1d", method="ddg")
        terms = compute_discrete_terms(problem, poisson1d_exact)
        residual_rms = torch.sqrt(torch.mean(terms.residual**2)).detach().item()
        gradient_rms = torch.sqrt(torch.mean(terms.gradients[0] ** 2)).detach().item()
        self.assertLess(residual_rms, 0.01)
        # The paper's 13-point order-3 third-derivative stencil on 65 support
        # nodes has an RMS error of about 0.332 for this high-frequency field.
        self.assertLess(gradient_rms, 0.4)

    def test_gpinn_exact_residual_and_gradient(self):
        problem = make_problem("poisson1d", method="gpinn")
        residual, gradients, constraints = compute_gpinn_terms(problem, poisson1d_exact)
        self.assertLess(torch.max(torch.abs(residual)).detach().item(), 2e-4)
        self.assertLess(torch.max(torch.abs(gradients[0])).detach().item(), 2e-3)
        self.assertEqual(constraints.detach().item(), 0.0)

    def test_dt_orders_three_and_four(self):
        for order in (3, 4):
            problem = make_problem("poisson1d", method=f"dt{order}")
            residual, constraints = compute_dt_terms(problem, poisson1d_exact, order)
            self.assertTrue(torch.all(torch.isfinite(residual)))
            self.assertLess(torch.sqrt(torch.mean(residual**2)).detach().item(), 0.01)
            self.assertEqual(constraints.detach().item(), 0.0)


class TestPaperBoundaryCondition(unittest.TestCase):
    def test_poisson_dirichlet_values_are_exact(self):
        problem = make_problem("poisson1d", method="pinn")
        model = ConstrainedSolution(
            "poisson1d", FullyConnectedTanh(problem.input_dim, [5])
        )
        coordinates = problem.tensor(np.array([[0.0], [pi]]))
        expected = torch.tensor([0.0, pi], dtype=problem.dtype)
        self.assertTrue(
            torch.allclose(model(coordinates).detach().reshape(-1), expected, atol=2e-6)
        )


class TestPaperTrainingPaths(unittest.TestCase):
    def test_all_paper_losses_have_finite_parameter_gradients(self):
        for method in PAPER_METHOD_NAMES:
            problem = make_problem("poisson1d", method=method)
            model = make_network(problem)
            loss = compute_loss(problem, model, method=method, lambda_weight=0.1)
            loss.total.backward()
            self.assertTrue(torch.isfinite(loss.total))
            self.assertTrue(
                all(
                    parameter.grad is not None and torch.all(torch.isfinite(parameter.grad))
                    for parameter in model.parameters()
                )
            )

    def test_one_update_for_every_paper_method(self):
        with tempfile.TemporaryDirectory() as destination:
            for method in PAPER_METHOD_NAMES:
                config = TrainingConfig(
                    method,
                    "poisson1d",
                    steps=1,
                    seed=202,
                    output_dir=destination,
                    log_every=1,
                )
                _, records, _ = run_training(
                    config, save_outputs=False, verbose=False
                )
                self.assertEqual(len(records), 1)
                self.assertTrue(np.isfinite(records[0]["total_loss"]))


if __name__ == "__main__":
    unittest.main(verbosity=2)
