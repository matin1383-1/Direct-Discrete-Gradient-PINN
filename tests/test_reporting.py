"""Regression coverage for numerical acceptance and scientific reporting."""
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from methods import compute_loss
from paper_experiment import PAPER_CONFIGURATIONS, paper_test_points, paper_validation_points
from problems import make_problem
from rbf_fd import build_operator
from train import TrainingConfig, relative_errors, run_training


class ReportingRegression(unittest.TestCase):
    def test_paper_driver_encodes_all_seven_final_configurations(self):
        self.assertEqual(len(PAPER_CONFIGURATIONS), 7)
        self.assertEqual(
            [configuration.key for configuration in PAPER_CONFIGURATIONS],
            ['pinn', 'gpinn_w0.1', 'ddg_w0.01', 'ddg_w1', 'ddg_w0.1', 'dt3', 'dt4'],
        )
        validation = paper_validation_points()
        test = paper_test_points()
        self.assertEqual(validation.shape, (1000, 1))
        self.assertEqual(test.shape, (1001, 1))
        self.assertGreaterEqual(validation.min(), 0.0)
        self.assertLessEqual(validation.max(), np.pi)
        np.testing.assert_allclose(test[[0, -1], 0], [0.0, np.pi])

    def test_poisson1d_matches_paper_cloud_and_ddg_stencils(self):
        problem = make_problem('poisson1d', method='ddg')
        self.assertEqual(problem.support_nodes.shape, (65, 1))
        self.assertEqual(problem.target_nodes.shape, (15, 1))
        np.testing.assert_allclose(
            problem.target_nodes[:, 0], np.arange(1, 16) * np.pi / 16.0
        )
        self.assertEqual(problem.diagnostics['dxx'].stencil_size, 11)
        self.assertEqual(problem.diagnostics['dxxx'].stencil_size, 13)
        self.assertEqual(problem.diagnostics['dxx'].phs_exponent, 5)
        self.assertEqual(problem.diagnostics['dxxx'].phs_exponent, 5)

    def test_saved_state_and_metrics_agree(self):
        with tempfile.TemporaryDirectory() as directory:
            validation = paper_validation_points()
            config = TrainingConfig('dt4', 'poisson1d', steps=2, log_every=1,
                                    output_dir=directory, dtype='float32')
            model, records, _ = run_training(
                config, verbose=False, evaluation_points=validation
            )
            problem = make_problem('poisson1d', method='dt4', dtype=torch.float32)
            measured = compute_loss(problem, model, method='dt4', lambda_weight=.1)
            self.assertAlmostEqual(records[-1]['total_loss'], measured.total.item(), places=6)
            self.assertEqual(tuple(records[-1][k] for k in ('relative_l1_error', 'relative_l2_error')),
                             relative_errors(problem, model, validation))
            folder = Path(directory) / 'poisson1d_dt4_seed200'
            saved = torch.load(folder / 'checkpoint_final.pt', weights_only=False)
            for key, value in model.state_dict().items():
                torch.testing.assert_close(saved['model_state_dict'][key], value)
            summary = json.loads((folder / 'run_summary.json').read_text())
            self.assertGreater(summary['total_process_time'], records[-1]['total_process_time'])
            self.assertTrue((folder / 'checkpoint_best_observed_reference_l2.pt').is_file())
            with self.assertRaises(FileExistsError):
                run_training(config, verbose=False)

    def test_only_required_operators_are_built(self):
        for method, expected in [('pinn', set()), ('gpinn', set()),
                                 ('dt4', {'dxx_p4'}),
                                 ('ddg', {'dxx', 'dxxx'})]:
            p = make_problem('poisson1d', method=method)
            self.assertEqual(set(p.operators_scipy), expected)
            if method in ('pinn', 'gpinn'):
                self.assertEqual(p.operator_seconds, 0.0)

    def test_incompatible_reused_problem_is_rejected_clearly(self):
        problem = make_problem('poisson1d', method='dt4')
        config = TrainingConfig('ddg', 'poisson1d', steps=1)
        with self.assertRaisesRegex(ValueError, 'missing operators required by ddg'):
            run_training(config, problem=problem, save_outputs=False, verbose=False)

    def test_bad_solve_and_bad_reproduction_are_rejected(self):
        nodes = np.linspace(0, 1, 32)[:, None]
        with patch('numpy.linalg.solve', side_effect=lambda a, b: np.zeros_like(b)):
            with self.assertRaises(np.linalg.LinAlgError):
                build_operator(nodes, (2,))
        with patch('rbf_fd.polynomial_reproduction_error', return_value=1.0):
            with self.assertRaises(np.linalg.LinAlgError):
                build_operator(nodes, (2,))

    def test_invalid_weights_and_learning_rate_rejected(self):
        for kwargs in ({'learning_rate': 0}, {'lambda_weight': -1}):
            with self.assertRaises(ValueError):
                TrainingConfig('gpinn', 'poisson1d', **kwargs).resolved()


if __name__ == '__main__':
    unittest.main(verbosity=2)
