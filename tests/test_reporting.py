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
from problems import evaluation_coordinates, make_problem
from rbf_fd import build_operator
from train import TrainingConfig, relative_errors, run_training


class ReportingRegression(unittest.TestCase):
    def test_saved_state_and_metrics_agree(self):
        with tempfile.TemporaryDirectory() as directory:
            config = TrainingConfig('dt4', 'poisson1d', steps=2, log_every=1,
                                    output_dir=directory, dtype='float64')
            model, records, _ = run_training(config, verbose=False)
            problem = make_problem('poisson1d', method='dt4', dtype=torch.float64)
            measured = compute_loss(problem, model, method='dt4', lambda_weight=.1)
            self.assertAlmostEqual(records[-1]['total_loss'], measured.total.item(), places=10)
            self.assertEqual(tuple(records[-1][k] for k in ('relative_l1_error', 'relative_l2_error')),
                             relative_errors(problem, model))
            folder = Path(directory) / 'poisson1d_dt4_seed200'
            saved = torch.load(folder / 'checkpoint_final.pt', weights_only=False)
            for key, value in model.state_dict().items():
                torch.testing.assert_close(saved['model_state_dict'][key], value)
            summary = json.loads((folder / 'run_summary.json').read_text())
            self.assertGreater(summary['total_process_time'], records[-1]['total_process_time'])
            self.assertTrue((folder / 'checkpoint_best_observed_reference_l2.pt').is_file())
            with self.assertRaises(FileExistsError):
                run_training(config, verbose=False)

    def test_evaluation_grid_is_independent(self):
        p = make_problem('poisson1d', method='gpinn')
        points = evaluation_coordinates(p.name)
        self.assertGreater(len(points), len(p.target_nodes))
        self.assertGreater(np.min(np.abs(points - p.target_nodes.T)), 1e-8)

    def test_only_required_operators_are_built(self):
        for method, expected in [('gpinn', set()), ('dt4', {'dx_p4', 'dxx_p4'}),
                                 ('direct', {'dx', 'dxx', 'dxxx'})]:
            p = make_problem('poisson1d', method=method)
            self.assertEqual(set(p.operators_scipy), expected)
            if method == 'gpinn':
                self.assertEqual(p.operator_seconds, 0.0)

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
