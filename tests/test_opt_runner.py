"""
Unit tests for tests/opt_runner.py.

Run with: python -m unittest tests.test_opt_runner
"""

import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from tests.opt_runner import OptimizationTrial
from utils.noise import NoiseType


class OptimizationTrialTests(unittest.TestCase):
    def test_initial_evaluation_counts_against_budget(self):
        trial = OptimizationTrial(
            problem_name="least-squares",
            grad_est_name="truth",
            maxevals=3,
            dims=2,
            randseed=1,
        )
        # The initial sampled point is unconditionally recorded before any
        # solver step runs, so it already counts toward the budget.
        self.assertEqual(trial.history.Zn.size, 1)

        result = trial.run()

        # Budget is enforced per objective call; the run must never exceed it.
        self.assertLessEqual(trial.history.Zn.size, trial.maxevals)
        self.assertEqual(result["n_evals"], trial.history.Zn.size)

    def test_budget_exhaustion_mid_line_search_is_not_an_error(self):
        # A tiny budget forces StopIteration to fire inside the line search;
        # run() must swallow it and return normally.
        trial = OptimizationTrial(
            problem_name="least-squares",
            grad_est_name="truth",
            maxevals=2,
            dims=2,
            randseed=1,
        )
        result = trial.run()
        self.assertLessEqual(result["n_evals"], trial.maxevals)

    def test_history_is_per_evaluation_not_per_accepted_step(self):
        # A huge initial stepsize forces at least one line-search rejection,
        # so history entries must outnumber accepted optimizer steps.
        trial = OptimizationTrial(
            problem_name="least-squares",
            grad_est_name="truth",
            maxevals=50,
            dims=2,
            randseed=1,
            stepsize=1e8,
        )
        trial.run()

        self.assertGreater(trial.history.Zn.size, trial.solver.k)
        # Every evaluation has a matching iterate-tracking entry.
        self.assertEqual(trial.history.z_k_eval_hist.size, trial.history.Zn.size)
        self.assertEqual(trial.history.z_k_true_hist.size, trial.history.Zn.size)

    def test_denominator_uses_deterministic_optimizer_start_value(self):
        trial = OptimizationTrial(
            problem_name="least-squares",
            grad_est_name="truth",
            maxevals=5,
            dims=2,
            randseed=1,
            noise_type=NoiseType.UNIFORM,
            noise_param=5.0,
        )
        result = trial.run()

        expected = trial.problem.eval(trial.X_start, trial.noise_type, 0.0)
        # Deterministic: repeated noiseless evaluation at the same point is stable.
        self.assertEqual(expected, trial.problem.eval(trial.X_start, trial.noise_type, 0.0))
        self.assertEqual(result["Z_start_true"], expected)
        self.assertEqual(result["Z_start_true"], trial.Z_start_true)

    def test_non_sage_optimizer_starts_from_initial_point(self):
        trial = OptimizationTrial(
            problem_name="least-squares",
            grad_est_name="truth",
            maxevals=5,
            dims=2,
            randseed=1,
        )
        np.testing.assert_array_equal(trial.X_start, trial.X_initial)
        self.assertEqual(trial.Z_start_eval, trial.Z_initial_eval)
        self.assertEqual(trial.Z_start_true, trial.Z_initial_true)


if __name__ == "__main__":
    unittest.main()
