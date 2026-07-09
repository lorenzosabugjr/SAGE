"""
Unit tests for optimizers/descent.py.

Run with: python -m unittest tests.test_optimizers
"""

import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import optimizers
from optimizers import GradientDescent, StepSizeMode
from estimators import TruthEstimator
from utils.history import HistoryBuffer


class _Quadratic:
    """Deterministic quadratic f(x) = 0.5 * ||x||^2, gradient(x) = x."""

    def eval(self, x, noise_type=None, noise_param=0.0):
        x = np.asarray(x, dtype=float)
        return float(0.5 * np.dot(x, x))

    def gradient(self, x):
        return np.asarray(x, dtype=float).copy()


class GradientDescentTests(unittest.TestCase):
    def _make_solver(self, x0, **kwargs):
        history = HistoryBuffer()
        problem = _Quadratic()

        def obj_func(x):
            val = problem.eval(x)
            history.add(x, val)
            return val

        estimator = TruthEstimator(obj_func, dim=x0.shape[0], problem=problem, history=history)
        solver = GradientDescent(
            fun=obj_func,
            x0=x0,
            grad_estimator=estimator,
            **kwargs,
        )
        return solver, history

    def test_adaptive_step_reduces_objective_and_records_history(self):
        x0 = np.array([10.0, 10.0])
        solver, history = self._make_solver(x0)

        z_before = solver.z_k
        n_before = history.Zn.size

        solver.step()

        self.assertLess(solver.z_k, z_before)
        self.assertGreater(history.Zn.size, n_before)
        # Every line-search evaluation (rejected or accepted) is recorded.
        self.assertGreaterEqual(history.Zn.size - n_before, 1)
        self.assertEqual(solver.k, 1)

    def test_rejected_evaluations_enter_history(self):
        # A tiny min_stepsize combined with a huge initial stepsize forces at
        # least one rejection before the Armijo condition is satisfied.
        x0 = np.array([10.0, 10.0])
        solver, history = self._make_solver(x0, stepsize=1e6)

        n_before = history.Zn.size
        solver.step()
        n_after = history.Zn.size

        # More than one evaluation means a rejected sample was recorded.
        self.assertGreater(n_after - n_before, 1)

    def test_no_standard_descent_alias_exported(self):
        self.assertFalse(hasattr(optimizers, "StandardDescent"))
        self.assertNotIn("StandardDescent", optimizers.__all__)

    def test_step_size_mode_enum_values(self):
        self.assertEqual(StepSizeMode.FIXED.value, 0)
        self.assertEqual(StepSizeMode.ADAPTIVE.value, 1)


if __name__ == "__main__":
    unittest.main()
