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


class GradientDescentIncumbentHistoryTests(unittest.TestCase):
    """Milestone 3: the optimizer -- not the objective wrapper -- decides
    when the shared history's incumbent moves, and it must move on the same
    evaluation index that produced the accepted point (not the next one)."""

    def _make_solver(self, x0, **kwargs):
        history = HistoryBuffer()
        problem = _Quadratic()

        def obj_func(x):
            val = problem.eval(x)
            history.add(x, val, z_true=val)
            return val

        z0 = problem.eval(x0)
        history.init_incumbent(z0, z0)
        history.add(x0, z0, z_true=z0)

        estimator = TruthEstimator(obj_func, dim=x0.shape[0], problem=problem, history=history)
        solver = GradientDescent(
            fun=obj_func,
            x0=x0,
            grad_estimator=estimator,
            z0=z0,
            **kwargs,
        )
        return solver, history

    def test_initial_center_is_the_incumbent(self):
        x0 = np.array([10.0, 10.0])
        _, history = self._make_solver(x0)

        self.assertEqual(history.z_k_eval_hist.size, 1)
        self.assertEqual(history.z_k_eval_hist[0], history.Zn[0])
        self.assertEqual(history.z_k_true_hist[0], history.Zn[0])

    def test_rejected_then_accepted_step_marks_incumbent_on_accept_index(self):
        # A huge initial stepsize forces at least one rejection before the
        # Armijo condition is satisfied.
        x0 = np.array([10.0, 10.0])
        solver, history = self._make_solver(x0, stepsize=1e6)
        center = history.z_k_eval_hist[0]

        solver.step()

        # Raw history retains every evaluation, one-to-one with the
        # incumbent-history record.
        self.assertEqual(history.Zn.size, history.z_k_eval_hist.size)
        self.assertGreater(history.Zn.size, 2)  # center + reject(s) + accept

        # Every evaluation up to (but not including) the accepted one still
        # forward-fills the original center: rejections do not move the
        # incumbent.
        np.testing.assert_array_equal(history.z_k_eval_hist[:-1], center)

        # The accepted evaluation changes the incumbent on its own index,
        # not the following one, and matches its own raw evaluation.
        self.assertEqual(history.z_k_eval_hist[-1], history.Zn[-1])
        self.assertNotEqual(history.z_k_eval_hist[-1], center)
        self.assertEqual(solver.k, 1)

    def test_fixed_step_mode_marks_incumbent_on_accept_index(self):
        # Fixed-step mode has no rejection path: every step is an
        # acceptance, and must still land on the same evaluation's index.
        x0 = np.array([10.0, 10.0])
        solver, history = self._make_solver(x0, stepsize=0.1, stepsizemode=StepSizeMode.FIXED)
        center = history.z_k_eval_hist[0]

        solver.step()

        self.assertEqual(history.Zn.size, 2)
        self.assertEqual(history.z_k_eval_hist[-1], history.Zn[-1])
        self.assertNotEqual(history.z_k_eval_hist[-1], center)

        solver.step()

        self.assertEqual(history.Zn.size, 3)
        self.assertEqual(history.z_k_eval_hist[-1], history.Zn[-1])
        # Incumbent-history is monotonically updated: the earlier accepted
        # evaluation's own entry is untouched by the later acceptance.
        self.assertEqual(history.z_k_eval_hist[1], history.Zn[1])


class HistoryBufferIncumbentTests(unittest.TestCase):
    """Milestone 3: HistoryBuffer's own incumbent bookkeeping API."""

    def test_add_without_incumbent_init_leaves_incumbent_hist_empty(self):
        history = HistoryBuffer()
        history.add(np.array([1.0, 2.0]), 3.0)
        self.assertEqual(history.z_k_eval_hist.size, 0)
        self.assertEqual(history.Zn.size, 1)

    def test_add_forward_fills_current_incumbent(self):
        history = HistoryBuffer()
        history.init_incumbent(5.0, 5.0)
        history.add(np.array([1.0]), 9.0, z_true=9.0)
        history.add(np.array([2.0]), 7.0, z_true=7.0)

        np.testing.assert_array_equal(history.z_k_eval_hist, [5.0, 5.0])
        np.testing.assert_array_equal(history.z_k_true_hist, [5.0, 5.0])
        # Raw evaluations are unaffected by forward-filling.
        np.testing.assert_array_equal(history.Zn, [9.0, 7.0])

    def test_accept_incumbent_overwrites_only_the_last_entry(self):
        history = HistoryBuffer()
        history.init_incumbent(5.0, 5.0)
        history.add(np.array([1.0]), 9.0, z_true=9.0)
        history.add(np.array([2.0]), 7.0, z_true=7.0)

        history.accept_incumbent()

        np.testing.assert_array_equal(history.z_k_eval_hist, [5.0, 7.0])
        np.testing.assert_array_equal(history.z_k_true_hist, [5.0, 7.0])

        # A further observation forward-fills the *new* incumbent.
        history.add(np.array([3.0]), 4.0, z_true=4.0)
        np.testing.assert_array_equal(history.z_k_eval_hist, [5.0, 7.0, 7.0])

    def test_accept_incumbent_accepts_explicit_values(self):
        history = HistoryBuffer()
        history.init_incumbent(5.0, 5.0)
        history.add(np.array([1.0]), 9.0, z_true=9.0)

        history.accept_incumbent(z_eval=1.0, z_true=2.0)

        self.assertEqual(history.z_k_eval_hist[-1], 1.0)
        self.assertEqual(history.z_k_true_hist[-1], 2.0)

    def test_raw_history_preserves_duplicate_evaluations(self):
        history = HistoryBuffer()
        history.init_incumbent(0.0, 0.0)
        history.add(np.array([1.0]), 1.0, z_true=1.0)
        history.add(np.array([1.0]), 1.0, z_true=1.0)

        self.assertEqual(history.Zn.size, 2)
        self.assertEqual(history.Xn.shape[0], 2)


if __name__ == "__main__":
    unittest.main()
