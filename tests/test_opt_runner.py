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


class SageCenteredStartTests(unittest.TestCase):
    """Milestone 1: SAGE must keep the initial iterate at its own stencil
    center, not reassign it to the best-valued stencil sample."""

    def _make_sage_trial(self, noise_param: float, dims: int = 2, maxevals: int = 30):
        return OptimizationTrial(
            problem_name="least-squares",
            grad_est_name="sage",
            maxevals=maxevals,
            dims=dims,
            randseed=1,
            noise_type=NoiseType.UNIFORM,
            noise_param=noise_param,
        )

    def _assert_centered_start(self, trial):
        np.testing.assert_array_equal(trial.X_start, trial.X_initial)
        self.assertEqual(trial.Z_start_eval, trial.Z_initial_eval)
        self.assertEqual(trial.Z_start_true, trial.Z_initial_true)
        # 1 center + the 2*D-point CFD seed stencil, all charged to budget.
        self.assertEqual(trial.history.Zn.size, 1 + 2 * trial.dims)
        # The optimizer's first gradient query is at the stencil center.
        np.testing.assert_array_equal(trial.solver.x_k, trial.X_initial)

    def test_noiseless_sage_trial_starts_centered(self):
        trial = self._make_sage_trial(noise_param=0.0)
        self._assert_centered_start(trial)

    def test_noisy_sage_trial_starts_centered(self):
        trial = self._make_sage_trial(noise_param=2.0)
        self._assert_centered_start(trial)


class SageIncumbentHistoryTests(unittest.TestCase):
    """Milestone 3: initial stencil, auxiliary, and rejected evaluations
    must forward-fill the incumbent; only an accepted evaluation moves it,
    on its own evaluation index."""

    def test_stencil_and_rejections_forward_fill_center_until_accepted(self):
        trial = OptimizationTrial(
            problem_name="least-squares",
            grad_est_name="sage",
            maxevals=80,
            dims=2,
            randseed=1,
            noise_type=NoiseType.UNIFORM,
            noise_param=0.0,
            stepsize=1e8,  # forces at least one line-search rejection
        )
        trial.run()

        hist_eval = trial.history.z_k_eval_hist
        center = trial.Z_initial_eval

        # Raw history retains every evaluated point, one-to-one with the
        # incumbent-history record.
        self.assertEqual(trial.history.Zn.size, hist_eval.size)
        self.assertEqual(trial.history.Xn.shape[0], trial.history.Zn.size)
        # SAGE's seed stencil alone forces more evaluations than just the
        # center before the optimizer's first line-search decision.
        self.assertGreater(trial.history.Zn.size, 1 + 2 * trial.dims)

        changed = np.flatnonzero(hist_eval != center)
        self.assertGreater(changed.size, 0, "expected at least one accepted step")
        first_change = int(changed[0])
        self.assertGreater(first_change, 0)

        # Every observation before the first accepted evaluation -- the
        # initial stencil, any auxiliary refinement, and any rejected
        # line-search trials -- forward-fills the original center.
        np.testing.assert_array_equal(hist_eval[:first_change], center)
        # The accepted evaluation records the new incumbent on its own
        # index, matching its own raw evaluation.
        self.assertEqual(hist_eval[first_change], trial.history.Zn[first_change])


class BudgetExhaustionAfterAcceptanceTests(unittest.TestCase):
    """Milestone 3: budget exhaustion immediately after an accepted
    evaluation must not lose that accepted improvement from the history."""

    def test_accepted_improvement_survives_budget_exhaustion(self):
        def make_trial(maxevals):
            return OptimizationTrial(
                problem_name="least-squares",
                grad_est_name="truth",
                maxevals=maxevals,
                dims=2,
                randseed=1,
                stepsize=1e8,  # forces at least one rejection before acceptance
            )

        probe = make_trial(maxevals=50)
        probe.run()
        hist_eval = probe.history.z_k_eval_hist
        center = probe.Z_initial_eval
        changed = np.flatnonzero(hist_eval != center)
        self.assertGreater(changed.size, 0, "test setup must force an acceptance")
        accept_idx = int(changed[0])

        # Deterministic (fixed seed, noiseless gradient): cut the budget
        # exactly at the accepted evaluation.
        trial = make_trial(maxevals=accept_idx + 1)
        trial.run()

        self.assertEqual(trial.history.Zn.size, accept_idx + 1)
        self.assertEqual(trial.history.z_k_eval_hist[-1], trial.history.Zn[accept_idx])
        self.assertNotEqual(trial.history.z_k_eval_hist[-1], center)


class SagePersistentHistoryTests(unittest.TestCase):
    """Milestone 5: exercise persistent history without changing SAGE.
    Line-search rejections and acceptances land in the shared HistoryBuffer
    through lightweight updates, and the next SAGE gradient call detects the
    enlarged history and rebuilds its LP there -- with reset_on_step false
    throughout, matching the production optimization config."""

    def test_next_sage_call_sees_rejected_and_accepted_line_search_points(self):
        trial = OptimizationTrial(
            problem_name="least-squares",
            grad_est_name="sage",
            maxevals=80,
            dims=2,
            randseed=1,
            noise_type=NoiseType.UNIFORM,
            noise_param=0.0,
            stepsize=1e8,  # forces at least one line-search rejection
        )
        self.assertFalse(trial.estimator.reset_on_step)

        # SAGE's centered seed stencil (Milestone 1) is evaluated during
        # construction, before any solver step runs.
        seed_stencil_size = 1 + 2 * trial.dims
        self.assertEqual(trial.history.Zn.size, seed_stencil_size)

        trial.run()

        diagnostics = trial.estimator.call_diagnostics
        self.assertGreaterEqual(len(diagnostics), 2, "expected at least two SAGE gradient calls")

        # First public __call__ (the first solver.step()) starts with
        # exactly the already-seeded stencil history.
        self.assertEqual(diagnostics[0].hist_size, seed_stencil_size)

        # A later call must see a strictly larger call-start history than
        # the first: the forced rejection(s) and the eventual acceptance
        # grew the shared buffer in between, and SAGE's diagnostics report
        # that accumulated history at call start.
        self.assertGreater(diagnostics[1].hist_size, diagnostics[0].hist_size)
        # reset_on_step is false, so the call-start history size coincides
        # with the trial-wide evaluation index.
        self.assertEqual(diagnostics[1].eval_index, diagnostics[1].hist_size)

        # Those raw rejected/accepted samples remain available to the next
        # call's selector -- not discarded between calls.
        self.assertEqual(trial.estimator.Xn.shape[0], trial.history.Zn.size)
        self.assertEqual(trial.estimator.history.Zn.size, trial.history.Zn.size)


if __name__ == "__main__":
    unittest.main()
