"""
Unit tests for estimators/sage.py.

Covers Milestones 1-2 of plans/plan_20260709_202428.md:
  - Milestone 1: SAGE API/state for `noise_bound`.
  - Milestone 2: LP assembly split between estimated and fixed noise modes.

Covers Milestones 1-4 of plans/plan_20260710_132820.md (see
plans/prd_20260710_132820.md for background):
  - Milestone 2: axis-cycling auxiliary sampling radius growth.
  - Milestone 3: bounding-box center point estimate with segment-clip
    projection.

Covers the certificate-driven refinement redesign (2026-07-12):
  - estimate-mode noise self-calibration from seed second differences
    (_maybe_calibrate_noise), replacing the removed noise-reseed;
  - informed auxiliary radius sized to the stopping certificate
    (_compute_informed_radius) with the 1.5*init_step alpha floor;
  - second-difference pilot curvature guard (_consume_pilot_feedback);
  - relative-accuracy-certificate stopping with the 2*dim auxiliary cap
    (_should_stop_refinement), replacing the absolute diameter target and
    the stagnation stop.

Run with: python -m unittest tests.test_sage
"""

import os
import sys
import types
import unittest
from unittest.mock import patch

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from estimators.sage import SAGE, SageStopReason
from utils.history import HistoryBuffer


def constant_objective(_x):
    return 0.0


def quadratic_objective(x):
    x = np.asarray(x, dtype=float)
    return float(0.5 * x @ x)


def _seed_history(est: SAGE, dim: int, n_extra: int) -> None:
    rng = np.random.default_rng(0)
    X = rng.normal(size=(n_extra, dim)) * 0.1
    Z = rng.normal(size=(n_extra,))
    est.history.add_batch(X, Z)
    est._sync_history()


def _fake_linprog_result(ncols: int) -> types.SimpleNamespace:
    return types.SimpleNamespace(
        success=True,
        x=np.zeros(ncols),
        status=0,
        message="optimal",
        fun=0.0,
    )


def _fake_linprog_failure(ncols: int) -> types.SimpleNamespace:
    return types.SimpleNamespace(
        success=False,
        x=np.zeros(ncols),
        status=2,
        message="infeasible",
        fun=None,
    )


class SageNoiseBoundConstructionTests(unittest.TestCase):
    def test_accepts_none(self):
        est = SAGE(constant_objective, dim=2)
        self.assertIsNone(est.noise_bound)
        self.assertFalse(est.noise_bound_is_fixed)
        self.assertEqual(est.ns_est, 0.0)

    def test_accepts_zero(self):
        est = SAGE(constant_objective, dim=2, noise_bound=0.0)
        self.assertEqual(est.noise_bound, 0.0)
        self.assertTrue(est.noise_bound_is_fixed)
        self.assertEqual(est.ns_est, 0.0)

    def test_accepts_positive_value(self):
        est = SAGE(constant_objective, dim=2, noise_bound=0.25)
        self.assertEqual(est.noise_bound, 0.25)
        self.assertTrue(est.noise_bound_is_fixed)
        self.assertEqual(est.ns_est, 0.25)

    def test_rejects_negative(self):
        with self.assertRaises(ValueError):
            SAGE(constant_objective, dim=2, noise_bound=-0.1)

    def test_rejects_nan(self):
        with self.assertRaises(ValueError):
            SAGE(constant_objective, dim=2, noise_bound=float("nan"))

    def test_rejects_infinite(self):
        with self.assertRaises(ValueError):
            SAGE(constant_objective, dim=2, noise_bound=float("inf"))


class SageResetOnStepTests(unittest.TestCase):
    def test_fixed_ns_est_preserved_across_reset_on_step(self):
        est = SAGE(
            constant_objective,
            dim=2,
            noise_bound=0.3,
            reset_on_step=True,
            init_step=1e-3,
        )
        self.assertEqual(est.ns_est, 0.3)

        est(np.array([0.0, 0.0]))
        self.assertEqual(est.ns_est, 0.3)

        # Jumping to a new iterate triggers the reset_on_step history reset.
        est(np.array([1.0, 1.0]))
        self.assertEqual(est.ns_est, 0.3)

    def test_estimated_ns_est_resets_to_zero_across_reset_on_step(self):
        est = SAGE(
            constant_objective,
            dim=2,
            reset_on_step=True,
            init_step=1e-3,
        )
        est(np.array([0.0, 0.0]))
        est.ns_est = 5.0  # simulate a nonzero estimate carried from the LP
        est(np.array([1.0, 1.0]))
        self.assertEqual(est.ns_est, 0.0)


class SageLpDimensionTests(unittest.TestCase):
    """Milestone 2: monkeypatched linprog objective-length gate check."""

    def test_estimated_mode_objective_length_is_d_plus_3(self):
        dim = 3
        est = SAGE(constant_objective, dim=dim, init_step=1e-3)
        _seed_history(est, dim, dim + 2)

        captured = {}

        def fake_linprog(c, **kwargs):
            captured["c_len"] = len(c)
            return _fake_linprog_result(len(c))

        with patch("estimators.sage.cp.optimize.linprog", side_effect=fake_linprog):
            est._grad_est_lp(np.zeros(dim))

        self.assertEqual(captured["c_len"], dim + 3)

    def test_fixed_mode_objective_length_is_d_plus_2(self):
        dim = 3
        est = SAGE(constant_objective, dim=dim, noise_bound=0.1, init_step=1e-3)
        _seed_history(est, dim, dim + 2)

        captured = {}

        def fake_linprog(c, **kwargs):
            captured["c_len"] = len(c)
            return _fake_linprog_result(len(c))

        with patch("estimators.sage.cp.optimize.linprog", side_effect=fake_linprog):
            est._grad_est_lp(np.zeros(dim))

        self.assertEqual(captured["c_len"], dim + 2)


class SageFixedModeNsEstTests(unittest.TestCase):
    """Milestone 2: deterministic seeded-history ns_est invariant gate check."""

    def test_fixed_mode_keeps_ns_est_equal_to_noise_bound_after_lp(self):
        dim = 2
        noise_bound = 0.15

        est = SAGE(
            quadratic_objective,
            dim=dim,
            noise_bound=noise_bound,
            init_step=1e-2,
            quickmode=True,
        )

        self.assertEqual(est.ns_est, noise_bound)
        est(np.zeros(dim))
        self.assertEqual(est.ns_est, noise_bound)

        est._grad_est_lp(np.zeros(dim))
        self.assertEqual(est.ns_est, noise_bound)


class SageLpFailureHandlingTests(unittest.TestCase):
    """Milestone 3: fixed-mode LP failure raises RuntimeError; estimated mode does not."""

    def test_fixed_mode_raises_runtime_error_on_lp_failure(self):
        dim = 3
        noise_bound = 0.2
        est = SAGE(constant_objective, dim=dim, noise_bound=noise_bound, init_step=1e-3)
        _seed_history(est, dim, dim + 2)

        def fake_linprog(c, **kwargs):
            return _fake_linprog_failure(len(c))

        with patch("estimators.sage.cp.optimize.linprog", side_effect=fake_linprog):
            with self.assertRaises(RuntimeError) as ctx:
                est._grad_est_lp(np.zeros(dim))

        message = str(ctx.exception)
        self.assertIn("2", message)  # LP status
        self.assertIn("infeasible", message)  # LP message
        self.assertIn(str(noise_bound), message)  # fixed noise_bound

    def test_estimated_mode_does_not_raise_on_same_lp_failure(self):
        dim = 3
        est = SAGE(constant_objective, dim=dim, init_step=1e-3)
        _seed_history(est, dim, dim + 2)

        def fake_linprog(c, **kwargs):
            return _fake_linprog_failure(len(c))

        with patch("estimators.sage.cp.optimize.linprog", side_effect=fake_linprog):
            # Should not raise; matches current (pre-existing) estimated-mode behavior.
            est._grad_est_lp(np.zeros(dim))


class SageNoiseCalibrationTests(unittest.TestCase):
    """Estimate-mode noise self-calibration from the seed stencil's second
    differences (_maybe_calibrate_noise): the LP's estimated ns_est is a
    lower bound (it minimizes eps subject to feasibility), so SAGE
    recalibrates from D2 statistics and switches to fixed-bound mode."""

    def _stenciled(self, dim: int, values: dict, init_step: float = 1.0,
                   **kwargs) -> tuple[SAGE, np.ndarray]:
        """SAGE with a seed stencil whose z-values come from a lookup table
        keyed by the sampled point (rounded tuple)."""
        def fun(x):
            return values[tuple(np.round(np.asarray(x, dtype=float), 9))]
        est = SAGE(fun, dim=dim, init_step=init_step, **kwargs)
        x0 = np.zeros(dim)
        est._eval_and_record(x0)
        est._seed_cfd_stencil_if_singleton(x0)
        return est, x0

    def _values_for_d2(self, dim: int, d2: np.ndarray, h: float = 1.0) -> dict:
        """z-values giving exactly D2_i = d2[i]: z0 = 0, z(+h e_i) = 2*d2_i,
        z(-h e_i) = 0 (so |z+ + z- - 2 z0|/2 = d2_i)."""
        values = {tuple(np.zeros(dim)): 0.0}
        for i in range(dim):
            e = np.zeros(dim)
            e[i] = h
            values[tuple(np.round(e, 9))] = 2.0 * d2[i]
            values[tuple(np.round(-e, 9))] = 0.0
        return values

    def test_calibration_sets_conservative_fixed_bound(self):
        dim = 3
        d2 = np.array([0.3, 0.9, 0.6])
        est, x0 = self._stenciled(dim, self._values_for_d2(dim, d2))
        est.ns_est = 0.05  # simulate the LP's (lower-bound) estimate

        est._maybe_calibrate_noise(x0)

        expected = max(float(np.sqrt(2.0 * np.mean(d2 ** 2))),
                       float(np.max(d2)) / 1.5, 0.05)
        self.assertTrue(est.noise_bound_is_fixed)
        self.assertAlmostEqual(est.noise_bound, expected)
        self.assertAlmostEqual(est.ns_est, expected)

    def test_calibration_never_below_lp_estimate(self):
        dim = 2
        d2 = np.array([1e-6, 2e-6])  # tame draws
        est, x0 = self._stenciled(dim, self._values_for_d2(dim, d2))
        est.ns_est = 0.4  # LP already proves at least this much noise

        est._maybe_calibrate_noise(x0)

        self.assertAlmostEqual(est.noise_bound, 0.4)

    def test_calibration_skipped_when_bound_already_fixed(self):
        dim = 2
        d2 = np.array([0.3, 0.9])
        est, x0 = self._stenciled(dim, self._values_for_d2(dim, d2),
                                  noise_bound=0.25)

        est._maybe_calibrate_noise(x0)

        self.assertEqual(est.noise_bound, 0.25)

    def test_calibration_skipped_when_lp_found_no_noise(self):
        dim = 2
        d2 = np.array([0.3, 0.9])
        est, x0 = self._stenciled(dim, self._values_for_d2(dim, d2))
        est.ns_est = 0.0

        est._maybe_calibrate_noise(x0)

        self.assertFalse(est.noise_bound_is_fixed)

    def test_calibration_skipped_when_stencil_not_in_history(self):
        est = SAGE(constant_objective, dim=2, init_step=1.0)
        _seed_history(est, 2, 5)  # pre-seeded history, no stencil around x0
        est.ns_est = 0.1

        est._maybe_calibrate_noise(np.zeros(2))

        self.assertFalse(est.noise_bound_is_fixed)

    def test_full_call_calibrates_in_estimate_mode(self):
        rng = np.random.RandomState(3)

        def noisy(x):
            return rng.uniform(-0.5, 0.5)

        est = SAGE(noisy, dim=3, init_step=1.0)
        est(np.zeros(3))

        self.assertTrue(est.noise_bound_is_fixed)
        self.assertGreater(est.noise_bound, 0.0)

    def test_full_call_marks_attempted_and_fixed_on_noisy_stencil(self):
        # Milestone 2: the first (centered) gradient call must attempt
        # calibration and, for a genuinely noisy stencil, switch to
        # fixed-bound mode -- without changing that outcome.
        rng = np.random.RandomState(3)

        def noisy(x):
            return rng.uniform(-0.5, 0.5)

        est = SAGE(noisy, dim=3, init_step=1.0)
        self.assertFalse(est.calibration_attempted)
        self.assertFalse(est.calibration_fixed)

        est(np.zeros(3))

        self.assertTrue(est.calibration_attempted)
        self.assertTrue(est.calibration_fixed)
        self.assertTrue(est.noise_bound_is_fixed)

    def test_full_call_marks_attempted_without_fixing_on_noiseless_stencil(self):
        # Milestone 2: a zero-noise seed stencil still attempts calibration
        # (the gate is entered) but must intentionally leave estimated-noise
        # mode active -- the existing early-exit behavior is unchanged.
        est = SAGE(constant_objective, dim=3, init_step=1.0)

        est(np.zeros(3))

        self.assertTrue(est.calibration_attempted)
        self.assertFalse(est.calibration_fixed)
        self.assertFalse(est.noise_bound_is_fixed)

    def test_calibration_not_attempted_when_bound_supplied_up_front(self):
        # A supplied noise_bound short-circuits self-calibration entirely:
        # there is no estimate-mode gate to enter.
        est = SAGE(constant_objective, dim=2, init_step=1.0, noise_bound=0.1)

        est(np.zeros(2))

        self.assertFalse(est.calibration_attempted)
        self.assertFalse(est.calibration_fixed)


class SageInformedRadiusTests(unittest.TestCase):
    """Per-query informed auxiliary radius (_compute_informed_radius):
    r* = 4*eps*sqrt(dim)/(rel_tol*||g_seed||), clipped to [1.5, 16] times
    the informative scale max(init_step, 2*sqrt(eps)), consumed by
    _model_alpha with growth folded in."""

    def _est(self, dim=4, init_step=1.0, rel_tol=0.5):
        return SAGE(constant_objective, dim=dim, init_step=init_step,
                    rel_tol=rel_tol)

    def test_formula_and_clip_midrange(self):
        est = self._est()
        est.ns_est = 0.5
        est.gdt_est = np.array([1.0, 0.0, 0.0, 0.0])
        est._compute_informed_radius()
        # r = 4*0.5*sqrt(4)/(0.5*1) = 8, inside the clip band
        self.assertAlmostEqual(est._r_target, 8.0)

    def test_weak_gradient_clips_to_cap(self):
        est = self._est()
        est.ns_est = 0.5
        est.gdt_est = np.array([1e-4, 0.0, 0.0, 0.0])
        est._compute_informed_radius()
        # scale = max(1.0, 2*sqrt(0.5)); cap = 16*scale
        self.assertAlmostEqual(est._r_target, 16.0 * 2.0 * np.sqrt(0.5))

    def test_strong_gradient_clips_to_floor(self):
        est = self._est()
        est.ns_est = 0.5
        est.gdt_est = np.array([100.0, 0.0, 0.0, 0.0])
        est._compute_informed_radius()
        self.assertAlmostEqual(est._r_target, 1.5 * 2.0 * np.sqrt(0.5))

    def test_radius_scale_rescues_tiny_init_step(self):
        # A production run launched with the default init_step=1e-6 under
        # noise 1.0 must not pin the radius to ~1.5e-6: the informative
        # scale falls back to 2*sqrt(eps) (the old noise-reseed's absolute
        # jump target), keeping auxiliary samples at a noise-informative
        # distance.
        est = self._est(init_step=1e-6)
        est.ns_est = 0.5
        self.assertAlmostEqual(est._radius_scale(), 2.0 * np.sqrt(0.5))
        est.gdt_est = np.array([1e6, 0.0, 0.0, 0.0])  # noise-inflated seed g
        est._compute_informed_radius()
        self.assertAlmostEqual(est._r_target, 1.5 * 2.0 * np.sqrt(0.5))

    def test_radius_scale_is_init_step_when_noise_small(self):
        est = self._est(init_step=1.0)
        est.ns_est = 0.01  # 2*sqrt(eps) = 0.2 < init_step
        self.assertAlmostEqual(est._radius_scale(), 1.0)

    def test_not_computed_without_noise_or_gradient(self):
        est = self._est()
        est.ns_est = 0.0
        est.gdt_est = np.ones(4)
        est._compute_informed_radius()
        self.assertIsNone(est._r_target)

        est.ns_est = 0.5
        est.gdt_est = np.zeros(4)
        est._compute_informed_radius()
        self.assertIsNone(est._r_target)

    def test_model_alpha_uses_informed_radius_with_growth(self):
        est = self._est()
        est._r_target = 8.0
        est._aux_radius_growth = 2.0
        alpha, resolved = est._model_alpha()
        self.assertAlmostEqual(alpha, 16.0)
        self.assertTrue(resolved)

    def test_model_alpha_floored_without_informed_radius(self):
        est = self._est(init_step=1.0)
        est.ns_est = 1e-6  # bootstrap 2*sqrt(1e-6) = 2e-3 << floor of 1.5
        alpha, resolved = est._model_alpha()
        self.assertAlmostEqual(alpha, 1.5)
        self.assertFalse(resolved)

    def test_reset_per_query(self):
        est = self._est()
        est._r_target = 8.0
        est._pilot_shrinks = 2
        est._pending_pair = [(0, 1.0, np.zeros(4))]
        est._reset_query_state()
        self.assertIsNone(est._r_target)
        self.assertEqual(est._pilot_shrinks, 0)
        self.assertEqual(est._pending_pair, [])


class SagePilotGuardTests(unittest.TestCase):
    """Second-difference pilot guard (_consume_pilot_feedback): a completed
    axis pair whose D2 exceeds what noise alone can produce shrinks the
    informed radius globally and restarts the sweep."""

    def _est_with_pair(self, d2: float, r: float = 8.0, eps: float = 0.5):
        dim = 2
        est = SAGE(constant_objective, dim=dim, noise_bound=eps, init_step=1.0)
        x0 = np.zeros(dim)
        xp = np.array([r, 0.0])
        xm = np.array([-r, 0.0])
        # z-values giving exactly |z+ + z- - 2 z0|/2 = d2
        est.history.add(x0, 0.0)
        est.history.add(xp, 2.0 * d2)
        est.history.add(xm, 0.0)
        est._sync_history()
        est._r_target = r
        est._aux_radius_growth = 2.0
        est._used_probe_axes = {0}
        est._axis_probe_queue = [np.array([0.0, 1.0])]
        est._pending_pair = [(0, r, xp), (0, r, xm)]
        return est, x0

    def test_violating_pair_shrinks_radius_and_restarts_sweep(self):
        eps, r, d2 = 0.5, 8.0, 8.0  # threshold = 3*eps = 1.5 < d2
        est, x0 = self._est_with_pair(d2, r=r, eps=eps)

        est._consume_pilot_feedback(x0)

        self.assertAlmostEqual(est._r_target, r * np.sqrt(2.0 * eps / d2))
        self.assertEqual(est._aux_radius_growth, 1.0)
        self.assertEqual(est._axis_probe_queue, [])
        self.assertEqual(est._used_probe_axes, set())
        self.assertEqual(est._pilot_shrinks, 1)

    def test_noise_level_d2_does_not_trigger(self):
        eps = 0.5
        est, x0 = self._est_with_pair(d2=1.0, r=8.0, eps=eps)  # 1.0 <= 3*eps

        est._consume_pilot_feedback(x0)

        self.assertEqual(est._r_target, 8.0)
        self.assertEqual(est._pilot_shrinks, 0)

    def test_shrink_count_capped(self):
        est, x0 = self._est_with_pair(d2=8.0)
        est._pilot_shrinks = est._pilot_max_shrinks

        est._consume_pilot_feedback(x0)

        self.assertEqual(est._r_target, 8.0)

    def test_mismatched_pair_slides_without_consuming(self):
        est, x0 = self._est_with_pair(d2=8.0)
        # Second pending entry is a different axis: not a completed pair.
        est._pending_pair = [(0, 8.0, np.array([8.0, 0.0])),
                             (1, 8.0, np.array([0.0, 8.0]))]

        est._consume_pilot_feedback(x0)

        self.assertEqual(est._r_target, 8.0)
        self.assertEqual(len(est._pending_pair), 2)

    def test_shrink_never_goes_below_alpha_floor(self):
        # Enormous D2 would imply a radius below the alpha floor; a shrink
        # to the floor itself is applied at most once (a_new >= 0.9*a1
        # no-ops). Floor = 1.5 * max(init_step, 2*sqrt(eps)).
        est, x0 = self._est_with_pair(d2=1e6, r=8.0, eps=0.5)

        est._consume_pilot_feedback(x0)

        self.assertAlmostEqual(est._r_target, 1.5 * 2.0 * np.sqrt(0.5))


class SageAuxRadiusGrowthTests(unittest.TestCase):
    """Milestone 2 (plan_20260710_132820) as originally shipped, then
    superseded by the P1' axis-only refinement change: the refinement loop
    used to alternate diameter-direction probes with axis probes, growing
    the radius reactively when a diameter probe wasn't "informative" (per a
    contraction/rotation heuristic on _current_diameter_direction()). That
    was found to be structurally broken in approx/box diam_mode: gd_v is
    elementwise non-negative, so the "diameter direction" was always the
    same fixed all-positive-orthant diagonal, and cutting it never moves an
    axis-aligned bound -- so roughly half of every query's aux budget was
    spent on directions that could not inform the box metric or the
    box-center point estimate. _next_aux_direction now cycles proactively
    through every axis and only grows the radius (restarting a fresh sweep)
    once all axes have been probed at the current radius -- no diameter
    direction, no informativeness gate. Verified on the standard diagnostic:
    strict win over the old alternation on every point tested, same budget.
    """

    def test_growth_resets_per_query(self):
        est = SAGE(constant_objective, dim=2, init_step=1e-3)
        est._aux_radius_growth = 5.0
        est._reset_query_state()
        self.assertEqual(est._aux_radius_growth, 1.0)

    def test_compute_aux_step_scales_alpha_not_model_alpha(self):
        est = SAGE(constant_objective, dim=2, init_step=1e-3)
        est.hess_norm = 0.0
        est.hess_lipsc = 0.0
        # Forces the 2*sqrt(ns_est) fallback (= 1.0), which the alpha floor
        # then lifts to 1.5 * max(init_step, 2*sqrt(ns_est)) = 1.5.
        est.ns_est = 0.25
        expected = 1.5 * 1.0

        est._aux_radius_growth = 1.0
        alpha, model_alpha = est._compute_aux_step()
        self.assertAlmostEqual(alpha, expected)
        self.assertAlmostEqual(model_alpha, expected)
        diath_ungrown = est.gdtset_diath

        est._aux_radius_growth = 3.0
        alpha, model_alpha = est._compute_aux_step()
        self.assertAlmostEqual(alpha, 3.0 * expected)
        self.assertAlmostEqual(model_alpha, expected)
        # gdtset_diath must stay tied to the unscaled model alpha (non-goal:
        # do not change the gdtset_diath = 1.01*alpha formula).
        self.assertAlmostEqual(est.gdtset_diath, diath_ungrown)

    def test_next_aux_direction_cycles_every_axis_before_growing(self):
        est = SAGE(constant_objective, dim=2, init_step=1e-3)
        est.gd_v = np.array([5.0, 3.0])  # axis 0 wider than axis 1

        d1 = est._next_aux_direction()
        np.testing.assert_array_equal(d1, [1.0, 0.0])
        self.assertEqual(est._aux_radius_growth, 1.0)

        d2 = est._next_aux_direction()  # other half of axis 0's pair
        np.testing.assert_array_equal(d2, [-1.0, 0.0])

        # Axis 0 is used up but axis 1 remains: no growth yet.
        d3 = est._next_aux_direction()
        np.testing.assert_array_equal(d3, [0.0, 1.0])
        self.assertEqual(est._aux_radius_growth, 1.0)
        self.assertEqual(est._used_probe_axes, {0, 1})

        d4 = est._next_aux_direction()
        np.testing.assert_array_equal(d4, [0.0, -1.0])

        # Both axes now used at the current radius: growth triggers and the
        # sweep restarts from the widest axis.
        d5 = est._next_aux_direction()
        self.assertEqual(est._aux_radius_growth, 2.0)
        self.assertEqual(est._used_probe_axes, {0})
        np.testing.assert_array_equal(d5, [1.0, 0.0])

    def test_growth_not_triggered_while_axis_probes_remain(self):
        est = SAGE(constant_objective, dim=2, init_step=1e-3)
        est.gd_v = np.array([5.0, 5.0])

        est._next_aux_direction()

        self.assertEqual(est._aux_radius_growth, 1.0)
        self.assertEqual(len(est._axis_probe_queue), 1)
        self.assertEqual(est._used_probe_axes, {0})


class SagePointEstimateBoxCenterTests(unittest.TestCase):
    """Milestone 3 (plan_20260710_132820): bounding-box center point
    estimate with segment-clip projection."""

    def test_feasible_box_center_is_used_directly(self):
        est = SAGE(constant_objective, dim=2, init_step=1e-3)
        est.Al = np.array([[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0], [0.0, -1.0]])
        est.bl = np.array([10.0, 10.0, 10.0, 10.0])
        est.min_g = np.array([-1.0, -1.0])
        est.max_g = np.array([1.0, 1.0])
        est.gdt_est = np.array([0.9, 0.9])  # raw vertex, feasible

        est._update_point_estimate()

        np.testing.assert_allclose(est.gdt_est, np.array([0.0, 0.0]))

    def test_infeasible_box_center_is_projected_with_zero_slack_on_binding_row(self):
        est = SAGE(constant_objective, dim=2, init_step=1e-3)
        # x0 + x1 <= 1 is the binding constraint; the other two rows are
        # deliberately loose so their slack should stay non-negative.
        est.Al = np.array([[1.0, 1.0], [1.0, 0.0], [0.0, 1.0]])
        est.bl = np.array([1.0, 10.0, 10.0])
        est.min_g = np.array([0.0, 0.0])
        est.max_g = np.array([2.0, 2.0])  # box center = (1, 1): infeasible
        est.gdt_est = np.array([0.0, 0.0])  # raw vertex, feasible

        est._update_point_estimate()

        np.testing.assert_allclose(est.gdt_est, np.array([0.5, 0.5]))
        slack = est.bl - est.Al @ est.gdt_est
        self.assertAlmostEqual(slack[0], 0.0, places=9)
        self.assertGreaterEqual(slack[1], -1e-9)
        self.assertGreaterEqual(slack[2], -1e-9)

    def test_unbounded_box_axis_keeps_raw_vertex(self):
        est = SAGE(constant_objective, dim=2, init_step=1e-3)
        est.Al = np.array([[1.0, 0.0], [-1.0, 0.0]])
        est.bl = np.array([10.0, 10.0])
        est.min_g = np.array([-1.0, -np.inf])
        est.max_g = np.array([1.0, np.inf])
        raw = np.array([0.3, 7.0])
        est.gdt_est = raw.copy()

        est._update_point_estimate()

        np.testing.assert_allclose(est.gdt_est, raw)

    def test_recompute_at_wires_box_center_into_gdt_est(self):
        dim = 2
        est = SAGE(
            quadratic_objective,
            dim=dim,
            noise_bound=0.05,
            init_step=0.5,
            quickmode=True,
        )
        x0 = np.zeros(dim)
        est(x0)

        self.assertIsNotNone(est.Al)
        self.assertIsNotNone(est.bl)

        # Independently recompute the box bounds and expected point estimate
        # using the same public per-axis solves the implementation reuses,
        # to confirm gdt_est matches the box center (or its projection)
        # rather than an arbitrary raw vertex.
        min_g = np.empty(dim)
        max_g = np.empty(dim)
        for i in range(dim):
            direction = np.zeros(dim)
            direction[i] = 1.0
            max_g[i] = est._solve_direction_bound(direction, maximize=True)
            min_g[i] = est._solve_direction_bound(direction, maximize=False)
        self.assertTrue(np.all(np.isfinite(min_g)) and np.all(np.isfinite(max_g)))
        g_box = 0.5 * (max_g + min_g)

        slack = est.bl - est.Al @ g_box
        if np.all(slack >= -1e-9):
            np.testing.assert_allclose(est.gdt_est, g_box, atol=1e-6)
        else:
            # Projected point must be feasible (within tolerance).
            self.assertTrue(np.all(est.bl - est.Al @ est.gdt_est >= -1e-6))


class SageCertificateStoppingTests(unittest.TestCase):
    """Relative-accuracy-certificate stopping (_should_stop_refinement):
    stop when gd_vm < rel_tol * ||gdt_est|| (or below the absolute
    gdtset_diaid floor for the noiseless/near-optimum regime), on the
    forced flag, or at the 2*dim auxiliary cap. The old absolute
    1.01*alpha* diameter target and the box-centered stagnation stop are
    both gone; raw LP-vertex stability (_stable_count) stays
    diagnostic-only."""

    def _est(self, dim=2):
        est = SAGE(constant_objective, dim=dim, init_step=1e-3)
        est.gdt_est_frc = False
        est.aux_samples_count = 0
        return est

    def test_certified_relative_diameter_stops(self):
        est = self._est()
        est.gdt_est = np.array([1.0, 0.0])
        est.gd_vm = 0.4  # < rel_tol (0.5) * ||g|| (1.0)
        self.assertTrue(est._should_stop_refinement())

    def test_uncertified_relative_diameter_does_not_stop(self):
        est = self._est()
        est.gdt_est = np.array([1.0, 0.0])
        est.gd_vm = 0.6  # > rel_tol * ||g||, > gdtset_diaid
        self.assertFalse(est._should_stop_refinement())

    def test_absolute_floor_stops_when_noiseless_and_gradient_tiny(self):
        est = self._est()
        est.ns_est = 0.0
        est.gdt_est = np.zeros(2)  # relative target degenerate
        est.gd_vm = 0.01  # < gdtset_diaid = 0.05
        self.assertTrue(est._should_stop_refinement())

    def test_absolute_floor_does_not_apply_under_noise(self):
        # Near-zero-gradient noisy points are where refinement pays most;
        # an absolute floor was measured to cut them off 10x short.
        est = self._est()
        est.ns_est = 0.5
        est.gdt_est = np.zeros(2)
        est.gd_vm = 0.01
        self.assertFalse(est._should_stop_refinement())

    def test_raw_vertex_stability_alone_does_not_stop_refinement(self):
        est = self._est()
        est._stable_count = 10
        est.gdt_est = np.array([1.0, 0.0])
        est.gd_vm = 100.0
        self.assertFalse(est._should_stop_refinement())

    def test_stops_on_forced_flag_even_if_uncertified(self):
        est = self._est()
        est.gdt_est = np.array([1.0, 0.0])
        est.gd_vm = 100.0
        est.gdt_est_frc = True
        self.assertTrue(est._should_stop_refinement())

    def test_stops_at_two_dim_aux_cap_even_if_uncertified(self):
        dim = 2
        est = self._est(dim)
        est.gdt_est = np.array([1.0, 0.0])
        est.gd_vm = 100.0
        est.aux_samples_count = 2.0 * dim
        self.assertTrue(est._should_stop_refinement())

    def test_infinite_diameter_never_certifies(self):
        est = self._est()
        est.gdt_est = np.array([1e30, 0.0])
        est.gd_vm = np.inf
        self.assertFalse(est._should_stop_refinement())

    def test_pending_axis_probe_blocks_stop_even_at_cap(self):
        # A pending axis probe blocks both the certificate and cap triggers
        # (only the forced-stop flag is unguarded).
        dim = 2
        est = self._est(dim)
        est.gdt_est = np.array([1.0, 0.0])
        est.gd_vm = 0.1
        est.aux_samples_count = 2.0 * dim
        est._axis_probe_queue = [np.array([1.0, 0.0])]
        self.assertFalse(est._should_stop_refinement())


class SageCallDiagnosticsTests(unittest.TestCase):
    """Milestone 4 (plan_20260712_170450.md): one compact SageCallDiagnostic
    row per public __call__ invocation, covering a zero-auxiliary return, an
    auxiliary-refined return, and budget exhaustion."""

    def test_zero_auxiliary_call_records_one_aligned_row(self):
        # A noiseless quadratic certifies immediately at the seed CFD
        # stencil: no auxiliary evaluations are needed.
        est = SAGE(quadratic_objective, dim=2, init_step=0.5)
        x0 = np.array([5.0, 5.0])

        est(x0)

        self.assertEqual(len(est.call_diagnostics), 1)
        d = est.call_diagnostics[0]
        self.assertEqual(d.eval_index, 0)
        self.assertEqual(d.hist_size, 0)
        self.assertEqual(d.n_aux, 0)
        self.assertGreater(d.n_neighbors, 0)
        self.assertIn(
            d.stop_reason,
            (SageStopReason.RELATIVE_CRITERION, SageStopReason.NOISELESS_FLOOR),
        )
        # The diagnostic row must reflect the estimator's actual
        # calibration state after the call, whatever the LP concluded.
        self.assertTrue(d.calibration_attempted)
        self.assertEqual(d.calibration_attempted, est.calibration_attempted)
        self.assertEqual(d.calibration_fixed, est.calibration_fixed)

    def test_auxiliary_refined_call_records_nonzero_aux(self):
        # A pure-noise objective gives no exploitable gradient signal at the
        # seed stencil, forcing the refinement loop to run.
        rng = np.random.RandomState(3)

        def noisy(_x):
            return rng.uniform(-0.5, 0.5)

        est = SAGE(noisy, dim=2, init_step=1.0)

        est(np.zeros(2))

        self.assertEqual(len(est.call_diagnostics), 1)
        d = est.call_diagnostics[0]
        self.assertEqual(d.eval_index, 0)
        self.assertEqual(d.hist_size, 0)
        self.assertGreater(d.n_aux, 0)
        self.assertEqual(d.n_aux, int(est.hist_aux_samples[-1]))
        self.assertIn(
            d.stop_reason,
            (
                SageStopReason.RELATIVE_CRITERION,
                SageStopReason.NOISELESS_FLOOR,
                SageStopReason.AUXILIARY_CAP,
                SageStopReason.NO_AUX_DIRECTION,
            ),
        )
        self.assertTrue(d.calibration_attempted)

    def test_budget_exhausted_call_records_one_row_and_reraises(self):
        history = HistoryBuffer()
        budget = 3

        def fun(x):
            if history.Zn.size >= budget:
                raise StopIteration("Budget exhausted")
            z = float(np.sum(x))
            history.add(x, z)
            return z

        est = SAGE(fun, dim=2, init_step=0.5, history=history)

        with self.assertRaises(StopIteration):
            est(np.zeros(2))

        self.assertEqual(len(est.call_diagnostics), 1)
        d = est.call_diagnostics[0]
        self.assertEqual(d.eval_index, 0)
        self.assertEqual(d.hist_size, 0)
        self.assertEqual(d.stop_reason, SageStopReason.BUDGET_EXHAUSTION)

    def test_diagnostic_rows_align_one_per_call_with_call_start_state(self):
        est = SAGE(quadratic_objective, dim=2, init_step=0.5)
        x0 = np.array([5.0, 5.0])

        est(x0)
        n_hist_after_first = est.history.Zn.size
        est(x0, force=True)

        self.assertEqual(len(est.call_diagnostics), 2)
        self.assertEqual(est.call_diagnostics[0].eval_index, 0)
        self.assertEqual(est.call_diagnostics[0].hist_size, 0)
        self.assertEqual(est.call_diagnostics[1].eval_index, n_hist_after_first)
        self.assertEqual(est.call_diagnostics[1].hist_size, n_hist_after_first)


class SageWarmVsColdHistoryFlowTests(unittest.TestCase):
    """Milestone 5 (plan_20260712_170450.md): history-flow regression
    comparing a warm second query -- reusing an estimator's own
    accumulated history -- against a freshly constructed cold query at the
    same point, without changing SAGE's selector, stopping rule, or
    numerical settings. This checks that history flows into the next call;
    it intentionally does not assert a universal evaluation-count advantage
    from a single synthetic instance (see the PRD's Risks/Non-Goals)."""

    def test_warm_query_call_start_history_exceeds_cold_query(self):
        rng = np.random.RandomState(7)

        def noisy_quadratic(x):
            x = np.asarray(x, dtype=float)
            return float(0.5 * x @ x) + rng.uniform(-0.05, 0.05)

        dim = 2
        x1 = np.array([3.0, 3.0])
        x2 = np.array([3.5, 2.5])

        warm = SAGE(noisy_quadratic, dim=dim, init_step=0.1)
        warm(x1)  # seeds history with the first query's stencil/aux samples
        hist_before_warm_second_call = warm.history.Zn.size
        warm(x2)

        cold = SAGE(noisy_quadratic, dim=dim, init_step=0.1)
        cold(x2)

        self.assertEqual(len(warm.call_diagnostics), 2)
        self.assertEqual(len(cold.call_diagnostics), 1)

        warm_second_call = warm.call_diagnostics[1]
        cold_call = cold.call_diagnostics[0]

        # History-flow check only: the warm call starts with the first
        # query's accumulated samples available; the cold call starts from
        # nothing. Neither call is asserted to use fewer evaluations.
        self.assertEqual(warm_second_call.hist_size, hist_before_warm_second_call)
        self.assertGreater(warm_second_call.hist_size, cold_call.hist_size)
        self.assertEqual(cold_call.hist_size, 0)


if __name__ == "__main__":
    unittest.main()
