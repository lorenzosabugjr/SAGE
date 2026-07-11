"""
Unit tests for estimators/sage.py.

Covers Milestones 1-2 of plans/plan_20260709_202428.md:
  - Milestone 1: SAGE API/state for `noise_bound`.
  - Milestone 2: LP assembly split between estimated and fixed noise modes.

Covers Milestones 1-4 of plans/plan_20260710_132820.md (see
plans/prd_20260710_132820.md for background):
  - Milestone 1: skip redundant noise-reseed when the seed step is already
    in-band around alpha.
  - Milestone 2: feedback-driven auxiliary sampling radius growth.
  - Milestone 3: bounding-box center point estimate with segment-clip
    projection.
  - Milestone 4: diameter-gated stopping criterion (stagnation branch no
    longer independently terminates refinement).

Run with: python -m unittest tests.test_sage
"""

import os
import sys
import types
import unittest
from unittest.mock import patch

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from estimators.sage import SAGE


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


class SageNoiseReseedBandSkipTests(unittest.TestCase):
    """Milestone 1 (plan_20260710_132820): skip redundant noise-reseed when
    the seed stencil is already in-band around the current alpha."""

    def _seeded(self, dim: int, noise_bound: float, init_step: float) -> tuple[SAGE, np.ndarray]:
        est = SAGE(constant_objective, dim=dim, noise_bound=noise_bound, init_step=init_step)
        x0 = np.zeros(dim)
        est._eval_and_record(x0)
        est._seed_cfd_stencil_if_singleton(x0)
        return est, x0

    def test_seed_step_in_band_skips_reseed(self):
        dim = 2
        noise_bound = 0.25
        # alpha = 2*sqrt(noise_bound) = 1.0 when H = gamma = 0; init_step is
        # chosen to land exactly on it, well inside [alpha/2, alpha*2].
        est, x0 = self._seeded(dim, noise_bound, init_step=1.0)
        hist_before = est.history.Zn.size
        self.assertEqual(hist_before, 1 + 2 * dim)

        est._maybe_noise_reseed(x0)

        self.assertEqual(est.history.Zn.size, hist_before)
        self.assertTrue(est._did_noise_reseed)

    def test_seed_step_out_of_band_triggers_reseed(self):
        dim = 2
        noise_bound = 0.25
        # init_step far below the [0.5, 2.0] band implied by alpha = 1.0.
        est, x0 = self._seeded(dim, noise_bound, init_step=1e-6)
        hist_before = est.history.Zn.size
        self.assertEqual(hist_before, 1 + 2 * dim)

        est._maybe_noise_reseed(x0)

        self.assertEqual(est.history.Zn.size, hist_before + 2 * dim)
        self.assertTrue(est._did_noise_reseed)

    def test_full_call_skips_reseed_when_in_band(self):
        est = SAGE(constant_objective, dim=2, noise_bound=0.25, init_step=1.0)
        with patch.object(SAGE, "_reseed_at_alpha") as mock_reseed:
            est(np.zeros(2))
        mock_reseed.assert_not_called()

    def test_full_call_reseeds_when_out_of_band(self):
        est = SAGE(constant_objective, dim=2, noise_bound=0.25, init_step=1e-6)
        with patch.object(SAGE, "_reseed_at_alpha", wraps=est._reseed_at_alpha) as mock_reseed:
            est(np.zeros(2))
        mock_reseed.assert_called_once()


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
        est.ns_est = 0.25  # forces the 2*sqrt(ns_est) fallback: alpha = 1.0

        est._aux_radius_growth = 1.0
        alpha, model_alpha = est._compute_aux_step()
        self.assertAlmostEqual(alpha, 1.0)
        self.assertAlmostEqual(model_alpha, 1.0)
        diath_ungrown = est.gdtset_diath

        est._aux_radius_growth = 3.0
        alpha, model_alpha = est._compute_aux_step()
        self.assertAlmostEqual(alpha, 3.0)
        self.assertAlmostEqual(model_alpha, 1.0)
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


class SageDiameterGatedStoppingTests(unittest.TestCase):
    """Milestone 4 (plan_20260710_132820) originally removed stagnation as a
    stopping signal entirely; it was later reinstated (see docs/theory.md
    Sec. 7 and docs/implementation.md Sec. 5) as a deliberate eval-cost
    control, tracking the *box-centered* estimate's stability
    (`_stable_count_final`) rather than the raw LP vertex's
    (`_stable_count`, still diagnostic-only, does not gate stopping). So
    stopping is gated by diameter-met, forced-stop, the 5*dim auxiliary cap,
    OR box-centered-estimate stagnation -- four independent triggers, not
    three."""

    def test_raw_vertex_stability_alone_does_not_stop_refinement(self):
        # _stable_count (the raw, solver-path-dependent LP vertex) is
        # diagnostic-only and must never gate stopping on its own --
        # distinct from _stable_count_final below, which does.
        est = SAGE(constant_objective, dim=2, init_step=1e-3)
        est._stable_count = 3
        est.gd_vm = 100.0
        est.gdtset_diath = 1.0
        est.gdt_est_frc = False
        est.aux_samples_count = 0
        self.assertFalse(est._should_stop_refinement())

    def test_stable_count_final_alone_stops_refinement(self):
        # _stable_count_final (the box-centered estimate) DOES independently
        # gate stopping -- this is the deliberate eval-cost-control
        # mechanism described in docs/theory.md Sec. 7.
        est = SAGE(constant_objective, dim=2, init_step=1e-3)
        est._stable_count_final = 3
        est.gd_vm = 100.0
        est.gdtset_diath = 1.0
        est.gdt_est_frc = False
        est.aux_samples_count = 0
        self.assertTrue(est._should_stop_refinement())

    def test_stops_once_diameter_meets_threshold_regardless_of_stability(self):
        est = SAGE(constant_objective, dim=2, init_step=1e-3)
        est._stable_count = 0
        est.gd_vm = 0.5
        est.gdtset_diath = 1.0
        est.gdt_est_frc = False
        est.aux_samples_count = 0
        self.assertTrue(est._should_stop_refinement())

    def test_stops_on_forced_flag_even_if_diameter_is_large(self):
        est = SAGE(constant_objective, dim=2, init_step=1e-3)
        est._stable_count = 0
        est.gd_vm = 100.0
        est.gdtset_diath = 1.0
        est.gdt_est_frc = True
        est.aux_samples_count = 0
        self.assertTrue(est._should_stop_refinement())

    def test_stops_at_aux_sample_cap_even_if_diameter_is_large_and_not_stable(self):
        dim = 2
        est = SAGE(constant_objective, dim=dim, init_step=1e-3)
        est._stable_count = 0
        est.gd_vm = 100.0
        est.gdtset_diath = 1.0
        est.gdt_est_frc = False
        est.aux_samples_count = 5.0 * dim
        self.assertTrue(est._should_stop_refinement())

    def test_pending_axis_probe_blocks_stop_even_at_cap(self):
        # A pending axis probe blocks all three of the diameter-met,
        # cap-reached, and stagnation triggers simultaneously (only the
        # forced-stop flag is unguarded).
        dim = 2
        est = SAGE(constant_objective, dim=dim, init_step=1e-3)
        est._stable_count_final = 3
        est.gd_vm = 0.1
        est.gdtset_diath = 1.0
        est.gdt_est_frc = False
        est.aux_samples_count = 5.0 * dim
        est._axis_probe_queue = [np.array([1.0, 0.0])]
        self.assertFalse(est._should_stop_refinement())


if __name__ == "__main__":
    unittest.main()
