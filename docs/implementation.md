# Implementation Notes

This document ties the theory in the paper (https://arxiv.org/abs/2508.19400) to the current implementation in `estimators/sage.py`.

## 1. State and data structures

`SAGE` maintains the full sample history:

- `Xn`: evaluated points, shape `(N, D)`
- `Zn`: function values, shape `(N,)`
- `gdt_est`: current gradient estimate
- `hess_norm`: estimated Hessian norm `H_i`
- `hess_lipsc`: estimated Hessian Lipschitz constant `gamma_H`
- `ns_est`: active noise bound `eps`. In estimated mode (`noise_bound=None`) this is the
  latest LP-estimated value until the seed-stencil self-calibration switches the estimator
  to fixed mode (see Section 5); in fixed mode it is pinned to `noise_bound` and never
  overwritten by the LP.
- `noise_bound`: constructor value (`None` for estimated mode or a fixed float `>= 0`),
  replaced by the calibrated bound once estimate-mode self-calibration runs.
- `noise_bound_is_fixed`: `True` when `noise_bound` was supplied or calibration has run.
- `rel_tol`: relative-accuracy certificate target (constructor kwarg, default 0.5); the
  main cost/accuracy knob (see Section 5's stopping criterion)
- `gdtset_diaid`: absolute ideal diameter, used as a stopping floor in the noiseless
  regime only
- `gdtset_diath`: legacy absolute threshold `1.01*alpha*`; still computed for diagnostics
  but no longer gates stopping
- `aux_samples_count`: number of refinement samples added in the current call, capped at
  `2*D`
- `init_step`: step size for the initial simplex when history has 0 or 1 samples
- `min_g`, `max_g`: per-axis bounding-box bounds cached by `_calc_diam_approx` and reused
  by the point-estimate step (see Section 4) instead of re-solving the same LPs
- `gd_v`, `gd_vm`: per-axis width vector and its norm (the diameter used by the
  certificate)
- `_r_target`: per-query informed auxiliary radius (see Section 5); `None` until the seed
  LP has run
- `_aux_radius_growth`: per-call multiplier applied on top of the informed/modeled radius
  when a full axis sweep completes without certifying; reset to `1.0` each call
- `_pending_pair`, `_pilot_shrinks`: state for the pilot second-difference curvature
  guard (see Section 5)
- `_stable_count`: diagnostic count of consecutive LP solves where the raw point estimate
  didn't move; retained for diagnostics only and no longer gates stopping (see Section 5)

## 2. LP construction

SAGE builds one of two LP layouts depending on whether `noise_bound` was supplied.

### Estimated mode (`noise_bound=None`, default)

For each neighbor `x^(j)`, the code builds two inequality rows (slab constraints)
including the noise term `eps` as a decision variable:

```
[-u_ij, -0.5*mu_ij, -1/6*mu_ij^2, -2/mu_ij] * [g, H_i, gamma_H, eps]^T <= -g~_ij
[ u_ij, -0.5*mu_ij, -1/6*mu_ij^2, -2/mu_ij] * [g, H_i, gamma_H, eps]^T <=  g~_ij
```

Non-negativity is enforced via extra rows:

```
H_i >= 0, gamma_H >= 0, eps >= 0
```

The LP objective matches the theory:

```
min H_i + gamma_H + eps
```

After the LP solves, `ns_est` is set to the solved `eps` value.

### Fixed mode (`noise_bound` set)

The `eps` column is dropped from the variable list; its known contribution (the `eps`
column is always `-2/mu_ij`) is folded into the RHS instead, using the fixed
`noise_bound`:

```
[-u_ij, -0.5*mu_ij, -1/6*mu_ij^2] * [g, H_i, gamma_H]^T <= -g~_ij + 2*noise_bound/mu_ij
[ u_ij, -0.5*mu_ij, -1/6*mu_ij^2] * [g, H_i, gamma_H]^T <=  g~_ij + 2*noise_bound/mu_ij
```

Non-negativity uses two rows instead of three (`H_i >= 0, gamma_H >= 0`), and the
objective drops the `eps` term:

```
min H_i + gamma_H
```

This makes the fixed-mode LP `D + 2` columns wide, versus `D + 3` in estimated mode.
`ns_est` is not touched by the LP result in this mode; it stays equal to `noise_bound`.
If the fixed-mode LP does not solve successfully (`res.success is False`), SAGE raises a
`RuntimeError` including the LP status, message, and the fixed `noise_bound`. Estimated
mode keeps its prior behavior of not raising in that case.

The code uses `scipy.optimize.linprog` to solve the LP in both modes.

## 3. Quickmode (filtered neighbors)

When `quickmode=True` and the history is large, SAGE uses a subset of samples:

- Compute the optimal radius `alpha` from the cubic:
  ```
  (1/3) * gamma_H * alpha^3 + (1/2) * H_i * alpha^2 - 2*eps = 0
  ```
- Select the `5*D` samples that minimize:
  ```
  | ||x^(j) - x^(i)||^2 - alpha^2 |
  ```

This reduces LP size while keeping the most informative samples for the current noise level.

## 4. Gradient set and diameter

After solving the LP, the code constructs `A2, b2` for the diameter optimization over two gradients:

```
g1, g2 in G~^(i)  ->  A2 * [g1, g2] <= b2
```

The diameter is approximated by a non-convex optimization using SLSQP. This is a local solve and is not guaranteed to find the global maximum diameter, but works well in practice.

### Fast approximate mode

When `diam_mode="approx"` (default when `quickmode=True`), SAGE computes an axis-aligned
bounding box by solving `2*D` LPs for the min/max of each coordinate. The resulting
diagonal length is an upper bound on the true diameter. This is faster than SLSQP but
can be conservative (may trigger extra refinement).

### Point estimate: bounding-box center

The LP objective (`min H_i + gamma_H [+ eps]`) never costs `g` itself, so once noise makes
`H_i = gamma_H = 0` (and `eps = 0` in estimated mode) LP-feasible, any `g` inside the
polytope is equally optimal — the raw vertex `linprog` returns is solver-path-dependent and
can be an arbitrary point (often exactly zero). `_update_point_estimate` replaces it with a
more representative point after the LP and diameter solve:

1. Reuse the per-axis `min_g`/`max_g` bounds already computed by `_calc_diam_approx` (or
   solve them directly when `diam_mode == "exact"`, which does not compute them as a
   byproduct) and take the box center `g_box = 0.5*(min_g + max_g)`.
2. If `g_box` is feasible (`Al @ g_box <= bl`, within a small numerical tolerance), use it
   directly as `gdt_est`.
3. Otherwise, segment-clip from the raw LP-solved vertex `g_v` (guaranteed feasible) toward
   `g_box`: for each row `i` with `Al_i @ (g_box - g_v) > 0`, compute
   `t_i = (bl_i - Al_i @ g_v) / (Al_i @ (g_box - g_v))`, take `t* = min(1, min_i t_i)`, and
   set `gdt_est = g_v + t* * (g_box - g_v)` — the nearest point to the box center still on
   the polytope.

This keys off `Al`/`bl` directly, so it applies uniformly regardless of `diam_mode`. It is
not a Chebyshev center (deliberately out of scope); it is a cheap reuse of bounds already
computed for the diameter.

## 5. Active sampling loop

For each call:

1. Solve the LP and compute the diameter.
2. If refinement should stop (see below), return `gdt_est`.
3. Otherwise, sample along a queued axis probe (see below):
   ```
   x_new = x + alpha * d_hat
   ```
4. Update history and repeat.

### Noise self-calibration (estimate mode)

The LP's estimated `ns_est` **minimizes** `eps` subject to feasibility, so it is a lower
bound on the noise, not an estimator (measured at 0.3–0.7x the true bound on the standard
benchmark). Everything derived from it — the informed radius, the pilot-guard threshold,
the certificate tightness — would inherit that bias. So on the first call in estimate
mode, once the seed LP has detected noise (`ns_est > 1e-9`), `_maybe_calibrate_noise`
re-estimates the bound from the seed stencil's own per-axis second differences
`D2_i = |z(x + h e_i) + z(x - h e_i) - 2 z(x)| / 2`: for noise-only D2 (uniform noise,
locally-linear `f` at scale `h`), `sd(D2) ≈ 0.7*eps` and `max(D2) <= 2*eps`, giving

```
eps_cal = max( sqrt(2*mean(D2^2)),  max(D2)/1.5,  ns_est )
```

(measured nearly unbiased: 0.45–0.50 vs a true bound of 0.5 on the log-type problems).
SAGE then **switches to fixed-bound mode** with `eps_cal` — one extra LP re-solve, no new
evaluations — so the LP, radius, guard, and certificate all use it consistently. On
functions with real curvature or kinks at scale `h` the D2s also contain curvature signal
and `eps_cal` overshoots (least-squares/lasso: ~20x); this errs conservative (looser
certificate, larger informed radius that the pilot guard walks back) and those problem
types certify at the seed anyway. Calibration runs once per estimator lifetime; the old
noise-aware **reseed** (`_maybe_noise_reseed`/`_reseed_at_alpha`, a second full stencil at
`alpha`) was removed outright: measured on the production configs it burned `2*D`
evaluations to place samples at a *smaller* radius than the seed (e.g. `alpha = 0.045`
vs `h = 0.1` at noise 1e-3), which degraded least-squares accuracy ~11x via the quickmode
filter crowding out the better seed samples.

### Auxiliary sampling radius

- If `ns_est` is near zero, SAGE uses `alpha = init_step` and resets the threshold to
  `gdtset_diaid`.
- **Informed one-shot radius** (`_compute_informed_radius`): once the (calibrated) seed LP
  has produced `gdt_est` and `ns_est`, the per-query radius is sized directly to the
  stopping certificate. A full axis sweep at radius `r` tightens each axis width to the
  information floor `~4*eps/r`, i.e. a box diameter `~sqrt(D)*4*eps/r`; setting that equal
  to the certificate target `rel_tol*||g||` gives
  `r* = 4*eps*sqrt(D) / (rel_tol*||g_seed||)`, clipped to `[1.5, 16]` times the
  **informative scale** `max(init_step, 2*sqrt(eps))` (`_radius_scale`).
  Strong-gradient points get a small `r*` (their certificate is nearly met already);
  weak-gradient points get a large one — and their curvature-bias tolerance also scales
  with `rel_tol*||g||`, which is exactly what the pilot guard (below) checks.
- When the informed radius is not yet available (during the seed LP itself), the
  cubic-root optimal radius (or the `2*sqrt(eps)` bootstrap when curvature is unresolved)
  is used as before. Either way the radius is **floored at 1.5x the informative scale**:
  the seed stencil already extracted the information available at `init_step`, sampling
  inside that radius only adds strictly noisier constraints — and under noise no sample
  inside `~2*sqrt(eps)` is informative regardless of `init_step`. Anchoring the floor on
  `init_step` alone was measured catastrophic when a production run was launched with the
  default `init_step=1e-6` under noise 1.0 (every auxiliary sample pinned to `1.5e-6`,
  mean rel err ~1e5); the `2*sqrt(eps)` component restores the absolute noise-derived
  jump the removed reseed used to provide, and SAGE now recovers from a mis-set
  `init_step` at the cost of the wasted seed stencil (measured at `init_step=1e-6`,
  noise 1.0: least-squares 1.8e-4, l2-log-reg 2.6 — still beating NMXFD at its properly
  tuned step).
- **Axis-only cycling with exhaustion-driven growth** (`_next_aux_direction`): each call
  proactively queues a symmetric `+e_i`/`-e_i` pair along the widest not-yet-probed
  coordinate (ranked by `|gd_v_i|`). Only once every axis has been probed at the current
  radius does it scale the radius by a geometric growth factor
  (`_aux_radius_growth *= factor`) and restart a fresh sweep. Growth resets to `1.0` at
  the start of each call. An earlier version alternated axis probes with probes along the
  composite diameter direction; that was removed after finding it structurally inert in
  `diam_mode="approx"` — see `docs/theory.md` §7.

### Pilot curvature guard

Each completed symmetric axis pair at radius `r` yields a free second difference
`D2 = |z+ + z- - 2 z0|/2` against the query-point sample. Noise alone can push `D2` to at
most `2*eps`, so `D2 > 3*eps` **proves** real curvature at scale `r` on that axis — the
regime where the LP (which structurally prefers `H = gamma = 0` whenever feasible) folds
the residual into a biased gradient instead of raising `H`. `_consume_pilot_feedback` then
shrinks the informed radius globally to `r*sqrt(2*eps/D2)` (where the quadratic residual
matches the noise slab) and restarts the sweep, so the first pair of each sweep acts as a
pilot and at most one pair is committed at a too-large radius; at most 3 shrinks per
query. This matters for *scale-dependent* curvature that a small-step Hessian misses
entirely (e.g. an l2-log-reg point measuring `H = 1e-3*I` at step 1e-4 but with a logistic
transition inside `r = 8`: `D2 = 7.2` vs noise-max 1.0 — sweeping at 16 without the guard
gave rel err 20.4, with it 4.6). A per-axis backoff (re-sampling only the violating axis
at the small radius) was tried and rejected: the replacement samples fall below the LP
band filter (`search_alpha/10`) and never reach the LP while the poisoned far samples
remain — measured strictly worse than no guard.

### Stopping criterion

`_should_stop_refinement` returns `True` when any of:

- the **relative-accuracy certificate** is met — `gd_vm < rel_tol * ||gdt_est||`
  (constructor kwarg `rel_tol`, default 0.5), or in the noiseless regime only
  (`ns_est <= 1e-9`) the absolute floor `gd_vm < gdtset_diaid` — and no axis probe is
  pending, or
- a forced stop is requested (`gdt_est_frc`), or
- the auxiliary sample cap is reached (`aux_samples_count >= 2.0*D`) and no axis probe is
  pending.

The certificate is the load-bearing change: the box then brackets the gradient to
`~rel_tol/2` **relative** accuracy, which is the quantity consumers of the estimate care
about. Strong-gradient points certify at the seed stencil itself (0 auxiliary samples,
`2*D + 1` evaluations — CFD cost with a certificate); weak-gradient points refine until
certified or capped, which is exactly where refinement pays (their per-eval information
gain stays positive all the way to the cap on the standard diagnostic).

Two earlier criteria were removed after being measured mistimed in both directions on the
production configs: the absolute diameter target `gd_vm < gdtset_diath = 1.01*alpha*`
(derived from `H_i`/`gamma_H`/`eps` LP estimates that are usually stuck at a degenerate
fallback — an unreachable moving target) and the box-centered stagnation stop
(`_stable_count_final >= 3`, which cut off weak-gradient points mid-improvement at ~3x
worse error while spending ~40 useless evaluations on points that should stop at the
seed). The absolute floor under noise was also measured harmful (l1-log-reg at noise
1e-3: median rel err 1.33 with it vs 0.131 without), hence its noiseless-only gate.
`gdtset_diath` is still computed for diagnostics but no longer gates stopping; the raw
LP-vertex stability counter `_stable_count` likewise remains diagnostic-only.

Benchmark placement (20D, condnum 1e8, 5 problem types, both noise levels, vs CFD at
`2*D` evals and NMXFD at `4*D` evals with noise-tuned steps): every easy cell ties CFD at
`2*D + 1` evaluations; every hard cell beats NMXFD 7–15x in median relative error at or
below NMXFD's evaluation count, in both known and estimate (self-calibrated) noise modes.

## 6. Differences from the ideal theory

- The gradient estimate is the bounding-box center of the LP-solved polytope (or its
  boundary projection when the center is infeasible), not the Chebyshev center — see
  Section 4.
- The diameter computation is approximate and uses a local solver.
- The filtered subset in quickmode does not guarantee a closed polytope, but the refinement loop compensates.

## 7. Evaluation accounting

SAGE may evaluate `fun` at:

- the initial simplex points `x0` and `x0 + init_step * e_i` when history has 0 or 1 samples
- the query point `x` if it is not in history
- the current center point after a move (even if already in history) to keep evaluation order consistent
- multiple refinement points during a single call

Track evaluation budgets explicitly if needed.
