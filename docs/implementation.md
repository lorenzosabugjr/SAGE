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
  latest LP-estimated value; in fixed mode it is pinned to `noise_bound` and never
  overwritten by the LP.
- `noise_bound`: constructor value, `None` (estimated mode) or a fixed float `>= 0`.
- `noise_bound_is_fixed`: `True` when `noise_bound` was supplied.
- `gdtset_diaid`: target diameter
- `gdtset_diath`: current threshold
- `aux_samples_count`: number of refinement samples added in the current call
- `init_step`: step size for the initial simplex when history has 0 or 1 samples
- `min_g`, `max_g`: per-axis bounding-box bounds cached by `_calc_diam_approx` and reused
  by the point-estimate step (see Section 4) instead of re-solving the same LPs
- `gd_v`, `gd_vm`: per-axis width vector and its norm (the diameter used against
  `gdtset_diath`)
- `_aux_radius_growth`: per-call multiplier applied to the modeled auxiliary radius when
  observed feedback shows the diameter is not contracting; reset to `1.0` at the start of
  each call (see Section 5)
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

### Auxiliary sampling radius

- If `ns_est` is near zero, SAGE uses `alpha = init_step` and resets the threshold to
  `gdtset_diaid`.
- Otherwise it uses the cubic-root optimal radius (or the `2*sqrt(eps)` fallback when
  curvature can't yet be resolved), and sets `gdtset_diath = 1.01 * alpha` — this formula
  is unchanged from the theory.
- **Axis-only cycling with exhaustion-driven growth** (`_next_aux_direction`): each call
  proactively queues a symmetric `+e_i`/`-e_i` pair along the widest not-yet-probed
  coordinate (ranked by `|gd_v_i|`). Only once every axis has been probed at the current
  radius (`_enqueue_axis_probe_pair` finds none left) does it scale the *used* radius by a
  geometric growth factor (`_aux_radius_growth *= factor`) and restart a fresh sweep over
  all axes, applied on top of the modeled `alpha` inside `_compute_aux_step` without
  changing the modeled value that `gdtset_diath` derives from. Growth resets to `1.0` at
  the start of each call. An earlier version of this mechanism alternated axis probes with
  probes along the composite diameter direction, growing reactively when a diameter probe
  didn't contract; that was removed after finding it was structurally inert in
  `diam_mode="approx"` — see `docs/theory.md` §7.

### Noise-aware reseed skip

When the first LP detects noise (`ns_est > 1e-9`), SAGE normally places a full coordinate
stencil at the noise-appropriate radius `alpha` (`_reseed_at_alpha`) so the LP has
well-conditioned constraints from the start, since the tiny `init_step` seed is otherwise
uninformative at that noise level. `_maybe_noise_reseed` skips this when it would be
redundant: if the seed stencil (placed at `self._seed_step`, normally `init_step`) already
falls within `[alpha/2, alpha*2]` of the current `alpha`, it is already appropriately
placed, so the reseed is skipped and no extra evaluations are added. Otherwise the
coordinate stencil is added as before.

### Stopping criterion

`_should_stop_refinement` returns `True` when any of:

- the diameter target is met (`gd_vm < gdtset_diath`) and no axis probe is pending, or
- a forced stop is requested (`gdt_est_frc`), or
- the auxiliary sample cap is reached (`aux_samples_count >= 5.0*D`) and no axis probe is
  pending, or
- the *box-centered* point estimate has stabilized: `_stable_count_final >= 3` (< 2%
  relative change across 3 consecutive solves) and no axis probe is pending.

This is deliberately **not** diameter-target-or-cap only. Measured on the standard 20D,
condnum-1e8, noise-1.0 diagnostic across all 5 problem types (50 points): with the
stagnation clause active, it is the actual stopping reason on 76% of calls (median 83
evals, median abs error 0.66); removing it more than halves error (median 0.27) but pushes
94% of calls to diameter-met-or-cap, with the cap alone firing 64% of the time (median 141
evals, near the `5*D` ceiling on nearly every call) — an eval-cost profile judged
operationally unacceptable. Two considerations justify keeping the stagnation branch
rather than treating it as inferior to a "pure" threshold criterion: (1) it is the primary
lever that keeps typical eval cost well below the cap, and (2) `gdtset_diath` is not a
known, fixed quantity — it is derived from `H_i`/`gamma_H`/`eps`, themselves LP estimates
from the same noisy sample set, and can jump discontinuously as curvature resolves (see
Section 6's radius-growth discussion), so treating it as the sole authoritative signal is
not obviously more correct than accepting a stable box-centered estimate.

An earlier version tracked the **raw** LP-solved vertex's stability (`_stable_count`) for
this purpose and found it unreliable, since that vertex is solver-path-dependent and can be
an arbitrary point of the optimal face (Section 4) rather than a meaningful summary —
`_stable_count` is still computed and logged as a diagnostic signal but does not gate
stopping. The current criterion instead tracks stability of the box-centered/projected
estimate (`_stable_count_final`, updated in `_recompute_at` after `_update_point_estimate`
runs), which is far less solver-path-dependent.

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
