# Theory: Set-Based Adaptive Gradient Estimation

This note summarizes the theory used by SAGE. It mirrors the results in the paper (https://arxiv.org/abs/2508.19400).

## 1. Assumptions and notation

Let `f: R^D -> R` be the objective, `g(x)` its gradient, and `H(x)` its Hessian. We assume:

```
||H(x1) - H(x2)|| <= gamma_H * ||x1 - x2||
```

Given a dataset of samples:

```
X^n = { (x^(i), z^(i)), i = 1..n },   z^(i) = f(x^(i))
```

For any two samples `(x^(i), z^(i))` and `(x^(j), z^(j))`:

```
a_ij = x^(j) - x^(i)
mu_ij = ||a_ij||
u_ij = a_ij / mu_ij
g~_ij = (z^(j) - z^(i)) / mu_ij
```

Let `H_i = ||H(x^(i))||` denote the Hessian spectral norm at `x^(i)`.

## 2. Directional slope bound (noiseless)

Lemma (paper): for any pair of samples,

```
| g~_ij - g_i^T u_ij | <= (1/2) * H_i * mu_ij + (1/6) * gamma_H * mu_ij^2
```

This defines a **gradient slab** for `g_i` along direction `u_ij`.

### Noisy evaluations

If samples are noisy with bounded noise `|epsilon| <= eps`, then:

```
| g~_ij - g_i^T u_ij | <= (1/2) * H_i * mu_ij + (1/6) * gamma_H * mu_ij^2 + 2*eps / mu_ij
```

The last term grows as `mu_ij` gets small, which motivates an optimal sampling radius.

## 3. Gradient set polytope

Collecting all slabs for `x^(i)` gives the polytope:

```
G^(i) = { g in R^D : A g <= b }
```

where each pair `(i, j)` contributes two inequalities:

```
-u_ij^T g <= -g~_ij + (1/2) H_i mu_ij + (1/6) gamma_H mu_ij^2
 u_ij^T g <=  g~_ij + (1/2) H_i mu_ij + (1/6) gamma_H mu_ij^2
```

In the noisy case, the term `2*eps / mu_ij` appears on the right-hand side.

## 4. Estimating H_i, gamma_H, and noise bound

In practice, `H_i` and `gamma_H` are unknown. SAGE estimates them together with `g_i` by solving a linear program (LP).

### Noiseless LP

```
min_{g, H_i, gamma_H}  H_i + gamma_H
subject to:
    A [g, H_i, gamma_H]^T <= b
    H_i >= 0, gamma_H >= 0
```

### Noisy LP

```
min_{g, H_i, gamma_H, eps}  H_i + gamma_H + eps
subject to:
    A_eps [g, H_i, gamma_H, eps]^T <= b
    H_i >= 0, gamma_H >= 0, eps >= 0
```

The solution returns an estimated gradient `g~^(i)` and the tightest unfalsified set `G~^(i)`.

### Known fixed noise bound

The noisy LP above treats `eps` as unknown and estimates it from data. If instead the
noise bound is known a priori (e.g. from the measurement device or simulator), `eps` need
not be a free decision variable: substitute the known bound `eps_fixed` directly and move
its contribution to the right-hand side:

```
min_{g, H_i, gamma_H}  H_i + gamma_H
subject to:
    A [g, H_i, gamma_H]^T <= b + 2*eps_fixed/mu_ij
    H_i >= 0, gamma_H >= 0
```

This differs from the noiseless LP above (which assumes `eps = 0` structurally) in that
`eps_fixed` may be any known nonnegative constant, not necessarily zero. Both cases share
the same reduced `[g, H_i, gamma_H]` variable set and are exposed together through the
`noise_bound` constructor argument on `SAGE` (`noise_bound=0.0` recovers the noiseless
LP; `noise_bound>0` recovers this fixed-bound LP). See `docs/implementation.md` for the
concrete matrix layout used in the code.

## 5. Gradient set diameter and refinement

Define the diameter:

```
rho(G~^(i)) = max_{g1, g2 in G~^(i)} ||g1 - g2||
```

This is a non-convex optimization. If `rho(G~^(i))` is larger than a desired threshold, SAGE refines the set by sampling:

```
x_new = x^(i) + alpha * d_hat
```

where `d_hat` is the unit vector along the diameter direction.

### Optimal radius under noise

With bounded noise, the optimal radius `alpha*` solves:

```
(1/3) * gamma_H * alpha^3 + (1/2) * H_i * alpha^2 - 2*eps = 0
```

This gives the **theoretical best achievable** refinement for the noisy case.

### Certificate-sized radius (implementation detail)

The `alpha*` formula above requires `H_i`, `gamma_H`, and `eps` to already be resolved by
the LP — but whenever noise makes the LP's curvature terms feasible at zero (which is
common), the cubic degenerates and no informative root exists. Rather than chasing the
unresolvable optimum, the implementation sizes the radius directly to its stopping
criterion: a full sweep of symmetric axis pairs at radius `r` tightens each axis of the
gradient box to the information floor `~4*eps/r`, so the radius whose *single* sweep
would meet the relative certificate `rho < rel_tol * ||g||` (Section 7) is

```
r* = 4 * eps * sqrt(D) / (rel_tol * ||g_seed||)
```

computed once per query from the seed-stencil LP estimate and clipped to
`[1.5, 16] * max(init_step, 2*sqrt(eps))` — never below the seed radius (samples inside
it carry strictly worse noise-to-signal) nor below the noise-informative scale
`2*sqrt(eps)` (which protects against a mis-chosen `init_step`). Two safety nets correct `r*` when its implicit
assumptions fail: exhaustion-driven geometric growth (if a full sweep at `r*` did not
certify, the next sweep runs at a doubled radius), and a pilot second-difference guard
that *shrinks* `r*` when a completed axis pair proves real curvature at that scale —
`|z+ + z- - 2 z0|/2` can exceed `2*eps` only through curvature, in which case the radius
backs off to where the quadratic residual matches the noise slab (`r*sqrt(2*eps/D2)`)
before more than one pair is committed at the too-large scale. The `alpha*` cubic is
still used while the seed LP itself is being built (no `g_seed` yet). See
`docs/implementation.md` for the exact mechanics.

## 6. Filtered sets for computation

To reduce LP size, SAGE can filter the dataset by selecting samples closest to the target radius:

```
xi(x^(j)) = | ||x^(j) - x^(i)|| - alpha* |
```

The `N_f` samples with smallest `xi` values are used to construct the constraints. In the noiseless case this reduces to nearest neighbors; in the noisy case it forms a hollow shell around `x^(i)`.

## 7. Practical refinements (implementation detail)

Beyond the derivations above, the implementation makes a few practical choices to keep the
refinement loop well-behaved on real (noisy, finite-sample) data. None of these change the
theory above; see `docs/implementation.md` for the exact mechanics:

- **Certificate-sized radius with pilot guard**, described above (Section 5).
- **Noise self-calibration (estimated-noise mode)**: the noisy LP of Section 4 *minimizes*
  `eps` subject to feasibility, so its `eps` is a lower bound on the noise, not an
  estimator. The implementation re-estimates the bound once from the seed stencil's own
  per-axis second differences (whose noise-only distribution is known:
  `max_i |z+ + z- - 2 z0|/2 <= 2*eps`) and then treats it as a fixed bound (Section 4's
  known-bound LP) for the rest of the estimator's lifetime.
- **Axis-only refinement (approx mode)**: with `diam_mode="approx"` (the default),
  `d_hat` above is never the composite diameter direction — the implementation cycles
  through axis-aligned probes exclusively, growing the radius once every axis has been
  probed at the current radius. In approx mode, `gd_v` is the elementwise-non-negative
  bounding-box diagonal (`max_g - min_g`), so its normalized direction is always the
  same fixed all-positive-orthant vector; a probe along it cannot move any axis-aligned
  bound (verified numerically) and so cannot inform the box diameter or the box-center
  point estimate (Section 4) — only axis probes can. This does not apply to
  `diam_mode="exact"`, where `gd_v` is a genuine signed direction from the SLSQP solve;
  the implementation currently uses the same axis-only loop for both modes, but this has
  only been validated for approx mode (the mode all production benchmarks use).
- **Point estimate**: `g~^(i)` is reported as the axis-aligned bounding-box center of
  `G~^(i)` (or its boundary projection when the center is infeasible), since the LP
  objective in Section 4 never costs `g`, so the raw LP vertex can be an arbitrary point
  of the optimal face rather than a meaningful summary.
- **Stopping criterion — relative-accuracy certificate**: refinement stops once
  `rho(G~^(i)) < rel_tol * ||g~^(i)||` (default `rel_tol = 0.5`), i.e. once the set
  brackets the reported gradient to a certified *relative* accuracy, or at a `2*D`
  auxiliary-sample cap (plus an absolute-diameter floor in the noiseless regime only).
  An absolute diameter threshold (`gdtset_diath = 1.01*alpha*`) is *not* used,
  deliberately: the target derived from a dense-directional-coverage assumption is
  frequently unreachable at polynomial sample budgets in higher `D` (a coverage
  limitation), and the threshold is itself an LP estimate from the same noisy samples,
  changing discontinuously as curvature resolves — an uncertain, moving target. A
  point-estimate stagnation stop used in an earlier revision was also removed: measured on
  the production configs it fired mid-improvement on exactly the points where refinement
  pays (weak-gradient points, ~3x accuracy left on the table) while wasting `~2*D`
  evaluations on points that should stop at the seed. The relative certificate handles
  both ends for free: strong-gradient points certify at the seed stencil itself (CFD-cost
  with a certificate), weak-gradient points refine until certified or capped. See
  `docs/implementation.md` for the exact mechanics and measured numbers.

## 8. Implementation note

The implementation approximates the diameter and uses local optimization in that step. See `docs/implementation.md` for code-level details.
