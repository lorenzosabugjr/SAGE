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

### Radius growth when alpha* can't yet be computed (implementation detail)

The `alpha*` formula above requires `H_i`, `gamma_H`, and `eps` to already be resolved by
the LP. Early in a call — or whenever noise makes the LP's curvature terms feasible at
zero, which is common under noise — the implementation cannot compute `alpha*` directly
and instead uses a fallback radius (`alpha ~ 2*sqrt(eps)`, i.e. assuming unit Hessian
norm). If that fallback radius turns out to be too small for the gradient-set diameter to
actually contract, the implementation grows the *used* radius geometrically based on this
observed feedback — not on any assumed curvature bound — and lets the cubic-root formula
take back over automatically once the LP starts resolving real curvature at the larger
radius. This is a fallback-only behavior: the `alpha*` derivation and the
`gdtset_diath = 1.01 * alpha` relationship above are unchanged. See
`docs/implementation.md` for the exact trigger and mechanics.

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

- **Radius growth from feedback** when `alpha*` can't yet be computed, described above.
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
- **Stopping criterion**: refinement stops once the diameter `rho(G~^(i))` meets the
  threshold, the sample cap is reached, *or* the box-centered point estimate has stabilized
  across 3 consecutive solves. The threshold-only version (chase the diameter target until
  it is met or the cap is hit) is *not* used, deliberately, for two reasons: (1) eval-cost
  control — empirically the diameter target is reached on a small minority of calls (the
  target derived from a dense-directional-coverage assumption is frequently unreachable at
  polynomial sample budgets in higher `D`, a coverage limitation, not a bug), so
  threshold-only chasing routinely exhausts the full sample cap on most calls, which is an
  unacceptable operational cost; and (2) the threshold itself is not a fixed, known
  quantity — `gdtset_diath` is derived from `H_i`, `gamma_H`, and (in estimated-noise mode)
  `eps`, all of which the LP itself is estimating from the same noisy, finite sample set,
  and which can change discontinuously as curvature resolves. Insisting on exactly meeting
  an uncertain, moving target is not obviously more principled than accepting a
  well-converged point estimate. Point-estimate stability of the *box-centered* estimate
  (not the raw LP vertex, which is solver-path-dependent — see Section 4) is used as the
  practical convergence signal instead. See `docs/implementation.md` for the exact
  mechanics.
- **Noise-aware reseed skip**: the coordinate stencil placed when noise is first detected
  (Section 4's noisy-LP case) is skipped when the existing seed is already at an
  appropriate distance from the current `alpha`, avoiding a redundant full stencil.

## 8. Implementation note

The implementation approximates the diameter and uses local optimization in that step. See `docs/implementation.md` for code-level details.
