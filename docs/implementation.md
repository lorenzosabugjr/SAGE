# Implementation Notes

This document ties the theory in the paper (https://arxiv.org/abs/2508.19400) to the current implementation in `estimators/sage.py`.

## 1. State and data structures

`SAGE` maintains the full sample history:

- `Xn`: evaluated points, shape `(N, D)`
- `Zn`: function values, shape `(N,)`
- `gdt_est`: current gradient estimate
- `hess_norm`: estimated Hessian norm `H_i`
- `hess_lipsc`: estimated Hessian Lipschitz constant `gamma_H`
- `ns_est`: estimated noise bound `eps`
- `gdtset_diaid`: target diameter
- `gdtset_diath`: current threshold
- `aux_samples_count`: number of refinement samples added in the current call
- `init_step`: step size for the initial simplex when history is empty

## 2. LP construction

For each neighbor `x^(j)`, the code builds two inequality rows (slab constraints)
including the noise term `eps`:

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

The code uses `scipy.optimize.linprog` to solve the LP.

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

## 5. Active sampling loop

For each call:

1. Solve the LP and compute the diameter.
2. If the diameter is below the threshold, return `gdt_est`.
3. Otherwise, sample along the diameter direction:
   ```
   x_new = x + alpha * d_hat
   ```
4. Update history and repeat.

Additional details:

- If `ns_est` is near zero, SAGE uses `alpha = 1e-6` and resets the threshold to `gdtset_diaid`.
- Otherwise, it uses the cubic root estimate and sets `gdtset_diath = 1.01 * alpha`.
- The loop caps at `2.5*D` refinement samples per call.

## 6. Differences from the ideal theory

- The gradient estimate is the LP solution, not the Chebyshev center.
- The diameter computation is approximate and uses a local solver.
- The filtered subset in quickmode does not guarantee a closed polytope, but the refinement loop compensates.

## 7. Evaluation accounting

SAGE may evaluate `fun` at:

- the initial simplex points `x0` and `x0 + init_step * e_i` when history is empty
- the query point `x` if it is not in history
- the current center point after a move (even if already in history) to keep evaluation order consistent
- multiple refinement points during a single call

Track evaluation budgets explicitly if needed.
