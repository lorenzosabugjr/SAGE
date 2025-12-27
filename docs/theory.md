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

## 6. Filtered sets for computation

To reduce LP size, SAGE can filter the dataset by selecting samples closest to the target radius:

```
xi(x^(j)) = | ||x^(j) - x^(i)|| - alpha* |
```

The `N_f` samples with smallest `xi` values are used to construct the constraints. In the noiseless case this reduces to nearest neighbors; in the noisy case it forms a hollow shell around `x^(i)`.

## 7. Implementation note

The implementation approximates the diameter and uses local optimization in that step. See `docs/implementation.md` for code-level details.
