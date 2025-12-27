# SAGE: A Set-based Adaptive Gradient Estimator

**SAGE** is a Python implementation of the Set-based Adaptive Gradient Estimator for gradients of noisy black-box scalar functions. It uses set membership ideas to build a **set of admissible gradients** from existing samples and refines that set only when needed.

## Theory Snapshot (from the paper: https://arxiv.org/abs/2508.19400)

SAGE assumes a Lipschitz continuous Hessian:

```
||H(x1) - H(x2)|| <= gamma_H * ||x1 - x2||
```

For two samples `(xi, zi)` and `(xj, zj)`, define:

```
a_ij = xj - xi
mu_ij = ||a_ij||
u_ij = a_ij / mu_ij
g~_ij = (zj - zi) / mu_ij
```

Then the directional slope error satisfies:

```
| g~_ij - g_i^T u_ij | <= (1/2) * H_i * mu_ij + (1/6) * gamma_H * mu_ij^2
```

With bounded noise `|epsilon| <= eps`, the bound becomes:

```
| g~_ij - g_i^T u_ij | <= (1/2) * H_i * mu_ij + (1/6) * gamma_H * mu_ij^2 + 2*eps / mu_ij
```

Each pair `(xi, xj)` yields a **slab** of admissible gradients. The intersection of all slabs is a polytope `G^(i)` guaranteed to contain the true gradient.

## Algorithm Overview

1. **Build constraints** from all samples (or a filtered subset).
2. **Solve an LP** to estimate the gradient and unknown constants (Hessian norm, Hessian Lipschitz constant, and optionally noise bound).
3. **Estimate the diameter** of the gradient set. If it exceeds a target threshold, **sample again** along the diameter direction.
4. **Noisy case:** the optimal sampling radius `alpha*` solves:

```
(1/3) * gamma_H * alpha^3 + (1/2) * H_i * alpha^2 - 2*eps = 0
```

## Practical Notes

- **Stateful:** SAGE aggregates the full history of evaluations `(x_k, f(x_k))`. You can pass `initial_history` to seed it, or let SAGE auto-seed a simplex around the first query with `init_step` (default `1e-3`).
- **Extra evaluations:** SAGE may call `fun` multiple times per gradient estimate to refine the set, so track evaluation budgets accordingly.
- **Noise handling:** SAGE estimates the noise bound internally; you only need to define the noisy objective.

## Installation

```bash
pip install -r requirements.txt
```

## Usage

```python
import numpy as np
from estimators import SAGE

def black_box(x):
    return np.sum(x**2) + np.random.normal(0, 0.01)

estimator = SAGE(
    fun=black_box,
    dim=10,
    quickmode=True,
)

x = np.random.rand(10)
grad = estimator(x)
```

## Documentation

- [Getting Started](docs/quickstart.md)
- [Mathematical Theory](docs/theory.md)
- [Implementation Notes](docs/implementation.md)
- [Benchmarks](docs/benchmarks.md)
- [API Reference](docs/api.md)
- [Examples](examples/README.md)

## References

If you use SAGE in your research, please cite:

> **SAGE: A Set-based Adaptive Gradient Estimator**  
> Lorenzo Sabug Jr., Fredy Ruiz, Lorenzo Fagiano  
> arXiv:2508.19400, 2025.
