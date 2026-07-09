"""Reproduce the reviewer's finite-difference example in NumPy.

This script deliberately uses scalar, coordinate-wise evaluations rather than
vectorising the stencil, so its operation order mirrors the MATLAB code:

    f(x + h*e_i) - f(x)          # FFD
    f(x + h*e_i) - f(x - h*e_i) # CFD

Set ``DTYPE`` to choose the arithmetic used for the stencil points, objective
values, and gradient arithmetic.

Run from the repository root:

    python -m tests.reviewer_cfd_reference
"""

from __future__ import annotations

import numpy as np


DIM = 20
DTYPE = np.float64
H = DTYPE("1e-6")
CURVATURE = DTYPE("1e8")


def objective(x: np.ndarray):
    """MATLAB: sum(x(1:dim-1).^2) + 10^(8)*x(dim)^2."""
    x = np.asarray(x, dtype=DTYPE)
    return np.sum(x[: DIM - 1] ** 2, dtype=DTYPE) + CURVATURE * x[DIM - 1] ** 2


def true_gradient(x: np.ndarray) -> np.ndarray:
    """Analytical gradient of ``objective`` at ``x``."""
    x = np.asarray(x, dtype=DTYPE)
    gradient = DTYPE("2.0") * x.copy()
    gradient[-1] = DTYPE("2.0") * CURVATURE * x[-1]
    return gradient


def finite_difference_gradients(x0: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return FFD and CFD estimates using the reviewer's operation order."""
    x0 = np.asarray(x0, dtype=DTYPE)
    g_ffd = np.zeros(DIM, dtype=DTYPE)
    g_cfd = np.zeros(DIM, dtype=DTYPE)
    f_x0 = objective(x0)

    for i in range(DIM):
        e = np.zeros(DIM, dtype=DTYPE)
        e[i] = DTYPE("1.0")
        g_ffd[i] = (objective(x0 + H * e) - f_x0) / H
        g_cfd[i] = (objective(x0 + H * e) - objective(x0 - H * e)) / (DTYPE("2.0") * H)

    return g_ffd, g_cfd


def relative_error(estimate: np.ndarray, reference: np.ndarray):
    """The same vector 2-norm relative error used by ``benchmark_grad.py``."""
    estimate = np.asarray(estimate, dtype=DTYPE)
    reference = np.asarray(reference, dtype=DTYPE)
    return np.linalg.norm(estimate - reference) / np.linalg.norm(reference)


def main() -> None:
    x0 = np.ones(DIM, dtype=DTYPE)
    truth = true_gradient(x0)
    g_ffd, g_cfd = finite_difference_gradients(x0)
    ffd_error = relative_error(g_ffd, truth)
    cfd_error = relative_error(g_cfd, truth)

    assert x0.dtype == np.dtype(DTYPE)
    assert g_ffd.dtype == np.dtype(DTYPE)
    assert g_cfd.dtype == np.dtype(DTYPE)

    print(f"NumPy dtype: {x0.dtype} ({np.finfo(DTYPE).bits}-bit)")
    print(f"f(x0):       {objective(x0):.17g}")
    print(f"FFD error:   {ffd_error:.15e}")
    print(f"CFD error:   {cfd_error:.15e}")
    print(f"FFD / CFD:   {ffd_error / cfd_error:.6e}")


if __name__ == "__main__":
    main()
