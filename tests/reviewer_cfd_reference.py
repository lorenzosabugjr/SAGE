"""Reproduce the reviewer's MATLAB finite-difference example in NumPy.

Both MATLAB's default ``double`` and NumPy's ``float64`` implement IEEE-754
binary64 arithmetic.  This script deliberately uses scalar, coordinate-wise
evaluations rather than vectorising the stencil, so its operation order mirrors
the MATLAB code:

    f(x + h*e_i) - f(x)          # FFD
    f(x + h*e_i) - f(x - h*e_i) # CFD

Run from the repository root:

    python -m tests.reviewer_cfd_reference
"""

from __future__ import annotations

import numpy as np


DIM = 20
H = np.float64(1e-6)
CURVATURE = np.float64(1e8)


def objective(x: np.ndarray) -> np.float64:
    """MATLAB: sum(x(1:dim-1).^2) + 10^(8)*x(dim)^2."""
    x = np.asarray(x, dtype=np.float64)
    return np.sum(x[: DIM - 1] ** 2, dtype=np.float64) + CURVATURE * x[DIM - 1] ** 2


def true_gradient(x: np.ndarray) -> np.ndarray:
    """Analytical gradient of ``objective`` at ``x``."""
    x = np.asarray(x, dtype=np.float64)
    gradient = np.float64(2.0) * x.copy()
    gradient[-1] = np.float64(2.0) * CURVATURE * x[-1]
    return gradient


def finite_difference_gradients(x0: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return FFD and CFD estimates using the reviewer's operation order."""
    x0 = np.asarray(x0, dtype=np.float64)
    g_ffd = np.zeros(DIM, dtype=np.float64)
    g_cfd = np.zeros(DIM, dtype=np.float64)
    f_x0 = objective(x0)

    for i in range(DIM):
        e = np.zeros(DIM, dtype=np.float64)
        e[i] = np.float64(1.0)
        g_ffd[i] = (objective(x0 + H * e) - f_x0) / H
        g_cfd[i] = (objective(x0 + H * e) - objective(x0 - H * e)) / (np.float64(2.0) * H)

    return g_ffd, g_cfd


def relative_error(estimate: np.ndarray, reference: np.ndarray) -> np.float64:
    """The same vector 2-norm relative error used by ``benchmark_grad.py``."""
    estimate = np.asarray(estimate, dtype=np.float64)
    reference = np.asarray(reference, dtype=np.float64)
    return np.linalg.norm(estimate - reference) / np.linalg.norm(reference)


def main() -> None:
    x0 = np.ones(DIM, dtype=np.float64)
    truth = true_gradient(x0)
    g_ffd, g_cfd = finite_difference_gradients(x0)
    ffd_error = relative_error(g_ffd, truth)
    cfd_error = relative_error(g_cfd, truth)

    assert x0.dtype == np.dtype(np.float64)
    assert g_ffd.dtype == np.dtype(np.float64)
    assert g_cfd.dtype == np.dtype(np.float64)
    assert np.dtype(np.float64).itemsize == 8

    print(f"NumPy dtype: {x0.dtype} ({np.finfo(np.float64).bits}-bit IEEE-754 binary64)")
    print(f"f(x0):       {objective(x0):.17g}")
    print(f"FFD error:   {ffd_error:.15e}")
    print(f"CFD error:   {cfd_error:.15e}")
    print(f"FFD / CFD:   {ffd_error / cfd_error:.6e}")


if __name__ == "__main__":
    main()
