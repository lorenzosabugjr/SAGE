from pathlib import Path
import sys

import numpy as np
from scipy.optimize import minimize

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from estimators import SAGE

rng = np.random.default_rng(0)


def noisy_rosenbrock(x):
    x = np.asarray(x)
    base = np.sum(100.0 * (x[1:] - x[:-1] ** 2) ** 2 + (1.0 - x[:-1]) ** 2)
    return base + rng.normal(0.0, 1e-2)


dim = 5
x0 = rng.normal(size=dim)

grad = SAGE(
    fun=noisy_rosenbrock,
    dim=dim,
    quickmode=True,
)

res = minimize(
    fun=noisy_rosenbrock,
    x0=x0,
    jac=grad,
    method="BFGS",
    options={"maxiter": 200},
)

print("x*:", res.x)
print("f(x*):", res.fun)
print("nfev:", res.nfev, "njev:", res.njev)
