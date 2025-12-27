from pathlib import Path
import sys

import numpy as np
from scipy.optimize import minimize

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from estimators import SAGE, FFDEstimator, CFDEstimator


class CallCounter:
    def __init__(self, fun):
        self.fun = fun
        self.calls = 0

    def __call__(self, x):
        self.calls += 1
        return self.fun(x)


def make_quadratic(dim, seed):
    rng = np.random.default_rng(seed)
    a = rng.standard_normal((dim, dim))
    q = a.T @ a + 0.1 * np.eye(dim)
    c = rng.standard_normal(dim)
    return q, c


def quadratic_value(q, c, x):
    return 0.5 * x @ q @ x + c @ x


def run_bfgs(estimator_ctor, q, c, x0, noise_sigma, seed, maxiter=80):
    rng = np.random.default_rng(seed)

    def noisy_fun(x):
        val = quadratic_value(q, c, x)
        if noise_sigma > 0.0:
            val += rng.normal(0.0, noise_sigma)
        return val

    counter = CallCounter(noisy_fun)
    estimator = estimator_ctor(counter)

    res = minimize(
        fun=counter,
        x0=x0,
        jac=estimator,
        method="BFGS",
        options={"maxiter": maxiter},
    )
    return res, counter.calls


def main():
    dim = 6
    noise_sigma = 1e-2  # Set to 0.0 for deterministic comparisons.
    q, c = make_quadratic(dim, seed=0)
    rng = np.random.default_rng(1)
    x0 = rng.normal(size=dim)

    x_star = -np.linalg.solve(q, c)
    f_star = quadratic_value(q, c, x_star)

    configs = [
        (
            "SAGE",
            lambda fun: SAGE(
                fun=fun,
                dim=dim,
                quickmode=True,
            ),
        ),
        ("FFD", lambda fun: FFDEstimator(fun=fun, dim=dim, step=1e-4)),
        ("CFD", lambda fun: CFDEstimator(fun=fun, dim=dim, step=1e-4)),
    ]

    header = "Estimator | f_det gap | nfev | njev | total_calls"
    print(header)
    print("-" * len(header))
    # total_calls counts objective evaluations triggered by SciPy and the estimator.
    for i, (name, ctor) in enumerate(configs):
        res, calls = run_bfgs(
            ctor,
            q,
            c,
            x0,
            noise_sigma,
            seed=123 + i,
        )
        f_det = quadratic_value(q, c, res.x)
        gap = f_det - f_star
        nfev = res.nfev if res.nfev is not None else -1
        njev = res.njev if res.njev is not None else -1
        print(f"{name:9s} | {gap: .3e} | {nfev:4d} | {njev:4d} | {calls:11d}")


if __name__ == "__main__":
    main()
