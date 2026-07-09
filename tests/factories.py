"""
Shared factory functions for creating problems and gradient estimators.

Used by benchmark scripts, smoke tests, and the optimization runner.
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from problems import LeastSquares, Lasso, L1LogReg, L2LogReg, LogSumExp
from estimators import (
    FFDEstimator, CFDEstimator, NMXFDEstimator,
    GSGEstimator, cGSGEstimator, SAGE, TruthEstimator,
)
from utils.history import HistoryBuffer


def create_problem(name: str, dims: int, condnum: float, randseed: int = 0):
    """Instantiate a benchmark problem by name.

    Supported names: "least-squares", "lasso", "l1-log-reg",
    "l2-log-reg", "log-sum-exp".
    """
    if name == "least-squares":
        return LeastSquares(dims, condnum, randseed=randseed)
    elif name == "lasso":
        return Lasso(dims, condnum, randseed=randseed)
    elif name == "l1-log-reg":
        return L1LogReg(dims, condnum, randseed=randseed)
    elif name == "l2-log-reg":
        return L2LogReg(dims, condnum, randseed=randseed)
    elif name == "log-sum-exp":
        return LogSumExp(dims, condnum, randseed=randseed)
    else:
        raise ValueError(f"Unknown problem: {name}")


def create_estimator(
    name: str,
    obj_func,
    dims: int,
    history: HistoryBuffer,
    gdtcalcstep: float = 1e-6,
    randseed: int = 0,
    dtype=np.float128,
    **sage_kwargs,
):
    """Instantiate a gradient estimator by name.

    Supported names: "ffd", "cfd", "gsg", "cgsg",
    "nmxfd", "sage", "truth".

    Extra keyword arguments are forwarded to the SAGE constructor
    (e.g. ``quickmode``, ``diam_mode``, ``init_step``).
    """
    if name == "ffd":
        return FFDEstimator(obj_func, dims, step=gdtcalcstep, history=history, dtype=dtype)
    elif name == "cfd":
        return CFDEstimator(obj_func, dims, step=gdtcalcstep, history=history, dtype=dtype)
    elif name == "gsg":
        return GSGEstimator(obj_func, dims, m=dims, u=gdtcalcstep, seed=randseed, history=history)
    elif name == "cgsg":
        return cGSGEstimator(obj_func, dims, m=dims, u=gdtcalcstep, seed=randseed, history=history)
    elif name == "nmxfd":
        return NMXFDEstimator(obj_func, dims, sigma=gdtcalcstep, history=history)
    elif name == "sage":
        # Default SAGE settings; caller can override via sage_kwargs
        kw = dict(quickmode=True, diam_mode="approx", init_step=gdtcalcstep)
        kw.update(sage_kwargs)
        return SAGE(obj_func, dims, history=history, **kw)
    elif name == "truth":
        # TruthEstimator needs the problem object passed via sage_kwargs["problem"]
        problem = sage_kwargs.pop("problem", None)
        return TruthEstimator(obj_func, dims, problem=problem, history=history)
    else:
        raise ValueError(f"Unknown estimator: {name}")
