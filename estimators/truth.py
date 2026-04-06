import numpy as np
from typing import Callable, Optional
from .base import BaseGradientEstimator
from utils.history import HistoryBuffer
from problems.base import BaseProblem


class TruthEstimator(BaseGradientEstimator):
    """
    Gradient "estimator" that returns the true analytical gradient.

    This consumes zero function evaluations — it calls problem.gradient(x)
    directly on the deterministic objective. Useful as:
      - A benchmark reference for gradient accuracy comparisons.
      - An oracle gradient source to measure optimizer convergence
        in the absence of estimation noise.
    """

    def __init__(
        self,
        fun: Callable[[np.ndarray], float],
        dim: int,
        problem: BaseProblem,
        history: Optional[HistoryBuffer] = None,
    ):
        super().__init__(fun, dim, history=history)
        self.problem = problem

    def __call__(self, x: np.ndarray, force: bool = False) -> np.ndarray:
        """Return the exact gradient at x (no function evaluations consumed)."""
        return self.problem.gradient(x)
