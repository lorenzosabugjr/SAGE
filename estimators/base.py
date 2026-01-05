from abc import ABC, abstractmethod
import numpy as np
from typing import Callable, Optional
from utils.history import HistoryBuffer

class BaseGradientEstimator(ABC):
    """
    Abstract base class for gradient estimators.
    
    The estimator acts as a callable that takes a point x and returns the estimated gradient.
    """
    def __init__(
        self,
        fun: Callable[[np.ndarray], float],
        dim: int,
        history: Optional[HistoryBuffer] = None,
    ):
        self.fun = fun
        self.dim = dim
        self.history = history
    @abstractmethod
    def __call__(self, x: np.ndarray, force: bool = False) -> np.ndarray:
        """
        Estimate the gradient at x.
        
        Args:
            x: The point at which to estimate the gradient.
            force: If True, recompute even if the estimator would normally reuse state.
            
        Returns:
            The estimated gradient vector.
        """
        pass
    
    def update(self, x: np.ndarray, y: float):
        """
        Optional method to update internal state if the estimator is stateful.
        Note: history.add() is handled by obj_func, so we don't add here.
        """
        pass
