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
        # Indicates whether update() recomputes the gradient estimate.
        self.recompute_on_update = False

    @abstractmethod
    def __call__(self, x: np.ndarray, **kwargs) -> np.ndarray:
        """
        Estimate the gradient at x.
        
        Args:
            x: The point at which to estimate the gradient.
            **kwargs: Additional arguments for specific estimators.
            
        Returns:
            The estimated gradient vector.
        """
        pass
    
    def update(self, x: np.ndarray, y: float):
        """
        Optional method to update internal history if the estimator is stateful.
        """
        if self.history is not None:
            self.history.add(x, y)
