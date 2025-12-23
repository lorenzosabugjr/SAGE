import numpy as np
from typing import Callable
from .base import BaseGradientEstimator

class GSGEstimator(BaseGradientEstimator):
    """
    Gaussian Smoothing Gradient (GSG) Estimator.

    Estimates the gradient by averaging directional derivatives along `m` random Gaussian directions.
    This acts as a smoothed gradient approximation, effectively convolving the function with a Gaussian.

    Formula:
        g ~ (1/m) * sum_{i=1}^m [ (f(x + u*e_i) - f(x)) / u * e_i ]
        where e_i ~ N(0, I)

    Attributes:
        fun (Callable): The objective function.
        dim (int): Input dimensionality.
        m (int): Number of random directions to sample.
        u (float): Smoothing radius / step size.
    """
    def __init__(
        self,
        fun: Callable[[np.ndarray], float],
        dim: int,
        m: int,
        u: float = 1e-6,
        seed: int = None,
        history=None,
    ):
        """
        Initialize the GSG estimator.

        Args:
            fun: Objective function.
            dim: Input dimension.
            m: Number of random directions.
            u: Smoothing parameter.
            seed: Random seed for reproducibility.
        """
        super().__init__(fun, dim, history=history)
        self.m = m  # Number of directions
        self.u = u  # Smoothing radius / step size
        self.rng = np.random.default_rng(seed)

    def __call__(self, x: np.ndarray) -> np.ndarray:
        """Estimate gradient using GSG."""
        grad = np.zeros(self.dim)
        z_k = None
        if self.history is not None:
            x_idx = self.history.find_indices(x)
            if x_idx.size > 0:
                z_k = self.history.Zn[x_idx[0]]
        if z_k is None:
            z_k = self.fun(x)
            self.update(x, z_k)
        
        # Generate m random directions
        e = self.rng.normal(size=(self.m, self.dim))
        
        for i in range(self.m):
            # Sample forward
            x_new = x + self.u * e[i]
            z_new = self.fun(x_new)
            self.update(x_new, z_new)
            
            # Update gradient estimate
            grad += (z_new - z_k) / self.u * e[i]
            
        grad /= self.m
        return grad

class cGSGEstimator(BaseGradientEstimator):
    """
    Centered Gaussian Smoothing Gradient (cGSG) Estimator.

    Similar to GSG but uses antithetic sampling (forward and backward steps) for variance reduction.
    Requires `2 * m` function evaluations.

    Formula:
        g ~ (1/m) * sum_{i=1}^m [ (f(x + u*e_i) - f(x - u*e_i)) / (2*u) * e_i ]

    Attributes:
        fun (Callable): The objective function.
        dim (int): Input dimensionality.
        m (int): Number of random direction pairs.
        u (float): Smoothing radius / step size.
    """
    def __init__(
        self,
        fun: Callable[[np.ndarray], float],
        dim: int,
        m: int,
        u: float = 1e-6,
        seed: int = None,
        history=None,
    ):
        """
        Initialize the cGSG estimator.

        Args:
            fun: Objective function.
            dim: Input dimension.
            m: Number of antithetic pairs.
            u: Smoothing parameter.
            seed: Random seed.
        """
        super().__init__(fun, dim, history=history)
        self.m = m  # Number of pairs
        self.u = u
        self.rng = np.random.default_rng(seed)

    def __call__(self, x: np.ndarray) -> np.ndarray:
        """Estimate gradient using cGSG."""
        grad = np.zeros(self.dim)
        
        # Generate m random directions
        e = self.rng.normal(size=(self.m, self.dim))
        
        for i in range(self.m):
            # Sample forward
            x_fwd = x + self.u * e[i]
            z_fwd = self.fun(x_fwd)
            self.update(x_fwd, z_fwd)
            
            # Sample backward
            x_bwd = x - self.u * e[i]
            z_bwd = self.fun(x_bwd)
            self.update(x_bwd, z_bwd)
            
            # Update gradient estimate
            grad += (z_fwd - z_bwd) / (2 * self.u) * e[i]
            
        grad /= self.m
        return grad
