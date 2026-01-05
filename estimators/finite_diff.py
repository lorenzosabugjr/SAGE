import numpy as np
from .base import BaseGradientEstimator
from typing import Callable, Tuple

class FFDEstimator(BaseGradientEstimator):
    """
    Forward Finite Difference (FFD) Estimator.

    Estimates the gradient by perturbing each coordinate direction by a small step `h`.
    This method requires `dim + 1` function evaluations per gradient estimate.

    Formula:
        g_i = (f(x + h*e_i) - f(x)) / h

    Attributes:
        fun (Callable): The objective function.
        dim (int): Input dimensionality.
        step (float): The step size `h`.
    """
    def __init__(
        self,
        fun: Callable[[np.ndarray], float],
        dim: int,
        step: float = 1e-6,
        history=None,
    ):
        """
        Initialize the FFD estimator.

        Args:
            fun: Objective function.
            dim: Input dimension.
            step: Finite difference step size.
        """
        super().__init__(fun, dim, history=history)
        self.step = step

    def __call__(self, x: np.ndarray, force: bool = False) -> np.ndarray:
        """
        Estimate gradient at x using forward differences.
        """
        grad = np.zeros(self.dim)
        z_k = None
        if self.history is not None:
            x_idx = self.history.find_indices(x)
            if x_idx.size > 0:
                z_k = self.history.Zn[x_idx[0]]
        if z_k is None:
            z_k = self.fun(x)
            self.update(x, z_k)
        
        for i in range(self.dim):
            x_step = x.copy()
            x_step[i] += self.step
            z_step = self.fun(x_step)
            self.update(x_step, z_step)
            grad[i] = (z_step - z_k) / self.step
            
        return grad

class CFDEstimator(BaseGradientEstimator):
    """
    Central Finite Difference (CFD) Estimator.

    Estimates the gradient by perturbing each coordinate direction forward and backward.
    This method is more accurate (O(h^2) error) than FFD but requires `2 * dim` evaluations.

    Formula:
        g_i = (f(x + h*e_i) - f(x - h*e_i)) / (2*h)

    Attributes:
        fun (Callable): The objective function.
        dim (int): Input dimensionality.
        step (float): The step size `h`.
    """
    def __init__(
        self,
        fun: Callable[[np.ndarray], float],
        dim: int,
        step: float = 1e-6,
        history=None,
    ):
        """
        Initialize the CFD estimator.

        Args:
            fun: Objective function.
            dim: Input dimension.
            step: Finite difference step size.
        """
        super().__init__(fun, dim, history=history)
        self.step = step

    def __call__(self, x: np.ndarray, force: bool = False) -> np.ndarray:
        """
        Estimate gradient at x using central differences.
        """
        grad = np.zeros(self.dim)
        
        for i in range(self.dim):
            x_fwd = x.copy()
            x_fwd[i] += self.step
            z_fwd = self.fun(x_fwd)
            self.update(x_fwd, z_fwd)
            
            x_bwd = x.copy()
            x_bwd[i] -= self.step
            z_bwd = self.fun(x_bwd)
            self.update(x_bwd, z_bwd)
            
            grad[i] = (z_fwd - z_bwd) / (2 * self.step)
            
        return grad

class NMXFDEstimator(BaseGradientEstimator):
    """
    Numerical Integration-based Gradient Estimator (NMXFD).

    Estimates the gradient by integrating the function against the derivative of a Gaussian kernel.
    This method can be more robust to noise than standard finite differences.

    Attributes:
        fun (Callable): The objective function.
        dim (int): Input dimensionality.
        n_u (int): Number of integration points per dimension.
        sigma (float): Width of the Gaussian kernel.
    """
    def __init__(
        self,
        fun: Callable[[np.ndarray], float],
        dim: int,
        rangeintegral: Tuple[float, float] = (-2, 2),
        numpoints: int = 4,
        sigma: float = 1e-2,
        history=None,
    ):
        """
        Initialize the NMXFD estimator.

        Args:
            fun: Objective function.
            dim: Input dimension.
            rangeintegral: Integration range (in sigma units usually).
            numpoints: Number of points for numerical integration.
            sigma: Kernel width.
        """
        super().__init__(fun, dim, history=history)
        self.n_u = numpoints
        self.u = np.linspace(rangeintegral[0], rangeintegral[1], numpoints, dtype='d')
        self.sigma = sigma
        self.dphi = -self.gaussian_derivative(self.u, 0, 1).reshape(-1, 1)

    def gaussian_derivative(self, x, mu, sigma):
        denominator = np.sqrt(2 * np.pi) * (sigma**3)
        numerator = (x - mu) * np.exp(-((x - mu)**2) / (2 * sigma**2))
        return -numerator / denominator

    def __call__(self, x: np.ndarray, force: bool = False) -> np.ndarray:
        grad = np.zeros(self.dim)
        
        # Calculate integration coefficients
        # Note: In the original code, this was done inside the loop but it seems constant per dim?
        # Actually checking original code: it seems to calculate coeff inside the loop over dims.
        # But u is constant. So coeff is constant.
        
        h = (self.u[-1] - self.u[0]) / (self.n_u - 1)
        mult = self.u[::-1][:self.n_u // 2].reshape(-1, 1).astype(float)
        phi_coeff = np.abs(self.dphi[:self.n_u // 2])
        coeff = phi_coeff * mult * h
        coeff[0] /= 2  # Adjust for boundary conditions
        normd_coeff = coeff / np.sum(coeff)

        for j in range(self.dim):
            # Collect samples
            res = np.zeros(self.n_u)
            for k, u_val in enumerate(self.u):
                dx = np.zeros(self.dim)
                dx[j] = u_val
                res[k] = self.fun(x + dx)
                self.update(x + dx, res[k])
                
            differences = np.array([
                res[::-1][k] - res[k] for k in range(self.n_u // 2)
            ]).reshape(-1, 1)
            
            output = normd_coeff * differences / (mult * self.sigma) / 2
            grad[j] = np.sum(output)
            
        return grad
