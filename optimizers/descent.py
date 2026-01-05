from enum import Enum, unique
import numpy as np
from numpy.linalg import norm
from typing import Callable, Optional
from estimators.base import BaseGradientEstimator

@unique
class StepSizeMode(Enum):
    FIXED = 0
    ADAPTIVE = 1

class StandardDescent:
    """
    A standard gradient descent optimizer with optional BFGS and Armijo line search.
    It delegates gradient estimation to a BaseGradientEstimator.
    """
    def __init__(
        self,
        fun: Callable[[np.ndarray], float],
        x0: np.ndarray,
        grad_estimator: BaseGradientEstimator,
        stepsize: float = 1.0,
        stepsizemode: StepSizeMode = StepSizeMode.ADAPTIVE,
        bfgs: bool = False,
        z0: Optional[float] = None,
        callback: Optional[Callable[[float], None]] = None,
    ):
        self.fun = fun
        self.x_k = x0.copy()
        self.D = x0.shape[0]
        self.grad_estimator = grad_estimator
        self.callback = callback
        
        # Optimization state
        if z0 is not None:
            self.z_k = z0
        else:
            self.z_k = self.fun(self.x_k)
            
        self.k = 0
        
        # Step size parameters
        self.eta0 = stepsize
        self.eta = self.eta0
        self.etaM = 0.5   # Backtracking multiplier
        self.etaT = 1e-6  # Armijo condition factor
        self.eta_mode = stepsizemode
        
        # BFGS state
        self.bfgs = bfgs
        self.bfgs_hinv = np.identity(self.D)
        self.bfgs_gdtp = np.zeros(self.D)
        self.gdt_est = np.zeros(self.D)
        
        # Helper for BFGS updates
        self.x_kp = self.x_k.copy() # x_{k-1}

    def step(self):
        """
        Perform a single optimization step.

        1. estimates the gradient at the current point x_k using the estimator.
        2. Updates the BFGS Hessian approximation (if enabled).
        3. Computes the search direction p_k (-g or -H*g).
        4. Performs a backtracking line search (if ADAPTIVE) to find a valid step length.
        5. Updates x_k and z_k.
        """
        if self.callback:
            self.callback(self.z_k)

        # 1. Estimate Gradient
        self.gdt_est = self.grad_estimator(self.x_k)
        
        # 2. BFGS Update (if enabled and not first step)
        if self.bfgs and self.k > 0:
            if np.isnan(self.bfgs_hinv).any():
                self.bfgs_hinv = np.identity(self.D)
                
            sk = np.atleast_2d(self.x_k - self.x_kp).T
            yk = np.atleast_2d(self.gdt_est - self.bfgs_gdtp).T
            
            if not (yk.T @ sk == 0.0).all():
                rho = 1 / (yk.T @ sk)
                I = np.identity(self.D)
                # BFGS inverse Hessian update
                term1 = (I - rho * sk @ yk.T)
                term2 = (I - rho * yk @ sk.T)
                self.bfgs_hinv = term1 @ self.bfgs_hinv @ term2 + (rho * sk @ sk.T)
        
        self.bfgs_gdtp = self.gdt_est.copy()
        self.x_kp = self.x_k.copy()
        
        # 3. Determine Search Direction
        if self.bfgs:
            p_k = -self.bfgs_hinv @ self.gdt_est
        else:
            p_k = -self.gdt_est

        # 4. Line Search
        if self.eta_mode == StepSizeMode.ADAPTIVE:
            # Active Line Search Loop
            ls_iter = 0
            while True:
                ls_iter += 1
                if ls_iter > 100:
                    # Safety break
                    break
                
                x_next = self.x_k + self.eta * p_k
                z_next = self.fun(x_next)
                
                # Update estimator history with the new point
                if np.array_equal(x_next, self.x_k):
                    self.z_k = z_next
                self.grad_estimator.update(x_next, z_next)

                # Check Armijo condition using the current gradient estimate.
                descent_term = norm(self.gdt_est)**2
                
                if z_next <= self.z_k - self.etaT * self.eta * descent_term:
                    # Accepted
                    self.x_k = x_next
                    self.z_k = z_next
                    self.k += 1
                    self.eta = self.eta / self.etaM
                    if self.callback:
                        self.callback(self.z_k)
                    break
                else:
                    # Rejected
                    if self.eta > 1e-6:
                        self.eta = self.eta * self.etaM
                    else:
                        # Reset
                        # Recompute gradient before restarting with the reset step size.
                        self.gdt_est = self.grad_estimator(self.x_k, force=True)
                        self.eta = self.eta0
                        # Note: Reset eta before computing x_n = x_k - eta * g.
                        # Since we loop, we update p_k below and then x_next will be computed with eta0.
                    
                    # Update search direction with NEW gradient
                    if self.bfgs:
                         p_k = -self.bfgs_hinv @ self.gdt_est
                    else:
                         p_k = -self.gdt_est
                    
                    if self.callback:
                        self.callback(self.z_k)
                         
                    # Continue loop to try new x_next with new p_k and new/old eta
        else:
            # Fixed step size
            self.x_k = self.x_k + self.eta0 * p_k
            self.z_k = self.fun(self.x_k)
            self.grad_estimator.update(self.x_k, self.z_k)
            self.k += 1
            if self.callback:
                self.callback(self.z_k)
            
    def run(self, max_evals: int):
        """Run until max evaluations reached (approximate control)."""
        if max_evals <= 0:
            return

        history = getattr(self.grad_estimator, "history", None)
        if history is not None:
            while history.Zn.size < max_evals:
                self.step()
        else:
            for _ in range(max_evals):
                self.step()
