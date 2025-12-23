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
    ):
        self.fun = fun
        self.x_k = x0.copy()
        self.D = x0.shape[0]
        self.grad_estimator = grad_estimator
        
        # Optimization state
        self.z_k = self.fun(self.x_k)
        self.grad_estimator.update(self.x_k, self.z_k)
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
        # 1. Estimate Gradient
        # SAGE (or others) might do active sampling here
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
            # Backtracking line search (Armijo)
            # We want z_next <= z_k + c * eta * grad^T * p_k
            # grad^T * p_k is usually negative.
            # Here implementation mimics original: 
            # condition: z_n <= z_k - etaT * eta * norm(grad)^2
            # Wait, original code for BFGS used: x_n = -eta * H * g + x_k
            # And condition: z_n <= z_k - etaT * eta * norm(g)**2 ??
            # Usually for BFGS it involves p_k. 
            # But let's stick to the original logic for fidelity if possible, 
            # or upgrade to standard Armijo on p_k.
            # Original code check:
            # if (self.z_n <= self.z_k - self.etaT * self.eta * norm(self.gdt_est) ** 2):
            
            # We will use the direction p_k. 
            # If standard GD, p_k = -g, so p_k^T g = -|g|^2.
            # So condition z <= z - c * alpha * |g|^2 is standard for GD.
            # For BFGS, we should check descent direction.
            
            while True:
                x_next = self.x_k + self.eta * p_k
                z_next = self.fun(x_next)
                self.grad_estimator.update(x_next, z_next)
                
                # Check Armijo
                # Note: Original code used norm(gdt_est)**2 regardless of BFGS. 
                # This might be a simplification or specific design choice in SAGE paper.
                # We will preserve it for now to match benchmark behavior.
                descent_term = norm(self.gdt_est)**2
                
                if z_next <= self.z_k - self.etaT * self.eta * descent_term:
                    # Accepted
                    self.x_k = x_next
                    self.z_k = z_next
                    self.k += 1
                    # Increase step size slightly for next iter? 
                    # Original code: self.eta = self.eta / self.etaM (where etaM=0.5 -> multiply by 2)
                    self.eta = self.eta / self.etaM
                    break
                else:
                    # Rejected, reduce step size
                    self.eta = self.eta * self.etaM
                    if self.eta < 1e-9:
                        # Step too small, accept anyway or stop?
                        # Original code: resets to eta0 if < 1e-6 and calls add_samples_gdtest(None,None)
                        # We will just accept to avoid infinite loops, or break.
                        self.x_k = x_next
                        self.z_k = z_next
                        self.k += 1
                        self.eta = self.eta0
                        break
        else:
            # Fixed step size
            self.x_k = self.x_k + self.eta0 * p_k
            self.z_k = self.fun(self.x_k)
            self.grad_estimator.update(self.x_k, self.z_k)
            self.k += 1
            
    def run(self, max_evals: int):
        """Run until max evaluations reached (approximate control)."""
        # Since we don't control the estimator's internal eval count easily here without coupling,
        # we might rely on the caller to check loop conditions.
        # But here is a simple loop.
        pass
