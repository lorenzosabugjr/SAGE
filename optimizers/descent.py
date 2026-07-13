from enum import Enum, unique
import numpy as np
from numpy.linalg import norm
from typing import Callable, Optional
from estimators.base import BaseGradientEstimator


@unique
class StepSizeMode(Enum):
    FIXED = 0
    ADAPTIVE = 1


class GradientDescent:
    """
    Plain gradient descent with an adaptive Armijo line search.

    Delegates gradient estimation to a BaseGradientEstimator. The search
    direction is always ``p = -g`` (no BFGS/quasi-Newton support).
    """
    def __init__(
        self,
        fun: Callable[[np.ndarray], float],
        x0: np.ndarray,
        grad_estimator: BaseGradientEstimator,
        stepsize: float = 1.0,
        stepsizemode: StepSizeMode = StepSizeMode.ADAPTIVE,
        armijo_beta: float = 0.5,
        armijo_c: float = 1e-6,
        min_stepsize: float = 1e-6,
        max_line_search_iters: int = 100,
        recompute_grad_every_ls_failures: int = 5,
        reset_stepsize_at_floor: bool = True,
        z0: Optional[float] = None,
        callback: Optional[Callable[[float], None]] = None,
        verbose: bool = False,
    ):
        self.fun = fun
        self.x_k = x0.copy()
        self.D = x0.shape[0]
        self.grad_estimator = grad_estimator
        self.callback = callback
        self.verbose = verbose

        # Optimization state
        if z0 is not None:
            self.z_k = z0
        else:
            self.z_k = self.fun(self.x_k)

        self.k = 0

        # Step size parameters
        self.eta0 = stepsize
        self.eta = self.eta0
        self.eta_mode = stepsizemode
        self.armijo_beta = armijo_beta
        self.armijo_c = armijo_c
        self.min_stepsize = min_stepsize
        self.max_line_search_iters = max_line_search_iters
        self.recompute_grad_every_ls_failures = recompute_grad_every_ls_failures
        self.reset_stepsize_at_floor = reset_stepsize_at_floor

        self.gdt_est = np.zeros(self.D)

    def _mark_incumbent_accepted(self) -> None:
        """Tell the shared history (if any) that the most recently recorded
        raw evaluation is the newly accepted iterate, so that evaluation's
        incumbent-history entry reflects it in place instead of one
        evaluation late."""
        history = getattr(self.grad_estimator, "history", None)
        if history is not None:
            history.accept_incumbent()

    def step(self):
        """
        Perform a single optimization step.

        1. Estimate the gradient at the current point x_k using the estimator.
        2. Compute the search direction p_k = -g.
        3. Perform a backtracking line search (if ADAPTIVE) to find a valid step length.
        4. Update x_k and z_k.
        """
        if self.callback:
            self.callback(self.z_k)

        # 1. Estimate Gradient
        self.gdt_est = self.grad_estimator(self.x_k)

        # 2. Search Direction
        p_k = -self.gdt_est

        # 3. Line Search
        if self.eta_mode == StepSizeMode.ADAPTIVE:
            ls_iter = 0
            ls_fail_count = 0
            while True:
                ls_iter += 1
                if ls_iter > self.max_line_search_iters:
                    # Safety break: return without accepting a step or
                    # incrementing the iteration counter.
                    if self.verbose:
                        n_hist = getattr(getattr(self.grad_estimator, "history", None), "Zn", np.empty(0)).size
                        print(
                            f"[OPT-DIAG] step#{self.k}  ls_iter={self.max_line_search_iters}(BREAK)  "
                            f"eta={self.eta:.4e}  z_k={self.z_k:.6e}  hist={n_hist}",
                            flush=True,
                        )
                    break

                x_next = self.x_k + self.eta * p_k
                z_next = self.fun(x_next)

                # Update estimator history with the new point
                if np.array_equal(x_next, self.x_k):
                    self.z_k = z_next
                self.grad_estimator.update(x_next, z_next, lightweight=True)

                # Check Armijo condition using the current gradient estimate.
                descent_term = norm(self.gdt_est) ** 2

                if z_next <= self.z_k - self.armijo_c * self.eta * descent_term:
                    # Accepted
                    self.x_k = x_next
                    self.z_k = z_next
                    self.k += 1
                    self.eta = self.eta / self.armijo_beta
                    self._mark_incumbent_accepted()
                    if self.verbose:
                        n_hist = getattr(getattr(self.grad_estimator, "history", None), "Zn", np.empty(0)).size
                        print(
                            f"[OPT-DIAG] step#{self.k}  ls_iter={ls_iter}(OK)  "
                            f"eta={self.eta:.4e}  z_k={self.z_k:.6e}  hist={n_hist}",
                            flush=True,
                        )
                    if self.callback:
                        self.callback(self.z_k)
                    break
                else:
                    # Rejected — shrink eta and track failures
                    self.eta = self.eta * self.armijo_beta
                    ls_fail_count += 1

                    # Recalc gradient every N failures or at the eta floor
                    need_recalc = (
                        (ls_fail_count % self.recompute_grad_every_ls_failures == 0)
                        or (self.eta <= self.min_stepsize)
                    )
                    if need_recalc:
                        self.gdt_est = self.grad_estimator(self.x_k, force=True)

                    # Reset eta to eta0 at the floor
                    if self.eta <= self.min_stepsize and self.reset_stepsize_at_floor:
                        self.eta = self.eta0

                    # Update search direction with (possibly refreshed) gradient
                    p_k = -self.gdt_est

                    if self.callback:
                        self.callback(self.z_k)
        else:
            # Fixed step size
            self.x_k = self.x_k + self.eta0 * p_k
            self.z_k = self.fun(self.x_k)
            # Mark accepted before update(): a non-lightweight update() can
            # trigger further (observation-only) evaluations, which must
            # not be mistaken for this accepted evaluation's own row.
            self._mark_incumbent_accepted()
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
