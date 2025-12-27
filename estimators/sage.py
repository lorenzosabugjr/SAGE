import numpy as np
import scipy as cp
from numpy.linalg import norm
from typing import Callable, Optional
from .base import BaseGradientEstimator
from utils.history import HistoryBuffer

class SAGE(BaseGradientEstimator):
    """
    Set-membership Active Gradient Estimator (SAGE).

    SAGE is a robust, data-efficient gradient estimator designed for noisy black-box optimization.
    Unlike finite difference methods which are stateless, SAGE maintains a history of past
    function evaluations to construct a "consistency set" (a polytope) that contains the true gradient.

    If the uncertainty in the gradient estimate (the diameter of the consistency set) is too large,
    SAGE actively samples new points in directions that maximally reduce this uncertainty.

    Attributes:
        fun (Callable): The black-box objective function f(x).
        dim (int): Dimensionality of the input space.
        autonoise (bool): If True, attempts to estimate noise level alongside gradient.
        quickmode (bool): If True, uses a local subset of samples for faster LP solving.
        Xn (np.ndarray): History of evaluated points (N x dim).
        Zn (np.ndarray): History of function values (N,).
        history (HistoryBuffer): Optional shared history buffer.
    """
    def __init__(
        self,
        fun: Callable[[np.ndarray], float],
        dim: int,
        noise_param: float = 0.0,
        autonoise: bool = True,
        quickmode: bool = True,
        initial_history: Optional[tuple[np.ndarray, np.ndarray]] = None,
        history: Optional[HistoryBuffer] = None,
        diam_mode: Optional[str] = None,
        callback: Optional[Callable[[], None]] = None,
        init_step: float = 1e-3,
    ):
        """
        Initialize the SAGE estimator.

        Args:
            fun: The objective function to estimate gradients for.
            dim: The dimension of the input vector x.
            noise_param: Noise bound used when autonoise is False (or an initial value otherwise).
            autonoise: Whether to automatically estimate the noise bound.
            quickmode: Whether to use a subset of neighbors for faster computation.
            initial_history: Optional tuple (X, Z) of past evaluations to seed the history.
            history: Optional shared HistoryBuffer used to collect evaluations.
            diam_mode: "exact" or "approx". Defaults to "approx" when quickmode is True.
            callback: Optional callback invoked after each auxiliary evaluation.
            init_step: Step size used to seed an initial simplex when history is empty.
        """
        super().__init__(fun, dim, history=history)
        self.autonoise = autonoise
        self.quickmode = quickmode
        self.callback = callback
        self.init_step = init_step
        self.recompute_on_update = True
        self.center_sample_on_move = True
        if diam_mode is None:
            self.diam_mode = "approx" if quickmode else "exact"
        else:
            self.diam_mode = diam_mode

        if self.history is None:
            self.history = HistoryBuffer()

        if initial_history is not None:
            self.history.add_batch(initial_history[0], initial_history[1])

        self.Xn, self.Zn = self.history.snapshot()
        
        # Internal state for gradient estimation
        self.gdtset_diaid = 0.05   # Ideal gradient set diameter
        self.gdtset_diath = 0.05   # Current threshold
        
        # If autonoise is True, noise bound is estimated by the LP
        # But we initialize it to noise_param (if provided) for consistency
        self.ns_est = noise_param
        
        self.gdt_est = np.zeros(dim)
        self.hess_norm = 0.0
        self.hess_lipsc = 0.0
        
        self.A2 = None
        self.b2 = None
        self.gd_v = np.nan
        self.gd_vm = np.inf
        
        self.gdt_est_frc = False
        self.hist_aux_samples = np.empty((0,))

        self._last_update_n = None
        self._last_update_x = None
        self._center_sample_pending = False
        self.x_current = None
        
        # Tracking aux samples for the current estimation step
        self.aux_samples_count = 0
        
        # Perform initial gradient update when history is pre-seeded
        if self.Xn.size > 0:
            best_idx = np.argmin(self.Zn)
            self._recompute_at(self.Xn[best_idx])

    def _recompute_at(self, x: np.ndarray) -> None:
        self.x_current = x
        self._grad_est_lp(x)
        self._calc_diam()
        self._last_update_n = self.history.Zn.size
        self._last_update_x = self.x_current.copy()

    def _grad_est_lp(self, x: np.ndarray):
        """
        Constructs and solves the Linear Program (LP) to find the gradient estimate.

        This method:
        1. Identifies relevant samples (neighbors) around x.
        2. Constructs constraints based on the Lipschitz continuity and noise bounds.
           |f(y) - f(x) - g^T(y-x)| <= L/2 ||y-x||^2 + noise
        3. Solves the LP to find the gradient g, Hessian norm L, and noise M/e.
        4. Updates the consistency set polytopes (self.A2, self.b2).

        Args:
            x: The point at which to estimate the gradient.
        """
        D = self.dim
        
        # 1. Identify relevant samples
        self._sync_history()
        x_in = np.all(np.equal(self.Xn, x), axis=1)
        if not np.any(x_in):
            # If x is not in history, evaluate it
            z = self.fun(x)
            self._add_sample(x, z)
            x_idx = [self.Zn.size - 1]
        else:
            x_idx = np.nonzero(x_in)[0]

        # 2. Select neighbors (Quickmode logic)
        if self.Zn.size > 5*D + 1 and self.quickmode:
            coll_x   = [self.Xn[j] for j in range(self.Zn.size) if j not in x_idx]
            coll_idx_raw = [j for j in range(self.Zn.size) if j not in x_idx]
            
            # Root finding for optimal radius
            aa = 1 / 3 * self.hess_lipsc
            bb = 1 / 2 * self.hess_norm
            dd = -2 * self.ns_est
            rt = np.roots([aa, bb, 0, dd])
            alpha_roots = rt[np.isreal(rt) & (rt.real >= 0)]
            if alpha_roots.size == 0:
                alpha = 0
            else:
                alpha = alpha_roots.real[0]

            cost_fn = np.abs(np.sum((coll_x - x)**2, axis=1) - alpha**2)
            sort_idx = np.argsort(cost_fn)[:5*D]
            coll_idx = [coll_idx_raw[j] for j in sort_idx]
        else:
            coll_idx = [j for j in range(self.Zn.size) if j not in x_idx]

        # 3. Build LP Matrices
        A = np.empty([0, D + 2 + 1 * self.autonoise])
        b = np.empty([0, 1])
        
        z_curr = self.Zn[x_idx[0]]
        
        for j in coll_idx:
            dij = np.linalg.norm(self.Xn[j] - x)
            if dij == 0.0: dij = 1.0
            uij = (self.Xn[j] - x) / dij
            gij = (self.Zn[j] - z_curr) / dij
            
            # Constraints
            if not self.autonoise:
                # Cols: [g (D), L (1), M (1)]
                row1 = np.hstack((-uij, -0.5 * dij, -1 / 6 * dij**2))
                row2 = np.hstack((uij, -0.5 * dij, -1 / 6 * dij**2))
                A = np.vstack((A, row1, row2))
            else:
                # Cols: [g (D), L (1), M (1), e (1)]
                row1 = np.hstack((-uij, -0.5 * dij, -1 / 6 * dij**2, -2 / dij))
                row2 = np.hstack((uij, -0.5 * dij, -1 / 6 * dij**2, -2 / dij))
                A = np.vstack((A, row1, row2))
                
            b = np.vstack((b, -gij, gij))

        # 4. Solve LP
        if not self.autonoise:
            # Bounds for L and M are >= 0. g is unbounded.
            # A_ub * [g, L, M]^T <= b_ub
            # We want to minimize nothing? Or just find feasible point?
            # Original used linprog with obj [0...0, 1, 1] to minimize L+M?
            Ae = np.vstack((A, np.hstack((np.zeros(A.shape[1] - 2), -1, 0))))
            Ae = np.vstack((Ae, np.hstack((np.zeros(A.shape[1] - 2), 0, -1))))
            be = np.vstack((b, 0, 0)).flatten()
            
            c = np.hstack((np.zeros(Ae.shape[1] - 2), 1, 1))
            res = cp.optimize.linprog(c, A_ub=Ae, b_ub=be, bounds=(None, None), method="highs-ipm")
            
            if res.success:
                self.gdt_est = res.x[:-2]
                self.hess_norm = res.x[-2]
                self.hess_lipsc = np.max([res.x[-1], self.hess_lipsc])
        else:
            Ae = np.vstack((A, np.hstack((np.zeros(A.shape[1] - 3), -1, 0, 0))))
            Ae = np.vstack((Ae, np.hstack((np.zeros(A.shape[1] - 3), 0, -1, 0))))
            Ae = np.vstack((Ae, np.hstack((np.zeros(A.shape[1] - 3), 0, 0, -1))))
            be = np.vstack((b, 0, 0, 0)).flatten()
            
            c = np.hstack((np.zeros(Ae.shape[1] - 3), 1, 1, 1))
            res = cp.optimize.linprog(c, A_ub=Ae, b_ub=be, bounds=(None, None))
            
            if res.success:
                self.gdt_est = res.x[:-3]
                self.hess_norm = res.x[-3]
                self.hess_lipsc = np.max([res.x[-2], self.hess_lipsc])
                self.ns_est = res.x[-1]

        # 5. Construct Gradient Set Polytopes (A2, b2)
        # Used for diameter calculation
        self.Al = A[:, 0 : D]
        Ar = A[:, D :]
        self.A2 = np.vstack((
            np.hstack((self.Al, np.zeros(self.Al.shape))),
            np.hstack((np.zeros(self.Al.shape), self.Al))
        ))
        
        if not self.autonoise:
            params = np.array([self.hess_norm, self.hess_lipsc])
        else:
            params = np.array([self.hess_norm, self.hess_lipsc, self.ns_est])
            
        self.bl = b.flatten() - (Ar @ params)
        self.b2 = np.vstack((self.bl, self.bl)).flatten()


    def _calc_diam(self):
        if self.diam_mode == "approx":
            return self._calc_diam_approx()
        return self._calc_diam_exact()

    def _calc_diam_exact(self):
        """
        Calculates the diameter of the current gradient consistency set.

        The diameter is defined as the maximum distance between any two gradients
        that satisfy the consistency constraints. This is solved as a maximization
        problem (or minimization of negative distance) over the polytope defined
        by self.A2 and self.b2.

        Returns:
            float: The diameter of the set (scalar).
        """
        D = self.dim
        if self.A2 is None:
            return np.inf

        P = np.vstack((
            np.hstack((np.identity(D), -np.identity(D))),
            np.hstack((-np.identity(D), np.identity(D)))
        ))

        def obj(x):
            return -x.T @ P @ x

        # Ensure gdt_est is shaped correctly
        x0 = np.hstack((self.gdt_est, self.gdt_est)) + 1e-3 * np.random.rand(2 * D)
        
        cons = {"type": "ineq", "fun": lambda x: -(self.A2 @ x - self.b2)}
        res = cp.optimize.minimize(
            obj, x0, method="SLSQP", constraints=cons, options={"disp": False}
        )

        self.gd_v = res.x[:D] - res.x[D:]
        self.gd_vm = np.linalg.norm(self.gd_v)
        return self.gd_vm

    def _calc_diam_approx(self):
        """
        Fast approximate diameter using axis-aligned bounds from LPs.

        This computes a bounding-box diameter, which is an upper bound on the true
        diameter and avoids the non-convex SLSQP solve.
        """
        D = self.dim
        if self.Al is None or self.bl is None or self.Al.size == 0:
            return np.inf

        max_g = np.empty(D)
        min_g = np.empty(D)

        for i in range(D):
            c = np.zeros(D)
            c[i] = -1.0
            res = cp.optimize.linprog(c, A_ub=self.Al, b_ub=self.bl, bounds=(None, None), method="highs")
            if not res.success:
                self.gd_vm = np.inf
                return self.gd_vm
            max_g[i] = -res.fun

            c[i] = 1.0
            res = cp.optimize.linprog(c, A_ub=self.Al, b_ub=self.bl, bounds=(None, None), method="highs")
            if not res.success:
                self.gd_vm = np.inf
                return self.gd_vm
            min_g[i] = res.fun

        self.gd_v = max_g - min_g
        self.gd_vm = np.linalg.norm(self.gd_v)
        return self.gd_vm

    def __call__(self, x: np.ndarray, force: bool = False) -> np.ndarray:
        """
        Estimate the gradient at point x.
        """
        # Ensure x is in history
        self._sync_history()
        sample_added = False
        if self.Xn.size == 0:
            x0 = np.asarray(x)
            self._add_sample(x0, self.fun(x0))
            for i in range(self.dim):
                x_step = x0.copy()
                x_step[i] += self.init_step
                self._add_sample(x_step, self.fun(x_step))
            self._sync_history()
            sample_added = True
        if not np.any(np.all(np.equal(self.Xn, x), axis=1)):
            self._add_sample(x, self.fun(x))
            sample_added = True
            self._sync_history()

        history_changed = self._last_update_n is None or self._last_update_n != self.history.Zn.size
        x_changed = self.x_current is None or not np.array_equal(self.x_current, x)
        needs_recompute = force or sample_added or history_changed or x_changed

        if needs_recompute:
            self._recompute_at(x)
        else:
            self.x_current = x

        if self._last_update_x is None or not np.array_equal(x, self._last_update_x):
            return self.gdt_est

        self.aux_samples_count = 0

        while True:
            # 1. Get current diameter (already calculated if we just updated or entered)
            # We assume gdt_est and gd_v are current for self.x_current
            diam = self.gd_vm 
            
            # 2. Check Termination Conditions
            if diam < self.gdtset_diath or self.gdt_est_frc or (self.aux_samples_count >= 2.5 * self.dim):
                self.hist_aux_samples = np.hstack((self.hist_aux_samples, self.aux_samples_count))
                self.aux_samples_count = 0
                self.gdt_est_frc = False
                break
            
            # 3. Active Sampling (Refinement)
            if self.ns_est <= 1e-9:
                alpha = 1e-6
                self.gdtset_diath = self.gdtset_diaid
            else:
                aa = 1 / 3 * self.hess_lipsc
                bb = 1 / 2 * self.hess_norm
                dd = -2 * self.ns_est
                rt = np.roots([aa, bb, 0, dd])
                roots = rt[np.isreal(rt) & (rt.real >= 0)]
                if len(roots) > 0:
                    alpha = roots.real[0]
                else:
                    alpha = 1e-6
                self.gdtset_diath = 1.01 * alpha

            # Next sample point
            x_new = x + alpha * self.gd_v / norm(self.gd_v)
            
            z_new = self.fun(x_new)
            self.aux_samples_count += 1
            
            # Update history
            self.update(x_new, z_new)
            
            if self.callback:
                self.callback()
            
        return self.gdt_est

    def next_aux_sample(self, x: np.ndarray) -> Optional[np.ndarray]:
        """
        Decide whether to take a single auxiliary sample for gradient refinement.

        Returns:
            The next auxiliary sample point if refinement is needed, otherwise None.
        """
        if self._center_sample_pending:
            return x

        if self.gd_vm < self.gdtset_diath or self.gdt_est_frc:
            self.hist_aux_samples = np.hstack((self.hist_aux_samples, self.aux_samples_count))
            self.aux_samples_count = 0
            self.gdt_est_frc = False
            return None

        if self.ns_est <= 1e-9:
            alpha = 1e-6
            self.gdtset_diath = self.gdtset_diaid
        else:
            aa = 1 / 3 * self.hess_lipsc
            bb = 1 / 2 * self.hess_norm
            dd = -2 * self.ns_est
            rt = np.roots([aa, bb, 0, dd])
            roots = rt[np.isreal(rt) & (rt.real >= 0)]
            if len(roots) > 0:
                alpha = roots.real[0]
            else:
                alpha = 1e-6
            self.gdtset_diath = 1.01 * alpha

        x_new = x + alpha * self.gd_v / norm(self.gd_v)
        self.aux_samples_count += 1
        if self.aux_samples_count >= 2.5 * self.dim:
            self.gdt_est_frc = True

        return x_new

    def _add_sample(self, x: np.ndarray, z: float):
        self.history.add(x, z)

    def update(self, x: np.ndarray, z: float):
        self._add_sample(x, z)
        self._sync_history()
        if hasattr(self, "x_current") and self.x_current is not None:
            if np.array_equal(x, self.x_current):
                self._center_sample_pending = False
        # Update gradient estimate and diameter at the current center point
        # This matches the behavior where every add_samples triggers update_gradient(x_k)
        if hasattr(self, 'x_current') and self.x_current is not None:
            self._recompute_at(self.x_current)

    def _sync_history(self):
        self.Xn, self.Zn = self.history.snapshot()
