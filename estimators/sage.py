import numpy as np
import scipy as cp
import time
import sys
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
        quickmode (bool): If True, uses a local subset of samples for faster LP solving.
        Xn (np.ndarray): History of evaluated points (N x dim).
        Zn (np.ndarray): History of function values (N,).
        history (HistoryBuffer): Optional shared history buffer.
    """
    def __init__(
        self,
        fun: Callable[[np.ndarray], float],
        dim: int,
        quickmode: bool = True,
        initial_history: Optional[tuple[np.ndarray, np.ndarray]] = None,
        history: Optional[HistoryBuffer] = None,
        diam_mode: Optional[str] = None,
        callback: Optional[Callable[[], None]] = None,
        init_step: float = 1e-6,
        reset_on_step: bool = False,
    ):
        """
        Initialize the SAGE estimator.

        Args:
            fun: The objective function to estimate gradients for.
            dim: The dimension of the input vector x.
            quickmode: Whether to use a subset of neighbors for faster computation.
            initial_history: Optional tuple (X, Z) of past evaluations to seed the history.
            history: Optional shared HistoryBuffer used to collect evaluations.
            diam_mode: "exact" or "approx". Defaults to "approx" when quickmode is True.
            callback: Optional callback invoked after each auxiliary evaluation.
            init_step: Step size used to seed a CFD-like stencil when history has 0 or 1 samples.
        """
        super().__init__(fun, dim, history=history)
        self.quickmode = quickmode
        self.callback = callback
        self.init_step = init_step
        self.reset_on_step = reset_on_step
        if diam_mode is None:
            self.diam_mode = "approx" if quickmode else "exact"
        else:
            self.diam_mode = diam_mode

        if self.history is None:
            self.history = HistoryBuffer()
        self._shared_history = self.history  # keep ref for budget tracking

        if initial_history is not None:
            self.history.add_batch(initial_history[0], initial_history[1])

        self.Xn, self.Zn = self.history.snapshot()
        
        # Internal state for gradient estimation
        self.gdtset_diaid = 0.05   # Ideal gradient set diameter
        self.gdtset_diath = 0.05   # Current threshold
        
        # Noise bound is estimated by the LP; initialize to zero.
        self.ns_est = 0.0
        
        self.gdt_est = np.zeros(dim)
        self.hess_norm = 0.0
        self.hess_lipsc = 0.0
        
        self.Al = None
        self.bl = None
        self.A2 = None
        self.b2 = None
        self.gd_v = np.nan
        self.gd_vm = np.inf
        
        self.gdt_est_frc = False
        self.hist_aux_samples = np.empty((0,))
        self.aux_step_sizes_current = np.empty((0,))

        self._last_update_n = None
        self._last_update_x = None
        self.x_current = None
        self._pending_aux_feedback = None
        self._axis_probe_queue = []

        # Tracking aux samples for the current estimation step
        self.aux_samples_count = 0
        self._used_probe_axes = set()

        self._did_noise_reseed = False
        self._prev_gdt_est = None
        self._stable_count = 0

        # Active-sampling feedback thresholds
        self._directional_contraction_ratio_max = 0.95
        self._directional_alignment_tol = 0.995
        self.aux_log = []

        # ── Diagnostic counters ──
        self._diag_lp_count = 0          # total LP solves
        self._diag_lp_time = 0.0         # total LP wall-time
        self._diag_call_count = 0        # number of __call__ invocations
        self._diag_call_time = 0.0       # total __call__ wall-time
        self._diag_diam_lp_count = 0     # LP solves inside diameter calc
        self._diag_diam_lp_time = 0.0
        self._diag_update_count = 0      # update() calls
        self._diag_update_time = 0.0
        self._diag_enabled = True        # set False to silence
        
        # Perform initial gradient update when history is pre-seeded
        if self.Xn.size > 0:
            best_idx = np.argmin(self.Zn)
            self._recompute_at(self.Xn[best_idx])

    def _eval_and_record(self, x: np.ndarray) -> float:
        n_before = self.history.Zn.size
        z = self.fun(x)
        if self.history.Zn.size == n_before:
            self._add_sample(x, z)
        self._sync_history()
        return z

    def _seed_cfd_stencil_if_singleton(self, x: np.ndarray) -> None:
        if self.dim <= 0:
            return

        if self.history.Zn.size != 1:
            return

        # Symmetric coordinate-aligned seed (CFD-like stencil).
        # For each axis i, sample x + h*e_i and x - h*e_i.
        # This gives the LP tight, per-coordinate directional constraints
        # that pin down each gradient component via symmetric differences.
        # In the noiseless case this already yields near-machine-precision
        # accuracy; in the noisy case the adaptive refinement loop will
        # detect the large diameter and sample farther automatically.
        h = self.init_step
        for i in range(self.dim):
            e_i = np.zeros(self.dim)
            e_i[i] = 1.0
            self._eval_and_record(x + h * e_i)
            self._eval_and_record(x - h * e_i)

    def _reset_query_state(self) -> None:
        self.aux_log = []
        self.aux_step_sizes_current = np.empty((0,))
        self._pending_aux_feedback = None
        self._axis_probe_queue = []
        self._used_probe_axes = set()
        self.aux_samples_count = 0
        self._prev_gdt_est = None
        self._stable_count = 0
        self._did_noise_reseed = False
        # Reset per-iterate Hessian norm: H is local to each iterate, so it
        # should not carry over when jumping to a new query point.
        # In contrast, hess_lipsc (γ) is a global Lipschitz constant and
        # remains monotone non-decreasing across queries.
        self.hess_norm = 0.0

    def _ensure_query_sample(self, x: np.ndarray) -> bool:
        self._sync_history()

        if self.Xn.size == 0:
            n_before = self.history.Zn.size
            self._eval_and_record(x)
            self._seed_cfd_stencil_if_singleton(np.asarray(x))
            self._sync_history()
            return self.history.Zn.size != n_before

        if self.history.find_indices(x).size == 0:
            n_before = self.history.Zn.size
            self._eval_and_record(x)
            self._sync_history()
            return self.history.Zn.size != n_before

        return False

    def _recompute_at(self, x: np.ndarray) -> None:
        self.x_current = x
        _lp_t0 = time.perf_counter()
        self._grad_est_lp(x)
        _lp_dt = time.perf_counter() - _lp_t0
        _diam_t0 = time.perf_counter()
        self._calc_diam()
        _diam_dt = time.perf_counter() - _diam_t0
        if self._diag_enabled and (_lp_dt > 1.0 or _diam_dt > 1.0):
            print(
                f"[SAGE-RECOMPUTE] lp_t={_lp_dt:.4f}s  diam_t={_diam_dt:.4f}s  "
                f"hist={self.history.Zn.size}",
                flush=True,
            )
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
        x_idx = self.history.find_indices(x)
        if x_idx.size == 0:
            # If x is not in history, evaluate it
            self._eval_and_record(x)
            x_idx = self.history.find_indices(x)

        # Seed a CFD-like coordinate stencil if only one sample exists.
        self._seed_cfd_stencil_if_singleton(x)
        self._sync_history()
        x_idx = self.history.find_indices(x)

        # 2. Select neighbors
        coll_x   = [self.Xn[j] for j in range(self.Zn.size) if j not in x_idx]
        coll_idx_raw = [j for j in range(self.Zn.size) if j not in x_idx]

        # Compute optimal sampling radius for neighbor selection
        aa = 1 / 3 * self.hess_lipsc
        bb = 1 / 2 * self.hess_norm
        dd = -2 * self.ns_est
        rt = np.roots([aa, bb, 0, dd])
        alpha_roots = rt[np.isreal(rt) & (rt.real >= 0)]
        if alpha_roots.size == 0:
            alpha = 2.0 * np.sqrt(max(self.ns_est, 1e-30))
        else:
            alpha = float(alpha_roots.real[0])

        # Distance band filter: when noise is detected, exclude samples
        # whose distance from x is so far from alpha that their LP
        # constraints are noise-dominated (too close) or curvature-
        # dominated (too far).  This prevents tiny-step seed samples
        # from poisoning the LP with enormous slab widths.
        if self.ns_est > 1e-9 and len(coll_x) > 0:
            coll_dists = np.array([norm(cx - x) for cx in coll_x])
            band_lo = alpha / 10.0
            band_hi = alpha * 10.0
            in_band = (coll_dists >= band_lo) & (coll_dists <= band_hi)
            if np.sum(in_band) >= D + 1:
                coll_x = [coll_x[j] for j in range(len(coll_x)) if in_band[j]]
                coll_idx_raw = [coll_idx_raw[j] for j in range(len(coll_idx_raw)) if in_band[j]]

        # Quickmode: keep at most 5D nearest-to-alpha samples
        if len(coll_x) > 5*D and self.quickmode:
            cost_fn = np.abs(np.sum((np.array(coll_x) - x)**2, axis=1) - alpha**2)
            sort_idx = np.argsort(cost_fn)[:5*D]
            coll_idx = [coll_idx_raw[j] for j in sort_idx]
        else:
            coll_idx = list(coll_idx_raw)

        # [DEEP-DIAG] sample geometry at LP construction
        if self._diag_enabled:
            dists = np.array([norm(self.Xn[j] - x) for j in coll_idx]) if coll_idx else np.array([])
            print(
                f"[LP-DIAG] call#{self._diag_call_count}  "
                f"n_samples={len(coll_idx)}  alpha={alpha:.4e}  "
                f"hess_lipsc_in={self.hess_lipsc:.4e}  "
                f"hess_norm_in={self.hess_norm:.4e}  ns_in={self.ns_est:.4e}  "
                f"d_min={dists.min():.4e}  d_max={dists.max():.4e}  "
                f"d_median={np.median(dists):.4e}  "
                f"n_close(d<1e-4)={np.sum(dists<1e-4)}  "
                f"n_far(d>1)={np.sum(dists>1)}",
                flush=True,
            )

        # 3. Build LP Matrices
        rows = []
        rhs = []
        z_curr = self.Zn[x_idx[0]]
        
        for j in coll_idx:
            dij = np.linalg.norm(self.Xn[j] - x)
            if dij == 0.0: dij = 1.0
            uij = (self.Xn[j] - x) / dij
            gij = (self.Zn[j] - z_curr) / dij
            
            # Cols: [g (D), L (1), M (1), e (1)]
            row1 = np.hstack((-uij, -0.5 * dij, -1 / 6 * dij**2, -2 / dij))
            row2 = np.hstack((uij, -0.5 * dij, -1 / 6 * dij**2, -2 / dij))
            rows.extend((row1, row2))
            rhs.extend((-gij, gij))

        A = np.asarray(rows, dtype=float).reshape(-1, D + 3)
        b = np.asarray(rhs, dtype=float).reshape(-1, 1)

        # 4. Solve LP
        nonneg_rows = np.zeros((3, A.shape[1]))
        nonneg_rows[0, -3] = -1.0
        nonneg_rows[1, -2] = -1.0
        nonneg_rows[2, -1] = -1.0
        Ae = np.vstack((A, nonneg_rows))
        be = np.concatenate((b.ravel(), np.zeros(3)))
        
        c = np.hstack((np.zeros(Ae.shape[1] - 3), 1, 1, 1))
        _lp_t0 = time.perf_counter()
        res = cp.optimize.linprog(
            c,
            A_ub=Ae,
            b_ub=be,
            bounds=(None, None),
            options={"output_flag": False},
        )
        _lp_dt = time.perf_counter() - _lp_t0
        self._diag_lp_count += 1
        self._diag_lp_time += _lp_dt
        
        if res.success:
            self.gdt_est = res.x[:-3]
            self.hess_norm = np.max([res.x[-3], self.hess_norm])
            self.hess_lipsc = np.max([res.x[-2], self.hess_lipsc])
            self.ns_est = res.x[-1]

            if self._diag_enabled:
                print(
                    f"[LP-RESULT] call#{self._diag_call_count}  "
                    f"H={self.hess_norm:.4e}  γ={self.hess_lipsc:.4e}  "
                    f"ε={self.ns_est:.4e}  |g|={norm(self.gdt_est):.4e}  "
                    f"lp_status={res.status}",
                    flush=True,
                )

            # Track gradient estimate stability for stagnation detection
            if self._prev_gdt_est is not None:
                g_norm = max(norm(self.gdt_est), 1e-30)
                change = norm(self.gdt_est - self._prev_gdt_est) / g_norm
                if change < 0.02:
                    self._stable_count += 1
                else:
                    self._stable_count = 0
            self._prev_gdt_est = self.gdt_est.copy()
        else:
            if self._diag_enabled:
                print(
                    f"[LP-RESULT] call#{self._diag_call_count}  "
                    f"LP FAILED  status={res.status}  msg={res.message}",
                    flush=True,
                )

        # 5. Construct Gradient Set Polytopes (A2, b2)
        # Used for diameter calculation
        self.Al = A[:, 0 : D]
        Ar = A[:, D :]
        
        params = np.array([self.hess_norm, self.hess_lipsc, self.ns_est])
            
        self.bl = b.flatten() - (Ar @ params)

        if self._diag_enabled:
            # Show slab widths: bl[2i] and bl[2i+1] are the pair for sample i
            # A positive bl means the constraint is "loose"
            print(
                f"[SLAB-DIAG] call#{self._diag_call_count}  "
                f"bl_min={self.bl.min():.4e}  bl_max={self.bl.max():.4e}  "
                f"bl_median={np.median(self.bl):.4e}  "
                f"n_positive_bl={np.sum(self.bl > 0)}/{self.bl.size}",
                flush=True,
            )

        if self.diam_mode == "exact":
            self.A2 = np.vstack((
                np.hstack((self.Al, np.zeros(self.Al.shape))),
                np.hstack((np.zeros(self.Al.shape), self.Al))
            ))
            self.b2 = np.vstack((self.bl, self.bl)).flatten()
        else:
            self.A2 = None
            self.b2 = None

    def _current_diameter_direction(self) -> Optional[np.ndarray]:
        if np.ndim(self.gd_v) != 1 or len(self.gd_v) != self.dim:
            return None
        v = np.array(self.gd_v, dtype=float)
        # Replace inf entries with a large finite value so the direction
        # points toward unbounded axes (the ones most in need of probing).
        inf_mask = ~np.isfinite(v)
        if np.all(inf_mask):
            # All axes unbounded — pick the first one
            v = np.zeros(self.dim)
            v[0] = 1.0
            return v
        if np.any(inf_mask):
            v[inf_mask] = 10.0 * np.max(np.abs(v[~inf_mask]))
            if np.max(np.abs(v)) < 1e-30:
                v[inf_mask] = 1.0
        n = norm(v)
        if n < 1e-30:
            return None
        return v / n

    def _solve_direction_bound(self, direction: np.ndarray, maximize: bool) -> float:
        c = -direction if maximize else direction
        _lp_t0 = time.perf_counter()
        res = cp.optimize.linprog(
            c,
            A_ub=self.Al,
            b_ub=self.bl,
            bounds=(None, None),
            method="highs",
            options={"output_flag": False},
        )
        _lp_dt = time.perf_counter() - _lp_t0
        self._diag_lp_count += 1
        self._diag_lp_time += _lp_dt
        self._diag_diam_lp_count += 1
        self._diag_diam_lp_time += _lp_dt
        if not res.success:
            return np.inf if maximize else -np.inf
        return float(-res.fun if maximize else res.fun)

    def _directional_width(self, direction: Optional[np.ndarray]) -> float:
        if direction is None or self.Al is None or self.bl is None or self.Al.size == 0:
            return np.inf

        direction = np.asarray(direction, dtype=float).reshape(-1)
        max_val = self._solve_direction_bound(direction, maximize=True)
        min_val = self._solve_direction_bound(direction, maximize=False)
        if not np.isfinite(max_val) or not np.isfinite(min_val):
            return np.inf
        return max(0.0, max_val - min_val)

    def _record_aux_feedback(
        self,
        pending: dict,
        prev_width: float,
        new_width: float,
        width_ratio: float,
        dir_cosine: float,
        informative: bool,
        needs_growth: bool,
    ) -> None:
        self.aux_log.append({
            "alpha_model": float(pending["alpha_model"]),
            "alpha_used": float(pending["alpha_used"]),
            "direction_source": pending["direction_source"],
            "axis_queue_len": int(len(self._axis_probe_queue)),
            "width_before": prev_width,
            "width_after": new_width,
            "width_ratio": width_ratio,
            "dir_cosine": dir_cosine,
            "informative": bool(informative),
            "needs_growth": bool(needs_growth),
        })

    def _compute_aux_step(self) -> tuple[float, float]:
        if self.ns_est <= 1e-9:
            model_alpha = self.init_step
            self.gdtset_diath = self.gdtset_diaid
        else:
            aa = 1 / 3 * self.hess_lipsc
            bb = 1 / 2 * self.hess_norm
            dd = -2 * self.ns_est
            rt = np.roots([aa, bb, 0, dd])
            roots = rt[np.isreal(rt) & (rt.real >= 0)]
            if len(roots) > 0:
                model_alpha = float(roots.real[0])
            else:
                # Curvature unresolvable from current samples (H ≈ 0, γ ≈ 0
                # but noise detected).  Use the optimal-α formula assuming
                # unit Hessian norm as bootstrap: α ≈ 2√ε.  Once the LP can
                # resolve curvature at this distance, the cubic takes over.
                model_alpha = 2.0 * np.sqrt(max(self.ns_est, 1e-30))
            self.gdtset_diath = 1.01 * model_alpha

        return model_alpha, float(model_alpha)

    def _queue_aux_feedback(
        self,
        direction: np.ndarray,
        alpha_used: float,
        alpha_model: float,
        direction_source: str,
    ) -> None:
        self._pending_aux_feedback = {
            "direction": np.array(direction, copy=True),
            "alpha_used": float(alpha_used),
            "alpha_model": float(alpha_model),
            "direction_source": direction_source,
            "width_before": float(self._directional_width(direction)),
        }

    def _select_probe_axis(
        self,
        reference_direction: Optional[np.ndarray],
    ) -> Optional[int]:
        if np.ndim(self.gd_v) != 1:
            return None

        widths = np.abs(np.asarray(self.gd_v, dtype=float).reshape(-1))
        if widths.size != self.dim:
            return None

        order = np.argsort(-widths)
        ref = None if reference_direction is None else np.asarray(reference_direction, dtype=float).reshape(-1)

        for axis in order:
            if widths[axis] <= 0.0 or int(axis) in self._used_probe_axes:
                continue
            if ref is not None and np.abs(ref[axis]) >= self._directional_alignment_tol:
                continue

            return int(axis)

        return None

    def _enqueue_axis_probe_pair(
        self,
        reference_direction: Optional[np.ndarray],
    ) -> bool:
        if self._axis_probe_queue:
            return True

        axis = self._select_probe_axis(reference_direction)
        if axis is None:
            return False

        direction = np.zeros(self.dim)
        direction[axis] = 1.0
        # Use a symmetric shell pair so the directional derivative is pinned by
        # matched far samples instead of a one-sided constraint.
        self._axis_probe_queue.extend((direction, -direction))
        self._used_probe_axes.add(int(axis))
        return True

    def _next_aux_direction(self) -> tuple[Optional[np.ndarray], str]:
        if self._axis_probe_queue:
            return self._axis_probe_queue.pop(0), "axis"

        return self._current_diameter_direction(), "diameter"

    def _consume_aux_feedback(self) -> None:
        if self._pending_aux_feedback is None:
            return

        pending = self._pending_aux_feedback
        self._pending_aux_feedback = None

        prev_dir = pending["direction"]
        prev_width = float(pending["width_before"])
        new_dir = self._current_diameter_direction()
        new_width = float(self._directional_width(prev_dir))

        if new_dir is None:
            dir_cosine = np.nan
        else:
            dir_cosine = float(np.clip(np.abs(prev_dir @ new_dir), 0.0, 1.0))

        if np.isfinite(prev_width) and prev_width > 1e-30 and np.isfinite(new_width):
            width_ratio = float(new_width / prev_width)
        else:
            width_ratio = np.nan

        contracted = np.isfinite(width_ratio) and width_ratio <= self._directional_contraction_ratio_max
        rotated = (
            np.isfinite(dir_cosine)
            and dir_cosine < self._directional_alignment_tol
            and np.isfinite(new_width)
            and (not np.isfinite(prev_width) or new_width <= prev_width)
        )
        informative = contracted or rotated
        needs_growth = (
            np.isfinite(width_ratio) and width_ratio > self._directional_contraction_ratio_max
            and (not np.isfinite(dir_cosine) or dir_cosine >= self._directional_alignment_tol)
        )
        # If width is still infinite, we definitely need axis probes —
        # the diameter direction probe didn't bound the axis at all.
        if not np.isfinite(new_width):
            needs_growth = True

        # If the diameter direction didn't contract, try axis-aligned probes
        # to reduce uncertainty along the widest coordinate axes.
        if needs_growth and not informative:
            self._enqueue_axis_probe_pair(prev_dir)

        self._record_aux_feedback(
            pending,
            prev_width,
            new_width,
            width_ratio,
            dir_cosine,
            informative,
            needs_growth,
        )

    def _should_stop_refinement(self) -> bool:
        pending_axis_pair = bool(self._axis_probe_queue)
        # Stagnation: gradient estimate hasn't changed by >2% for several
        # consecutive LP solves.  Further sampling is unlikely to help.
        stagnated = self._stable_count >= 3 and not pending_axis_pair
        return (
            (self.gd_vm < self.gdtset_diath and not pending_axis_pair)
            or self.gdt_est_frc
            or stagnated
            or (self.aux_samples_count >= 5.0 * self.dim and not pending_axis_pair)
        )

    def _finish_refinement(self) -> None:
        self.hist_aux_samples = np.hstack((self.hist_aux_samples, self.aux_samples_count))
        self.aux_samples_count = 0
        self.gdt_est_frc = False

    def _prepare_aux_sample(self, x: np.ndarray) -> Optional[np.ndarray]:
        alpha, model_alpha = self._compute_aux_step()
        direction, direction_source = self._next_aux_direction()
        if direction is None:
            return None

        x_new = x + alpha * direction
        self.aux_step_sizes_current = np.hstack((self.aux_step_sizes_current, float(alpha)))
        self._queue_aux_feedback(direction, alpha, model_alpha, direction_source)
        self.aux_samples_count += 1
        return x_new


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

        Important: we always compute ALL axes so that gd_v records which
        axes are unbounded (inf).  The refinement loop uses gd_v to decide
        where to probe next, so it must know about every axis.
        """
        D = self.dim
        if self.Al is None or self.bl is None or self.Al.size == 0:
            return np.inf

        max_g = np.empty(D)
        min_g = np.empty(D)

        for i in range(D):
            direction = np.zeros(D)
            direction[i] = 1.0
            max_g[i] = self._solve_direction_bound(direction, maximize=True)
            min_g[i] = self._solve_direction_bound(direction, maximize=False)

        self.gd_v = max_g - min_g
        # Guard against nan from inf - (-inf) or similar edge cases
        self.gd_v = np.where(np.isnan(self.gd_v), np.inf, self.gd_v)
        if np.all(np.isfinite(self.gd_v)):
            self.gd_vm = np.linalg.norm(self.gd_v)
        else:
            self.gd_vm = np.inf

        if self._diag_enabled:
            inf_axes = np.where(~np.isfinite(self.gd_v))[0]
            finite_widths = self.gd_v[np.isfinite(self.gd_v)]
            print(
                f"[DIAM-DIAG] call#{self._diag_call_count}  "
                f"diam={self.gd_vm:.4e}  "
                f"n_inf_axes={len(inf_axes)}/{D}  "
                f"inf_axes={inf_axes.tolist()[:10]}  "
                f"finite_w_range=[{finite_widths.min():.4e},{finite_widths.max():.4e}]"
                if finite_widths.size > 0 else
                f"[DIAM-DIAG] call#{self._diag_call_count}  "
                f"diam={self.gd_vm:.4e}  ALL axes unbounded",
                flush=True,
            )

        return self.gd_vm

    def _reseed_at_alpha(self, x: np.ndarray, alpha: float) -> None:
        """Add a symmetric coordinate stencil at distance *alpha* from *x*.

        Called when the first LP detects noise, making the initial tiny-step
        seed uninformative.  This gives the LP a full set of coordinate-aligned
        constraints at a noise-appropriate distance so that every gradient
        component is well-constrained immediately.
        """
        for i in range(self.dim):
            e_i = np.zeros(self.dim)
            e_i[i] = 1.0
            self._eval_and_record(x + alpha * e_i)
            self._eval_and_record(x - alpha * e_i)

    def _get_aux_alpha(self) -> float:
        """Return the current auxiliary step size based on noise / curvature."""
        if self.ns_est <= 1e-9:
            return self.init_step
        aa = 1 / 3 * self.hess_lipsc
        bb = 1 / 2 * self.hess_norm
        dd = -2 * self.ns_est
        rt = np.roots([aa, bb, 0, dd])
        roots = rt[np.isreal(rt) & (rt.real >= 0)]
        if len(roots) > 0:
            return float(roots.real[0])
        return 2.0 * np.sqrt(max(self.ns_est, 1e-30))

    def _probe_then_batch_inf_axes(self, x: np.ndarray) -> bool:
        """Probe-then-batch: quickly cover all unbounded gradient axes.

        When the diameter is infinite and multiple axes are unbounded,
        placing axis probes one-by-one with a full LP+diameter recompute
        after each is extremely wasteful.

        Strategy:
          1. Identify unbounded axes from self.gd_v.
          2. Place ONE pilot pair (±α·eᵢ) for the widest axis.
          3. Recompute and check: did the pilot bound its axis?
             Did the step size α change significantly?
          4. If the pilot was informative and α is stable, batch-place
             pairs for ALL remaining unbounded axes at the same α,
             then recompute once.
          5. If the pilot was uninformative (e.g. wrong α due to
             undetected noise), fall back to the normal refinement loop
             which will detect noise and trigger the noise-reseed path.

        Returns True if any samples were added, False otherwise.
        """
        if not np.isinf(self.gd_vm):
            return False
        if np.ndim(self.gd_v) != 1 or len(self.gd_v) != self.dim:
            return False

        inf_mask = ~np.isfinite(self.gd_v)
        inf_axes = np.where(inf_mask)[0]
        if len(inf_axes) <= 1:
            return False  # single inf axis: normal refinement handles it fine

        alpha_before = self._get_aux_alpha()

        # --- Step 1: pilot one axis pair ---
        pilot_axis = int(inf_axes[0])
        e_pilot = np.zeros(self.dim)
        e_pilot[pilot_axis] = 1.0
        self._eval_and_record(x + alpha_before * e_pilot)
        self._eval_and_record(x - alpha_before * e_pilot)
        self.aux_samples_count += 2
        self._recompute_at(x)

        # Check if pilot was informative
        pilot_bounded = np.isfinite(self.gd_v[pilot_axis])
        alpha_after = self._get_aux_alpha()
        alpha_ratio = alpha_after / max(alpha_before, 1e-30)
        alpha_stable = 0.1 <= alpha_ratio <= 10.0

        if self._diag_enabled:
            print(
                f"[BATCH-PILOT] call#{self._diag_call_count}  "
                f"pilot_axis={pilot_axis}  α={alpha_before:.4e}  "
                f"bounded={pilot_bounded}  "
                f"α_after={alpha_after:.4e}  α_ratio={alpha_ratio:.2f}  "
                f"stable={alpha_stable}  "
                f"ns_after={self.ns_est:.4e}",
                flush=True,
            )

        if not pilot_bounded or not alpha_stable:
            # Pilot failed: alpha was wrong (likely undetected noise).
            # Fall back — the normal flow (noise reseed or refinement)
            # will handle it.
            return True  # we did add 2 samples

        # --- Step 2: batch remaining inf axes ---
        remaining_inf = np.where(~np.isfinite(self.gd_v))[0]
        if len(remaining_inf) == 0:
            return True  # pilot already fixed everything

        alpha_batch = alpha_after  # use the (possibly updated) alpha
        for axis in remaining_inf:
            e_i = np.zeros(self.dim)
            e_i[axis] = 1.0
            self._eval_and_record(x + alpha_batch * e_i)
            self._eval_and_record(x - alpha_batch * e_i)
            self.aux_samples_count += 2

        self._recompute_at(x)

        if self._diag_enabled:
            inf_after = np.sum(~np.isfinite(self.gd_v))
            print(
                f"[BATCH-DONE] call#{self._diag_call_count}  "
                f"batched={len(remaining_inf)} axes  "
                f"α={alpha_batch:.4e}  "
                f"diam={self.gd_vm:.4e}  "
                f"inf_remaining={inf_after}  "
                f"hist={self.history.Zn.size}",
                flush=True,
            )
        return True

    def __call__(self, x: np.ndarray, force: bool = False) -> np.ndarray:
        """
        Estimate the gradient at point x.
        """
        _call_t0 = time.perf_counter()
        _call_lp0 = self._diag_lp_count
        self._diag_call_count += 1
        self._reset_query_state()

        x_changed = self.x_current is None or not np.array_equal(self.x_current, x)

        # reset_on_step: when jumping to a new iterate, replace the internal
        # history with a fresh buffer so the CFD stencil seeding triggers.
        # The shared history (used by obj_func / budget) is unaffected.
        if self.reset_on_step and x_changed:
            self.history = HistoryBuffer()
            self.Xn, self.Zn = self.history.snapshot()
            self._last_update_n = None
            self._last_update_x = None
            self.x_current = None
            self.Al = None
            self.bl = None
            self.A2 = None
            self.b2 = None
            self.gd_v = np.nan
            self.gd_vm = np.inf
            self.hess_norm = 0.0
            self.ns_est = 0.0

        sample_added = self._ensure_query_sample(x)

        history_changed = self._last_update_n is None or self._last_update_n != self.history.Zn.size
        needs_recompute = force or sample_added or history_changed or x_changed

        if self._diag_enabled:
            print(
                f"[SAGE-PHASE] call#{self._diag_call_count} "
                f"recompute={needs_recompute} hist={self.history.Zn.size} "
                f"reset={self.reset_on_step and x_changed} "
                f"t={time.perf_counter()-_call_t0:.4f}s",
                flush=True,
            )

        if needs_recompute:
            _rc_t0 = time.perf_counter()
            self._recompute_at(x)
            if self._diag_enabled:
                print(
                    f"[SAGE-PHASE] call#{self._diag_call_count} "
                    f"recompute done  t={time.perf_counter()-_rc_t0:.4f}s  "
                    f"diam={self.gd_vm:.4e}  H={self.hess_norm:.4e}  "
                    f"ns={self.ns_est:.4e}",
                    flush=True,
                )
        else:
            self.x_current = x

        # Noise-aware re-stencil: if the first LP detected noise, the tiny-step
        # seed is uninformative.  Immediately place a full coordinate stencil at
        # the noise-appropriate distance so the band filter can exclude the seed
        # and the LP gets well-conditioned constraints from the start.
        if self.ns_est > 1e-9 and not self._did_noise_reseed:
            self._did_noise_reseed = True
            if self._diag_enabled:
                print(
                    f"[SAGE-PHASE] call#{self._diag_call_count} "
                    f"noise reseed triggered  ns={self.ns_est:.4e}",
                    flush=True,
                )
            aa = 1 / 3 * self.hess_lipsc
            bb = 1 / 2 * self.hess_norm
            dd = -2 * self.ns_est
            rt = np.roots([aa, bb, 0, dd])
            roots = rt[np.isreal(rt) & (rt.real >= 0)]
            if len(roots) > 0:
                alpha = float(roots.real[0])
            else:
                alpha = 2.0 * np.sqrt(max(self.ns_est, 1e-30))
            self._reseed_at_alpha(x, alpha)
            self._recompute_at(x)
            if self._diag_enabled:
                print(
                    f"[SAGE-PHASE] call#{self._diag_call_count} "
                    f"noise reseed done  hist={self.history.Zn.size}  "
                    f"diam={self.gd_vm:.4e}",
                    flush=True,
                )

        # Probe-then-batch: if multiple axes are unbounded after the
        # initial recompute (and possible noise reseed), quickly cover
        # them with a pilot + batch instead of one-by-one refinement.
        if np.isinf(self.gd_vm):
            self._probe_then_batch_inf_axes(x)

        if self._last_update_x is None or not np.array_equal(x, self._last_update_x):
            if self._diag_enabled:
                print(
                    f"[SAGE-PHASE] call#{self._diag_call_count} "
                    f"early return (x != last_update_x)  "
                    f"t={time.perf_counter()-_call_t0:.4f}s",
                    flush=True,
                )
            return self.gdt_est

        if self._diag_enabled:
            print(
                f"[SAGE-PHASE] call#{self._diag_call_count} "
                f"entering refinement loop  diam={self.gd_vm:.4e}  "
                f"th={self.gdtset_diath:.4e}",
                flush=True,
            )

        _refine_aux = 0
        while True:
            if self._should_stop_refinement():
                if self._diag_enabled:
                    pending_axis = bool(self._axis_probe_queue)
                    stagnated = self._stable_count >= 3 and not pending_axis
                    print(
                        f"[REFINE-STOP] call#{self._diag_call_count}  "
                        f"reason: diam_ok={self.gd_vm < self.gdtset_diath}  "
                        f"stagnated={stagnated}(stable={self._stable_count})  "
                        f"max_aux={self.aux_samples_count >= 5.0*self.dim}  "
                        f"forced={self.gdt_est_frc}  "
                        f"axis_q={len(self._axis_probe_queue)}  "
                        f"diam={self.gd_vm:.4e}  th={self.gdtset_diath:.4e}",
                        flush=True,
                    )
                self._finish_refinement()
                break

            x_new = self._prepare_aux_sample(x)
            if x_new is None:
                break

            _aux_t0 = time.perf_counter()
            z_new = self.fun(x_new)
            _fun_dt = time.perf_counter() - _aux_t0
            _upd_t0 = time.perf_counter()
            self.update(x_new, z_new)
            _upd_dt = time.perf_counter() - _upd_t0
            _refine_aux += 1

            if self._diag_enabled and (_refine_aux <= 5 or _refine_aux % 10 == 0 or _upd_dt > 2.0):
                inf_axes = np.where(~np.isfinite(self.gd_v))[0] if np.ndim(self.gd_v) == 1 else []
                print(
                    f"[SAGE-REFINE] call#{self._diag_call_count} "
                    f"aux#{_refine_aux}  alpha={self.aux_step_sizes_current[-1]:.4e}  "
                    f"fun_t={_fun_dt:.4f}s  upd_t={_upd_dt:.4f}s  "
                    f"diam={self.gd_vm:.4e}  "
                    f"n_inf={len(inf_axes)}  "
                    f"stable={self._stable_count}  "
                    f"axis_q={len(self._axis_probe_queue)}  "
                    f"hist={self.history.Zn.size}",
                    flush=True,
                )

            if self.callback:
                self.callback()

        _call_dt = time.perf_counter() - _call_t0
        _call_lps = self._diag_lp_count - _call_lp0
        self._diag_call_time += _call_dt
        if self._diag_enabled:
            n_hist = self.history.Zn.size
            print(
                f"[SAGE-DIAG] call#{self._diag_call_count}  "
                f"hist={n_hist}  aux={_refine_aux}  "
                f"LPs={_call_lps}  call_t={_call_dt:.4f}s  "
                f"diam={self.gd_vm:.4e}  th={self.gdtset_diath:.4e}  "
                f"H={self.hess_norm:.4e}  ns={self.ns_est:.4e}  "
                f"stable={self._stable_count}",
                flush=True,
            )
        return self.gdt_est

    def _add_sample(self, x: np.ndarray, z: float):
        self.history.add(x, z)

    def update(self, x: np.ndarray, z: float, lightweight: bool = False):
        # Note: history.add() is handled by obj_func, so we just sync and recompute
        _upd_t0 = time.perf_counter()
        self._diag_update_count += 1
        self._sync_history()
        # In lightweight mode (e.g. during line search), skip the expensive
        # LP + diameter recomputation.  The next __call__ will detect
        # history_changed and do a full recompute automatically.
        if not lightweight and self.x_current is not None:
            self._recompute_at(self.x_current)
            self._consume_aux_feedback()
        self._diag_update_time += time.perf_counter() - _upd_t0

    def _sync_history(self):
        self.Xn, self.Zn = self.history.snapshot()
