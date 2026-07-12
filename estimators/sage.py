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
        noise_bound: Optional[float] = None,
        rel_tol: float = 0.5,
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
            noise_bound: Optional a priori bound on the noise magnitude. If None
                (default), SAGE self-calibrates the bound at the seed stencil
                (see _maybe_calibrate_noise) and then treats it as fixed. If a
                finite value >= 0 is given, SAGE fixes the noise bound up front
                and removes the noise-bound decision variable from the LP.
            rel_tol: Relative-accuracy certificate target: refinement stops once
                the gradient-set diameter falls below rel_tol * ||gdt_est||
                (see _should_stop_refinement). This is the main cost/accuracy
                knob: smaller values buy accuracy with more evaluations.
        """
        if noise_bound is not None:
            if not np.isfinite(noise_bound) or noise_bound < 0.0:
                raise ValueError(
                    f"noise_bound must be None or a finite value >= 0, got {noise_bound!r}"
                )
            noise_bound = float(noise_bound)

        super().__init__(fun, dim, history=history)
        self.quickmode = quickmode
        self.callback = callback
        self.init_step = init_step
        # Step size actually used by the last singleton CFD seed stencil;
        # the noise self-calibration (_maybe_calibrate_noise) reconstructs
        # the stencil's sample locations from it.
        self._seed_step = init_step
        self.reset_on_step = reset_on_step
        self.noise_bound = noise_bound
        self.noise_bound_is_fixed = noise_bound is not None
        self.rel_tol = float(rel_tol)
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
        
        # Noise bound is estimated by the LP unless a fixed bound is supplied.
        self.ns_est = self.noise_bound if self.noise_bound_is_fixed else 0.0
        
        self.gdt_est = np.zeros(dim)
        self.hess_norm = 0.0
        self.hess_lipsc = 0.0
        
        self.Al = None
        self.bl = None
        self.A2 = None
        self.b2 = None
        self.gd_v = np.nan
        self.gd_vm = np.inf
        # Axis-aligned bounding-box bounds of the gradient consistency set,
        # populated by _calc_diam_approx (reused for the point estimate) or
        # left None when diam_mode == "exact" / unavailable.
        self.min_g = None
        self.max_g = None
        
        self.gdt_est_frc = False
        self.hist_aux_samples = np.empty((0,))
        self.aux_step_sizes_current = np.empty((0,))

        self._last_update_n = None
        self._last_update_x = None
        self.x_current = None
        self._axis_probe_queue = []

        # Tracking aux samples for the current estimation step
        self.aux_samples_count = 0
        self._used_probe_axes = set()

        self._prev_gdt_est = None
        self._stable_count = 0

        # Adaptive auxiliary sampling radius: geometric growth factor applied
        # on top of the modeled alpha once every axis has been probed at the
        # current radius and the box is still not tight enough. Reset per
        # query in _reset_query_state so it never leaks across calls.
        self._aux_radius_growth = 1.0
        self._aux_radius_growth_factor = 2.0

        # ── Informed auxiliary radius + pilot curvature guard ──
        # _r_target: per-query one-shot sampling radius, sized so that a
        # single full axis sweep's information floor (4*eps/r per axis,
        # aggregated over dim axes) would just meet the rel_tol certificate
        # (see _compute_informed_radius). None until the seed LP has run.
        self._r_target = None
        # Auxiliary radius never drops below this multiple of the
        # informative scale (_radius_scale): sampling inside the radius the
        # seed stencil already covered adds strictly noisier constraints
        # (verified empirically: reseeding at alpha < init_step degraded
        # least-squares accuracy ~11x).
        self._alpha_floor_mult = 1.5
        # Cap on the informed radius, as a multiple of the informative scale.
        self._informed_r_cap_mult = 16.0
        # Pilot second-difference guard: for a completed symmetric axis pair
        # at radius r, D2 = |z(x+r e) + z(x-r e) - 2 z(x)|/2 can reach at
        # most 2*eps from noise alone, so D2 > _pilot_guard_mult * eps
        # proves real curvature at scale r and triggers a global shrink of
        # _r_target to r*sqrt(2*eps/D2) (the radius where the quadratic
        # residual ~ the noise slab), restarting the sweep. At most
        # _pilot_max_shrinks shrinks per query.
        self._pilot_guard_mult = 3.0
        self._pilot_max_shrinks = 3
        self._pilot_shrinks = 0
        self._pending_pair = []
        # Auxiliary sample cap per query, as a multiple of dim. With the
        # 2*dim+1 seed stencil this bounds a query's total evaluations by
        # (2 + _aux_cap_mult)*dim + 1.
        self._aux_cap_mult = 2.0

        # ── Diagnostic counters ──
        self._diag_lp_count = 0          # total LP solves
        self._diag_lp_time = 0.0         # total LP wall-time
        self._diag_call_count = 0        # number of __call__ invocations
        self._diag_call_time = 0.0       # total __call__ wall-time
        self._diag_diam_lp_count = 0     # LP solves inside diameter calc
        self._diag_diam_lp_time = 0.0
        self._diag_update_count = 0      # update() calls
        self._diag_update_time = 0.0
        self._diag_enabled = False        # set False to silence
        
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
        self._seed_step = h
        for i in range(self.dim):
            e_i = np.zeros(self.dim)
            e_i[i] = 1.0
            self._eval_and_record(x + h * e_i)
            self._eval_and_record(x - h * e_i)

    def _reset_query_state(self) -> None:
        self.aux_step_sizes_current = np.empty((0,))
        self._axis_probe_queue = []
        self._used_probe_axes = set()
        self.aux_samples_count = 0
        self._prev_gdt_est = None
        self._stable_count = 0
        self._aux_radius_growth = 1.0
        self._r_target = None
        self._pending_pair = []
        self._pilot_shrinks = 0
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
        self._update_point_estimate()

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

        # Compute optimal sampling radius for neighbor selection. Growth
        # applies only when alpha* is still the unresolved-curvature
        # fallback (see _model_alpha) -- once the cubic resolves a real
        # root, alpha* is already the correct radius for the
        # currently-estimated H_i/gamma_H, and multiplying it by a stale
        # growth factor would reintroduce the runaway-radius hazard
        # _model_alpha's docstring describes.
        alpha, resolved = self._model_alpha()

        # Neighbor selection tracks the radius aux samples are actually
        # being placed at (alpha scaled by any axis-exhaustion-driven growth
        # from _next_aux_direction), not just the un-grown model alpha.
        # Otherwise growth pushes new samples further out while the band
        # filter/quickmode ranking keep re-anchoring on the stale un-grown
        # alpha, discarding exactly the far samples growth was trying to
        # create -- H_i/gamma_H then can never resolve nonzero from that
        # evidence, no matter how far growth pushes.
        search_alpha = alpha if resolved else alpha * self._aux_radius_growth

        # Distance band filter: when noise is detected, exclude samples
        # whose distance from x is so far from search_alpha that their LP
        # constraints are noise-dominated (too close) or curvature-
        # dominated (too far).  This prevents tiny-step seed samples
        # from poisoning the LP with enormous slab widths.
        if self.ns_est > 1e-9 and len(coll_x) > 0:
            coll_dists = np.array([norm(cx - x) for cx in coll_x])
            band_lo = search_alpha / 10.0
            band_hi = search_alpha * 10.0
            in_band = (coll_dists >= band_lo) & (coll_dists <= band_hi)
            if np.sum(in_band) >= D + 1:
                coll_x = [coll_x[j] for j in range(len(coll_x)) if in_band[j]]
                coll_idx_raw = [coll_idx_raw[j] for j in range(len(coll_idx_raw)) if in_band[j]]

        # Quickmode: keep at most 5D nearest-to-search_alpha samples
        if len(coll_x) > 5*D and self.quickmode:
            cost_fn = np.abs(np.sum((np.array(coll_x) - x)**2, axis=1) - search_alpha**2)
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
                f"search_alpha={search_alpha:.4e}  growth={self._aux_radius_growth:.4e}  "
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
        # Estimated mode keeps the full [g, H, gamma, eps] LP. Fixed mode
        # drops the eps decision variable and folds its known contribution
        # (the eps column is always -2/dij) into the RHS instead.
        if self.noise_bound_is_fixed:
            A_lp = A[:, : D + 2]
            b_lp = b.flatten() - A[:, D + 2] * self.noise_bound
            n_nonneg = 2
        else:
            A_lp = A
            b_lp = b.flatten()
            n_nonneg = 3

        nonneg_rows = np.zeros((n_nonneg, A_lp.shape[1]))
        for i in range(n_nonneg):
            nonneg_rows[i, A_lp.shape[1] - n_nonneg + i] = -1.0
        Ae = np.vstack((A_lp, nonneg_rows))
        be = np.concatenate((b_lp, np.zeros(n_nonneg)))

        c = np.hstack((np.zeros(Ae.shape[1] - n_nonneg), np.ones(n_nonneg)))
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
            if self.noise_bound_is_fixed:
                self.gdt_est = res.x[:-2]
                self.hess_norm = np.max([res.x[-2], self.hess_norm])
                self.hess_lipsc = np.max([res.x[-1], self.hess_lipsc])
                # ns_est stays fixed at self.noise_bound; not overwritten.
            else:
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
            if self.noise_bound_is_fixed:
                raise RuntimeError(
                    "SAGE fixed-noise-bound LP did not solve successfully "
                    f"(status={res.status!r}, message={res.message!r}, "
                    f"noise_bound={self.noise_bound!r})"
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

    def _model_alpha(self) -> tuple[float, bool]:
        """Auxiliary sampling radius and whether it is "resolved" (should
        NOT be multiplied by the axis-exhaustion growth factor again).

        Priority order:
          1. The per-query informed radius _r_target, when available
             (see _compute_informed_radius): returned with the growth
             factor already folded in, flagged resolved so callers use it
             as-is. Growth still escalates the radius sweep-by-sweep, and
             the pilot guard (_consume_pilot_feedback) can shrink _r_target
             globally when it proves curvature at the current scale.
          2. Otherwise the cubic-root alpha* (theory.md Sec. 5) when the LP
             has resolved curvature, or the unit-Hessian-norm bootstrap
             2*sqrt(eps) when it hasn't (unresolved: growth applies).

        Either way the result is floored at _alpha_floor_mult times the
        informative scale (_radius_scale): the seed stencil already
        extracted the information available at init_step, so sampling
        inside that radius only adds constraints with strictly worse
        noise-to-signal (empirically ~11x accuracy loss on least-squares
        when the old noise-reseed did exactly that), and under noise no
        sample inside ~2*sqrt(eps) is informative regardless of init_step.
        """
        floor = self._alpha_floor_mult * self._radius_scale()
        if self._r_target is not None:
            return max(self._r_target, floor) * self._aux_radius_growth, True

        aa = 1 / 3 * self.hess_lipsc
        bb = 1 / 2 * self.hess_norm
        dd = -2 * self.ns_est
        rt = np.roots([aa, bb, 0, dd])
        roots = rt[np.isreal(rt) & (rt.real >= 0)]
        if roots.size > 0:
            return max(float(roots.real[0]), floor), True
        return max(2.0 * np.sqrt(max(self.ns_est, 1e-30)), floor), False

    def _radius_scale(self) -> float:
        """Informative sampling scale: max(init_step, 2*sqrt(eps)).

        The radius floor and the informed-radius clip band are anchored on
        this rather than on init_step alone. Under noise, a sample at
        distance d contributes a slab of width ~4*eps/d in gradient units,
        so nothing inside ~2*sqrt(eps) (the unit-Hessian bootstrap radius)
        is informative no matter how init_step was chosen -- anchoring on
        init_step alone pinned every auxiliary sample to ~1.5e-6 when a
        production run was launched with the default init_step=1e-6 under
        noise 1.0, producing rel errors of ~1e5 (the old noise-reseed's
        absolute jump to 2*sqrt(eps) was what used to absorb this mistake).
        When init_step is already noise-appropriate, init_step dominates
        and behavior is unchanged.
        """
        return max(float(self.init_step),
                   2.0 * float(np.sqrt(max(self.ns_est, 0.0))))

    def _compute_aux_step(self) -> tuple[float, float]:
        if self.ns_est <= 1e-9:
            model_alpha = self.init_step
            self.gdtset_diath = self.gdtset_diaid
            alpha = model_alpha * self._aux_radius_growth
            return alpha, float(model_alpha)

        model_alpha, resolved = self._model_alpha()
        self.gdtset_diath = 1.01 * model_alpha

        # Axis-exhaustion-driven radius growth (see _next_aux_direction):
        # scales the step actually used without altering the modeled alpha
        # that gdtset_diath is derived from -- but only while alpha* is
        # still unresolved (see _model_alpha).
        alpha = model_alpha if resolved else model_alpha * self._aux_radius_growth
        return alpha, float(model_alpha)

    def _select_probe_axis(self) -> Optional[int]:
        if np.ndim(self.gd_v) != 1:
            return None

        widths = np.abs(np.asarray(self.gd_v, dtype=float).reshape(-1))
        if widths.size != self.dim:
            return None

        order = np.argsort(-widths)
        for axis in order:
            if widths[axis] <= 0.0 or int(axis) in self._used_probe_axes:
                continue
            return int(axis)

        return None

    def _enqueue_axis_probe_pair(self) -> bool:
        if self._axis_probe_queue:
            return True

        axis = self._select_probe_axis()
        if axis is None:
            return False

        direction = np.zeros(self.dim)
        direction[axis] = 1.0
        # Use a symmetric shell pair so the directional derivative is pinned by
        # matched far samples instead of a one-sided constraint.
        self._axis_probe_queue.extend((direction, -direction))
        self._used_probe_axes.add(int(axis))
        return True

    def _next_aux_direction(self) -> Optional[np.ndarray]:
        if self._axis_probe_queue:
            return self._axis_probe_queue.pop(0)
        if self._enqueue_axis_probe_pair():
            return self._axis_probe_queue.pop(0)

        # Every axis has been probed at the current radius and the box is
        # still not tight enough: grow the radius and restart a fresh sweep
        # over all axes. (Diameter-direction probes were removed entirely --
        # in approx/box diam_mode, gd_v is elementwise non-negative, so the
        # "diameter direction" was always the same fixed all-positive-orthant
        # diagonal; a diagonal cut leaves every axis-aligned bound unchanged,
        # so those probes could never inform the box metric or the box-center
        # point estimate, only burn budget. Verified: axis-only refinement
        # beat the old diameter+axis alternation on every point of the
        # diagnostic benchmark, at equal budget.)
        self._aux_radius_growth *= self._aux_radius_growth_factor
        self._used_probe_axes.clear()
        if self._diag_enabled:
            print(
                f"[SAGE-PHASE] call#{self._diag_call_count} "
                f"aux radius growth  factor={self._aux_radius_growth:.4e}",
                flush=True,
            )
        if self._enqueue_axis_probe_pair():
            return self._axis_probe_queue.pop(0)
        return None

    def _should_stop_refinement(self) -> bool:
        """Relative-accuracy-certificate stopping.

        Stop when the gradient-set diameter is certifiably small RELATIVE
        to the estimated gradient itself (gd_vm < rel_tol * ||gdt_est||):
        the box then brackets the gradient to ~rel_tol/2 relative accuracy,
        which is the quantity the benchmark (and any consumer of the
        estimate) actually cares about. Points with strong gradients
        certify at the seed stencil (0 auxiliary samples, CFD-cost);
        weak-gradient points refine until certified or capped.

        The absolute gdtset_diaid floor applies ONLY in the noiseless
        regime (ns_est ~ 0), where ||gdt_est|| -> 0 near an optimum makes
        the relative target degenerate and the seed stencil is already
        near-machine-precision. Under noise it must NOT apply: near-zero-
        gradient points are exactly where refinement pays most, and an
        absolute floor was measured to cut them off 10x short
        (l1-log-reg at noise 1e-3: median rel err 1.33 with the floor vs
        0.131 without, at 47 vs 64 evaluations).

        The old absolute target (gd_vm < gdtset_diath = 1.01*alpha*) and
        the box-centered stagnation stop were both removed: the former is
        derived from H_i/gamma_H estimates that are usually stuck at a
        degenerate fallback (unreachable moving target), and the latter was
        measured to trigger mid-improvement on exactly the points where
        refinement pays (cutting achievable accuracy ~3x) while wasting
        ~40 evaluations on points that should stop at the seed.

        _stable_count (raw LP-vertex stability) remains diagnostic-only.
        """
        pending_axis_pair = bool(self._axis_probe_queue)
        g_norm = max(float(norm(np.asarray(self.gdt_est, dtype=float))), 1e-30)
        certified = np.isfinite(self.gd_vm) and (
            self.gd_vm < self.rel_tol * g_norm
            or (self.ns_est <= 1e-9 and self.gd_vm < self.gdtset_diaid)
        )
        return (
            (certified and not pending_axis_pair)
            or self.gdt_est_frc
            or (self.aux_samples_count >= self._aux_cap_mult * self.dim
                and not pending_axis_pair)
        )

    def _finish_refinement(self) -> None:
        self.hist_aux_samples = np.hstack((self.hist_aux_samples, self.aux_samples_count))
        self.aux_samples_count = 0
        self.gdt_est_frc = False

    def _prepare_aux_sample(self, x: np.ndarray) -> Optional[np.ndarray]:
        alpha, _ = self._compute_aux_step()
        direction = self._next_aux_direction()
        if direction is None:
            return None

        x_new = x + alpha * direction
        self.aux_step_sizes_current = np.hstack((self.aux_step_sizes_current, float(alpha)))
        self.aux_samples_count += 1
        # Track (axis, radius, location) so _consume_pilot_feedback can
        # compute the pair's second difference once both members land.
        axis = int(np.argmax(np.abs(direction)))
        self._pending_pair.append((axis, float(alpha), np.asarray(x_new)))
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
        # Exact mode does not compute per-axis bounds as a byproduct, so the
        # point estimate must fall back to computing its own bounding box.
        self.min_g = None
        self.max_g = None
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
            self.min_g = None
            self.max_g = None
            return np.inf

        max_g = np.empty(D)
        min_g = np.empty(D)

        for i in range(D):
            direction = np.zeros(D)
            direction[i] = 1.0
            max_g[i] = self._solve_direction_bound(direction, maximize=True)
            min_g[i] = self._solve_direction_bound(direction, maximize=False)

        # Stored so _update_point_estimate can reuse these per-axis solves
        # for the bounding-box center instead of re-solving them.
        self.min_g = min_g
        self.max_g = max_g

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

    def _update_point_estimate(self) -> None:
        """Replace the raw LP-solved vertex in self.gdt_est with a
        representative point: the center of the axis-aligned bounding box of
        the gradient consistency set (self.Al, self.bl), or its projection
        onto the polytope if the box center itself is infeasible.

        The LP objective never costs g itself, so once noise makes H/gamma
        (and eps, in estimated mode) LP-feasible at zero, any g inside the
        polytope is equally optimal and the raw vertex returned by the
        solver is solver-path-dependent and potentially meaningless (often
        the exact zero vector). The box center is a much more representative
        summary of the consistency set.

        Works off Al/bl directly so it applies regardless of diam_mode:
        reuses the per-axis bounds already solved by _calc_diam_approx when
        available, and falls back to solving them here (e.g. diam_mode ==
        "exact") otherwise.
        """
        if self.Al is None or self.bl is None or self.Al.size == 0:
            return

        D = self.dim
        if self.min_g is not None and self.max_g is not None:
            min_g, max_g = self.min_g, self.max_g
        else:
            min_g = np.empty(D)
            max_g = np.empty(D)
            for i in range(D):
                direction = np.zeros(D)
                direction[i] = 1.0
                max_g[i] = self._solve_direction_bound(direction, maximize=True)
                min_g[i] = self._solve_direction_bound(direction, maximize=False)

        if not (np.all(np.isfinite(min_g)) and np.all(np.isfinite(max_g))):
            # Box is unbounded along some axis; no well-defined center to
            # project toward, so keep the raw LP vertex.
            return

        g_box = 0.5 * (max_g + min_g)

        tol = 1e-9
        slack = self.bl - self.Al @ g_box
        if np.all(slack >= -tol):
            self.gdt_est = g_box
            return

        # Box center is infeasible: segment-clip from the raw LP-solved
        # vertex (guaranteed feasible) toward the box center, stopping at
        # the first constraint boundary crossed.
        g_v = np.asarray(self.gdt_est, dtype=float)
        delta = g_box - g_v
        Ad = self.Al @ delta
        Av = self.Al @ g_v
        active = Ad > tol
        if not np.any(active):
            return

        t_star = min(1.0, float(np.min((self.bl[active] - Av[active]) / Ad[active])))
        t_star = max(0.0, t_star)
        self.gdt_est = g_v + t_star * delta

    def _seed_second_differences(self, x: np.ndarray) -> Optional[np.ndarray]:
        """Per-axis second differences of the seed stencil around *x*:
        D2_i = |z(x + h e_i) + z(x - h e_i) - 2 z(x)| / 2 with h = _seed_step.

        Returns None when the stencil samples cannot all be found in the
        history (e.g. the history was pre-seeded without a stencil).
        """
        h = self._seed_step
        i0 = self.history.find_indices(np.asarray(x))
        if i0.size == 0:
            return None
        z0 = float(self.Zn[i0[0]])
        d2s = np.empty(self.dim)
        for i in range(self.dim):
            e_i = np.zeros(self.dim)
            e_i[i] = 1.0
            ip = self.history.find_indices(x + h * e_i)
            im = self.history.find_indices(x - h * e_i)
            if ip.size == 0 or im.size == 0:
                return None
            d2s[i] = abs(float(self.Zn[ip[0]]) + float(self.Zn[im[0]]) - 2.0 * z0) / 2.0
        return d2s

    def _maybe_calibrate_noise(self, x: np.ndarray) -> None:
        """Estimate-mode noise self-calibration at the seed stencil.

        The LP's estimated ns_est MINIMIZES eps subject to feasibility, so
        it is a lower bound on the noise, not an estimator (measured at
        0.3-0.7x the true bound). Everything derived from it (informed
        radius, pilot guard threshold, certificate tightness) inherits the
        bias. Instead, estimate eps from the seed stencil's own second
        differences: for noise-only D2 (uniform noise, locally-linear f at
        scale h), sd(D2) ~ 0.7*eps and max(D2) <= 2*eps, so
            eps_cal = max( sqrt(2*mean(D2^2)),  max(D2)/1.5,  ns_est )
        is nearly unbiased (measured 0.45-0.50 vs true 0.5 on the log-type
        benchmark problems). Then switch to fixed-bound mode with eps_cal so
        the LP, the informed radius, the pilot guard and the stopping
        certificate all use it consistently, and re-solve once (no new
        evaluations).

        On functions with real curvature or kinks at scale h the D2s also
        contain curvature signal, inflating eps_cal (least-squares/lasso:
        ~20x). This errs conservative -- a larger eps loosens the certificate
        and enlarges the informed radius, and the pilot guard walks the
        radius back if the far samples then prove curvature -- and those
        problem types certify at the seed anyway. Calibration runs once per
        estimator lifetime (noise is a property of the oracle, not of the
        query point) and is skipped entirely when a noise_bound was supplied
        or the seed LP found the data consistent with zero noise.
        """
        if self.noise_bound_is_fixed or self.ns_est <= 1e-9:
            return
        d2s = self._seed_second_differences(np.asarray(x))
        if d2s is None:
            return
        eps_cal = max(
            float(np.sqrt(2.0 * np.mean(d2s ** 2))),
            float(np.max(d2s)) / 1.5,
            float(self.ns_est),
        )
        self.noise_bound = eps_cal
        self.noise_bound_is_fixed = True
        self.ns_est = eps_cal
        if self._diag_enabled:
            print(
                f"[SAGE-PHASE] call#{self._diag_call_count} "
                f"noise calibrated  eps_cal={eps_cal:.4e}  "
                f"d2_max={np.max(d2s):.4e}",
                flush=True,
            )
        self._recompute_at(np.asarray(x))

    def _compute_informed_radius(self) -> None:
        """Size the auxiliary sampling radius to the stopping certificate.

        A full axis sweep at radius r can tighten each axis width to the
        information floor ~4*eps/r, giving a box diameter ~sqrt(dim)*4*eps/r.
        Setting that equal to the certificate target rel_tol*||g|| yields
            r* = 4 * eps * sqrt(dim) / (rel_tol * ||g_seed||),
        the radius whose single sweep would just certify -- instead of
        crawling toward it via 2x growth per exhausted sweep. Clipped to
        [_alpha_floor_mult, _informed_r_cap_mult] * _radius_scale(). Computed once
        per query from the post-seed LP estimate; the pilot guard may shrink
        it afterwards if the first pair proves curvature at that scale.

        Points with strong gradients get a small r* (safe: their certificate
        is nearly met already), weak-gradient points get a large r* -- and
        their curvature bias tolerance also scales with rel_tol*||g||, which
        is what the pilot guard checks against.
        """
        if self._r_target is not None or self.ns_est <= 1e-9:
            return
        g_norm = float(norm(np.asarray(self.gdt_est, dtype=float)))
        if not np.isfinite(g_norm) or g_norm <= 0.0:
            return
        scale = self._radius_scale()
        r = 4.0 * float(self.ns_est) * np.sqrt(self.dim) / (self.rel_tol * g_norm)
        self._r_target = float(np.clip(
            r, self._alpha_floor_mult * scale, self._informed_r_cap_mult * scale))
        if self._diag_enabled:
            print(
                f"[SAGE-PHASE] call#{self._diag_call_count} "
                f"informed radius  r*={self._r_target:.4e}  "
                f"(raw={r:.4e}  |g|={g_norm:.4e}  eps={self.ns_est:.4e})",
                flush=True,
            )

    def _consume_pilot_feedback(self, x: np.ndarray) -> None:
        """Second-difference curvature guard over completed axis pairs.

        For a completed symmetric pair (+r e_i, -r e_i), noise alone (|n| <=
        eps) can push D2 = |z+ + z- - 2 z0|/2 to at most 2*eps, so
        D2 > _pilot_guard_mult*eps proves real curvature at scale r on that
        axis -- the regime where the LP (which structurally prefers
        H = gamma = 0 whenever feasible) folds the residual into a biased
        gradient instead. Response: shrink _r_target globally to
        r*sqrt(2*eps/D2) (where the quadratic residual ~ the noise slab) and
        restart the sweep, so the first pair of each sweep acts as a pilot
        and at most one pair is committed at a too-large radius.

        A per-axis backoff (resampling only the violating axis at the small
        radius) was tried and REJECTED: the replacement samples fall below
        the LP band filter (search_alpha/10) and never reach the LP, while
        the poisoned far samples remain -- measured strictly worse than no
        guard at all.
        """
        if len(self._pending_pair) < 2:
            return
        (ax1, a1, xp), (ax2, a2, xm) = self._pending_pair[-2:]
        if ax1 != ax2 or abs(a1 - a2) > 1e-12 * max(a1, a2):
            return  # not a matched pair; keep sliding
        self._pending_pair = []
        i_p = self.history.find_indices(xp)
        i_m = self.history.find_indices(xm)
        i_0 = self.history.find_indices(np.asarray(x))
        if i_p.size == 0 or i_m.size == 0 or i_0.size == 0:
            return
        d2 = abs(float(self.Zn[i_p[0]]) + float(self.Zn[i_m[0]])
                 - 2.0 * float(self.Zn[i_0[0]])) / 2.0
        eps = max(float(self.ns_est), 1e-30)
        if d2 <= self._pilot_guard_mult * eps:
            return
        if self._pilot_shrinks >= self._pilot_max_shrinks:
            return
        a_new = max(a1 * float(np.sqrt(2.0 * eps / d2)),
                    self._alpha_floor_mult * self._radius_scale())
        if a_new >= 0.9 * a1:
            return  # already at/near the floor; nothing to shrink
        self._pilot_shrinks += 1
        self._r_target = a_new
        self._aux_radius_growth = 1.0
        self._axis_probe_queue = []
        self._used_probe_axes.clear()
        if self._diag_enabled:
            print(
                f"[SAGE-PHASE] call#{self._diag_call_count} "
                f"pilot shrink #{self._pilot_shrinks}  axis={ax1}  "
                f"D2={d2:.4e}  r {a1:.4e} -> {a_new:.4e}",
                flush=True,
            )

    def _get_aux_alpha(self) -> float:
        """Return the current auxiliary step size based on noise / curvature."""
        if self.ns_est <= 1e-9:
            return self.init_step
        alpha, _ = self._model_alpha()
        return alpha

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
            self.ns_est = self.noise_bound if self.noise_bound_is_fixed else 0.0

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

        # Estimate-mode noise self-calibration from the seed stencil's own
        # second differences (see _maybe_calibrate_noise). No new
        # evaluations; switches to fixed-bound mode on first success.
        self._maybe_calibrate_noise(x)

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

        # Size the auxiliary radius to the certificate target, now that the
        # (possibly recalibrated) seed LP has produced gdt_est and ns_est.
        self._compute_informed_radius()

        if self._diag_enabled:
            print(
                f"[SAGE-PHASE] call#{self._diag_call_count} "
                f"entering refinement loop  diam={self.gd_vm:.4e}  "
                f"|g|={norm(np.asarray(self.gdt_est, dtype=float)):.4e}  "
                f"r*={self._r_target}",
                flush=True,
            )

        _refine_aux = 0
        while True:
            if self._should_stop_refinement():
                if self._diag_enabled:
                    g_norm = max(float(norm(np.asarray(self.gdt_est, dtype=float))), 1e-30)
                    print(
                        f"[REFINE-STOP] call#{self._diag_call_count}  "
                        f"reason: certified={self.gd_vm < self.rel_tol * g_norm or self.gd_vm < self.gdtset_diaid}  "
                        f"max_aux={self.aux_samples_count >= self._aux_cap_mult*self.dim}  "
                        f"forced={self.gdt_est_frc}  "
                        f"axis_q={len(self._axis_probe_queue)}  "
                        f"diam={self.gd_vm:.4e}  rel_target={self.rel_tol * g_norm:.4e}",
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
            self._consume_pilot_feedback(x)
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
        self._diag_update_time += time.perf_counter() - _upd_t0

    def _sync_history(self):
        self.Xn, self.Zn = self.history.snapshot()
