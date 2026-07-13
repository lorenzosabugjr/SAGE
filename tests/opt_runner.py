"""
Per-trial optimization harness.

Wires a problem, a gradient estimator, and a GradientDescent optimizer
together, enforcing a shared evaluation budget across every objective call
(initial sample, estimator initialization/auxiliary samples, and line-search
trial points).
"""

import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from tests.factories import create_estimator, create_problem
from optimizers import GradientDescent, StepSizeMode
from utils.noise import NoiseType
from utils.history import HistoryBuffer


class OptimizationTrial:
    """
    A single optimization trial: instantiate a problem, a gradient estimator,
    and a GradientDescent optimizer, then run until the evaluation budget is
    exhausted.
    """

    def __init__(
        self,
        problem_name: str,
        grad_est_name: str,
        maxevals: int,
        dims: int = 2,
        condnum: float = 1.0,
        randseed: int = 1,
        noise_type: NoiseType = NoiseType.UNIFORM,
        noise_param: float = 0.0,
        gdtcalcstep: float = 1e-6,
        dtype=np.float128,
        stepsize: float = 1.0,
        stepsizemode: StepSizeMode = StepSizeMode.ADAPTIVE,
        armijo_beta: float = 0.5,
        armijo_c: float = 1e-6,
        min_stepsize: float = 1e-6,
        max_line_search_iters: int = 100,
        recompute_grad_every_ls_failures: int = 5,
        reset_stepsize_at_floor: bool = True,
        sage_reset_on_step: bool = False,
        verbose: bool = False,
    ):
        self.maxevals = maxevals
        self.noise_type = noise_type
        self.noise_param = noise_param
        self.dims = dims
        self.problem_name = problem_name

        # Initialize timing/solver state (set properly once available below).
        self.start_time = None
        self.solver = None

        # 1. Instantiate Problem
        self.problem = create_problem(problem_name, dims, condnum, randseed=randseed)

        # Objective function wrapper (binds noise params and records every
        # charged evaluation). Every call is an observation only: it
        # forward-fills whatever incumbent the shared history currently
        # holds. Accept/reject decisions are the optimizer's job (see
        # GradientDescent._mark_incumbent_accepted), applied to this same
        # evaluation's history entry right after this call returns.
        def obj_func(x):
            # Check budget: every objective call (initial eval, estimator
            # init/auxiliary samples, line-search trial points) counts.
            if self.history.Zn.size >= self.maxevals:
                raise StopIteration("Budget exhausted")

            val = self.problem.eval(x, self.noise_type, self.noise_param)
            z_true = self.problem.eval(x, self.noise_type, 0.0)
            t = time.perf_counter() - self.start_time if self.start_time is not None else 0.0

            self.history.add(x, val, z_true=z_true, t=t)
            return val
        self.obj_func = obj_func

        # 2. Initial Point Logic
        np.random.seed(randseed)
        X0 = 1e2 * (np.random.rand(dims) - 0.5)
        # Use 0.0 noise for the resampling check loop.
        Z0_tmp = self.problem.eval(X0, self.noise_type, 0.0)
        while Z0_tmp <= 1.0:
            X0 = 1e2 * (np.random.rand(dims) - 0.5)
            Z0_tmp = self.problem.eval(X0, self.noise_type, 0.0)

        self.X_initial = X0
        # Evaluate the initial point with actual noise for the history.
        self.Z_initial_eval = self.problem.eval(self.X_initial, self.noise_type, self.noise_param)
        self.Z_initial_true = self.problem.eval(self.X_initial, self.noise_type, 0.0)

        # Shared History Buffer. The original sampled center is the initial
        # incumbent; every subsequent raw evaluation forward-fills it until
        # an accepted line-search step moves it.
        self.history = HistoryBuffer()
        self.history.init_incumbent(self.Z_initial_eval, self.Z_initial_true)
        self.history.add(
            self.X_initial,
            self.Z_initial_eval,
            z_true=self.Z_initial_true,
            t=0.0,
        )

        # 3. Instantiate Estimator
        est_kwargs = dict(gdtcalcstep=gdtcalcstep, randseed=randseed, dtype=dtype)
        if grad_est_name == "sage":
            est_kwargs["reset_on_step"] = sage_reset_on_step
        if grad_est_name == "truth":
            est_kwargs["problem"] = self.problem

        self.estimator = create_estimator(
            grad_est_name, self.obj_func, dims, self.history, **est_kwargs
        )

        # Actual optimizer-start point/values. The optimizer always starts at
        # the original sampled center: this is the point around which SAGE's
        # initial stencil and first gradient are constructed, so reassigning
        # the start elsewhere would decenter the estimator's own seed.
        self.X_start = self.X_initial
        self.Z_start_eval = self.Z_initial_eval
        self.Z_start_true = self.Z_initial_true

        # 4. Instantiate Optimizer
        self.solver = GradientDescent(
            fun=self.obj_func,
            x0=self.X_start,
            grad_estimator=self.estimator,
            stepsize=stepsize,
            stepsizemode=stepsizemode,
            armijo_beta=armijo_beta,
            armijo_c=armijo_c,
            min_stepsize=min_stepsize,
            max_line_search_iters=max_line_search_iters,
            recompute_grad_every_ls_failures=recompute_grad_every_ls_failures,
            reset_stepsize_at_floor=reset_stepsize_at_floor,
            z0=self.Z_start_eval,
            verbose=verbose,
        )

    def run(self) -> dict:
        self.start_time = time.perf_counter()

        try:
            while self.history.Zn.size < self.maxevals:
                self.solver.step()
        except StopIteration:
            # Budget exhaustion mid-line-search is normal termination.
            pass

        return {
            # Per-evaluation iterate history (aligned with evaluation count).
            "res_hist_eval": self.history.z_k_eval_hist.reshape(-1, 1),
            "res_hist_true": self.history.z_k_true_hist.reshape(-1, 1),
            "time_hist": self.history.t_hist.reshape(-1, 1),
            # Original sampled initial point.
            "Z_initial_eval": self.Z_initial_eval,
            "Z_initial_true": self.Z_initial_true,
            # Actual optimizer-start point (equals Z_initial_* in current runs).
            "Z_start_eval": self.Z_start_eval,
            "Z_start_true": self.Z_start_true,
            "n_evals": self.history.Zn.size,
        }
