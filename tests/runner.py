import numpy as np
from datetime import datetime
from problems import LeastSquares, Lasso, L1LogReg, L2LogReg, LogSumExp
from estimators import SAGE, FFDEstimator, CFDEstimator, GSGEstimator, cGSGEstimator, NMXFDEstimator
from optimizers import StandardDescent, StepSizeMode
from utils.noise import NoiseType
from utils.history import HistoryBuffer


class SolverTest:
    def __init__(
        self,
        problem_name: str,
        grad_est_name: str,
        maxevals: int,
        dims: int = 2,
        condnum: float = 1.0,
        bfgs: bool = False,
        randseed: int = 1,
        noise_type: NoiseType = NoiseType.UNIFORM,
        noise_param: float = 0.0,
        gdtcalcstep: float = 1e-6,
    ):
        self.maxevals = maxevals
        self.noise_type = noise_type
        self.noise_param = noise_param
        self.dims = dims
        self.problem_name = problem_name
        
        # Initialize timing (will be set properly in run())
        self.start_time = None
        self.solver = None  # Will be set after estimator creation

        # 1. Instantiate Problem
        if problem_name == "least-squares":
            self.problem = LeastSquares(dims, condnum, randseed=randseed)
        elif problem_name == "lasso":
            self.problem = Lasso(dims, condnum, randseed=randseed)
        elif problem_name == "l1-log-reg":
            self.problem = L1LogReg(dims, condnum, randseed=randseed)
        elif problem_name == "l2-log-reg":
            self.problem = L2LogReg(dims, condnum, randseed=randseed)
        elif problem_name == "log-sum-exp":
            self.problem = LogSumExp(dims, condnum, randseed=randseed)
        else:
            raise ValueError(f"Unknown problem: {problem_name}")

        # Objective function wrapper (binds noise params and tracks iterate state)
        def obj_func(x):
            # Check budget
            if self.history.Zn.size >= self.maxevals:
                raise StopIteration("Budget exhausted")
            
            val = self.problem.eval(x, self.noise_type, self.noise_param)
            
            # Get current iterate state for tracking
            if self.solver is not None:
                z_k_eval = self.solver.z_k
                z_k_true = self.problem.eval(self.solver.x_k, self.noise_type, 0.0)
                t = (datetime.now() - self.start_time).total_seconds() if self.start_time else 0.0
            else:
                # Before solver exists, use the evaluated point itself
                z_k_eval = val
                z_k_true = self.problem.eval(x, self.noise_type, 0.0)
                t = 0.0
            
            # Add to history with iterate tracking
            self.history.add(x, val, z_k_eval=z_k_eval, z_k_true=z_k_true, t=t)
            return val
        self.obj_func = obj_func

        # 2. Initial Point Logic
        np.random.seed(randseed)
        X0 = 1e2 * (np.random.rand(dims) - 0.5)
        # Use 0.0 noise for the check loop
        Z0_tmp = self.problem.eval(X0, self.noise_type, 0.0) 
        while Z0_tmp <= 1.0:
            X0 = 1e2 * (np.random.rand(dims) - 0.5)
            Z0_tmp = self.problem.eval(X0, self.noise_type, 0.0)

        self.X0 = X0
        # Evaluate the initial point with actual noise for the history
        self.Z0_eval = self.problem.eval(self.X0, self.noise_type, self.noise_param)
        self.Z0_true = self.problem.eval(self.X0, self.noise_type, 0.0)
        
        # Shared History Buffer
        self.history = HistoryBuffer()
        # Add initial point with iterate tracking (before solver exists)
        self.history.add(self.X0, self.Z0_eval, z_k_eval=self.Z0_eval, z_k_true=self.Z0_true, t=0.0)

        # 3. Instantiate Estimator
        if grad_est_name == "ffd":
            self.estimator = FFDEstimator(self.obj_func, dims, step=gdtcalcstep, history=self.history)
        elif grad_est_name == "cfd":
            self.estimator = CFDEstimator(self.obj_func, dims, step=gdtcalcstep, history=self.history)
        elif grad_est_name == "gsg":
            self.estimator = GSGEstimator(self.obj_func, dims, m=dims, u=gdtcalcstep, seed=randseed, history=self.history)
        elif grad_est_name == "cgsg":
            self.estimator = cGSGEstimator(self.obj_func, dims, m=dims, u=gdtcalcstep, seed=randseed, history=self.history)
        elif grad_est_name == "nmxfd":
            self.estimator = NMXFDEstimator(self.obj_func, dims, history=self.history)
        elif grad_est_name == "sage":
            self.estimator = SAGE(
                self.obj_func,
                dims,
                quickmode=True,
                diam_mode="exact",
                history=self.history,
            )

            # Start from the best point in history to match the estimator's initialization.
            Xn_hist, Zn_hist = self.history.snapshot()
            best_idx = int(np.argmin(Zn_hist))
            self.X0 = Xn_hist[best_idx].copy()
            self.Z0_eval = Zn_hist[best_idx]
            
        else:
            raise ValueError(f"Unknown gradient estimator: {grad_est_name}")

        # 4. Instantiate Optimizer
        self.solver = StandardDescent(
            fun=self.obj_func,
            x0=self.X0,
            grad_estimator=self.estimator,
            stepsize=1.0,
            stepsizemode=StepSizeMode.ADAPTIVE,
            bfgs=bfgs,
            z0=self.Z0_eval
        )
        
    def run(self):
        self.start_time = datetime.now()
        
        try:
            while self.history.Zn.size < self.maxevals:
                self.solver.step()
        except StopIteration:
            pass
        
        # Return iterate history from HistoryBuffer (aligned with evaluation count)
        return (
            self.history.z_k_eval_hist.reshape(-1, 1),
            self.history.z_k_true_hist.reshape(-1, 1),
            self.history.t_hist.reshape(-1, 1),
            self.Z0_eval,
            self.Z0_true,
            self.history.Zn.size,
        )
