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

        # Objective function wrapper (binds noise params)
        def obj_func(x):
            # Check budget
            if self.history.Zn.size >= self.maxevals:
                raise StopIteration("Budget exhausted")
            val = self.problem.eval(x, self.noise_type, self.noise_param)
            return val
        self.obj_func = obj_func

        # 2. Initial Point Logic
        np.random.seed(randseed)
        X0 = 1e2 * (np.random.rand(dims) - 0.5)
        # Use 0.0 noise for the check loop
        Z0_tmp = self.problem.eval(X0, self.noise_type, 0.0) 
        while Z0_tmp <= 10 * max(1e-9, noise_param):
            X0 = 1e2 * (np.random.rand(dims) - 0.5)
            Z0_tmp = self.problem.eval(X0, self.noise_type, 0.0)

        self.X0 = X0
        # Evaluate the initial point with actual noise for the history
        self.Z0_eval = self.problem.eval(self.X0, self.noise_type, self.noise_param)
        self.Z0_true = self.problem.eval(self.X0, self.noise_type, 0.0)
        
        # Shared History Buffer
        self.history = HistoryBuffer()
        self.history.add(self.X0, self.Z0_eval)

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
            X_init = np.tile(X0, (dims + 1, 1)) + np.vstack(
                (np.zeros((1, dims)), np.identity(dims))
            )
            for i in range(1, X_init.shape[0]):
                z_val = self.problem.eval(X_init[i], self.noise_type, self.noise_param)  # Direct call
                self.history.add(X_init[i], z_val)

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
        self.hist_z_k_eval = []
        self.hist_z_k_true = []
        self.hist_t = []
        self.start_time = datetime.now()
        
        # StandardDescent invokes its callback at the start of each step (current z_k)
        # and after each evaluation (line search or auxiliary sample). We record each
        # callback so hist_z_k_eval includes the initial z0 plus per-evaluation values.
        
        def record_state(z_val=None):
            # Record current z_k (or the explicit evaluation value when provided).
            # Avoid recording if budget exhausted (handled by StopIteration usually,
            # but callback might fire right before exception).
            if self.history.Zn.size > self.maxevals:
                return

            current_z = z_val if z_val is not None else self.solver.z_k
            # Calculate noiseless value at current point
            current_z_true = self.problem.eval(self.solver.x_k, self.noise_type, 0.0)
            
            # Calculate elapsed time
            elapsed = (datetime.now() - self.start_time).total_seconds()
            
            self.hist_z_k_eval.append(current_z)
            self.hist_z_k_true.append(current_z_true)
            self.hist_t.append(elapsed)

        # Attach callbacks
        self.solver.callback = record_state
        if isinstance(self.estimator, SAGE):
            self.estimator.callback = lambda: record_state(None)

        try:
            while self.history.Zn.size < self.maxevals:
                self.solver.step()
        except StopIteration:
            pass
            
        return (
            np.array(self.hist_z_k_eval).reshape(-1, 1),
            np.array(self.hist_z_k_true).reshape(-1, 1),
            np.array(self.hist_t).reshape(-1, 1),
            self.Z0_eval,
            self.Z0_true,
            self.history.Zn.size,
        )
