import numpy as np
from datetime import datetime
from problems import LeastSquares, Lasso, L1LogReg, L2LogReg, LogSumExp
from optimizers import FFDOpt, CFDOpt, GSGOpt, cGSGOpt, NMXFDOpt, SAGEOpt
from utils.noise import NoiseType


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

        # Initialize History buffers
        self.hist_z_k = np.empty((0, 1))
        self.hist_t = np.empty((0, 1))
        self.Z0 = np.inf

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

        # 2. Initial Point Logic
        np.random.seed(randseed)
        X0 = 1e2 * (np.random.rand(dims) - 0.5)
        Z0_tmp = self.problem.eval(X0, noise_type=NoiseType.UNIFORM, noise_param=0.0)
        while Z0_tmp <= 10 * noise_param:
            X0 = 1e2 * (np.random.rand(dims) - 0.5)
            Z0_tmp = self.problem.eval(X0, noise_type=NoiseType.UNIFORM, noise_param=0.0)

        self.X0 = X0
        self.Z0 = self.problem.eval(self.X0, self.noise_type, self.noise_param)

        # 3. Instantiate Solver (state-machine based)
        if grad_est_name == "ffd":
            self.solver = FFDOpt(self.X0, self.Z0, bfgs=bfgs, ffdstep=gdtcalcstep)
        elif grad_est_name == "cfd":
            self.solver = CFDOpt(self.X0, self.Z0, bfgs=bfgs, cfdstep=gdtcalcstep)
        elif grad_est_name == "gsg":
            self.solver = GSGOpt(self.X0, self.Z0, bfgs=bfgs, m=dims, u=gdtcalcstep)
        elif grad_est_name == "cgsg":
            self.solver = cGSGOpt(self.X0, self.Z0, bfgs=bfgs, m=dims, u=gdtcalcstep)
        elif grad_est_name == "nmxfd":
            self.solver = NMXFDOpt(self.X0, self.Z0, bfgs=bfgs, m=dims)
        elif grad_est_name == "sage":
            X_init = np.tile(X0, (dims + 1, 1)) + 1 * np.vstack(
                (np.zeros((1, dims)), np.identity(dims))
            )
            Z_init = np.empty(0)
            for i in range(X_init.shape[0]):
                if i == 0:
                    Z_tmp = self.Z0
                else:
                    Z_tmp = self.problem.eval(X_init[i], self.noise_type, self.noise_param)
                Z_init = np.hstack((Z_init, Z_tmp))
            self.solver = SAGEOpt(X_init, Z_init, bfgs=bfgs, quickmode=True)
        else:
            raise ValueError(f"Unknown gradient estimator: {grad_est_name}")

    def run(self):
        while self.solver.n < self.maxevals:
            z_n = self.problem.eval(self.solver.x_n, self.noise_type, self.noise_param)

            start_t = datetime.now()
            self.solver.add_samples(self.solver.x_n, z_n)
            end_t = datetime.now()

            self.hist_t = np.vstack((self.hist_t, (end_t - start_t).total_seconds()))
            self.hist_z_k = np.vstack((self.hist_z_k, self.solver.z_k))

        return self.hist_z_k, self.hist_t, self.Z0, self.solver.n
