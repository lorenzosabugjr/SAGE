import numpy as np
import matplotlib.pyplot as plt
import os
from datetime import datetime
from scipy.io import savemat
from benchmark_problems import *
from solvers import *

bmk_maxtrials     = 100

list_dims         = [5, 10, 20]         # 5, 10, 20
list_condnum      = [1.0, 1e4, 1e8]     # 1e0, 1e4, 1e8
list_noise        = [1.0, 1e-3, 0.0]    # 1.0, 1e-3, 0.0

# Available options:
# 'least-squares', 'lasso', 'l1-log-reg', 'l2-log-reg', 'log-sum-exp'
list_problem      = ['least-squares', 'lasso', 'l1-log-reg', 'l2-log-reg', 'log-sum-exp']

# Available options: 
# 'ffd', 'cfd', 'gsg', 'cgsg', 'nmxfd', 'sage'
list_solver       = ['ffd', 'cfd', 'gsg', 'cgsg', 'nmxfd', 'sage']

class SolverTest:
    def __init__(
        self,
        problem: str,
        solver: str,
        maxevals: int,
        dims: int = 2,
        condnum: int = 1,
        bfgs: bool = False,
        randseed: int = 1,
        noisebnd: float = 0.0,
        gdtcalcstep: float = 1e-6,
    ):

        self.maxevals = maxevals
        self.noisebnd = noisebnd
        self.hist_z_k = np.empty((0,1))
        self.hist_t   = np.empty((0,1))
        self.Z0       = np.inf

        match problem:
            case "least-squares":
                self.problem = LeastSquares(dims, condnum, randseed=randseed)            
            case "lasso":
                self.problem = Lasso(dims, condnum, randseed=randseed)            
            case "l1-log-reg":
                self.problem = L1LogReg(dims, condnum, randseed=randseed)
            case "l2-log-reg":
                self.problem = L2LogReg(dims, condnum, randseed=randseed)
            case "log-sum-exp":
                self.problem = LogSumExp(dims, condnum, randseed=randseed)
            case _:
                return

        np.random.seed(randseed)
        X0 = 1e2 * (np.random.rand(dims) - 0.5)
        Z0_tmp = self.problem.eval(X0, 0.0)
        while Z0_tmp <= 10*noisebnd:
            X0 = 1e2 * (np.random.rand(dims) - 0.5)
            Z0_tmp = self.problem.eval(X0, 0.0)

        self.Xn = X0
        self.Zn = self.problem.eval(self.Xn, noisebnd)
        self.Z0 = self.Zn
        match solver:
            case "ffd":
                self.solver = FFDOpt(self.Xn, self.Zn, bfgs=bfgs, ffdstep=gdtcalcstep)
            case "cfd":
                self.solver = CFDOpt(self.Xn, self.Zn, bfgs=bfgs, cfdstep=gdtcalcstep)
            case 'gsg':
                self.solver = GSGOpt(self.Xn, self.Zn, bfgs=bfgs, m=dims, u=gdtcalcstep)
            case 'cgsg':
                self.solver = cGSGOpt(self.Xn, self.Zn, bfgs=bfgs, m=dims, u=gdtcalcstep)
            case 'nmxfd':
                self.solver = NMXFDOpt(self.Xn, self.Zn, bfgs=bfgs, m=dims)
            case "sage":
                self.Xn = np.tile(X0, (dims + 1, 1)) + 1 * np.vstack(
                    (np.zeros((1, dims)), np.identity(dims))
                )
                
                self.Zn = np.empty(0)
                for i in range(self.Xn.shape[0]):
                    if i == 0:
                        ZnT = self.Z0
                    else:
                        ZnT = self.problem.eval(self.Xn[i], noisebnd)
                    self.Zn = np.hstack((self.Zn, ZnT))

                self.solver = SAGEOpt(self.Xn, self.Zn, bfgs=bfgs, quickmode=True)
            case _:
                return

    def run(self):
        while self.solver.n < self.maxevals:
            z_n = self.problem.eval(self.solver.x_n, self.noisebnd)
            
            start_t = datetime.now()
            self.solver.add_samples(self.solver.x_n, z_n)

            end_t = datetime.now()
            del_t = end_t - start_t
            self.hist_t = np.vstack((self.hist_t, del_t.total_seconds()))

            self.z_k = self.solver.z_k
            self.hist_z_k = np.vstack((self.hist_z_k, self.solver.z_k))

# Prepare 'results' directory
if not os.path.exists('results'):
    os.makedirs('results')
    
# ======================================
# RANDOMLY-GENERATED OPTIMIZATION RUNS
# ======================================
for bmk_D in list_dims:
    for bmk_prob in list_problem:
        print("==============")
        print("%s" % bmk_prob.upper())
        print("==============")
        
        for bmk_condnum in list_condnum:
            print("COND #: %d" % bmk_condnum)

            for bmk_solv in list_solver:
                print("    %s" % bmk_solv.upper())

                for bmk_noise in list_noise:
                    res_vec = np.empty(0)
                    res_auxs = np.empty(0)
                    Z0_vec   = np.empty(0)
                    res_hist = None
                    time_hist = None

                    for trial_i in range(bmk_maxtrials):
                        bmk_test = SolverTest(
                            problem=bmk_prob,
                            solver=bmk_solv,
                            maxevals=50*bmk_D,
                            dims=bmk_D,
                            condnum=bmk_condnum,
                            randseed=trial_i,
                            noisebnd=bmk_noise,
                        )
                        bmk_test.run()
                        
                        if res_hist is None:
                            res_hist  = bmk_test.hist_z_k
                            time_hist = bmk_test.hist_t
                        else:
                            res_hist  = np.hstack((res_hist, bmk_test.hist_z_k))
                            time_hist = np.hstack((time_hist, bmk_test.hist_t))

                        Z0_vec  = np.hstack((Z0_vec, bmk_test.Z0))
                        res_vec = np.hstack((res_vec, bmk_test.z_k))

                        if bmk_solv == "sage":
                            res_auxs = np.hstack(
                                (res_auxs, np.mean(bmk_test.solver.hist_aux_samples))
                            )

                    print(
                        "      NS: %f, Z0: %.6E, MN: %.6E, STD: %.6E, TM: %.6f"
                        % (bmk_noise, np.mean(Z0_vec), np.mean(res_vec), np.std(res_vec), np.mean(np.sum(time_hist, axis=0)))
                    )

                    # Save data to MAT
                    save_filename = "results/%dD-%s-%d-%s-%.6f.mat" % (
                        bmk_D,
                        bmk_prob,
                        bmk_condnum,
                        bmk_solv,
                        bmk_noise,
                    )
                    save_dict = {"res_hist": res_hist, "res_vec": res_vec, "time_hist": time_hist, "Z0_vec": Z0_vec}
                    if bmk_solv == "sage":
                        save_dict["auxs_hist"] = res_auxs
                    savemat(save_filename, save_dict)
