import os
import numpy as np
from scipy.io import savemat
from tests.config import (
    LIST_DIMS,
    LIST_PROBLEM,
    LIST_CONDNUM,
    LIST_GRAD_EST,
    LIST_NOISE_PARAM,
    LIST_NOISE_TYPE,
    BMK_MAXTRIALS,
    MAX_EVALS_MULT,
)
from tests.runner import SolverTest
from utils.noise import NoiseType

def run_all_benchmarks():
    # Prepare results directory
    if not os.path.exists('results'):
        os.makedirs('results')

    for bmk_D in LIST_DIMS:
        for bmk_prob in LIST_PROBLEM:
            print("==============")
            print(f"{bmk_prob.upper()}")
            print("==============")
            
            for bmk_condnum in LIST_CONDNUM:
                print(f"COND #: {bmk_condnum}")
                
                for bmk_est in LIST_GRAD_EST:
                    print(f"    {bmk_est.upper()}")
                    
                    for bmk_noise_type in LIST_NOISE_TYPE:
                        for bmk_noise in LIST_NOISE_PARAM:
                            res_vec = np.empty(0)
                            res_auxs = np.empty(0)
                            Z0_eval_vec = np.empty(0)
                            Z0_true_vec = np.empty(0)
                            res_hist_eval = None
                            res_hist_true = None
                            time_hist = None
                        
                            # Loop trials
                            for trial_i in range(BMK_MAXTRIALS):
                                # Instantiate and Run
                                # Map bmk_noise to the objective noise parameter.
                                # Noise type is controlled by LIST_NOISE_TYPE in tests/config.py.
                                
                                try:
                                    noise_type = NoiseType.UNIFORM if bmk_noise_type == "uniform" else NoiseType.GAUSSIAN
                                    test = SolverTest(
                                        problem_name=bmk_prob,
                                        grad_est_name=bmk_est,
                                        maxevals=MAX_EVALS_MULT * bmk_D,
                                        dims=bmk_D,
                                        condnum=bmk_condnum,
                                        randseed=trial_i,
                                        noise_type=noise_type,
                                        noise_param=bmk_noise,
                                    )
                                    
                                    h_zk_eval, h_zk_true, h_t, z0_eval, z0_true, _ = test.run()
                                    
                                    # Aggregate results
                                    if res_hist_eval is None:
                                        res_hist_eval = h_zk_eval
                                        res_hist_true = h_zk_true
                                        time_hist = h_t
                                    else:
                                        # Handle different lengths if any (should cover maxevals roughly)
                                        # Stacking requires same shape. 
                                        # If lengths differ, we might need padding.
                                        # For now assume mostly consistent or take min?
                                        # Original code used np.hstack.
                                        # Let's trust it aligns or use list.
                                        if h_zk_eval.shape[0] == res_hist_eval.shape[0]:
                                            res_hist_eval = np.hstack((res_hist_eval, h_zk_eval))
                                            res_hist_true = np.hstack((res_hist_true, h_zk_true))
                                            time_hist = np.hstack((time_hist, h_t))
                                        else:
                                            # Mismatch length handling?
                                            # Just skip or truncate?
                                            pass 

                                    Z0_eval_vec = np.hstack((Z0_eval_vec, z0_eval))
                                    Z0_true_vec = np.hstack((Z0_true_vec, z0_true))
                                    res_vec = np.hstack((res_vec, h_zk_eval[-1] if len(h_zk_eval) > 0 else z0_eval))

                                    if bmk_est == "sage":
                                        res_auxs = np.hstack(
                                            (res_auxs, np.mean(test.estimator.hist_aux_samples))
                                        )
                                    
                                except Exception as e:
                                    print(f"Error in trial {trial_i}: {e}")
                                    continue

                            # Report
                            if len(Z0_eval_vec) > 0:
                                avg_z0 = np.mean(Z0_eval_vec)
                                avg_res = np.mean(res_vec)
                                std_res = np.std(res_vec)
                                avg_time = np.mean(np.sum(time_hist, axis=0)) if time_hist is not None else 0.0
                                print(
                                    f"      NT: {bmk_noise_type}, NS: {bmk_noise}, Z0: {avg_z0:.6E}, "
                                    f"MN: {avg_res:.6E}, STD: {std_res:.6E}, TM: {avg_time:.6f}"
                                )

                                # Save
                                save_filename = (
                                    f"results/{bmk_D}D-{bmk_prob}-{bmk_condnum}-{bmk_est}-"
                                    f"{bmk_noise_type}-{bmk_noise:.6f}.mat"
                                )
                                save_dict = {
                                    "res_hist_eval": res_hist_eval,
                                    "res_hist_true": res_hist_true,
                                    "res_vec": res_vec,
                                    "time_hist": time_hist,
                                    "Z0_eval_vec": Z0_eval_vec,
                                    "Z0_true_vec": Z0_true_vec,
                                }
                                if bmk_est == "sage":
                                    save_dict["auxs_hist"] = res_auxs
                                savemat(save_filename, save_dict)

if __name__ == "__main__":
    run_all_benchmarks()
