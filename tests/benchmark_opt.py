"""
Optimization Benchmark
=======================
For each (dim, problem, condnum, estimator, noise_type, noise_param) combo
from config, run BMK_MAXTRIALS independent OptimizationTrial runs (trial
seed == trial index) and aggregate the per-evaluation accepted-iterate
history, success/eval-to-target statistics, and SAGE auxiliary-sample
summary into one .mat file per combo.

Results saved as .mat files in a timestamped run folder under results/,
alongside a config snapshot and stdout log.
"""

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import yaml

# Ensure project root is on path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from optimizers import StepSizeMode
from utils.benchmark_artifacts import copy_config, create_run_dir, get_git_commit, tee_stdout
from utils.config_coercion import (
    coerce_bool,
    coerce_float,
    coerce_int,
    coerce_numeric_list,
    coerce_optional_scalar,
    require_keys,
)
from utils.noise import NoiseType

RESULTS_ROOT = Path("results")

REQUIRED_KEYS = [
    "list_dims",
    "list_condnum",
    "list_noise_param",
    "list_noise_type",
    "list_problem",
    "list_grad_est",
    "bmk_maxtrials",
    "max_evals_mult",
]

# (key, coerce_fn, default) for optimizer/execution settings that fall back
# to SAGE-mother defaults when omitted from the config.
_OPTIONAL_SCALARS = [
    ("opt_bmk_dtype", None, "float128"),
    ("gdtcalcstep", coerce_float, 1e-6),
    ("stepsize_mode", None, "adaptive"),
    ("stepsize", coerce_float, 1.0),
    ("armijo_beta", coerce_float, 0.5),
    ("armijo_c", coerce_float, 1e-6),
    ("min_stepsize", coerce_float, 1e-6),
    ("max_line_search_iters", coerce_int, 100),
    ("recompute_grad_every_ls_failures", coerce_int, 5),
    ("reset_stepsize_at_floor", coerce_bool, True),
    ("sage_reset_on_step", coerce_bool, False),
    ("verbose", coerce_bool, False),
    ("save_eval_history", coerce_bool, False),
]


def load_config(path: str) -> dict:
    """Load and validate optimization benchmark configuration from a YAML file."""
    with open(path, "r") as f:
        cfg = yaml.safe_load(f)

    if cfg is None:
        cfg = {}
    if not isinstance(cfg, dict):
        raise ValueError("Benchmark config must be a mapping")

    require_keys(cfg, REQUIRED_KEYS)

    coerce_numeric_list(cfg, "list_dims", coerce_int)
    coerce_numeric_list(cfg, "list_condnum", coerce_float)
    coerce_numeric_list(cfg, "list_noise_param", coerce_float)
    coerce_optional_scalar(cfg, "bmk_maxtrials", coerce_int)
    coerce_optional_scalar(cfg, "max_evals_mult", coerce_int)

    for key, coerce, default in _OPTIONAL_SCALARS:
        if coerce is None:
            cfg.setdefault(key, default)
        else:
            coerce_optional_scalar(cfg, key, coerce, default=default)

    if key_error := _validate_choice(cfg, "stepsize_mode", ("adaptive", "fixed")):
        raise ValueError(key_error)

    if "target_ratios" in cfg:
        coerce_numeric_list(cfg, "target_ratios", coerce_float)
    else:
        cfg["target_ratios"] = [0.1, 0.01, 0.001]

    return cfg


def _validate_choice(cfg: dict, key: str, choices) -> str:
    if cfg.get(key) not in choices:
        return f"{key} must be one of {choices}, got {cfg.get(key)!r}"
    return ""


def run_optimization_benchmark(config_path: str):
    """Public entry point: sets up run artifacts, then runs the benchmark."""
    run_dir = create_run_dir(RESULTS_ROOT)
    config_snapshot = copy_config(config_path, run_dir)
    log_path = run_dir / "log.txt"

    metadata = {
        "config_path": str(config_snapshot),
        "output_dir": str(run_dir),
        "run_timestamp": run_dir.name,
        "git_commit": get_git_commit(),
    }

    with open(log_path, "w") as log_file:
        with tee_stdout(log_file):
            print(f"Benchmark output directory: {run_dir}")
            print(f"Config snapshot: {config_snapshot}")
            print(f"Stdout log: {log_path}")
            _run_optimization_benchmark(config_path, run_dir, metadata)


def _run_optimization_benchmark(config_path: str, output_dir: Path, metadata: dict):
    from scipy.io import savemat

    from tests.opt_runner import OptimizationTrial

    cfg = load_config(config_path)

    LIST_DIMS = cfg["list_dims"]
    LIST_PROBLEM = cfg["list_problem"]
    LIST_CONDNUM = cfg["list_condnum"]
    LIST_GRAD_EST = cfg["list_grad_est"]
    LIST_NOISE_PARAM = cfg["list_noise_param"]
    LIST_NOISE_TYPE = cfg["list_noise_type"]
    BMK_MAXTRIALS = cfg["bmk_maxtrials"]
    MAX_EVALS_MULT = cfg["max_evals_mult"]

    OPT_BMK_DTYPE = np.dtype(cfg["opt_bmk_dtype"])
    GDTCALCSTEP = cfg["gdtcalcstep"]
    STEPSIZE_MODE = StepSizeMode.ADAPTIVE if cfg["stepsize_mode"] == "adaptive" else StepSizeMode.FIXED
    STEPSIZE = cfg["stepsize"]
    ARMIJO_BETA = cfg["armijo_beta"]
    ARMIJO_C = cfg["armijo_c"]
    MIN_STEPSIZE = cfg["min_stepsize"]
    MAX_LINE_SEARCH_ITERS = cfg["max_line_search_iters"]
    RECOMPUTE_GRAD_EVERY_LS_FAILURES = cfg["recompute_grad_every_ls_failures"]
    RESET_STEPSIZE_AT_FLOOR = cfg["reset_stepsize_at_floor"]
    SAGE_RESET_ON_STEP = cfg["sage_reset_on_step"]
    VERBOSE = cfg["verbose"]
    TARGET_RATIOS = cfg["target_ratios"]
    N_TARGETS = len(TARGET_RATIOS)

    output_dir = Path(output_dir)

    for bmk_D in LIST_DIMS:
        for bmk_prob in LIST_PROBLEM:
            for bmk_condnum in LIST_CONDNUM:
                for bmk_est in LIST_GRAD_EST:
                    is_sage = (bmk_est == "sage")

                    for bmk_noise_type in LIST_NOISE_TYPE:
                        noise_enum = (
                            NoiseType.UNIFORM if bmk_noise_type == "uniform" else NoiseType.GAUSSIAN
                        )

                        for bmk_noise in LIST_NOISE_PARAM:
                            max_evals = MAX_EVALS_MULT * bmk_D

                            print("=" * 60)
                            print(
                                f"OPT BMK | {bmk_D}D {bmk_prob} cond={bmk_condnum:.0e} "
                                f"{bmk_est} {bmk_noise_type} noise={bmk_noise:.6f} "
                                f"({BMK_MAXTRIALS} trials x {max_evals} max_evals)"
                            )
                            print("=" * 60)

                            res_hist_eval = np.full((max_evals, BMK_MAXTRIALS), np.nan)
                            res_hist_true = np.full((max_evals, BMK_MAXTRIALS), np.nan)
                            time_hist = np.full((max_evals, BMK_MAXTRIALS), np.nan)
                            Z_initial_eval_vec = np.full(BMK_MAXTRIALS, np.nan)
                            Z_initial_true_vec = np.full(BMK_MAXTRIALS, np.nan)
                            Z_start_eval_vec = np.full(BMK_MAXTRIALS, np.nan)
                            Z_start_true_vec = np.full(BMK_MAXTRIALS, np.nan)
                            final_eval = np.full(BMK_MAXTRIALS, np.nan)
                            final_true = np.full(BMK_MAXTRIALS, np.nan)
                            n_evals = np.zeros(BMK_MAXTRIALS, dtype=int)
                            evals_to_target = np.full((N_TARGETS, BMK_MAXTRIALS), np.nan)
                            success_by_target = np.zeros((N_TARGETS, BMK_MAXTRIALS), dtype=bool)
                            trial_status = np.empty(BMK_MAXTRIALS, dtype=object)
                            trial_error = np.empty(BMK_MAXTRIALS, dtype=object)
                            auxs_hist = np.full(BMK_MAXTRIALS, np.nan) if is_sage else None

                            # Per-call SAGE diagnostics (Milestone 6): one row per
                            # public SAGE.__call__ invocation, aligned across trials
                            # and NaN/""-padded like the per-evaluation history above.
                            # A trial has at most one SAGE call per optimizer step, so
                            # `max_evals` rows is always enough.
                            if is_sage:
                                sage_diag_eval_index = np.full((max_evals, BMK_MAXTRIALS), np.nan)
                                sage_diag_hist_size = np.full((max_evals, BMK_MAXTRIALS), np.nan)
                                sage_diag_n_neighbors = np.full((max_evals, BMK_MAXTRIALS), np.nan)
                                sage_diag_n_aux = np.full((max_evals, BMK_MAXTRIALS), np.nan)
                                sage_diag_calibration_attempted = np.full((max_evals, BMK_MAXTRIALS), np.nan)
                                sage_diag_calibration_fixed = np.full((max_evals, BMK_MAXTRIALS), np.nan)
                                sage_diag_stop_reason = np.full((max_evals, BMK_MAXTRIALS), "", dtype=object)
                                sage_diag_n_calls = np.zeros(BMK_MAXTRIALS, dtype=int)

                            for trial_i in range(BMK_MAXTRIALS):
                                try:
                                    trial = OptimizationTrial(
                                        problem_name=bmk_prob,
                                        grad_est_name=bmk_est,
                                        maxevals=max_evals,
                                        dims=bmk_D,
                                        condnum=bmk_condnum,
                                        randseed=trial_i,
                                        noise_type=noise_enum,
                                        noise_param=bmk_noise,
                                        gdtcalcstep=GDTCALCSTEP,
                                        dtype=OPT_BMK_DTYPE.type,
                                        stepsize=STEPSIZE,
                                        stepsizemode=STEPSIZE_MODE,
                                        armijo_beta=ARMIJO_BETA,
                                        armijo_c=ARMIJO_C,
                                        min_stepsize=MIN_STEPSIZE,
                                        max_line_search_iters=MAX_LINE_SEARCH_ITERS,
                                        recompute_grad_every_ls_failures=RECOMPUTE_GRAD_EVERY_LS_FAILURES,
                                        reset_stepsize_at_floor=RESET_STEPSIZE_AT_FLOOR,
                                        sage_reset_on_step=SAGE_RESET_ON_STEP,
                                        verbose=VERBOSE,
                                    )
                                    result = trial.run()
                                except Exception as e:
                                    trial_status[trial_i] = "error"
                                    trial_error[trial_i] = str(e)
                                    print(f"  trial {trial_i}: ERROR: {e}")
                                    continue

                                trial_status[trial_i] = "ok"
                                trial_error[trial_i] = ""

                                h_eval = np.asarray(result["res_hist_eval"]).reshape(-1)
                                h_true = np.asarray(result["res_hist_true"]).reshape(-1)
                                h_t = np.asarray(result["time_hist"]).reshape(-1)
                                n = h_eval.size

                                res_hist_eval[:n, trial_i] = h_eval
                                res_hist_true[:n, trial_i] = h_true
                                time_hist[:n, trial_i] = h_t

                                Z_initial_eval_vec[trial_i] = result["Z_initial_eval"]
                                Z_initial_true_vec[trial_i] = result["Z_initial_true"]
                                Z_start_eval_vec[trial_i] = result["Z_start_eval"]
                                Z_start_true_vec[trial_i] = result["Z_start_true"]
                                n_evals[trial_i] = result["n_evals"]

                                if n > 0:
                                    final_eval[trial_i] = h_eval[-1]
                                    final_true[trial_i] = h_true[-1]

                                denom = result["Z_start_true"]
                                if denom and not np.isnan(denom) and n > 0:
                                    ratios = h_true / denom
                                    for k, target in enumerate(TARGET_RATIOS):
                                        hits = np.nonzero(ratios <= target)[0]
                                        if hits.size > 0:
                                            evals_to_target[k, trial_i] = hits[0] + 1
                                            success_by_target[k, trial_i] = True

                                if is_sage:
                                    aux_samples = getattr(trial.estimator, "hist_aux_samples", None)
                                    if aux_samples is not None and aux_samples.size > 0:
                                        auxs_hist[trial_i] = float(np.mean(aux_samples))

                                    call_diagnostics = getattr(trial.estimator, "call_diagnostics", None) or []
                                    n_calls = min(len(call_diagnostics), max_evals)
                                    sage_diag_n_calls[trial_i] = len(call_diagnostics)
                                    for ci in range(n_calls):
                                        d = call_diagnostics[ci]
                                        sage_diag_eval_index[ci, trial_i] = d.eval_index
                                        sage_diag_hist_size[ci, trial_i] = d.hist_size
                                        sage_diag_n_neighbors[ci, trial_i] = d.n_neighbors
                                        sage_diag_n_aux[ci, trial_i] = d.n_aux
                                        sage_diag_calibration_attempted[ci, trial_i] = float(d.calibration_attempted)
                                        sage_diag_calibration_fixed[ci, trial_i] = float(d.calibration_fixed)
                                        sage_diag_stop_reason[ci, trial_i] = d.stop_reason

                                print(
                                    f"  trial {trial_i}: ok  evals={result['n_evals']}  "
                                    f"final_true={final_true[trial_i]:.6e}"
                                )

                            # Last recorded (non-NaN) row per trial column, extracted
                            # from the padded matrices, as a cross-check against final_*.
                            last_hist_eval = np.full(BMK_MAXTRIALS, np.nan)
                            last_hist_true = np.full(BMK_MAXTRIALS, np.nan)
                            for trial_i in range(BMK_MAXTRIALS):
                                valid = np.nonzero(~np.isnan(res_hist_true[:, trial_i]))[0]
                                if valid.size > 0:
                                    last_idx = valid[-1]
                                    last_hist_true[trial_i] = res_hist_true[last_idx, trial_i]
                                    last_hist_eval[trial_i] = res_hist_eval[last_idx, trial_i]

                            n_ok = int(np.sum(trial_status == "ok"))
                            print(
                                f"  -> {n_ok}/{BMK_MAXTRIALS} trials ok; "
                                f"success@{TARGET_RATIOS}: "
                                f"{np.nanmean(success_by_target, axis=1) if n_ok else 'n/a'}"
                            )

                            save_dict = {
                                "res_hist_true": res_hist_true,
                                "res_hist_eval": res_hist_eval,
                                "time_hist": time_hist,
                                "Z_initial_true_vec": Z_initial_true_vec,
                                "Z_initial_eval_vec": Z_initial_eval_vec,
                                "Z_start_true_vec": Z_start_true_vec,
                                "Z_start_eval_vec": Z_start_eval_vec,
                                "Z0_true_vec": Z_start_true_vec.copy(),
                                "Z0_eval_vec": Z_start_eval_vec.copy(),
                                "final_true": final_true,
                                "final_eval": final_eval,
                                "last_hist_true": last_hist_true,
                                "last_hist_eval": last_hist_eval,
                                "n_evals": n_evals,
                                "target_ratios": np.array(TARGET_RATIOS, dtype=float),
                                "evals_to_target": evals_to_target,
                                "success_by_target": success_by_target,
                                "trial_status": trial_status,
                                "trial_error": trial_error,
                            }
                            if is_sage:
                                save_dict["auxs_hist"] = auxs_hist
                                save_dict["sage_diag_n_calls"] = sage_diag_n_calls
                                save_dict["sage_diag_eval_index"] = sage_diag_eval_index
                                save_dict["sage_diag_hist_size"] = sage_diag_hist_size
                                save_dict["sage_diag_n_neighbors"] = sage_diag_n_neighbors
                                save_dict["sage_diag_n_aux"] = sage_diag_n_aux
                                save_dict["sage_diag_calibration_attempted"] = sage_diag_calibration_attempted
                                save_dict["sage_diag_calibration_fixed"] = sage_diag_calibration_fixed
                                save_dict["sage_diag_stop_reason"] = sage_diag_stop_reason

                            save_dict.update(metadata)
                            save_dict.update({
                                "max_evals": max_evals,
                                "problem": bmk_prob,
                                "dim": bmk_D,
                                "condnum": bmk_condnum,
                                "estimator": bmk_est,
                                "noise_type": bmk_noise_type,
                                "noise_param": bmk_noise,
                                "stepsize_mode": cfg["stepsize_mode"],
                                "stepsize": STEPSIZE,
                                "armijo_beta": ARMIJO_BETA,
                                "armijo_c": ARMIJO_C,
                                "min_stepsize": MIN_STEPSIZE,
                                "max_line_search_iters": MAX_LINE_SEARCH_ITERS,
                                "recompute_grad_every_ls_failures": RECOMPUTE_GRAD_EVERY_LS_FAILURES,
                                "reset_stepsize_at_floor": RESET_STEPSIZE_AT_FLOOR,
                                "sage_reset_on_step": SAGE_RESET_ON_STEP,
                                "opt_bmk_dtype": cfg["opt_bmk_dtype"],
                                "gdtcalcstep": GDTCALCSTEP,
                            })

                            fname = output_dir / (
                                f"opt-bmk-{bmk_D}D-{bmk_prob}-{bmk_condnum}-"
                                f"{bmk_est}-{bmk_noise_type}-{bmk_noise:.6f}.mat"
                            )
                            savemat(str(fname), save_dict)
                            print(f"  -> saved {fname}")
                            print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Optimization Benchmark")
    parser.add_argument("--config", required=True, help="Path to YAML config file")
    args = parser.parse_args()
    run_optimization_benchmark(args.config)
