"""
Gradient Accuracy Benchmark
============================
For each (dim, problem, condnum, noise_type, noise_param) combo from config,
generate GRAD_BMK_NPOINTS random test points, compute the true gradient at each,
then evaluate every estimator's gradient estimate and measure error.

Metrics per point:
  - Relative error:    ||g_hat - g_true|| / ||g_true||
  - Absolute error:    ||g_hat - g_true||
  - Cosine similarity: g_hat · g_true / (||g_hat|| ||g_true||)
  - Max component err: max_i |g_hat_i - g_true_i|

Results saved as .mat files in a timestamped run folder under results/,
alongside a config snapshot and stdout log, and printed to console.
"""

import argparse
import os
import sys
import warnings
from pathlib import Path

import numpy as np
import yaml

# Ensure project root is on path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from utils.benchmark_artifacts import copy_config, create_run_dir, get_git_commit, tee_stdout
from utils.config_coercion import coerce_float, coerce_int, coerce_numeric_list, coerce_optional_scalar
from utils.noise import NoiseType
from utils.history import HistoryBuffer

RESULTS_ROOT = Path("results")


def load_config(path: str) -> dict:
    """Load benchmark configuration from a YAML file."""
    with open(path, "r") as f:
        cfg = yaml.safe_load(f)

    if cfg is None:
        cfg = {}
    if not isinstance(cfg, dict):
        raise ValueError("Benchmark config must be a mapping")

    coerce_numeric_list(cfg, "list_dims", coerce_int)
    coerce_numeric_list(cfg, "list_condnum", coerce_float)
    coerce_numeric_list(cfg, "list_noise_param", coerce_float)
    coerce_optional_scalar(cfg, "grad_bmk_npoints", coerce_int)
    coerce_optional_scalar(cfg, "grad_bmk_nproblems", coerce_int)
    cfg.setdefault("sage_noise_bound_mode", "estimate")
    if cfg["sage_noise_bound_mode"] not in ("estimate", "known"):
        raise ValueError(
            "sage_noise_bound_mode must be 'estimate' or 'known', "
            f"got {cfg['sage_noise_bound_mode']!r}"
        )
    return cfg


# ---------------------------------------------------------------------------
# Metric helpers
# ---------------------------------------------------------------------------
EPS = 1e-12  # guard against division by zero


def _compute_metrics(g_hat: np.ndarray, g_true: np.ndarray):
    """Return (rel_err, abs_err, cos_sim, max_comp_err)."""
    diff = g_hat - g_true
    abs_err = float(np.linalg.norm(diff))
    g_true_norm = float(np.linalg.norm(g_true))
    g_hat_norm = float(np.linalg.norm(g_hat))

    rel_err = abs_err / g_true_norm if g_true_norm > EPS else np.nan
    if g_true_norm > EPS and g_hat_norm > EPS:
        cos_sim = float(np.dot(g_hat, g_true) / (g_hat_norm * g_true_norm))
    else:
        cos_sim = np.nan
    max_comp_err = float(np.max(np.abs(diff)))
    return rel_err, abs_err, cos_sim, max_comp_err


# ---------------------------------------------------------------------------
# Main benchmark
# ---------------------------------------------------------------------------
def run_gradient_benchmark(config_path: str):
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
            _run_gradient_benchmark(config_path, run_dir, metadata)


def _run_gradient_benchmark(config_path: str, output_dir: Path, metadata: dict):
    from scipy.io import savemat
    from tests.factories import create_problem, create_estimator

    cfg = load_config(config_path)

    LIST_DIMS = cfg["list_dims"]
    LIST_PROBLEM = cfg["list_problem"]
    LIST_CONDNUM = cfg["list_condnum"]
    LIST_NOISE_PARAM = cfg["list_noise_param"]
    LIST_NOISE_TYPE = cfg["list_noise_type"]
    GRAD_BMK_NPOINTS = cfg["grad_bmk_npoints"]
    GRAD_BMK_NPROBLEMS = cfg["grad_bmk_nproblems"]
    GRAD_BMK_ESTIMATORS = cfg["grad_bmk_estimators"]
    GRAD_BMK_DTYPE = np.dtype(cfg.get("grad_bmk_dtype", "float128"))
    GRAD_BMK_STEP = GRAD_BMK_DTYPE.type(cfg.get("gdtcalcstep", "1e-6"))
    SAGE_NOISE_BOUND_MODE = cfg["sage_noise_bound_mode"]

    output_dir = Path(output_dir)

    total_pts = GRAD_BMK_NPOINTS * GRAD_BMK_NPROBLEMS

    for bmk_D in LIST_DIMS:
        for bmk_prob in LIST_PROBLEM:
            for bmk_condnum in LIST_CONDNUM:

                # Pre-generate all problems and test points
                # For each random seed, create a problem + 250 test points
                problems = []
                X_tests = []
                G_trues = []
                for seed in range(1, GRAD_BMK_NPROBLEMS + 1):
                    problem = create_problem(bmk_prob, bmk_D, bmk_condnum, randseed=seed)
                    rng = np.random.RandomState(42 + seed)
                    X_test = GRAD_BMK_DTYPE.type("1e2") * (
                        rng.rand(GRAD_BMK_NPOINTS, bmk_D).astype(GRAD_BMK_DTYPE)
                        - GRAD_BMK_DTYPE.type("0.5")
                    )
                    G_true = np.array(
                        [problem.gradient(X_test[j]) for j in range(GRAD_BMK_NPOINTS)],
                        dtype=GRAD_BMK_DTYPE,
                    )
                    problems.append(problem)
                    X_tests.append(X_test)
                    G_trues.append(G_true)

                for bmk_noise_type in LIST_NOISE_TYPE:
                    noise_enum = (
                        NoiseType.UNIFORM if bmk_noise_type == "uniform" else NoiseType.GAUSSIAN
                    )

                    for bmk_noise in LIST_NOISE_PARAM:
                        print("=" * 60)
                        print(
                            f"GRAD BMK | {bmk_D}D {bmk_prob} cond={bmk_condnum:.0e} "
                            f"{bmk_noise_type} noise={bmk_noise:.6f} "
                            f"({GRAD_BMK_NPROBLEMS} problems x {GRAD_BMK_NPOINTS} pts) "
                            f"dtype={GRAD_BMK_DTYPE.name}"
                        )
                        print("=" * 60)

                        # SAGE noise_bound kwarg for this (noise_type, noise_param)
                        # combo, computed once per combo rather than per point.
                        sage_extra_kwargs = {}
                        if SAGE_NOISE_BOUND_MODE == "known":
                            if bmk_noise_type == "uniform":
                                # utils/noise.py draws uniform noise from
                                # [-bmk_noise/2, bmk_noise/2], so the true bound
                                # (max |noise|) is half the configured interval
                                # width, not the width itself.
                                sage_extra_kwargs["noise_bound"] = bmk_noise / 2.0
                            else:
                                # Gaussian noise has no hard bound; bmk_noise is a
                                # standard deviation, not an interval width, so
                                # "known" mode is not well-defined here. Warn and
                                # fall back to estimate-mode behavior (no
                                # noise_bound kwarg) for this run.
                                warnings.warn(
                                    "sage_noise_bound_mode='known' is not "
                                    "well-defined for Gaussian noise (no hard "
                                    "bound); falling back to estimate mode for "
                                    f"this run (noise={bmk_noise})."
                                )

                        for est_name in GRAD_BMK_ESTIMATORS:
                            is_sage = (est_name == "sage")

                            # Flat arrays across all problems × points
                            rel_errs = np.full(total_pts, np.nan)
                            abs_errs = np.full(total_pts, np.nan)
                            cos_sims = np.full(total_pts, np.nan)
                            max_errs = np.full(total_pts, np.nan)
                            n_evals  = np.zeros(total_pts, dtype=int)
                            if is_sage:
                                aux_step_sizes_counts = np.zeros(total_pts, dtype=int)
                                aux_step_sizes_flat = np.empty((0,))

                            for pi in range(GRAD_BMK_NPROBLEMS):
                                problem = problems[pi]
                                X_test = X_tests[pi]
                                G_true = G_trues[pi]

                                if is_sage:
                                    print(f"  {est_name:>10s} | Problem {pi+1}/{GRAD_BMK_NPROBLEMS}")

                                offset = pi * GRAD_BMK_NPOINTS
                                for j in range(GRAD_BMK_NPOINTS):
                                    idx = offset + j

                                    # Fresh history per query (memoryless)
                                    history = HistoryBuffer()

                                    def obj_func(
                                        x,
                                        _hist=history,
                                        _prob=problem,
                                        _nt=noise_enum,
                                        _noise_param=GRAD_BMK_DTYPE.type(bmk_noise),
                                        _dtype=GRAD_BMK_DTYPE,
                                    ):
                                        x = np.asarray(x, dtype=_dtype)
                                        val = _dtype.type(_prob.eval(x, _nt, _noise_param))
                                        _hist.add(x, val)
                                        return val

                                    extra_kwargs = sage_extra_kwargs if is_sage else {}

                                    try:
                                        estimator = create_estimator(
                                            est_name, obj_func, bmk_D, history,
                                            gdtcalcstep=GRAD_BMK_STEP,
                                            randseed=pi + 1,
                                            dtype=GRAD_BMK_DTYPE,
                                            **extra_kwargs,
                                        )
                                    except Exception as e:
                                        if pi == 0 and j == 0:
                                            print(f"  {est_name:>10s} |   ERROR creating estimator: {e}")
                                        continue

                                    evals_before = history.Zn.size
                                    try:
                                        g_hat = estimator(X_test[j])
                                        evals_after = history.Zn.size
                                        n_evals[idx] = evals_after - evals_before
                                        (rel_errs[idx], abs_errs[idx],
                                         cos_sims[idx], max_errs[idx]) = _compute_metrics(
                                            g_hat, G_true[j]
                                        )
                                        if is_sage and hasattr(estimator, "aux_step_sizes_current"):
                                            steps = np.asarray(estimator.aux_step_sizes_current).reshape(-1)
                                            aux_step_sizes_counts[idx] = steps.size
                                            if steps.size > 0:
                                                aux_step_sizes_flat = np.hstack((aux_step_sizes_flat, steps))
                                    except Exception as e:
                                        evals_after = history.Zn.size
                                        n_evals[idx] = evals_after - evals_before
                                        if is_sage:
                                            aux_step_sizes_counts[idx] = 0
                                        if pi == 0 and j == 0:
                                            print(f"  {est_name:>10s} |   point error: {e}")

                            # Aggregate statistics (ignoring NaN)
                            mean_rel = float(np.nanmean(rel_errs))
                            mean_abs = float(np.nanmean(abs_errs))
                            mean_cos = float(np.nanmean(cos_sims))
                            mean_max = float(np.nanmean(max_errs))
                            mean_ev  = float(np.mean(n_evals))

                            print(
                                f"  {est_name:>10s} | "
                                f"rel={mean_rel:.3e}  abs={mean_abs:.3e}  "
                                f"cos={mean_cos:.6f}  maxcomp={mean_max:.3e}  "
                                f"evals/pt={mean_ev:.1f}"
                            )

                            # Save .mat for this estimator
                            save_dict = {
                                "rel_err": rel_errs,
                                "abs_err": abs_errs,
                                "cos_sim": cos_sims,
                                "max_err": max_errs,
                                "n_evals": n_evals,
                            }
                            if is_sage:
                                save_dict["aux_step_sizes_flat"] = aux_step_sizes_flat
                                save_dict["aux_step_sizes_counts"] = aux_step_sizes_counts
                            save_dict.update(metadata)
                            fname = output_dir / (
                                f"grad-bmk-{bmk_D}D-{bmk_prob}-{bmk_condnum}-"
                                f"{est_name}-{bmk_noise_type}-{bmk_noise:.6f}.mat"
                            )
                            savemat(str(fname), save_dict)
                            print(f"  -> saved {fname}")

                        print()  # blank line after all estimators for this config


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Gradient Accuracy Benchmark")
    parser.add_argument("--config", required=True, help="Path to YAML config file")
    args = parser.parse_args()
    run_gradient_benchmark(args.config)
