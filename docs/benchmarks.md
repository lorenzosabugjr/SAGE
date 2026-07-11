# Benchmarks

This document summarizes the benchmark setup used in the paper (https://arxiv.org/abs/2508.19400) and implemented in `tests/`.

## 1. Problems (P1-P5)

The benchmark problems map to the `problems/` modules:

- **P1 (Least Squares):** `LeastSquares` in `problems/linear.py`
- **P2 (L1-regularized Least Squares):** `Lasso` in `problems/linear.py`
- **P3 (Log-sum-exp):** `LogSumExp` in `problems/misc.py`
- **P4 (L1-regularized Logistic Regression):** `L1LogReg` in `problems/logistic.py`
- **P5 (L2-regularized Logistic Regression):** `L2LogReg` in `problems/logistic.py`

These match the definitions listed in the paper.

## 2. Settings

Defaults are specified in `tests/config_benchmark_grad.yaml`:

- Dimensions: `[20]`
- Condition numbers: `[1e8]`
- Noise parameters: `[0.0]`
- Noise types: `["uniform"]`
- Problems per config: `grad_bmk_nproblems: 100` (random seeds 1–100)
- Test points per problem: `grad_bmk_npoints: 250`

## 3. Estimators and defaults

The benchmark supports the following gradient estimators (set `grad_bmk_estimators` in the YAML config):

- `ffd` / `cfd`: Forward / Central Finite Differences, `step = gdtcalcstep`
- `gsg` / `cgsg`: Gaussian Smoothing, `m = D`, `u = gdtcalcstep`, `seed = problem_seed`
- `nmxfd`: Normalized Mixed Finite Differences, `sigma = gdtcalcstep`
- `sage`: SAGE with `quickmode=True`, `diam_mode="approx"`, `init_step = gdtcalcstep`
- `truth`: Analytical gradient (reference, not typically benchmarked)

Factory functions in `tests/factories.py` instantiate problems and estimators by name.

### SAGE noise-bound mode

The YAML config's top-level `sage_noise_bound_mode` key controls whether SAGE estimates
the noise bound or is given the benchmark's own noise parameter as a fixed bound:

```yaml
sage_noise_bound_mode: "estimate"  # or "known"
```

- `"estimate"` (default): SAGE estimates `eps` from data via the LP, as it always has.
- `"known"`: for each benchmark noise setting, the runner passes a calibrated
  `noise_bound` into `create_estimator("sage", ...)`, so SAGE uses the benchmark noise
  level as a fixed bound and drops `eps` from the LP. The calibration depends on the
  noise type:
  - Uniform: `noise_bound = bmk_noise / 2.0`. `utils/noise.py` draws uniform noise from
    `[-bmk_noise/2, bmk_noise/2]`, so `bmk_noise` itself is the full interval width, not
    the true bound (max `|noise|`) that SAGE expects.
  - Gaussian: `bmk_noise` is a standard deviation, not a hard bound, so `"known"` mode
    is not well-defined. The runner emits a warning and does not pass `noise_bound` for
    that run (equivalent to falling back to `"estimate"` mode).

This only changes what is passed into the `sage` estimator; it does not affect other
estimators, the noise generator, or how `noise_param` is interpreted. Result filenames and
the saved `.mat` schema are unchanged between modes — the copied config snapshot in each
run's results folder is what distinguishes an `"estimate"` run from a `"known"` run.

## 4. Running benchmarks

From the repo root:

```bash
python -m tests.benchmark_grad --config tests/config_benchmark_grad.yaml
```

Each run creates a self-contained, timestamped folder under `results/`,
named with the local start time (`YYYY-MM-DD HH-MM-SS`):

```text
results/2026-07-04 23-18-42/
  config_benchmark_grad.yaml
  log.txt
  grad-bmk-...mat
```

- If a run directory for the same second already exists, a `_02`, `_03`, ...
  suffix is appended; existing run directories are never overwritten.
- The folder contains a raw, byte-preserving copy of the config file passed
  via `--config`, and all `.mat` result files produced by that run.
- `log.txt` mirrors stdout: benchmark output is printed live to the terminal
  and simultaneously written to `log.txt` (stderr is not captured).
- On a crash or exception, the partial run folder (including whatever config
  copy, log, and `.mat` files were written so far) is intentionally left in
  place rather than cleaned up.

## 5. Metrics

Per test point, the benchmark computes:

- **Relative error**: `||g_hat - g_true|| / ||g_true||`
- **Absolute error**: `||g_hat - g_true||`
- **Cosine similarity**: `g_hat · g_true / (||g_hat|| ||g_true||)`
- **Max component error**: `max_i |g_hat_i - g_true_i|`

The `.mat` files include:

- `rel_err`, `abs_err`, `cos_sim`, `max_err`: per-point metric arrays
- `n_evals`: number of function evaluations per point
- `aux_step_sizes_flat`, `aux_step_sizes_counts`: auxiliary step sizes (SAGE only)
- `config_path`, `output_dir`, `run_timestamp`, `git_commit`: run metadata,
  additive fields describing which config, run folder, and commit produced
  the file (`git_commit` is `"unknown"` outside a git checkout)

## 6. Plotting rel_err / cos_sim histograms

`utils/plot_grad_results.py` reads a YAML settings file and, for each
requested `(dim, problem, condnum, noise_type, noise_param)` combination,
generates one PDF with stacked `rel_err` (log-scale, shared log bins) and
`cos_sim` (shared linear bins) step histograms across the configured
estimators, plus a `grad_hist_summary.csv` with per-estimator counts and
quantiles.

```bash
python -m utils.plot_grad_results --options docs/config_plot_grad.yaml --dry-run
python -m utils.plot_grad_results --options docs/config_plot_grad.yaml
```

See `docs/config_plot_grad.yaml` for a documented example (required fields:
`source_dirs`, `dims`, `problems`, `condnums`, `noise_types`, `noise_params`,
`estimators`). Matching is filename-based (works on files with or without
the metadata fields above), duplicate matches raise an error listing all
duplicate paths, and missing matches raise by default (`missing_policy:
warn_skip` produces partial plots instead).

`results_plot_grad.m` is an equivalent MATLAB script for interactive use
(user-editable settings at the top, no YAML parsing needed).
