# Benchmarks

This document summarizes the benchmark setup used in the paper (https://arxiv.org/abs/2508.19400) and implemented in `tests/`.

There are two independent benchmark families:

- **Gradient accuracy benchmarks** (`tests/benchmark_grad.py`): compare
  estimated gradients against the true gradient at fixed test points, with
  no optimization involved. See sections 1-6 below.
- **Optimization benchmarks** (`tests/benchmark_opt.py`): run
  `optimizers.GradientDescent` end-to-end with each gradient estimator and
  record data-profile-style success statistics. See sections 7-9.

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

## 7. Optimization benchmark

`tests/benchmark_opt.py` runs `optimizers.GradientDescent` (plain gradient
descent with an adaptive Armijo line search) on each configured
`(dim, problem, condnum, estimator, noise_type, noise_param)` combo for
`bmk_maxtrials` independent trials (trial seed == trial index), and
aggregates the per-evaluation accepted-iterate history plus success/
eval-to-target statistics into one `.mat` file per combo.

```bash
python -m tests.benchmark_opt --config tests/config_benchmark_opt.yaml
```

Run artifacts follow the same convention as the gradient benchmark:
a timestamped folder under `results/` containing a byte-preserving config
copy, `log.txt`, and the `.mat` result files, named:

```text
opt-bmk-{D}D-{problem}-{condnum}-{estimator}-{noise_type}-{noise_param:.6f}.mat
```

Two small diagnostic configs are also provided for quick manual runs:
`tests/config_benchmark_opt_diag.yaml` and
`tests/config_benchmark_opt_diag_ls.yaml`.

### SAGE: one estimator, two history regimes

`sage` is included in the production `list_grad_est` alongside `ffd`, `cfd`,
`gsg`, `cgsg`, and `nmxfd`. It is the *same* `SAGE` class, constructed with
the *same* defaults, used by the gradient-accuracy benchmark (Section 3):
`quickmode=True`, `diam_mode="approx"`, `init_step = gdtcalcstep`, default
`rel_tol`, and estimated noise-bound mode. The two benchmark families are
not comparing different SAGE variants — they differ only in what history is
available when SAGE is called:

- the gradient-accuracy benchmark builds a fresh, empty `HistoryBuffer` for
  every isolated query, so each call's stencil and any auxiliary sampling
  start from nothing;
- the optimization benchmark shares one `HistoryBuffer` across the whole
  trial (`sage_reset_on_step: false`), so a later SAGE call can reuse the
  initial stencil, prior gradient calls' auxiliary points, and every
  accepted/rejected line-search evaluation already charged to the budget.

See "SAGE per-call diagnostics" in Section 8 for how to measure the effect
of that reuse from the saved `.mat` fields.

### SAGE-mother compatibility behavior

This port intentionally preserves the following SAGE-mother computational
behaviors, since the goal of the port is to reproduce mother's optimization
results rather than redesign them:

- **Evaluation budget** (`max_evals = max_evals_mult * D`) counts *every*
  objective evaluation against the same budget: the initial evaluation,
  SAGE's own initialization samples, rejected line-search trial points, and
  accepted line-search trial points. Budget exhaustion (`StopIteration`) is
  normal termination, not a failed trial.
- **Rejected line-search points** are still passed to the estimator's
  history via a lightweight update, so SAGE (and other stateful estimators)
  benefit from every function call, not just accepted steps.
- **SAGE optimizer start is centered**: the optimizer always starts at the
  original sampled point `X_initial`, which is also the center of SAGE's
  initial stencil and first gradient call. `Z_start_eval`/`Z_start_true`
  therefore equal `Z_initial_eval`/`Z_initial_true` in every current SAGE
  trial. (Earlier versions of this harness reassigned the SAGE start to the
  best point already present in SAGE's initialized history; that
  reassignment decentered the stencil the estimator had just built and has
  been removed.)
- **Two initial objective values are recorded per trial**:
  `Z_initial_*` (the original sampled initial point) and `Z_start_*` (the
  actual point optimization starts from — equal to `Z_initial_*` for
  SAGE trials, kept as a separate field pair for schema compatibility with
  non-SAGE estimators and older result files). Success ratios and
  `evals_to_target` use the deterministic **true** objective at `Z_start`
  as the denominator, never `Z_initial`.
- **One SAGE estimator, two history regimes**: both benchmark families call
  the same `SAGE` class with the same LP, sample selector, point estimate,
  stopping criterion, sampling radius, auxiliary policy, cap, and numerical
  settings (`quickmode=True`, `diam_mode="approx"`, `init_step =
  gdtcalcstep`, default `rel_tol`, estimated noise-bound mode). They differ
  only in what history SAGE sees when called: the gradient-accuracy
  benchmark gives every query a fresh, empty `HistoryBuffer`, while the
  optimization benchmark shares one `HistoryBuffer` (`reset_on_step:
  false`) across the initial stencil, every later gradient call, and every
  accepted/rejected line-search evaluation for the whole trial — so a later
  SAGE call can reuse samples an earlier call or line search already paid
  for.
- **Accepted-iterate history** (`res_hist_true` / `res_hist_eval` /
  `time_hist`) is recorded once per objective evaluation (not once per
  accepted step), and is padded to exactly `max_evals` rows with `NaN`.
  Initial-stencil and auxiliary evaluations forward-fill the current
  accepted iterate (they are observations, not iterates); rejected
  line-search evaluations likewise forward-fill the old iterate; an
  accepted line-search (or fixed-step) evaluation records the *newly*
  accepted iterate on that same evaluation's row, not the following one.
- Failed trials (e.g. an unknown problem name) remain represented in the
  output with `trial_status="error"` and a `trial_error` message; their
  history stays fully `NaN` and they count as unsolved for every target
  ratio.

## 8. Optimization benchmark artifact fields

Each `opt-bmk-*.mat` file contains:

- `res_hist_true`, `res_hist_eval`, `time_hist`: `(max_evals, n_trials)`,
  `NaN`-padded per-evaluation accepted-iterate history (true objective,
  evaluated/noisy objective, and elapsed `time.perf_counter()` seconds).
- `Z_initial_true_vec` / `Z_initial_eval_vec`: original sampled initial
  objective values, per trial.
- `Z_start_true_vec` / `Z_start_eval_vec` (aliased as `Z0_true_vec` /
  `Z0_eval_vec`): actual optimizer-start objective values, per trial.
- `final_true` / `final_eval`, `last_hist_true` / `last_hist_eval`: final
  recorded objective values per trial (the `last_hist_*` fields are
  extracted from the padded history as a cross-check against `final_*`).
- `n_evals`: number of objective evaluations consumed by each trial.
- `target_ratios`: the configured target success ratios.
- `evals_to_target`, `success_by_target`: `(n_targets, n_trials)`, the
  number of evaluations to first reach `true_objective / Z_start_true <=
  target_ratio`, and whether that target was reached at all.
- `trial_status`, `trial_error`: per-trial `"ok"` / `"error"` status and
  any error message.
- `auxs_hist`: SAGE-only, mean auxiliary-sample count per trial.
- Metadata: `config_path`, `output_dir`, `run_timestamp`, `git_commit`,
  `max_evals`, `problem`, `dim`, `condnum`, `estimator`, `noise_type`,
  `noise_param`, `stepsize_mode`, `stepsize`, line-search parameters,
  `opt_bmk_dtype`, `gdtcalcstep`.

### SAGE per-call diagnostics

`sage`-estimator `.mat` files additionally contain one compact diagnostic
record per public `SAGE.__call__` invocation for the trial (one call per
optimizer gradient estimation or forced refresh; *not* one row per line-search
trial point). These fields let the optimization benchmark show whether
accumulated history reduces fresh auxiliary sampling, without changing SAGE's
estimation or stopping mechanism:

- `sage_diag_n_calls`: `(n_trials,)`, the number of SAGE calls made in each
  trial — how many of the leading rows in the fields below are populated for
  that trial's column.
- `sage_diag_eval_index`, `sage_diag_hist_size`, `sage_diag_n_neighbors`,
  `sage_diag_n_aux`: `(max_evals, n_trials)`, `NaN`-padded beyond
  `sage_diag_n_calls[trial]` (like `res_hist_true`/`res_hist_eval`). Row `ci`
  of trial `trial_i` describes that trial's `(ci+1)`-th SAGE call:
  - `eval_index`: the trial-wide objective-evaluation count *before* this
    call added any evaluations of its own (unaffected by `reset_on_step`).
  - `hist_size`: the raw history size actually available to this call's
    selector at call start (equal to `eval_index` unless `reset_on_step`
    just wiped it). The gap between `hist_size` and `eval_index` across
    calls in a trial is the reuse signal the optimization benchmark is
    meant to surface; both are always equal in the gradient-accuracy
    benchmark's fresh-history calls.
  - `n_neighbors`: number of samples selected by SAGE's existing selector
    for the LP that produced the returned gradient estimate.
  - `n_aux`: number of auxiliary objective evaluations this call added
    before returning.
- `sage_diag_stop_reason`: `(max_evals, n_trials)` array of stopping-reason
  code strings (`""` beyond `sage_diag_n_calls[trial]`), one of:
  - `relative_criterion`: the relative-diameter criterion
    `gd_vm < rel_tol * ||gdt_est||` was met.
  - `noiseless_floor`: the noiseless absolute floor was met.
  - `forced_stop`: the call was forced to stop (e.g. `force=True` path).
  - `auxiliary_cap`: the `2*D` auxiliary-sample cap was reached.
  - `no_aux_direction`: refinement stopped because no further auxiliary
    direction was available.
  - `budget_exhaustion`: the call ended mid-refinement because the shared
    evaluation budget ran out (`StopIteration`); still produces one
    diagnostic row.
  - `stale_estimate`: a rare defensive early-return path where the query
    point drifted from the LP's `x_current` between recompute and
    refinement; not one of the `_should_stop_refinement` conditions above.
- `sage_diag_calibration_attempted`, `sage_diag_calibration_fixed`:
  `(max_evals, n_trials)`, `1.0`/`0.0` (`NaN`-padded like the fields above).
  `calibration_attempted` marks that estimate-mode noise self-calibration
  (`_maybe_calibrate_noise`) was invoked on that call; `calibration_fixed`
  marks that it switched to a fixed noise bound (it can be attempted and
  intentionally not fix a bound, e.g. when the seed LP is already
  consistent with zero noise).

These diagnostics never feed back into estimator decisions, and full raw
`Xn`/`Zn` histories are not saved by default — only this compact per-call
summary.

## 9. Plotting optimization data profiles

`utils/plot_opt_profiles.py` reads a YAML settings file and, for each
requested `(dim, problem, condnum, noise_type, noise_param)` combination
and each configured target ratio, generates one PDF data profile: the
fraction of trials solved (y-axis) as a function of function evaluations
divided by dimension (x-axis, default) or raw evaluations
(`x_axis: raw_evals`), with one line per configured estimator. It also
writes an `opt_profile_summary.csv` with per-`(combo, target_ratio,
estimator)` trial counts, success rate, and solved-trial median/mean
evaluations to target.

```bash
python -m utils.plot_opt_profiles --options docs/config_plot_opt.yaml --dry-run
python -m utils.plot_opt_profiles --options docs/config_plot_opt.yaml
```

See `docs/config_plot_opt.yaml` for a documented example (required fields:
`source_dirs`, `dims`, `problems`, `condnums`, `noise_types`,
`noise_params`, `estimators`). Matching is filename-based; duplicate
matches raise an error listing all duplicate paths, and missing matches
raise by default (`missing_policy: warn_skip` produces partial plots
instead).
