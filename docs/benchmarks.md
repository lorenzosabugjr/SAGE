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

Defaults in `tests/config.py`:

- Dimensions: `D in {5, 10, 20}`
- Condition numbers: `{1.0, 1e4, 1e8}`
- Noise parameters: `{1.0, 1e-3, 0.0}`
- Trials: `BMK_MAXTRIALS = 100`
- Evaluation budget: `50 * D`

The benchmarks in `tests/run_benchmarks.py` use **uniform noise** by default. If you want Gaussian noise, update the `noise_type` in that script.

## 3. Estimators and defaults

The benchmark runner uses the following gradient estimators:

- `FFD`: step size `1e-6`
- `CFD`: step size `1e-6`
- `GSG` / `cGSG`: `m = D`, `u = 1e-6`, `seed = trial_id`
- `NMXFD`: default parameters in `estimators/finite_diff.py`
- `SAGE`: uses a simplex-like initial history around `x0`

All gradient estimators are run through the state-machine optimizers in `optimizers/state_machine.py`, which include backtracking line search.

For deterministic comparisons, pass a fixed `seed` to `GSGOpt`/`cGSGOpt` (state-machine path). The
default benchmarks keep RNG unseeded to mirror the original repository behavior.

## 4. Running benchmarks

From the repo root:

```bash
python tests/run_benchmarks.py
```

You can adjust parameters in `tests/config.py`. Results are saved to `results/` as `.mat` files.

## 5. Metrics

The paper uses:

```
sigma1 = z_N / z_1
sigma2 = (1/N) * sum_{n=1}^N (z_n / z_1)
```

The `.mat` files include:

- `res_hist`: history of `z_k`
- `res_vec`: final `z_k` for each trial
- `time_hist`: per-iteration time
- `Z0_vec`: initial values `z_1`
- `auxs_hist`: mean auxiliary samples per iteration (SAGE only)

You can compute `sigma1` and `sigma2` from these arrays. The helper `results_parse.m` provides a Matlab parsing example.
