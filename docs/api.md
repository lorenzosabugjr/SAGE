# API Reference

## Estimators (`estimators/`)

### `SAGE`

Set-based Adaptive Gradient Estimator. It is stateful and may perform extra function
evaluations to refine its gradient set.

```python
class SAGE(BaseGradientEstimator):
    def __init__(
        self,
        fun: Callable[[np.ndarray], float],
        dim: int,
        quickmode: bool = True,
        initial_history: Optional[tuple[np.ndarray, np.ndarray]] = None,
        history: Optional[HistoryBuffer] = None,
        diam_mode: Optional[str] = None,
        callback: Optional[Callable[[], None]] = None,
        init_step: float = 1e-6,
    )
```

**Parameters:**
*   `fun`: The black-box objective function `f: R^n -> R`.
*   `dim`: Dimensionality of the problem.
*   `quickmode`: If `True`, uses a filtered subset of samples for faster LP solving.
*   `initial_history`: Optional tuple `(X, Z)` to seed the history, where `X` is `(N, dim)` and `Z` is `(N,)`.
*   `history`: Optional shared `HistoryBuffer` used to collect all evaluations (e.g., from line search).
*   `diam_mode`: `"exact"` or `"approx"`. Defaults to `"approx"` when `quickmode=True`.
*   `callback`: Optional callback invoked after each auxiliary evaluation.
*   `init_step`: Step size for the initial simplex when history has 0 or 1 samples.

SAGE estimates the noise bound internally.
If history has 0 or 1 samples on the first call, SAGE evaluates `x0` and `x0 + init_step * e_i` to seed the history.

---

### `FFDEstimator` / `CFDEstimator`

Standard Finite Difference Estimators (Forward and Central).

```python
class FFDEstimator(BaseGradientEstimator):
    def __init__(self, fun, dim, step=1e-6, history=None)

class CFDEstimator(BaseGradientEstimator):
    def __init__(self, fun, dim, step=1e-6, history=None)
```

**Parameters:**
*   `step`: The finite difference step size $h$.
*   `history`: Optional shared `HistoryBuffer` to record evaluations.

---

### `NMXFDEstimator`

Normalized Mixed Finite Differences (NMXFD) baseline used in the paper. The current
implementation uses a numerical integration of a Gaussian derivative to mix step sizes.

```python
class NMXFDEstimator(BaseGradientEstimator):
    def __init__(self, fun, dim, rangeintegral=(-2, 2), numpoints=4, sigma=1e-2, history=None)
```

**Parameters:**
*   `rangeintegral`: Tuple `(min, max)` for the integration range (in sigma units).
*   `numpoints`: Number of integration points.
*   `sigma`: Width of the Gaussian kernel.
*   `history`: Optional shared `HistoryBuffer` to record evaluations.

---

### `GSGEstimator` / `cGSGEstimator`

Randomized Gaussian Smoothing Estimators.

```python
class GSGEstimator(BaseGradientEstimator):
    def __init__(self, fun, dim, m, u=1e-6, seed=None, history=None)

class cGSGEstimator(BaseGradientEstimator):
    def __init__(self, fun, dim, m, u=1e-6, seed=None, history=None)
```

**Parameters:**
*   `m`: Number of random directions to sample.
*   `u`: Smoothing radius.
*   `seed`: Random seed for reproducibility.
*   `history`: Optional shared `HistoryBuffer` to record evaluations.

---

## Optimizers (`optimizers/`)

### `GradientDescent`

Plain gradient descent with an adaptive Armijo line search. Delegates
gradient estimation to a `BaseGradientEstimator`; the search direction is
always `p = -g` (no BFGS/quasi-Newton support).

```python
class GradientDescent:
    def __init__(
        self,
        fun: Callable[[np.ndarray], float],
        x0: np.ndarray,
        grad_estimator: BaseGradientEstimator,
        stepsize: float = 1.0,
        stepsizemode: StepSizeMode = StepSizeMode.ADAPTIVE,
        armijo_beta: float = 0.5,
        armijo_c: float = 1e-6,
        min_stepsize: float = 1e-6,
        max_line_search_iters: int = 100,
        recompute_grad_every_ls_failures: int = 5,
        reset_stepsize_at_floor: bool = True,
        z0: Optional[float] = None,
        callback: Optional[Callable[[float], None]] = None,
        verbose: bool = False,
    )

    def step(self):
        """Perform a single optimization step (one line-search pass)."""

    def run(self, max_evals: int):
        """Run until the estimator's shared history reaches max_evals evaluations."""
```

**Parameters:**
*   `fun`: The (possibly noisy) black-box objective `f: R^n -> R`.
*   `x0`: Initial point.
*   `grad_estimator`: Any `BaseGradientEstimator` (e.g. `SAGE`, `FFDEstimator`).
*   `stepsize`: Initial/fixed step size `eta0`.
*   `stepsizemode`: `StepSizeMode.ADAPTIVE` (Armijo line search) or `StepSizeMode.FIXED`.
*   `armijo_beta`: Shrink/grow factor for `eta` on rejection/acceptance.
*   `armijo_c`: Armijo sufficient-decrease constant, applied to `||g||^2`.
*   `min_stepsize`: Step-size floor; triggers a gradient refresh and (if
    `reset_stepsize_at_floor`) resets `eta` back to `eta0`.
*   `max_line_search_iters`: Safety cap; if exceeded, `step()` returns
    without accepting a step or incrementing the iteration counter.
*   `recompute_grad_every_ls_failures`: Refresh the gradient estimate every
    this many consecutive line-search rejections.
*   `reset_stepsize_at_floor`: Whether to reset `eta` to `eta0` when it
    hits `min_stepsize`.
*   `z0`: Optional known objective value at `x0` (skips one evaluation).
*   `callback`: Optional callback invoked with `z_k` after each accepted or
    rejected line-search trial.
*   `verbose`: Print per-step line-search diagnostics.

Rejected line-search trial points are passed to `grad_estimator` via a
lightweight `update(..., lightweight=True)` call, so their function
evaluations still inform stateful estimators like SAGE. There is no
`StandardDescent` compatibility alias; only plain gradient descent is
provided.

### `StepSizeMode`

```python
class StepSizeMode(Enum):
    FIXED = 0
    ADAPTIVE = 1
```

---

## Utilities (`utils/`)

### `NoiseType`

Enum for defining the noise model.

*   `NoiseType.UNIFORM`: Assumes bounded noise within `[-delta/2, delta/2]`.
*   `NoiseType.GAUSSIAN`: Gaussian noise with standard deviation `param`.

### `HistoryBuffer`

Tracks evaluation history `(x, z)` pairs. You can share the same history between an objective function and a gradient estimator.
