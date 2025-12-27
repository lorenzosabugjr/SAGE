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
        noise_param: float = 0.0,
        autonoise: bool = True,
        quickmode: bool = True,
        initial_history: Optional[tuple[np.ndarray, np.ndarray]] = None,
        history: Optional[HistoryBuffer] = None,
        diam_mode: Optional[str] = None,
        callback: Optional[Callable[[], None]] = None,
        init_step: float = 1e-3,
    )
```

**Parameters:**
*   `fun`: The black-box objective function `f: R^n -> R`.
*   `dim`: Dimensionality of the problem.
*   `noise_param`: Noise bound used only when `autonoise=False` (or an initial value otherwise).
*   `autonoise`: If `True`, estimates the noise bound inside the LP (adds an extra decision variable).
*   `quickmode`: If `True`, uses a filtered subset of samples for faster LP solving.
*   `initial_history`: Optional tuple `(X, Z)` to seed the history, where `X` is `(N, dim)` and `Z` is `(N,)`.
*   `history`: Optional shared `HistoryBuffer` used to collect all evaluations (e.g., from line search).
*   `diam_mode`: `"exact"` or `"approx"`. Defaults to `"approx"` when `quickmode=True`.
*   `callback`: Optional callback invoked after each auxiliary evaluation.
*   `init_step`: Step size for the initial simplex when history is empty.

If history is empty on the first call, SAGE evaluates `x0` and `x0 + init_step * e_i` to seed the history.

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

### `StandardDescent`

A flexible gradient descent optimizer supporting BFGS and Adaptive Line Search.

```python
class StandardDescent:
    def __init__(
        self,
        fun: Callable,
        x0: np.ndarray,
        grad_estimator: BaseGradientEstimator,
        stepsize: float = 1.0,
        stepsizemode: StepSizeMode = StepSizeMode.ADAPTIVE,
        bfgs: bool = False,
        z0: Optional[float] = None,
        callback: Optional[Callable[[float], None]] = None,
    )
```

**Parameters:**
*   `grad_estimator`: An instance of a SAGE or FD estimator.
*   `bfgs`: If `True`, approximates the Hessian inverse using BFGS updates.
*   `stepsizemode`: `StepSizeMode.FIXED` or `StepSizeMode.ADAPTIVE` (Armijo backtracking).
*   `z0`: Optional initial function value at `x0` (avoids re-evaluation).
*   `callback`: Optional callback invoked with each recorded evaluation value.

---

## Utilities (`utils/`)

### `NoiseType`

Enum for defining the noise model.

*   `NoiseType.UNIFORM`: Assumes bounded noise within `[-delta/2, delta/2]`.
*   `NoiseType.GAUSSIAN`: Gaussian noise with standard deviation `param`.

### `HistoryBuffer`

Tracks evaluation history `(x, z)` pairs. You can share the same history between an optimizer and a gradient estimator.
