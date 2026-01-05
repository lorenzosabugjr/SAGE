# Quickstart Guide

This guide demonstrates how to integrate SAGE into your optimization pipelines. SAGE is stateful and may perform extra evaluations during gradient estimation.

## Basic Integration

If you have a noisy black-box function `my_noisy_func(x)`, you can replace your gradient estimator with SAGE directly.
Keep in mind that SAGE may call `fun` multiple times per gradient estimate.
SAGE estimates the noise bound internally. If the history has 0 or 1 samples on the first call, SAGE seeds forward-coordinate points around the first query using `init_step` (default `1e-6`).

```python
import numpy as np
from estimators import SAGE

# 1. Define your black-box function (e.g., a simulation or experiment)
def my_noisy_func(x):
    return np.sum(x**2) + np.random.normal(0, 0.01)

# 2. Initialize SAGE
#    - dim: Dimension of x
#    - quickmode: use a filtered subset of samples
#    - init_step: step size for the auto-initial simplex when history has 0 or 1 samples
grad_estimator = SAGE(
    fun=my_noisy_func, 
    dim=5, 
    quickmode=True,
    init_step=1e-6,
    # diam_mode defaults to "approx" when quickmode=True
)

# 3. Compute the gradient
x = np.random.rand(5)
gradient = grad_estimator(x)

print("Estimated Gradient:", gradient)
```

## Seeding with Existing Data

SAGE can reuse existing evaluations via `initial_history`. This is useful if you already have a dataset or if you want to warm-start with a simplex around `x0`. When `initial_history` contains two or more samples, SAGE skips the auto-initial simplex; if it contains only one sample, SAGE will add forward-coordinate points.

```python
import numpy as np
from estimators import SAGE

x0 = np.random.rand(5)
X_init = np.vstack([x0, x0 + np.eye(5)])
Z_init = np.array([my_noisy_func(x) for x in X_init])

grad_estimator = SAGE(
    fun=my_noisy_func,
    dim=5,
    initial_history=(X_init, Z_init),
)
```

## Integration with Scipy

SAGE behaves like a standard callable, making it compatible with `scipy.optimize.minimize`.
Since SAGE may perform extra evaluations, keep your own evaluation budget if you need strict limits.
SAGE returns a gradient vector for scalar objectives (the `jac` callable in `minimize`), not a full Jacobian for vector-valued objectives.

If you want SAGE to reuse *all* evaluations (including line search steps), share a `HistoryBuffer`
between the objective and the estimator.

```python
from scipy.optimize import minimize
from utils.history import HistoryBuffer

history = HistoryBuffer()

def fun_logged(x):
    z = my_noisy_func(x)
    history.add(x, z)
    return z

grad_estimator = SAGE(
    fun=fun_logged,
    dim=5,
    history=history,
)

# Run BFGS using SAGE for gradients
res = minimize(
    fun=fun_logged,
    x0=np.random.rand(5),
    jac=grad_estimator,  # Gradient callable (vector) for scalar objective
    method='BFGS'
)

print("Optimized Result:", res.x)
```

## Using the StandardDescent Optimizer

For explicit control over the optimization loop, SAGE includes a `StandardDescent` optimizer tailored for robust gradient descent.

```python
from optimizers import StandardDescent, StepSizeMode

opt = StandardDescent(
    fun=my_noisy_func,
    x0=np.random.rand(5),
    grad_estimator=grad_estimator,
    stepsize=1.0,
    stepsizemode=StepSizeMode.ADAPTIVE,
    bfgs=True
)

for i in range(100):
    opt.step()
    print(f"Iter {i}: Value = {opt.z_k:.6f}")
```
