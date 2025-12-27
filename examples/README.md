# Examples

These scripts are small, runnable demos. Install deps with `pip install -r requirements.txt`.
You can run them from the repo root with `python examples/<script>.py`.
If you do not pass `initial_history`, SAGE auto-seeds a simplex around the first query using `init_step`.

- `scipy_minimize_bfgs.py`: Use SAGE as the `jac` callable in `scipy.optimize.minimize`.
- `scipy_compare_estimators.py`: Compare SAGE against finite-difference estimators in BFGS.
