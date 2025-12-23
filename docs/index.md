# SAGE Documentation

Welcome to the documentation for **SAGE** (Set-based Adaptive Gradient Estimator).

SAGE is a Python library implementing a robust, data-efficient gradient estimation algorithm for noisy black-box functions, based on Set Membership Identification theory. It is also available as a SciPy-compatible gradient callable; see the "Integration with Scipy" section in the Quickstart.

## Contents

*   **[Getting Started](quickstart.md)**
    *   Integration guides for `scipy.optimize` and custom optimization loops.
    *   Configuration of noise parameters.

*   **[Mathematical Theory](theory.md)**
    *   Formal derivation of the Set Membership Identification framework.
    *   Consistency set construction and Linear Programming formulation.
    *   Active sampling strategy for uncertainty reduction.

*   **[Implementation Notes](implementation.md)**
    *   Mapping from theory to the current codebase.
    *   LP construction, diameter approximation, and sampling loop details.

*   **[Benchmarks](benchmarks.md)**
    *   Problem definitions, settings, and how to run experiments.

*   **[API Reference](api.md)**
    *   **Estimators**: Detailed specifications for `SAGE`, `FFDEstimator`, `GSGEstimator`.
    *   **Optimizers**: Documentation for `StandardDescent`.
    *   **Utilities**: Configuration of noise models.

## Installation

SAGE requires **Python 3.9+**.

```bash
git clone https://github.com/lorenzosabugjr/SAGE.git
cd SAGE
pip install -r requirements.txt
```

## Contributing

We welcome contributions! Please submit Pull Requests to the main repository.
