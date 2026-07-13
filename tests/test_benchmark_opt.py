"""
Unit tests for tests/benchmark_opt.py.

Run with: python -m unittest tests.test_benchmark_opt
"""

import os
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import yaml

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from tests.benchmark_opt import _run_optimization_benchmark

REQUIRED_FIELDS = [
    "res_hist_true",
    "res_hist_eval",
    "time_hist",
    "Z_initial_true_vec",
    "Z_initial_eval_vec",
    "Z_start_true_vec",
    "Z_start_eval_vec",
    "Z0_true_vec",
    "Z0_eval_vec",
    "final_true",
    "final_eval",
    "last_hist_true",
    "last_hist_eval",
    "n_evals",
    "target_ratios",
    "evals_to_target",
    "success_by_target",
    "trial_status",
    "trial_error",
]

SAGE_ONLY_FIELDS = [
    "auxs_hist",
    "sage_diag_n_calls",
    "sage_diag_eval_index",
    "sage_diag_hist_size",
    "sage_diag_n_neighbors",
    "sage_diag_n_aux",
    "sage_diag_calibration_attempted",
    "sage_diag_calibration_fixed",
    "sage_diag_stop_reason",
]

METADATA_FIELDS = [
    "config_path",
    "output_dir",
    "run_timestamp",
    "git_commit",
    "max_evals",
    "problem",
    "dim",
    "condnum",
    "estimator",
    "noise_type",
    "noise_param",
    "stepsize_mode",
    "stepsize",
    "armijo_beta",
    "armijo_c",
    "min_stepsize",
    "max_line_search_iters",
    "recompute_grad_every_ls_failures",
    "reset_stepsize_at_floor",
    "sage_reset_on_step",
    "opt_bmk_dtype",
    "gdtcalcstep",
]


def _write_yaml(path: Path, cfg: dict):
    path.write_text(yaml.dump(cfg), encoding="utf-8")


def _run(tmp: Path, cfg: dict):
    config_path = tmp / "config.yaml"
    _write_yaml(config_path, cfg)
    out_dir = tmp / "out"
    out_dir.mkdir()
    metadata = {
        "config_path": str(config_path),
        "output_dir": str(out_dir),
        "run_timestamp": "2026-07-09 00-00-00",
        "git_commit": "deadbeef",
    }
    _run_optimization_benchmark(str(config_path), out_dir, metadata)
    return out_dir


class RunOptimizationBenchmarkTests(unittest.TestCase):
    def test_saves_mat_with_required_fields_shapes_and_metadata(self):
        from scipy.io import loadmat

        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            n_trials = 3
            max_evals_mult = 10
            dims = 2
            target_ratios = [0.5, 0.1]
            out_dir = _run(
                tmp,
                {
                    "list_dims": [dims],
                    "list_condnum": [1.0],
                    "list_noise_param": [0.0],
                    "list_noise_type": ["uniform"],
                    "list_problem": ["least-squares"],
                    "list_grad_est": ["truth"],
                    "bmk_maxtrials": n_trials,
                    "max_evals_mult": max_evals_mult,
                    "target_ratios": target_ratios,
                },
            )

            files = list(out_dir.glob("opt-bmk-*.mat"))
            self.assertEqual(len(files), 1)
            self.assertEqual(
                files[0].name,
                f"opt-bmk-{dims}D-least-squares-1.0-truth-uniform-0.000000.mat",
            )

            mat = loadmat(str(files[0]))

            for field in REQUIRED_FIELDS:
                self.assertIn(field, mat, msg=f"missing field {field}")
            for field in METADATA_FIELDS:
                self.assertIn(field, mat, msg=f"missing metadata field {field}")

            max_evals = max_evals_mult * dims
            self.assertEqual(mat["res_hist_true"].shape, (max_evals, n_trials))
            self.assertEqual(mat["res_hist_eval"].shape, (max_evals, n_trials))
            self.assertEqual(mat["time_hist"].shape, (max_evals, n_trials))
            self.assertEqual(mat["evals_to_target"].shape, (len(target_ratios), n_trials))
            self.assertEqual(mat["success_by_target"].shape, (len(target_ratios), n_trials))

            # SAGE-only fields must not be present for a non-SAGE estimator.
            for field in SAGE_ONLY_FIELDS:
                self.assertNotIn(field, mat)

            # All trials succeed for "truth" on a tiny least-squares problem; every
            # trial column should have exactly max_evals real (non-NaN) rows.
            self.assertTrue(np.all(~np.isnan(mat["res_hist_true"])))
            self.assertTrue(np.all(~np.isnan(mat["res_hist_eval"])))
            self.assertEqual(int(mat["max_evals"].squeeze()), max_evals)

    def test_sage_estimator_saves_auxs_hist(self):
        from scipy.io import loadmat

        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            out_dir = _run(
                tmp,
                {
                    "list_dims": [2],
                    "list_condnum": [1.0],
                    "list_noise_param": [0.0],
                    "list_noise_type": ["uniform"],
                    "list_problem": ["least-squares"],
                    "list_grad_est": ["sage"],
                    "bmk_maxtrials": 2,
                    "max_evals_mult": 15,
                },
            )
            files = list(out_dir.glob("opt-bmk-*.mat"))
            mat = loadmat(str(files[0]))
            self.assertIn("auxs_hist", mat)
            self.assertEqual(mat["auxs_hist"].shape, (1, 2))

    def test_sage_estimator_saves_call_diagnostics(self):
        """Milestone 6 gate check: SAGE per-call diagnostic arrays are
        present, aligned to (max_evals, n_trials), and every valid row
        (below sage_diag_n_calls) carries a real eval_index/hist_size and a
        known stopping-reason code."""
        from scipy.io import loadmat

        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            dims = 2
            max_evals_mult = 15
            n_trials = 3
            out_dir = _run(
                tmp,
                {
                    "list_dims": [dims],
                    "list_condnum": [1.0],
                    "list_noise_param": [0.0],
                    "list_noise_type": ["uniform"],
                    "list_problem": ["least-squares"],
                    "list_grad_est": ["sage"],
                    "bmk_maxtrials": n_trials,
                    "max_evals_mult": max_evals_mult,
                },
            )
            files = list(out_dir.glob("opt-bmk-*.mat"))
            self.assertEqual(len(files), 1)
            mat = loadmat(str(files[0]))

            max_evals = max_evals_mult * dims
            for field in SAGE_ONLY_FIELDS:
                self.assertIn(field, mat, msg=f"missing field {field}")
            for field in [
                "sage_diag_eval_index", "sage_diag_hist_size",
                "sage_diag_n_neighbors", "sage_diag_n_aux",
                "sage_diag_calibration_attempted", "sage_diag_calibration_fixed",
                "sage_diag_stop_reason",
            ]:
                self.assertEqual(mat[field].shape, (max_evals, n_trials), msg=field)

            n_calls = np.asarray(mat["sage_diag_n_calls"]).reshape(-1)
            self.assertEqual(n_calls.shape, (n_trials,))
            self.assertTrue(np.all(n_calls >= 1))

            known_codes = {
                "relative_criterion", "noiseless_floor", "forced_stop",
                "auxiliary_cap", "no_aux_direction", "budget_exhaustion",
                "stale_estimate",
            }
            for trial_i in range(n_trials):
                n = int(n_calls[trial_i])
                eval_idx_col = mat["sage_diag_eval_index"][:, trial_i]
                hist_size_col = mat["sage_diag_hist_size"][:, trial_i]
                stop_reason_col = mat["sage_diag_stop_reason"][:, trial_i]

                # Every one of this trial's real calls has a non-NaN
                # call-start eval index/history size...
                self.assertTrue(np.all(~np.isnan(eval_idx_col[:n])))
                self.assertTrue(np.all(~np.isnan(hist_size_col[:n])))
                # ...and rows beyond n_calls stay unpopulated (NaN-padded).
                if n < max_evals:
                    self.assertTrue(np.all(np.isnan(eval_idx_col[n:])))

                for ci in range(n):
                    cell = stop_reason_col[ci]
                    # loadmat unwraps object-array cells into a 1-element
                    # array/string depending on scipy version; normalize both.
                    code = str(cell[0]) if hasattr(cell, "__len__") and not isinstance(cell, str) else str(cell)
                    self.assertIn(code, known_codes)

    def test_sage_alongside_other_estimators_produces_one_file_each(self):
        """Milestone 6 gate check: a benchmark containing every existing
        estimator plus SAGE produces one output file per estimator, with
        unchanged target-ratio computation, a centered SAGE start, and SAGE
        diagnostic arrays present only in the SAGE artifact."""
        from scipy.io import loadmat

        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            dims = 2
            estimators = ["ffd", "cfd", "gsg", "cgsg", "nmxfd", "sage"]
            target_ratios = [0.5]
            out_dir = _run(
                tmp,
                {
                    "list_dims": [dims],
                    "list_condnum": [1.0],
                    "list_noise_param": [0.0],
                    "list_noise_type": ["uniform"],
                    "list_problem": ["least-squares"],
                    "list_grad_est": estimators,
                    "bmk_maxtrials": 2,
                    "max_evals_mult": 15,
                    "target_ratios": target_ratios,
                },
            )

            files = {p.name: p for p in out_dir.glob("opt-bmk-*.mat")}
            self.assertEqual(len(files), len(estimators))

            for est in estimators:
                fname = f"opt-bmk-{dims}D-least-squares-1.0-{est}-uniform-0.000000.mat"
                self.assertIn(fname, files)
                mat = loadmat(str(files[fname]))

                for field in REQUIRED_FIELDS:
                    self.assertIn(field, mat, msg=f"{est}: missing {field}")
                self.assertEqual(mat["evals_to_target"].shape, (len(target_ratios), 2))
                self.assertEqual(mat["success_by_target"].shape, (len(target_ratios), 2))

                if est == "sage":
                    for field in SAGE_ONLY_FIELDS:
                        self.assertIn(field, mat, msg=f"sage: missing {field}")
                    # Centered start: SAGE no longer reassigns X_start to the
                    # best point in its initialized history.
                    np.testing.assert_array_equal(
                        mat["Z_start_eval_vec"], mat["Z_initial_eval_vec"]
                    )
                    np.testing.assert_array_equal(
                        mat["Z_start_true_vec"], mat["Z_initial_true_vec"]
                    )
                else:
                    for field in SAGE_ONLY_FIELDS:
                        self.assertNotIn(field, mat, msg=f"{est}: unexpected {field}")

    def test_failed_trial_is_represented_and_serializable(self):
        from scipy.io import loadmat

        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            n_trials = 2
            max_evals_mult = 10
            dims = 2
            out_dir = _run(
                tmp,
                {
                    "list_dims": [dims],
                    "list_condnum": [1.0],
                    "list_noise_param": [0.0],
                    "list_noise_type": ["uniform"],
                    "list_problem": ["not-a-real-problem"],
                    "list_grad_est": ["truth"],
                    "bmk_maxtrials": n_trials,
                    "max_evals_mult": max_evals_mult,
                },
            )
            files = list(out_dir.glob("opt-bmk-*.mat"))
            self.assertEqual(len(files), 1)
            mat = loadmat(str(files[0]))

            statuses = [str(s[0]) for s in mat["trial_status"].reshape(-1)]
            errors = [str(s[0]) for s in mat["trial_error"].reshape(-1)]
            self.assertEqual(statuses, ["error"] * n_trials)
            self.assertTrue(all("Unknown problem" in e for e in errors))

            # Failed trials stay unsolved: no successes and fully padded/NaN history.
            self.assertFalse(np.any(mat["success_by_target"]))
            self.assertTrue(np.all(np.isnan(mat["res_hist_true"])))
            self.assertTrue(np.all(np.isnan(mat["evals_to_target"])))

    def test_final_true_matches_last_hist_true(self):
        from scipy.io import loadmat

        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            out_dir = _run(
                tmp,
                {
                    "list_dims": [2],
                    "list_condnum": [1.0],
                    "list_noise_param": [0.0],
                    "list_noise_type": ["uniform"],
                    "list_problem": ["least-squares"],
                    "list_grad_est": ["truth"],
                    "bmk_maxtrials": 2,
                    "max_evals_mult": 10,
                },
            )
            files = list(out_dir.glob("opt-bmk-*.mat"))
            mat = loadmat(str(files[0]))
            np.testing.assert_array_equal(mat["final_true"], mat["last_hist_true"])
            np.testing.assert_array_equal(mat["final_eval"], mat["last_hist_eval"])

    def test_multiple_combos_produce_multiple_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            out_dir = _run(
                tmp,
                {
                    "list_dims": [2],
                    "list_condnum": [1.0],
                    "list_noise_param": [0.0],
                    "list_noise_type": ["uniform"],
                    "list_problem": ["least-squares", "lasso"],
                    "list_grad_est": ["truth"],
                    "bmk_maxtrials": 1,
                    "max_evals_mult": 10,
                },
            )
            files = sorted(p.name for p in out_dir.glob("opt-bmk-*.mat"))
            self.assertEqual(
                files,
                [
                    "opt-bmk-2D-lasso-1.0-truth-uniform-0.000000.mat",
                    "opt-bmk-2D-least-squares-1.0-truth-uniform-0.000000.mat",
                ],
            )


if __name__ == "__main__":
    unittest.main()
