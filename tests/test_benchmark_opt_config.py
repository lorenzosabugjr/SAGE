"""
Unit tests for optimization benchmark config loading.

Run with: python -m unittest tests.test_benchmark_opt_config
"""

import os
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from tests.benchmark_opt import load_config

REQUIRED_LINES = [
    "list_dims: [2]",
    "list_condnum: [1.0]",
    "list_noise_param: [0.0]",
    "list_noise_type: ['uniform']",
    "list_problem: ['least-squares']",
    "list_grad_est: ['truth']",
    "bmk_maxtrials: 2",
    "max_evals_mult: 20",
]


def _write_config(tmp: Path, extra_lines=()) -> Path:
    config_path = tmp / "config.yaml"
    config_path.write_text("\n".join(list(REQUIRED_LINES) + list(extra_lines)), encoding="utf-8")
    return config_path


class LoadConfigTests(unittest.TestCase):
    def test_coerces_yaml_exponent_strings_for_numeric_fields(self):
        with tempfile.TemporaryDirectory() as tmp:
            config_path = _write_config(
                Path(tmp),
                extra_lines=[
                    "list_dims: ['20']",
                    "list_condnum: [1.0, 1.0e4, '1.0e8']",
                    "gdtcalcstep: '1.0e-6'",
                    "stepsize: '2.0'",
                    "armijo_beta: '0.25'",
                    "armijo_c: '1e-6'",
                    "min_stepsize: '1e-8'",
                    "max_line_search_iters: '50'",
                    "recompute_grad_every_ls_failures: '3'",
                    "target_ratios: ['0.1', 0.01]",
                ],
            )

            cfg = load_config(str(config_path))

            self.assertEqual(cfg["list_dims"], [20])
            self.assertEqual(cfg["list_condnum"], [1.0, 1.0e4, 1.0e8])
            self.assertEqual(cfg["gdtcalcstep"], 1e-6)
            self.assertEqual(cfg["stepsize"], 2.0)
            self.assertEqual(cfg["armijo_beta"], 0.25)
            self.assertEqual(cfg["armijo_c"], 1e-6)
            self.assertEqual(cfg["min_stepsize"], 1e-8)
            self.assertEqual(cfg["max_line_search_iters"], 50)
            self.assertEqual(cfg["recompute_grad_every_ls_failures"], 3)
            self.assertEqual(cfg["target_ratios"], [0.1, 0.01])

    def test_mother_defaults_applied_when_omitted(self):
        with tempfile.TemporaryDirectory() as tmp:
            config_path = _write_config(Path(tmp))
            cfg = load_config(str(config_path))

            self.assertEqual(cfg["opt_bmk_dtype"], "float128")
            self.assertEqual(cfg["gdtcalcstep"], 1e-6)
            self.assertEqual(cfg["stepsize_mode"], "adaptive")
            self.assertEqual(cfg["stepsize"], 1.0)
            self.assertEqual(cfg["armijo_beta"], 0.5)
            self.assertEqual(cfg["armijo_c"], 1e-6)
            self.assertEqual(cfg["min_stepsize"], 1e-6)
            self.assertEqual(cfg["max_line_search_iters"], 100)
            self.assertEqual(cfg["recompute_grad_every_ls_failures"], 5)
            self.assertIs(cfg["reset_stepsize_at_floor"], True)
            self.assertIs(cfg["sage_reset_on_step"], False)
            self.assertIs(cfg["verbose"], False)
            self.assertIs(cfg["save_eval_history"], False)
            self.assertEqual(cfg["target_ratios"], [0.1, 0.01, 0.001])

    def test_booleans_rejected_for_numeric_fields(self):
        with tempfile.TemporaryDirectory() as tmp:
            for key in ("gdtcalcstep", "stepsize", "armijo_beta", "armijo_c", "min_stepsize"):
                config_path = _write_config(Path(tmp), extra_lines=[f"{key}: true"])
                with self.assertRaises(ValueError, msg=f"{key} should reject bool"):
                    load_config(str(config_path))

    def test_numbers_rejected_for_boolean_fields(self):
        with tempfile.TemporaryDirectory() as tmp:
            for key in ("reset_stepsize_at_floor", "sage_reset_on_step", "verbose", "save_eval_history"):
                config_path = _write_config(Path(tmp), extra_lines=[f"{key}: 1"])
                with self.assertRaises(ValueError, msg=f"{key} should reject non-bool"):
                    load_config(str(config_path))

    def test_invalid_stepsize_mode_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            config_path = _write_config(Path(tmp), extra_lines=["stepsize_mode: 'bogus'"])
            with self.assertRaises(ValueError):
                load_config(str(config_path))

    def test_missing_required_field_fails_before_a_run(self):
        with tempfile.TemporaryDirectory() as tmp:
            config_path = Path(tmp) / "config.yaml"
            config_path.write_text("list_dims: [2]\n", encoding="utf-8")
            with self.assertRaises(ValueError):
                load_config(str(config_path))

    def test_non_mapping_config_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            config_path = Path(tmp) / "config.yaml"
            config_path.write_text("- 1\n- 2\n", encoding="utf-8")
            with self.assertRaises(ValueError):
                load_config(str(config_path))


if __name__ == "__main__":
    unittest.main()
