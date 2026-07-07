"""
Unit tests for benchmark gradient config loading.

Run with: python -m unittest tests.test_benchmark_grad_config
"""

import os
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from tests.benchmark_grad import load_config


class LoadConfigTests(unittest.TestCase):
    def test_coerces_yaml_exponent_strings_for_numeric_fields(self):
        with tempfile.TemporaryDirectory() as tmp:
            config_path = Path(tmp) / "config.yaml"
            config_path.write_text(
                "\n".join(
                    [
                        "list_dims: ['20']",
                        "list_condnum: [1.0, 1.0e4, '1.0e8']",
                        "list_noise_param: [0.0, 1.0e-3, '2e-2']",
                        "list_noise_type: ['uniform']",
                        "list_problem: ['least-squares']",
                        "grad_bmk_npoints: '3'",
                        "grad_bmk_nproblems: 2.0",
                        "grad_bmk_estimators: ['ffd']",
                    ]
                ),
                encoding="utf-8",
            )

            cfg = load_config(str(config_path))

            self.assertEqual(cfg["list_dims"], [20])
            self.assertEqual(cfg["list_condnum"], [1.0, 1.0e4, 1.0e8])
            self.assertEqual(cfg["list_noise_param"], [0.0, 1.0e-3, 2e-2])
            self.assertEqual(cfg["grad_bmk_npoints"], 3)
            self.assertEqual(cfg["grad_bmk_nproblems"], 2)


if __name__ == "__main__":
    unittest.main()
