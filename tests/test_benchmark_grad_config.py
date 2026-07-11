"""
Unit tests for benchmark gradient config loading.

Run with: python -m unittest tests.test_benchmark_grad_config
"""

import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from tests.benchmark_grad import load_config, _run_gradient_benchmark


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


def _write_minimal_config(path: Path, extra_lines=()) -> None:
    lines = [
        "list_dims: [2]",
        "list_condnum: [1.0]",
        "list_noise_param: [0.05]",
        "list_noise_type: ['uniform']",
        "list_problem: ['least-squares']",
        "grad_bmk_npoints: 1",
        "grad_bmk_nproblems: 1",
        "grad_bmk_estimators: ['sage']",
    ]
    lines.extend(extra_lines)
    path.write_text("\n".join(lines), encoding="utf-8")


class SageNoiseBoundModeConfigTests(unittest.TestCase):
    def test_defaults_to_estimate_when_absent(self):
        with tempfile.TemporaryDirectory() as tmp:
            config_path = Path(tmp) / "config.yaml"
            _write_minimal_config(config_path)

            cfg = load_config(str(config_path))

            self.assertEqual(cfg["sage_noise_bound_mode"], "estimate")

    def test_accepts_known_mode(self):
        with tempfile.TemporaryDirectory() as tmp:
            config_path = Path(tmp) / "config.yaml"
            _write_minimal_config(config_path, ["sage_noise_bound_mode: 'known'"])

            cfg = load_config(str(config_path))

            self.assertEqual(cfg["sage_noise_bound_mode"], "known")

    def test_rejects_invalid_mode(self):
        with tempfile.TemporaryDirectory() as tmp:
            config_path = Path(tmp) / "config.yaml"
            _write_minimal_config(config_path, ["sage_noise_bound_mode: 'bogus'"])

            with self.assertRaises(ValueError):
                load_config(str(config_path))


class SageKnownModeFactoryForwardingTests(unittest.TestCase):
    """Milestone 4/5: known mode results in SAGE receiving a calibrated
    noise_bound (bmk_noise / 2.0 for uniform noise, since utils/noise.py
    draws uniform noise from [-bmk_noise/2, bmk_noise/2] and SAGE wants the
    true bound, not the full interval width)."""

    def test_known_mode_passes_half_bmk_noise_as_noise_bound_for_uniform(self):
        import numpy as np

        captured = {}

        class FakeEstimator:
            def __call__(self, x):
                return np.zeros_like(np.asarray(x, dtype=float))

        def fake_create_estimator(name, obj_func, dims, history, **kwargs):
            if name == "sage":
                captured["noise_bound"] = kwargs.get("noise_bound")
            return FakeEstimator()

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            config_path = tmp_path / "config.yaml"
            _write_minimal_config(
                config_path,
                ["sage_noise_bound_mode: 'known'"],
            )
            output_dir = tmp_path / "out"
            output_dir.mkdir()

            with patch("tests.factories.create_estimator", side_effect=fake_create_estimator):
                _run_gradient_benchmark(str(config_path), output_dir, metadata={})

        self.assertEqual(captured["noise_bound"], 0.025)

    def test_known_mode_with_gaussian_noise_warns_and_omits_noise_bound(self):
        import numpy as np

        captured = {}

        class FakeEstimator:
            def __call__(self, x):
                return np.zeros_like(np.asarray(x, dtype=float))

        def fake_create_estimator(name, obj_func, dims, history, **kwargs):
            if name == "sage":
                captured["called"] = True
                captured["has_noise_bound"] = "noise_bound" in kwargs
            return FakeEstimator()

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            config_path = tmp_path / "config.yaml"
            _write_minimal_config(
                config_path,
                [
                    "sage_noise_bound_mode: 'known'",
                    "list_noise_type: ['gaussian']",
                ],
            )
            output_dir = tmp_path / "out"
            output_dir.mkdir()

            with patch("tests.factories.create_estimator", side_effect=fake_create_estimator):
                with self.assertWarns(UserWarning):
                    _run_gradient_benchmark(str(config_path), output_dir, metadata={})

        self.assertTrue(captured.get("called"))
        self.assertFalse(captured.get("has_noise_bound"))

    def test_estimate_mode_does_not_pass_noise_bound(self):
        import numpy as np

        captured = {}

        class FakeEstimator:
            def __call__(self, x):
                return np.zeros_like(np.asarray(x, dtype=float))

        def fake_create_estimator(name, obj_func, dims, history, **kwargs):
            if name == "sage":
                captured["called"] = True
                captured["has_noise_bound"] = "noise_bound" in kwargs
            return FakeEstimator()

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            config_path = tmp_path / "config.yaml"
            _write_minimal_config(config_path)
            output_dir = tmp_path / "out"
            output_dir.mkdir()

            with patch("tests.factories.create_estimator", side_effect=fake_create_estimator):
                _run_gradient_benchmark(str(config_path), output_dir, metadata={})

        self.assertTrue(captured.get("called"))
        self.assertFalse(captured.get("has_noise_bound"))


if __name__ == "__main__":
    unittest.main()
