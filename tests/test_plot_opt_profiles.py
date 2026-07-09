"""
Unit tests for utils/plot_opt_profiles.py.

Run with: python -m unittest tests.test_plot_opt_profiles
"""

import io
import os
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path

import numpy as np
import yaml
from scipy.io import savemat

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from utils.plot_opt_profiles import (
    DuplicateFileError,
    MissingFileError,
    SettingsError,
    build_data_profile,
    build_summary_rows,
    combo_target_ratios,
    discover,
    load_profile_data,
    load_settings,
    parse_opt_filename,
    run,
    summarize_target,
    write_summary_csv,
)

ESTIMATORS = ["sage", "ffd", "truth"]


def _write_mat(path, target_ratios, evals_to_target, success_by_target, trial_status, max_evals):
    n_trials = len(trial_status)
    status_arr = np.empty(n_trials, dtype=object)
    for i, s in enumerate(trial_status):
        status_arr[i] = s
    savemat(str(path), {
        "target_ratios": np.asarray(target_ratios, dtype=float),
        "evals_to_target": np.asarray(evals_to_target, dtype=float),
        "success_by_target": np.asarray(success_by_target, dtype=bool),
        "trial_status": status_arr,
        "max_evals": max_evals,
    })


def _settings_dict(source_dirs, **overrides):
    cfg = {
        "source_dirs": [str(d) for d in source_dirs],
        "dims": [2],
        "problems": ["least-squares"],
        "condnums": [1.0],
        "noise_types": ["uniform"],
        "noise_params": [0.0],
        "estimators": ["sage", "ffd"],
    }
    cfg.update(overrides)
    return cfg


def _write_settings_yaml(path, cfg):
    with open(path, "w") as f:
        yaml.safe_dump(cfg, f)


# ---------------------------------------------------------------------------
# Filename parsing
# ---------------------------------------------------------------------------
class ParseOptFilenameTests(unittest.TestCase):
    def test_parses_simple_problem_name(self):
        match = parse_opt_filename(
            "opt-bmk-2D-lasso-1.0-ffd-uniform-0.000000.mat", ESTIMATORS
        )
        self.assertEqual(match["dim"], 2)
        self.assertEqual(match["problem"], "lasso")
        self.assertEqual(match["condnum"], 1.0)
        self.assertEqual(match["estimator"], "ffd")
        self.assertEqual(match["noise_type"], "uniform")
        self.assertEqual(match["noise_param"], 0.0)

    def test_parses_hyphenated_problem_name(self):
        match = parse_opt_filename(
            "opt-bmk-20D-least-squares-100000000.0-sage-uniform-1.000000.mat", ESTIMATORS
        )
        self.assertEqual(match["problem"], "least-squares")
        self.assertEqual(match["condnum"], 1.0e8)
        self.assertEqual(match["estimator"], "sage")

    def test_returns_none_for_non_matching_prefix(self):
        self.assertIsNone(parse_opt_filename("not-an-opt-file.mat", ESTIMATORS))

    def test_returns_none_when_no_estimator_anchor_found(self):
        match = parse_opt_filename(
            "opt-bmk-2D-lasso-1.0-nmxfd-uniform-0.000000.mat", ESTIMATORS
        )
        self.assertIsNone(match)


# ---------------------------------------------------------------------------
# Discovery and matching
# ---------------------------------------------------------------------------
class DiscoverAndMatchTests(unittest.TestCase):
    def _make_file(self, tmp, dim, problem, condnum, estimator, noise_type, noise_param):
        fname = f"opt-bmk-{dim}D-{problem}-{condnum}-{estimator}-{noise_type}-{noise_param:.6f}.mat"
        path = Path(tmp) / fname
        _write_mat(
            path,
            target_ratios=[0.1, 0.01],
            evals_to_target=[[3, np.nan], [5, np.nan]],
            success_by_target=[[True, False], [True, False]],
            trial_status=["ok", "ok"],
            max_evals=20,
        )
        return path

    def _write_cfg(self, tmp, cfg):
        path = Path(tmp) / "plot_settings.yaml"
        _write_settings_yaml(path, cfg)
        return path

    def test_numeric_condnum_and_noise_param_matching(self):
        with tempfile.TemporaryDirectory() as tmp:
            self._make_file(tmp, 2, "least-squares", 10000.0, "sage", "uniform", 0.001)
            self._make_file(tmp, 2, "least-squares", 10000.0, "ffd", "uniform", 0.001)

            settings = load_settings(self._write_cfg(tmp, _settings_dict(
                [tmp], condnums=[1.0e4], noise_params=[1.0e-3],
            )))
            resolved = discover(settings)
            combo = (2, "least-squares", 1.0e4, "uniform", 1.0e-3)
            self.assertEqual(set(resolved[combo].keys()), {"sage", "ffd"})

    def test_duplicate_matches_raise_with_all_paths(self):
        with tempfile.TemporaryDirectory() as tmp:
            p1 = self._make_file(tmp, 2, "lasso", 1.0, "ffd", "uniform", 0.0)
            sub = Path(tmp) / "sub"
            sub.mkdir()
            p2 = self._make_file(sub, 2, "lasso", 1.0, "ffd", "uniform", 0.0)

            settings = load_settings(self._write_cfg(tmp, _settings_dict(
                [tmp], problems=["lasso"], recursive=True, estimators=["ffd"],
            )))
            with self.assertRaises(DuplicateFileError) as ctx:
                discover(settings)
            self.assertIn(str(p1), str(ctx.exception))
            self.assertIn(str(p2), str(ctx.exception))

    def test_missing_file_raises_by_default(self):
        with tempfile.TemporaryDirectory() as tmp:
            self._make_file(tmp, 2, "lasso", 1.0, "ffd", "uniform", 0.0)
            settings = load_settings(self._write_cfg(tmp, _settings_dict(
                [tmp], problems=["lasso"],
            )))
            with self.assertRaises(MissingFileError):
                discover(settings)

    def test_missing_file_warn_skip_continues(self):
        with tempfile.TemporaryDirectory() as tmp:
            self._make_file(tmp, 2, "lasso", 1.0, "ffd", "uniform", 0.0)
            settings = load_settings(self._write_cfg(tmp, _settings_dict(
                [tmp], problems=["lasso"], missing_policy="warn_skip",
            )))
            buf = io.StringIO()
            with redirect_stdout(buf):
                resolved = discover(settings)
            combo = (2, "lasso", 1.0, "uniform", 0.0)
            self.assertEqual(set(resolved[combo].keys()), {"ffd"})
            self.assertIn("WARNING", buf.getvalue())

    def test_dry_run_prints_matches_and_writes_no_artifacts(self):
        with tempfile.TemporaryDirectory() as tmp:
            self._make_file(tmp, 2, "lasso", 1.0, "ffd", "uniform", 0.0)
            self._make_file(tmp, 2, "lasso", 1.0, "sage", "uniform", 0.0)
            output_dir = Path(tmp) / "plots"
            settings = load_settings(self._write_cfg(tmp, _settings_dict(
                [tmp], problems=["lasso"], output_dir=str(output_dir),
            )))
            buf = io.StringIO()
            with redirect_stdout(buf):
                run(settings, dry_run=True)
            self.assertIn("ffd", buf.getvalue())
            self.assertIn("sage", buf.getvalue())
            self.assertFalse(output_dir.exists())


class LoadSettingsValidationTests(unittest.TestCase):
    def test_missing_required_field_raises(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg = _settings_dict([tmp])
            del cfg["dims"]
            path = Path(tmp) / "cfg.yaml"
            _write_settings_yaml(path, cfg)
            with self.assertRaises(SettingsError):
                load_settings(path)

    def test_invalid_x_axis_raises(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg = _settings_dict([tmp], x_axis="bogus")
            path = Path(tmp) / "cfg.yaml"
            _write_settings_yaml(path, cfg)
            with self.assertRaises(SettingsError):
                load_settings(path)

    def test_bool_target_ratio_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg = _settings_dict([tmp], target_ratios=[True])
            path = Path(tmp) / "cfg.yaml"
            _write_settings_yaml(path, cfg)
            with self.assertRaises(SettingsError):
                load_settings(path)


# ---------------------------------------------------------------------------
# Data-profile computation
# ---------------------------------------------------------------------------
class DataProfileTests(unittest.TestCase):
    def test_build_data_profile_steps_and_plateaus(self):
        evals_row = np.array([2.0, 4.0, np.nan, np.nan])
        xs, ys = build_data_profile(evals_row, n_trials=4, dim=2, max_evals=20, x_axis="raw_evals")
        np.testing.assert_allclose(xs, [0.0, 2.0, 4.0, 20.0])
        np.testing.assert_allclose(ys, [0.0, 0.25, 0.5, 0.5])

    def test_build_data_profile_evals_per_dim(self):
        evals_row = np.array([4.0, np.nan])
        xs, ys = build_data_profile(evals_row, n_trials=2, dim=2, max_evals=20, x_axis="evals_per_dim")
        np.testing.assert_allclose(xs, [0.0, 2.0, 10.0])
        np.testing.assert_allclose(ys, [0.0, 0.5, 0.5])

    def test_summarize_target_counts_and_stats(self):
        evals_row = np.array([2.0, 4.0, np.nan])
        success_row = np.array([True, True, False])
        trial_status = ["ok", "ok", "error"]
        summary = summarize_target(evals_row, success_row, trial_status)
        self.assertEqual(summary["n_trials"], 3)
        self.assertEqual(summary["n_ok"], 2)
        self.assertEqual(summary["n_error"], 1)
        self.assertEqual(summary["n_solved"], 2)
        self.assertAlmostEqual(summary["success_rate"], 2 / 3)
        self.assertAlmostEqual(summary["median_evals_to_target"], 3.0)
        self.assertAlmostEqual(summary["mean_evals_to_target"], 3.0)

    def test_combo_target_ratios_uses_config_when_given(self):
        settings = {"target_ratios": [0.5, 0.25], "estimators": ["ffd"]}
        self.assertEqual(combo_target_ratios({}, settings), [0.5, 0.25])

    def test_combo_target_ratios_falls_back_to_file(self):
        loaded = {"ffd": {"target_ratios": np.array([0.1, 0.01])}}
        settings = {"target_ratios": None, "estimators": ["sage", "ffd"]}
        self.assertEqual(combo_target_ratios(loaded, settings), [0.1, 0.01])


class SummaryCsvTests(unittest.TestCase):
    def test_write_summary_csv_and_build_rows(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "a.mat"
            _write_mat(
                path,
                target_ratios=[0.1, 0.01],
                evals_to_target=[[3, np.nan], [np.nan, np.nan]],
                success_by_target=[[True, False], [False, False]],
                trial_status=["ok", "ok"],
                max_evals=20,
            )
            loaded = {"ffd": load_profile_data(path)}
            settings = {"estimators": ["ffd"]}
            combo = (2, "lasso", 1.0, "uniform", 0.0)
            rows = build_summary_rows(combo, 0.1, loaded, settings)
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["n_solved"], 1)
            self.assertEqual(rows[0]["target_ratio"], 0.1)

            csv_path = Path(tmp) / "summary.csv"
            write_summary_csv(rows, csv_path)
            content = csv_path.read_text()
            self.assertIn("dim,problem,condnum", content.splitlines()[0])
            self.assertIn("lasso", content)


# ---------------------------------------------------------------------------
# End-to-end PDF + CSV generation
# ---------------------------------------------------------------------------
class PlotGenerationTests(unittest.TestCase):
    def test_run_generates_pdf_and_summary_csv(self):
        with tempfile.TemporaryDirectory() as tmp:
            rng = np.random.RandomState(0)
            for estimator in ("sage", "ffd"):
                fname = f"opt-bmk-2D-lasso-1.0-{estimator}-uniform-0.000000.mat"
                evals = rng.randint(1, 20, size=(2, 10)).astype(float)
                success = np.ones((2, 10), dtype=bool)
                _write_mat(
                    Path(tmp) / fname,
                    target_ratios=[0.1, 0.01],
                    evals_to_target=evals,
                    success_by_target=success,
                    trial_status=["ok"] * 10,
                    max_evals=20,
                )

            output_dir = Path(tmp) / "plots"
            summary_csv = Path(tmp) / "summary.csv"
            cfg = _settings_dict(
                [tmp],
                problems=["lasso"],
                output_dir=str(output_dir),
                summary_csv=str(summary_csv),
                target_ratios=[0.1, 0.01],
            )
            cfg_path = Path(tmp) / "cfg.yaml"
            _write_settings_yaml(cfg_path, cfg)
            settings = load_settings(cfg_path)

            run(settings, dry_run=False)

            expected_pdf_1 = output_dir / "opt-profile-2D-lasso-cond1-uniform-noise0-target0.1.pdf"
            expected_pdf_2 = output_dir / "opt-profile-2D-lasso-cond1-uniform-noise0-target0.01.pdf"
            self.assertTrue(expected_pdf_1.exists())
            self.assertTrue(expected_pdf_2.exists())
            self.assertGreater(expected_pdf_1.stat().st_size, 0)
            self.assertTrue(summary_csv.exists())
            content = summary_csv.read_text()
            self.assertIn("sage", content)
            self.assertIn("ffd", content)

    def test_dry_run_does_not_create_pdf(self):
        with tempfile.TemporaryDirectory() as tmp:
            for estimator in ("sage", "ffd"):
                fname = f"opt-bmk-2D-lasso-1.0-{estimator}-uniform-0.000000.mat"
                _write_mat(
                    Path(tmp) / fname,
                    target_ratios=[0.1],
                    evals_to_target=[[3, np.nan]],
                    success_by_target=[[True, False]],
                    trial_status=["ok", "ok"],
                    max_evals=20,
                )

            output_dir = Path(tmp) / "plots"
            cfg = _settings_dict([tmp], problems=["lasso"], output_dir=str(output_dir))
            cfg_path = Path(tmp) / "cfg.yaml"
            _write_settings_yaml(cfg_path, cfg)
            settings = load_settings(cfg_path)

            run(settings, dry_run=True)
            self.assertFalse(output_dir.exists())


if __name__ == "__main__":
    unittest.main()
