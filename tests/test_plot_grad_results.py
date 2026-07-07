"""
Unit tests for utils/plot_grad_results.py.

Run with: python -m unittest tests.test_plot_grad_results
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

from utils.plot_grad_results import (
    DuplicateFileError,
    MissingFileError,
    SettingsError,
    build_summary_rows,
    discover,
    load_settings,
    parse_grad_filename,
    run,
    summarize_cos_sim,
    summarize_rel_err,
    write_summary_csv,
)

ESTIMATORS = ["ffd", "cfd", "sage"]


def _write_mat(path, rel_err, cos_sim):
    savemat(str(path), {
        "rel_err": np.asarray(rel_err, dtype=float),
        "cos_sim": np.asarray(cos_sim, dtype=float),
    })


def _settings_dict(source_dirs, **overrides):
    cfg = {
        "source_dirs": [str(d) for d in source_dirs],
        "dims": [20],
        "problems": ["least-squares"],
        "condnums": [1.0],
        "noise_types": ["uniform"],
        "noise_params": [0.001],
        "estimators": ["ffd", "cfd"],
    }
    cfg.update(overrides)
    return cfg


def _write_settings_yaml(path, cfg):
    with open(path, "w") as f:
        yaml.safe_dump(cfg, f)


# ---------------------------------------------------------------------------
# Milestone 1: filename parsing, numeric matching, duplicates, missing, dry-run
# ---------------------------------------------------------------------------
class ParseGradFilenameTests(unittest.TestCase):
    def test_parses_simple_problem_name(self):
        match = parse_grad_filename(
            "grad-bmk-20D-lasso-1.0-cfd-uniform-0.001000.mat", ESTIMATORS
        )
        self.assertEqual(match["dim"], 20)
        self.assertEqual(match["problem"], "lasso")
        self.assertEqual(match["condnum"], 1.0)
        self.assertEqual(match["estimator"], "cfd")
        self.assertEqual(match["noise_type"], "uniform")
        self.assertEqual(match["noise_param"], 0.001)

    def test_parses_hyphenated_problem_name(self):
        match = parse_grad_filename(
            "grad-bmk-20D-least-squares-10000.0-ffd-gaussian-0.000000.mat", ESTIMATORS
        )
        self.assertEqual(match["problem"], "least-squares")
        self.assertEqual(match["condnum"], 10000.0)
        self.assertEqual(match["estimator"], "ffd")

    def test_parses_multiply_hyphenated_problem_name(self):
        match = parse_grad_filename(
            "grad-bmk-20D-log-sum-exp-100000000.0-sage-uniform-1.000000.mat", ESTIMATORS
        )
        self.assertEqual(match["problem"], "log-sum-exp")
        self.assertEqual(match["condnum"], 1.0e8)
        self.assertEqual(match["estimator"], "sage")

    def test_returns_none_for_non_matching_prefix(self):
        self.assertIsNone(parse_grad_filename("not-a-grad-file.mat", ESTIMATORS))

    def test_returns_none_when_no_estimator_anchor_found(self):
        # "nmxfd" is not in the configured estimator list.
        match = parse_grad_filename(
            "grad-bmk-20D-lasso-1.0-nmxfd-uniform-0.001000.mat", ESTIMATORS
        )
        self.assertIsNone(match)


class DiscoverAndMatchTests(unittest.TestCase):
    def _make_file(self, tmp, dim, problem, condnum, estimator, noise_type, noise_param):
        fname = f"grad-bmk-{dim}D-{problem}-{condnum}-{estimator}-{noise_type}-{noise_param:.6f}.mat"
        path = Path(tmp) / fname
        _write_mat(path, [1e-6, 2e-6], [0.9, 0.95])
        return path

    def test_numeric_condnum_and_noise_param_matching(self):
        with tempfile.TemporaryDirectory() as tmp:
            self._make_file(tmp, 20, "least-squares", 10000.0, "ffd", "uniform", 0.001)
            self._make_file(tmp, 20, "least-squares", 10000.0, "cfd", "uniform", 0.001)

            settings = load_settings(self._write_cfg(tmp, _settings_dict(
                [tmp], condnums=[1.0e4], noise_params=[1.0e-3],
            )))
            resolved = discover(settings)
            combo = (20, "least-squares", 1.0e4, "uniform", 1.0e-3)
            self.assertEqual(set(resolved[combo].keys()), {"ffd", "cfd"})

    def test_duplicate_matches_raise_with_all_paths(self):
        with tempfile.TemporaryDirectory() as tmp:
            p1 = self._make_file(tmp, 20, "lasso", 1.0, "ffd", "uniform", 0.001)
            sub = Path(tmp) / "sub"
            sub.mkdir()
            p2 = self._make_file(sub, 20, "lasso", 1.0, "ffd", "uniform", 0.001)

            settings = load_settings(self._write_cfg(tmp, _settings_dict(
                [tmp], problems=["lasso"], recursive=True, estimators=["ffd"],
            )))
            with self.assertRaises(DuplicateFileError) as ctx:
                discover(settings)
            self.assertIn(str(p1), str(ctx.exception))
            self.assertIn(str(p2), str(ctx.exception))

    def test_missing_file_raises_by_default(self):
        with tempfile.TemporaryDirectory() as tmp:
            self._make_file(tmp, 20, "lasso", 1.0, "ffd", "uniform", 0.001)
            settings = load_settings(self._write_cfg(tmp, _settings_dict(
                [tmp], problems=["lasso"],
            )))
            with self.assertRaises(MissingFileError):
                discover(settings)

    def test_missing_file_warn_skip_continues(self):
        with tempfile.TemporaryDirectory() as tmp:
            self._make_file(tmp, 20, "lasso", 1.0, "ffd", "uniform", 0.001)
            settings = load_settings(self._write_cfg(tmp, _settings_dict(
                [tmp], problems=["lasso"], missing_policy="warn_skip",
            )))
            buf = io.StringIO()
            with redirect_stdout(buf):
                resolved = discover(settings)
            combo = (20, "lasso", 1.0, "uniform", 0.001)
            self.assertEqual(set(resolved[combo].keys()), {"ffd"})
            self.assertIn("WARNING", buf.getvalue())

    def test_dry_run_prints_matches_and_writes_no_artifacts(self):
        with tempfile.TemporaryDirectory() as tmp:
            self._make_file(tmp, 20, "lasso", 1.0, "ffd", "uniform", 0.001)
            self._make_file(tmp, 20, "lasso", 1.0, "cfd", "uniform", 0.001)
            output_dir = Path(tmp) / "plots"
            settings = load_settings(self._write_cfg(tmp, _settings_dict(
                [tmp], problems=["lasso"], output_dir=str(output_dir),
            )))
            buf = io.StringIO()
            with redirect_stdout(buf):
                run(settings, dry_run=True)
            self.assertIn("ffd", buf.getvalue())
            self.assertIn("cfd", buf.getvalue())
            self.assertFalse(output_dir.exists())

    def _write_cfg(self, tmp, cfg):
        path = Path(tmp) / "plot_settings.yaml"
        _write_settings_yaml(path, cfg)
        return path


class LoadSettingsValidationTests(unittest.TestCase):
    def test_missing_required_field_raises(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg = _settings_dict([tmp])
            del cfg["dims"]
            path = Path(tmp) / "cfg.yaml"
            _write_settings_yaml(path, cfg)
            with self.assertRaises(SettingsError):
                load_settings(path)

    def test_invalid_missing_policy_raises(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg = _settings_dict([tmp], missing_policy="bogus")
            path = Path(tmp) / "cfg.yaml"
            _write_settings_yaml(path, cfg)
            with self.assertRaises(SettingsError):
                load_settings(path)


# ---------------------------------------------------------------------------
# Milestone 2: MAT loading and summary statistics
# ---------------------------------------------------------------------------
class SummaryStatsTests(unittest.TestCase):
    def test_summarize_rel_err_counts_and_stats(self):
        rel_err = np.array([1e-6, 2e-6, 3e-6, 4e-6, np.nan, 0.0, -1.0])
        summary = summarize_rel_err(rel_err)
        self.assertEqual(summary["n_valid"], 4)
        self.assertEqual(summary["n_nan"], 1)
        self.assertEqual(summary["n_nonpositive"], 2)
        self.assertAlmostEqual(summary["min"], 1e-6)
        self.assertAlmostEqual(summary["max"], 4e-6)
        self.assertAlmostEqual(summary["median"], 2.5e-6)

    def test_summarize_cos_sim_counts_and_stats(self):
        cos_sim = np.array([0.9, 0.95, 1.0, -1.0, np.nan, 1.5, -1.5])
        summary = summarize_cos_sim(cos_sim, (-1.0, 1.0))
        self.assertEqual(summary["n_valid"], 4)
        self.assertEqual(summary["n_nan"], 1)
        self.assertEqual(summary["n_nonpositive"], 2)
        self.assertAlmostEqual(summary["min"], -1.0)
        self.assertAlmostEqual(summary["max"], 1.0)

    def test_summarize_with_no_valid_values_returns_none_stats(self):
        rel_err = np.array([np.nan, 0.0, -1.0])
        summary = summarize_rel_err(rel_err)
        self.assertEqual(summary["n_valid"], 0)
        self.assertIsNone(summary["min"])
        self.assertIsNone(summary["max"])


class SummaryCsvTests(unittest.TestCase):
    def test_write_summary_csv_and_build_rows(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "a.mat"
            _write_mat(path, [1e-6, 2e-6, np.nan], [0.9, 1.0, np.nan])
            from utils.plot_grad_results import load_metrics

            metrics = {"ffd": load_metrics(path)}
            settings = {
                "estimators": ["ffd"],
                "cos_sim_range": (-1.0, 1.0),
            }
            combo = (20, "lasso", 1.0, "uniform", 0.001)
            rows = build_summary_rows(combo, metrics, settings)
            self.assertEqual(len(rows), 2)
            metrics_seen = {row["metric"] for row in rows}
            self.assertEqual(metrics_seen, {"rel_err", "cos_sim"})

            csv_path = Path(tmp) / "summary.csv"
            write_summary_csv(rows, csv_path)
            content = csv_path.read_text()
            self.assertIn("dim,problem,condnum", content.splitlines()[0])
            self.assertIn("lasso", content)


# ---------------------------------------------------------------------------
# Milestone 3: Python/Matplotlib PDF generation
# ---------------------------------------------------------------------------
class PlotGenerationTests(unittest.TestCase):
    def test_run_generates_pdf_and_summary_csv(self):
        with tempfile.TemporaryDirectory() as tmp:
            rng = np.random.RandomState(0)
            for estimator in ("ffd", "cfd"):
                fname = f"grad-bmk-20D-lasso-1.0-{estimator}-uniform-0.001000.mat"
                _write_mat(
                    Path(tmp) / fname,
                    rng.uniform(1e-8, 1e-4, size=100),
                    rng.uniform(0.9, 1.0, size=100),
                )

            output_dir = Path(tmp) / "plots"
            summary_csv = Path(tmp) / "summary.csv"
            cfg = _settings_dict(
                [tmp],
                problems=["lasso"],
                output_dir=str(output_dir),
                summary_csv=str(summary_csv),
            )
            cfg_path = Path(tmp) / "cfg.yaml"
            _write_settings_yaml(cfg_path, cfg)
            settings = load_settings(cfg_path)

            run(settings, dry_run=False)

            expected_pdf = output_dir / "grad-hist-20D-lasso-cond1-uniform-noise0.001.pdf"
            self.assertTrue(expected_pdf.exists())
            self.assertGreater(expected_pdf.stat().st_size, 0)
            self.assertTrue(summary_csv.exists())

    def test_dry_run_does_not_create_pdf(self):
        with tempfile.TemporaryDirectory() as tmp:
            for estimator in ("ffd", "cfd"):
                fname = f"grad-bmk-20D-lasso-1.0-{estimator}-uniform-0.001000.mat"
                _write_mat(Path(tmp) / fname, [1e-6, 2e-6], [0.9, 0.95])

            output_dir = Path(tmp) / "plots"
            cfg = _settings_dict([tmp], problems=["lasso"], output_dir=str(output_dir))
            cfg_path = Path(tmp) / "cfg.yaml"
            _write_settings_yaml(cfg_path, cfg)
            settings = load_settings(cfg_path)

            run(settings, dry_run=True)
            self.assertFalse(output_dir.exists())


if __name__ == "__main__":
    unittest.main()
