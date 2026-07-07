"""
Gradient Benchmark Histogram Plotting
======================================
CLI tool that discovers gradient-benchmark ``.mat`` result files, matches
them against a YAML settings file, and plots relative-error / cosine-
similarity histograms per (dim, problem, condnum, noise_type, noise_param)
combination.

Run from the repo root:

    python -m utils.plot_grad_results --options docs/config_plot_grad.yaml
    python -m utils.plot_grad_results --options docs/config_plot_grad.yaml --dry-run
"""

import argparse
import csv
import itertools
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import yaml
from scipy.io import loadmat

FILENAME_PREFIX = "grad-bmk-"
FILENAME_SUFFIX = ".mat"

REQUIRED_FIELDS = (
    "source_dirs",
    "dims",
    "problems",
    "condnums",
    "noise_types",
    "noise_params",
    "estimators",
)

DEFAULT_SETTINGS = {
    "recursive": False,
    "output_dir": "plots",
    "missing_policy": "error",
    "duplicate_policy": "error",
    "rel_err_bins": 80,
    "cos_sim_bins": 80,
    "cos_sim_range": [-1.0, 1.0],
    "write_summary_csv": True,
    "summary_csv": "plots/grad_hist_summary.csv",
    "overwrite": True,
}

CSV_COLUMNS = [
    "dim", "problem", "condnum", "noise_type", "noise_param",
    "estimator", "metric", "n_valid", "n_nan", "n_nonpositive",
    "min", "q05", "median", "q95", "max",
]


class SettingsError(ValueError):
    """Raised when the YAML plot settings file is missing or invalid."""


class DuplicateFileError(RuntimeError):
    """Raised when more than one file matches the same combination."""


class MissingFileError(RuntimeError):
    """Raised when missing_policy is "error" and a match is absent."""


# ---------------------------------------------------------------------------
# Settings loading
# ---------------------------------------------------------------------------
def load_settings(path) -> dict:
    """Load and validate the YAML plot settings file."""
    with open(path, "r") as f:
        cfg = yaml.safe_load(f)

    if cfg is None:
        cfg = {}
    if not isinstance(cfg, dict):
        raise SettingsError("Plot settings file must be a YAML mapping")

    for field in REQUIRED_FIELDS:
        if field not in cfg:
            raise SettingsError(f"Missing required setting: {field}")
        value = cfg[field]
        if not isinstance(value, list) or len(value) == 0:
            raise SettingsError(f"{field} must be a non-empty list")

    settings = dict(DEFAULT_SETTINGS)
    settings.update(cfg)

    settings["source_dirs"] = [str(v) for v in settings["source_dirs"]]
    settings["dims"] = [_coerce_int(v, "dims") for v in settings["dims"]]
    settings["problems"] = [str(v) for v in settings["problems"]]
    settings["condnums"] = [_coerce_float(v, "condnums") for v in settings["condnums"]]
    settings["noise_types"] = [str(v) for v in settings["noise_types"]]
    settings["noise_params"] = [_coerce_float(v, "noise_params") for v in settings["noise_params"]]
    settings["estimators"] = [str(v) for v in settings["estimators"]]

    settings["recursive"] = bool(settings["recursive"])
    settings["output_dir"] = str(settings["output_dir"])
    settings["overwrite"] = bool(settings["overwrite"])
    settings["write_summary_csv"] = bool(settings["write_summary_csv"])
    settings["summary_csv"] = str(settings["summary_csv"])
    settings["rel_err_bins"] = _coerce_int(settings["rel_err_bins"], "rel_err_bins")
    settings["cos_sim_bins"] = _coerce_int(settings["cos_sim_bins"], "cos_sim_bins")

    cos_range = settings["cos_sim_range"]
    if not isinstance(cos_range, list) or len(cos_range) != 2:
        raise SettingsError("cos_sim_range must be a two-element list [low, high]")
    lo = _coerce_float(cos_range[0], "cos_sim_range")
    hi = _coerce_float(cos_range[1], "cos_sim_range")
    if lo >= hi:
        raise SettingsError("cos_sim_range must have low < high")
    settings["cos_sim_range"] = (lo, hi)

    if settings["missing_policy"] not in ("error", "warn_skip"):
        raise SettingsError(
            f"missing_policy must be 'error' or 'warn_skip', got {settings['missing_policy']!r}"
        )
    if settings["duplicate_policy"] != "error":
        raise SettingsError(
            f"duplicate_policy must be 'error', got {settings['duplicate_policy']!r}"
        )

    return settings


def _coerce_float(value, key: str) -> float:
    if isinstance(value, bool):
        raise SettingsError(f"{key} must be numeric, got {value!r}")
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise SettingsError(f"{key} must be numeric, got {value!r}") from exc


def _coerce_int(value, key: str) -> int:
    if isinstance(value, bool):
        raise SettingsError(f"{key} must be an integer, got {value!r}")
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise SettingsError(f"{key} must be an integer, got {value!r}") from exc
    if not number.is_integer():
        raise SettingsError(f"{key} must be an integer, got {value!r}")
    return int(number)


# ---------------------------------------------------------------------------
# Filename parsing and discovery
# ---------------------------------------------------------------------------
def parse_grad_filename(filename: str, estimators):
    """Parse a ``grad-bmk-...mat`` filename into its components.

    Expected shape: ``grad-bmk-{D}D-{problem}-{condnum}-{estimator}-
    {noise_type}-{noise:.6f}.mat``. Since ``problem`` may itself contain
    hyphens (e.g. "least-squares"), the configured ``estimators`` names are
    used as anchors to split the ambiguous middle section. Returns None if
    the filename does not match.
    """
    if not (filename.startswith(FILENAME_PREFIX) and filename.endswith(FILENAME_SUFFIX)):
        return None

    core = filename[len(FILENAME_PREFIX):-len(FILENAME_SUFFIX)]
    tokens = core.split("-")
    if len(tokens) < 5:
        return None

    dim_token = tokens[0]
    if not dim_token.endswith("D"):
        return None
    try:
        dim = int(dim_token[:-1])
    except ValueError:
        return None

    rest = tokens[1:]
    if len(rest) < 4:
        return None

    noise_param_token = rest[-1]
    noise_type = rest[-2]
    try:
        noise_param = float(noise_param_token)
    except ValueError:
        return None

    body = rest[:-2]
    estimator_set = set(estimators)
    for i in range(len(body) - 1, 0, -1):
        if body[i] not in estimator_set:
            continue
        condnum_token = body[i - 1]
        try:
            condnum = float(condnum_token)
        except ValueError:
            continue
        problem_tokens = body[:i - 1]
        if not problem_tokens:
            continue
        return {
            "dim": dim,
            "problem": "-".join(problem_tokens),
            "condnum": condnum,
            "estimator": body[i],
            "noise_type": noise_type,
            "noise_param": noise_param,
        }
    return None


def discover_mat_files(source_dirs, recursive: bool):
    """Return all ``.mat`` files found under source_dirs (exact or recursive)."""
    files = []
    for source_dir in source_dirs:
        path = Path(source_dir)
        if not path.is_dir():
            raise SettingsError(f"source_dir does not exist or is not a directory: {source_dir}")
        pattern = "**/*.mat" if recursive else "*.mat"
        files.extend(path.glob(pattern))
    return files


def parse_source_files(files, estimators):
    """Parse a list of file paths, discarding any that don't match the pattern."""
    parsed = []
    for f in files:
        match = parse_grad_filename(f.name, estimators)
        if match is not None:
            match["path"] = f
            parsed.append(match)
    return parsed


def _condnum_matches(a: float, b: float) -> bool:
    return math.isclose(a, b, rel_tol=1e-9, abs_tol=1e-9)


def _noise_param_matches(a: float, b: float) -> bool:
    # Filenames only preserve 6 decimal places (see ``{noise:.6f}``).
    return abs(a - b) < 5e-7


def iter_plot_combos(settings):
    """Yield (dim, problem, condnum, noise_type, noise_param) in config order."""
    return itertools.product(
        settings["dims"],
        settings["problems"],
        settings["condnums"],
        settings["noise_types"],
        settings["noise_params"],
    )


def resolve_combo(parsed, combo, estimators, missing_policy):
    """Match each configured estimator to exactly one file for this combo.

    Raises DuplicateFileError on multiple matches. Raises MissingFileError
    on an absent match unless missing_policy is "warn_skip", in which case
    the estimator is omitted from the returned mapping and a warning is
    printed instead.
    """
    dim, problem, condnum, noise_type, noise_param = combo
    files = {}
    missing = []

    for estimator in estimators:
        matches = [
            entry for entry in parsed
            if entry["estimator"] == estimator
            and entry["problem"] == problem
            and entry["noise_type"] == noise_type
            and entry["dim"] == dim
            and _condnum_matches(entry["condnum"], condnum)
            and _noise_param_matches(entry["noise_param"], noise_param)
        ]
        if len(matches) > 1:
            paths = sorted(str(m["path"]) for m in matches)
            raise DuplicateFileError(
                "Duplicate files for dim={}D problem={} condnum={} noise_type={} "
                "noise_param={} estimator={}:\n{}".format(
                    dim, problem, condnum, noise_type, noise_param, estimator,
                    "\n".join(paths),
                )
            )
        if not matches:
            missing.append(estimator)
            continue
        files[estimator] = matches[0]["path"]

    if missing:
        message = (
            "Missing files for dim={}D problem={} condnum={} noise_type={} "
            "noise_param={}: estimators {}".format(
                dim, problem, condnum, noise_type, noise_param, missing,
            )
        )
        if missing_policy == "error":
            raise MissingFileError(message)
        print(f"WARNING: {message}")

    return files


# ---------------------------------------------------------------------------
# MAT loading and summary statistics
# ---------------------------------------------------------------------------
def load_metrics(path):
    """Load and flatten the rel_err and cos_sim arrays from a .mat file."""
    data = loadmat(str(path))
    rel_err = np.asarray(data["rel_err"]).ravel()
    cos_sim = np.asarray(data["cos_sim"]).ravel()
    return rel_err, cos_sim


def _quantile_stats(values):
    if values.size == 0:
        return {"min": None, "q05": None, "median": None, "q95": None, "max": None}
    return {
        "min": float(np.min(values)),
        "q05": float(np.percentile(values, 5)),
        "median": float(np.median(values)),
        "q95": float(np.percentile(values, 95)),
        "max": float(np.max(values)),
    }


def summarize_rel_err(rel_err):
    """Count finite positive, NaN, and nonpositive rel_err values, plus stats."""
    finite = np.isfinite(rel_err)
    valid_mask = finite & (rel_err > 0)
    summary = {
        "n_valid": int(np.count_nonzero(valid_mask)),
        "n_nan": int(np.count_nonzero(np.isnan(rel_err))),
        "n_nonpositive": int(np.count_nonzero(finite & (rel_err <= 0))),
    }
    summary.update(_quantile_stats(rel_err[valid_mask]))
    return summary


def summarize_cos_sim(cos_sim, value_range):
    """Count finite in-range, NaN, and out-of-range cos_sim values, plus stats."""
    lo, hi = value_range
    finite = np.isfinite(cos_sim)
    valid_mask = finite & (cos_sim >= lo) & (cos_sim <= hi)
    summary = {
        "n_valid": int(np.count_nonzero(valid_mask)),
        "n_nan": int(np.count_nonzero(np.isnan(cos_sim))),
        "n_nonpositive": int(np.count_nonzero(finite & ~valid_mask)),
    }
    summary.update(_quantile_stats(cos_sim[valid_mask]))
    return summary


def build_summary_rows(combo, metrics, settings):
    """Build one summary CSV row per (estimator, metric) present in metrics."""
    dim, problem, condnum, noise_type, noise_param = combo
    rows = []
    for estimator in settings["estimators"]:
        if estimator not in metrics:
            continue
        rel_err, cos_sim = metrics[estimator]
        summaries = (
            ("rel_err", summarize_rel_err(rel_err)),
            ("cos_sim", summarize_cos_sim(cos_sim, settings["cos_sim_range"])),
        )
        for metric_name, summary in summaries:
            row = {
                "dim": dim,
                "problem": problem,
                "condnum": condnum,
                "noise_type": noise_type,
                "noise_param": noise_param,
                "estimator": estimator,
                "metric": metric_name,
            }
            row.update(summary)
            rows.append(row)
    return rows


def write_summary_csv(rows, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------
def _fmt_num(x: float) -> str:
    return f"{x:g}"


def _shared_log_bins(arrays, n_bins):
    non_empty = [a for a in arrays if a.size > 0]
    if not non_empty:
        return None
    combined = np.concatenate(non_empty)
    lo, hi = float(np.min(combined)), float(np.max(combined))
    if lo == hi:
        lo, hi = lo / 2.0, hi * 2.0
    return np.logspace(np.log10(lo), np.log10(hi), n_bins + 1)


def _linear_bins(value_range, n_bins):
    lo, hi = value_range
    return np.linspace(lo, hi, n_bins + 1)


def _plot_step_hist(ax, estimators, valid_by_estimator, bins):
    for estimator in estimators:
        if estimator not in valid_by_estimator:
            continue
        values = valid_by_estimator[estimator]
        n_valid = values.size
        label = f"{estimator} (n={n_valid})"
        if bins is not None and n_valid > 0:
            weights = np.full(n_valid, 1.0 / n_valid)
            ax.hist(values, bins=bins, weights=weights, histtype="step", label=label)
        else:
            ax.plot([], [], label=label)


def plot_combo(combo, metrics, settings, output_dir):
    """Render and save the two-panel (rel_err, cos_sim) PDF for one combo."""
    dim, problem, condnum, noise_type, noise_param = combo
    cos_lo, cos_hi = settings["cos_sim_range"]

    rel_err_valid = {
        est: r[np.isfinite(r) & (r > 0)] for est, (r, _) in metrics.items()
    }
    cos_sim_valid = {
        est: c[np.isfinite(c) & (c >= cos_lo) & (c <= cos_hi)] for est, (_, c) in metrics.items()
    }

    rel_bins = _shared_log_bins(list(rel_err_valid.values()), settings["rel_err_bins"])
    cos_bins = _linear_bins(settings["cos_sim_range"], settings["cos_sim_bins"])

    fig, (ax_rel, ax_cos) = plt.subplots(2, 1, figsize=(7, 9))

    _plot_step_hist(ax_rel, settings["estimators"], rel_err_valid, rel_bins)
    ax_rel.set_xscale("log")
    ax_rel.set_xlabel("relative error")
    ax_rel.set_ylabel("probability")
    ax_rel.set_title(
        f"{dim}D {problem} cond={condnum:g} {noise_type} noise={noise_param:g}"
    )
    ax_rel.legend(fontsize="small")

    _plot_step_hist(ax_cos, settings["estimators"], cos_sim_valid, cos_bins)
    ax_cos.set_xlabel("cosine similarity")
    ax_cos.set_ylabel("probability")
    ax_cos.legend(fontsize="small")

    fig.tight_layout()

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / (
        f"grad-hist-{dim}D-{problem}-cond{_fmt_num(condnum)}-"
        f"{noise_type}-noise{_fmt_num(noise_param)}.pdf"
    )
    if out_path.exists() and not settings["overwrite"]:
        plt.close(fig)
        raise FileExistsError(f"Output already exists and overwrite is false: {out_path}")

    fig.savefig(out_path)
    plt.close(fig)
    return out_path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def discover(settings):
    """Run discovery + matching for every combo. Shared by --dry-run and real runs."""
    files = discover_mat_files(settings["source_dirs"], settings["recursive"])
    parsed = parse_source_files(files, settings["estimators"])

    resolved = {}
    for combo in iter_plot_combos(settings):
        resolved[combo] = resolve_combo(parsed, combo, settings["estimators"], settings["missing_policy"])
    return resolved


def _print_dry_run(resolved):
    for combo, combo_files in resolved.items():
        dim, problem, condnum, noise_type, noise_param = combo
        print(f"{dim}D {problem} cond={condnum:g} {noise_type} noise={noise_param:g}")
        if not combo_files:
            print("    (no matching files)")
            continue
        for estimator, path in combo_files.items():
            print(f"    {estimator}: {path}")


def run(settings, dry_run: bool = False):
    resolved = discover(settings)

    if dry_run:
        _print_dry_run(resolved)
        return

    all_rows = []
    for combo, combo_files in resolved.items():
        if not combo_files:
            print(f"WARNING: no data found for combo {combo}, skipping")
            continue
        metrics = {est: load_metrics(path) for est, path in combo_files.items()}
        out_path = plot_combo(combo, metrics, settings, settings["output_dir"])
        print(f"-> saved {out_path}")
        if settings["write_summary_csv"]:
            all_rows.extend(build_summary_rows(combo, metrics, settings))

    if settings["write_summary_csv"]:
        write_summary_csv(all_rows, settings["summary_csv"])
        print(f"-> saved {settings['summary_csv']}")


def main(argv=None):
    parser = argparse.ArgumentParser(description="Plot gradient benchmark histograms")
    parser.add_argument("--options", required=True, help="Path to YAML plot settings file")
    parser.add_argument("--dry-run", action="store_true", help="Discover and print matches only")
    args = parser.parse_args(argv)

    settings = load_settings(args.options)
    run(settings, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
