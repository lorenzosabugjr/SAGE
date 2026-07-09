"""
Optimization Data-Profile Plotting
====================================
CLI tool that discovers optimization-benchmark ``opt-bmk-*.mat`` result
files, matches them against a YAML settings file, and plots data profiles
(fraction of trials solved vs. function evaluations) per (dim, problem,
condnum, noise_type, noise_param, target_ratio) combination.

Run from the repo root:

    python -m utils.plot_opt_profiles --options docs/config_plot_opt.yaml
    python -m utils.plot_opt_profiles --options docs/config_plot_opt.yaml --dry-run
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

FILENAME_PREFIX = "opt-bmk-"
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
    "x_axis": "evals_per_dim",
    "target_ratios": None,
    "write_summary_csv": True,
    "summary_csv": "plots/opt_profile_summary.csv",
    "overwrite": True,
}

CSV_COLUMNS = [
    "dim", "problem", "condnum", "noise_type", "noise_param",
    "estimator", "target_ratio", "n_trials", "n_ok", "n_error",
    "n_solved", "success_rate", "median_evals_to_target", "mean_evals_to_target",
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

    if settings["missing_policy"] not in ("error", "warn_skip"):
        raise SettingsError(
            f"missing_policy must be 'error' or 'warn_skip', got {settings['missing_policy']!r}"
        )
    if settings["duplicate_policy"] != "error":
        raise SettingsError(
            f"duplicate_policy must be 'error', got {settings['duplicate_policy']!r}"
        )
    if settings["x_axis"] not in ("evals_per_dim", "raw_evals"):
        raise SettingsError(
            f"x_axis must be 'evals_per_dim' or 'raw_evals', got {settings['x_axis']!r}"
        )

    if settings["target_ratios"] is not None:
        if not isinstance(settings["target_ratios"], list) or len(settings["target_ratios"]) == 0:
            raise SettingsError("target_ratios must be a non-empty list when provided")
        settings["target_ratios"] = [
            _coerce_float(v, "target_ratios") for v in settings["target_ratios"]
        ]

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
def parse_opt_filename(filename: str, estimators):
    """Parse an ``opt-bmk-...mat`` filename into its components.

    Expected shape: ``opt-bmk-{D}D-{problem}-{condnum}-{estimator}-
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
        match = parse_opt_filename(f.name, estimators)
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
# MAT loading
# ---------------------------------------------------------------------------
def load_profile_data(path):
    """Load the fields needed for data-profile plotting from a .mat file."""
    data = loadmat(str(path))
    trial_status = [str(s[0]) for s in np.asarray(data["trial_status"]).reshape(-1)]
    return {
        "target_ratios": np.asarray(data["target_ratios"], dtype=float).ravel(),
        "evals_to_target": np.asarray(data["evals_to_target"], dtype=float),
        "success_by_target": np.asarray(data["success_by_target"]).astype(bool),
        "trial_status": trial_status,
        "max_evals": int(np.asarray(data["max_evals"]).ravel()[0]),
    }


def _target_ratio_index(target_ratios, target_ratio):
    for idx, value in enumerate(target_ratios):
        if math.isclose(value, target_ratio, rel_tol=1e-9, abs_tol=1e-12):
            return idx
    return None


def combo_target_ratios(loaded_by_estimator, settings):
    """Determine which target ratios to plot for a combo.

    Uses the configured ``target_ratios`` if given, otherwise falls back to
    the target ratios stored in the first (in estimator-config order)
    resolved file.
    """
    if settings["target_ratios"] is not None:
        return list(settings["target_ratios"])
    for estimator in settings["estimators"]:
        if estimator in loaded_by_estimator:
            return list(loaded_by_estimator[estimator]["target_ratios"])
    return []


# ---------------------------------------------------------------------------
# Data-profile computation
# ---------------------------------------------------------------------------
def build_data_profile(evals_row, n_trials: int, dim: int, max_evals: int, x_axis: str):
    """Build a step-function data profile: fraction of trials solved by x.

    Returns (x_values, y_values) suitable for a "steps-post" step plot,
    anchored at (0, 0) and extended flat out to the evaluation budget.
    """
    values = np.sort(evals_row[np.isfinite(evals_row)])
    x_max = max_evals
    if x_axis == "evals_per_dim":
        values = values / dim
        x_max = x_max / dim

    n_solved = values.size
    xs = np.concatenate(([0.0], values, [x_max]))
    ys = np.concatenate(([0.0], np.arange(1, n_solved + 1) / n_trials, [n_solved / n_trials]))
    return xs, ys


def summarize_target(evals_row, success_row, trial_status):
    """Summarize one (estimator, target_ratio) column for the CSV report."""
    n_trials = len(trial_status)
    n_ok = sum(1 for s in trial_status if s == "ok")
    n_error = sum(1 for s in trial_status if s == "error")
    n_solved = int(np.count_nonzero(success_row))
    solved_evals = evals_row[np.asarray(success_row, dtype=bool) & np.isfinite(evals_row)]
    return {
        "n_trials": n_trials,
        "n_ok": n_ok,
        "n_error": n_error,
        "n_solved": n_solved,
        "success_rate": (n_solved / n_trials) if n_trials else None,
        "median_evals_to_target": float(np.median(solved_evals)) if solved_evals.size else None,
        "mean_evals_to_target": float(np.mean(solved_evals)) if solved_evals.size else None,
    }


def build_summary_rows(combo, target_ratio, loaded_by_estimator, settings):
    """Build one summary CSV row per estimator present for this combo/target."""
    dim, problem, condnum, noise_type, noise_param = combo
    rows = []
    for estimator in settings["estimators"]:
        if estimator not in loaded_by_estimator:
            continue
        data = loaded_by_estimator[estimator]
        idx = _target_ratio_index(data["target_ratios"], target_ratio)
        if idx is None:
            continue
        summary = summarize_target(
            data["evals_to_target"][idx], data["success_by_target"][idx], data["trial_status"]
        )
        row = {
            "dim": dim,
            "problem": problem,
            "condnum": condnum,
            "noise_type": noise_type,
            "noise_param": noise_param,
            "estimator": estimator,
            "target_ratio": target_ratio,
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


def plot_combo_target(combo, target_ratio, loaded_by_estimator, settings, output_dir):
    """Render and save the data-profile PDF for one (combo, target_ratio)."""
    dim, problem, condnum, noise_type, noise_param = combo
    x_axis = settings["x_axis"]

    fig, ax = plt.subplots(figsize=(7, 5))
    plotted = False
    for estimator in settings["estimators"]:
        if estimator not in loaded_by_estimator:
            continue
        data = loaded_by_estimator[estimator]
        idx = _target_ratio_index(data["target_ratios"], target_ratio)
        if idx is None:
            continue
        n_trials = len(data["trial_status"])
        xs, ys = build_data_profile(
            data["evals_to_target"][idx], n_trials, dim, data["max_evals"], x_axis
        )
        ax.step(xs, ys, where="post", label=estimator)
        plotted = True

    xlabel = "function evaluations / D" if x_axis == "evals_per_dim" else "function evaluations"
    ax.set_xlabel(xlabel)
    ax.set_ylabel("fraction solved")
    ax.set_ylim(0.0, 1.05)
    ax.set_title(
        f"{dim}D {problem} cond={condnum:g} {noise_type} noise={noise_param:g} "
        f"target_ratio={target_ratio:g}"
    )
    if plotted:
        ax.legend(fontsize="small")
    fig.tight_layout()

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / (
        f"opt-profile-{dim}D-{problem}-cond{_fmt_num(condnum)}-"
        f"{noise_type}-noise{_fmt_num(noise_param)}-target{_fmt_num(target_ratio)}.pdf"
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
        loaded_by_estimator = {est: load_profile_data(path) for est, path in combo_files.items()}
        target_ratios = combo_target_ratios(loaded_by_estimator, settings)
        if not target_ratios:
            print(f"WARNING: no target ratios found for combo {combo}, skipping")
            continue
        for target_ratio in target_ratios:
            out_path = plot_combo_target(combo, target_ratio, loaded_by_estimator, settings, settings["output_dir"])
            print(f"-> saved {out_path}")
            if settings["write_summary_csv"]:
                all_rows.extend(build_summary_rows(combo, target_ratio, loaded_by_estimator, settings))

    if settings["write_summary_csv"]:
        write_summary_csv(all_rows, settings["summary_csv"])
        print(f"-> saved {settings['summary_csv']}")


def main(argv=None):
    parser = argparse.ArgumentParser(description="Plot optimization data profiles")
    parser.add_argument("--options", required=True, help="Path to YAML plot settings file")
    parser.add_argument("--dry-run", action="store_true", help="Discover and print matches only")
    args = parser.parse_args(argv)

    settings = load_settings(args.options)
    run(settings, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
