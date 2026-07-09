"""
Shared YAML config coercion helpers for benchmark scripts.

YAML lets numeric-looking values slip in as strings (e.g. exponent
notation like "1.0e-6") or as booleans where a number is expected.
These helpers make that coercion explicit and consistent across
benchmark config loaders.
"""


def coerce_float(value, key: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{key} must be numeric, got {value!r}")
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{key} must be numeric, got {value!r}") from exc


def coerce_int(value, key: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{key} must be an integer, got {value!r}")
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{key} must be an integer, got {value!r}") from exc

    if not number.is_integer():
        raise ValueError(f"{key} must be an integer, got {value!r}")
    return int(number)


def coerce_bool(value, key: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"{key} must be a boolean, got {value!r}")
    return value


def coerce_numeric_list(cfg: dict, key: str, coerce) -> None:
    """In-place coerce cfg[key] (a list) elementwise, if present."""
    if key not in cfg:
        return
    values = cfg[key]
    if not isinstance(values, list):
        raise ValueError(f"{key} must be a list")
    cfg[key] = [coerce(value, f"{key}[{idx}]") for idx, value in enumerate(values)]


def coerce_optional_scalar(cfg: dict, key: str, coerce, default=None) -> None:
    """In-place coerce cfg[key] if present; otherwise set it to default."""
    if key in cfg:
        cfg[key] = coerce(cfg[key], key)
    elif default is not None:
        cfg[key] = default


def require_keys(cfg: dict, keys) -> None:
    missing = [key for key in keys if key not in cfg]
    if missing:
        raise ValueError(f"Missing required config key(s): {', '.join(missing)}")
