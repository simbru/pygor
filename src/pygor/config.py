"""
Pygor configuration management.

Loads configuration from:
1. Package defaults (src/pygor/defaults.toml) - always applied
2. Optional user config file - passed explicitly via config_path

User config values override package defaults. Only specify values you want to change.
"""

from pathlib import Path
from typing import Union
import warnings

# Try to import tomllib (Python 3.11+) or tomli as fallback
try:
    import tomllib
    HAS_TOML = True
except ImportError:
    try:
        import tomli as tomllib
        HAS_TOML = True
    except ImportError:
        HAS_TOML = False

# Try to import tomli_w for TOML writing
try:
    import tomli_w
    HAS_TOML_WRITE = True
except ImportError:
    HAS_TOML_WRITE = False


# -----------------------------------------------------------------------------
# Path helpers
# -----------------------------------------------------------------------------

def _get_defaults_path() -> Path:
    """Get path to package defaults.toml."""
    return Path(__file__).parent / "defaults.toml"


# -----------------------------------------------------------------------------
# Config loading
# -----------------------------------------------------------------------------

def _load_toml_file(path: Path) -> dict:
    """Load a TOML file, returning empty dict if not found or toml not available.

    Raises on parse errors so invalid TOML is caught immediately,
    with the file path included in the error message.
    """
    if not HAS_TOML:
        return {}

    if not path.exists():
        return {}

    try:
        with open(path, "rb") as f:
            config = tomllib.load(f)
            return config if config else {}
    except Exception as e:
        raise type(e)(
            f"{e}\n  File: {path}\n"
            f"  Hint: TOML requires 0.5 not .5, and has no null (use \"none\" string)"
        ) from None


def _deep_merge(base: dict, override: dict) -> dict:
    """
    Deep merge two dicts. Values in override take precedence.

    Nested dicts are merged recursively.
    """
    result = base.copy()
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def load_config(config_path: Union[str, Path, None] = None) -> dict:
    """
    Load configuration from package defaults and optional user config.

    Parameters
    ----------
    config_path : str, Path, or None
        Path to a TOML config file to merge with package defaults.
        If None, only package defaults are used.

    Returns
    -------
    dict
        Merged configuration with all sections:
        {
            "instrument": {...},
            "preprocessing": {...},
            "registration": {...},
            "triggers": {...},
            "segmentation": {...},
            "strf": {...},
        }

    Examples
    --------
    >>> # Use package defaults only
    >>> config = load_config()

    >>> # Merge with a project-specific config
    >>> config = load_config("configs/high_zoom.toml")
    """
    # Load package defaults
    defaults = _load_toml_file(_get_defaults_path())

    # Merge user config if provided
    if config_path is not None:
        config_path = Path(config_path)
        if not config_path.exists():
            raise FileNotFoundError(
                f"Config file not found: {config_path.resolve()}"
            )
        user_config = _load_toml_file(config_path)
        return _deep_merge(defaults, user_config)

    return defaults


def get_defaults(section: str, config_path: Union[str, Path, None] = None) -> dict:
    """
    Get defaults for a specific section.

    Supports nested sections using dot notation (e.g., 'strf.spatial').

    Parameters
    ----------
    section : str
        Section name, e.g., 'preprocessing', 'registration', 'strf.spatial',
        'segmentation.cellpose'
    config_path : str, Path, or None
        Optional path to a config file to merge with defaults.

    Returns
    -------
    dict
        Configuration values for the specified section

    Examples
    --------
    >>> get_defaults('preprocessing')
    {'artifact_width': 2, 'flip_x': True, ...}

    >>> get_defaults('strf.spatial')
    {'snr_threshold': 3.0, 'kernel_width': 3, ...}

    >>> get_defaults('segmentation.cellpose')
    {'diameter': 0, 'flow_threshold': 0.9, ...}
    """
    config = load_config(config_path)
    keys = section.split(".")
    result = config
    for key in keys:
        result = result.get(key, {})
    return result.copy() if isinstance(result, dict) else result

