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
    """Load a TOML file, returning empty dict if not found or toml not available."""
    if not HAS_TOML:
        return {}

    if not path.exists():
        return {}

    try:
        with open(path, "rb") as f:
            config = tomllib.load(f)
            return config if config else {}
    except Exception as e:
        warnings.warn(f"Failed to load config from {path}: {e}")
        return {}


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
        user_config = _load_toml_file(Path(config_path))
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


# -----------------------------------------------------------------------------
# Convenience functions (backward compatible)
# -----------------------------------------------------------------------------

def get_preprocess_defaults() -> dict:
    """
    Get preprocessing defaults from package config.

    Returns
    -------
    dict
        Preprocessing parameters with keys:
        - artifact_width: int
        - flip_x: bool
        - detrend: bool
        - smooth_window_s: float
        - time_bin: int
        - fix_first_frame: bool
    """
    return get_defaults("preprocessing")


def get_registration_defaults() -> dict:
    """
    Get registration defaults from package config.

    Returns
    -------
    dict
        Registration parameters with keys:
        - n_reference_frames: int
        - batch_size: int
        - upsample_factor: int
        - normalization: str or None
        - order: int
        - mode: str
    """
    return get_defaults("registration")


def get_trigger_defaults() -> dict:
    """
    Get trigger detection defaults from package config.

    Returns
    -------
    dict
        Trigger parameters with keys:
        - threshold: int
        - min_gap_seconds: float
    """
    return get_defaults("triggers")


def get_segmentation_defaults() -> dict:
    """
    Get segmentation defaults from package config.

    Returns
    -------
    dict
        Segmentation parameters with nested keys:
        - cellpose: {...}
        - postprocess: {...}
    """
    return get_defaults("segmentation")


def get_instrument_defaults() -> dict:
    """
    Get instrument calibration defaults from package config.

    Returns
    -------
    dict
        Instrument parameters with keys:
        - frame_rate_hz: float
        - fish_screen_dist_mm: float
        - screen_width_mm: float
        - screen_width_pix_au: int
        - screen_height_pix_au: int
        - screen_width_visang: float
        - lens_to_retina_distance_um: float
    """
    return get_defaults("instrument")


# -----------------------------------------------------------------------------
# Legacy constants for backward compatibility
# -----------------------------------------------------------------------------
# These are kept for any code that imports them directly.
# New code should use get_defaults() or the convenience functions.

PREPROCESS_DEFAULTS = get_preprocess_defaults()
REGISTRATION_DEFAULTS = get_registration_defaults()
TRIGGER_DEFAULTS = get_trigger_defaults()
SEGMENTATION_DEFAULTS = get_segmentation_defaults()
