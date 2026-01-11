"""
Pygor configuration management.

Loads and merges configuration from multiple sources:
1. Package defaults (built-in)
2. User config (~/.pygor/config.toml)

Later sources override earlier ones.
"""

from pathlib import Path
import os
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

# -----------------------------------------------------------------------------
# Package defaults
# -----------------------------------------------------------------------------

PREPROCESS_DEFAULTS = {
    "artifact_width": 2,
    "flip_x": True,
    "detrend": True,
    "smooth_window_s": 1000.0,
    "time_bin": 10,
    "fix_first_frame": True,
}
"""Default preprocessing parameters matching IGOR OS Scripts."""

REGISTRATION_DEFAULTS = {
    "n_reference_frames": 1000,
    "batch_size": 10,
    "upsample_factor": 10,
    "normalization": None,
    "order": 1,
    "mode": "reflect",
}
"""Default registration (motion correction) parameters."""

TRIGGER_DEFAULTS = {
    "threshold": 20000,
    "min_gap_seconds": 0.05,
}
"""Default trigger detection parameters."""


# -----------------------------------------------------------------------------
# Config loading
# -----------------------------------------------------------------------------

def _get_user_config_path() -> Path:
    """Get path to user config file (~/.pygor/config.toml)."""
    return Path(__file__).parent.parent.parent / ".pygor" / "config.toml"

# def _get_project_config_path() -> Path:
#     """Get path to project config file (./pygor.toml)."""
#     return Path.cwd() / "pygor.toml"


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


def load_config() -> dict:
    """
    Load merged configuration from all sources.

    Returns
    -------
    dict
        Merged configuration with structure:
        {
            "preprocessing": {...},
            "registration": {...},
            "triggers": {...},
            ...
        }
    """
    # Start with package defaults
    config = {
        "preprocessing": PREPROCESS_DEFAULTS.copy(),
        "registration": REGISTRATION_DEFAULTS.copy(),
        "triggers": TRIGGER_DEFAULTS.copy(),
    }

    # Merge user config
    user_config = _load_toml_file(_get_user_config_path())
    config = _deep_merge(config, user_config)

    # # Merge project config (highest priority)
    # project_config = _load_toml_file(_get_project_config_path())
    # config = _deep_merge(config, project_config)

    return config


def get_preprocess_defaults() -> dict:
    """
    Get preprocessing defaults from merged config.

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
    config = load_config()
    return config.get("preprocessing", PREPROCESS_DEFAULTS.copy())


def get_registration_defaults() -> dict:
    """
    Get registration defaults from merged config.

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
    config = load_config()
    return config.get("registration", REGISTRATION_DEFAULTS.copy())


def get_trigger_defaults() -> dict:
    """
    Get trigger detection defaults from merged config.

    Returns
    -------
    dict
        Trigger parameters with keys:
        - threshold: int
        - min_gap_seconds: float
    """
    config = load_config()
    return config.get("triggers", TRIGGER_DEFAULTS.copy())


def create_user_config_template():
    """
    Create a template config file at ~/.pygor/config.toml.

    Only creates if the file doesn't already exist.
    """
    config_path = _get_user_config_path()

    if config_path.exists():
        print(f"Config file already exists: {config_path}")
        return config_path

    # Create directory if needed
    config_path.parent.mkdir(parents=True, exist_ok=True)

    template = """\
# Pygor configuration file
# These values override package defaults.
# Delete any lines you don't want to customize.

[preprocessing]
artifact_width = 2        # Pixels affected by light artifact (IGOR: LightArtifact_cut)
flip_x = true             # X-flip image (standard for ScanM data)
detrend = true            # Apply temporal baseline subtraction
smooth_window_s = 1000.0  # Detrending smooth window in seconds
time_bin = 10             # Temporal binning for detrending speed
fix_first_frame = true    # Copy frame 2 to frame 1 (first-frame artifact)

[registration]
n_reference_frames = 1000  # Frames to average for reference
batch_size = 10            # Frames per batch for shift computation
upsample_factor = 10       # Subpixel precision factor
# normalization = "phase"  # Uncomment for high-SNR data (default: none)
order = 1                  # Spline interpolation order (0-5)
mode = "reflect"           # Edge handling mode

[triggers]
threshold = 20000          # Trigger fires when value < 2^16 - threshold
min_gap_seconds = 0.1      # Minimum time between triggers (debounce)
"""

    with open(config_path, "w") as f:
        f.write(template)

    print(f"Created config template: {config_path}")
    return config_path

if __name__ == "__main__":
    create_user_config_template()