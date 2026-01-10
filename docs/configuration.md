# Configuration

Pygor supports user and project-level configuration to customize default behaviors.

## Configuration Sources

Configuration is loaded from multiple sources (later sources override earlier):

1. **Package defaults** - Built-in values
2. **User config** - `~/.pygor/config.yaml`
3. **Project config** - `./pygor.yaml` (in working directory)

## Configuration File Format

Configuration files use YAML format:

```yaml
# ~/.pygor/config.yaml

preprocessing:
  artifact_width: 2       # Pixels affected by light artifact
  flip_x: true            # X-flip image (standard for ScanM)
  detrend: true           # Apply temporal baseline subtraction
  smooth_window_s: 1000.0 # Detrending smooth window in seconds
  time_bin: 10            # Temporal binning for detrending speed
  fix_first_frame: true   # Fix first-frame artifact

triggers:
  threshold: 20000        # Trigger fires when value < 2^16 - threshold
  min_gap_seconds: 0.1    # Minimum time between triggers (debounce)
```

## Creating a Config Template

Generate a template configuration file:

```python
from pygor.config import create_user_config_template

# Creates ~/.pygor/config.yaml with all options documented
create_user_config_template()
```

## Accessing Configuration

```python
from pygor.config import get_preprocess_defaults, get_trigger_defaults

# Get current preprocessing defaults (merged from all sources)
preproc = get_preprocess_defaults()
print(f"Detrend enabled: {preproc['detrend']}")

# Get trigger detection defaults
triggers = get_trigger_defaults()
print(f"Threshold: {triggers['threshold']}")
```

## Project-Level Config

For project-specific settings, create `pygor.yaml` in your project root:

```yaml
# ./pygor.yaml (in project directory)

preprocessing:
  detrend: false  # This project skips detrending
  artifact_width: 3
```

## Package Defaults

If no config files exist, these defaults are used:

| Section | Parameter | Default |
|---------|-----------|---------|
| `preprocessing` | `artifact_width` | 2 |
| `preprocessing` | `flip_x` | True |
| `preprocessing` | `detrend` | True |
| `preprocessing` | `smooth_window_s` | 1000.0 |
| `preprocessing` | `time_bin` | 10 |
| `preprocessing` | `fix_first_frame` | True |
| `triggers` | `threshold` | 20000 |
| `triggers` | `min_gap_seconds` | 0.1 |

## YAML Dependency

Configuration loading requires PyYAML:

```bash
pip install pyyaml
```

If PyYAML is not installed, only package defaults will be used.

## See Also

- [Preprocessing](modules/preprocessing.md) - Preprocessing parameters
