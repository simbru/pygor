# Pygor Configuration System

## Overview

```
defaults.toml (package defaults)
       ↓
    config.py (load_config, get_defaults)
       ↓
    params.py (AnalysisParams dataclass)
       ↓
    Core.__post_init__ → self.params = AnalysisParams.from_config(self.config)
       ↓
    Methods like register(), preprocess() read from self.params.get_defaults("section")
```

## Key Files

- `src/pygor/defaults.toml` - All package defaults
- `src/pygor/config.py` - TOML loading utilities
- `src/pygor/params.py` - AnalysisParams dataclass (tracks processing state)
- `src/pygor/configs/example.toml` - Example user config

## Using Config on Objects

```python
obj = pygor.load.Core(path, config="my_config.toml")

# Access defaults for any section
obj.params.get_defaults("registration")
obj.params.get_defaults("segmentation")
obj.params.get_defaults("strf")  # nested: general, contouring, spatial, etc.

# Check processing state
obj.params.preprocessed  # True/False
obj.params.registered
obj.params.segmented

# See everything
print(obj.params.summary())
```

## Config Sections

| Section | Keys |
|---------|------|
| `instrument` | fish_screen_dist_mm, screen_width_mm, etc. |
| `preprocessing` | artifact_width, flip_x, detrend, etc. |
| `registration` | n_reference_frames, batch_size, upsample_factor, etc. |
| `triggers` | threshold, min_gap_seconds |
| `segmentation.cellpose` | diameter, flow_threshold, cellprob_threshold |
| `segmentation.postprocess` | split_large, size_multiplier, etc. |
| `strf.general` | num_colours |
| `strf.contouring` | global_thresh_val, min_targets, etc. |
| `strf.spatial` | snr_threshold, kernel_width, etc. |
| `strf.temporal` | exclude_firstlast, extrema_threshold |
| `strf.calculate` | sta_past_window, sta_future_window, etc. |
| `strf.centsurr` | n_clusters, amplitude_ratio_threshold, etc. |

## Retrofitting Old Code

### Before (hardcoded defaults)
```python
def some_function(data, threshold=3.0, kernel_width=3):
    ...
```

### After (config-aware)
```python
def some_function(data, threshold=None, kernel_width=None):
    defaults = data.params.get_defaults("strf").get("spatial", {})

    if threshold is None:
        threshold = defaults.get("snr_threshold", 3.0)
    if kernel_width is None:
        kernel_width = defaults.get("kernel_width", 3)

    ...
```

### Pattern for Core/STRF methods
```python
def register(self, batch_size=None, upsample_factor=None, ...):
    # Load defaults
    defaults = self.params.get_defaults("registration")

    # Use provided value or config default
    if batch_size is None:
        batch_size = defaults.get("batch_size", 100)
    if upsample_factor is None:
        upsample_factor = defaults.get("upsample_factor", 3)

    # ... do registration ...

    # Track what was used
    self.params.mark_registration({
        "batch_size": batch_size,
        "upsample_factor": upsample_factor,
    })
```

## Key Principle

Function parameters default to `None`, then:
1. Check if user passed explicit value → use it
2. Otherwise check config → use config value
3. Otherwise use hardcoded fallback

This preserves backwards compatibility while enabling config-driven workflows.

## TOML Gotcha: None values

TOML has no `None` type. Use string `"None"` and handle in code:
```python
if value in ("None", "none", ""):
    value = None
```

See `registration.py:217-219` for example.
