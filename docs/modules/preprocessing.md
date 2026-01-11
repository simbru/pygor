# Preprocessing Module

The `pygor.preproc` module provides functions for loading and preprocessing ScanM microscopy data directly in Python, eliminating the need for IGOR Pro as an intermediary.

## Overview

**Before (IGOR dependency):**
```
SMP/SMH → IGOR Pro → H5 → Pygor
```

**Now (direct loading):**
```
SMP/SMH → Pygor → (optional) H5
```

## Quick Start

```python
from pygor.classes.core_data import Core

# Load and preprocess in one step
data = Core.from_scanm("recording.smp", preprocess=True)

# Or load raw, then preprocess with custom settings
data = Core.from_scanm("recording.smp")
data.preprocess(artifact_width=2, detrend=True)

# View the result
data.view_images_interactive()
```

## ScanM File Loading

### File Format

ScanM produces paired files for each recording:
- `.smh` - Header file (UTF-16-LE encoded text, metadata)
- `.smp` - Pixel data file (binary, uint16)

### Loading Functions

```python
from pygor.preproc.scanm import load_scanm, read_smh_header

# Load header only
header = read_smh_header("recording.smh")
print(f"Frames: {header['NumberOfFrames']}")
print(f"Dimensions: {header['FrameWidth']} x {header['FrameHeight']}")

# Load header and pixel data
header, data = load_scanm("recording.smp")
images = data[0]  # Channel 0 (imaging)
triggers = data[2]  # Channel 2 (trigger signal)
```

### Channel Convention

| Channel | Typical Use |
|---------|-------------|
| 0 | Primary imaging (GCaMP, etc.) |
| 1 | Secondary imaging (dual-color) |
| 2 | Stimulus trigger signal |
| 3 | Additional channel (rare) |

## Preprocessing Pipeline

The preprocessing pipeline matches IGOR's OS_DetrendStack algorithm:

1. **X-flip** - Flip image horizontally (corrects for scan direction)
2. **Light artifact fix** - Fill artifact pixels with mean value
3. **First frame fix** - Copy frame 2 to frame 1 (first-frame artifact)
4. **Detrending** - Remove slow baseline drift (optional)

### Using Core.preprocess()

```python
data = Core.from_scanm("recording.smp")

# Preprocess with defaults
data.preprocess()

# Preprocess with custom parameters
data.preprocess(
    artifact_width=2,      # Pixels affected by light artifact
    flip_x=True,           # X-flip the image
    detrend=True,          # Apply temporal detrending
    smooth_window_s=1000,  # Detrend smooth window (seconds)
    time_bin=10,           # Temporal binning for speed
    fix_first_frame=True,  # Fix first-frame artifact
)

# Skip detrending (faster)
data.preprocess(detrend=False)
```

### Loading with Preprocessing

```python
# Preprocess with defaults on load
data = Core.from_scanm("recording.smp", preprocess=True)

# Preprocess with custom parameters on load
data = Core.from_scanm("recording.smp", preprocess={
    "artifact_width": 3,
    "detrend": False,
})
```

### Re-preprocessing

By default, calling `preprocess()` twice will warn and skip:

```python
data.preprocess()  # First time: applies
data.preprocess()  # Second time: warns, skips

# Force re-preprocessing
data.preprocess(force=True)
```

## Trigger Detection

Triggers are detected from the trigger channel (typically channel 2) using IGOR-compatible logic:

```python
from pygor.preproc.scanm import detect_triggers

# Detect triggers from trigger channel
trigger_frames, trigger_times = detect_triggers(
    trigger_stack,          # 3D array (frames, lines, pixels)
    line_duration=0.001,    # Line duration in seconds
    trigger_threshold=20000,  # Threshold parameter
    min_gap_seconds=0.1,    # Debounce gap
)

print(f"Found {len(trigger_frames)} triggers")
print(f"First trigger at frame {trigger_frames[0]}, time {trigger_times[0]:.3f}s")
```

### Trigger Detection Algorithm

Triggers fire when the signal drops below a threshold:
- Threshold value = 2^16 - trigger_threshold (e.g., 65536 - 20000 = 45536)
- Only column 0 of the trigger channel is checked
- Debouncing prevents re-triggering within `min_gap_seconds`

## Default Parameters

Default values match IGOR's OS_Parameters:

| Parameter | Default | IGOR Name |
|-----------|---------|-----------|
| `artifact_width` | 2 | LightArtifact_cut |
| `flip_x` | True | (always done) |
| `detrend` | True | Detrend_skip (inverted) |
| `smooth_window_s` | 1000.0 | Detrend_smooth_window |
| `time_bin` | 10 | Detrend_nTimeBin |
| `fix_first_frame` | True | (always done) |
| `trigger_threshold` | 20000 | Trigger_Threshold |
| `min_gap_seconds` | 0.1 | Trigger_after_skip_s |

## Configuration

Defaults can be customized via configuration files. See [Configuration](../configuration.md) for details.

```yaml
# ~/.pygor/config.yaml
preprocessing:
  artifact_width: 2
  detrend: true
  smooth_window_s: 1000.0
```

## Low-Level Functions

For advanced use, the underlying functions are available:

```python
from pygor.preproc.scanm import (
    fix_light_artifact,    # X-flip and artifact handling
    fill_light_artifact,   # Fill artifact region
    detrend_stack,         # Temporal detrending
    preprocess_stack,      # Full pipeline
)

# Manual preprocessing
result, stack_ave = fix_light_artifact(images, artifact_width=2)
result = fill_light_artifact(result, stack_ave, artifact_width=2)
result = detrend_stack(result, frame_rate=15.6, smooth_window_s=1000)
```

## Verification

The preprocessing implementation has been verified against IGOR output:

```
Mean difference: 0.27
Pixels within ±100: 99.89%
Pixels within ±50:  97.89%
Match: ✅ TRUE
```

### Implementation Notes

The detrending uses binomial (Gaussian) smoothing to match IGOR's `Smooth` function behavior.
Key implementation details:

- **Smoothing type**: Gaussian approximation of binomial smoothing (`sigma = sqrt(num/2)`)
- **Edge handling**: Reflect mode (IGOR's "bounce" default)
- **Negative values**: Clipped to 0 (IGOR wraps to unsigned, creating misleading bright pixels)
- **Line duration**: Uses actual timing from header (IGOR hardcodes 2ms)

See `pygor/dev/compare_detrend_params.py` for the verification script.

## See Also

- [Core Data](core_data.md) - The Core class documentation
- [Configuration](../configuration.md) - Setting up user defaults
- `pygor.preproc.scanm` - Module source code
