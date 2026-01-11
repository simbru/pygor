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

## Registration (Motion Correction)

Registration corrects for sample drift and motion artifacts using batch-averaged phase cross-correlation. This is optimized for low-SNR calcium imaging data.

### Using Core.register()

```python
data = Core.from_scanm("recording.smp", preprocess=True)

# Register with defaults
stats = data.register()
print(f"Mean drift: {stats['mean_shift']}")
print(f"Registration quality: {stats['mean_error']:.4f}")

# Register with custom parameters
stats = data.register(
    n_reference_frames=500,  # Frames for reference
    batch_size=20,           # Frames per batch
    upsample_factor=5,       # Subpixel precision
)

# Force re-registration
stats = data.register(force=True)
```

### Registration Algorithm

The batch-averaged approach dramatically improves registration quality for noisy data:

1. **Create reference**: Average the first `n_reference_frames` (default: 1000)
2. **Divide into batches**: Group frames into batches of `batch_size` (default: 10)
3. **Compute shifts**: For each batch:
   - Average all frames in the batch
   - Compute shift via phase cross-correlation with reference
4. **Apply shifts**: Shift all frames in each batch by the batch's computed shift

### Default Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `n_reference_frames` | 1000 | Frames to average for stable reference |
| `batch_size` | 10 | Frames per batch (higher = better SNR, lower temporal resolution) |
| `upsample_factor` | 10 | Subpixel precision factor |
| `normalization` | None | Phase correlation mode (None is crucial for low SNR!) |
| `artifact_width` | 0 | Pixels to fill at left edge (0 = match preprocessing) |
| `order` | 1 | Spline interpolation order (0-5) |
| `mode` | 'reflect' | Edge handling mode |

### Registration Statistics

The `register()` method returns a dictionary with:

```python
stats = {
    'mean_shift': (dy, dx),      # Mean shift in pixels
    'std_shift': (dy, dx),       # Std deviation of shifts
    'max_shift': (dy, dx),       # Maximum shift
    'mean_error': 0.024,         # Mean error (lower is better)
    'shifts': array([[...], ...]),  # Per-batch shifts (n_batches, 2)
    'errors': array([...]),      # Per-batch errors
}
```

### Low-Level Functions

For advanced use cases:

```python
from pygor.preproc.registration import (
    register_stack,
    compute_batch_shifts,
    apply_shifts_to_stack,
)

# Compute shifts without applying
shifts, errors = compute_batch_shifts(stack, batch_size=10)

# Apply pre-computed shifts
registered = apply_shifts_to_stack(stack, shifts, batch_size=10)

# Or do both in one step
registered, shifts, errors = register_stack(stack, return_shifts=True)
```

### ROI Transfer Between Recordings

Transfer ROI masks between tandem recordings using registration:

```python
from pygor.preproc.registration import transfer_rois

# Load both recordings
data_a = Core.from_scanm("recording_a.smp", preprocess=True)
data_b = Core.from_scanm("recording_b.smp", preprocess=True)

# Transfer ROIs from A to B
shifted_rois, transform = transfer_rois(
    roi_mask=data_a.rois,
    ref_projection=data_a.average_stack,
    target_projection=data_b.average_stack,
)

print(f"Detected offset: {transform['shift']} pixels")
print(f"Registration error: {transform['error']:.4f}")

# Apply to recording B
data_b.rois = shifted_rois
```

### Important Notes

- **Order of operations**: Always run `preprocess()` before `register()`
- **normalization=None**: Critical for low-SNR calcium imaging. Using 'phase' mode will give poor results.
- **Batch size tradeoff**: Larger batches give better shift estimates but lower temporal resolution
- **Artifact width**: Should match preprocessing settings to handle light artifacts consistently

### Configuration

Defaults can be customized via config files:

```yaml
# ~/.pygor/config.yaml
registration:
  n_reference_frames: 500
  batch_size: 20
  upsample_factor: 5
```

See [Configuration](../configuration.md) for details.

## See Also

- [Core Data](core_data.md) - The Core class documentation
- [Configuration](../configuration.md) - Setting up user defaults
- `pygor.preproc.scanm` - ScanM loading and preprocessing
- `pygor.preproc.registration` - Registration module source code
