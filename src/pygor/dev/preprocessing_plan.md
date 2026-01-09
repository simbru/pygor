# Pygor Preprocessing Module Plan

*Document created: 2026-01-09*
*Focus: Core preprocessing functionality to replace IGOR*

---

## 1. Goal

Enable direct loading and preprocessing of ScanM data (SMP/SMH files) in Python, eliminating the IGOR dependency.

**Current flow:**
```
SMP/SMH → IGOR → H5 → Pygor
```

**Target flow:**
```
SMP/SMH → Pygor (preprocessing module) → H5
```

---

## 2. SMP/SMH File Format Reference

### File Pairs

Each recording produces two files:
```
recording_name.smh   # Header (text, metadata)
recording_name.smp   # Pixel data (binary)
```

### SMH Header Format

- **Encoding:** latin-1 (Unicode text)
- **Structure:** `type,variable_name=value;` pairs
- **Key parameters:**

| Key | Type | Description |
|-----|------|-------------|
| `uFrameWidth` | uint32 | Frame width in pixels |
| `uFrameHeight` | uint32 | Frame height in pixels |
| `uNumberOfFrames` | uint32 | Total frames in recording |
| `uInputChannelMask` | uint32 | Bitmask of recorded channels |
| `fXCoord_um` | float32 | Stage X position (microns) |
| `fYCoord_um` | float32 | Stage Y position (microns) |
| `fZCoord_um` | float32 | Stage Z position (microns) |
| `sDateStamp` | string | Acquisition date |
| `sTimeStamp` | string | Acquisition time |
| `fRealPixelDuration_µs` | float32 | Pixel dwell time |

### SMP Binary Format

**Pre-header (64 bytes):** 8 × uint64
```
[0] File type ID
[1-2] GUID
[3] Header size (bytes)
[4] Header length (key-value pairs)
[5] Header start offset (bytes)
[6] Pixel data length (bytes)
[7] Analog data length (bytes)
```

**Pixel data:** 16-bit signed integers, multi-channel interleaved

### Channel Convention

- **Ch0:** Primary imaging channel (GCaMP, etc.)
- **Ch1:** Secondary imaging channel (if dual-color)
- **Ch2:** Stimulus trigger signal
- **Ch3:** Additional channel (rare)

---

## 3. Existing Reference Implementation

**Euler Lab Python reader:** https://github.com/eulerlab/processing_pypeline

Key functions from `readScanM.py`:
- `read_in_header()` - Parse SMH to dict
- `read_in_data()` - Read SMP binary, separate channels
- `to_frame()` - Reshape to (nFrames, height, width)
- `trigger_detection()` - Find triggers from Ch2

**IGOR reference files:**
- `C:\Users\SimenLab\...\User Procedures\ScM\ScanM_FileIO\ScM_FileIO.ipf` - File I/O
- `C:\Users\SimenLab\...\User Procedures\OS\OS_DetrendStack.ipf` - Detrending
- `C:\Users\SimenLab\...\User Procedures\OS\OS_TracesAndTriggers.ipf` - Trigger detection
- `C:\Users\SimenLab\...\User Procedures\OS\OS_Register.ipf` - Registration

---

## 4. Module Structure

```
pygor/
└── preprocessing/
    ├── __init__.py          # Public API exports
    ├── scanm.py             # SMP/SMH loading
    ├── registration.py      # Frame registration, ROI transfer
    ├── detrend.py           # Bleach correction
    ├── triggers.py          # Trigger detection from Ch2
    └── export.py            # H5 file creation (IGOR-compatible)
```

---

## 5. API Design

### scanm.py - SMP/SMH Loading

```python
from pathlib import Path
import numpy as np

def read_smh_header(path: str | Path) -> dict:
    """
    Parse SMH header file to dictionary.

    Parameters
    ----------
    path : str or Path
        Path to .smh file (or .smp - will find partner)

    Returns
    -------
    dict
        Header key-value pairs with appropriate type conversion
    """

def read_smp_data(
    path: str | Path,
    header: dict,
    channels: list[int] = [0, 1]
) -> dict[int, np.ndarray]:
    """
    Read SMP binary pixel data.

    Parameters
    ----------
    path : str or Path
        Path to .smp file
    header : dict
        Parsed header from read_smh_header()
    channels : list of int
        Which channels to load (default: [0, 1])

    Returns
    -------
    dict
        {channel_idx: 3D array (frames, height, width)}
    """

def load_scanm(
    path: str | Path,
    channels: list[int] = [0, 1]
) -> tuple[dict, dict[int, np.ndarray]]:
    """
    Load SMP/SMH file pair.

    Convenience function that loads header and data together.

    Parameters
    ----------
    path : str or Path
        Path to either .smp or .smh file
    channels : list of int
        Which channels to load

    Returns
    -------
    header : dict
        Parsed header
    data : dict
        {channel_idx: 3D stack}

    Examples
    --------
    >>> header, data = load_scanm("recording.smp")
    >>> stack = data[0]  # Primary channel
    >>> print(stack.shape)  # (n_frames, height, width)
    """
```

### registration.py - Frame Registration & ROI Transfer

```python
def register_stack(
    stack: np.ndarray,
    method: str = 'phase',
    reference: str | np.ndarray = 'mean',
    max_shift: int = 10,
    allow_rotation: bool = False
) -> tuple[np.ndarray, list[dict]]:
    """
    Register frames within a stack (motion correction).

    Parameters
    ----------
    stack : ndarray
        3D array (frames, height, width)
    method : str
        Registration method: 'phase' (cross-correlation) or 'ecc' (enhanced correlation)
    reference : str or ndarray
        'mean' (default), 'first', 'median', or a 2D reference image
    max_shift : int
        Maximum allowed shift in pixels
    allow_rotation : bool
        If True, also estimate and correct rotation

    Returns
    -------
    registered : ndarray
        Aligned stack
    transforms : list of dict
        Per-frame transform info: {'shift': (dy, dx), 'rotation': angle, 'quality': float}
    """

def register_to_reference(
    stack: np.ndarray,
    reference: np.ndarray,
    max_shift: int = 20,
    allow_rotation: bool = False
) -> tuple[np.ndarray, dict]:
    """
    Register stack to an external reference image.

    Used for aligning tandem recordings to each other.

    Parameters
    ----------
    stack : ndarray
        3D stack to align
    reference : ndarray
        2D reference image (e.g., mean of another recording)
    max_shift : int
        Maximum shift in pixels
    allow_rotation : bool
        Allow rotation correction

    Returns
    -------
    registered : ndarray
        Aligned stack
    transform : dict
        {'shift': (dy, dx), 'rotation': angle}
    """

def transfer_rois(
    roi_mask: np.ndarray,
    ref_projection: np.ndarray,
    target_projection: np.ndarray,
    max_shift: int = 20,
    allow_rotation: bool = False
) -> tuple[np.ndarray, dict]:
    """
    Transfer ROI mask from one recording to another with shift correction.

    Parameters
    ----------
    roi_mask : ndarray
        2D ROI mask (background=1, ROIs=-1,-2,...)
    ref_projection : ndarray
        Mean projection of source recording
    target_projection : ndarray
        Mean projection of target recording
    max_shift : int
        Maximum shift in pixels
    allow_rotation : bool
        Allow rotation correction

    Returns
    -------
    shifted_mask : ndarray
        ROI mask aligned to target
    transform : dict
        {'shift': (dy, dx), 'rotation': angle, 'quality': float}

    Examples
    --------
    >>> # Recording A has ROIs, apply to recording B
    >>> shifted_rois, transform = transfer_rois(
    ...     rois_a,
    ...     stack_a.mean(axis=0),
    ...     stack_b.mean(axis=0)
    ... )
    >>> print(f"Detected offset: {transform['shift']} pixels")
    """
```

### detrend.py - Bleach Correction

```python
def detrend_stack(
    stack: np.ndarray,
    method: str = 'polynomial',
    order: int = 2,
    percentile: float = 8.0
) -> np.ndarray:
    """
    Remove slow drift/bleaching from imaging stack.

    Parameters
    ----------
    stack : ndarray
        3D array (frames, height, width)
    method : str
        'polynomial' - fit polynomial to baseline
        'exponential' - fit exponential decay
        'rolling' - rolling percentile baseline
    order : int
        Polynomial order (for 'polynomial' method)
    percentile : float
        Percentile for baseline estimation (for 'rolling')

    Returns
    -------
    detrended : ndarray
        Bleach-corrected stack
    """
```

### triggers.py - Trigger Detection

```python
def detect_triggers(
    trigger_channel: np.ndarray,
    threshold: float | None = None,
    min_separation: int = 10,
    mode: str = 'rising'
) -> np.ndarray:
    """
    Detect stimulus trigger frames from Ch2.

    Parameters
    ----------
    trigger_channel : ndarray
        3D trigger channel stack or 1D trigger trace
    threshold : float, optional
        Detection threshold (auto-computed if None)
    min_separation : int
        Minimum frames between triggers
    mode : str
        'rising' - detect rising edges
        'falling' - detect falling edges
        'both' - detect both

    Returns
    -------
    trigger_frames : ndarray
        1D array of frame indices where triggers occurred
    """

def downsample_trigger_channel(
    trigger_stack: np.ndarray,
    method: str = 'mean'
) -> np.ndarray:
    """
    Downsample 3D trigger stack to 1D trace (one value per frame).

    Parameters
    ----------
    trigger_stack : ndarray
        3D array (frames, height, width)
    method : str
        'mean', 'max', or 'sum'

    Returns
    -------
    trace : ndarray
        1D array (frames,)
    """
```

### export.py - H5 File Creation

```python
def create_h5_from_scanm(
    output_path: str | Path,
    header: dict,
    channels: dict[int, np.ndarray],
    rois: np.ndarray | None = None,
    traces: np.ndarray | None = None,
    trigger_frames: np.ndarray | None = None,
    detrended: bool = False
) -> None:
    """
    Create IGOR-compatible H5 file from ScanM data.

    Parameters
    ----------
    output_path : str or Path
        Output .h5 file path
    header : dict
        Parsed SMH header
    channels : dict
        {channel_idx: 3D stack}
    rois : ndarray, optional
        ROI mask
    traces : ndarray, optional
        Extracted traces (n_rois, n_frames)
    trigger_frames : ndarray, optional
        Trigger frame indices
    detrended : bool
        If True, save as wDataCh0_detrended

    Notes
    -----
    Output structure matches IGOR export for backwards compatibility:
    - wDataCh0 (or wDataCh0_detrended)
    - Stack_Ave
    - ROIs
    - Traces0_raw, Traces0_znorm
    - Triggertimes, Triggertimes_Frame
    - OS_Parameters (attributes)
    - wParamsNum, wParamsStr
    """

def update_h5_rois(
    h5_path: str | Path,
    rois: np.ndarray,
    overwrite: bool = True
) -> None:
    """Update ROIs in existing H5 file."""

def update_h5_traces(
    h5_path: str | Path,
    traces_raw: np.ndarray,
    traces_znorm: np.ndarray | None = None,
    overwrite: bool = True
) -> None:
    """Update traces in existing H5 file."""
```

---

## 6. Implementation Priority

### Phase 1A: Core Loading (Highest Priority)

1. `scanm.py` - Get data into Python
   - `read_smh_header()`
   - `read_smp_data()`
   - `load_scanm()`

2. `export.py` - Get data back out
   - `create_h5_from_scanm()` (basic version)

**Milestone:** Can load SMP/SMH → save H5 → load in existing pygor

### Phase 1B: Preprocessing

3. `detrend.py`
   - `detrend_stack()` (polynomial method first)

4. `triggers.py`
   - `downsample_trigger_channel()`
   - `detect_triggers()`

**Milestone:** Full preprocessing pipeline matches IGOR output

### Phase 1C: Registration & ROI Transfer

5. `registration.py`
   - `register_stack()` (phase correlation, no rotation first)
   - `transfer_rois()`
   - Add rotation support

**Milestone:** Can do tandem analysis workflow

---

## 7. Testing Strategy

### Unit Tests

```python
# tests/test_preprocessing/test_scanm.py

def test_read_smh_header():
    """Test header parsing with known file."""

def test_read_smp_data_shape():
    """Verify output shape matches header dimensions."""

def test_load_scanm_roundtrip():
    """Load SMP → save H5 → load H5 → compare."""
```

### Integration Tests

```python
def test_full_preprocessing_pipeline():
    """
    Load SMP/SMH
    → Detrend
    → Detect triggers
    → Save H5
    → Compare against IGOR-produced H5
    """

def test_roi_transfer_accuracy():
    """
    Load tandem pair
    → Draw ROIs on A
    → Transfer to B
    → Verify overlap with manually-drawn ROIs on B
    """
```

### Test Data

Need sample files:
- [ ] Simple single-plane SMP/SMH pair
- [ ] Multi-channel recording
- [ ] IGOR-produced H5 for comparison
- [ ] Tandem pair for ROI transfer testing

---

## 8. Dependencies

```toml
[project.optional-dependencies]
preprocessing = [
    "scikit-image>=0.21",    # Registration
]
```

Core dependencies (already in pygor):
- `numpy`
- `h5py`
- `scipy` (for ndimage.shift)

---

## 9. Open Questions

1. **Detrending method** - Which method does IGOR use by default? Need to match.

2. **Trigger detection parameters** - What threshold/separation works best? May need to expose tuning.

3. **H5 attribute details** - Need to inspect IGOR H5 files closely to match exact attribute structure.

4. **Error handling** - How to handle corrupted SMP files? Missing channels?

---

## 10. Next Steps

1. [ ] Get sample SMP/SMH files for testing
2. [x] Implement SMP/SMH loading → `pygor.preproc.scanm` (2026-01-09)
3. [ ] Test loading with real files
4. [ ] Implement basic H5 export
5. [ ] Compare output with IGOR-produced H5
6. [ ] Implement detrending
7. [ ] Implement trigger detection
8. [ ] Implement registration
9. [ ] Implement ROI transfer

---

## 11. References

- Euler Lab reader: https://github.com/eulerlab/processing_pypeline
- IGOR ScM_FileIO.ipf: `C:\Users\SimenLab\...\ScM\ScanM_FileIO\ScM_FileIO.ipf`
- IGOR OS_GUI.ipf: `C:\Users\SimenLab\...\OS\OS_GUI.ipf`
