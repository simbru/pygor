# Tandem Workflow Design

*Document created: 2026-01-09*
*Focus: Processing paired recordings (same cells, different stimuli)*

---

## Related Documents

- **[preprocessing_plan.md](preprocessing_plan.md)** - Core preprocessing functions this workflow depends on
- **[gui_notes_and_plans.md](gui_notes_and_plans.md)** - GUI that will wrap this workflow

---

## 1. The Problem

When analyzing the same cells under different stimuli (e.g., SWN for receptive fields + moving bars for direction selectivity), the current workflow requires:

1. Open Recording A in IGOR
2. Register frames
3. Draw ROIs
4. Extract traces
5. Save H5
6. **Repeat steps 1-5 for Recording B** (same cells!)
7. **Manually ensure ROIs match** between A and B

For hundreds of scan planes, this is tedious and error-prone.

---

## 2. The Solution: Tandem Workflow

Process paired recordings together, drawing ROIs once and automatically transferring them.

```
┌─────────────────┐     ┌─────────────────┐
│  Recording A    │     │  Recording B    │
│  (e.g., SWN)    │     │  (e.g., Bars)   │
└────────┬────────┘     └────────┬────────┘
         │                       │
         ▼                       │
   ┌───────────┐                 │
   │ Register  │                 │
   │ Frames A  │                 │
   └─────┬─────┘                 │
         │                       │
         ▼                       │
   ┌───────────┐                 │
   │ Draw ROIs │                 │
   │ (once!)   │                 │
   └─────┬─────┘                 │
         │                       │
         ├───────────────────────┤
         │                       │
         ▼                       ▼
   ┌───────────┐           ┌───────────┐
   │ Extract   │           │ Detect    │
   │ Traces A  │           │ Offset    │
   └─────┬─────┘           │ A → B     │
         │                 └─────┬─────┘
         │                       │
         │                       ▼
         │                 ┌───────────┐
         │                 │ Transfer  │
         │                 │ ROIs      │
         │                 └─────┬─────┘
         │                       │
         │                       ▼
         │                 ┌───────────┐
         │                 │ Register  │
         │                 │ Frames B  │
         │                 └─────┬─────┘
         │                       │
         │                       ▼
         │                 ┌───────────┐
         │                 │ Extract   │
         │                 │ Traces B  │
         │                 └─────┬─────┘
         │                       │
         ▼                       ▼
   ┌─────────────────────────────────┐
   │  Save both H5 files             │
   │  (ROIs guaranteed to match)     │
   └─────────────────────────────────┘
```

---

## 3. Use Cases

### Use Case 1: Simple Tandem (Two Recordings)

Most common case - compare two stimuli on same cells.

```python
pair = TandemPair(
    primary="0_0_SWN_200_White.smp",
    secondary="0_0_RGBUVAll_0.smp"
)
pair.process()  # Register, draw ROIs, transfer, extract, save
```

### Use Case 2: Multi-Tandem (N Recordings)

Same cells, multiple stimuli (SWN + Bars + FullField + ...).

```python
group = TandemGroup(
    primary="0_0_SWN_200_White.smp",
    secondaries=[
        "0_0_RGBUVAll_0.smp",
        "0_0_gradient_contrast_400_white.smp",
    ]
)
group.process()  # ROIs drawn once, transferred to all secondaries
```

### Use Case 3: Batch Tandem (Many Planes)

Process all planes from an experiment session.

```python
batch = TandemBatch.from_directory(
    "D:/experiment_2025-11-12/",
    primary_pattern="*_SWN_*.smp",
    secondary_pattern="*_RGBUVAll_*.smp",
    pair_by="plane"  # Match by plane index (0_0_, 0_1_, etc.)
)
batch.process_all(parallel=True)
```

---

## 4. API Design

### TandemPair - Core Class

```python
from dataclasses import dataclass, field
from pathlib import Path
import numpy as np

@dataclass
class TandemPair:
    """
    Manage a pair of recordings for tandem analysis.

    The primary recording is where ROIs are drawn.
    The secondary recording receives transferred ROIs.

    Parameters
    ----------
    primary : str or Path
        Path to primary recording (SMP/SMH or H5)
    secondary : str or Path
        Path to secondary recording

    Examples
    --------
    >>> pair = TandemPair("recording_SWN.smp", "recording_Bars.smp")
    >>> pair.load()
    >>> pair.register_primary()
    >>> pair.draw_rois()  # Opens Napari
    >>> pair.transfer_rois()  # Auto-detect offset, apply to secondary
    >>> pair.extract_traces()
    >>> pair.save("output/")
    """

    primary: str | Path
    secondary: str | Path

    # Loaded data (populated by load())
    primary_header: dict = field(default=None, init=False, repr=False)
    secondary_header: dict = field(default=None, init=False, repr=False)
    primary_stack: np.ndarray = field(default=None, init=False, repr=False)
    secondary_stack: np.ndarray = field(default=None, init=False, repr=False)

    # Processing results
    roi_mask: np.ndarray = field(default=None, init=False, repr=False)
    secondary_roi_mask: np.ndarray = field(default=None, init=False, repr=False)
    offset: dict = field(default=None, init=False)  # {'shift': (dy, dx), 'rotation': angle}

    primary_traces: np.ndarray = field(default=None, init=False, repr=False)
    secondary_traces: np.ndarray = field(default=None, init=False, repr=False)

    # State tracking
    _is_loaded: bool = field(default=False, init=False)
    _is_registered: bool = field(default=False, init=False)
    _has_rois: bool = field(default=False, init=False)
    _is_transferred: bool = field(default=False, init=False)

    def load(self, channels: list[int] = [0, 1]) -> "TandemPair":
        """
        Load both recordings into memory.

        Parameters
        ----------
        channels : list of int
            Which channels to load (default: [0, 1])

        Returns
        -------
        self : TandemPair
            For method chaining
        """
        ...

    def register_primary(
        self,
        method: str = 'phase',
        reference: str = 'mean',
        max_shift: int = 10
    ) -> "TandemPair":
        """
        Register frames in primary recording (motion correction).

        Returns
        -------
        self : TandemPair
            For method chaining
        """
        ...

    def draw_rois(self, existing_rois: np.ndarray = None) -> "TandemPair":
        """
        Launch Napari for interactive ROI drawing on primary.

        Parameters
        ----------
        existing_rois : ndarray, optional
            Pre-existing ROI mask to edit

        Returns
        -------
        self : TandemPair
            For method chaining
        """
        ...

    def compute_offset(
        self,
        max_shift: int = 20,
        allow_rotation: bool = False
    ) -> dict:
        """
        Compute spatial offset between primary and secondary.

        Uses cross-correlation on mean projections.

        Returns
        -------
        offset : dict
            {'shift': (dy, dx), 'rotation': angle, 'quality': float}
        """
        ...

    def transfer_rois(
        self,
        max_shift: int = 20,
        allow_rotation: bool = False,
        verify: bool = True
    ) -> "TandemPair":
        """
        Transfer ROIs from primary to secondary with offset correction.

        Parameters
        ----------
        max_shift : int
            Maximum allowed shift in pixels
        allow_rotation : bool
            Allow rotation correction
        verify : bool
            If True, show Napari viewer to verify transfer (optional edit)

        Returns
        -------
        self : TandemPair
            For method chaining
        """
        ...

    def extract_traces(self, parallel: bool = True) -> "TandemPair":
        """
        Extract traces from both recordings using their respective ROIs.

        Returns
        -------
        self : TandemPair
            For method chaining
        """
        ...

    def save(
        self,
        output_dir: str | Path,
        format: str = 'h5'
    ) -> tuple[Path, Path]:
        """
        Save both recordings as H5 files.

        Parameters
        ----------
        output_dir : str or Path
            Output directory
        format : str
            Output format ('h5' only for now)

        Returns
        -------
        primary_path, secondary_path : tuple of Path
            Paths to saved files
        """
        ...

    def process(
        self,
        output_dir: str | Path,
        register: bool = True,
        verify_transfer: bool = True
    ) -> tuple[Path, Path]:
        """
        Run full workflow: load → register → draw ROIs → transfer → extract → save.

        Convenience method that chains all steps.

        Parameters
        ----------
        output_dir : str or Path
            Output directory for H5 files
        register : bool
            Whether to register frames (motion correction)
        verify_transfer : bool
            Whether to show verification viewer after ROI transfer

        Returns
        -------
        primary_path, secondary_path : tuple of Path
            Paths to saved files
        """
        ...

    # Properties
    @property
    def num_rois(self) -> int:
        """Number of ROIs drawn."""
        ...

    @property
    def primary_projection(self) -> np.ndarray:
        """Mean projection of primary stack."""
        ...

    @property
    def secondary_projection(self) -> np.ndarray:
        """Mean projection of secondary stack."""
        ...
```

### TandemGroup - Multiple Secondaries

```python
@dataclass
class TandemGroup:
    """
    Manage one primary with multiple secondary recordings.

    ROIs are drawn once on primary and transferred to all secondaries.

    Parameters
    ----------
    primary : str or Path
        Path to primary recording
    secondaries : list of str or Path
        Paths to secondary recordings

    Examples
    --------
    >>> group = TandemGroup(
    ...     primary="0_0_SWN.smp",
    ...     secondaries=["0_0_Bars.smp", "0_0_FullField.smp"]
    ... )
    >>> group.process("output/")
    """

    primary: str | Path
    secondaries: list[str | Path]

    # Internal pairs
    _pairs: list[TandemPair] = field(default_factory=list, init=False)

    def load(self) -> "TandemGroup":
        """Load primary and all secondaries."""
        ...

    def draw_rois(self) -> "TandemGroup":
        """Draw ROIs on primary (once)."""
        ...

    def transfer_all(self, verify: bool = False) -> "TandemGroup":
        """Transfer ROIs to all secondaries."""
        ...

    def process(self, output_dir: str | Path) -> list[Path]:
        """Full workflow for all recordings."""
        ...
```

### TandemBatch - Many Planes

```python
@dataclass
class TandemBatch:
    """
    Batch process many tandem pairs from a directory.

    Examples
    --------
    >>> batch = TandemBatch.from_directory(
    ...     "D:/experiment/",
    ...     primary_pattern="*_SWN_*.smp",
    ...     secondary_pattern="*_Bars_*.smp"
    ... )
    >>> batch.process_all(parallel=True)
    """

    pairs: list[TandemPair]

    @classmethod
    def from_directory(
        cls,
        directory: str | Path,
        primary_pattern: str,
        secondary_pattern: str,
        pair_by: str = 'plane'
    ) -> "TandemBatch":
        """
        Create batch from directory by matching file patterns.

        Parameters
        ----------
        directory : str or Path
            Directory containing recordings
        primary_pattern : str
            Glob pattern for primary files
        secondary_pattern : str
            Glob pattern for secondary files
        pair_by : str
            How to match pairs:
            - 'plane': Match by plane index (0_0_, 0_1_, etc.)
            - 'name': Match by base filename

        Returns
        -------
        TandemBatch
        """
        ...

    @classmethod
    def from_pairs(
        cls,
        pairs: list[tuple[str, str]]
    ) -> "TandemBatch":
        """
        Create batch from explicit list of (primary, secondary) pairs.
        """
        ...

    def process_all(
        self,
        output_dir: str | Path,
        parallel: bool = True,
        n_jobs: int = -1,
        progress: bool = True
    ) -> list[tuple[Path, Path]]:
        """
        Process all pairs.

        Parameters
        ----------
        output_dir : str or Path
            Output directory
        parallel : bool
            Use parallel processing
        n_jobs : int
            Number of parallel jobs (-1 = all cores)
        progress : bool
            Show progress bar

        Returns
        -------
        list of (primary_path, secondary_path) tuples
        """
        ...

    def __len__(self) -> int:
        return len(self.pairs)

    def __iter__(self):
        return iter(self.pairs)
```

---

## 5. Workflow States and Validation

The `TandemPair` tracks state to prevent invalid operations:

```python
# Valid workflow order:
pair.load()           # Required first
pair.register_primary()  # Optional but recommended
pair.draw_rois()      # Required before transfer
pair.transfer_rois()  # Requires ROIs
pair.extract_traces() # Requires transfer (or manual secondary ROIs)
pair.save()           # Requires traces

# Invalid operations raise errors:
pair.transfer_rois()  # Error: Must draw ROIs first
pair.draw_rois()      # Error: Must load first
```

State transitions:

```
CREATED → load() → LOADED → register_primary() → REGISTERED
                      ↓                               ↓
                 draw_rois()                    draw_rois()
                      ↓                               ↓
                 HAS_ROIS ←───────────────────── HAS_ROIS
                      ↓
              transfer_rois()
                      ↓
                TRANSFERRED
                      ↓
              extract_traces()
                      ↓
                 EXTRACTED
                      ↓
                   save()
                      ↓
                   SAVED
```

---

## 6. Integration with Existing Pygor

### Using with Existing H5 Files

The tandem workflow should work with both:
- Raw SMP/SMH files (new preprocessing)
- Existing H5 files (from IGOR)

```python
# From raw files
pair = TandemPair("recording.smp", "recording2.smp")

# From existing H5 (skip preprocessing, just transfer ROIs)
pair = TandemPair("recording.h5", "recording2.h5")
pair.load()
pair.draw_rois()  # Or load existing: pair.draw_rois(existing_rois=h5_rois)
pair.transfer_rois()
pair.save()  # Updates existing H5 files
```

### Using ROIs from Core Class

```python
from pygor.classes import Core

# Load existing processed recording
core = Core("processed.h5")

# Create tandem pair and use existing ROIs
pair = TandemPair("processed.h5", "new_recording.smp")
pair.load()
pair.draw_rois(existing_rois=core.rois)  # Start with existing ROIs
pair.transfer_rois()
```

---

## 7. Napari Integration

ROI drawing uses the existing `NapariRoiPrompt` from `pygor.core.gui.methods`:

```python
def draw_rois(self, existing_rois=None):
    from pygor.core.gui.methods import NapariRoiPrompt

    # Prepare display stack (mean projection or full stack)
    display = self.primary_projection

    # Launch Napari
    prompt = NapariRoiPrompt(
        stack=self.primary_stack,
        average=display,
        existing_rois=existing_rois
    )
    prompt.run()

    self.roi_mask = prompt.roi_mask
    self._has_rois = True
```

For transfer verification:

```python
def _verify_transfer(self):
    """Show side-by-side comparison after ROI transfer."""
    import napari

    viewer = napari.Viewer()

    # Left: Primary with ROIs
    viewer.add_image(self.primary_projection, name="Primary")
    viewer.add_labels(self.roi_mask, name="Primary ROIs")

    # Right: Secondary with transferred ROIs
    viewer.add_image(self.secondary_projection, name="Secondary")
    viewer.add_labels(self.secondary_roi_mask, name="Transferred ROIs")

    # Show offset info
    print(f"Detected offset: {self.offset['shift']} pixels")
    if self.offset.get('rotation'):
        print(f"Detected rotation: {self.offset['rotation']:.2f}°")

    napari.run()
```

---

## 8. Error Handling

### Common Errors

```python
class TandemError(Exception):
    """Base class for tandem workflow errors."""
    pass

class NotLoadedError(TandemError):
    """Raised when operation requires loaded data."""
    pass

class NoRoisError(TandemError):
    """Raised when operation requires ROIs."""
    pass

class TransferFailedError(TandemError):
    """Raised when ROI transfer fails (e.g., offset too large)."""
    pass

class IncompatibleRecordingsError(TandemError):
    """Raised when recordings can't be paired (different dimensions, etc.)."""
    pass
```

### Validation on Load

```python
def load(self):
    # Load both
    self.primary_header, primary_data = load_scanm(self.primary)
    self.secondary_header, secondary_data = load_scanm(self.secondary)

    self.primary_stack = primary_data[0]
    self.secondary_stack = secondary_data[0]

    # Validate compatibility
    if self.primary_stack.shape[1:] != self.secondary_stack.shape[1:]:
        raise IncompatibleRecordingsError(
            f"Frame dimensions don't match: "
            f"{self.primary_stack.shape[1:]} vs {self.secondary_stack.shape[1:]}"
        )

    self._is_loaded = True
    return self
```

---

## 9. Progress Reporting

For GUI and batch processing, operations should report progress:

```python
from typing import Callable

def transfer_rois(
    self,
    progress_callback: Callable[[float, str], None] = None,
    ...
):
    if progress_callback:
        progress_callback(0.0, "Computing offset...")

    self.offset = self.compute_offset()

    if progress_callback:
        progress_callback(0.5, f"Applying offset {self.offset['shift']}...")

    self.secondary_roi_mask = transfer_rois(
        self.roi_mask,
        self.primary_projection,
        self.secondary_projection,
        ...
    )

    if progress_callback:
        progress_callback(1.0, "Transfer complete")
```

---

## 10. Module Structure

```
pygor/
└── tandem/
    ├── __init__.py       # Public exports
    ├── pair.py           # TandemPair class
    ├── group.py          # TandemGroup class
    ├── batch.py          # TandemBatch class
    └── exceptions.py     # Error classes
```

`__init__.py`:
```python
from .pair import TandemPair
from .group import TandemGroup
from .batch import TandemBatch
from .exceptions import (
    TandemError,
    NotLoadedError,
    NoRoisError,
    TransferFailedError,
    IncompatibleRecordingsError,
)

__all__ = [
    "TandemPair",
    "TandemGroup",
    "TandemBatch",
    "TandemError",
    "NotLoadedError",
    "NoRoisError",
    "TransferFailedError",
    "IncompatibleRecordingsError",
]
```

---

## 11. Dependencies

The tandem module depends on:
- `pygor.preprocessing` (Phase 1) - loading, registration, ROI transfer
- `pygor.core.gui.methods` - Napari ROI drawing (existing)
- `numpy`, `h5py` - data handling (existing)
- `joblib` - parallel batch processing (existing)

No new dependencies required.

---

## 12. Implementation Priority

1. **TandemPair** - Core functionality
   - `load()`, `draw_rois()`, `compute_offset()`, `transfer_rois()`
   - `extract_traces()`, `save()`

2. **Verification viewer** - Visual confirmation of ROI transfer

3. **TandemGroup** - Multiple secondaries (straightforward extension)

4. **TandemBatch** - Batch processing with progress

---

## 13. Testing Strategy

### Unit Tests

```python
def test_tandem_pair_load():
    """Test loading paired recordings."""

def test_compute_offset_known_shift():
    """Test offset detection with synthetic data (known shift)."""

def test_transfer_rois_preserves_count():
    """Transferred ROIs should have same count as original."""

def test_incompatible_recordings_raises():
    """Different frame sizes should raise IncompatibleRecordingsError."""
```

### Integration Tests

```python
def test_full_tandem_workflow():
    """
    Load pair → register → draw ROIs → transfer → extract → save
    Verify both H5 files have matching ROI structure.
    """

def test_tandem_batch_from_directory():
    """Test batch creation and processing from real experiment directory."""
```

---

## 14. Open Questions

1. **Verification workflow** - Should verification be:
   - Automatic popup after every transfer?
   - Optional (default off)?
   - Only when offset exceeds threshold?

2. **Failed transfers** - When offset is too large or quality is poor:
   - Warn and continue?
   - Abort and require manual intervention?
   - Allow manual offset entry?

3. **Partial processing** - If batch processing fails on one pair:
   - Skip and continue?
   - Abort entire batch?
   - Mark as failed and continue?

---

## 15. Next Steps

1. [ ] Implement `TandemPair.load()` (depends on preprocessing.scanm)
2. [ ] Implement `TandemPair.compute_offset()` (depends on preprocessing.registration)
3. [ ] Implement `TandemPair.transfer_rois()`
4. [ ] Implement `TandemPair.draw_rois()` (integrate existing Napari tools)
5. [ ] Implement `TandemPair.extract_traces()`
6. [ ] Implement `TandemPair.save()` (depends on preprocessing.export)
7. [ ] Add verification viewer
8. [ ] Implement `TandemGroup`
9. [ ] Implement `TandemBatch`
10. [ ] Write tests
