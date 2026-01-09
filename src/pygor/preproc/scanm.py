"""
ScanM file loading (SMP/SMH format).

This module provides functions to load ScanM microscopy data files directly
into Python, without requiring IGOR Pro as an intermediary.

File Format Reference
---------------------
ScanM produces paired files for each recording:
- .smh - Header file (text, metadata)
- .smp - Pixel data file (binary)

The SMH header is latin-1 encoded text with entries in format:
    type,variable_name=value;

The SMP file contains:
- 64-byte pre-header (8 x uint64)
- Pixel data as 16-bit signed integers, multi-channel interleaved

Format specification derived from ScanM documentation and
ScM_FileIO.ipf (Thomas Euler, MPImF/Heidelberg, CIN/Uni Tübingen).
"""

from pathlib import Path
import struct
import numpy as np
import datetime

__all__ = [
    "read_smh_header",
    "read_smp_data",
    "load_scanm",
    "to_pygor_data",
]


# -----------------------------------------------------------------------------
# Header parsing
# -----------------------------------------------------------------------------

def read_smh_header(path: str | Path) -> dict:
    """
    Parse SMH header file to dictionary.

    Parameters
    ----------
    path : str or Path
        Path to .smh file. If .smp path is given, will find the .smh partner.

    Returns
    -------
    dict
        Header key-value pairs with appropriate type conversion.
        Keys are cleaned (prefix stripped), values are converted to
        int, float, or str as appropriate.

    Examples
    --------
    >>> header = read_smh_header("recording.smh")
    >>> print(header["FrameWidth"], header["FrameHeight"])
    512 512
    >>> print(header["NumberOfFrames"])
    1000
    """
    path = Path(path)

    # Handle .smp input - find .smh partner
    if path.suffix.lower() == ".smp":
        path = path.with_suffix(".smh")

    if not path.exists():
        raise FileNotFoundError(f"Header file not found: {path}")

    header = {}

    # Try reading as UTF-16-LE first (common for ScanM files on Windows)
    # Fall back to latin-1 if that fails
    try:
        with open(path, "r", encoding="utf-16-le") as f:
            content = f.read()
    except UnicodeDecodeError:
        # Fall back to latin-1 and skip every other character
        # (handles mixed binary/text format from older ScanM versions)
        with open(path, "r", encoding="latin-1") as f:
            raw_content = f.read()
        # Skip null bytes that appear in UTF-16-like encoding
        content = raw_content[1:-1:2] if raw_content else ""

    # Parse entries: "type,key=value;"
    # Example: "UINT32,uFrameWidth=512;"
    entries = content.split(";")

    for entry in entries:
        entry = entry.strip()
        if not entry or "=" not in entry:
            continue

        try:
            # Split "type,key=value"
            type_and_key, value = entry.split("=", 1)

            if "," in type_and_key:
                type_str, key = type_and_key.split(",", 1)
            else:
                type_str = "String"
                key = type_and_key

            key = key.strip()
            value = value.strip()
            type_str = type_str.strip().upper()

            # Convert value based on type
            if type_str in ("UINT32", "UINT64", "INT32", "INT64"):
                header[key] = int(value)
            elif type_str in ("REAL32", "REAL64", "FLOAT"):
                header[key] = float(value)
            else:
                header[key] = value

        except ValueError:
            # Skip malformed entries
            continue

    return header


def _get_channel_count(input_channel_mask: int) -> int:
    """Count number of active channels from bitmask."""
    return bin(input_channel_mask).count("1")


def _get_active_channels(input_channel_mask: int) -> list[int]:
    """Get list of active channel indices from bitmask."""
    channels = []
    for i in range(32):  # Max 32 channels
        if input_channel_mask & (1 << i):
            channels.append(i)
    return channels


# -----------------------------------------------------------------------------
# Binary data reading
# -----------------------------------------------------------------------------

def read_smp_data(
    path: str | Path,
    header: dict | None = None,
    channels: list[int] | None = None,
) -> dict[int, np.ndarray]:
    """
    Read SMP binary pixel data.

    Parameters
    ----------
    path : str or Path
        Path to .smp file.
    header : dict, optional
        Parsed header from read_smh_header(). If None, will read from
        partner .smh file.
    channels : list of int, optional
        Which channels to load. If None, loads all available channels.

    Returns
    -------
    dict[int, np.ndarray]
        Dictionary mapping channel index to 3D array (frames, height, width).

    Examples
    --------
    >>> header = read_smh_header("recording.smh")
    >>> data = read_smp_data("recording.smp", header)
    >>> stack = data[0]  # Channel 0
    >>> print(stack.shape)
    (1000, 512, 512)
    """
    path = Path(path)

    # Handle .smh input - find .smp partner
    if path.suffix.lower() == ".smh":
        path = path.with_suffix(".smp")

    if not path.exists():
        raise FileNotFoundError(f"Data file not found: {path}")

    # Read header if not provided
    if header is None:
        header = read_smh_header(path)

    # Extract dimensions from header - try multiple possible key names
    frame_width = (
        header.get("uFrameWidth") or
        header.get("FrameWidth") or
        header.get("udxFrDecoded")  # Decoded frame width
    )
    frame_height = (
        header.get("uFrameHeight") or
        header.get("FrameHeight") or
        header.get("udyFrDecoded")  # Decoded frame height
    )
    n_frames = (
        header.get("uNumberOfFrames") or
        header.get("NumberOfFrames") or
        header.get("uFrameCounter")
    )
    input_mask = (
        header.get("uInputChannelMask") or
        header.get("InputChannelMask") or
        1
    )

    if frame_width is None or frame_height is None:
        # Debug: show available keys
        dimension_keys = [k for k in header.keys() if any(
            x in k.lower() for x in ['width', 'height', 'frame', 'dx', 'dy', 'size']
        )]
        raise ValueError(
            f"Could not determine frame dimensions from header. "
            f"Dimension-related keys found: {dimension_keys}. "
            # f"All keys: {list(header.keys())[:20]}..."
        )
    if n_frames is None:
        raise ValueError("Could not determine number of frames from header")

    # Determine which channels are available and which to load
    available_channels = _get_active_channels(input_mask)
    n_channels = len(available_channels)

    if channels is None:
        channels_to_load = available_channels
    else:
        channels_to_load = [ch for ch in channels if ch in available_channels]
        if not channels_to_load:
            raise ValueError(
                f"None of requested channels {channels} are available. "
                f"Available: {available_channels}"
            )

    # Read binary data
    with open(path, "rb") as f:
        # Skip pre-header (64 bytes = 8 x uint64)
        f.seek(64)

        # Read all pixel data as int16
        # Data is interleaved: all channels for pixel 0, then pixel 1, etc.
        pixels_per_frame = frame_width * frame_height
        total_samples = pixels_per_frame * n_frames * n_channels

        raw_data = np.fromfile(f, dtype=np.int16, count=total_samples)

    if len(raw_data) < total_samples:
        # Adjust n_frames if file is shorter than expected
        actual_samples = len(raw_data)
        n_frames = actual_samples // (pixels_per_frame * n_channels)
        raw_data = raw_data[:n_frames * pixels_per_frame * n_channels]

    # Reshape and de-interleave channels
    # Data layout: [ch0_px0, ch1_px0, ch2_px0, ch0_px1, ch1_px1, ch2_px1, ...]
    raw_data = raw_data.reshape(-1, n_channels)  # (total_pixels, n_channels)

    result = {}
    for i, ch_idx in enumerate(available_channels):
        if ch_idx in channels_to_load:
            ch_data = raw_data[:, i]  # Extract this channel
            # Reshape to (frames, height, width)
            ch_data = ch_data.reshape(n_frames, frame_height, frame_width)
            result[ch_idx] = ch_data

    return result


# -----------------------------------------------------------------------------
# Convenience function
# -----------------------------------------------------------------------------

def load_scanm(
    path: str | Path,
    channels: list[int] | None = None,
) -> tuple[dict, dict[int, np.ndarray]]:
    """
    Load SMP/SMH file pair.

    Convenience function that loads header and data together.

    Parameters
    ----------
    path : str or Path
        Path to either .smp or .smh file. Will find the partner file.
    channels : list of int, optional
        Which channels to load. If None, loads all available channels.

    Returns
    -------
    header : dict
        Parsed header metadata.
    data : dict[int, np.ndarray]
        Dictionary mapping channel index to 3D stack (frames, height, width).

    Examples
    --------
    >>> header, data = load_scanm("0_0_SWN_200_White.smp")
    >>> print(f"Recording: {header.get('sOriginalPixelDataFileName', 'unknown')}")
    >>> print(f"Frames: {header['uNumberOfFrames']}")
    >>> print(f"Dimensions: {header['uFrameWidth']} x {header['uFrameHeight']}")
    >>>
    >>> stack = data[0]  # Primary imaging channel
    >>> print(f"Stack shape: {stack.shape}")

    >>> # Load only channel 0
    >>> header, data = load_scanm("recording.smp", channels=[0])
    """
    path = Path(path)

    # Determine base path (works with either .smp or .smh)
    if path.suffix.lower() in (".smp", ".smh"):
        smh_path = path.with_suffix(".smh")
        smp_path = path.with_suffix(".smp")
    else:
        raise ValueError(f"Expected .smp or .smh file, got: {path.suffix}")

    header = read_smh_header(smh_path)
    data = read_smp_data(smp_path, header, channels)

    return header, data


# -----------------------------------------------------------------------------
# Pygor object creation
# -----------------------------------------------------------------------------

def _parse_scanm_datetime(header: dict) -> tuple[datetime.date, datetime.time]:
    """Parse date and time from ScanM header."""
    date_str = header.get("DateStamp", "1970-01-01")
    time_str = header.get("TimeStamp", "00-00-00-00")
    
    try:
        date_parts = date_str.split("-")
        date = datetime.date(int(date_parts[0]), int(date_parts[1]), int(date_parts[2]))
    except (ValueError, IndexError):
        date = datetime.date(1970, 1, 1)
    
    try:
        time_parts = time_str.split("-")
        time = datetime.time(int(time_parts[0]), int(time_parts[1]), int(time_parts[2]))
    except (ValueError, IndexError):
        time = datetime.time(0, 0, 0)
    
    return date, time


def _compute_timing_params(header: dict, images: np.ndarray) -> dict:
    """
    Compute timing parameters from ScanM header.
    
    Returns dict with:
    - line_duration_s: time per line (seconds)
    - frame_duration_s: time per frame (seconds)
    - frame_hz: frame rate (Hz)
    """
    # Get pixel duration in microseconds
    pixel_duration_us = header.get("RealPixelDuration_µs", header.get("TargetedPixelDuration_µs", 5.0))
    
    # Frame dimensions
    frame_width = header.get("FrameWidth", images.shape[2])
    frame_height = header.get("FrameHeight", images.shape[1])
    
    # Retrace length (dead pixels at end of each line)
    retrace_len = header.get("PixRetraceLen", 0)
    line_offset = header.get("XPixLineOffs", 0)
    
    # Total pixels per line including retrace
    total_pixels_per_line = frame_width + retrace_len + line_offset
    
    # Line duration in seconds
    line_duration_s = total_pixels_per_line * pixel_duration_us * 1e-6
    
    # Frame duration (lines * line_duration)
    # Account for n_planes if doing volumetric imaging
    n_planes = header.get("dZPixels", 1)
    if n_planes == 0:
        n_planes = 1
        
    frame_duration_s = frame_height * line_duration_s
    
    # Frame rate
    frame_hz = 1.0 / frame_duration_s if frame_duration_s > 0 else 0.0
    
    return {
        "line_duration_s": line_duration_s,
        "frame_duration_s": frame_duration_s,
        "frame_hz": frame_hz,
        "n_planes": n_planes,
    }


def _detect_triggers_simple(
    trigger_stack: np.ndarray,
    threshold: float | None = None,
    min_separation: int = 5,
) -> np.ndarray:
    """
    Simple trigger detection from trigger channel stack.
    
    Parameters
    ----------
    trigger_stack : np.ndarray
        3D array (frames, height, width) from trigger channel
    threshold : float, optional
        Detection threshold. If None, auto-computed as mean + 3*std
    min_separation : int
        Minimum frames between triggers
        
    Returns
    -------
    trigger_frames : np.ndarray
        1D array of frame indices where triggers occurred
    """
    # Downsample to 1D trace (max of each frame)
    trace = trigger_stack.max(axis=(1, 2))
    
    # Auto-threshold
    if threshold is None:
        baseline = np.percentile(trace, 10)
        peak = np.percentile(trace, 99)
        threshold = baseline + 0.5 * (peak - baseline)
    
    # Find frames above threshold
    above_thresh = trace > threshold
    
    # Find rising edges
    trigger_frames = []
    last_trigger = -min_separation
    
    for i, is_trigger in enumerate(above_thresh):
        if is_trigger and (i - last_trigger) >= min_separation:
            trigger_frames.append(i)
            last_trigger = i
    
    return np.array(trigger_frames, dtype=int)


class ScanMData:
    """
    Pygor-compatible data object created from ScanM files.
    
    This class provides the same interface as pygor.classes.Core but
    is initialized directly from ScanM SMP/SMH files without requiring
    an intermediate H5 file.
    
    Parameters
    ----------
    path : str or Path
        Path to .smp or .smh file
    imaging_channel : int, optional
        Channel index for imaging data (default: 0)
    trigger_channel : int, optional
        Channel index for trigger detection (default: 2)
    skip_first_triggers : int, optional
        Number of initial triggers to skip (default: 0)
    skip_last_triggers : int, optional
        Number of final triggers to skip (default: 0)
    trigger_mode : int, optional
        Trigger detection mode (default: 1)
        
    Attributes
    ----------
    filename : Path
        Path to the source file
    images : np.ndarray
        3D imaging stack (frames, height, width)
    metadata : dict
        Recording metadata
    frame_hz : float
        Frame rate in Hz
    linedur_s : float
        Line duration in seconds
    triggertimes_frame : np.ndarray
        Frame indices of detected triggers
    rois : np.ndarray or None
        ROI mask (initially None, can be set later)
    traces_raw : np.ndarray or None
        Raw traces (initially None, computed after ROIs set)
    traces_znorm : np.ndarray or None
        Z-normalized traces
        
    Examples
    --------
    >>> data = ScanMData("recording.smp")
    >>> print(data.images.shape)
    (1000, 64, 200)
    >>> print(f"Frame rate: {data.frame_hz:.2f} Hz")
    Frame rate: 7.81 Hz
    """
    
    def __init__(
        self,
        path: str | Path,
        imaging_channel: int = 0,
        trigger_channel: int = 2,
        skip_first_triggers: int = 0,
        skip_last_triggers: int = 0,
        trigger_mode: int = 1,
    ):
        self.filename = Path(path)
        self.type = self.__class__.__name__
        
        # Load data
        channels_to_load = list(set([imaging_channel, trigger_channel]))
        header, channel_data = load_scanm(path, channels=channels_to_load)
        
        # Store header for reference
        self._header = header
        
        # Get actual number of frames recorded
        # (NumberOfFrames is max capacity, FrameCounter counts down)
        n_frames_total = header.get("NumberOfFrames", 0)
        frame_counter = header.get("FrameCounter", 0)
        stim_buf_per_fr = header.get("StimBufPerFr", 1)
        actual_frames = (n_frames_total - frame_counter) * stim_buf_per_fr
        
        # Get imaging data
        if imaging_channel in channel_data:
            self.images = channel_data[imaging_channel][:actual_frames]
        else:
            raise ValueError(f"Imaging channel {imaging_channel} not found in data")
        
        # Compute timing parameters
        timing = _compute_timing_params(header, self.images)
        self.frame_hz = timing["frame_hz"]
        self.linedur_s = timing["line_duration_s"]
        self.n_planes = timing["n_planes"]
        
        # Parse metadata
        exp_date, exp_time = _parse_scanm_datetime(header)
        self.metadata = {
            "filename": str(self.filename),
            "exp_date": exp_date,
            "exp_time": exp_time,
            "objectiveXYZ": (
                header.get("XCoord_um", 0),
                header.get("YCoord_um", 0),
                header.get("ZCoord_um", 0),
            ),
            "scanm_header": header,  # Keep full header for reference
        }
        
        # Detect triggers
        if trigger_channel in channel_data:
            trigger_stack = channel_data[trigger_channel][:actual_frames]
            all_triggers = _detect_triggers_simple(trigger_stack)
            
            # Apply skip settings
            if skip_last_triggers > 0:
                all_triggers = all_triggers[skip_first_triggers:-skip_last_triggers]
            else:
                all_triggers = all_triggers[skip_first_triggers:]
                
            self.triggertimes_frame = all_triggers
            self.triggertimes = all_triggers.astype(float)  # For compatibility
        else:
            self.triggertimes_frame = np.array([], dtype=int)
            self.triggertimes = np.array([], dtype=float)
        
        self.trigger_mode = trigger_mode
        
        # Initialize ROI-related attributes (to be filled later)
        self.rois = None
        self.num_rois = 0
        self.roi_sizes = None
        self.traces_raw = None
        self.traces_znorm = None
        self.averages = None
        self.snippets = None
        self.ms_dur = None
        self.quality_indices = None
        
        # Compute average stack
        self.average_stack = self.images.mean(axis=0)
        self.correlation_projection = None  # Can be computed on demand
        
        # IPL depths (for multi-plane recordings)
        self.ipl_depths = None
        
        # Set name
        self.name = self.filename.stem
    
    def __repr__(self):
        date = self.metadata["exp_date"].strftime("%d-%m-%Y")
        return f"{date}:{self.__class__.__name__}:{self.filename.stem}"
    
    @property
    def frametime_ms(self):
        """Time array in milliseconds for each frame."""
        time_arr = np.arange(self.images.shape[0]) / self.frame_hz * 1000
        return time_arr
    
    def set_rois(self, rois: np.ndarray):
        """
        Set ROI mask and extract traces.
        
        Parameters
        ----------
        rois : np.ndarray
            2D ROI mask where background=1 and ROIs are negative integers
            (-1, -2, -3, ...)
        """
        self.rois = rois
        
        # Ensure background is 1 (pygor convention)
        if np.any(self.rois == 0):
            self.rois[self.rois == 0] = 1
            
        # Count ROIs (unique negative values)
        roi_ids = np.unique(self.rois[self.rois < 0])
        self.num_rois = len(roi_ids)
        
        # Extract traces
        self._extract_traces()
    
    def _extract_traces(self):
        """Extract mean traces from each ROI."""
        if self.rois is None or self.num_rois == 0:
            return
            
        roi_ids = np.unique(self.rois[self.rois < 0])
        n_frames = self.images.shape[0]
        
        traces = np.zeros((self.num_rois, n_frames))
        roi_sizes = []
        
        for i, roi_id in enumerate(roi_ids):
            mask = self.rois == roi_id
            roi_sizes.append(np.sum(mask))
            for t in range(n_frames):
                traces[i, t] = self.images[t][mask].mean()
        
        self.traces_raw = traces
        self.roi_sizes = np.array(roi_sizes)
        
        # Z-normalize traces
        self.traces_znorm = (traces - traces.mean(axis=1, keepdims=True)) / traces.std(axis=1, keepdims=True)


def to_pygor_data(
    path: str | Path,
    imaging_channel: int = 0,
    trigger_channel: int = 2,
    **kwargs,
) -> ScanMData:
    """
    Load ScanM file and return a pygor-compatible data object.
    
    This is a convenience function that creates a ScanMData object,
    which provides the same interface as pygor.classes.Core.
    
    Parameters
    ----------
    path : str or Path
        Path to .smp or .smh file
    imaging_channel : int, optional
        Channel index for imaging data (default: 0)
    trigger_channel : int, optional
        Channel index for trigger detection (default: 2)
    **kwargs
        Additional arguments passed to ScanMData constructor
        
    Returns
    -------
    ScanMData
        Pygor-compatible data object
        
    Examples
    --------
    >>> import pygor.preproc as preproc
    >>> data = preproc.to_pygor_data("recording.smp")
    >>> print(f"Loaded {data.images.shape[0]} frames at {data.frame_hz:.2f} Hz")
    >>> print(f"Detected {len(data.triggertimes_frame)} triggers")
    """
    return ScanMData(
        path,
        imaging_channel=imaging_channel,
        trigger_channel=trigger_channel,
        **kwargs,
    )
