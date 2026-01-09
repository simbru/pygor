"""
ScanM file loading (SMP/SMH format).

This module provides functions to load ScanM microscopy data files directly
into Python, without requiring IGOR Pro as an intermediary.

File Format Reference
---------------------
ScanM produces paired files for each recording:
- .smh - Header file (UTF-16-LE encoded text, metadata)
- .smp - Pixel data file (binary)

The SMH header contains entries in format:
    type,variable_name=value;

The SMP file contains:
- Pixel data as 16-bit signed integers
- Channels are interleaved in buffer blocks (PixelBuffer_#0_Length pixels per block)
- Data layout: [ch0_buf0, ch1_buf0, ch0_buf1, ch1_buf1, ...]

Key header parameters for decoding:
- FrameWidth: Raw pixels per line (including retrace)
- FrameHeight: Number of lines per frame
- PixRetraceLen: Pixels for line retrace (to be cropped)
- XPixLineOffs: Line offset pixels (to be cropped from start)
- PixelBuffer_#0_Length: Buffer chunk size for channel interleaving
- InputChannelMask: Bitmask of active channels (e.g., 5 = channels 0,2)

Format specification derived from ScanM documentation,
ScM_FileIO.ipf (Thomas Euler, MPImF/Heidelberg, CIN/Uni Tübingen),
and eulerlab/processing_pypeline (Andre Chagas).
"""

from pathlib import Path
import struct
import numpy as np
import datetime
import h5py

__all__ = [
    "read_smh_header",
    "read_smp_data",
    "load_scanm",
    "to_pygor_data",
    "ScanMData",
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
    decode: bool = True,
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
    decode : bool, optional
        If True (default), crops retrace and line offset pixels from each line
        to produce the actual imaging data. If False, returns raw uncropped data.
        
        For XY scans, decoded width = FrameWidth - PixRetraceLen - XPixLineOffs.
        This matches what IGOR's ScM_FileIO produces.

    Returns
    -------
    dict[int, np.ndarray]
        Dictionary mapping channel index to 3D array (frames, height, width).
        If decode=True, width is the decoded imaging width.
        If decode=False, width is the raw FrameWidth from header.

    Examples
    --------
    >>> header = read_smh_header("recording.smh")
    >>> data = read_smp_data("recording.smp", header)
    >>> stack = data[0]  # Channel 0 (decoded)
    >>> print(stack.shape)
    (1000, 64, 128)  # Decoded dimensions
    
    >>> # Get raw data with retrace pixels
    >>> data_raw = read_smp_data("recording.smp", header, decode=False)
    >>> stack_raw = data_raw[0]
    >>> print(stack_raw.shape)
    (1000, 64, 200)  # Raw dimensions including retrace
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

    # Get pixel buffer length for de-interleaving
    # Channels are stored in blocks of pixBuffer length, NOT per-pixel interleaved
    # Reference: eulerlab/processing_pypeline/readScanM.py
    pix_buffer = (
        header.get("PixelBuffer_#0_Length") or
        header.get("uPixelBuffer_#0_Length") or
        frame_width * frame_height  # Fallback: one frame per buffer
    )

    # Read binary data
    with open(path, "rb") as f:
        # NO header offset - SMP files start with pixel data immediately
        f.seek(0)

        # Read all pixel data as int16
        raw_data = np.fromfile(f, dtype=np.int16)

    # De-interleave channels using buffer blocks
    # Data layout: [ch0_buf0, ch1_buf0, ch0_buf1, ch1_buf1, ...]
    # where each buffer chunk is pix_buffer pixels
    chunk_size = n_channels * pix_buffer
    
    # Initialize lists to collect channel data
    channel_data = {ch: [] for ch in available_channels}
    
    for i in range(0, len(raw_data), chunk_size):
        for ch_idx, ch in enumerate(available_channels):
            start = i + ch_idx * pix_buffer
            end = start + pix_buffer
            if end <= len(raw_data):
                channel_data[ch].append(raw_data[start:end])
    
    # Calculate decode parameters for cropping retrace/offset pixels
    # For XY scans: decoded_width = FrameWidth - PixRetraceLen - XPixLineOffs
    retrace_len = header.get("PixRetraceLen") or header.get("RtrcLen") or 0
    line_offset = header.get("XPixLineOffs") or header.get("LineOffSet") or 0
    decoded_width = frame_width - retrace_len - line_offset
    
    # Ensure decoded_width is sensible
    if decode and decoded_width <= 0:
        print(f"Warning: decoded_width={decoded_width} is invalid, falling back to raw data")
        decode = False

    result = {}
    pixels_per_frame = frame_width * frame_height
    
    for ch in channels_to_load:
        if ch in channel_data and channel_data[ch]:
            ch_data = np.concatenate(channel_data[ch])
            # Calculate actual number of complete frames
            actual_frames = len(ch_data) // pixels_per_frame
            ch_data = ch_data[:actual_frames * pixels_per_frame]
            # Reshape to (frames, height, raw_width)
            ch_data = ch_data.reshape(actual_frames, frame_height, frame_width)
            
            if decode:
                # Crop to imaging region: skip line offset, remove retrace
                # XPixLineOffs is the number of pixels to skip at the start
                ch_data = ch_data[:, :, line_offset:line_offset + decoded_width]
            
            result[ch] = ch_data

    return result


# -----------------------------------------------------------------------------
# Convenience function
# -----------------------------------------------------------------------------

def load_scanm(
    path: str | Path,
    channels: list[int] | None = None,
    decode: bool = True,
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
    decode : bool, optional
        If True (default), crops retrace and line offset pixels from each line
        to produce the actual imaging data. If False, returns raw uncropped data.
        
        For XY scans, decoded width = FrameWidth - PixRetraceLen - XPixLineOffs.

    Returns
    -------
    header : dict
        Parsed header metadata.
    data : dict[int, np.ndarray]
        Dictionary mapping channel index to 3D stack (frames, height, width).
        Width is decoded (cropped) if decode=True, raw if decode=False.

    Examples
    --------
    >>> header, data = load_scanm("0_0_SWN_200_White.smp")
    >>> print(f"Recording: {header.get('sOriginalPixelDataFileName', 'unknown')}")
    >>> print(f"Frames: {header['NumberOfFrames']}")
    >>> stack = data[0]  # Primary imaging channel (decoded)
    >>> print(f"Stack shape: {stack.shape}")  # e.g., (20291, 64, 128)

    >>> # Load raw data with retrace pixels
    >>> header, data = load_scanm("recording.smp", decode=False)
    >>> print(f"Raw shape: {data[0].shape}")  # e.g., (20291, 64, 200)
    """
    path = Path(path)

    # Determine base path (works with either .smp or .smh)
    if path.suffix.lower() in (".smp", ".smh"):
        smh_path = path.with_suffix(".smh")
        smp_path = path.with_suffix(".smp")
    else:
        raise ValueError(f"Expected .smp or .smh file, got: {path.suffix}")

    header = read_smh_header(smh_path)
    data = read_smp_data(smp_path, header, channels, decode=decode)

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
    
    Notes
    -----
    The header's FrameWidth already includes retrace and line offset pixels.
    So line duration = FrameWidth * pixel_duration (not adding retrace again).
    """
    # Get pixel duration in microseconds
    pixel_duration_us = header.get("RealPixelDuration_µs", header.get("TargetedPixelDuration_µs", 5.0))
    
    # Raw frame dimensions (FrameWidth includes retrace and line offset)
    frame_width_raw = header.get("FrameWidth", 200)
    frame_height = header.get("FrameHeight", images.shape[1])
    
    # Get decode parameters for reporting
    retrace_len = header.get("PixRetraceLen", 0)
    line_offset = header.get("XPixLineOffs", 0)
    decoded_width = frame_width_raw - retrace_len - line_offset
    
    # Line duration: raw FrameWidth already includes all pixels per line
    # Do NOT add retrace/offset again - they're already in FrameWidth
    line_duration_s = frame_width_raw * pixel_duration_us * 1e-6
    
    # Frame duration (lines * line_duration)
    frame_duration_s = frame_height * line_duration_s
    
    # Account for n_planes if doing volumetric imaging
    n_planes = header.get("dZPixels", 1)
    if n_planes == 0 or n_planes is None:
        n_planes = 1
    
    # Frame rate
    frame_hz = 1.0 / frame_duration_s if frame_duration_s > 0 else 0.0
    
    return {
        "line_duration_s": line_duration_s,
        "frame_duration_s": frame_duration_s,
        "frame_hz": frame_hz,
        "n_planes": n_planes,
        "decoded_width": decoded_width,
        "raw_width": frame_width_raw,
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

    def export_to_h5(
        self,
        output_path: str | Path | None = None,
        overwrite: bool = False,
    ) -> Path:
        """
        Export data to IGOR-compatible H5 file for loading as pygor.classes.Core.
        
        Parameters
        ----------
        output_path : str or Path, optional
            Output H5 file path. If None, uses same name as source with .h5 extension.
        overwrite : bool, optional
            If True, overwrite existing file. Default False.
            
        Returns
        -------
        Path
            Path to the created H5 file.
            
        Examples
        --------
        >>> data = ScanMData("recording.smp")
        >>> data.set_rois(my_rois)  # Optional preprocessing
        >>> h5_path = data.export_to_h5("recording.h5")
        >>> 
        >>> # Now load as Core with all methods
        >>> import pygor
        >>> core_data = pygor.load(pygor.classes.Core, h5_path)
        """
        if output_path is None:
            output_path = self.filename.with_suffix(".h5")
        else:
            output_path = Path(output_path)
        
        if output_path.exists() and not overwrite:
            raise FileExistsError(
                f"File already exists: {output_path}. Use overwrite=True to replace."
            )
        
        # Get header for metadata
        header = self._header
        
        with h5py.File(output_path, "w") as f:
            # === Image data ===
            # Core expects (width, height, frames) - transposed from our (frames, height, width)
            images_t = self.images.transpose(2, 1, 0)
            f.create_dataset("wDataCh0_detrended", data=images_t, dtype=np.int16)
            
            # Average stack - Core expects (width, height) transposed
            f.create_dataset("Stack_Ave", data=self.average_stack.T, dtype=np.float32)
            
            # === ROIs ===
            if self.rois is not None:
                # Core expects transposed ROI mask
                f.create_dataset("ROIs", data=self.rois.T, dtype=np.int16)
                
            if self.roi_sizes is not None:
                f.create_dataset("RoiSizes", data=self.roi_sizes, dtype=np.int32)
            
            # === Traces ===
            if self.traces_raw is not None:
                # Core expects (frames, rois) - transposed from our (rois, frames)
                f.create_dataset("Traces0_raw", data=self.traces_raw.T, dtype=np.float32)
                
            if self.traces_znorm is not None:
                f.create_dataset("Traces0_znorm", data=self.traces_znorm.T, dtype=np.float32)
            
            # === Trigger times ===
            # Pad with NaNs to fixed size (Core expects this)
            max_triggers = max(len(self.triggertimes_frame), 1000)
            triggertimes = np.full(max_triggers, np.nan)
            triggertimes[:len(self.triggertimes)] = self.triggertimes
            f.create_dataset("Triggertimes", data=triggertimes, dtype=np.float64)
            
            triggertimes_frame = np.full(max_triggers, np.nan)
            triggertimes_frame[:len(self.triggertimes_frame)] = self.triggertimes_frame
            f.create_dataset("Triggertimes_Frame", data=triggertimes_frame, dtype=np.float64)
            
            # === wParamsStr (date/time metadata) ===
            # Core expects specific format: index 4 = date, index 5 = time
            exp_date = self.metadata["exp_date"]
            exp_time = self.metadata["exp_time"]
            date_str = f"{exp_date.year}-{exp_date.month:02d}-{exp_date.day:02d}"
            time_str = f"{exp_time.hour:02d}-{exp_time.minute:02d}-{exp_time.second:02d}-00"
            
            # Create wParamsStr array (minimum 6 elements, with date at 4, time at 5)
            params_str = [""] * 10
            params_str[4] = date_str
            params_str[5] = time_str
            params_str[0] = str(self.filename.stem)  # Recording name
            
            # Encode as bytes for h5py
            dt = h5py.special_dtype(vlen=str)
            params_str_ds = f.create_dataset("wParamsStr", (len(params_str),), dtype=dt)
            for i, s in enumerate(params_str):
                params_str_ds[i] = s.encode("utf-8")
            
            # === wParamsNum (XYZ position, etc.) ===
            # Core reads XYZ from indices 26, 27, 28
            params_num = np.zeros(50, dtype=np.float64)
            xyz = self.metadata.get("objectiveXYZ", (0, 0, 0))
            params_num[26] = xyz[0]  # X
            params_num[27] = xyz[2]  # Z (swapped in Core)
            params_num[28] = xyz[1]  # Y
            f.create_dataset("wParamsNum", data=params_num, dtype=np.float64)
            
            # === OS_Parameters (timing, trigger settings) ===
            # Create OS_Parameters dataset with attributes
            os_params_keys = [
                "placeholder",  # Index 0 is skipped in Core
                "LineDuration",
                "nPlanes", 
                "Trigger_Mode",
                "Skip_First_Triggers",
                "Skip_Last_Triggers",
            ]
            os_params_values = np.array([
                0,  # placeholder
                self.linedur_s,
                self.n_planes,
                self.trigger_mode,
                0,  # skip_first (applied already)
                0,  # skip_last (applied already)
            ], dtype=np.float64)
            
            os_params_ds = f.create_dataset("OS_Parameters", data=os_params_values)
            # Store keys as attribute (Core reads keys from here)
            os_params_ds.attrs["OS_Parameters"] = np.array(
                [b"Keys"] + [k.encode() for k in os_params_keys], 
                dtype=object
            )
            
            # === Optional: averages, snippets ===
            if self.averages is not None:
                f.create_dataset("Averages0", data=self.averages.T, dtype=np.float32)
                
            if self.snippets is not None:
                f.create_dataset("Snippets0", data=self.snippets.T, dtype=np.float32)
                
            if self.ipl_depths is not None:
                f.create_dataset("Positions", data=self.ipl_depths, dtype=np.float64)
                
            if self.correlation_projection is not None:
                f.create_dataset("correlation_projection", data=self.correlation_projection.T, dtype=np.float32)
                
            if self.quality_indices is not None:
                f.create_dataset("QualityCriterion", data=self.quality_indices, dtype=np.float64)
        
        print(f"Exported to: {output_path}")
        return output_path


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
