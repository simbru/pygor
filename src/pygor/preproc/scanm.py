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
import numpy as np
import datetime
import h5py

__all__ = [
    "read_smh_header",
    "read_smp_data",
    "load_scanm",
    "fix_light_artifact",
    "fill_light_artifact",
    "detrend_stack",
    "preprocess_stack",
    "detect_triggers",
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

        # Read all pixel data as uint16 (unsigned)
        # ScanM stores raw ADC values as unsigned 16-bit integers
        # Reading as int16 and using .view(uint16) gives identical results,
        # but reading directly as uint16 is cleaner and matches IGOR's wDataCh0
        raw_data = np.fromfile(f, dtype=np.uint16)

    # De-interleave channels using buffer blocks (VECTORIZED)
    # Data layout: [ch0_buf0, ch1_buf0, ch0_buf1, ch1_buf1, ...]
    # where each buffer chunk is pix_buffer pixels
    chunk_size = n_channels * pix_buffer
    
    # Truncate to complete chunks
    n_complete_chunks = len(raw_data) // chunk_size
    raw_truncated = raw_data[:n_complete_chunks * chunk_size]
    
    # Reshape to (n_chunks, n_channels, pix_buffer) for vectorized de-interleaving
    raw_reshaped = raw_truncated.reshape(n_complete_chunks, n_channels, pix_buffer)
    
    # Extract each channel by slicing (no Python loop needed)
    channel_data = {}
    for ch_idx, ch in enumerate(available_channels):
        # Flatten (n_chunks, pix_buffer) back to 1D
        channel_data[ch] = raw_reshaped[:, ch_idx, :].ravel()
    
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
        if ch in channel_data and len(channel_data[ch]) > 0:
            ch_data = channel_data[ch]  # Already 1D array from vectorized extraction
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


# -----------------------------------------------------------------------------
# Preprocessing: Light artifact and detrending
# -----------------------------------------------------------------------------

def fix_light_artifact(
    stack: np.ndarray,
    artifact_width: int = 2,
    flip_x: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Handle scanning light artifact by X-flipping the image.
    
    The leftmost pixels in each frame often contain a light artifact from the
    scanning laser turn-around. IGOR handles this by:
    1. X-flipping the non-artifact region only
    2. Later filling the artifact region (done after detrending)
    
    This function performs the X-flip. Use `fill_light_artifact()` after 
    any detrending to complete the artifact handling.
    
    Parameters
    ----------
    stack : np.ndarray
        3D array (frames, lines, width) of imaging data
    artifact_width : int, default=2
        Number of pixels at left edge affected by light artifact.
        IGOR default is 2 (OS_Parameters[%LightArtifact_cut]).
        Note: IGOR uses inclusive indexing, so artifact_width=2 means
        pixels 0, 1, 2 are affected (3 pixels total).
    flip_x : bool, default=True
        Whether to flip the image horizontally (IGOR does this).
        
    Returns
    -------
    result : np.ndarray
        Stack with X-flip applied (artifact region not yet filled)
    stack_ave : np.ndarray
        Mean image (lines, width) computed from the x-flipped data
        
    Notes
    -----
    IGOR's x-flip algorithm (OS_DetrendStack.ipf, line ~93):
    ``InputData[LightArtifactCut,nX-1][][]=wDataCh0[nX-1-(p-LightArtifactCut)][q][r]``
    
    This flips positions [artifact:] by mapping:
    - position artifact → source position nX-1
    - position artifact+1 → source position nX-2
    - etc.
    """
    n_frames, n_lines, n_width = stack.shape
    result = stack.astype(np.float32).copy()
    
    if flip_x and artifact_width < n_width:
        # IGOR-compatible X-flip: flip only [artifact_width:] region
        # Maps position p to source position (nX-1) - (p - artifact_width)
        for p in range(artifact_width, n_width):
            source_idx = n_width - 1 - (p - artifact_width)
            result[:, :, p] = stack[:, :, source_idx]
    
    # Compute Stack_Ave from x-flipped data (used for artifact fill later)
    stack_ave = result.mean(axis=0)  # (lines, width)
    
    return result, stack_ave


def fill_light_artifact(
    stack: np.ndarray,
    stack_ave: np.ndarray,
    artifact_width: int = 2,
) -> np.ndarray:
    """
    Fill the light artifact region with mean of non-artifact pixels.
    
    This should be called after detrending (or after x-flip if not detrending)
    to complete the artifact handling.
    
    Parameters
    ----------
    stack : np.ndarray
        3D array (frames, lines, width) - typically already x-flipped
    stack_ave : np.ndarray
        Mean image (lines, width) from x-flipped data
    artifact_width : int, default=2
        IGOR LightArtifact_cut parameter. With inclusive indexing,
        this fills pixels 0 through artifact_width (inclusive).
        
    Returns
    -------
    np.ndarray
        Stack with artifact region filled
        
    Notes
    -----
    IGOR fills artifact with mean of Stack_Ave[artifact_width+1:, :]:
    ``ImageStats/Q tempimage`` (where tempimage = Stack_Ave[LightArtifactCut:])
    ``InputData[0,LightArtifactCut][][]=V_Avg``
    """
    result = stack.copy()
    
    # IGOR: artifact region is [0, LightArtifact_cut] INCLUSIVE
    # So with artifact_width=2, we fill pixels 0, 1, 2 (indices 0:3 in Python)
    fill_width = artifact_width + 1
    
    # Compute fill value: mean of Stack_Ave for non-artifact region
    non_artifact_mean = stack_ave[:, fill_width:].mean()
    
    # Fill artifact region in all frames
    result[:, :, :fill_width] = non_artifact_mean
    
    return result


def detrend_stack(
    stack: np.ndarray,
    frame_rate: float,
    smooth_window_s: float = 1000.0,
    time_bin: int = 10,
) -> np.ndarray:
    """
    Remove slow baseline drift from imaging stack.

    For each pixel, subtracts a heavily smoothed version of its time course,
    then adds back the pixel's mean to preserve intensity scale.

    This matches IGOR's OS_DetrendStack algorithm:
    ``InputData -= SubtractionStack - Stack_Ave``

    Parameters
    ----------
    stack : np.ndarray
        3D array (frames, lines, width) of imaging data
    frame_rate : float
        Frame rate in Hz
    smooth_window_s : float, default=1000.0
        Smoothing window in seconds. IGOR default is 1000.
    time_bin : int, default=10
        Temporal binning factor for speed. IGOR default is 10.

    Returns
    -------
    np.ndarray
        Detrended stack (same shape as input)

    Notes
    -----
    IGOR's Smooth function (without /B flag) uses binomial (Gaussian) smoothing,
    not boxcar smoothing. The relationship between IGOR's smoothing iterations
    and Gaussian sigma is: sigma = sqrt(num / 2).

    IGOR's default edge handling is "bounce" (reflect), which mirrors values
    at the boundaries.

    References
    ----------
    IGOR Pro Smooth documentation
    Marchand, P., and L. Marmet, "Binomial smoothing filter",
    Rev. Sci. Instrum. 54(8), 1034-1041, 1983.
    """
    from scipy.ndimage import gaussian_filter1d

    n_frames, n_lines, n_width = stack.shape

    # Compute mean per pixel (to add back after subtracting baseline)
    pixel_mean = stack.mean(axis=0)  # Shape: (n_lines, n_width)

    # Calculate smoothing iterations (IGOR's 'num' parameter)
    # IGOR: Smoothingfactor = (Framerate * nSeconds_smooth) / nTimeBin
    smooth_iterations = int(frame_rate * smooth_window_s / time_bin)
    smooth_iterations = min(smooth_iterations, 2**15 - 1)  # IGOR limit
    smooth_iterations = max(smooth_iterations, 1)

    # Convert IGOR binomial iterations to Gaussian sigma
    # For binomial smoothing: sigma = sqrt(num / 2)
    sigma = np.sqrt(smooth_iterations / 2)

    # Temporal binning for speed (IGOR uses simple subsampling)
    binned_stack = stack[::time_bin, :, :]

    # Apply Gaussian (binomial) smoothing along time dimension
    # IGOR's Smooth uses binomial smoothing by default with 'bounce' (reflect) edge handling
    smoothed = gaussian_filter1d(binned_stack.astype(np.float32),
                                  sigma=sigma, axis=0, mode='reflect')

    # Upsample back to original frame count by repeating
    # (IGOR: SubtractionStack[p][q][r/nTimeBin])
    baseline = np.repeat(smoothed, time_bin, axis=0)[:n_frames]

    # Subtract baseline and add back mean
    # IGOR: InputData -= SubtractionStack - Stack_Ave
    result = stack.astype(np.float32) - baseline + pixel_mean

    # Clip to valid uint16 range (0-65535)
    # IGOR wraps negative values to unsigned, which creates misleading bright pixels.
    # Clipping to 0 is more analytically sound - negative values indicate
    # the baseline was overestimated at those locations.
    result = np.clip(result, 0, 65535)

    return result


def preprocess_stack(
    stack: np.ndarray,
    frame_rate: float,
    artifact_width: int = 2,
    flip_x: bool = True,
    detrend: bool = True,
    smooth_window_s: float = 1000.0,
    time_bin: int = 10,
    fix_first_frame: bool = True,
) -> np.ndarray:
    """
    Full preprocessing pipeline matching IGOR's OS_DetrendStack.
    
    Applies light artifact correction and optional detrending.
    
    Parameters
    ----------
    stack : np.ndarray
        3D array (frames, lines, width) of imaging data
    frame_rate : float
        Frame rate in Hz
    artifact_width : int, default=2
        Pixels to mask for light artifact. IGOR default is 2.
    flip_x : bool, default=True
        Whether to X-flip the image.
    detrend : bool, default=True
        Whether to apply temporal detrending.
    smooth_window_s : float, default=1000.0
        Detrending smooth window in seconds.
    time_bin : int, default=10
        Temporal binning for detrending speed.
    fix_first_frame : bool, default=True
        Copy frame 2 into frame 1 to fix first-frame artifact.
        
    Returns
    -------
    np.ndarray
        Preprocessed stack
        
    Notes
    -----
    IGOR preprocessing order (OS_DetrendStack.ipf):
    1. X-flip the non-artifact region
    2. Compute Stack_Ave (mean per pixel)
    3. Detrend (if not skipped): subtract smoothed baseline, add back Stack_Ave
    4. Fix frame 0 by copying from frame 1
    5. Fill artifact region with mean of non-artifact Stack_Ave
    """
    # Step 1: X-flip (artifact region initially just copied)
    result, stack_ave = fix_light_artifact(stack, artifact_width=artifact_width, flip_x=flip_x)
    
    # Step 2: Detrending (uses stack_ave internally)
    if detrend:
        result = detrend_stack(result, frame_rate, 
                               smooth_window_s=smooth_window_s, 
                               time_bin=time_bin)
    
    # Step 3: Fix first frame artifact (IGOR copies frame 2 to frame 1)
    # IGOR: InputData[][][0]=InputData[p][q][1]
    if fix_first_frame and result.shape[0] > 1:
        result[0] = result[1]
    
    # Step 4: Fill artifact region with mean of non-artifact Stack_Ave
    result = fill_light_artifact(result, stack_ave, artifact_width=artifact_width)
    
    return result


# -----------------------------------------------------------------------------
# Trigger detection
# -----------------------------------------------------------------------------

def detect_triggers(
    trigger_stack: np.ndarray,
    line_duration: float,
    trigger_threshold: int = 20000,
    min_gap_seconds: float = 0.1,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Detect stimulus triggers from trigger channel stack.
    
    Uses IGOR-compatible detection: triggers fire when signal drops below
    a threshold value (2^16 - trigger_threshold). This matches the algorithm
    in OS_TracesAndTriggers.ipf.
    
    Parameters
    ----------
    trigger_stack : np.ndarray
        3D array (frames, lines, width) from trigger channel
    line_duration : float
        Duration of one scan line in seconds
    trigger_threshold : int, default=20000
        Threshold parameter. Trigger fires when value < 2^16 - threshold.
        IGOR default is 20000 (OS_Parameters[%Trigger_Threshold]).
    min_gap_seconds : float, default=0.1
        Minimum time between triggers in seconds. Prevents re-triggering.
        IGOR default is 0.1 (OS_Parameters[%Trigger_after_skip_s]).
        
    Returns
    -------
    trigger_frames : np.ndarray
        1D array of frame indices where triggers occurred
    trigger_times : np.ndarray
        1D array of trigger times in seconds (with line precision)
    """
    n_frames, n_lines, _ = trigger_stack.shape
    
    # Use column 0 only (IGOR convention), flatten to 1D for vectorized ops
    # Shape: (n_frames * n_lines,)
    signal = trigger_stack[:, :, 0].ravel()
    
    # Threshold: trigger when signal drops below this value
    threshold_value = 2**16 - trigger_threshold
    
    # Find all samples below threshold (potential triggers)
    below_thresh = signal < threshold_value
    
    # Find falling edges: current sample is low AND previous was high
    # This is the trigger onset
    falling_edges = below_thresh & ~np.concatenate([[False], below_thresh[:-1]])
    
    # Get flat indices of all trigger onsets
    candidate_indices = np.where(falling_edges)[0]
    
    if len(candidate_indices) == 0:
        return np.array([], dtype=int), np.array([], dtype=float)
    
    # Minimum gap in samples (lines)
    min_gap_samples = int(min_gap_seconds / line_duration)
    
    # Filter by minimum gap - keep first, then only those far enough apart
    # This is the debounce logic (IGOR's "expectlow" + skip)
    kept_indices = [candidate_indices[0]]
    for idx in candidate_indices[1:]:
        if idx - kept_indices[-1] >= min_gap_samples:
            kept_indices.append(idx)
    
    kept_indices = np.array(kept_indices, dtype=int)
    
    # Convert flat indices to frame indices and times
    trigger_frames = kept_indices // n_lines
    trigger_lines = kept_indices % n_lines
    trigger_times = (trigger_frames * n_lines + trigger_lines) * line_duration
    
    return trigger_frames, trigger_times





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
            trigger_frames, trigger_times = detect_triggers(
                trigger_stack, 
                line_duration=self.linedur_s,
            )
            
            # Apply skip settings
            if skip_last_triggers > 0:
                trigger_frames = trigger_frames[skip_first_triggers:-skip_last_triggers]
                trigger_times = trigger_times[skip_first_triggers:-skip_last_triggers]
            else:
                trigger_frames = trigger_frames[skip_first_triggers:]
                trigger_times = trigger_times[skip_first_triggers:]
                
            self.triggertimes_frame = trigger_frames
            self.triggertimes = trigger_times  # Now has accurate line-precision times
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
        
        with h5py.File(output_path, "w") as f:
            # === Image data ===
            # Core expects (width, height, frames) - transposed from our (frames, height, width)
            # IGOR stores as uint16 (unsigned), matching raw ADC values
            images_t = self.images.transpose(2, 1, 0)
            f.create_dataset("wDataCh0_detrended", data=images_t, dtype=np.uint16)
            
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
