"""
Trigger signal preprocessing functions.

This module provides functions to detect and correct issues with TTL trigger
signals from ScanM recordings.
"""

import numpy as np

__all__ = ["correct_ttl_baseline"]


def correct_ttl_baseline(
    trigger_stack: np.ndarray,
    high_value: int | None = None,
    threshold_diff: int = 10000,
    baseline_threshold: int = 45000,
) -> tuple[np.ndarray, int]:
    """
    Correct TTL signal that starts in LOW state before actual triggers begin.

    Some recordings start with the TTL trigger signal already low (baseline ~2000)
    before the stimulus protocol begins. This causes trigger detection to register
    a ghost trigger at index 0. This function detects and corrects this condition
    by setting the baseline period to match the actual signal ceiling level.

    Parameters
    ----------
    trigger_stack : np.ndarray
        3D array (frames, lines, width) from trigger channel
    high_value : int, optional
        Value to set baseline frames to. If None (default), automatically
        detects the ceiling level from the actual TTL signal.
    threshold_diff : int, default=10000
        Minimum difference from baseline to detect transition to actual TTL pulses.
    baseline_threshold : int, default=45000
        If first frame mean is above this value, no correction is needed
        (signal already starts high).

    Returns
    -------
    corrected_stack : np.ndarray
        Corrected trigger stack (copy if correction applied, original if not)
    n_corrected : int
        Number of frames corrected (0 if no correction needed)

    Notes
    -----
    This mirrors the IGOR function CorrectTTLSignal which was written to handle
    the same issue. The algorithm:
    1. Check if first frame average is below baseline_threshold
    2. If yes, scan forward to find where TTL pulses actually start (big jump)
    3. Detect the ceiling level from the actual signal (95th percentile of high frames)
    4. Set all baseline frames (0 to baseline_end) to the detected ceiling

    Examples
    --------
    >>> # In preprocess(), this is called automatically:
    >>> trigger_images, n_corrected = correct_ttl_baseline(trigger_images)
    >>> if n_corrected > 0:
    ...     print(f"Corrected {n_corrected} baseline frames")
    """
    n_frames = trigger_stack.shape[0]

    # Check if first frame is already high (no correction needed)
    first_frame_avg = trigger_stack[0].mean()

    if first_frame_avg >= baseline_threshold:
        # Signal starts high, no correction needed
        return trigger_stack, 0

    # Signal starts low - find where the actual TTL pulses begin
    baseline = first_frame_avg
    baseline_end = -1

    for i in range(1, n_frames):
        frame_avg = trigger_stack[i].mean()

        # If we see a big jump from baseline, this is where TTL pulses start
        if abs(frame_avg - baseline) > threshold_diff:
            baseline_end = i - 1  # Last frame that was still baseline
            break

    if baseline_end == -1:
        # Could not find where TTL pulses start - no correction possible
        return trigger_stack, 0

    # Detect ceiling level from actual signal if not specified
    if high_value is None:
        # Look at frames after baseline to find the "high" level
        # Use 95th percentile of frame means to get ceiling (between trigger pulses)
        signal_region = trigger_stack[baseline_end + 1:]
        frame_means = signal_region.mean(axis=(1, 2))
        # The ceiling is the high state - use 95th percentile to be robust
        high_value = int(np.percentile(frame_means, 95))

    # Check if we're in the middle of a partial trigger at the boundary
    # If frames just after baseline_end are below the ceiling, extend the fill
    # to avoid creating a false trigger at the boundary
    fill_end = baseline_end
    trigger_threshold_value = 2**16 - 20000  # Standard trigger threshold

    for i in range(baseline_end + 1, min(baseline_end + 100, n_frames)):
        frame_avg = trigger_stack[i].mean()
        if frame_avg < trigger_threshold_value:
            # Still in a low state (partial trigger), extend fill
            fill_end = i
        else:
            # Reached high state, stop extending
            break

    # Apply correction: set baseline + any partial trigger frames to ceiling value
    corrected = trigger_stack.copy()
    corrected[:fill_end + 1] = high_value

    n_corrected = fill_end + 1
    if fill_end > baseline_end:
        print(f"TTL baseline correction: set frames 0-{fill_end} to {high_value} "
              f"(baseline avg was {baseline:.0f}, extended {fill_end - baseline_end} frames for partial trigger)")
    else:
        print(f"TTL baseline correction: set frames 0-{baseline_end} to {high_value} "
              f"(baseline avg was {baseline:.0f})")

    return corrected, n_corrected
