"""Turn sample counts into seconds, using the recording's own timing.

Nothing in the GUI should put a frame or a sample index on an axis. A
frame number means nothing without the frame rate, and two recordings at
different rates cannot be compared by eye when both are labelled "Frame".
The recording already carries what is needed, so the conversion always
uses its attributes rather than anything the GUI computes for itself:

``frame_hz``
    Frames per second. Set at load from the ScanM header, and what
    :mod:`pygor.classes.core_data` uses for its own time axes (see the
    ``time_arr`` it builds for plotting raw traces).

``linedur_s``
    Duration of one scan line. Averages and snippets are upsampled to
    line precision by ``compute_snippets_and_averages``, so one sample of
    those arrays is one scan line, not one frame. ``Core`` makes the same
    assumption when it computes ``ms_dur``.

Both are absent on partially built objects, so every helper falls back to
plain indices and says so in the label it returns. A wrong axis is worse
than an honest one.
"""

import numpy as np

#: Label used whenever a real time base was available
SECONDS_LABEL = "Time (s)"


def frame_rate(recording):
    """Frames per second, or None when the recording cannot say."""
    hz = getattr(recording, "frame_hz", None)
    try:
        hz = float(hz)
    except (TypeError, ValueError):
        return None
    return hz if hz > 0 else None


def line_duration(recording):
    """Seconds per scan line, or None when the recording cannot say."""
    dur = getattr(recording, "linedur_s", None)
    try:
        dur = float(dur)
    except (TypeError, ValueError):
        return None
    return dur if dur > 0 else None


def trace_axis(recording, n_samples):
    """X values and label for an array sampled once per frame.

    Returns ``(x, label)``. Falls back to frame indices when the
    recording has no frame rate.
    """
    hz = frame_rate(recording)
    if hz is None:
        return np.arange(n_samples), "Frame"
    return np.arange(n_samples) / hz, SECONDS_LABEL


def average_axis(recording, n_samples):
    """X values and label for an upsampled average or snippet.

    Returns ``(x, label)``. Falls back to sample indices when the
    recording has no line duration.
    """
    dur = line_duration(recording)
    if dur is None:
        return np.arange(n_samples), "Sample"
    return np.arange(n_samples) * dur, SECONDS_LABEL


def frame_to_seconds(recording, frame):
    """Position of a frame on a :func:`trace_axis`.

    Returns the frame index unchanged when there is no frame rate, which
    is exactly what :func:`trace_axis` plotted in that case.
    """
    hz = frame_rate(recording)
    if hz is None:
        return frame
    return frame / hz


def frames_as_seconds(recording, n_frames):
    """Human-readable duration of a frame count, or None.

    For annotating parameters that pygor takes in frames, so the widget
    can say what the number means without changing what is stored.
    """
    hz = frame_rate(recording)
    if hz is None or n_frames is None:
        return None
    return f"{n_frames / hz:.2f} s at {hz:.1f} Hz"
