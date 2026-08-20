"""Stimulus triggers: where they are, and which of them get averaged.

``triggertimes`` is in seconds from the start of the recording, which is
the same axis :mod:`pygor.gui.timebase` puts traces on, so trigger marks
can be drawn straight onto a trace without conversion.

Which triggers actually contribute to an average is decided by three
numbers on the recording, and they are easy to get wrong by eye:

``trigger_mode``
    Triggers per stimulus loop. One loop is one repetition, so this is
    the number of triggers each averaged snippet spans.

``_Core__skip_first_frames`` / ``_Core__skip_last_frames``
    Triggers ignored at each end, set at load. The last one is stored
    negative or zero, matching how ``compute_snippets_and_averages``
    adds rather than subtracts it.

The loop arithmetic here is deliberately the same as that function's, so
what the GUI shows is what an average would actually be built from.
"""

import numpy as np


def trigger_times(recording):
    """Trigger times in seconds, or None when the recording has none."""
    times = getattr(recording, "triggertimes", None)
    if times is None:
        return None
    times = np.asarray(times, dtype=float).ravel()
    return times if times.size else None


def trigger_frames(recording):
    """Frame index of each trigger, or None."""
    frames = getattr(recording, "triggertimes_frame", None)
    if frames is None:
        return None
    frames = np.asarray(frames).ravel()
    return frames if frames.size else None


def triggers_per_loop(recording):
    """How many triggers one averaged repetition spans. At least 1."""
    try:
        mode = int(getattr(recording, "trigger_mode", 1))
    except (TypeError, ValueError):
        return 1
    return max(1, mode)


def skipped(recording):
    """(first, last) trigger counts ignored at each end, both positive."""
    first = getattr(recording, "_Core__skip_first_frames", 0) or 0
    last = getattr(recording, "_Core__skip_last_frames", 0) or 0
    return int(first), abs(int(last))


def loop_count(recording, mode=None):
    """Complete loops available, by the same arithmetic as averaging.

    ``mode`` overrides the recording's ``trigger_mode``, for previewing
    what a different setting would give without writing it first.
    """
    times = trigger_times(recording)
    if times is None:
        return 0
    mode = triggers_per_loop(recording) if mode is None else max(1, int(mode))
    first, last = skipped(recording)
    valid = times.size - first - last
    return max(0, valid // mode)


def loop_start_times(recording, mode=None):
    """Time of the trigger that opens each complete loop."""
    times = trigger_times(recording)
    if times is None:
        return np.empty(0)
    mode = triggers_per_loop(recording) if mode is None else max(1, int(mode))
    first, _ = skipped(recording)
    n_loops = loop_count(recording, mode)
    starts = first + np.arange(n_loops) * mode
    return times[starts] if n_loops else np.empty(0)


def within_loop_times(recording, mode=None):
    """Trigger offsets inside one loop, measured from its first trigger.

    These are the marks that belong on an averaged snippet, whose x axis
    restarts at zero for every repetition. Taken from the first complete
    loop, so an uneven trigger train shows its real spacing rather than
    an assumed one.
    """
    times = trigger_times(recording)
    if times is None:
        return np.empty(0)
    mode = triggers_per_loop(recording) if mode is None else max(1, int(mode))
    if loop_count(recording, mode) == 0:
        return np.empty(0)
    first, _ = skipped(recording)
    loop = times[first : first + mode]
    return loop - loop[0]


def trigger_table(recording, mode=None):
    """One row per trigger, in the order they were recorded.

    Each row is ``(index, time_s, interval_s, frame, loop, used)`` where
    ``interval_s`` is the gap from the previous trigger (NaN for the
    first), ``loop`` is the repetition the trigger belongs to or None,
    and ``used`` says whether an average would include it.
    """
    times = trigger_times(recording)
    if times is None:
        return []

    mode = triggers_per_loop(recording) if mode is None else max(1, int(mode))
    first, _ = skipped(recording)
    n_loops = loop_count(recording, mode)
    last_used = first + n_loops * mode

    frames = trigger_frames(recording)
    intervals = np.diff(times, prepend=np.nan)

    rows = []
    for i, (time, interval) in enumerate(zip(times, intervals)):
        used = first <= i < last_used
        rows.append(
            {
                "index": i,
                "time_s": float(time),
                "interval_s": float(interval),
                "frame": None if frames is None or i >= frames.size else int(frames[i]),
                "loop": (i - first) // mode if used else None,
                "used": used,
            }
        )
    return rows


def summary(recording, mode=None):
    """One line describing what the current trigger settings produce."""
    times = trigger_times(recording)
    if times is None:
        return "No triggers on this recording"

    mode = triggers_per_loop(recording) if mode is None else max(1, int(mode))
    first, last = skipped(recording)
    n_loops = loop_count(recording, mode)
    dropped = times.size - first - last - n_loops * mode

    parts = [f"{times.size} triggers", f"{mode} per loop", f"{n_loops} complete loops"]
    if first or last:
        parts.append(f"{first} skipped at the start, {last} at the end")
    if dropped > 0:
        parts.append(f"{dropped} left over")
    return ", ".join(parts)
