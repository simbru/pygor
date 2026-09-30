"""What counts as a recording on disk, and how to open one.

Three places need the same answer and did not have one: ``pygor-tui``'s path
argument, the file browser, and the napari hand-off. The hand-off only ever
called ``load_object``, so it could open a saved ``.recording.h5`` and nothing
else -- pointing it at the raw file the object came from raised ``No recording
groups found``. One reader, used by all three, so what the browser is willing
to list is exactly what the other two are willing to open.
"""

from __future__ import annotations

import pathlib

# A saved pygor object. Checked before the raw suffixes, since its name ends
# in ".h5" too and the two are loaded by completely different code paths.
SAVED_SUFFIX = ".recording.h5"

# IGOR exports and ScanM raw. ".smh" is the header half of a ScanM pair and
# Core accepts it, but listing both halves shows every ScanM recording twice,
# so the browser offers the ".smp" and lets Core find its partner.
RAW_SUFFIXES = (".h5", ".hdf5", ".smp")

KINDS = {
    "saved": "saved pygor object",
    "igor": "IGOR H5 export",
    "scanm": "ScanM raw",
}


def kind(path) -> str | None:
    """Which of :data:`KINDS` ``path`` is, or None if it is not a recording."""
    name = pathlib.Path(path).name
    if name.endswith(SAVED_SUFFIX):
        return "saved"
    if name.endswith((".h5", ".hdf5")):
        return "igor"
    if name.endswith(".smp"):
        return "scanm"
    return None


def is_recording(path) -> bool:
    return kind(path) is not None


def load_recording(path, n_colours=None):
    """Open any of :data:`KINDS` as a pygor object.

    A saved object comes back as whatever class saved it. A raw file is loaded
    as :class:`~pygor.classes.strf_data.STRF`, which is a Core with the STRF
    machinery on top: harmless on a recording that has no noise stimulus, and
    the alternative is asking the user to name a class for a file the class
    can be read off in a second.
    """
    import pygor.load

    path = pathlib.Path(path)
    if path.name.endswith(SAVED_SUFFIX):
        return pygor.load.Core.load_object(path)
    kwargs = {"n_colours": n_colours} if n_colours else {}
    return pygor.load.STRF(str(path), **kwargs)


def _shape(array) -> str:
    return "x".join(str(n) for n in array.shape)


def _count(value) -> int | None:
    """How many of something, or None for nothing, for a value that may be an
    array, a NaN placeholder, or an H5 scalar ``try_fetch`` passed through
    unconverted -- ``len()`` raises on the last two."""
    import numpy as np

    if value is None:
        return None
    array = np.asarray(value)
    if array.ndim == 0:
        return None if array.dtype.kind == "f" and np.isnan(array) else 1
    return len(array)


def summarise(recording) -> list[tuple[str, str]]:
    """Label/value rows describing a loaded recording.

    Every field is read defensively. The point of the inspector is to look at
    recordings that may be broken, so a missing attribute is a row saying so,
    never an exception that hides the twenty rows below it.
    """
    import numpy as np

    rows: list[tuple[str, str]] = [("class", type(recording).__name__)]

    meta = getattr(recording, "metadata", None) or {}
    date, time = meta.get("exp_date"), meta.get("exp_time")
    if date is not None:
        rows.append(("recorded", f"{date} {time or ''}".strip()))

    images = getattr(recording, "images", None)
    if images is not None:
        rows.append(("images", f"{_shape(images)}  ({images.dtype})"))
    else:
        rows.append(("images", "missing"))

    hz = getattr(recording, "frame_hz", None)
    if hz:
        rows.append(("frame rate", f"{hz:.2f} Hz"))
        if images is not None:
            rows.append(("duration", f"{len(images) / hz:.1f} s"))

    rois = getattr(recording, "rois", None)
    n_rois = getattr(recording, "num_rois", None)
    if rois is None:
        rows.append(("ROIs", "none segmented"))
    else:
        rows.append(("ROIs", f"{n_rois} on {_shape(rois)}"))

    triggers = _count(getattr(recording, "triggertimes", None))
    rows.append(("triggers", "none" if triggers is None else str(triggers)))
    rows.append(("trigger mode", str(getattr(recording, "trigger_mode", "?"))))

    try:
        rows.append(("registered", "yes" if recording.is_registered else "no"))
    except Exception:
        rows.append(("registered", "unknown"))

    depths = _count(getattr(recording, "ipl_depths", None))
    rows.append(("IPL depths", "none" if depths is None
                 else f"{depths} value{'s' if depths != 1 else ''}"))

    # np.nan is the "not computed" sentinel on Core for these, so a bare
    # `is None` check reports an absent average as present.
    for label, attr in (("averages", "averages"), ("snippets", "snippets"),
                        ("STRFs", "strfs")):
        value = getattr(recording, attr, None)
        if value is None or (np.isscalar(value) and np.isnan(value)):
            rows.append((label, "not computed"))
        else:
            rows.append((label, _shape(np.asarray(value))))

    return rows
