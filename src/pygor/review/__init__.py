"""Proofreading support: look at the data behind a row of an aggregate CSV.

The pipeline that builds those CSVs turns raw ScanM recordings into saved pygor
objects and then into one row per cell. Nothing in it lets a human check that a
row means what it says -- whether ROIs transferred onto the right cells, whether
a receptive field is real, whether a recording registered at all.

This subpackage is the front-end-agnostic half of that job: it finds the saved
objects, reads the cheap parts of them, renders evidence panels, and records
non-destructive verdicts to a sidecar. It deliberately holds no terminal or
napari code, and imports nothing the base install lacks, so it stays importable
headless and without any extras.
"""

from pygor.review.bundle import (
    AlignmentInfo,
    BundleCache,
    FovBundle,
    MissingPartner,
    group_refs,
    master_roi_index,
)
from pygor.review.index import RecordingRef, attach_status, read_arrays, scan_processed, scan_recording
from pygor.review.verdicts import Verdict, VerdictStore

__all__ = [
    "AlignmentInfo",
    "BundleCache",
    "FovBundle",
    "MissingPartner",
    "RecordingRef",
    "Verdict",
    "VerdictStore",
    "attach_status",
    "group_refs",
    "master_roi_index",
    "read_arrays",
    "scan_processed",
    "scan_recording",
]
