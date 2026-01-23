"""
Pygor preprocessing module.

This module provides functions to load and preprocess ScanM microscopy data
directly in Python, without requiring IGOR Pro as an intermediary.

Submodules
----------
scanm
    SMP/SMH file loading
registration
    Motion correction and ROI transfer
triggers
    Trigger signal preprocessing
"""

from .scanm import read_smh_header, read_smp_data, load_scanm, to_pygor_data, ScanMData
from .registration import register_stack, compute_batch_shifts, apply_shifts_to_stack, transfer_rois, transfer_rois_between
from .triggers import correct_ttl_baseline

__all__ = [
    # scanm.py
    "read_smh_header",
    "read_smp_data",
    "load_scanm",
    "to_pygor_data",
    "ScanMData",
    # registration.py
    "register_stack",
    "compute_batch_shifts",
    "apply_shifts_to_stack",
    "transfer_rois",
    "transfer_rois_between",
    # triggers.py
    "correct_ttl_baseline",
]
