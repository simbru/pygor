"""
Pygor preprocessing module.

This module provides functions to load and preprocess ScanM microscopy data
directly in Python, without requiring IGOR Pro as an intermediary.

Submodules
----------
scanm
    SMP/SMH file loading
"""

from .scanm import read_smh_header, read_smp_data, load_scanm, to_pygor_data, ScanMData

__all__ = [
    # scanm.py
    "read_smh_header",
    "read_smp_data",
    "load_scanm",
    "to_pygor_data",
    "ScanMData",
]
