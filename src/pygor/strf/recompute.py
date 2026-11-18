"""Compute STRFs for some ROIs and splice them in, leaving the rest alone.

``calculate_strf(roi=[...])`` computes just those ROIs but then replaces
``strfs`` with them, dropping every other ROI's. Each STRF depends only on
its own ROI's trace and the stimulus, and is normalised on its own, so the
rows it returns are the rows a full run would give -- verified bit for bit
on the demo recording. Spliced back at their own indices, they make
recomputing an edited ROI cost one ROI, not the recording.
"""

from __future__ import annotations

import pathlib

import numpy as np

from pygor.core import roi_alignment as ra

#: Attributes ``calculate_strf`` overwrites that describe the whole array
_WHOLE_ARRAY_STATE = (
    "strfs",
    "strf_keys",
    "num_strfs",
    "n_colours",
    "multicolour",
    "strf_dur_ms",
)

#: Derived per STRF, too slow to recompute wholesale: the spliced rows get NaN
_PER_STRF_DERIVED = ("_pval_time", "_pval_space")


def load_noise(path):
    """The noise stimulus as ``(y, x, triggers)``.

    ``.npy`` files hold it as saved; ``.h5`` files are read from their
    ``noise_jitter`` dataset, or their only dataset. Some files store
    triggers first; the trigger axis is the longest by far, so it is moved
    last when it is not.
    """
    path = pathlib.Path(path).expanduser()
    if not path.exists():
        raise FileNotFoundError(f"no noise stimulus at {path}")
    if path.suffix.lower() == ".npy":
        noise = np.load(path)
    elif path.suffix.lower() in (".h5", ".hdf5"):
        import h5py

        with h5py.File(path, "r") as f:
            if "noise_jitter" in f:
                noise = np.asarray(f["noise_jitter"])
            else:
                datasets = [k for k in f if isinstance(f[k], h5py.Dataset)]
                if len(datasets) != 1:
                    raise ValueError(
                        f"{path.name} has no 'noise_jitter' dataset and "
                        f"{len(datasets)} others; cannot tell which is the noise"
                    )
                noise = np.asarray(f[datasets[0]])
    else:
        raise ValueError(f"noise stimulus must be .npy or .h5, not {path.suffix}")
    if noise.ndim != 3:
        raise ValueError(f"noise stimulus must be 3-D, got shape {noise.shape}")
    longest = int(np.argmax(noise.shape))
    if longest != 2:
        noise = np.moveaxis(noise, longest, -1)
    return noise


def recompute_strfs(recording, noise_array, rois=None, **kwargs):
    """Compute STRFs for ``rois`` (row indices), or for every ROI if None.

    With ``rois``, only those rows of ``strfs`` change; every other ROI's
    STRF, and everything derived from it, is left exactly as it was. The
    window and colour settings must produce STRFs the shape of the stored
    ones, or there is nothing consistent to splice into: that raises
    ``ValueError`` and nothing changes, and a full recompute is the fix.

    Returns the row indices that were computed.
    """
    n_rois = ra.count(recording.rois)
    if rois is None:
        recording.calculate_strf(noise_array, **kwargs)
        # calculate_strf leaves these from the old STRFs; they would
        # describe RFs that no longer exist
        for name in _PER_STRF_DERIVED + ra.DROPPED_STRF_STATE:
            if name in vars(recording):
                delattr(recording, name)
        return list(range(n_rois))

    rois = sorted({int(r) for r in rois})
    if not rois:
        return []
    if rois[0] < 0 or rois[-1] >= n_rois:
        raise ValueError(f"ROI rows {rois} are outside 0..{n_rois - 1}")
    problems = ra.check_alignment(recording)
    if problems:
        raise ValueError("STRFs are not aligned with the ROIs: " + "; ".join(problems))
    old = getattr(recording, "strfs", None)
    if not ra._present(old):
        raise ValueError("no stored STRFs to splice into; compute all of them")

    n_colours = ra.n_colours_of(recording, n_rois)
    saved = {name: getattr(recording, name, None) for name in _WHOLE_ARRAY_STATE}
    try:
        # n_colours given explicitly: calculate_strf otherwise falls back on
        # config, which need not match what the stored STRFs were made with
        recording.calculate_strf(noise_array, roi=rois, n_colours=n_colours, **kwargs)
        # Kept masked: the mask is part of the result, not decoration
        fresh = recording.strfs
    finally:
        for name, value in saved.items():
            setattr(recording, name, value)

    expected = (len(rois) * n_colours, *old.shape[1:])
    if fresh.shape != expected:
        raise ValueError(
            f"new STRFs come out {fresh.shape[1:]} per ROI, the stored ones are "
            f"{old.shape[1:]}: the STA window or crop differs from what they were "
            f"made with, so they cannot be mixed. Compute all RFs instead."
        )

    flat = ra._strf_rows(rois, n_colours)
    masked = isinstance(old, np.ma.MaskedArray) or isinstance(fresh, np.ma.MaskedArray)
    if masked:
        # Data and mask both: assigning a masked array into a masked array
        # copies its mask, so the new rows are masked as calculate_strf made them
        spliced = np.ma.array(old, copy=True, dtype=old.dtype)
        spliced.mask = np.ma.getmaskarray(old).copy()
        spliced[flat] = np.ma.asarray(fresh).astype(old.dtype)
    else:
        spliced = np.array(old, copy=True)
        spliced[flat] = np.asarray(fresh, dtype=old.dtype)
    # A new array, not an in-place write: views cache on the array's identity
    recording.strfs = spliced
    for name in _PER_STRF_DERIVED:
        value = getattr(recording, name, None)
        if ra._present(value) and len(value) == len(old):
            value = np.array(value, dtype=float, copy=True)
            value[flat] = np.nan
            setattr(recording, name, value)
    for name in ra.DROPPED_STRF_STATE:
        if name in vars(recording):
            delattr(recording, name)
    if hasattr(recording, "_invalidate_strf_caches"):
        recording._invalidate_strf_caches()
    return rois
