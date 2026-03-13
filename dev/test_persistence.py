"""Minimal test: load real H5 files → save as .pygor.h5 → reload → compare."""

import pathlib
import tempfile

import numpy as np

from pygor.classes.experiment import Experiment

# ── Config ───────────────────────────────────────────────────────────────
h5_files = [
    pathlib.Path("/home/simen/Documents/Git_repos/2p_analysis/data/2023-11-7_0_4_ColourSWN_100.h5"),
]
print(f"Using {len(h5_files)} H5 file(s):")
for f in h5_files:
    print(f"  {f.name}")

# ── 1. Load from original IGOR H5 files ─────────────────────────────────
print("\n=== Loading from IGOR H5 files ===")
exp = Experiment.from_files(h5_files, "STRF", n_jobs=1)
print(exp.recording_id)

# ── 2. Save to .pygor.h5 ────────────────────────────────────────────────
out_path = pathlib.Path(tempfile.mktemp(suffix=".pygor.h5"))
print(f"\n=== Saving to {out_path} ===")
exp.save(out_path)
print(f"File size: {out_path.stat().st_size / 1024 / 1024:.2f} MB")

# ── 3. Reload from .pygor.h5 ────────────────────────────────────────────
print("\n=== Reloading from .pygor.h5 ===")
loaded = Experiment.load(out_path)
print(loaded.recording_id)

# ── 4. Compare ──────────────────────────────────────────────────────────
print("\n=== Comparing original vs loaded ===")
assert len(loaded.recording) == len(exp.recording), "Recording count mismatch"

for i, (orig, reloaded) in enumerate(zip(exp.recording, loaded.recording)):
    tag = f"rec[{i}] {orig.name}"

    # Class
    assert type(orig).__name__ == type(reloaded).__name__, f"{tag}: class mismatch"

    # Metadata
    assert reloaded.metadata["exp_date"] == orig.metadata["exp_date"], f"{tag}: date"
    assert reloaded.metadata["exp_time"] == orig.metadata["exp_time"], f"{tag}: time"
    assert reloaded.name == orig.name, f"{tag}: name"
    assert reloaded.type == orig.type, f"{tag}: type"
    assert reloaded.num_rois == orig.num_rois, f"{tag}: num_rois"

    # Arrays
    if orig.images is not None:
        assert np.allclose(reloaded.images, orig.images, equal_nan=True), f"{tag}: images"
    if orig.traces_raw is not None:
        assert np.allclose(reloaded.traces_raw, orig.traces_raw, equal_nan=True), f"{tag}: traces_raw"
    if orig.averages is not None and not np.isscalar(orig.averages):
        assert np.allclose(reloaded.averages, orig.averages, equal_nan=True), f"{tag}: averages"
    if orig.rois is not None:
        assert np.array_equal(reloaded.rois, orig.rois), f"{tag}: rois"

    # STRF-specific
    if hasattr(orig, "strfs") and orig.strfs is not None:
        assert np.allclose(reloaded.strfs, orig.strfs, equal_nan=True), f"{tag}: strfs"
        assert reloaded.numcolour == orig.numcolour, f"{tag}: numcolour"
        assert reloaded.multicolour == orig.multicolour, f"{tag}: multicolour"
        assert reloaded.strf_keys == orig.strf_keys, f"{tag}: strf_keys"

    # Timing
    assert np.allclose(reloaded.triggertimes, orig.triggertimes), f"{tag}: triggertimes"
    assert reloaded.frame_hz == orig.frame_hz, f"{tag}: frame_hz"

    # AnalysisParams
    assert reloaded.params.analysis_type == orig.params.analysis_type, f"{tag}: params.analysis_type"

    # Internal state reconstructed
    assert hasattr(reloaded, "_Core__compare_ops_map"), f"{tag}: compare_ops_map missing"
    assert hasattr(reloaded, "_Core__keyword_lables"), f"{tag}: keyword_lables missing"

    # repr works
    repr(reloaded)

    print(f"  {tag}: OK ({reloaded.num_rois} ROIs, strfs={reloaded.strfs.shape if reloaded.strfs is not None else None})")

# ── 5. Verify fetch still works on reloaded experiment ───────────────────
print("\n=== Testing fetch on reloaded experiment ===")
num_rois_list = loaded.fetch_all("num_rois")
print(f"num_rois across recordings: {num_rois_list}")

names = loaded.fetch_all("name")
print(f"names: {list(names)}")

# ── Cleanup ──────────────────────────────────────────────────────────────
out_path.unlink()
print(f"\n=== ALL TESTS PASSED ===")
