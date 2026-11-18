"""Recomputing some ROIs' STRFs splices them in without touching the rest.

The real-data tests need the demo recording and a noise stimulus; point
PYGOR_TEST_H5 and PYGOR_TEST_NOISE at them, or they are skipped.
"""

import os
import pathlib

import numpy as np
import pytest

from pygor.core import roi_alignment as ra
from pygor.strf.recompute import load_noise, recompute_strfs

_DEMO = pathlib.Path(
    os.environ.get(
        "PYGOR_TEST_H5",
        pathlib.Path(__file__).parents[3].joinpath("examples/strf_demo_data.h5"),
    )
)
_NOISE = pathlib.Path(os.environ.get("PYGOR_TEST_NOISE", "/nonexistent"))
needs_data = pytest.mark.skipif(
    not (_DEMO.exists() and _NOISE.exists()),
    reason="set PYGOR_TEST_H5 and PYGOR_TEST_NOISE to run against real data",
)


def test_noise_is_read_triggers_last(tmp_path):
    noise = np.random.default_rng(0).integers(0, 2, (6, 10, 500)).astype(np.uint8)
    np.save(tmp_path / "yx_t.npy", noise)
    np.save(tmp_path / "t_yx.npy", np.moveaxis(noise, -1, 0))
    assert np.array_equal(load_noise(tmp_path / "yx_t.npy"), noise)
    assert np.array_equal(load_noise(tmp_path / "t_yx.npy"), noise)


def test_noise_from_h5(tmp_path):
    import h5py

    noise = np.zeros((6, 10, 500), dtype=np.uint8)
    with h5py.File(tmp_path / "n.h5", "w") as f:
        f["noise_jitter"] = noise
        f["other"] = np.zeros(3)
    assert load_noise(tmp_path / "n.h5").shape == (6, 10, 500)


def test_missing_noise_file_is_reported(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_noise(tmp_path / "nope.npy")


def same(a, b):
    """Equal data (NaN equal to NaN) and an equal mask: masked STRFs that
    only compared ``.data`` would hide a border unmasked by a copy."""
    data_a = np.nan_to_num(np.ma.getdata(a))
    data_b = np.nan_to_num(np.ma.getdata(b))
    return np.array_equal(data_a, data_b) and np.array_equal(
        np.ma.getmaskarray(a), np.ma.getmaskarray(b)
    )


def _demo():
    import pygor.load

    return pygor.load.STRF(_DEMO)


@needs_data
def test_missing_rfs_are_computed_and_every_other_rf_is_untouched():
    noise = load_noise(_NOISE)
    rec = _demo()
    recompute_strfs(rec, noise)  # all, with pygor's own settings
    nc = ra.n_colours_of(rec)
    before = {
        float(np.flatnonzero(rec.rois.ravel() == roi_id).sum()): rec.strfs[r * nc : (r + 1) * nc].copy()
        for r, roi_id in enumerate(ra.roi_ids(rec.rois))
    }

    mask = rec.rois.copy()
    mask[mask == -1] = 1  # delete the first ROI
    rows, cols = np.nonzero(mask == 1)
    mask[rows[:6], cols[:6]] = -60  # draw a new one
    ra.apply_mask(rec, mask)
    missing = ra.rois_missing_strfs(rec)
    assert missing == [rec.num_rois - 1]

    untouched = rec.strfs.copy()
    assert recompute_strfs(rec, noise, missing) == missing
    for r, roi_id in enumerate(ra.roi_ids(rec.rois)):
        rows_ = slice(r * nc, (r + 1) * nc)
        key = float(np.flatnonzero(rec.rois.ravel() == roi_id).sum())
        if r in missing:
            assert not np.isnan(rec.strfs[rows_]).all(), "the new ROI got an RF"
        else:
            assert same(rec.strfs[rows_], untouched[rows_]), "a survivor's RF changed"
            assert same(rec.strfs[rows_], before[key]), "a survivor's RF moved"

    # The spliced RF is the one a full recompute gives that ROI
    spliced = rec.strfs.copy()
    recompute_strfs(rec, noise)
    assert same(spliced, rec.strfs), "data and mask both match a full recompute"
    assert ra.check_alignment(rec) == []


@needs_data
def test_splicing_into_rfs_of_another_shape_is_refused_untouched():
    """The demo's stored STRFs come from IGOR with a different STA window."""
    rec = _demo()
    mask = rec.rois.copy()
    rows, cols = np.nonzero(mask == 1)
    mask[rows[:6], cols[:6]] = -60
    ra.apply_mask(rec, mask)
    stored = rec.strfs.copy()
    with pytest.raises(ValueError, match="Compute all RFs instead"):
        recompute_strfs(rec, load_noise(_NOISE), ra.rois_missing_strfs(rec))
    assert same(rec.strfs, stored)
    assert rec.strfs.shape == stored.shape


@needs_data
def test_full_recompute_drops_stale_pvalues():
    rec = _demo()
    rec._pval_time = np.zeros(len(rec.strfs))
    recompute_strfs(rec, load_noise(_NOISE))
    assert "_pval_time" not in vars(rec)


@needs_data
def test_gui_recomputes_missing_rfs_from_the_menu():
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    napari = pytest.importorskip("napari")
    from pygor.gui.launch import attach
    from pygor.test.test_gui_spike import _pygor_menu, _submenu, _wait_for

    rec = _demo()
    recompute_strfs(rec, load_noise(_NOISE))
    viewer = napari.Viewer(show=False)
    try:
        session = attach(viewer, rec)
        actions = session.actions
        titles = [a.text() for a in _submenu(_pygor_menu(viewer), "Analysis").actions()]
        assert {"Compute RFs...", "Recompute missing RFs"} <= set(titles)

        layer = viewer.layers["ROIs"]
        data = np.asarray(layer.data).copy()
        rows, cols = np.nonzero(data == 0)
        data[rows[:6], cols[:6]] = data.max() + 1
        layer.data = data
        survivors = rec.strfs.copy()

        actions.bus.set("strf.calculate.noise_path", str(_NOISE))
        worker = actions.run_rfs(missing_only=True)
        assert worker is not None
        assert _wait_for(lambda: not ra.rois_missing_strfs(rec), timeout=120)
        n = len(survivors)
        assert same(rec.strfs[:n], survivors), "survivors' RFs untouched"
        assert ra.check_alignment(rec) == []
    finally:
        viewer.close()
