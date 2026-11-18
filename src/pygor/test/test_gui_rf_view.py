"""The RF view, the RF mosaic panel and the Gaussian-fit metrics in the GUI.

Headless via the Qt offscreen platform, like test_gui_spike. Tests on real
STRFs open a saved recording from the BC paper data and are skipped where that
data is not mounted.
"""

import os
import pathlib

import numpy as np
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

napari = pytest.importorskip("napari")

from pygor.gui.launch import RF_PANEL  # noqa: E402
from pygor.gui.metrics import available_metrics  # noqa: E402
from pygor.gui.roi_bridge import trace_index_to_label  # noqa: E402
from pygor.gui.widgets.rf_grid import mosaic, normalised  # noqa: E402
from pygor.test.test_gui_spike import StubRecording  # noqa: E402

# 261001 GB ablation, field 8_2: a single-colour recording with 117 STRFs
SAVED = pathlib.Path(
    "/mnt/data/BC paper data/Achromatic SWN/Processed/gb_ablation/261001 GB abl/"
    "8_2_SWN_200_Achrom.recording.h5"
)
RF_METRICS = {"gaussian_index", "gaussian_snr"}


def test_no_strfs_means_no_rf_view_panel_or_metrics():
    from pygor.gui.launch import launch

    recording = StubRecording()
    assert not RF_METRICS & {spec.key for spec in available_metrics(recording)}
    viewer = launch(recording, show=False, block=False)
    try:
        dock = viewer.window.dock_widgets["Plot"]
        assert dock.RF not in dock.available_views()
        assert dock.mode_box.findText(dock.RF) < 0
        assert RF_PANEL not in viewer.window.dock_widgets
    finally:
        viewer.close()


def test_mosaic_lays_tiles_out_row_by_row_with_gaps():
    tiles = np.arange(5 * 2 * 3, dtype=float).reshape(5, 2, 3)
    grid = mosaic(tiles, n_cols=2)
    assert grid.shape == (3 * 3 - 1, 2 * 4 - 1)  # 3 rows of 2, 1 px gaps
    np.testing.assert_array_equal(grid[0:2, 4:7], tiles[1])
    np.testing.assert_array_equal(grid[6:8, 0:3], tiles[4])
    assert np.isnan(grid[6:8, 4:7]).all()  # the unused sixth tile
    assert np.isnan(grid[2]).all()  # the gap between rows


def test_each_tile_is_scaled_to_its_own_peak():
    stack = np.array([[[1.0, -4.0]], [[0.5, 0.25]], [[0.0, 0.0]]])
    out = normalised(stack)
    np.testing.assert_allclose(out[0], [[0.25, -1.0]])
    np.testing.assert_allclose(out[1], [[1.0, 0.5]])
    np.testing.assert_array_equal(out[2], 0.0)


@pytest.fixture(scope="module")
def saved():
    if not SAVED.exists():
        pytest.skip("BC paper data not mounted")
    import pygor.load

    return pygor.load.Core.load_object(SAVED)


def test_metrics_are_the_fit_values_per_roi(saved):
    from pygor.gui.metrics import compute_metric, metric_by_key

    n_rois = len(saved.strfs) // saved.n_colours
    index = compute_metric(metric_by_key("gaussian_index"), saved, n_rois)
    # Single colour: the median over one channel is the value itself
    np.testing.assert_array_equal(index, saved.get_gaussian_fit_index())


def test_rf_view_puts_map_and_movie_side_by_side_fit_on_request(saved):
    from pygor.gui.launch import launch

    viewer = launch(saved, show=False, block=False)
    try:
        dock = viewer.window.dock_widgets["Plot"]
        dock.set_view(dock.RF)
        viewer.layers["ROIs"].selected_label = trace_index_to_label(0, saved.rois)
        static, movie, kernel = dock.figure.axes  # one colour: one row of three
        assert static.get_position().y0 == pytest.approx(movie.get_position().y0)
        # Centre and surround kernels, and the marker following the movie frame
        assert len(kernel.get_lines()) >= 3
        marker, frame_x = dock._rf_frame_lines[0]
        dock.frame_slider.setValue(5)
        assert marker.get_xdata()[0] == pytest.approx(frame_x[5])
        # Off by default, so the fit does not sway the call on the RF
        assert not dock.fits_box.isChecked()
        assert not static.patches and not movie.patches
        assert not dock.figure.legends
        assert "fit index" not in static.get_title(loc="left")

        dock.fits_box.setChecked(True)
        static, movie, _kernel = dock.figure.axes
        assert len(static.patches) == len(movie.patches) == 2  # fit and footprint
        assert dock.figure.legends
        assert "fit index" in static.get_title(loc="left") or "no fit" in static.get_title(loc="left")
        assert "non-Gaussian" not in static.get_title(loc="left")
    finally:
        viewer.close()


def test_rf_panel_shows_every_roi_and_selects_on_click(saved):
    from matplotlib.backend_bases import MouseEvent

    from pygor.gui.launch import launch

    viewer = launch(saved, show=False, block=False)
    try:
        panel = viewer.window.dock_widgets[RF_PANEL]
        panel.redraw()
        n_rois = len(saved.strfs) // saved.n_colours
        assert len(panel._maps) == n_rois
        assert len(panel.ax.texts) == n_rois  # every tile numbered

        # Click the middle of the third tile
        h, w = panel._maps.shape[-2:]
        x, y = panel._tile_origin(2)
        panel.canvas.draw()
        px, py = panel.ax.transData.transform((x + w / 2, y + h / 2))
        panel._on_click(MouseEvent("button_press_event", panel.canvas, px, py, 1))
        assert viewer.layers["ROIs"].selected_label == trace_index_to_label(2, saved.rois)
        assert panel._highlight is not None

        panel.movie_box.setChecked(True)
        panel.frame_slider.setValue(0)
        frame0 = mosaic(panel._movies[:, 0], panel._n_cols)
        np.testing.assert_array_equal(panel._image.get_array().filled(np.nan), frame0)
    finally:
        viewer.close()


def test_rf_panel_keeps_tiles_large_and_scrolls(saved):
    from pygor.gui.launch import launch

    viewer = launch(saved, show=False, block=False)
    try:
        panel = viewer.window.dock_widgets[RF_PANEL]
        panel.resize(400, 600)
        panel.scroll.viewport().resize(380, 500)
        panel.redraw()
        # Auto columns: as many ~110 px tiles as fit the width
        assert panel._n_cols == 380 // 110
        # The canvas grows past the viewport instead of shrinking the tiles
        assert panel.canvas.minimumHeight() > 500
        panel.columns_spin.setValue(5)
        assert panel._n_cols == 5
    finally:
        viewer.close()


def test_rf_movie_opens_on_the_peak_frame_and_steps(saved):
    from pygor.gui.launch import launch

    viewer = launch(saved, show=False, block=False)
    try:
        dock = viewer.window.dock_widgets["Plot"]
        dock.set_view(dock.RF)
        viewer.layers["ROIs"].selected_label = trace_index_to_label(0, saved.rois)
        frames = np.asarray(saved.strfs_no_border[0])
        peak = int(np.argmax(np.nan_to_num(np.abs(frames)).max(axis=(1, 2))))
        assert dock.frame_slider.value() == peak
        artist, _ = dock._rf_movies[0]
        dock.frame_slider.setValue(0)
        np.testing.assert_array_equal(artist.get_array(), frames[0])
        dock.play_button.setChecked(True)
        assert dock.rf_timer.isActive()
        dock.set_view(dock.TRACE)
        assert not dock.rf_timer.isActive()
    finally:
        viewer.close()


def test_reject_rf_marks_the_recording_and_the_panel(saved):
    from pygor.gui.launch import launch
    from pygor.strf.rejection import ATTR, rejected

    before = getattr(saved, ATTR, None)
    # Start clean: the real file carries curated rejections; put back below
    setattr(saved, ATTR, np.array([], dtype=int))
    viewer = launch(saved, show=False, block=False)
    try:
        dock = viewer.window.dock_widgets["Plot"]
        panel = viewer.window.dock_widgets[RF_PANEL]
        panel.redraw()
        dock.set_view(dock.RF)
        viewer.layers["ROIs"].selected_label = trace_index_to_label(5, saved.rois)
        assert not dock.reject_button.isChecked()

        dock.reject_button.setChecked(True)
        assert rejected(saved).tolist() == [5]
        assert "rejected" in dock.status.text()
        assert len(panel._rejected_marks) == 2  # the tile's cross

        # Another ROI shows its own state
        viewer.layers["ROIs"].selected_label = trace_index_to_label(6, saved.rois)
        assert not dock.reject_button.isChecked()

        viewer.layers["ROIs"].selected_label = trace_index_to_label(5, saved.rois)
        dock.reject_button.setChecked(False)
        assert rejected(saved).size == 0
        assert not panel._rejected_marks
    finally:
        viewer.close()
        if before is None:
            saved.__dict__.pop(ATTR, None)
        else:
            setattr(saved, ATTR, before)
