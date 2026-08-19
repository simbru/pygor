"""Feasibility tests for the napari GUI spike.

Runs headless via the Qt offscreen platform, so it needs napari and a Qt
binding but no display. Skipped entirely when the ``[gui]`` extra is absent.
"""

import os
import pathlib

import numpy as np
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

napari = pytest.importorskip("napari")
pytest.importorskip("magicgui")

from pygor.gui.roi_bridge import (  # noqa: E402
    is_igor_style,
    label_to_trace_index,
    labels_to_mask,
    mask_to_labels,
)


class StubRecording:
    """Minimal duck-typed stand-in for pygor.classes.core_data.Core."""

    def __init__(self, n_frames=20, shape=(16, 16), n_rois=3):
        rng = np.random.default_rng(0)
        self.filename = "stub.h5"
        self.images = rng.normal(size=(n_frames, *shape)).astype(np.float32)
        self.num_rois = n_rois

        mask = np.ones(shape, dtype=np.int32)
        for roi in range(1, n_rois + 1):
            mask[roi * 2 : roi * 2 + 2, :4] = -roi
        self.rois = mask

        self.traces_raw = rng.normal(size=(n_rois, n_frames)).astype(np.float32)
        self.traces_znorm = rng.normal(size=(n_rois, n_frames)).astype(np.float32)
        self._original_images = None
        self._pre_registration_images = None

    def calculate_image_average(self):
        return self.images.mean(axis=0)


@pytest.fixture
def recording():
    return StubRecording()


def test_igor_mask_roundtrip(recording):
    labels = mask_to_labels(recording.rois)
    assert labels.min() == 0
    assert labels.max() == recording.num_rois
    restored = labels_to_mask(labels, igor_style=True)
    np.testing.assert_array_equal(restored, recording.rois)


def test_positive_mask_detected_and_preserved():
    mask = np.zeros((8, 8), dtype=np.int32)
    mask[0:2, 0:2] = 1
    mask[4:6, 4:6] = 2
    assert is_igor_style(mask) is False
    labels = mask_to_labels(mask)
    np.testing.assert_array_equal(labels, mask)


def test_label_index_mapping():
    assert label_to_trace_index(1) == 0
    assert label_to_trace_index(3) == 2


def test_viewer_builds_with_layers_and_docks(recording, make_napari_viewer=None):
    from pygor.gui.launch import launch

    viewer = launch(recording, show=False, block=False)
    try:
        names = [layer.name for layer in viewer.layers]
        assert "Image stack" in names
        assert "ROIs" in names
        docked = viewer.window.dock_widgets
        assert "Traces" in docked
        assert "Analysis" in docked
    finally:
        viewer.close()


def test_trace_dock_follows_label_selection(recording):
    from pygor.gui.launch import launch

    viewer = launch(recording, show=False, block=False)
    try:
        dock = viewer.window.dock_widgets["Traces"]
        labels_layer = viewer.layers["ROIs"]

        labels_layer.selected_label = 2
        trace, source = dock.current_trace()
        assert trace is not None
        np.testing.assert_array_equal(trace, getattr(recording, source)[1])

        labels_layer.selected_label = 99
        missing, _ = dock.current_trace()
        assert missing is None
    finally:
        viewer.close()


def test_roi_navigation_steps_and_wraps(recording):
    from pygor.gui.launch import launch

    viewer = launch(recording, show=False, block=False)
    try:
        dock = viewer.window.dock_widgets["Traces"]
        assert dock.n_rois == recording.num_rois

        dock.set_roi(1)
        dock.step_roi(1)
        assert dock.selected_label == 2

        dock.set_roi(recording.num_rois)
        dock.step_roi(1)
        assert dock.selected_label == 1

        dock.step_roi(-1)
        assert dock.selected_label == recording.num_rois
    finally:
        viewer.close()


def test_roi_spinbox_and_layer_stay_in_sync(recording):
    from pygor.gui.launch import launch

    viewer = launch(recording, show=False, block=False)
    try:
        dock = viewer.window.dock_widgets["Traces"]
        labels_layer = viewer.layers["ROIs"]

        dock.roi_spin.setValue(3)
        assert labels_layer.selected_label == 3

        labels_layer.selected_label = 2
        assert dock.roi_spin.value() == 2
    finally:
        viewer.close()


def test_follow_frame_is_off_by_default(recording):
    from pygor.gui.launch import launch

    viewer = launch(recording, show=False, block=False)
    try:
        dock = viewer.window.dock_widgets["Traces"]
        assert dock.follow_box.isChecked() is False
        assert dock._cursor is None
        assert dock._background is None
    finally:
        viewer.close()


def test_follow_frame_uses_blitting_when_enabled(recording):
    from pygor.gui.launch import launch

    viewer = launch(recording, show=False, block=False)
    try:
        dock = viewer.window.dock_widgets["Traces"]
        dock.follow_box.setChecked(True)
        assert dock._cursor is not None
        # Animated artists are excluded from the cached background
        assert dock._cursor.get_animated() is True
        assert dock._background is not None

        viewer.dims.set_current_step(0, 5)
        assert dock._cursor.get_xdata()[0] == 5
    finally:
        viewer.close()


def test_centre_view_draws_and_moves_crosshair(recording):
    from pygor.gui.launch import launch
    from pygor.gui.widgets.traces import CENTRE_LAYER_NAME

    viewer = launch(recording, show=False, block=False)
    try:
        dock = viewer.window.dock_widgets["Traces"]
        assert CENTRE_LAYER_NAME not in viewer.layers

        dock.centre_box.setChecked(True)
        dock.set_roi(2)
        marker = viewer.layers[CENTRE_LAYER_NAME]
        first = np.array(marker.data, copy=True)
        assert marker.visible is True

        # Adding the marker must not steal the layer-list selection,
        # which would knock the Labels layer out of picker mode.
        assert [layer.name for layer in viewer.layers.selection] == ["ROIs"]

        dock.set_roi(3)
        assert not np.allclose(np.array(marker.data), first)

        dock.centre_box.setChecked(False)
        assert viewer.layers[CENTRE_LAYER_NAME].visible is False
    finally:
        viewer.close()


def test_crosshair_scales_with_image_width(recording):
    from pygor.gui.launch import launch
    from pygor.gui.widgets.traces import CENTRE_LAYER_NAME

    viewer = launch(recording, show=False, block=False)
    try:
        dock = viewer.window.dock_widgets["Traces"]
        dock.centre_box.setChecked(True)
        dock.set_roi(1)

        marker = viewer.layers[CENTRE_LAYER_NAME]
        expected = recording.rois.shape[-1] / 30
        np.testing.assert_allclose(np.asarray(marker.size).ravel(), expected)
        np.testing.assert_allclose(np.asarray(marker.face_color)[0], [1, 1, 1, 1])
        np.testing.assert_allclose(np.asarray(marker.border_color)[0], [0, 0, 0, 0])
    finally:
        viewer.close()


def test_new_roi_selects_next_free_label(recording):
    from pygor.gui.launch import launch

    viewer = launch(recording, show=False, block=False)
    try:
        dock = viewer.window.dock_widgets["Traces"]
        labels_layer = viewer.layers["ROIs"]
        start = dock.max_label

        dock.new_roi()
        assert dock.selected_label == start + 1
        assert dock.roi_spin.value() == start + 1

        # Drawing into it then asking again must advance, not repeat
        labels_layer.data[0:2, 0:2] = dock.selected_label
        labels_layer.refresh()
        dock.new_roi()
        assert dock.selected_label == start + 2
    finally:
        viewer.close()


def test_new_roi_label_survives_refresh(recording):
    """A pending label has no pixels yet and must not be clamped away."""
    from pygor.gui.launch import launch

    viewer = launch(recording, show=False, block=False)
    try:
        dock = viewer.window.dock_widgets["Traces"]
        dock.new_roi()
        pending = dock.selected_label

        dock.refresh()
        assert dock.selected_label == pending
        assert dock.roi_spin.value() == pending
    finally:
        viewer.close()


def _stroke(labels_layer, coord):
    """Simulate a completed brush stroke.

    napari wraps a drag in block_history and commits the staged undo
    history on release, which is what emits the paint event.
    """
    with labels_layer.block_history():
        labels_layer.paint(coord, labels_layer.selected_label, refresh=False)


def test_auto_new_advances_label_per_stroke(recording):
    from pygor.gui.launch import launch

    viewer = launch(recording, show=False, block=False)
    try:
        dock = viewer.window.dock_widgets["Traces"]
        labels_layer = viewer.layers["ROIs"]
        labels_layer.mode = "paint"

        dock.auto_new_box.setChecked(True)
        first = dock.selected_label
        _stroke(labels_layer, (10, 10))
        second = dock.selected_label
        assert second != first

        _stroke(labels_layer, (12, 12))
        assert dock.selected_label == second + 1
    finally:
        viewer.close()


def test_auto_new_off_keeps_label_for_multi_stroke_rois(recording):
    from pygor.gui.launch import launch

    viewer = launch(recording, show=False, block=False)
    try:
        dock = viewer.window.dock_widgets["Traces"]
        labels_layer = viewer.layers["ROIs"]
        labels_layer.mode = "paint"

        dock.auto_new_box.setChecked(False)
        label = dock.selected_label
        _stroke(labels_layer, (10, 10))
        _stroke(labels_layer, (11, 11))
        assert dock.selected_label == label
    finally:
        viewer.close()


def test_auto_new_ignores_erasing(recording):
    from pygor.gui.launch import launch

    viewer = launch(recording, show=False, block=False)
    try:
        dock = viewer.window.dock_widgets["Traces"]
        labels_layer = viewer.layers["ROIs"]

        dock.auto_new_box.setChecked(True)
        labels_layer.mode = "erase"
        label = dock.selected_label
        _stroke(labels_layer, (10, 10))
        assert dock.selected_label == label
    finally:
        viewer.close()


def test_navigation_keys_leave_brush_size_alone(recording):
    """[ and ] are napari's brush size controls, needed while painting."""
    from pygor.gui.launch import launch

    viewer = launch(recording, show=False, block=False)
    try:
        bound = {str(key) for key in viewer.keymap}
        assert "[" not in bound
        assert "]" not in bound
        assert {",", ".", "N"} <= bound
    finally:
        viewer.close()


def test_actions_dock_builds_all_buttons(recording):
    from pygor.gui.launch import launch

    viewer = launch(recording, show=False, block=False)
    try:
        actions = viewer.window.dock_widgets["Analysis"]
        assert actions.status.text() == "Idle"
    finally:
        viewer.close()


# Real-data smoke test. Skipped unless the demo recording is present, since
# it is not checked into the repository.
# Set PYGOR_TEST_H5 to point at any recording on the local machine.
_DEMO = pathlib.Path(
    os.environ.get(
        "PYGOR_TEST_H5",
        pathlib.Path(__file__).parents[3].joinpath("examples/strf_demo_data.h5"),
    )
)


@pytest.mark.skipif(not _DEMO.exists(), reason=f"no recording at {_DEMO}")
def test_launch_against_real_recording():
    import pygor.load
    from pygor.gui.launch import launch

    rec = pygor.load.Core(_DEMO)
    viewer = launch(rec, show=False, block=False)
    try:
        assert "Image stack" in [layer.name for layer in viewer.layers]
        dock = viewer.window.dock_widgets["Traces"]
        if "ROIs" in viewer.layers:
            viewer.layers["ROIs"].selected_label = 1
            trace, _ = dock.current_trace()
            assert trace is not None
    finally:
        viewer.close()
