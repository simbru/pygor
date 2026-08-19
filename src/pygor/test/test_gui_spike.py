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
