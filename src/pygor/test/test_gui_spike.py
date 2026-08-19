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
    roi_ids_in_order,
    trace_index_to_label,
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

        from pygor.params import AnalysisParams

        self.params = AnalysisParams.from_config(None, analysis_type="Core")

    def calculate_image_average(self):
        return self.images.mean(axis=0)

    def update_rois(self, roi_mask):
        self.rois = np.asarray(roi_mask)
        self.num_rois = int((np.unique(self.rois) < 0).sum())

    def extract_traces_from_rois(self, baseline_dur=10.0):
        n_frames = self.images.shape[0]
        shape = (self.num_rois, n_frames)
        self.traces_raw = np.zeros(shape, dtype=np.float32)
        self.traces_znorm = np.zeros(shape, dtype=np.float32)
        return self.traces_raw, self.traces_znorm


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


def test_roi_ids_follow_extraction_order():
    igor = np.array([[1, -1, -2], [-3, 1, 1]])
    np.testing.assert_array_equal(roi_ids_in_order(igor), [-1, -2, -3])

    positive = np.array([[0, 1, 2], [3, 0, 0]])
    np.testing.assert_array_equal(roi_ids_in_order(positive), [1, 2, 3])


def test_label_index_accounts_for_erased_rois():
    """Trace rows are packed, so a gap in the labels shifts later rows."""
    mask = np.ones((2, 8), dtype=int)
    for roi in (1, 2, 3, 5, 6):
        mask[0, roi - 1] = -roi

    # Contiguous up to the gap, then one row lower than the label suggests
    assert label_to_trace_index(1, mask) == 0
    assert label_to_trace_index(3, mask) == 2
    assert label_to_trace_index(5, mask) == 3
    assert label_to_trace_index(6, mask) == 4

    # Label 4 was erased and has no row
    assert label_to_trace_index(4, mask) == -1

    assert trace_index_to_label(3, mask) == 5
    assert trace_index_to_label(99, mask) == 0


def test_roi_palette_contains_no_greys():
    import colorsys

    from pygor.gui.colors import MIN_SATURATION, roi_colors

    colors = roi_colors(48)
    assert len(colors) == 48
    for rgba in colors:
        saturation = colorsys.rgb_to_hsv(*rgba[:3])[1]
        assert saturation >= MIN_SATURATION
        # Grey means the channels are equal, so require a real spread
        assert float(rgba[:3].max() - rgba[:3].min()) > 0.1


def test_roi_colormap_keeps_background_transparent():
    from pygor.gui.colors import roi_colormap

    colormap = roi_colormap(12)
    np.testing.assert_allclose(colormap.colors[0], [0, 0, 0, 0])


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


def _process_events():
    """Let queued Qt callbacks run, such as the deferred layer-lock restore."""
    from qtpy.QtWidgets import QApplication

    app = QApplication.instance()
    if app is not None:
        app.processEvents()


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


def test_labels_layer_defaults(recording):
    from pygor.gui.launch import launch

    viewer = launch(recording, show=False, block=False)
    try:
        labels_layer = viewer.layers["ROIs"]
        assert labels_layer.preserve_labels is True
        assert labels_layer.contiguous is True
        assert viewer.window.dock_widgets["Traces"].auto_new_box.isChecked() is True
    finally:
        viewer.close()


def test_entering_draw_mode_leaves_an_occupied_label(recording):
    """Selection starts on ROI 1, so drawing would otherwise extend it."""
    from pygor.gui.launch import launch

    viewer = launch(recording, show=False, block=False)
    try:
        dock = viewer.window.dock_widgets["Traces"]
        labels_layer = viewer.layers["ROIs"]
        assert dock.selected_label == 1

        labels_layer.mode = "paint"
        assert dock.selected_label == dock.max_label + 1

        # Inspecting an ROI must still be possible without being bumped off
        labels_layer.mode = "pan_zoom"
        dock.set_roi(1)
        assert dock.selected_label == 1
    finally:
        viewer.close()


def test_extract_traces_picks_up_hand_drawn_rois(recording):
    """Extraction reads recording.rois, which layer edits do not touch."""
    from pygor.gui.launch import launch

    viewer = launch(recording, show=False, block=False)
    try:
        dock = viewer.window.dock_widgets["Traces"]
        actions = viewer.window.dock_widgets["Analysis"]
        labels_layer = viewer.layers["ROIs"]
        before = recording.num_rois

        dock.new_roi()
        labels_layer.data[12:14, 12:14] = dock.selected_label
        labels_layer.refresh()

        assert recording.num_rois == before
        assert actions.sync_rois_from_layer() is True
        assert recording.num_rois == before + 1

        raw, znorm = recording.extract_traces_from_rois()
        assert raw.shape[0] == before + 1
        assert znorm.shape[0] == before + 1
    finally:
        viewer.close()


def test_sync_is_a_noop_when_nothing_changed(recording):
    from pygor.gui.launch import launch

    viewer = launch(recording, show=False, block=False)
    try:
        actions = viewer.window.dock_widgets["Analysis"]
        assert actions.sync_rois_from_layer() is False
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


def test_removing_roi_layer_is_detected_not_silent(recording):
    """A deleted layer stays alive in Python, so edits would go nowhere."""
    from pygor.gui.launch import launch

    viewer = launch(recording, show=False, block=False)
    try:
        dock = viewer.window.dock_widgets["Traces"]
        actions = viewer.window.dock_widgets["Analysis"]

        actions.lock_box.setChecked(False)
        viewer.layers.remove(viewer.layers["ROIs"])
        assert dock.labels_layer is None
        assert actions.labels_layer is None
        assert "Restore default layers" in actions.status.text()

        # Guarded paths must degrade quietly rather than raise
        assert dock.max_label == 0
        dock.set_roi(3)
        dock.refresh()
        assert actions.sync_rois_from_layer() is False
    finally:
        viewer.close()


def test_removing_roi_layer_rescues_unpushed_edits(recording):
    from pygor.gui.launch import launch

    viewer = launch(recording, show=False, block=False)
    try:
        dock = viewer.window.dock_widgets["Traces"]
        labels_layer = viewer.layers["ROIs"]
        before = recording.num_rois

        actions = viewer.window.dock_widgets["Analysis"]
        actions.lock_box.setChecked(False)
        dock.new_roi()
        labels_layer.data[10:12, 10:12] = dock.selected_label
        labels_layer.refresh()

        viewer.layers.remove(labels_layer)
        assert recording.num_rois == before + 1
    finally:
        viewer.close()


def test_restore_rebuilds_layer_and_rebinds_docks(recording):
    from pygor.gui.launch import launch

    viewer = launch(recording, show=False, block=False)
    try:
        dock = viewer.window.dock_widgets["Traces"]
        actions = viewer.window.dock_widgets["Analysis"]
        actions.lock_box.setChecked(False)
        viewer.layers.remove(viewer.layers["ROIs"])

        restored = actions.restore_roi_layer()
        assert restored is not None
        assert "ROIs" in viewer.layers
        assert dock.labels_layer is restored
        assert restored.preserve_labels is True

        # Rebound events must work again
        dock.set_roi(2)
        assert restored.selected_label == 2
        restored.selected_label = 3
        assert dock.roi_spin.value() == 3
    finally:
        viewer.close()


def test_removing_a_user_layer_is_ignored(recording):
    from pygor.gui.launch import launch

    viewer = launch(recording, show=False, block=False)
    try:
        actions = viewer.window.dock_widgets["Analysis"]
        before = actions.status.text()

        viewer.add_points([[1, 1]], name="scratch")
        viewer.layers.remove(viewer.layers["scratch"])
        _process_events()

        assert actions.labels_layer is not None
        assert actions.status.text() == before
    finally:
        viewer.close()


def test_closing_a_viewer_with_locked_layers_terminates(recording):
    """viewer.close() empties the layer list; the lock must not refill it."""
    from pygor.gui.launch import launch

    viewer = launch(recording, show=False, block=False)
    actions = viewer.window.dock_widgets["Analysis"]
    assert actions.lock_box.isChecked() is True
    viewer.close()
    _process_events()
    assert len(viewer.layers) == 0


def test_locked_default_layers_come_straight_back(recording):
    from pygor.gui.launch import launch

    viewer = launch(recording, show=False, block=False)
    try:
        actions = viewer.window.dock_widgets["Analysis"]
        dock = viewer.window.dock_widgets["Traces"]
        assert actions.lock_box.isChecked() is True

        viewer.layers.remove(viewer.layers["Image stack"])
        _process_events()
        assert "Image stack" in viewer.layers
        assert "Restored locked layer" in actions.status.text()

        viewer.layers.remove(viewer.layers["ROIs"])
        _process_events()
        assert "ROIs" in viewer.layers
        assert dock.labels_layer is not None

        # ROIs must stay above the image stack, and numbers above ROIs
        order = [layer.name for layer in viewer.layers]
        assert order.index("ROIs") > order.index("Image stack")
        assert order[-1] == "ROI numbers"
    finally:
        viewer.close()


def test_restore_defaults_rebuilds_every_missing_layer(recording):
    from pygor.gui.launch import launch

    viewer = launch(recording, show=False, block=False)
    try:
        actions = viewer.window.dock_widgets["Analysis"]
        actions.lock_box.setChecked(False)
        expected = list(actions.default_layers)

        for name in expected:
            viewer.layers.remove(viewer.layers[name])
        assert sorted(actions.missing_default_layers()) == sorted(expected)

        actions.restore_default_layers()
        assert actions.missing_default_layers() == []
        assert sorted(layer.name for layer in viewer.layers) == sorted(expected)
    finally:
        viewer.close()


def test_trace_dock_reads_correct_row_after_erasing_an_roi(recording):
    from pygor.gui.launch import launch

    viewer = launch(recording, show=False, block=False)
    try:
        dock = viewer.window.dock_widgets["Traces"]
        actions = viewer.window.dock_widgets["Analysis"]
        labels_layer = viewer.layers["ROIs"]

        # Give each ROI a distinguishable trace before erasing one
        n = recording.num_rois
        recording.traces_znorm = np.tile(
            np.arange(1, n + 1, dtype=np.float32)[:, None],
            (1, recording.images.shape[0]),
        )

        data = np.asarray(labels_layer.data)
        data[data == 2] = 0
        labels_layer.data = data
        actions.sync_rois_from_layer()
        recording.traces_znorm = np.delete(recording.traces_znorm, 1, axis=0)

        labels_layer.selected_label = 3
        trace, _ = dock.current_trace()
        # Label 3 is now the second surviving ROI, so row 1
        np.testing.assert_allclose(trace, recording.traces_znorm[1])

        labels_layer.selected_label = 2
        assert dock.current_trace()[0] is None
    finally:
        viewer.close()


def test_trace_colour_matches_the_roi_colour(recording):
    from pygor.gui.launch import launch

    viewer = launch(recording, show=False, block=False)
    try:
        dock = viewer.window.dock_widgets["Traces"]
        labels_layer = viewer.layers["ROIs"]

        for label in (1, 2, 3):
            dock.set_roi(label)
            line_colour = np.asarray(dock.ax.lines[0].get_color())[:4]
            np.testing.assert_allclose(
                line_colour, np.asarray(labels_layer.get_color(label)), atol=1e-6
            )
    finally:
        viewer.close()


def test_no_roi_renders_grey(recording):
    """Grey ROIs vanish against the greyscale image stack."""
    import colorsys

    from pygor.gui.colors import MIN_SATURATION
    from pygor.gui.launch import launch

    viewer = launch(recording, show=False, block=False)
    try:
        labels_layer = viewer.layers["ROIs"]
        for label in range(1, recording.num_rois + 1):
            rgba = np.asarray(labels_layer.get_color(label))
            assert colorsys.rgb_to_hsv(*rgba[:3])[1] >= MIN_SATURATION
    finally:
        viewer.close()


def test_number_layer_labels_each_roi(recording):
    from pygor.gui.launch import launch
    from pygor.gui.roi_numbers import NUMBER_LAYER_NAME

    viewer = launch(recording, show=False, block=False)
    try:
        numbers = viewer.layers[NUMBER_LAYER_NAME]
        assert numbers.features["label"].tolist() == list(
            range(1, recording.num_rois + 1)
        )
        assert numbers.data.shape == (recording.num_rois, 2)
        # Numbers must sit above the ROI labels to stay readable
        assert [layer.name for layer in viewer.layers][-1] == NUMBER_LAYER_NAME
        np.testing.assert_allclose(numbers.text.color.constant, [1, 1, 1, 1])
    finally:
        viewer.close()


def test_number_layer_follows_new_rois(recording):
    from pygor.gui.launch import launch
    from pygor.gui.roi_numbers import NUMBER_LAYER_NAME

    viewer = launch(recording, show=False, block=False)
    try:
        dock = viewer.window.dock_widgets["Traces"]
        labels_layer = viewer.layers["ROIs"]
        labels_layer.mode = "paint"
        before = len(viewer.layers[NUMBER_LAYER_NAME].data)

        dock.new_roi()
        _stroke(labels_layer, (10, 10))

        numbers = viewer.layers[NUMBER_LAYER_NAME]
        assert len(numbers.data) == before + 1
    finally:
        viewer.close()


def _pygor_menu(viewer):
    from pygor.gui.menus import MENU_TITLE

    for action in viewer.window.main_menu.actions():
        if action.text() == MENU_TITLE:
            return action.menu()
    raise AssertionError("Pygor menu not found")


def _submenu(menu, title):
    for action in menu.actions():
        if action.text() == title:
            return action.menu()
    raise AssertionError(f"submenu {title!r} not found")


def test_pygor_menu_is_built(recording):
    from pygor.gui.launch import launch

    viewer = launch(recording, show=False, block=False)
    try:
        menu = _pygor_menu(viewer)
        titles = [a.text() for a in menu.actions() if not a.isSeparator()]
        assert {"Analysis", "ROIs", "View", "Parameters...", "Save recording as..."} <= set(
            titles
        )
        assert [a.text() for a in _submenu(menu, "Analysis").actions()] == [
            "Segment ROIs...",
            "Extract traces",
            "Correlation projection",
        ]
    finally:
        viewer.close()


def test_menu_toggles_track_the_docks_both_ways(recording):
    from pygor.gui.launch import launch

    viewer = launch(recording, show=False, block=False)
    try:
        dock = viewer.window.dock_widgets["Traces"]
        view = _submenu(_pygor_menu(viewer), "View")
        items = {a.text(): a for a in view.actions()}

        items["Follow frame"].setChecked(True)
        assert dock.follow_box.isChecked() is True

        dock.auto_new_box.setChecked(False)
        assert items["Auto-new ROI after each stroke"].isChecked() is False
    finally:
        viewer.close()


def test_menu_tracks_roi_number_visibility(recording):
    from pygor.gui.launch import launch
    from pygor.gui.roi_numbers import NUMBER_LAYER_NAME

    viewer = launch(recording, show=False, block=False)
    try:
        view = _submenu(_pygor_menu(viewer), "View")
        item = {a.text(): a for a in view.actions()}["Show ROI numbers"]

        viewer.layers[NUMBER_LAYER_NAME].visible = False
        assert item.isChecked() is False

        item.setChecked(True)
        assert viewer.layers[NUMBER_LAYER_NAME].visible is True
    finally:
        viewer.close()


def test_parameter_editor_docks_and_survives(recording):
    """params.edit(blocking=False) returned a widget nothing kept alive."""
    from pygor.gui.launch import launch

    viewer = launch(recording, show=False, block=False)
    try:
        actions = viewer.window.dock_widgets["Analysis"]
        assert "Parameters" not in viewer.window.dock_widgets

        actions.open_param_editor()
        assert "Parameters" in viewer.window.dock_widgets

        # Reopening must raise the existing dock, not stack up duplicates
        actions.open_param_editor()
        names = [n for n in viewer.window.dock_widgets if n == "Parameters"]
        assert len(names) == 1
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
