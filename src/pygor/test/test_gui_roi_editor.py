"""The ROI editor: click to select ROIs, then delete, merge or redraw them.

Headless, like the other GUI tests. The stub recording has three ROIs,
each two rows tall in the first four columns: ROI 1 in rows 2-3, ROI 2
in rows 4-5, ROI 3 in rows 6-7.
"""

import os

import numpy as np
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

napari = pytest.importorskip("napari")
pytest.importorskip("magicgui")

from pygor.test.test_gui_spike import (  # noqa: E402
    StubRecording,
    _pygor_menu,
    _stroke,
    _submenu,
)

# One pixel inside each stub ROI, as (row, col)
INSIDE = {1: (2, 1), 2: (4, 1), 3: (6, 1)}


class FakeClick:
    """Enough of a napari mouse event for the editor's drag callback."""

    def __init__(self, row, col, modifiers=(), button=1):
        self.button = button
        self.modifiers = modifiers
        self.type = "mouse_press"
        # Screen and world positions; only their change matters for pos
        self.pos = (0.0, 0.0)
        self.position = (0.0, float(row), float(col))


def _click(editor, label=None, at=None, shift=False, drag=0.0):
    """Press and release over an ROI, optionally moving in between."""
    row, col = at if at is not None else INSIDE[label]
    event = FakeClick(row, col, modifiers=("Shift",) if shift else ())
    gen = editor._on_mouse(editor.viewer, event)
    try:
        next(gen)
    except StopIteration:
        return  # the callback declined the press
    if drag:
        event.type = "mouse_move"
        event.pos = (drag, 0.0)
        next(gen)
    event.type = "mouse_release"
    try:
        next(gen)
    except StopIteration:
        pass


@pytest.fixture
def session():
    from pygor.gui.launch import ROI_EDITOR_PANEL, attach

    viewer = napari.Viewer(show=False)
    session = attach(viewer, StubRecording())
    editor = session.docks[ROI_EDITOR_PANEL]
    editor.set_active(True)
    yield viewer, session, editor
    viewer.close()


def _labels(viewer):
    return np.asarray(viewer.layers["ROIs"].data)


def _present(viewer):
    values = np.unique(_labels(viewer))
    return set(values[values > 0].tolist())


def test_rois_menu_opens_the_editor():
    from pygor.gui.launch import ROI_EDITOR_PANEL, attach

    viewer = napari.Viewer(show=False)
    try:
        session = attach(viewer, StubRecording())
        editor = session.docks[ROI_EDITOR_PANEL]
        assert not editor.active, "the editor should start closed"

        rois = _submenu(_pygor_menu(viewer), "ROIs")
        entry = next(a for a in rois.actions() if a.text() == "Edit ROIs...")
        viewer.layers["ROIs"].mode = "paint"
        entry.trigger()

        assert editor.active
        assert str(viewer.layers["ROIs"].mode) == "pan_zoom"
        assert viewer.layers.selection.active is viewer.layers["ROIs"]
    finally:
        viewer.close()


def test_click_selects_and_shift_click_toggles(session):
    viewer, _session, editor = session

    _click(editor, 2)
    assert editor.selection == {2}
    assert viewer.layers["ROIs"].selected_label == 2, "the plot follows the pick"

    _click(editor, 3, shift=True)
    assert editor.selection == {2, 3}
    _click(editor, 2, shift=True)
    assert editor.selection == {3}

    _click(editor, at=(12, 12))
    assert editor.selection == set(), "clicking background clears"


def test_a_drag_pans_rather_than_selects(session):
    _viewer, _session, editor = session
    _click(editor, 1, drag=20.0)
    assert editor.selection == set()


def test_clicks_are_ignored_while_painting_or_closed(session):
    viewer, _session, editor = session

    viewer.layers["ROIs"].mode = "paint"
    _click(editor, 1)
    assert editor.selection == set()

    viewer.layers["ROIs"].mode = "pan_zoom"
    editor.set_active(False)
    _click(editor, 1)
    assert editor.selection == set()


def test_selection_is_outlined(session):
    from pygor.gui.widgets.roi_editor import SELECTION_LAYER_NAME

    viewer, _session, editor = session
    _click(editor, 1)
    _click(editor, 3, shift=True)

    outline = viewer.layers[SELECTION_LAYER_NAME]
    assert outline.visible
    expected = np.isin(_labels(viewer), [1, 3])
    assert np.array_equal(np.asarray(outline.data).astype(bool), expected)
    assert viewer.layers.selection.active is viewer.layers["ROIs"], (
        "adding the outline must not take clicks off the ROI layer"
    )

    editor.set_active(False)
    assert not outline.visible


def test_delete_removes_the_selection_and_undo_restores_it(session):
    viewer, _session, editor = session
    before = _labels(viewer).copy()

    _click(editor, 1)
    _click(editor, 3, shift=True)
    editor.delete_selected()

    assert _present(viewer) == {2}
    assert editor.selection == set()

    viewer.layers["ROIs"].undo()
    assert np.array_equal(_labels(viewer), before), "one undo takes the whole delete back"


def test_delete_key_deletes_the_selection(session):
    viewer, _session, editor = session
    _click(editor, 2)
    editor._on_delete_key(viewer)
    assert _present(viewer) == {1, 3}


def test_delete_does_not_trigger_auto_new(session):
    viewer, session_, editor = session
    session_.docks["Plot"].auto_new_box.setChecked(True)

    _click(editor, 2)
    editor.delete_selected()

    assert str(viewer.layers["ROIs"].mode) == "pan_zoom"
    assert viewer.layers["ROIs"].selected_label == 2


def test_merge_folds_into_the_lowest_label(session):
    viewer, _session, editor = session
    _click(editor, 3)
    _click(editor, 2, shift=True)
    editor.merge_selected()

    assert _present(viewer) == {1, 2}
    assert (_labels(viewer) == 2).sum() == 16, "both ROIs' pixels now carry label 2"
    assert editor.selection == {2}


def test_merge_needs_two_rois(session):
    viewer, _session, editor = session
    _click(editor, 1)
    editor.merge_selected()
    assert _present(viewer) == {1, 2, 3}


def test_paint_into_grows_only_the_selected_roi(session):
    viewer, session_, editor = session
    layer = viewer.layers["ROIs"]
    session_.docks["Plot"].auto_new_box.setChecked(True)

    _click(editor, 2)
    editor.edit_selected("paint")
    assert not session_.docks["Plot"].auto_new_box.isChecked()
    assert str(layer.mode) == "paint"

    layer.brush_size = 1
    _stroke(layer, (4, 8))
    assert _labels(viewer)[4, 8] == 2
    assert layer.selected_label == 2, "a second stroke still lands in ROI 2"


def test_erase_from_leaves_neighbours_alone(session):
    viewer, _session, editor = session
    layer = viewer.layers["ROIs"]
    layer.brush_size = 1

    _click(editor, 2)
    editor.edit_selected("erase")
    # napari's eraser paints background; preserve_labels limits it to
    # pixels of the selected label
    for coord in (INSIDE[2], INSIDE[1]):
        with layer.block_history():
            layer.paint(coord, 0, refresh=False)

    assert _labels(viewer)[INSIDE[2]] == 0
    assert _labels(viewer)[INSIDE[1]] == 1, "erasing ROI 2 must not touch ROI 1"


def test_push_compacts_ids_and_data_follows_the_pixels(session):
    viewer, session_, editor = session
    recording = session_.recording
    recording.ipl_depths = np.array([10.0, 20.0, 30.0])
    traces = recording.traces_raw.copy()

    _click(editor, 1)
    editor.delete_selected()
    applied = editor.push()

    assert (applied.kept, applied.deleted, applied.added) == (2, 1, 0)
    assert recording.num_rois == 2
    # What was ROI 2 is ROI 1 now, in the mask, the layer and every array
    assert recording.rois[INSIDE[2]] == -1
    assert _labels(viewer)[INSIDE[2]] == 1
    assert _labels(viewer)[INSIDE[3]] == 2
    assert recording.ipl_depths.tolist() == [20.0, 30.0]
    assert np.array_equal(recording.traces_raw, traces[1:])
    assert editor.selection == set(), "old numbers no longer name the same ROIs"


def test_push_gives_a_merged_roi_no_inherited_data(session):
    viewer, session_, editor = session
    recording = session_.recording
    recording.ipl_depths = np.array([10.0, 20.0, 30.0])

    _click(editor, 1)
    _click(editor, 2, shift=True)
    editor.merge_selected()
    editor.push()

    # ROI 3 is untouched and moves to the front; the merged one goes last
    assert recording.rois[INSIDE[3]] == -1
    assert recording.rois[INSIDE[1]] == recording.rois[INSIDE[2]] == -2
    assert recording.ipl_depths[0] == 30.0
    assert np.isnan(recording.ipl_depths[1]), "a merged ROI's depth is not either part's"
    assert np.isnan(recording.traces_raw[1]).all()


def test_selection_drops_rois_removed_elsewhere(session):
    viewer, _session, editor = session
    _click(editor, 1)
    _click(editor, 2, shift=True)

    layer = viewer.layers["ROIs"]
    data = _labels(viewer).copy()
    data[data == 2] = 0
    layer.data = data
    assert editor.selection == {1}


def test_detach_releases_the_mouse_hook():
    from pygor.gui.launch import ROI_EDITOR_PANEL, attach, detach

    viewer = napari.Viewer(show=False)
    try:
        session = attach(viewer, StubRecording())
        editor = session.docks[ROI_EDITOR_PANEL]
        detach(viewer, session)
        assert editor._on_mouse not in viewer.mouse_drag_callbacks
    finally:
        viewer.close()
