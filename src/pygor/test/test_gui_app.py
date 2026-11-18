"""Tests for the pygor-gui app shell: one window, recordings swapped in it.

Headless via the Qt offscreen platform, like ``test_gui_spike``, whose stub
recording and event helpers these reuse.
"""

import os

import numpy as np
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

napari = pytest.importorskip("napari")
pytest.importorskip("magicgui")

from pygor.test.test_gui_spike import (  # noqa: E402
    StubRecording,
    _process_events,
    _stroke,
    _wait_for,
)


def _menu_titles(viewer):
    return [action.text() for action in viewer.window.main_menu.actions()]


@pytest.fixture
def viewer():
    viewer = napari.Viewer(show=False)
    yield viewer
    viewer.close()


def test_detach_leaves_nothing_behind(viewer):
    from pygor.gui.launch import PANEL_NAMES, attach, detach
    from pygor.gui.menus import MENU_TITLE

    session = attach(viewer, StubRecording())
    session.actions.open_param_editor()
    detach(viewer, session)
    _process_events()

    assert len(viewer.layers) == 0, "the lock put layers back"
    assert not set(viewer.window.dock_widgets) & {*PANEL_NAMES, "Parameters"}
    assert MENU_TITLE not in _menu_titles(viewer)


def test_reattach_builds_one_set_for_the_new_recording(viewer):
    from pygor.gui.launch import attach, detach
    from pygor.gui.menus import MENU_TITLE

    detach(viewer, attach(viewer, StubRecording(shape=(16, 16))))
    # Rebinding the navigation keys would raise if detach had not unbound them
    session = attach(viewer, StubRecording(shape=(24, 20)))

    assert viewer.layers["Image stack"].data.shape[1:] == (24, 20)
    assert _menu_titles(viewer).count(MENU_TITLE) == 1
    assert viewer.window.dock_widgets["Plot"] is session.docks["Plot"]


def test_detached_plot_no_longer_follows_the_frame(viewer, monkeypatch):
    from pygor.gui.launch import attach, detach
    from pygor.gui.widgets.plot import PlotDock

    # Patched on the class before attaching: napari holds a bound method as
    # (object, name) and looks the name up on each emit
    calls = []
    monkeypatch.setattr(PlotDock, "_on_frame_changed", lambda *_: calls.append(1))
    session = attach(viewer, StubRecording())
    detach(viewer, session)
    viewer.add_image(np.zeros((5, 4, 4)))
    viewer.dims.set_current_step(0, 3)
    assert calls == []


def test_detach_keeps_unpushed_roi_edits(viewer):
    from pygor.gui.launch import attach, detach

    recording = StubRecording()
    session = attach(viewer, recording)
    labels = viewer.layers["ROIs"]
    data = labels.data.copy()
    data[-3:, -3:] = 9
    labels.data = data
    detach(viewer, session)
    # Ids are compacted on the way in, so the drawn ROI is -4, not -9; its
    # pixels are what has to survive, as one ROI
    drawn = recording.rois[-3:, -3:]
    assert (drawn < 0).all() and np.unique(drawn).size == 1
    assert (recording.rois == drawn[0, 0]).sum() == 9


def test_new_roi_makes_a_layer_on_a_recording_without_rois(viewer):
    """A raw recording has no ROI layer; New ROI used to do nothing."""
    from pygor.gui.launch import attach

    recording = StubRecording(shape=(16, 12))
    recording.rois = None
    session = attach(viewer, recording)
    assert "ROIs" not in viewer.layers

    plot = session.docks["Plot"]
    plot.new_roi()
    layer = viewer.layers["ROIs"]
    assert layer.data.shape == (16, 12)
    assert layer.selected_label == 1
    assert layer.mode == "paint"
    assert viewer.layers.selection.active is layer

    # Every dock follows the new layer, so a stroke reaches the recording
    _stroke(layer, (8, 6))
    assert session.docks["Segmentation"].push_rois()
    assert (recording.rois < 0).any()


def test_new_roi_enters_paint_mode(viewer):
    from pygor.gui.launch import attach

    session = attach(viewer, StubRecording())
    layer = viewer.layers["ROIs"]
    layer.mode = "pan_zoom"
    session.docks["Plot"].new_roi()
    assert layer.mode == "paint"

    layer.mode = "polygon"
    session.docks["Plot"].new_roi()
    assert layer.mode == "polygon", "a drawing mode already chosen is kept"


def _wheel(dy=0.0, dx=0.0, modifiers=("Shift",)):
    from types import SimpleNamespace

    return SimpleNamespace(modifiers=modifiers, delta=(dx, dy), handled=False)


def test_shift_wheel_scales_the_brush(viewer):
    from pygor.gui.launch import attach, scale_brush_on_shift_wheel

    attach(viewer, StubRecording())
    layer = viewer.layers["ROIs"]
    assert scale_brush_on_shift_wheel in layer.mouse_wheel_callbacks

    layer.mode = "paint"
    layer.brush_size = 10
    event = _wheel(dy=1)
    scale_brush_on_shift_wheel(layer, event)
    assert layer.brush_size == 12 and event.handled
    scale_brush_on_shift_wheel(layer, _wheel(dy=-1))
    assert layer.brush_size == 10

    # Some platforms turn a Shift-scroll horizontal
    scale_brush_on_shift_wheel(layer, _wheel(dx=1))
    assert layer.brush_size == 12

    layer.brush_size = 1
    scale_brush_on_shift_wheel(layer, _wheel(dy=-1))
    assert layer.brush_size == 1, "never below one pixel"


def test_wheel_without_shift_or_outside_painting_leaves_the_brush(viewer):
    from pygor.gui.launch import attach, scale_brush_on_shift_wheel

    attach(viewer, StubRecording())
    layer = viewer.layers["ROIs"]
    layer.mode = "paint"
    layer.brush_size = 10
    event = _wheel(dy=1, modifiers=())
    scale_brush_on_shift_wheel(layer, event)
    assert layer.brush_size == 10 and not event.handled

    layer.mode = "pan_zoom"
    scale_brush_on_shift_wheel(layer, _wheel(dy=1))
    assert layer.brush_size == 10


def _app(monkeypatch):
    from pygor.gui import app as app_module

    # Nothing is saved in tests, and the recent list must not leak into the
    # user's own settings
    monkeypatch.setattr(app_module, "_SETTINGS", ("pygor", "gui-tests"))
    return app_module.open_app(show=False)


def test_app_opens_empty_with_its_menu(monkeypatch):
    from pygor.gui.app import RECORDING_MENU_TITLE

    app = _app(monkeypatch)
    try:
        assert app.session is None
        assert _menu_titles(app.viewer)[0] == RECORDING_MENU_TITLE
        assert not app.save_action.isEnabled()
        assert len(app.viewer.layers) == 0
    finally:
        app.viewer.close()


def test_recording_shortcuts_do_not_clash_with_napari(monkeypatch):
    # Qt runs neither action when two share a shortcut, silently
    from napari._app_model import get_app_model
    from qtpy.QtGui import QKeySequence

    app = _app(monkeypatch)
    try:
        napari_keys = {
            QKeySequence(str(kb.keybinding)).toString()
            for kb in get_app_model().keybindings
        }
        ours = [a.shortcut().toString() for a in app.menu.actions()
                if not a.shortcut().isEmpty()]
        assert ours, "the Recording menu has shortcuts to check"
        assert not set(ours) & napari_keys
    finally:
        app.viewer.close()


def test_app_swaps_one_recording_for_the_next(monkeypatch, tmp_path):
    from pygor.tui import reader

    app = _app(monkeypatch)
    try:
        first, second = StubRecording(), StubRecording(shape=(24, 20))
        second.filename = "second.h5"
        queue = [first, second]
        monkeypatch.setattr(reader, "load_recording", lambda *a, **k: queue.pop(0))
        monkeypatch.setattr(app, "_confirm_close", lambda: True)

        app.open(tmp_path / "first.h5")
        assert _wait_for(lambda: app.recording is first)
        assert app.save_action.isEnabled()

        app.open(tmp_path / "second.h5")
        assert _wait_for(lambda: app.recording is second)
        assert app.viewer.title == "second.h5"
        assert app.viewer.layers["Image stack"].data.shape[1:] == (24, 20)
        assert app.recent_files()[0] == str((tmp_path / "second.h5").resolve())

        app.close_recording(confirm=False)
        assert app.session is None and len(app.viewer.layers) == 0
    finally:
        app._settings.clear()
        app.viewer.close()


def test_failed_open_keeps_the_current_recording(monkeypatch, tmp_path):
    from qtpy.QtWidgets import QMessageBox

    from pygor.tui import reader

    app = _app(monkeypatch)
    try:
        current = StubRecording()
        app.attach_recording(current)

        def broken(*_args, **_kwargs):
            raise OSError("not a recording")

        shown = []
        monkeypatch.setattr(reader, "load_recording", broken)
        monkeypatch.setattr(app, "_confirm_close", lambda: True)
        monkeypatch.setattr(QMessageBox, "critical", lambda *a: shown.append(a))
        app.open(tmp_path / "bad.h5", background=False)

        assert shown
        assert app.recording is current
    finally:
        app._settings.clear()
        app.viewer.close()


def test_cancelling_the_save_prompt_keeps_the_recording(monkeypatch, tmp_path):
    from qtpy.QtWidgets import QMessageBox

    app = _app(monkeypatch)
    try:
        current = StubRecording()
        app.attach_recording(current)
        monkeypatch.setattr(
            QMessageBox, "question", lambda *a, **k: QMessageBox.Cancel
        )
        assert app.open(tmp_path / "other.h5") is None
        assert app.close_recording() is False
        assert app.recording is current
    finally:
        app._settings.clear()
        app.viewer.close()


def test_open_dialog_fits_its_fields_to_the_file(tmp_path):
    from pygor.gui.widgets.open_dialog import OpenDialog

    raw = tmp_path / "a.smp"
    saved = tmp_path / "a.recording.h5"
    raw.touch()
    saved.touch()

    dialog = OpenDialog(path=raw, class_name="Core")
    try:
        assert dialog.class_box.isEnabled()
        assert dialog.n_colours is None, "Core takes no colour count"
        dialog.class_box.setCurrentText("STRF")
        assert dialog.n_colours == 1

        dialog.path_edit.setText(str(saved))
        assert not dialog.class_box.isEnabled()
        assert not dialog.config_edit.isEnabled()
        assert dialog.n_colours is None

        dialog.path_edit.setText(str(tmp_path / "missing.h5"))
        assert not dialog._open_button.isEnabled()
    finally:
        dialog.deleteLater()
