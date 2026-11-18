"""A Pygor menu on napari's menu bar.

The Analysis dock holds the buttons used constantly. Everything else —
saving, one-off toggles, ROI navigation — belongs in a menu rather than
adding to a panel that already has to be scrolled.
"""

import pathlib

from qtpy.QtWidgets import QFileDialog, QMessageBox

from pygor.gui.launch import PANEL_NAMES, panel_docks, restore_panels
from pygor.gui.roi_numbers import NUMBER_LAYER_NAME

MENU_TITLE = "&Pygor"


def build_pygor_menu(
    viewer,
    actions_dock,
    plot_dock,
    metrics_dock,
    preprocessing_dock,
    segmentation_dock,
    recording,
    open_roi_editor=None,
):
    """Add the Pygor menu to a viewer, returning the QMenu."""
    menu = viewer.window.main_menu.addMenu(MENU_TITLE)

    analysis = menu.addMenu("Analysis")
    _add(analysis, "Extract traces", lambda: actions_dock.run_extraction())
    _add(analysis, "Compute averages", lambda: actions_dock.run_averaging())
    _add(
        analysis,
        "Correlation projection",
        lambda: actions_dock.run_correlation_projection(),
    )
    if actions_dock.can_compute_strfs:
        analysis.addSeparator()
        _add(analysis, "Compute RFs...", lambda: actions_dock.run_rfs())
        # New, merged or redrawn ROIs only; every other RF is left as it is
        _add(
            analysis,
            "Recompute missing RFs",
            lambda: actions_dock.run_rfs(missing_only=True),
        )
        _add(analysis, "Choose noise stimulus...", lambda: _choose_noise(actions_dock))

    rois = menu.addMenu("ROIs")
    _add(rois, "New ROI", plot_dock.new_roi, "N")
    _add(rois, "Next ROI", lambda: plot_dock.step_roi(1), ".")
    _add(rois, "Previous ROI", lambda: plot_dock.step_roi(-1), ",")
    if open_roi_editor is not None:
        _add(rois, "Edit ROIs...", open_roi_editor)
    rois.addSeparator()
    _add(rois, "Segment ROIs", lambda: segmentation_dock.run_segmentation())
    _add(rois, "Push ROIs to recording", segmentation_dock.push_rois)
    _add(rois, "Restore default layers", lambda: _restore(actions_dock))

    preprocessing = menu.addMenu("Preprocessing")
    _add(
        preprocessing,
        "Preprocess with current settings",
        lambda: preprocessing_dock.run_preprocess(),
    )
    _add(
        preprocessing,
        "Register with current settings",
        lambda: preprocessing_dock.run_registration(),
    )
    preprocessing.addSeparator()
    _add(preprocessing, "Reset images", preprocessing_dock.reset_images)

    ipl = menu.addMenu("IPL depth")
    _add(ipl, "Draw boundaries", actions_dock.draw_ipl_boundaries)
    _add(ipl, "Compute depths from boundaries", actions_dock.compute_ipl_depths)
    _add(ipl, "Estimate depths automatically", actions_dock.estimate_ipl_depths)

    view = menu.addMenu("View")
    plot = view.addMenu("Plot shows")
    for label in plot_dock.available_views():
        _add(plot, label, lambda name=label: plot_dock.set_view(name))
    view.addSeparator()
    numbers_action = _add_checkable(
        view,
        "Show ROI numbers",
        lambda checked: _set_layer_visible(viewer, NUMBER_LAYER_NAME, checked),
        checked=_layer_visible(viewer, NUMBER_LAYER_NAME),
    )
    _track_layer_visibility(viewer, NUMBER_LAYER_NAME, numbers_action)
    for text, box in (
        ("Show histogram", metrics_dock.histogram_box),
        ("Show stimulus triggers", plot_dock.trigger_box),
        ("Follow frame", plot_dock.follow_box),
        ("Centre on selected ROI", plot_dock.centre_box),
        ("Auto-new ROI after each stroke", plot_dock.auto_new_box),
        ("Lock default layers", actions_dock.lock_box),
    ):
        _mirror_checkbox(view, text, box)

    panels = menu.addMenu("Panels")
    _add_panel_toggles(panels, viewer)
    panels.addSeparator()
    _add(panels, "Restore all panels", lambda: restore_panels(viewer))

    menu.addSeparator()
    _add(menu, "Parameter table...", actions_dock.open_param_editor)
    _add(menu, "Save recording as...", lambda: save_recording_as(viewer, recording))

    return menu


def _add_panel_toggles(menu, viewer):
    """One checkable entry per panel, driven by Qt's own dock action.

    A closed dock is only hidden, so its toggleViewAction already knows
    how to bring it back and stays in step when the panel is closed by
    its title-bar button instead.
    """
    docks = panel_docks(viewer)
    for name in PANEL_NAMES:
        dock = docks.get(name)
        if dock is not None:
            menu.addAction(dock.toggleViewAction())


def _add(menu, text, callback, shortcut=None):
    action = menu.addAction(text)
    if shortcut:
        action.setShortcut(shortcut)
        # napari already handles these keys on the canvas; the menu entry is
        # there to advertise them, not to bind them a second time.
        action.setShortcutVisibleInContextMenu(True)
    action.triggered.connect(lambda *_: callback())
    return action


def _add_checkable(menu, text, callback, checked=False):
    action = menu.addAction(text)
    action.setCheckable(True)
    action.setChecked(checked)
    action.toggled.connect(callback)
    return action


def _mirror_checkbox(menu, text, box):
    """Add a menu item that tracks a dock checkbox in both directions.

    setChecked only emits when the value actually changes, so the two
    staying in step does not loop.
    """
    action = _add_checkable(menu, text, box.setChecked, checked=box.isChecked())
    box.toggled.connect(action.setChecked)
    return action


def _track_layer_visibility(viewer, name, action):
    """Keep a menu item in step with a layer's own visibility toggle."""
    if name not in viewer.layers:
        return
    layer = viewer.layers[name]
    layer.events.visible.connect(lambda *_: action.setChecked(layer.visible))


def _layer_visible(viewer, name):
    return name in viewer.layers and viewer.layers[name].visible


def _set_layer_visible(viewer, name, visible):
    if name in viewer.layers:
        viewer.layers[name].visible = visible


def _choose_noise(actions_dock):
    """Pick a different noise stimulus than the remembered one."""
    bus = actions_dock.bus
    key = "strf.calculate.noise_path"
    previous = bus.get(key, "") if bus is not None else ""
    if bus is not None:
        bus.set(key, "")
    path = actions_dock.noise_path()
    if not path and bus is not None:
        bus.set(key, previous)  # cancelled: keep what was there
    actions_dock._set_status(
        f"Noise stimulus: {path}" if path else "Noise stimulus unchanged"
    )


def _restore(actions_dock):
    missing = actions_dock.missing_default_layers()
    actions_dock.restore_default_layers()
    actions_dock._set_status(
        f"Restored: {', '.join(missing)}" if missing else "Nothing to restore"
    )


def save_recording_as(viewer, recording, default=None):
    """Save the recording object, asking where to put it first.

    The suggested name ends in ``.recording.h5``, which is how a saved
    object is told apart from an IGOR export when it is opened again.
    ``default`` is the saved file the recording was opened from, if any:
    a saved object's ``filename`` is the raw file it was first analysed
    from, so without it the suggestion would land in the raw data tree.
    Returns whether it was saved.
    """
    from pygor.tui.reader import SAVED_SUFFIX

    if default is not None:
        suggestion = pathlib.Path(default)
    else:
        source = pathlib.Path(getattr(recording, "filename", "recording"))
        stem = source.name.removesuffix(SAVED_SUFFIX).split(".")[0]
        suggestion = source.with_name(stem + SAVED_SUFFIX)
    path, _ = QFileDialog.getSaveFileName(
        viewer.window._qt_window,
        "Save recording",
        str(suggestion),
        f"pygor recording (*{SAVED_SUFFIX});;All files (*)",
    )
    if not path:
        return False
    try:
        recording.save_object(path, overwrite=True)
    except Exception as exc:
        QMessageBox.critical(
            viewer.window._qt_window, "Save failed", f"Could not save:\n{exc}"
        )
        return False
    return True
