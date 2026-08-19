"""A Pygor menu on napari's menu bar.

The Analysis dock holds the buttons used constantly. Everything else —
saving, one-off toggles, ROI navigation — belongs in a menu rather than
adding to a panel that already has to be scrolled.
"""

import pathlib

from qtpy.QtWidgets import QFileDialog, QMessageBox

from pygor.gui.roi_numbers import NUMBER_LAYER_NAME

MENU_TITLE = "&Pygor"


def build_pygor_menu(
    viewer, actions_dock, plot_dock, population_dock, preprocessing_dock, recording
):
    """Add the Pygor menu to a viewer, returning the QMenu."""
    menu = viewer.window.main_menu.addMenu(MENU_TITLE)

    analysis = menu.addMenu("Analysis")
    _add(analysis, "Segment ROIs...", lambda: actions_dock.run_segmentation())
    _add(analysis, "Extract traces", lambda: actions_dock.run_extraction())
    _add(analysis, "Compute averages", lambda: actions_dock.run_averaging())
    _add(
        analysis,
        "Correlation projection",
        lambda: actions_dock.run_correlation_projection(),
    )

    rois = menu.addMenu("ROIs")
    _add(rois, "New ROI", plot_dock.new_roi, "N")
    _add(rois, "Next ROI", lambda: plot_dock.step_roi(1), ".")
    _add(rois, "Previous ROI", lambda: plot_dock.step_roi(-1), ",")
    rois.addSeparator()
    _add(rois, "Push ROIs to recording", lambda: _push(actions_dock))
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
    for label in (plot_dock.TRACE, plot_dock.AVERAGE):
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
        ("Show histogram", population_dock.histogram_box),
        ("Follow frame", plot_dock.follow_box),
        ("Centre on selected ROI", plot_dock.centre_box),
        ("Auto-new ROI after each stroke", plot_dock.auto_new_box),
        ("Lock default layers", actions_dock.lock_box),
    ):
        _mirror_checkbox(view, text, box)

    menu.addSeparator()
    _add(menu, "Parameters...", actions_dock.open_param_editor)
    _add(menu, "Save recording as...", lambda: _save_as(viewer, recording))

    return menu


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


def _push(actions_dock):
    if actions_dock.sync_rois_from_layer():
        actions_dock._set_status("Pushed ROIs to recording")
    else:
        actions_dock._set_status("ROIs already match the recording")


def _restore(actions_dock):
    missing = actions_dock.missing_default_layers()
    actions_dock.restore_default_layers()
    actions_dock._set_status(
        f"Restored: {', '.join(missing)}" if missing else "Nothing to restore"
    )


def _save_as(viewer, recording):
    """Save the recording object, asking where to put it first."""
    suggested = str(pathlib.Path(getattr(recording, "filename", "recording")).stem)
    path, _ = QFileDialog.getSaveFileName(
        viewer.window._qt_window,
        "Save recording",
        suggested,
        "HDF5 (*.h5);;All files (*)",
    )
    if not path:
        return
    try:
        recording.save_object(path, overwrite=True)
    except Exception as exc:
        QMessageBox.critical(
            viewer.window._qt_window, "Save failed", f"Could not save:\n{exc}"
        )
