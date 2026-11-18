"""Assemble a single napari viewer for a pygor recording.

The existing ``view_*`` methods each spin up their own viewer for one
purpose. This builds one viewer holding the image stack, ROI labels and
docked widgets for traces and analysis actions, so a whole recording can
be inspected and processed without leaving the window.

Usage
-----
>>> import pygor
>>> from pygor.gui import launch
>>> rec = pygor.load("my_recording.h5")
>>> viewer = launch(rec)
"""

import warnings
from dataclasses import dataclass

import numpy as np

from pygor.gui.colors import apply_roi_colormap
from pygor.gui.roi_bridge import mask_to_labels
from pygor.gui.roi_numbers import NUMBER_LAYER_NAME, ensure_number_layer

# Layer name the docks resolve the ROI mask by
ROI_LAYER_NAME = "ROIs"

#: The plot dock, along the bottom
PLOT_PANEL = "Plot"

#: The tabbed docks on the right, in the order the work is done
RIGHT_PANELS = ("Preprocessing", "Segmentation", "Analysis", "Metrics")

#: Every ROI's RF as a mosaic; a right-hand tab, only for recordings with STRFs
RF_PANEL = "RFs"

#: Click-to-select ROI editing; opened from the ROIs menu, not on launch
ROI_EDITOR_PANEL = "ROI editor"

#: Every panel the GUI can create
PANEL_NAMES = (PLOT_PANEL, *RIGHT_PANELS, RF_PANEL)


def _import_napari():
    """Import napari, raising a useful message when the extra is missing."""
    try:
        import napari
    except ImportError as exc:
        raise ImportError(
            "napari is required for the pygor GUI.\n"
            "Install it with:\n"
            "  uv pip install 'pygor[gui]'\n"
            "or:\n"
            "  uv pip install 'napari>=0.5.6' 'pyqt5>=5.15.11'"
        ) from exc
    return napari


def _image_sources(recording):
    """Yield (name, array, visible) for each image layer launch provides."""
    if getattr(recording, "images", None) is not None:
        yield "Image stack", np.asarray(recording.images), True

    extras = (
        ("_pre_registration_images", "Pre-registration"),
        ("_original_images", "Original"),
    )
    for attr, name in extras:
        stack = getattr(recording, attr, None)
        if stack is not None:
            yield name, np.asarray(stack), False

    # Returns None when the recording has no repetitions to average over.
    average = None
    if hasattr(recording, "calculate_image_average"):
        try:
            average = recording.calculate_image_average()
        except Exception as exc:
            warnings.warn(
                f"Could not compute average projection: {exc}", stacklevel=2
            )
    if average is not None:
        yield "Average", np.asarray(average), False


def ensure_image_layers(viewer, recording):
    """Add any missing image layers, leaving existing ones untouched.

    Returns the names of the image layers this recording provides,
    whether or not they had to be created.
    """
    names = []
    for name, data, visible in _image_sources(recording):
        names.append(name)
        if name in viewer.layers:
            continue
        viewer.add_image(data, name=name, colormap="Greys_r", visible=visible)
    return names


def ensure_roi_layer(viewer, recording, empty=False):
    """Return the ROI Labels layer, rebuilding it from the recording if gone.

    napari lets any layer be deleted from the layer list, and a deleted
    Labels layer takes ROI editing with it while ``recording.rois`` stays
    intact. Rebuilding from the recording is therefore always possible.

    Returns None when the recording has no ROI mask at all, unless
    ``empty`` asks for a blank layer the size of the image to draw into.
    """
    if ROI_LAYER_NAME in viewer.layers:
        return viewer.layers[ROI_LAYER_NAME]
    if getattr(recording, "rois", None) is not None:
        labels = mask_to_labels(recording.rois)
    elif empty and getattr(recording, "images", None) is not None:
        labels = np.zeros(np.shape(recording.images)[-2:], dtype=np.int32)
    else:
        return None
    layer = viewer.add_labels(labels, name=ROI_LAYER_NAME, opacity=0.4)
    # napari's default label colours include greys, which disappear against
    # the greyscale stack.
    apply_roi_colormap(layer)
    # Painting should not eat into ROIs already placed, and filling should
    # stay within the region under the cursor.
    layer.preserve_labels = True
    layer.contiguous = True
    layer.mouse_wheel_callbacks.append(scale_brush_on_shift_wheel)
    if labels.max() > 0:
        layer.selected_label = 1
    return layer


def scale_brush_on_shift_wheel(layer, event):
    """Shift + scroll grows or shrinks the brush while painting or erasing.

    napari has ``[`` and ``]`` for this but nothing on the wheel. Shift is
    free there: vispy's camera ignores wheel events with a modifier held,
    so this does not fight the zoom, and napari keeps Ctrl for the frame
    slider. Steps are a fifth of the current size, so large brushes do not
    take dozens of notches to resize.
    """
    if "Shift" not in event.modifiers or layer.mode not in ("paint", "erase"):
        return
    # Qt reports a Shift-scroll as horizontal on some platforms
    delta = event.delta[1] or event.delta[0]
    if not delta:
        return
    size = layer.brush_size
    step = max(1, round(size / 5))
    upper = max(layer.data.shape[-2:])
    layer.brush_size = int(np.clip(size + step if delta > 0 else size - step, 1, upper))
    event.handled = True


def refresh_image_layers(viewer, recording):
    """Point the image layers at the current arrays.

    Preprocessing and registration replace ``recording.images`` rather
    than editing it, so a layer built earlier still holds the old array
    and would show stale pixels. Backup stacks appear as layers only once
    a destructive step has created them.
    """
    for name, data, _visible in _image_sources(recording):
        if name in viewer.layers:
            viewer.layers[name].data = data
    ensure_image_layers(viewer, recording)


def ensure_default_layers(viewer, recording, empty_rois=False):
    """Rebuild any of launch's own layers that are missing.

    Returns ``(layer_names, roi_layer)``. Layers the user added themselves
    are left alone, and existing default layers keep their current
    settings rather than being replaced. ``empty_rois`` creates a blank
    ROI layer for a recording that has no ROIs yet.
    """
    names = ensure_image_layers(viewer, recording)
    roi_layer = ensure_roi_layer(viewer, recording, empty=empty_rois)
    if roi_layer is not None:
        names.append(ROI_LAYER_NAME)
        if ensure_number_layer(viewer, roi_layer) is not None:
            names.append(NUMBER_LAYER_NAME)
        # Rebuilt layers are appended on top, which would put an image
        # stack over the ROI labels and hide them. Numbers go above both.
        _move_to_top(viewer, ROI_LAYER_NAME)
        _move_to_top(viewer, NUMBER_LAYER_NAME)
    return names, roi_layer


def _move_to_top(viewer, name):
    """Move a layer to the top of the layer list, if it is present."""
    if name not in viewer.layers:
        return
    index = viewer.layers.index(viewer.layers[name])
    if index != len(viewer.layers) - 1:
        # move() inserts before the target index, so the end is len()
        viewer.layers.move(index, len(viewer.layers))


def launch(recording, show=True, block=False, title=None):
    """Open a napari viewer wired up to a pygor recording.

    Parameters
    ----------
    recording : pygor.classes.core_data.Core
        The recording to inspect. Layers and widgets read from it live,
        so analysis run from the GUI mutates this object.
    show : bool
        Show the viewer window. Set False for headless construction.
    block : bool
        Start the Qt event loop and block until the window closes. Leave
        False inside Jupyter or when driving the viewer from a script.
    title : str or None
        Window title. Defaults to the recording filename.

    Returns
    -------
    napari.Viewer
        The configured viewer. Docked widgets are available under
        ``viewer.window.dock_widgets``.
    """
    napari = _import_napari()

    if title is None:
        title = recording_title(recording)

    viewer = napari.Viewer(title=title, show=show)
    attach(viewer, recording)

    if block:
        napari.run()

    return viewer


def recording_title(recording):
    """The recording's file name, for a window title."""
    title = getattr(recording, "filename", None)
    return str(getattr(title, "name", title) or "pygor")


@dataclass
class Session:
    """What :func:`attach` put on a viewer, so :func:`detach` can take it off."""

    recording: object
    bus: object
    docks: dict
    menu: object

    @property
    def actions(self):
        return self.docks[RIGHT_PANELS[2]]


def attach(viewer, recording):
    """Build pygor's layers, docks and menu for a recording on a viewer.

    Returns a :class:`Session` for :func:`detach`. The viewer is assumed
    to hold nothing of another recording's; detach that one first.
    """
    default_layers, labels_layer = ensure_default_layers(viewer, recording)

    from pygor.gui.param_bus import ParamBus

    # Every panel edits this rather than its own copy of the config
    bus = ParamBus(recording.params)

    from pygor.gui.widgets.actions import ActionsDock
    from pygor.gui.widgets.preprocessing import PreprocessingDock
    from pygor.gui.widgets.segmentation import SegmentationDock
    from pygor.gui.widgets.metrics import MetricsDock
    from pygor.gui.widgets.plot import PlotDock

    from pygor.gui.metrics import has_strfs

    plot_dock = PlotDock(recording, viewer, labels_layer=labels_layer)
    metrics_dock = MetricsDock(recording, viewer, labels_layer=labels_layer)
    # Panels are offered by what the recording carries, not by its class
    rf_dock = None
    if has_strfs(recording):
        from pygor.gui.widgets.rf_grid import RFGridDock

        rf_dock = RFGridDock(recording, viewer, labels_layer=labels_layer)
        # Rejecting an RF in the plot dock crosses out its tile
        plot_dock.on_rf_rejected = rf_dock.mark_rejected

    def rebind_all(layer):
        """Point every dock at an ROI layer that was rebuilt or created."""
        plot_dock.rebind(layer)
        metrics_dock.rebind(layer)
        if rf_dock is not None:
            rf_dock.rebind(layer)
        segmentation_dock.rebind(layer)
        roi_editor.rebind(layer)
        actions_dock._labels_layer = layer
        actions_dock._bind_labels_events(layer)

    def create_roi_layer():
        """A blank ROI layer to draw into, for a recording with no ROIs."""
        names, layer = ensure_default_layers(viewer, recording, empty_rois=True)
        if layer is not None:
            actions_dock.default_layers = names
            rebind_all(layer)
        return layer

    plot_dock.create_roi_layer = create_roi_layer

    actions_dock = ActionsDock(
        recording,
        viewer,
        labels_layer=labels_layer,
        # Extraction changes which metrics exist, not just the plot
        on_traces_changed=lambda: (plot_dock.refresh(), metrics_dock.refresh()),
        on_layer_restored=rebind_all,
        default_layers=default_layers,
        on_depths_updated=lambda: _show_metric(metrics_dock, "IPL depth"),
        bus=bus,
        on_strfs_changed=lambda: _strfs_changed(
            plot_dock, metrics_dock, rf_dock, actions_dock
        ),
    )

    # napari owns the left dock area with its layer controls and layer
    # list, so anything put there competes with them for height. Both
    # pygor panels go right instead, tabbed so only one is visible at a
    # time, and the trace plot gets the full width along the bottom.
    segmentation_dock = SegmentationDock(
        recording,
        viewer,
        bus,
        labels_layer=labels_layer,
        on_rois_changed=lambda: (
            actions_dock.refresh_numbers(),
            metrics_dock.refresh(),
            plot_dock.refresh(),
        ),
        on_layer_created=rebind_all,
    )

    preprocessing_dock = PreprocessingDock(
        recording,
        viewer,
        bus=bus,
        on_images_changed=lambda: refresh_image_layers(viewer, recording),
    )

    from pygor.gui.widgets.roi_editor import RoiEditorDock

    roi_editor = RoiEditorDock(
        recording,
        viewer,
        labels_layer=labels_layer,
        auto_new_box=plot_dock.auto_new_box,
        on_push=segmentation_dock.push_rois,
    )

    viewer.window.add_dock_widget(plot_dock, name=PLOT_PANEL, area="bottom")
    # Tabs read left to right in the order they are added, so they are
    # added in the order the work is done: clean the images, find the
    # ROIs, run the analysis, then look at what came out.
    panels = dict(
        zip(
            RIGHT_PANELS,
            (preprocessing_dock, segmentation_dock, actions_dock, metrics_dock),
        )
    )
    first = None
    for name, dock in panels.items():
        area = viewer.window.add_dock_widget(
            dock, name=name, area="right", tabify=first is not None
        )
        first = first or area
    if rf_dock is not None:
        viewer.window.add_dock_widget(rf_dock, name=RF_PANEL, area="right", tabify=True)
        panels[RF_PANEL] = rf_dock
    # The editor is opened on request from the ROIs menu, so it starts
    # closed; closing a dock only hides it, so showing it puts the tab back
    editor_dock = viewer.window.add_dock_widget(
        roi_editor, name=ROI_EDITOR_PANEL, area="right", tabify=True
    )
    editor_dock.hide()
    # tabify raises whatever was added last; the first step of the
    # workflow should be the one on show when the window opens
    first.raise_()

    def open_roi_editor():
        editor_dock.show()
        editor_dock.raise_()
        roi_editor.set_active(True)

    from pygor.gui.menus import build_pygor_menu

    menu = build_pygor_menu(
        viewer,
        actions_dock,
        plot_dock,
        metrics_dock,
        preprocessing_dock,
        segmentation_dock,
        recording,
        open_roi_editor=open_roi_editor,
    )

    _size_docks(viewer)
    viewer.reset_view()

    return Session(
        recording=recording,
        bus=bus,
        docks={PLOT_PANEL: plot_dock, **panels, ROI_EDITOR_PANEL: roi_editor},
        menu=menu,
    )


def detach(viewer, session):
    """Take a recording's docks, layers and menu off the viewer.

    ROI edits not yet pushed are written to the recording first, so a
    save offered afterwards still has them. Every layer goes, including
    any the user added: they were drawn over this recording's images.
    """
    actions = session.actions
    actions.sync_rois_from_layer()
    actions.disconnect_events()
    session.docks[PLOT_PANEL].disconnect_events()
    if RF_PANEL in session.docks:
        session.docks[RF_PANEL].disconnect_events()
    if ROI_EDITOR_PANEL in session.docks:
        session.docks[ROI_EDITOR_PANEL].disconnect_events()

    window = viewer.window
    for dock in session.docks.values():
        window.remove_dock_widget(dock)
    if actions._param_dock is not None:
        window.remove_dock_widget(actions._param_dock)
    window.main_menu.removeAction(session.menu.menuAction())
    viewer.layers.clear()


def _strfs_changed(plot_dock, metrics_dock, rf_dock, actions_dock):
    """Redraw everything that shows an RF, after STRFs were computed.

    The views re-fetch when ``strfs`` is a new array, which both a full
    and a spliced recompute assign.
    """
    plot_dock.refresh()
    metrics_dock.reload_metrics()
    metrics_dock.refresh()
    if rf_dock is not None:
        rf_dock.redraw()
    else:
        actions_dock._set_status(
            "RFs computed. Reopen the recording to get the RF panel."
        )


def _show_metric(metrics_dock, label):
    """Bring a metric to the front after something recomputed it.

    Freshly computed depths are only useful if they can be seen, so the
    metrics panel switches to them and shows the distribution.
    """
    metrics_dock.reload_metrics()
    index = metrics_dock.metric_box.findText(label)
    if index < 0:
        return
    metrics_dock.metric_box.setCurrentIndex(index)
    metrics_dock.histogram_box.setChecked(True)
    metrics_dock.refresh()


def panel_docks(viewer):
    """The QDockWidget for each pygor panel, keyed by name.

    ``window.dock_widgets`` hands back the inner widget; the dock wrapper
    that knows how to hide and show itself is its parent. Reaching for
    the private ``_dock_widgets`` gets the wrapper directly but napari
    warns on every access.
    """
    inner = getattr(viewer.window, "dock_widgets", {})
    docks = {}
    for name in PANEL_NAMES:
        widget = inner.get(name)
        if widget is not None and widget.parent() is not None:
            docks[name] = widget.parent()
    return docks


def _size_docks(viewer):
    """Give the docks a usable size on open.

    Qt distributes space by size hints, which left the plots a few pixels
    tall until they were dragged out by hand every session.
    """
    from qtpy.QtCore import Qt

    docks = panel_docks(viewer)
    window = viewer.window._qt_window
    plot = docks.get(PLOT_PANEL)
    right = docks.get(RIGHT_PANELS[0])
    if plot is not None:
        window.resizeDocks([plot], [260], Qt.Vertical)
    if right is not None:
        window.resizeDocks([right], [360], Qt.Horizontal)


def restore_panels(viewer):
    """Reopen any pygor panel that was closed, and re-apply the sizing.

    Closing a dock only hides it — the widget and everything it holds
    survive — so this is a matter of showing it again. It needs saying
    because napari's Window menu lists only napari's own docks: without
    this the way back is Qt's dock context menu, which nothing advertises.

    Returns the names it reopened.
    """
    docks = panel_docks(viewer)
    restored = [
        name
        for name in PANEL_NAMES
        if name in docks and docks[name].isHidden()
    ]
    for name in restored:
        docks[name].show()
    if restored:
        _size_docks(viewer)
    return restored
