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

import numpy as np

from pygor.gui.colors import apply_roi_colormap
from pygor.gui.roi_bridge import mask_to_labels
from pygor.gui.roi_numbers import NUMBER_LAYER_NAME, ensure_number_layer

# Layer name the docks resolve the ROI mask by
ROI_LAYER_NAME = "ROIs"


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


def ensure_roi_layer(viewer, recording):
    """Return the ROI Labels layer, rebuilding it from the recording if gone.

    napari lets any layer be deleted from the layer list, and a deleted
    Labels layer takes ROI editing with it while ``recording.rois`` stays
    intact. Rebuilding from the recording is therefore always possible.

    Returns None only when the recording has no ROI mask at all.
    """
    if ROI_LAYER_NAME in viewer.layers:
        return viewer.layers[ROI_LAYER_NAME]
    if getattr(recording, "rois", None) is None:
        return None
    labels = mask_to_labels(recording.rois)
    layer = viewer.add_labels(labels, name=ROI_LAYER_NAME, opacity=0.4)
    # napari's default label colours include greys, which disappear against
    # the greyscale stack.
    apply_roi_colormap(layer)
    # Painting should not eat into ROIs already placed, and filling should
    # stay within the region under the cursor.
    layer.preserve_labels = True
    layer.contiguous = True
    if labels.max() > 0:
        layer.selected_label = 1
    return layer


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


def ensure_default_layers(viewer, recording):
    """Rebuild any of launch's own layers that are missing.

    Returns ``(layer_names, roi_layer)``. Layers the user added themselves
    are left alone, and existing default layers keep their current
    settings rather than being replaced.
    """
    names = ensure_image_layers(viewer, recording)
    roi_layer = ensure_roi_layer(viewer, recording)
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
        title = getattr(recording, "filename", None)
        title = str(getattr(title, "name", title) or "pygor")

    viewer = napari.Viewer(title=title, show=show)

    default_layers, labels_layer = ensure_default_layers(viewer, recording)

    from pygor.gui.param_bus import ParamBus

    # Every panel edits this rather than its own copy of the config
    bus = ParamBus(recording.params)

    from pygor.gui.widgets.actions import ActionsDock
    from pygor.gui.widgets.preprocessing import PreprocessingDock
    from pygor.gui.widgets.segmentation import SegmentationDock
    from pygor.gui.widgets.metrics import MetricsDock
    from pygor.gui.widgets.plot import PlotDock

    plot_dock = PlotDock(recording, viewer, labels_layer=labels_layer)
    metrics_dock = MetricsDock(recording, viewer, labels_layer=labels_layer)

    def rebind_all(layer):
        """Point every dock at an ROI layer that was rebuilt or created."""
        plot_dock.rebind(layer)
        metrics_dock.rebind(layer)
        segmentation_dock.rebind(layer)
        actions_dock._labels_layer = layer
        actions_dock._bind_labels_events(layer)

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

    plot_area = viewer.window.add_dock_widget(
        plot_dock, name="Plot", area="bottom"
    )
    # Tabs read left to right in the order they are added, so they are
    # added in the order the work is done: clean the images, find the
    # ROIs, run the analysis, then look at what came out.
    right_area = viewer.window.add_dock_widget(
        preprocessing_dock, name="Preprocessing", area="right"
    )
    viewer.window.add_dock_widget(
        segmentation_dock, name="Segmentation", area="right", tabify=True
    )
    viewer.window.add_dock_widget(
        actions_dock, name="Analysis", area="right", tabify=True
    )
    viewer.window.add_dock_widget(
        metrics_dock, name="Metrics", area="right", tabify=True
    )
    # tabify raises whatever was added last; the first step of the
    # workflow should be the one on show when the window opens
    right_area.raise_()

    from pygor.gui.menus import build_pygor_menu

    build_pygor_menu(
        viewer,
        actions_dock,
        plot_dock,
        metrics_dock,
        preprocessing_dock,
        segmentation_dock,
        recording,
    )

    _size_docks(viewer, plot_area, right_area)
    viewer.reset_view()

    if block:
        napari.run()

    return viewer


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


def _size_docks(viewer, plot_area, right_area):
    """Give the docks a usable size on open.

    Qt distributes space by size hints, which left the plots a few pixels
    tall until they were dragged out by hand every session.
    """
    from qtpy.QtCore import Qt

    window = viewer.window._qt_window
    window.resizeDocks([plot_area], [260], Qt.Vertical)
    window.resizeDocks([right_area], [360], Qt.Horizontal)
