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

from pygor.gui.colors import roi_colormap
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
    layer.colormap = roi_colormap()
    # Painting should not eat into ROIs already placed, and filling should
    # stay within the region under the cursor.
    layer.preserve_labels = True
    layer.contiguous = True
    if labels.max() > 0:
        layer.selected_label = 1
    return layer


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

    from pygor.gui.widgets.actions import ActionsDock
    from pygor.gui.widgets.traces import TraceDock

    trace_dock = TraceDock(recording, viewer, labels_layer=labels_layer)
    actions_dock = ActionsDock(
        recording,
        viewer,
        labels_layer=labels_layer,
        on_traces_changed=trace_dock.refresh,
        on_layer_restored=trace_dock.rebind,
        default_layers=default_layers,
    )

    viewer.window.add_dock_widget(trace_dock, name="Traces", area="bottom")
    viewer.window.add_dock_widget(actions_dock, name="Analysis", area="right")

    from pygor.gui.menus import build_pygor_menu

    build_pygor_menu(viewer, actions_dock, trace_dock, recording)

    if block:
        napari.run()

    return viewer
