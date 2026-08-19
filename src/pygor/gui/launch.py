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

import numpy as np

from pygor.gui.roi_bridge import mask_to_labels


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


def _add_image_layers(viewer, recording):
    """Add the main stack plus any stored comparison stacks."""
    if getattr(recording, "images", None) is not None:
        viewer.add_image(
            np.asarray(recording.images), name="Image stack", colormap="Greys_r"
        )

    extras = (
        ("_pre_registration_images", "Pre-registration"),
        ("_original_images", "Original"),
    )
    for attr, name in extras:
        stack = getattr(recording, attr, None)
        if stack is not None:
            viewer.add_image(
                np.asarray(stack), name=name, colormap="Greys_r", visible=False
            )

    average = None
    if hasattr(recording, "calculate_image_average"):
        try:
            average = recording.calculate_image_average()
        except Exception:
            average = None
    if average is not None:
        viewer.add_image(
            np.asarray(average), name="Average", colormap="Greys_r", visible=False
        )


def _add_roi_layer(viewer, recording):
    """Add the ROI mask as a Labels layer, returning it (or None)."""
    if getattr(recording, "rois", None) is None:
        return None
    labels = mask_to_labels(recording.rois)
    layer = viewer.add_labels(labels, name="ROIs", opacity=0.4)
    if labels.max() > 0:
        layer.selected_label = 1
    return layer


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

    _add_image_layers(viewer, recording)
    labels_layer = _add_roi_layer(viewer, recording)

    from pygor.gui.widgets.actions import ActionsDock
    from pygor.gui.widgets.traces import TraceDock

    trace_dock = TraceDock(recording, viewer, labels_layer=labels_layer)
    actions_dock = ActionsDock(
        recording,
        viewer,
        labels_layer=labels_layer,
        on_traces_changed=trace_dock.refresh,
    )

    viewer.window.add_dock_widget(trace_dock, name="Traces", area="bottom")
    viewer.window.add_dock_widget(actions_dock, name="Analysis", area="right")

    if block:
        napari.run()

    return viewer
