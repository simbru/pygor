"""A Points layer that labels each ROI with its number.

The Labels layer shows ROI extent but not identity, so telling ROI 7 from
ROI 8 on screen means clicking each one. This draws the number at each
ROI's centroid, coloured to match the ROI itself.
"""

import numpy as np
import scipy.ndimage

NUMBER_LAYER_NAME = "ROI numbers"

# White reads against both the greyscale stack and the ROI colours.
# napari text has no outline option, so a black border is not available.
TEXT_COLOR = "white"


def label_centroids(labels):
    """Return (label_values, centroid_coordinates) for a label image."""
    labels = np.asarray(labels)
    values = np.unique(labels)
    values = values[values > 0]
    if not values.size:
        return values, np.empty((0, labels.ndim))
    centroids = scipy.ndimage.center_of_mass(labels, labels, values)
    return values, np.atleast_2d(np.array(centroids))


def _number_properties(labels_layer):
    """Build the point coordinates and text for the current mask."""
    values, centroids = label_centroids(labels_layer.data)
    return values, centroids


def ensure_number_layer(viewer, labels_layer, visible=True):
    """Add or refresh the ROI number layer, returning it (None if no ROIs)."""
    if labels_layer is None:
        return None
    values, centroids = _number_properties(labels_layer)
    if not len(values):
        return viewer.layers[NUMBER_LAYER_NAME] if NUMBER_LAYER_NAME in viewer.layers else None

    features = {"label": values}
    text = {
        "string": "{label}",
        "size": 8,
        "color": TEXT_COLOR,
        "anchor": "center",
    }

    if NUMBER_LAYER_NAME in viewer.layers:
        layer = viewer.layers[NUMBER_LAYER_NAME]
        layer.data = centroids
        layer.features = features
        layer.text = text
        return layer

    selection = list(viewer.layers.selection)
    layer = viewer.add_points(
        centroids,
        name=NUMBER_LAYER_NAME,
        features=features,
        text=text,
        size=0,
        face_color="transparent",
        border_color="transparent",
        visible=visible,
    )
    viewer.layers.selection = set(selection)
    return layer
