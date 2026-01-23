import numpy as np


def _import_napari():
    """Lazy import of napari with helpful error message if not installed."""
    try:
        import napari
        from qtpy.QtCore import QEventLoop
        from qtpy.QtWidgets import QApplication
        return napari, QEventLoop, QApplication
    except ImportError as e:
        raise ImportError(
            "Napari is not installed. It is an optional dependency for pygor GUI features.\n\n"
            "To install napari, run:\n"
            "  uv pip install 'pygor[gui]'\n\n"
            "Or install napari directly:\n"
            "  uv pip install 'napari>=0.5.6' 'pyqt5>=5.15.11'\n"
        ) from e


class NapariSession:
    def __init__(self, strf_object):
        napari, QEventLoop, _ = _import_napari()
        self.napari = napari
        self.result = None  # Store the computed value
        self.viewer = napari.Viewer()
        self.strf_object = strf_object
        # Create a Qt event loop
        self.event_loop = QEventLoop()

    def run(self):
        """Launch Napari and block execution properly."""

        self.viewer.add_image(self.strf_object.strfs, name = "STRF")
        rfs = np.stack(self.strf_object.strfs_chroma)
        self.viewer.add_image(rfs, name = "Chromatic STRF")


        self.napari.run()
        # self.event_loop.exec_()  # Block until close event triggers
        return self.viewer  # Now `self.result` is updated before returning
