import numpy as np
import napari
from qtpy.QtWidgets import QApplication
from qtpy.QtCore import QEventLoop

class NapariSession:
    def __init__(self, strf_object):
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


        napari.run()
        # self.event_loop.exec_()  # Block until close event triggers
        return self.viewer  # Now `self.result` is updated before returning