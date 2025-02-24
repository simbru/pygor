import napari
from qtpy.QtWidgets import QApplication
from qtpy.QtCore import QEventLoop

class NapariSession:
    def __init__(self):
        self.result = None  # Store the computed value
        self.viewer = napari.Viewer()

        # Create a Qt event loop
        self.event_loop = QEventLoop()

        # Override the close event
        original_close_event = self.viewer.window._qt_window.closeEvent

        def custom_close_event(event):
            self.on_close()  # Run computations on close
            original_close_event(event)  # Ensure proper closing
            self.event_loop.quit()  # Exit the event loop

        self.viewer.window._qt_window.closeEvent = custom_close_event

    def process_data(self):
        """Placeholder for computation logic based on user selection."""
        print("Processing user selection...")
        self.result = 42  # Example result

    def on_close(self):
        """Function triggered when the viewer closes."""
        print("Viewer closed. Running final computation...")
        self.process_data()  # Compute the final result

    def run(self):
        """Launch Napari and block execution properly."""
        napari.run()
        self.event_loop.exec_()  # Block until close event triggers
        return self.result  # Now `self.result` is updated before returning

# Usage
session = NapariSession()
result = session.run()  # Blocks execution until Napari closes
print(f"Returned result: {result}")  # Now prints after window closes