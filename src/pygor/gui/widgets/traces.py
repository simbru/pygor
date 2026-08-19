"""Matplotlib trace viewer docked into a napari viewer.

Redraws whenever the selected label in the ROI Labels layer changes, and
moves a vertical cursor to follow the viewer's current frame.
"""

import numpy as np
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from qtpy.QtWidgets import QCheckBox, QComboBox, QHBoxLayout, QLabel, QVBoxLayout, QWidget

from pygor.gui.roi_bridge import label_to_trace_index

# Attribute names offered in the source dropdown, in display order
_TRACE_SOURCES = ("traces_znorm", "traces_raw", "traces_deconvolved", "averages")


def available_sources(recording):
    """Return trace attributes present and non-empty on the recording."""
    found = []
    for name in _TRACE_SOURCES:
        arr = getattr(recording, name, None)
        if arr is None:
            continue
        arr = np.asarray(arr)
        if arr.ndim == 2 and arr.size:
            found.append(name)
    return found


class TraceDock(QWidget):
    """Dock widget plotting the trace of the currently selected ROI."""

    def __init__(self, recording, viewer, labels_layer=None):
        super().__init__()
        self.recording = recording
        self.viewer = viewer
        self.labels_layer = labels_layer
        self._cursor = None

        self.source_box = QComboBox()
        self.source_box.addItems(available_sources(recording) or ["<no traces>"])
        self.source_box.currentIndexChanged.connect(lambda _: self.refresh())

        self.follow_box = QCheckBox("Follow frame")
        self.follow_box.setChecked(True)

        self.status = QLabel("No ROI selected")

        self.figure = Figure(figsize=(5, 2.5), layout="constrained")
        self.canvas = FigureCanvas(self.figure)
        self.ax = self.figure.add_subplot(111)

        controls = QHBoxLayout()
        controls.addWidget(QLabel("Source:"))
        controls.addWidget(self.source_box, stretch=1)
        controls.addWidget(self.follow_box)

        layout = QVBoxLayout()
        layout.addLayout(controls)
        layout.addWidget(self.status)
        layout.addWidget(self.canvas, stretch=1)
        self.setLayout(layout)

        self.connect_events()
        self.refresh()

    def connect_events(self):
        """Subscribe to label-selection and frame-position changes."""
        if self.labels_layer is not None:
            self.labels_layer.events.selected_label.connect(self._on_label_changed)
        self.viewer.dims.events.current_step.connect(self._on_frame_changed)

    @property
    def selected_label(self):
        if self.labels_layer is None:
            return 0
        return int(self.labels_layer.selected_label)

    def current_trace(self):
        """Return (trace, source_name) for the selected ROI, or (None, name)."""
        source = self.source_box.currentText()
        arr = getattr(self.recording, source, None)
        if arr is None:
            return None, source
        arr = np.asarray(arr)
        index = label_to_trace_index(self.selected_label)
        if index < 0 or index >= arr.shape[0]:
            return None, source
        return arr[index], source

    def _on_label_changed(self, event=None):
        self.refresh()

    def _on_frame_changed(self, event=None):
        if not self.follow_box.isChecked() or self._cursor is None:
            return
        frame = self._current_frame()
        if frame is None:
            return
        self._cursor.set_xdata([frame, frame])
        self.canvas.draw_idle()

    def _current_frame(self):
        step = self.viewer.dims.current_step
        if not step:
            return None
        return step[0]

    def refresh(self):
        """Redraw the axis for the current selection and source."""
        self.ax.clear()
        self._cursor = None
        trace, source = self.current_trace()

        if trace is None:
            self.status.setText(f"No trace for label {self.selected_label} in {source}")
            self.ax.set_axis_off()
            self.canvas.draw_idle()
            return

        n_rois = np.asarray(getattr(self.recording, source)).shape[0]
        self.status.setText(
            f"ROI {self.selected_label} of {n_rois} — {source}"
        )
        self.ax.set_axis_on()
        self.ax.plot(trace, lw=0.8, color="tab:blue")
        self.ax.set_xlabel("Frame")
        self.ax.set_ylabel(source.replace("traces_", ""))
        self.ax.margins(x=0)

        frame = self._current_frame()
        if frame is not None and self.follow_box.isChecked():
            self._cursor = self.ax.axvline(frame, color="tab:red", lw=0.8)

        self.canvas.draw_idle()
