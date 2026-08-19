"""The window's per-ROI plot, and the ROI navigation that drives it.

One matplotlib axis showing whatever the selected ROI looks like. The
view selector carries a single entry for now and hides itself until there
is a second, which is where the per-ROI detail views in
`dev/gui_workflow_map.md` belong: RF maps per channel, temporal kernels,
tuning functions.

Population-level plots live with the population table instead, since a
distribution is read alongside the values it summarises.
"""

import numpy as np
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from qtpy.QtWidgets import (
    QCheckBox,
    QStackedWidget,
    QComboBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from pygor.gui.roi_bridge import label_to_trace_index

# Attribute names offered in the source dropdown, in display order
_TRACE_SOURCES = ("traces_znorm", "traces_raw", "traces_deconvolved", "averages")

# Name of the crosshair layer marking the centred ROI, and its size as a
# fraction of image width so it scales with the field of view
CENTRE_LAYER_NAME = "Centred ROI"

# Labels modes that add to the mask, as opposed to erasing or inspecting
_DRAWING_MODES = ("paint", "polygon", "fill")


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


class PlotDock(QWidget):
    """Dock widget holding the window's single plot axis."""

    TRACE = "Trace"

    def __init__(self, recording, viewer, labels_layer=None):
        super().__init__()
        self.recording = recording
        self.viewer = viewer
        self._labels_layer = labels_layer
        self._cursor = None
        self._background = None

        self.mode_box = QComboBox()
        self.mode_box.addItems([self.TRACE])
        self.mode_box.setToolTip("Which view this plot shows")
        self.mode_box.currentIndexChanged.connect(self._on_view_changed)

        self.source_box = QComboBox()
        self.source_box.addItems(available_sources(recording) or ["<no traces>"])
        self.source_box.currentIndexChanged.connect(lambda _: self.refresh())

        self.follow_box = QCheckBox("Follow frame")
        self.follow_box.setChecked(False)
        self.follow_box.setToolTip(
            "Track the viewer's current frame with a cursor on the trace. "
            "Costs a blit per frame change."
        )
        self.follow_box.toggled.connect(self._on_follow_toggled)

        self.status = QLabel("No ROI selected")

        self.roi_spin = QSpinBox()
        self.roi_spin.setPrefix("ROI ")
        self.roi_spin.setMinimum(1)
        self.roi_spin.setMaximum(max(1, self.n_rois))
        self.roi_spin.valueChanged.connect(self._on_spin_changed)

        self.prev_button = QPushButton("< Prev")
        self.next_button = QPushButton("Next >")
        self.prev_button.setToolTip("Previous ROI (,)")
        self.next_button.setToolTip("Next ROI (.)")
        self.prev_button.clicked.connect(lambda: self.step_roi(-1))
        self.next_button.clicked.connect(lambda: self.step_roi(1))

        self.auto_new_box = QCheckBox("Auto-new")
        self.auto_new_box.setChecked(True)
        self.auto_new_box.setToolTip(
            "After each completed stroke, select the next free label. Suits "
            "drawing many blob ROIs quickly; turn off to refine one ROI "
            "across several strokes."
        )

        self.new_button = QPushButton("New ROI")
        self.new_button.setToolTip(
            "Select the next unused label (n), so a new ROI is drawn as its "
            "own ROI rather than merging into the current one."
        )
        self.new_button.clicked.connect(self.new_roi)

        self.centre_box = QCheckBox("Centre view")
        self.centre_box.setChecked(False)
        self.centre_box.setToolTip(
            "Move the camera to the selected ROI and mark it with a crosshair."
        )
        self.centre_box.toggled.connect(self._on_centre_toggled)

        # Guards the spinbox <-> layer selection loop from recursing
        self._syncing = False

        self.figure = Figure(figsize=(5, 2.5), layout="constrained")
        self.canvas = FigureCanvas(self.figure)
        # Below roughly this height matplotlib cannot fit axes into the
        # figure and constrained layout gives up, printing a warning over
        # the canvas.
        self.canvas.setMinimumHeight(140)
        self.ax = self.figure.add_subplot(111)

        # One page of controls per view, swapped with the view itself
        trace_controls = QWidget()
        trace_row = QHBoxLayout()
        trace_row.setContentsMargins(0, 0, 0, 0)
        trace_row.addWidget(QLabel("Source:"))
        trace_row.addWidget(self.source_box, stretch=1)
        trace_row.addWidget(self.follow_box)
        trace_controls.setLayout(trace_row)

        self.controls_stack = QStackedWidget()
        self.controls_stack.addWidget(trace_controls)

        # A one-entry selector is just noise; it appears once a second view
        # is registered.
        self.view_label = QLabel("View:")
        single_view = self.mode_box.count() < 2
        self.view_label.setVisible(not single_view)
        self.mode_box.setVisible(not single_view)

        controls = QHBoxLayout()
        controls.addWidget(self.view_label)
        controls.addWidget(self.mode_box)
        controls.addWidget(self.controls_stack, stretch=1)

        navigation = QHBoxLayout()
        navigation.addWidget(self.prev_button)
        navigation.addWidget(self.roi_spin)
        navigation.addWidget(self.next_button)
        navigation.addWidget(self.new_button)
        navigation.addWidget(self.auto_new_box)
        navigation.addWidget(self.centre_box)
        navigation.addStretch(1)

        layout = QVBoxLayout()
        layout.addLayout(controls)
        layout.addLayout(navigation)
        layout.addWidget(self.status)
        layout.addWidget(self.canvas, stretch=1)
        self.setLayout(layout)

        self.connect_events()
        self.refresh()

    @property
    def labels_layer(self):
        """The ROI layer, or None once it has been removed from the viewer.

        A layer deleted from the layer list stays alive as a Python object,
        so holding a direct reference would let edits carry on landing on a
        detached layer where nothing shows them.
        """
        layer = self._labels_layer
        if layer is None or layer not in self.viewer.layers:
            return None
        return layer

    def bind_layer(self, layer):
        """Attach to an ROI layer, subscribing to its editing events."""
        self._labels_layer = layer
        if layer is None:
            return
        layer.events.selected_label.connect(self._on_label_changed)
        layer.events.paint.connect(self._on_paint)
        layer.events.mode.connect(self._on_mode_changed)
        self.refresh()

    def connect_events(self):
        """Subscribe to label-selection and frame-position changes."""
        if self._labels_layer is not None:
            self.bind_layer(self._labels_layer)
        self.viewer.dims.events.current_step.connect(self._on_frame_changed)
        self.viewer.bind_key(",", lambda _viewer: self.step_roi(-1))
        self.viewer.bind_key(".", lambda _viewer: self.step_roi(1))
        self.viewer.bind_key("n", lambda _viewer: self.new_roi())

    @property
    def n_rois(self):
        """Number of ROIs available in the current trace source, else 0."""
        source = self.source_box.currentText() if hasattr(self, "source_box") else None
        arr = getattr(self.recording, source, None) if source else None
        if arr is None:
            return int(getattr(self.recording, "num_rois", 0) or 0)
        return int(np.asarray(arr).shape[0])

    @property
    def max_label(self):
        """Highest label present in the ROI mask, 0 when there are none."""
        if self.labels_layer is None:
            return 0
        return int(np.asarray(self.labels_layer.data).max())

    def _on_mode_changed(self, event=None):
        """Move off an occupied label when a drawing mode is entered.

        The selection starts on ROI 1 so its trace can be inspected. With
        auto-new on, the first stroke would otherwise be absorbed into that
        ROI instead of starting a new one.
        """
        layer = self.labels_layer
        if layer is None or not self.auto_new_box.isChecked():
            return
        if str(layer.mode) not in _DRAWING_MODES:
            return
        data = np.asarray(layer.data)
        if np.any(data == self.selected_label):
            self.new_roi()

    def _on_paint(self, event=None):
        """Advance to a fresh label after a completed stroke.

        napari commits a stroke's staged undo history on mouse release, and
        emits ``paint`` from there, so this fires once per stroke rather
        than once per mouse move. Erasing is excluded: it removes pixels
        rather than creating an ROI.
        """
        layer = self.labels_layer
        if layer is None or not self.auto_new_box.isChecked():
            return
        if str(layer.mode) == "erase":
            return
        self.new_roi()

    def _roi_upper_bound(self):
        """Highest selectable ROI label.

        Includes the current selection, which may point at a label that has
        no pixels and no trace yet: New ROI selects the next free label
        before anything is drawn into it, and clamping to the labels that
        already exist would snap the selection straight back.
        """
        return max(1, self.n_rois, self.max_label, self.selected_label)

    def new_roi(self):
        """Select the next unused label, ready to draw a fresh ROI.

        Without this, painting after selecting ROI 11 keeps adding to ROI
        11 rather than creating ROI 12. napari has no built-in binding for
        this, so it is provided here.
        """
        if self.labels_layer is None:
            return
        target = self.max_label + 1
        self.roi_spin.setMaximum(max(self.roi_spin.maximum(), target))
        self.set_roi(target)
        self.roi_spin.setMaximum(self._roi_upper_bound())

    def step_roi(self, delta):
        """Move the selection by delta ROIs, wrapping at both ends."""
        total = self._roi_upper_bound()
        if total < 1 or self.labels_layer is None:
            return
        current = max(1, min(self.selected_label, total))
        target = (current - 1 + delta) % total + 1
        self.set_roi(target)

    def set_roi(self, label):
        """Select an ROI by label, updating the layer and optionally the camera."""
        if self.labels_layer is None:
            return
        self.labels_layer.selected_label = int(label)
        if self.centre_box.isChecked():
            self._centre_on_roi(int(label))

    def _centre_on_roi(self, label):
        """Move the camera to the centroid of the given ROI and mark it."""
        centre = self._roi_centre(label)
        if centre is None:
            return
        self.viewer.camera.center = (0, float(centre[0]), float(centre[1]))
        self._update_centre_marker(centre)

    def _roi_centre(self, label):
        """Return the (row, col) centroid of a label, or None if absent."""
        data = np.asarray(self.labels_layer.data)
        coords = np.argwhere(data == label)
        if not coords.size:
            return None
        centre = coords.mean(axis=0)
        return float(centre[-2]), float(centre[-1])

    def _centre_layer(self):
        """Return the crosshair layer if it exists, else None."""
        if CENTRE_LAYER_NAME in self.viewer.layers:
            return self.viewer.layers[CENTRE_LAYER_NAME]
        return None

    def _update_centre_marker(self, centre):
        """Place the crosshair on the centred ROI.

        Adding a layer steals the layer-list selection, which would knock
        the Labels layer out of picker mode, so the previous selection is
        restored afterwards.
        """
        point = np.array([[centre[0], centre[1]]])
        size = self._marker_size()
        layer = self._centre_layer()
        if layer is not None:
            layer.data = point
            layer.size = size
            layer.visible = True
            return

        selection = list(self.viewer.layers.selection)
        self.viewer.add_points(
            point,
            name=CENTRE_LAYER_NAME,
            symbol="cross",
            size=size,
            face_color="white",
            border_color="transparent",
            opacity=1.0,
        )
        self.viewer.layers.selection = set(selection)

    def roi_color(self):
        """Colour the selected ROI is drawn in, for matching the trace to it."""
        layer = self.labels_layer
        if layer is None or self.selected_label < 1:
            return "tab:blue"
        return tuple(float(c) for c in layer.get_color(self.selected_label))

    def _marker_size(self):
        """Scale the crosshair to a thirtieth of the image width."""
        width = np.asarray(self.labels_layer.data).shape[-1]
        return width / 30

    def rebind(self, layer):
        """Point the dock at a rebuilt ROI layer."""
        self.bind_layer(layer)

    def _remove_centre_marker(self):
        """Hide the crosshair without disturbing the layer selection."""
        layer = self._centre_layer()
        if layer is not None:
            layer.visible = False

    def _on_spin_changed(self, value):
        if self._syncing:
            return
        self.set_roi(value)

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
        # Rows follow the mask the traces were extracted from, not the
        # label number: erasing an ROI leaves a gap in the labels but not
        # in the rows.
        index = label_to_trace_index(self.selected_label, self.recording.rois)
        if index < 0 or index >= arr.shape[0]:
            return None, source
        return arr[index], source

    def _on_label_changed(self, event=None):
        self._syncing = True
        try:
            label = self.selected_label
            if 1 <= label <= self.roi_spin.maximum():
                self.roi_spin.setValue(label)
        finally:
            self._syncing = False
        self.refresh()

    def _on_frame_changed(self, event=None):
        if not self.follow_box.isChecked() or self._cursor is None:
            return
        frame = self._current_frame()
        if frame is None:
            return
        self._cursor.set_xdata([frame, frame])
        self._blit_cursor()

    def _blit_cursor(self):
        """Redraw only the cursor over a cached background.

        A full draw re-renders the whole trace, which is tens of thousands
        of points for a typical recording and far too slow to do on every
        frame change.
        """
        if self._background is None:
            self.canvas.draw()
            return
        self.canvas.restore_region(self._background)
        self.ax.draw_artist(self._cursor)
        self.canvas.blit(self.ax.bbox)

    def _capture_background(self):
        """Cache the axis without the cursor, for blitting to restore."""
        if self._cursor is None:
            self._background = None
            return
        self._cursor.set_visible(False)
        self.canvas.draw()
        self._background = self.canvas.copy_from_bbox(self.ax.bbox)
        self._cursor.set_visible(True)
        self.ax.draw_artist(self._cursor)
        self.canvas.blit(self.ax.bbox)

    def _on_follow_toggled(self, checked):
        """Create or drop the cursor when the follow toggle changes."""
        self.refresh()

    def _on_centre_toggled(self, checked):
        """Show or hide the crosshair, centring immediately when enabled."""
        if checked:
            self.set_roi(self.selected_label)
        else:
            self._remove_centre_marker()

    def resizeEvent(self, event):
        """Invalidate the cached background, which is size-dependent."""
        super().resizeEvent(event)
        self._background = None

    def _current_frame(self):
        step = self.viewer.dims.current_step
        if not step:
            return None
        return step[0]

    # ------------------------------------------------------------------
    # View switching
    # ------------------------------------------------------------------

    @property
    def view(self):
        """Label of the view currently on the axis."""
        return self.mode_box.currentText()

    def set_view(self, name):
        """Switch the plot to a named view."""
        index = self.mode_box.findText(name)
        if index >= 0:
            self.mode_box.setCurrentIndex(index)

    def _on_view_changed(self, index=None):
        self.controls_stack.setCurrentIndex(self.mode_box.currentIndex())
        self.refresh()

    def refresh(self):
        """Redraw the axis for the current view."""
        # Labels can outrun the trace array: an ROI drawn by hand exists in
        # the mask before traces are extracted for it.
        self.roi_spin.setMaximum(self._roi_upper_bound())
        self.ax.clear()
        self._cursor = None
        self._background = None

        self._draw_trace()

    def _draw_trace(self):
        """Time course of the selected ROI."""
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
        self.ax.plot(trace, lw=0.8, color=self.roi_color())
        self.ax.set_xlabel("Frame")
        self.ax.set_ylabel(source.replace("traces_", ""))
        self.ax.margins(x=0)

        frame = self._current_frame()
        if frame is not None and self.follow_box.isChecked():
            self._cursor = self.ax.axvline(
                frame, color="tab:red", lw=0.8, animated=True
            )
            self._capture_background()
        else:
            self._background = None
            self.canvas.draw_idle()
