"""The window's per-ROI plot, and the ROI navigation that drives it.

One matplotlib figure showing whatever the selected ROI looks like. Views
are offered by what the recording carries, not by its class: the RF view
appears only when there are STRFs, so a full-field recording never shows
it. More per-ROI detail views from `dev/gui_workflow_map.md` (temporal
kernels, tuning functions) belong in the same selector.

Distribution plots live with the metrics table instead, since a
distribution is read alongside the values it summarises.
"""

import numpy as np
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from qtpy.QtCore import Qt, QTimer
from qtpy.QtWidgets import (
    QCheckBox,
    QStackedWidget,
    QComboBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSlider,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from pygor.gui.metrics import has_strfs
from pygor.gui.rf import RFData, draw_fit, draw_map, fit_legend, symmetric_limit
from pygor.gui.roi_bridge import label_to_trace_index
from pygor.gui.timebase import average_axis, frame_to_seconds, trace_axis
from pygor.gui.triggers import trigger_times, within_loop_times

# Attribute names offered in the source dropdown, in display order
_TRACE_SOURCES = ("traces_znorm", "traces_raw", "traces_deconvolved", "averages")

# Drawing every repetition of a noise stimulus is slow and unreadable
_MAX_TRIALS_DRAWN = 40

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
    AVERAGE = "Average"
    RF = "RF"

    def __init__(self, recording, viewer, labels_layer=None):
        super().__init__()
        self.recording = recording
        self.viewer = viewer
        self._labels_layer = labels_layer
        self._cursor = None
        self._background = None
        # Maps, movies and fits, fetched on first use of the RF view
        self._rf = RFData(recording)
        # Set by launch: makes a blank ROI layer when there is none, so
        # New ROI works on a recording that has never been segmented
        self.create_roi_layer = None

        self.mode_box = QComboBox()
        self.mode_box.addItems(self.available_views())
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

        self.trigger_box = QCheckBox("Triggers")
        self.trigger_box.setChecked(True)
        self.trigger_box.setToolTip(
            "Mark the stimulus triggers. On a trace these are their times "
            "in the recording; on an average they are their offsets within "
            "one loop."
        )
        self.trigger_box.toggled.connect(lambda _: self.refresh())

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

        self.trials_box = QCheckBox("Show trials")
        self.trials_box.setChecked(True)
        self.trials_box.setToolTip(
            "Draw the individual repetitions behind the average"
        )
        self.trials_box.toggled.connect(lambda _: self.refresh())

        average_controls = QWidget()
        average_row = QHBoxLayout()
        average_row.setContentsMargins(0, 0, 0, 0)
        average_row.addWidget(self.trials_box)
        average_row.addStretch(1)
        average_controls.setLayout(average_row)

        # The RF movie: the STRF frame by frame, beside the time-collapsed map
        self.play_button = QPushButton("Play")
        self.play_button.setCheckable(True)
        self.play_button.setToolTip("Play the STRF through time, looping")
        self.play_button.toggled.connect(self._on_play_toggled)
        self.frame_slider = QSlider(Qt.Horizontal)
        self.frame_slider.setToolTip(
            "STRF frame. Opens on the ROI's strongest frame"
        )
        self.frame_slider.valueChanged.connect(self._show_rf_frame)
        self.frame_label = QLabel()
        self.rf_timer = QTimer(self)
        self.rf_timer.timeout.connect(self._advance_rf_frame)
        # (image artist, its STRF frames) per colour, for the movie column
        self._rf_movies = []
        # (marker line, x of each frame) per colour, in the kernel column
        self._rf_frame_lines = []
        # STRF ROI index the RF view is showing, for the reject toggle
        self._rf_index = None

        # Marks the shown RF as not real, in the recording itself; saved with it
        self.reject_button = QPushButton("Reject RF")
        self.reject_button.setCheckable(True)
        self.reject_button.setToolTip(
            "Mark this ROI's RF as not real (all colour channels). Stored in the "
            "recording when it is saved; the STRF itself is kept. The paper pipeline "
            "then treats the cell as having no RF."
        )
        self.reject_button.toggled.connect(self._on_reject_toggled)
        # Set by launch, so the RF panel can mark the tile
        self.on_rf_rejected = None

        # Off by default: whether an RF is real is judged on the map and the
        # movie, and the fit is not to sway that
        self.fits_box = QCheckBox("Fits")
        self.fits_box.setToolTip(
            "Draw the Gaussian fit (solid) and the region its fit index is "
            "scored over (dashed), with the fit's numbers"
        )
        self.fits_box.toggled.connect(lambda _: self.refresh())

        rf_controls = QWidget()
        movie_row = QHBoxLayout()
        movie_row.setContentsMargins(0, 0, 0, 0)
        movie_row.addWidget(self.play_button)
        movie_row.addWidget(self.frame_slider, stretch=1)
        movie_row.addWidget(self.frame_label)
        movie_row.addWidget(self.fits_box)
        movie_row.addWidget(self.reject_button)
        rf_controls.setLayout(movie_row)

        # Pages in the same order as available_views()
        self.controls_stack = QStackedWidget()
        self.controls_stack.addWidget(trace_controls)
        self.controls_stack.addWidget(average_controls)
        self.controls_stack.addWidget(rf_controls)

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
        # Applies to both views, so it sits outside the swapped pages
        controls.addWidget(self.trigger_box)

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

    def disconnect_events(self):
        """Let go of the viewer, so another recording can be attached to it.

        Layer events go with the layers; the frame slider and the key
        bindings belong to the viewer and would otherwise keep driving
        this dock, and napari refuses to bind a key twice.
        """
        self.viewer.dims.events.current_step.disconnect(self._on_frame_changed)
        self.play_button.setChecked(False)
        for key in (",", ".", "n"):
            self.viewer.bind_key(key, None)

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
        than once per mouse move. Only drawing modes count: erasing removes
        pixels rather than creating an ROI, and edits made from code (the
        ROI editor's delete and merge) also emit ``paint`` while the layer
        sits in pan/zoom.
        """
        layer = self.labels_layer
        if layer is None or not self.auto_new_box.isChecked():
            return
        if str(layer.mode) not in _DRAWING_MODES:
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

        A recording with no ROIs has no layer to draw on, so one is made.
        Outside a drawing mode, selecting a label shows nothing, so the
        layer is made active and switched to painting.
        """
        layer = self.labels_layer
        if layer is None and self.create_roi_layer is not None:
            layer = self.create_roi_layer()
        if layer is None:
            return
        target = self.max_label + 1
        self.roi_spin.setMaximum(max(self.roi_spin.maximum(), target))
        self.set_roi(target)
        self.roi_spin.setMaximum(self._roi_upper_bound())
        self.viewer.layers.selection.active = layer
        if layer.mode not in ("paint", "fill", "polygon"):
            layer.mode = "paint"

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
        # The trace axis is in seconds, so the cursor has to be too
        position = frame_to_seconds(self.recording, frame)
        self._cursor.set_xdata([position, position])
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

    def available_views(self):
        """Views this recording has data for, in selector order."""
        views = [self.TRACE, self.AVERAGE]
        if has_strfs(self.recording):
            views.append(self.RF)
        return views

    def _reload_views(self):
        """Offer a view once its data exists, e.g. STRFs computed after opening."""
        for name in self.available_views():
            if self.mode_box.findText(name) < 0:
                self.mode_box.addItem(name)
        several = self.mode_box.count() > 1
        self.view_label.setVisible(several)
        self.mode_box.setVisible(several)

    def _on_view_changed(self, index=None):
        self.controls_stack.setCurrentIndex(self.mode_box.currentIndex())
        self.refresh()

    def reload_sources(self):
        """Rebuild the source list, keeping the current choice if it stays.

        A recording opened before its traces exist has none to offer, so
        the list has to be rebuilt once extraction has run rather than
        left as it was at construction.
        """
        available = available_sources(self.recording) or ["<no traces>"]
        current = [
            self.source_box.itemText(i) for i in range(self.source_box.count())
        ]
        if available == current:
            return

        selected = self.source_box.currentText()
        self.source_box.blockSignals(True)
        try:
            self.source_box.clear()
            self.source_box.addItems(available)
            index = self.source_box.findText(selected)
            self.source_box.setCurrentIndex(index if index >= 0 else 0)
        finally:
            self.source_box.blockSignals(False)

    def refresh(self):
        """Redraw the axis for the current view."""
        self.reload_sources()
        # Labels can outrun the trace array: an ROI drawn by hand exists in
        # the mask before traces are extracted for it.
        self.roi_spin.setMaximum(self._roi_upper_bound())
        self._reload_views()
        # The RF view lays out one axis per colour, so the figure is rebuilt
        # rather than its one axis cleared
        self.figure.clear()
        self.ax = self.figure.add_subplot(111)
        self._cursor = None
        self._background = None
        self._rf_movies = []

        if self.view != self.RF:
            self.play_button.setChecked(False)
        if self.view == self.RF:
            self._draw_rf()
            return
        if self.view == self.AVERAGE:
            self._draw_average()
            return
        self._draw_trace()

    def _draw_triggers(self, times):
        """Mark stimulus triggers, unless the view has them turned off.

        Drawn as one LineCollection rather than an axvline apiece: a long
        recording can carry thousands of triggers, and that many artists
        makes every redraw crawl.
        """
        if not self.trigger_box.isChecked() or times is None or not len(times):
            return
        self.ax.vlines(
            times,
            0,
            1,
            transform=self.ax.get_xaxis_transform(),
            color="black",
            lw=0.6,
            alpha=0.7,
            zorder=0,
        )

    def _draw_average(self):
        """Mean across stimulus repetitions, with the repetitions behind it.

        ``compute_snippets_and_averages`` returns snippets shaped
        (n_rois, n_loops, snippet_length) and averages (n_rois, snippet_length),
        despite what its docstring says.
        """
        averages = np.asarray(getattr(self.recording, "averages", None))
        if averages.ndim != 2:
            self.status.setText(
                "No averages — run Analysis, Compute averages, on a "
                "recording with repeated trials"
            )
            self.ax.set_axis_off()
            self.canvas.draw_idle()
            return

        index = label_to_trace_index(self.selected_label, self.recording.rois)
        if index < 0 or index >= averages.shape[0]:
            self.status.setText(f"No average for label {self.selected_label}")
            self.ax.set_axis_off()
            self.canvas.draw_idle()
            return

        self.ax.set_axis_on()
        colour = self.roi_color()
        shown = 0

        snippets = getattr(self.recording, "snippets", None)
        snippets = None if snippets is None else np.asarray(snippets)
        if (
            self.trials_box.isChecked()
            and snippets is not None
            and snippets.ndim == 3
            and index < snippets.shape[0]
        ):
            trials = snippets[index]
            # Thousands of noise loops would be an unreadable smear and slow
            # to draw, so only a sample of them is shown.
            step = max(1, int(np.ceil(trials.shape[0] / _MAX_TRIALS_DRAWN)))
            trial_x, _ = average_axis(self.recording, trials.shape[1])
            for trial in trials[::step]:
                self.ax.plot(trial_x, trial, lw=0.5, color="0.6", alpha=0.5, zorder=1)
                shown += 1

        mean = averages[index]
        x, xlabel = average_axis(self.recording, mean.shape[0])
        self.ax.plot(x, mean, lw=1.4, color=colour, zorder=2)
        self._draw_triggers(within_loop_times(self.recording))
        self.ax.set_xlabel(xlabel)
        self.ax.set_ylabel("Average")
        self.ax.margins(x=0)

        total = 0 if snippets is None or snippets.ndim != 3 else snippets.shape[1]
        detail = f", {shown} of {total} trials shown" if shown else ""
        self.status.setText(
            f"ROI {self.selected_label} average over {total} trials{detail}"
        )
        self.canvas.draw_idle()

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
        x, xlabel = trace_axis(self.recording, np.asarray(trace).shape[0])
        self.ax.plot(x, trace, lw=0.8, color=self.roi_color())
        self._draw_triggers(trigger_times(self.recording))
        self.ax.set_xlabel(xlabel)
        self.ax.set_ylabel(source.replace("traces_", ""))
        self.ax.margins(x=0)

        frame = self._current_frame()
        if frame is not None and self.follow_box.isChecked():
            self._cursor = self.ax.axvline(
                frame_to_seconds(self.recording, frame),
                color="tab:red",
                lw=0.8,
                animated=True,
            )
            self._capture_background()
        else:
            self._background = None
            self.canvas.draw_idle()

    def _draw_rf(self):
        """The selected ROI's RF in each colour, with its Gaussian fit.

        One row per colour: the time-collapsed map the fit is made on, and
        beside it the STRF itself, one frame at a time, driven by the slider
        and Play. Solid ellipse: the fitted Gaussian. Dashed: the
        data-derived footprint the fit index is scored over.
        """
        from pygor.strf.rejection import is_rejected

        data = self._rf.refresh()
        index = label_to_trace_index(self.selected_label, self.recording.rois)
        if index < 0 or index >= data.n_rois:
            self._rf_index = None
            self.reject_button.setEnabled(False)
            self.status.setText(f"No STRF for label {self.selected_label}")
            self.ax.set_axis_off()
            self.canvas.draw_idle()
            return
        self._rf_index = index
        rejected = is_rejected(self.recording, index)
        self.reject_button.blockSignals(True)
        self.reject_button.setEnabled(True)
        self.reject_button.setChecked(rejected)
        self.reject_button.blockSignals(False)

        self.figure.clear()
        axes = self.figure.subplots(
            data.n_colours, 3, squeeze=False, gridspec_kw={"width_ratios": [1, 1, 1.1]}
        )
        fits = self.fits_box.isChecked()
        self._rf_movies = []
        self._rf_frame_lines = []
        for colour in range(data.n_colours):
            i = data.flat(index, colour)
            static, movie, kernel = axes[colour]
            draw_map(static, data.maps[i], symmetric_limit(data.maps[i]))
            frames = data.movies[i]
            self._rf_movies.append(
                (draw_map(movie, frames[0], symmetric_limit(frames)), frames)
            )
            self._draw_kernel(kernel, data, i, len(frames))
            static.set_ylabel(f"ch{colour}", fontsize=8)
            if fits:
                for ax in (static, movie):
                    draw_fit(ax, data, i)
                static.set_title(self._rf_title(data, i), fontsize=8, loc="left")
        axes[0, 0].set_title(
            axes[0, 0].get_title(loc="left") or "time-collapsed", fontsize=8, loc="left"
        )
        axes[0, 1].set_title("over time", fontsize=8, loc="left")
        axes[0, 2].set_title("kernel: centre —, surround - -", fontsize=8, loc="left")
        if fits:
            self.figure.legend(
                handles=fit_legend(data.n_sigma), loc="outside lower center",
                ncols=2, fontsize=7, frameon=False,
            )

        n_frames = len(self._rf_movies[0][1])
        self.frame_slider.blockSignals(True)
        self.frame_slider.setRange(0, n_frames - 1)
        if not self.play_button.isChecked():
            # Each ROI opens on its strongest frame, not on the empty first one
            peak = np.abs(np.stack([f for _, f in self._rf_movies]))
            peak = np.nan_to_num(peak).max(axis=(0, 2, 3))
            self.frame_slider.setValue(int(np.argmax(peak)))
        self.frame_slider.blockSignals(False)
        self._show_rf_frame(self.frame_slider.value(), draw=False)

        if rejected:
            self.figure.suptitle("RF rejected", color="tab:red", fontsize=9)
        self.status.setText(
            f"ROI {self.selected_label} of {data.n_rois}"
            + (" — RF rejected (save the recording to keep this)" if rejected else "")
        )
        self.canvas.draw_idle()

    def _on_reject_toggled(self, reject):
        """Reject or restore the shown RF on the recording object."""
        from pygor.strf.rejection import set_rejected

        if self._rf_index is None:
            return
        set_rejected(self.recording, self._rf_index, reject)
        if self.on_rf_rejected is not None:
            self.on_rf_rejected()
        self.refresh()

    def _draw_kernel(self, ax, data, i, n_frames):
        """Centre (solid) and surround (dashed) time courses, with a marker at
        the movie's frame so the two can be read together."""
        pair = data.kernels(i)
        if pair is None:
            ax.text(0.5, 0.5, "no kernel", ha="center", va="center",
                    transform=ax.transAxes, fontsize=8, color="0.5")
            ax.set_axis_off()
            return
        times = data.frame_times_ms(pair.shape[-1])
        x = times if times is not None else np.arange(pair.shape[-1])
        for lobe, style in zip(pair, ("-", "--")):
            ax.plot(x, lobe, color="k", lw=1, ls=style)
        ax.axhline(0, color="0.7", lw=0.6)
        if times is not None:
            ax.axvline(0, color="0.7", lw=0.6, ls=":")
        ax.set_xlabel("ms" if times is not None else "frame", fontsize=7)
        ax.tick_params(labelsize=7)
        ax.spines[["top", "right"]].set_visible(False)
        # x of each movie frame, for the marker; the kernel may differ in length
        frame_x = (data.frame_times_ms(n_frames) if times is not None
                   else np.arange(n_frames))
        line = ax.axvline(frame_x[0], color="tab:orange", lw=1)
        self._rf_frame_lines.append((line, frame_x))

    def _show_rf_frame(self, frame, draw=True):
        """Put one STRF frame in the movie column, without redrawing the rest."""
        if not self._rf_movies:
            return
        for artist, frames in self._rf_movies:
            artist.set_data(frames[min(frame, len(frames) - 1)])
        for line, frame_x in getattr(self, "_rf_frame_lines", []):
            x = frame_x[min(frame, len(frame_x) - 1)]
            line.set_xdata([x, x])
        self.frame_label.setText(self._rf.frame_label(frame))
        if draw:
            self.canvas.draw_idle()

    def _advance_rf_frame(self):
        if not self._rf_movies:
            return
        self.frame_slider.setValue(
            (self.frame_slider.value() + 1) % (self.frame_slider.maximum() + 1)
        )

    def _on_play_toggled(self, playing):
        self.play_button.setText("Pause" if playing else "Play")
        if not playing:
            self.rf_timer.stop()
            return
        self.rf_timer.start(self._rf.refresh().frame_interval_ms())

    @staticmethod
    def _rf_title(data, i):
        """The fit's own numbers, for judging it; no call is made on them."""
        index = data.fits["index"][i]
        if not np.isfinite(index):
            return "no fit"
        return (f"fit index {index:.2f}, snr {data.fits['snr'][i]:.1f}, "
                f"peak {data.edge[i]:.0f} px from edge")
