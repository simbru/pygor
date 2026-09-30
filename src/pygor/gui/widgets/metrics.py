"""Whole-recording view of the ROIs, one metric at a time.

The plot dock answers "what does this ROI do"; this answers "how is this
metric distributed across the recording". One metric is selected at a time
and drives three things together: a sortable table of per-ROI values,
optionally the colour of the ROIs themselves, and a histogram.

The histogram sits under the table it summarises, and can be hidden when
the table alone is wanted.
"""

import numpy as np
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from qtpy.QtCore import Qt, Signal
from qtpy.QtWidgets import (
    QAbstractItemView,
    QCheckBox,
    QComboBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from pygor.gui.colors import apply_metric_colormap, apply_roi_colormap
from pygor.gui.metrics import available_metrics, compute_metric
from pygor.gui.roi_bridge import roi_ids_in_order
from pygor.gui import triggers


class MetricsDock(QWidget):
    """Dock showing one metric across every ROI."""

    #: Emitted whenever the selected metric or its values change
    metric_changed = Signal()

    def __init__(self, recording, viewer, labels_layer=None):
        super().__init__()
        self.recording = recording
        self.viewer = viewer
        self._labels_layer = labels_layer
        self._specs = ()
        self._values = None
        self._labels = np.empty(0, dtype=int)
        # Guards the table <-> layer selection loop from recursing
        self._syncing = False

        self.metric_box = QComboBox()
        self.metric_box.currentIndexChanged.connect(lambda _: self._on_metric_changed())

        self.compute_button = QPushButton("Compute")
        self.compute_button.setToolTip(
            "Run this metric. Shown for metrics too slow to run on selection."
        )
        self.compute_button.clicked.connect(lambda: self.refresh(force=True))
        self.compute_button.setVisible(False)

        self.colour_box = QCheckBox("Colour ROIs by metric")
        self.colour_box.setToolTip(
            "Replace the identity colours with a scale over this metric. "
            "ROIs with no value are drawn transparent."
        )
        self.colour_box.toggled.connect(self._on_colour_toggled)

        self.status = QLabel("No metric")
        self.status.setWordWrap(True)

        self.histogram_box = QCheckBox("Show histogram")
        self.histogram_box.setChecked(True)
        self.histogram_box.setToolTip(
            "Distribution of the selected metric, with the selected ROI marked"
        )
        self.histogram_box.toggled.connect(self._on_histogram_toggled)

        self.figure = Figure(figsize=(4, 2), layout="constrained")
        self.canvas = FigureCanvas(self.figure)
        self.canvas.setMinimumHeight(150)
        self.ax = self.figure.add_subplot(111)

        self.table = QTableWidget(0, 2)
        self.table.setHorizontalHeaderLabels(["ROI", "Value"])
        self.table.verticalHeader().setVisible(False)
        self.table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.table.setSelectionMode(QAbstractItemView.SingleSelection)
        self.table.setSortingEnabled(True)
        self.table.itemSelectionChanged.connect(self._on_row_selected)
        self.table.setMinimumHeight(160)

        self.trigger_status = QLabel()
        self.trigger_status.setWordWrap(True)

        self.trigger_table = QTableWidget(0, 5)
        self.trigger_table.setHorizontalHeaderLabels(
            ["#", "Time (s)", "Interval (s)", "Frame", "Loop"]
        )
        self.trigger_table.verticalHeader().setVisible(False)
        self.trigger_table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.trigger_table.setSelectionMode(QAbstractItemView.SingleSelection)
        self.trigger_table.setSortingEnabled(True)
        self.trigger_table.itemSelectionChanged.connect(self._on_trigger_selected)

        controls = QHBoxLayout()
        controls.addWidget(QLabel("Metric:"))
        controls.addWidget(self.metric_box, stretch=1)
        controls.addWidget(self.compute_button)

        roi_layout = QVBoxLayout()
        roi_layout.addLayout(controls)
        roi_layout.addWidget(self.colour_box)
        roi_layout.addWidget(self.status)
        roi_layout.addWidget(self.table, stretch=2)
        roi_layout.addWidget(self.histogram_box)
        roi_layout.addWidget(self.canvas, stretch=1)
        roi_page = QWidget()
        roi_page.setLayout(roi_layout)

        trigger_layout = QVBoxLayout()
        trigger_layout.addWidget(self.trigger_status)
        trigger_layout.addWidget(self.trigger_table, stretch=1)
        trigger_page = QWidget()
        trigger_page.setLayout(trigger_layout)

        # Triggers are a property of the recording rather than of any one
        # ROI, so they get their own page rather than a column in a
        # per-ROI table that would repeat the same value all the way down.
        self.tabs = QTabWidget()
        self.tabs.addTab(roi_page, "ROIs")
        self.tabs.addTab(trigger_page, "Triggers")

        layout = QVBoxLayout()
        layout.addWidget(self.tabs)
        self.setLayout(layout)

        self.reload_triggers()
        self.reload_metrics()
        self.bind_layer(labels_layer)
        # Without an explicit initial sort the tables come up descending
        self.table.sortByColumn(0, Qt.AscendingOrder)
        self.trigger_table.sortByColumn(0, Qt.AscendingOrder)

    # ------------------------------------------------------------------
    # Triggers
    # ------------------------------------------------------------------

    def reload_triggers(self):
        """Fill the trigger page from the recording's trigger train."""
        rows = triggers.trigger_table(self.recording)
        self.trigger_status.setText(triggers.summary(self.recording))

        table = self.trigger_table
        # Sorting reorders rows as they are inserted, which scrambles a
        # half-filled table
        table.setSortingEnabled(False)
        table.setRowCount(len(rows))
        for row, entry in enumerate(rows):
            values = (
                entry["index"],
                round(entry["time_s"], 4),
                None if np.isnan(entry["interval_s"]) else round(entry["interval_s"], 4),
                entry["frame"],
                entry["loop"],
            )
            for column, value in enumerate(values):
                item = QTableWidgetItem()
                if value is None:
                    item.setText("—")
                else:
                    item.setData(Qt.DisplayRole, value)
                item.setFlags(item.flags() & ~Qt.ItemIsEditable)
                if not entry["used"]:
                    # Skipped triggers stay visible: knowing which ones an
                    # average left out is the point of the table
                    item.setForeground(Qt.gray)
                    item.setToolTip("Not included in the average")
                table.setItem(row, column, item)
        table.setSortingEnabled(True)

    def _on_trigger_selected(self):
        """Jump the viewer to the frame a trigger fired on."""
        rows = self.trigger_table.selectionModel().selectedRows()
        if not rows:
            return
        item = self.trigger_table.item(rows[0].row(), 3)
        if item is None:
            return
        frame = item.data(Qt.DisplayRole)
        if not isinstance(frame, int):
            return
        try:
            self.viewer.dims.set_current_step(0, frame)
        except (IndexError, ValueError):
            pass

    # ------------------------------------------------------------------
    # Layer plumbing
    # ------------------------------------------------------------------

    @property
    def labels_layer(self):
        """The ROI layer, or None once removed from the viewer."""
        layer = self._labels_layer
        if layer is None or layer not in self.viewer.layers:
            return None
        return layer

    def bind_layer(self, layer):
        """Attach to an ROI layer and follow its selection."""
        self._labels_layer = layer
        if layer is not None:
            layer.events.selected_label.connect(self._on_label_selected)
        self.refresh()

    rebind = bind_layer

    @property
    def roi_labels(self):
        """Labels of the ROIs the recording currently holds, in row order."""
        if getattr(self.recording, "rois", None) is None:
            return np.empty(0, dtype=int)
        return np.abs(roi_ids_in_order(self.recording.rois)).astype(int)

    # ------------------------------------------------------------------
    # Metric selection
    # ------------------------------------------------------------------

    def reload_metrics(self):
        """Refill the dropdown, keeping the current choice if it survives.

        Which metrics a recording can offer changes as it is worked on:
        trace metrics only exist once traces have been extracted. The list
        is rebuilt when that set changes, rather than being fixed at
        construction.
        """
        specs = available_metrics(self.recording)
        if [s.key for s in specs] == [s.key for s in self._specs]:
            return False

        selected = self.metric_box.currentText()
        self._specs = specs
        self.metric_box.blockSignals(True)
        try:
            self.metric_box.clear()
            for spec in self._specs:
                self.metric_box.addItem(spec.label)
                index = self.metric_box.count() - 1
                self.metric_box.setItemData(index, spec.description, Qt.ToolTipRole)
            index = self.metric_box.findText(selected)
            self.metric_box.setCurrentIndex(index if index >= 0 else 0)
        finally:
            self.metric_box.blockSignals(False)

        spec = self.current_spec
        self.compute_button.setVisible(bool(spec and spec.expensive))
        return True

    @property
    def current_spec(self):
        index = self.metric_box.currentIndex()
        if index < 0 or index >= len(self._specs):
            return None
        return self._specs[index]

    def _on_metric_changed(self):
        spec = self.current_spec
        self.compute_button.setVisible(bool(spec and spec.expensive))
        self.refresh()

    def refresh(self, force=False):
        """Recompute the selected metric and update everything it drives."""
        self.reload_metrics()
        # trigger_mode is editable from the Analysis panel, so which
        # triggers count as used can change under this table
        self.reload_triggers()
        spec = self.current_spec
        self._labels = self.roi_labels

        if spec is None:
            self._values = None
            self.status.setText("No metric available for this recording")
        elif spec.expensive and not force:
            self._values = None
            self.status.setText(f"{spec.label}: press Compute to run")
        else:
            self._values = compute_metric(
                spec, self.recording, n_rois=len(self._labels)
            )
            self.status.setText(self._describe(spec, self._values))

        self._fill_table()
        self._apply_colouring()
        self.draw_histogram()
        self.metric_changed.emit()

    @staticmethod
    def _describe(spec, values):
        if values is None:
            return f"{spec.label} unavailable for this recording"
        finite = np.isfinite(values)
        if not finite.any():
            return f"{spec.label}: no finite values"
        return (
            f"{spec.label}: {finite.sum()} of {values.size} ROIs, "
            f"{np.nanmin(values):.3g} to {np.nanmax(values):.3g}"
        )

    # ------------------------------------------------------------------
    # Views
    # ------------------------------------------------------------------

    def _on_histogram_toggled(self, checked):
        self.canvas.setVisible(checked)
        if checked:
            self.draw_histogram()

    def _roi_color(self):
        """Colour of the selected ROI, for marking it in the distribution."""
        layer = self.labels_layer
        if layer is None or layer.selected_label < 1:
            return "tab:red"
        return tuple(float(c) for c in layer.get_color(layer.selected_label))

    def draw_histogram(self):
        """Redraw the distribution, marking where the selected ROI falls."""
        if not self.histogram_box.isChecked():
            return

        self.ax.clear()
        values = self._values
        if values is None or not np.isfinite(values).any():
            self.ax.set_axis_off()
            self.canvas.draw_idle()
            return

        finite = values[np.isfinite(values)]
        self.ax.set_axis_on()
        self.ax.hist(
            finite, bins=min(20, max(4, finite.size // 2)), color="0.6"
        )

        layer = self.labels_layer
        if layer is not None:
            selected = self.value_for_label(layer.selected_label)
            if selected is not None and np.isfinite(selected):
                self.ax.axvline(selected, color=self._roi_color(), lw=1.5)

        self.ax.set_xlabel(self.current_metric_label)
        self.ax.set_ylabel("ROIs")
        self.canvas.draw_idle()

    def _fill_table(self):
        self._syncing = True
        sorting = self.table.isSortingEnabled()
        self.table.setSortingEnabled(False)
        try:
            self.table.setRowCount(len(self._labels))
            for row, label in enumerate(self._labels):
                roi_item = QTableWidgetItem()
                roi_item.setData(Qt.DisplayRole, int(label))
                roi_item.setFlags(roi_item.flags() & ~Qt.ItemIsEditable)
                self.table.setItem(row, 0, roi_item)

                value_item = QTableWidgetItem()
                if self._values is None:
                    value_item.setData(Qt.DisplayRole, "")
                else:
                    value_item.setData(Qt.DisplayRole, float(self._values[row]))
                value_item.setFlags(value_item.flags() & ~Qt.ItemIsEditable)
                self.table.setItem(row, 1, value_item)
        finally:
            self.table.setSortingEnabled(sorting)
            self._syncing = False
        self._highlight_selected()

    def _apply_colouring(self):
        layer = self.labels_layer
        if layer is None:
            return
        if self.colour_box.isChecked() and self._values is not None:
            apply_metric_colormap(layer, self._labels, self._values)
        else:
            apply_roi_colormap(layer)

    def _on_colour_toggled(self, checked):
        self._apply_colouring()

    @property
    def current_values(self):
        """Values of the selected metric, one per ROI, or None."""
        return self._values

    @property
    def current_metric_label(self):
        spec = self.current_spec
        return spec.label if spec is not None else ""

    def value_for_label(self, label):
        """Metric value for one ROI label, or None if it has no row."""
        if self._values is None:
            return None
        matches = np.flatnonzero(self._labels == int(label))
        if not matches.size:
            return None
        return float(self._values[matches[0]])

    # ------------------------------------------------------------------
    # Selection, both directions
    # ------------------------------------------------------------------

    def _row_for_label(self, label):
        for row in range(self.table.rowCount()):
            item = self.table.item(row, 0)
            if item is not None and int(item.data(Qt.DisplayRole)) == int(label):
                return row
        return -1

    def _highlight_selected(self):
        layer = self.labels_layer
        if layer is None:
            return
        row = self._row_for_label(layer.selected_label)
        if row < 0:
            return
        self._syncing = True
        try:
            self.table.selectRow(row)
        finally:
            self._syncing = False

    def _on_label_selected(self, event=None):
        if self._syncing:
            return
        self._highlight_selected()
        self.draw_histogram()

    def _on_row_selected(self):
        if self._syncing:
            return
        layer = self.labels_layer
        if layer is None:
            return
        rows = self.table.selectionModel().selectedRows()
        if not rows:
            return
        item = self.table.item(rows[0].row(), 0)
        if item is None:
            return
        self._syncing = True
        try:
            layer.selected_label = int(item.data(Qt.DisplayRole))
        finally:
            self._syncing = False
