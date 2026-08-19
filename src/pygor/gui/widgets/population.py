"""Whole-recording view of the ROIs, for choosing which one to inspect.

The trace dock answers "what does this ROI do"; this answers "which ROI
should I be looking at". One metric is selected at a time and drives three
views together: a histogram of its distribution, a sortable table of
per-ROI values, and optionally the colour of the ROIs themselves.
"""

import numpy as np
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from qtpy.QtCore import Qt
from qtpy.QtWidgets import (
    QAbstractItemView,
    QCheckBox,
    QComboBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from pygor.gui.colors import apply_metric_colormap, apply_roi_colormap
from pygor.gui.metrics import available_metrics, compute_metric
from pygor.gui.roi_bridge import roi_ids_in_order


class PopulationDock(QWidget):
    """Dock showing one metric across every ROI."""

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

        self.figure = Figure(figsize=(4, 2), layout="constrained")
        self.canvas = FigureCanvas(self.figure)
        self.canvas.setMinimumHeight(140)
        self.ax = self.figure.add_subplot(111)

        self.table = QTableWidget(0, 2)
        self.table.setHorizontalHeaderLabels(["ROI", "Value"])
        self.table.verticalHeader().setVisible(False)
        self.table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.table.setSelectionMode(QAbstractItemView.SingleSelection)
        self.table.setSortingEnabled(True)
        self.table.itemSelectionChanged.connect(self._on_row_selected)
        self.table.setMinimumHeight(160)

        controls = QHBoxLayout()
        controls.addWidget(QLabel("Metric:"))
        controls.addWidget(self.metric_box, stretch=1)
        controls.addWidget(self.compute_button)

        layout = QVBoxLayout()
        layout.addLayout(controls)
        layout.addWidget(self.colour_box)
        layout.addWidget(self.status)
        layout.addWidget(self.canvas, stretch=1)
        layout.addWidget(self.table, stretch=2)
        self.setLayout(layout)

        self.reload_metrics()
        self.bind_layer(labels_layer)
        # Without an explicit initial sort the table comes up descending
        self.table.sortByColumn(0, Qt.AscendingOrder)

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
        """Refill the dropdown with the metrics this recording supports."""
        self._specs = available_metrics(self.recording)
        self.metric_box.blockSignals(True)
        self.metric_box.clear()
        for spec in self._specs:
            self.metric_box.addItem(spec.label)
            index = self.metric_box.count() - 1
            self.metric_box.setItemData(index, spec.description, Qt.ToolTipRole)
        self.metric_box.blockSignals(False)

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
        """Recompute the selected metric and redraw everything it drives."""
        spec = self.current_spec
        self._labels = self.roi_labels

        if spec is None:
            self._values = None
            self.status.setText("No metric available for this recording")
            self._draw_histogram()
            self._fill_table()
            return

        if spec.expensive and not force:
            self._values = None
            self.status.setText(f"{spec.label}: press Compute to run")
            self._draw_histogram()
            self._fill_table()
            return

        values = compute_metric(spec, self.recording, n_rois=len(self._labels))
        self._values = values
        if values is None:
            self.status.setText(f"{spec.label} unavailable for this recording")
        else:
            finite = np.isfinite(values)
            self.status.setText(
                f"{spec.label}: {finite.sum()} of {values.size} ROIs, "
                f"{np.nanmin(values):.3g} to {np.nanmax(values):.3g}"
                if finite.any()
                else f"{spec.label}: no finite values"
            )
        self._draw_histogram()
        self._fill_table()
        self._apply_colouring()

    # ------------------------------------------------------------------
    # Views
    # ------------------------------------------------------------------

    def _draw_histogram(self):
        self.ax.clear()
        if self._values is None or not np.isfinite(self._values).any():
            self.ax.set_axis_off()
            self.canvas.draw_idle()
            return
        finite = self._values[np.isfinite(self._values)]
        self.ax.set_axis_on()
        self.ax.hist(finite, bins=min(20, max(4, finite.size // 2)), color="tab:blue")
        self.ax.set_xlabel(self.current_spec.label)
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
