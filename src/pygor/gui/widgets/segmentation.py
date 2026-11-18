"""Segmentation, with the parameters of the chosen mode and nothing else.

Each mode reads a different set of parameters from the config, so showing
all of them at once is noise: only the selected mode's section is shown.

The widgets are a view of ``recording.params``, not a copy of it. Editing
one writes through the parameter bus, so the parameter table shows the
change, and a change made there updates these widgets. ``segment_rois``
is then called with no parameter arguments at all: it reads the config
itself, which is now what the panel has been editing.
"""

import numpy as np
from napari.qt.threading import thread_worker
from qtpy.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QSpinBox,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)

from pygor.gui.colors import apply_roi_colormap
from pygor.gui.roi_bridge import labels_to_mask, mask_to_labels, sync_rois_from_layer

# Modes segment_rois accepts. cellpose variants need a model installed, so
# they are offered but may fail at run time rather than being hidden.
MODES = ("blob", "watershed", "flood_fill", "cellpose", "cellpose+")

# Sections of the config that are not a mode's parameters
_NON_MODE_SECTIONS = ("general",)

_SPIN_RANGE = (-1_000_000, 1_000_000)


class ParamForm(QWidget):
    """A form over one config section, typed by the value it holds.

    Widgets are bound to a dotted path each: editing one writes straight
    through the bus, and a change arriving from elsewhere updates the
    widget without echoing back.
    """

    def __init__(self, bus, prefix, defaults=None):
        super().__init__()
        self.bus = bus
        self.prefix = prefix
        self._defaults = dict(
            defaults if defaults is not None else bus.section(prefix)
        )
        self._widgets = {}
        self._applying = False

        layout = QFormLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        for name, value in self._defaults.items():
            widget = self._widget_for(self.bus.get(self._path(name), value))
            self._widgets[name] = widget
            self._connect(name, widget)
            layout.addRow(name.replace("_", " "), widget)
        self.setLayout(layout)

        self.bus.changed.connect(self._on_bus_changed)

    def _path(self, name):
        return f"{self.prefix}.{name}"

    def _connect(self, name, widget):
        """Write edits through the bus as they happen."""
        if isinstance(widget, QCheckBox):
            signal = widget.toggled
        elif isinstance(widget, (QSpinBox, QDoubleSpinBox)):
            signal = widget.valueChanged
        else:
            signal = widget.editingFinished
        signal.connect(lambda *_: self._push(name))

    def _push(self, name):
        if self._applying:
            return
        self.bus.set(self._path(name), self.value(name))

    def _on_bus_changed(self, path):
        if not path.startswith(f"{self.prefix}."):
            return
        name = path[len(self.prefix) + 1 :]
        if name not in self._widgets:
            return
        self._apply(name, self.bus.get(path))

    def _apply(self, name, value):
        """Show a value without writing it back out again."""
        widget = self._widgets[name]
        self._applying = True
        try:
            if isinstance(widget, QCheckBox):
                widget.setChecked(bool(value))
            elif isinstance(widget, (QSpinBox, QDoubleSpinBox)):
                widget.setValue(value)
            else:
                widget.setText(str(value))
        finally:
            self._applying = False

    @staticmethod
    def _widget_for(value):
        # bool first: it is a subclass of int
        if isinstance(value, bool):
            widget = QCheckBox()
            widget.setChecked(value)
        elif isinstance(value, int):
            widget = QSpinBox()
            widget.setRange(*_SPIN_RANGE)
            widget.setValue(value)
        elif isinstance(value, float):
            widget = QDoubleSpinBox()
            widget.setRange(*_SPIN_RANGE)
            widget.setDecimals(3)
            widget.setSingleStep(0.05)
            widget.setValue(value)
        else:
            widget = QLineEdit(str(value))
        return widget

    def value(self, name):
        widget = self._widgets[name]
        if isinstance(widget, QCheckBox):
            return widget.isChecked()
        if isinstance(widget, (QSpinBox, QDoubleSpinBox)):
            return widget.value()
        return widget.text()

    def values(self):
        return {name: self.value(name) for name in self._widgets}

    def changed(self):
        """Parameters that differ from the values the config shipped with."""
        return {
            name: value
            for name, value in self.values().items()
            if value != self._defaults[name]
        }

    def reset(self):
        """Put the section back to the values the config shipped with."""
        for name, value in self._defaults.items():
            self.bus.set(self._path(name), value)
            self._apply(name, value)


class SegmentationDock(QWidget):
    """Dock for finding ROIs and pushing hand-drawn ones to the recording."""

    def __init__(
        self,
        recording,
        viewer,
        bus,
        labels_layer=None,
        on_rois_changed=None,
        on_layer_created=None,
    ):
        super().__init__()
        self.recording = recording
        self.viewer = viewer
        self.bus = bus
        self._labels_layer = labels_layer
        self.on_rois_changed = on_rois_changed
        self.on_layer_created = on_layer_created

        sections = self._config_sections()

        self.mode_box = QComboBox()
        self.mode_box.addItems([m for m in MODES if m.rstrip("+") in sections] or list(MODES))
        self.mode_box.currentIndexChanged.connect(self._on_mode_changed)

        self.overwrite_box = QCheckBox("Overwrite existing ROIs")
        self.overwrite_box.setChecked(True)

        self.forms = QStackedWidget()
        self._forms_by_mode = {}
        for index in range(self.mode_box.count()):
            mode = self.mode_box.itemText(index)
            section = mode.rstrip("+")
            form = ParamForm(
                self.bus,
                f"segmentation.{section}",
                defaults=sections.get(section, {}),
            )
            self._forms_by_mode[mode] = form
            self.forms.addWidget(form)

        self.segment_button = QPushButton("Segment ROIs")
        self.segment_button.clicked.connect(lambda: self.run_segmentation())

        self.reset_button = QPushButton("Reset to config")
        self.reset_button.setToolTip("Restore this mode's configured values")
        self.reset_button.clicked.connect(self._reset_current_form)

        self.push_button = QPushButton("Push edited ROIs to recording")
        self.push_button.setToolTip(
            "ROIs drawn by hand live in the layer until they are written "
            "back; analysis reads the recording."
        )
        self.push_button.clicked.connect(self.push_rois)

        self.status = QLabel()
        self.status.setWordWrap(True)

        mode_row = QHBoxLayout()
        mode_row.addWidget(QLabel("Mode:"))
        mode_row.addWidget(self.mode_box, stretch=1)

        buttons = QHBoxLayout()
        buttons.addWidget(self.segment_button, stretch=1)
        buttons.addWidget(self.reset_button)

        layout = QVBoxLayout()
        layout.addWidget(self.status)
        layout.addLayout(mode_row)
        layout.addWidget(self.overwrite_box)
        layout.addWidget(self.forms, stretch=1)
        layout.addLayout(buttons)
        layout.addWidget(self.push_button)
        layout.addStretch(1)
        self.setLayout(layout)

        self._set_default_mode(sections)
        self.refresh_status()

    # ------------------------------------------------------------------
    # Config
    # ------------------------------------------------------------------

    def _config_sections(self):
        try:
            segmentation = self.recording.params._defaults["segmentation"]
        except (AttributeError, KeyError, TypeError):
            return {}
        return {
            name: values
            for name, values in segmentation.items()
            if isinstance(values, dict) and name not in _NON_MODE_SECTIONS
        }

    def _set_default_mode(self, sections):
        try:
            configured = self.recording.params._defaults["segmentation"]["general"]["mode"]
        except (AttributeError, KeyError, TypeError):
            return
        index = self.mode_box.findText(configured)
        if index >= 0:
            self.mode_box.setCurrentIndex(index)
            self.forms.setCurrentIndex(index)

    def _on_mode_changed(self, index):
        self.forms.setCurrentIndex(index)

    def _reset_current_form(self):
        self.current_form.reset()
        self.refresh_status(f"{self.mode} reset to configured values")

    @property
    def mode(self):
        return self.mode_box.currentText()

    @property
    def current_form(self):
        return self._forms_by_mode[self.mode]

    # ------------------------------------------------------------------
    # Layer plumbing
    # ------------------------------------------------------------------

    @property
    def labels_layer(self):
        layer = self._labels_layer
        if layer is None or layer not in self.viewer.layers:
            return None
        return layer

    def rebind(self, layer):
        self._labels_layer = layer

    def refresh_status(self, message=None):
        count = int(getattr(self.recording, "num_rois", 0) or 0)
        state = f"{count} ROIs on the recording"
        self.status.setText(f"{state}\n{message}" if message else state)

    # ------------------------------------------------------------------
    # Actions
    # ------------------------------------------------------------------

    def run_segmentation(self, mode=None, overwrite=None, **overrides):
        """Segment with the current mode.

        No parameter arguments are passed: the widgets write through to
        the config, and ``segment_rois`` reads that config itself, so
        passing them again would only be a chance for the two to disagree.
        """
        mode = mode or self.mode
        if overwrite is None:
            overwrite = self.overwrite_box.isChecked()
        params = dict(overrides)

        self.refresh_status(f"Segmenting ({mode})...")

        recording = self.recording

        @thread_worker
        def job():
            # segment_rois swaps the mask under every per-ROI array. Take its
            # result and put the old mask back, so apply_mask can carry the
            # arrays across with ROIs matched by their pixels.
            old_rois = getattr(recording, "rois", None)
            old_count = getattr(recording, "num_rois", 0)
            recording.segment_rois(mode=mode, overwrite=overwrite, **params)
            new_rois = getattr(recording, "rois", None)
            origin = getattr(recording, "roi_origin", None)
            recording.rois = old_rois
            recording.num_rois = old_count
            return new_rois, origin

        worker = job()
        worker.returned.connect(self._on_segmented)
        worker.errored.connect(
            lambda exc: self.refresh_status(f"Segmentation failed: {exc}")
        )
        worker.start()
        return worker

    def _on_segmented(self, result=None):
        new_rois, origin = result if result is not None else (None, None)
        applied = None
        if new_rois is not None:
            from pygor.core.roi_alignment import apply_mask

            applied = apply_mask(
                self.recording, mask_to_igor(new_rois), refresh_traces=False
            )
            if isinstance(origin, dict):
                self.recording.roi_origin = origin
        self._roi_data_changed(
            describe_change(f"Segmented with {self.mode}", applied, self.recording)
        )

    def push_rois(self):
        """Write ROIs drawn in the layer back to the recording.

        Per-ROI data follows each ROI's pixels; ROIs that are new or were
        reshaped get NaN rows until traces and RFs are computed for them.
        """
        if self.labels_layer is None:
            self.refresh_status("No ROI layer to push")
            return False
        applied = sync_rois_from_layer(
            self.recording, self.labels_layer, refresh_traces=False
        )
        if not applied:
            self.refresh_status("ROIs already match the recording")
            return False
        self._roi_data_changed(describe_change("Pushed", applied, self.recording))
        return applied

    def _roi_data_changed(self, message):
        """Show the new mask everywhere, then report what happened."""
        if getattr(self.recording, "rois", None) is None:
            self.refresh_status(message)
            return

        layer = self.labels_layer
        if layer is None:
            # A recording opened without ROIs has no layer to update, so
            # the first segmentation has to create one.
            from pygor.gui.launch import ensure_default_layers

            _names, layer = ensure_default_layers(self.viewer, self.recording)
            self._labels_layer = layer
            if layer is not None and self.on_layer_created is not None:
                self.on_layer_created(layer)
        else:
            layer.data = mask_to_labels(self.recording.rois)
            apply_roi_colormap(layer)
        if self.on_rois_changed is not None:
            self.on_rois_changed()
        self.refresh_status(message)


def mask_to_igor(mask):
    """A mask in the IGOR convention, whichever one a backend returned."""
    return labels_to_mask(mask_to_labels(mask), igor_style=True)


def describe_change(verb, applied, recording):
    """What an ROI change kept, dropped and still needs computing."""
    if applied is None:
        return verb
    from pygor.core.roi_alignment import rois_missing_strfs

    text = f"{verb}: {applied.kept} kept, {applied.deleted} removed, {applied.added} new"
    if applied.added:
        text += ". Extract traces to fill the new ones"
        if rois_missing_strfs(recording):
            text += ", then Recompute missing RFs"
    return text
