"""Action buttons that run pygor analysis steps from inside napari.

Long-running calls go through ``thread_worker`` so the viewer stays
responsive; results are written back onto the recording and the relevant
layers are refreshed on the main thread.
"""

import numpy as np
from magicgui import magicgui
from napari.qt.threading import thread_worker
from qtpy.QtCore import QTimer
from qtpy.QtWidgets import QCheckBox, QLabel, QVBoxLayout, QWidget

from pygor.gui.roi_bridge import is_igor_style, labels_to_mask, mask_to_labels

SEGMENTATION_MODES = ("blob", "watershed", "flood_fill", "cellpose", "cellpose+")


class ActionsDock(QWidget):
    """Dock widget exposing segmentation, trace extraction and projections."""

    def __init__(
        self,
        recording,
        viewer,
        labels_layer=None,
        on_traces_changed=None,
        on_layer_restored=None,
        default_layers=None,
    ):
        super().__init__()
        self.recording = recording
        self.viewer = viewer
        self._labels_layer = labels_layer
        self.on_traces_changed = on_traces_changed
        self.on_layer_restored = on_layer_restored
        self.default_layers = list(default_layers or [])
        # Guards against reacting to the additions a restore itself makes
        self._restoring = False
        self._param_dock = None

        self.status = QLabel("Idle")
        self.status.setWordWrap(True)

        self.lock_box = QCheckBox("Lock default layers")
        self.lock_box.setChecked(True)
        self.lock_box.setToolTip(
            "Put back a layer this window created if it is deleted. The "
            "docks depend on those layers; untick to remove them for good."
        )

        self._bind_labels_events(labels_layer)
        self.viewer.layers.events.removing.connect(self._on_layer_removing)
        self.viewer.layers.events.removed.connect(self._on_layer_removed)

        layout = QVBoxLayout()
        layout.addWidget(self.status)
        layout.addWidget(self.lock_box)
        for widget in self._build_widgets():
            layout.addWidget(widget.native)
        layout.addStretch(1)
        self.setLayout(layout)

    def _build_widgets(self):
        return [
            self._segment_widget(),
            self._traces_widget(),
            self._projection_widget(),
            self._push_rois_widget(),
            self._restore_layer_widget(),
            self._params_widget(),
        ]

    @property
    def labels_layer(self):
        """The ROI layer, or None once it has been removed from the viewer."""
        layer = self._labels_layer
        if layer is None or layer not in self.viewer.layers:
            return None
        return layer

    def _on_layer_removing(self, event=None):
        """Save the mask before the ROI layer goes.

        Restoring rebuilds from ``recording.rois``, so edits that were never
        pushed would be lost with the layer. Writing them across first makes
        deletion recoverable rather than destructive.
        """
        layer = self.labels_layer
        if layer is None:
            return
        index = getattr(event, "index", None)
        if index is not None and self.viewer.layers[index] is not layer:
            return
        self.sync_rois_from_layer()

    def _on_layer_removed(self, event=None):
        """Put back a locked default layer, or say how to get it back."""
        if self._restoring:
            return
        missing = self.missing_default_layers()
        if not missing:
            return
        if self.lock_box.isChecked():
            # Restoring inline would fight viewer.close(), which empties the
            # layer list one layer at a time: each re-added layer would be
            # removed again and the close would never finish. Queueing the
            # restore means a teardown simply never gets round to it.
            QTimer.singleShot(0, self._restore_locked_layers)
            return
        self._set_status(
            f"Removed: {', '.join(missing)}. The data is still on the "
            "recording — press Restore default layers to bring them back."
        )

    def _restore_locked_layers(self):
        """Deferred half of the lock. Silent if the viewer has since gone."""
        if not self.lock_box.isChecked():
            return
        try:
            missing = self.missing_default_layers()
            if not missing:
                return
            self.restore_default_layers()
        except RuntimeError:
            # Viewer torn down between the removal and this callback
            return
        self._set_status(f"Restored locked layer(s): {', '.join(missing)}")

    def _bind_labels_events(self, layer):
        """Keep the ROI number layer in step with strokes on this layer."""
        if layer is None:
            return
        layer.events.paint.connect(self.refresh_numbers)

    def refresh_numbers(self, event=None):
        """Redraw the ROI numbers for the current mask."""
        from pygor.gui.roi_numbers import ensure_number_layer

        ensure_number_layer(self.viewer, self.labels_layer)

    def missing_default_layers(self):
        """Names of layers this window created that are no longer present."""
        return [name for name in self.default_layers if name not in self.viewer.layers]

    def restore_default_layers(self):
        """Rebuild every missing default layer and rebind the docks."""
        from pygor.gui.launch import ensure_default_layers

        self._restoring = True
        try:
            names, layer = ensure_default_layers(self.viewer, self.recording)
        finally:
            self._restoring = False

        self.default_layers = names
        if layer is not None and layer is not self._labels_layer:
            self._labels_layer = layer
            self._bind_labels_events(layer)
            if self.on_layer_restored is not None:
                self.on_layer_restored(layer)
        return layer

    def restore_roi_layer(self):
        """Rebuild just the ROI layer. Kept for callers that only need it."""
        self.restore_default_layers()
        if self.labels_layer is None:
            self._set_status("Recording has no ROIs to restore")
        return self.labels_layer

    def _set_status(self, text):
        self.status.setText(text)

    def _run(self, worker, done_message):
        """Start a worker, wiring status updates and error reporting."""
        worker.returned.connect(lambda _=None: self._set_status(done_message))
        worker.errored.connect(lambda exc: self._set_status(f"Failed: {exc}"))
        worker.start()

    def _refresh_labels(self):
        if self.recording.rois is None:
            return
        layer = self.labels_layer
        if layer is None:
            self.restore_roi_layer()
            return
        layer.data = mask_to_labels(self.recording.rois)
        self.refresh_numbers()

    def run_segmentation(self, mode="blob", overwrite=True):
        """Segment ROIs off the GUI thread, refreshing the layer when done."""
        self._set_status(f"Segmenting ({mode})...")

        @thread_worker
        def job():
            return self.recording.segment_rois(mode=mode, overwrite=overwrite)

        worker = job()
        worker.returned.connect(lambda _=None: self._refresh_labels())
        self._run(worker, f"Segmented: {self.recording.num_rois} ROIs")
        return worker

    def _segment_widget(self):
        @magicgui(
            call_button="Segment ROIs",
            mode={"choices": SEGMENTATION_MODES},
            layout="vertical",
        )
        def segment(mode: str = "blob", overwrite: bool = True):
            self.run_segmentation(mode=mode, overwrite=overwrite)

        return segment

    def run_extraction(self, baseline_dur=10.0):
        """Extract traces off the GUI thread, syncing hand-drawn ROIs first."""
        # Extraction reads recording.rois, so ROIs drawn in the layer are
        # invisible to it until they are written back.
        pushed = self.sync_rois_from_layer()
        self._set_status(
            f"Extracting traces ({self.recording.num_rois} ROIs)..."
            if pushed
            else "Extracting traces..."
        )

        @thread_worker
        def job():
            return self.recording.extract_traces_from_rois(baseline_dur=baseline_dur)

        worker = job()
        if self.on_traces_changed is not None:
            worker.returned.connect(lambda _=None: self.on_traces_changed())
        self._run(worker, "Traces extracted")
        return worker

    def _traces_widget(self):
        @magicgui(call_button="Extract traces", layout="vertical")
        def extract(baseline_dur: float = 10.0):
            self.run_extraction(baseline_dur=baseline_dur)

        return extract

    def run_correlation_projection(
        self, include_diagonals=True, timecompress=1, binpix=1
    ):
        """Compute the correlation projection and add it as a layer."""
        self._set_status("Computing correlation projection...")

        @thread_worker
        def job():
            return self.recording.compute_correlation_projection(
                include_diagonals=include_diagonals,
                timecompress=timecompress,
                binpix=binpix,
                overwrite=True,
            )

        worker = job()
        worker.returned.connect(self._add_projection_layer)
        self._run(worker, "Correlation projection done")
        return worker

    def _projection_widget(self):
        @magicgui(call_button="Correlation projection", layout="vertical")
        def project(
            include_diagonals: bool = True,
            timecompress: int = 1,
            binpix: int = 1,
        ):
            self.run_correlation_projection(
                include_diagonals=include_diagonals,
                timecompress=timecompress,
                binpix=binpix,
            )

        return project

    def _add_projection_layer(self, projection):
        if projection is None:
            return
        name = "Correlation projection"
        if name in self.viewer.layers:
            self.viewer.layers[name].data = np.asarray(projection)
        else:
            self.viewer.add_image(np.asarray(projection), name=name, colormap="viridis")

    def sync_rois_from_layer(self):
        """Write layer edits back to the recording, if there are any.

        Returns True when the recording's mask was updated. Analysis reads
        ``recording.rois``, not the layer, so anything drawn by hand has to
        be pushed across before it can be measured.
        """
        if self.labels_layer is None:
            return False
        current = self.recording.rois
        igor = True if current is None else is_igor_style(current)
        mask = labels_to_mask(self.labels_layer.data, igor_style=igor)
        if current is not None and np.array_equal(mask, np.asarray(current)):
            return False
        self.recording.update_rois(mask)
        self.refresh_numbers()
        return True

    def _push_rois_widget(self):
        @magicgui(call_button="Push edited ROIs to recording", layout="vertical")
        def push():
            if self.labels_layer is None:
                self._set_status("No ROI layer to push")
                return
            if self.sync_rois_from_layer():
                self._set_status(
                    f"Pushed {self.recording.num_rois} ROIs to recording"
                )
            else:
                self._set_status("ROIs already match the recording")

        return push

    def _restore_layer_widget(self):
        @magicgui(call_button="Restore default layers", layout="vertical")
        def restore():
            missing = self.missing_default_layers()
            self.restore_default_layers()
            self._set_status(
                f"Restored: {', '.join(missing)}"
                if missing
                else "All default layers already present"
            )

        return restore

    def _params_widget(self):
        @magicgui(call_button="Edit parameters", layout="vertical")
        def edit_params():
            self.open_param_editor()

        return edit_params

    def open_param_editor(self, section=None):
        """Dock the parameter editor, or raise it if already open.

        ``params.edit(blocking=False)`` returns a top-level widget that the
        caller has to keep alive; dropping it let Python collect the window
        the moment it appeared. Docking hands ownership to Qt instead, and
        keeps the editor in the same window as everything else.
        """
        from pygor.core.gui.param_editor import ParamEditorWidget

        name = "Parameters"
        if self._param_dock is not None:
            self._param_dock.show()
            self._param_dock.raise_()
            self._set_status("Parameter editor already open")
            return self._param_dock

        widget = ParamEditorWidget(
            self.recording.params,
            section=section,
            title=name,
        )
        self._param_dock = self.viewer.window.add_dock_widget(
            widget, name=name, area="right"
        )
        self._set_status("Parameter editor opened")
        return self._param_dock
