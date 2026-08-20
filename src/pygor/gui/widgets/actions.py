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

from pygor.gui.triggers import loop_count, summary, triggers_per_loop

from pygor.gui.roi_bridge import mask_to_labels, sync_rois_from_layer

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
        on_depths_updated=None,
        bus=None,
    ):
        super().__init__()
        self.recording = recording
        self.viewer = viewer
        self._labels_layer = labels_layer
        self.on_traces_changed = on_traces_changed
        self.on_layer_restored = on_layer_restored
        self.on_depths_updated = on_depths_updated
        self.bus = bus
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
            self._traces_widget(),
            self._averaging_widget(),
            self._projection_widget(),
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
        """Rescale the palette to the ROI count, then redraw the numbers."""
        from pygor.gui.colors import apply_roi_colormap
        from pygor.gui.roi_numbers import ensure_number_layer

        layer = self.labels_layer
        apply_roi_colormap(layer)
        ensure_number_layer(self.viewer, layer)
        if self.on_traces_changed is not None:
            # The selected ROI's colour may have moved with the palette
            self.on_traces_changed()

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
        """Start a worker, wiring status updates and error reporting.

        ``done_message`` may be a callable, for messages that report on
        what the job produced: an f-string built at call time reads the
        state from before the job ran.
        """

        def _done(_=None):
            self._set_status(
                done_message() if callable(done_message) else done_message
            )

        worker.returned.connect(_done)
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

    def run_averaging(self):
        """Split traces into stimulus repetitions and average them."""
        pushed = self.sync_rois_from_layer()
        self._set_status(
            f"Averaging ({self.recording.num_rois} ROIs)..."
            if pushed
            else "Averaging trials..."
        )

        @thread_worker
        def job():
            return self.recording.compute_snippets_and_averages()

        worker = job()
        worker.returned.connect(self._on_averages_ready)
        worker.errored.connect(
            lambda exc: self._set_status(f"Averaging failed: {exc}")
        )
        worker.start()
        return worker

    def _on_averages_ready(self, result=None):
        snippets, averages = result if result is not None else (None, None)
        if averages is None:
            self._set_status("Averaging produced nothing")
            return
        n_rois, length = np.shape(averages)
        n_trials = np.shape(snippets)[1] if np.ndim(snippets) == 3 else 0
        self._set_status(
            f"Averaged {n_trials} trials for {n_rois} ROIs, {length} samples each"
        )
        if self.on_traces_changed is not None:
            self.on_traces_changed()

    def _averaging_widget(self):
        default = triggers_per_loop(self.recording)

        @magicgui(
            call_button="Compute averages",
            layout="vertical",
            triggers_per_loop={
                "min": 1,
                "max": 1000,
                "label": "Triggers per loop",
            },
        )
        def average(triggers_per_loop: int = default):
            # compute_snippets_and_averages reads trigger_mode off the
            # recording rather than taking it as an argument
            self.recording.trigger_mode = int(triggers_per_loop)
            self.run_averaging()

        def describe(_=None):
            """Say up front how many repetitions this setting would average.

            Getting it wrong is silent otherwise: a stimulus whose loop is
            four triggers long averages four unrelated epochs together and
            still returns a perfectly plausible-looking array.
            """
            mode = average.triggers_per_loop.value
            text = summary(self.recording, mode)
            average.triggers_per_loop.tooltip = text
            average.call_button.text = (
                f"Compute averages ({loop_count(self.recording, mode)} loops)"
            )

        average.triggers_per_loop.changed.connect(describe)
        describe()
        self._averaging_form = average

        return average

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
        """Write layer edits back to the recording, if there are any."""
        if sync_rois_from_layer(self.recording, self.labels_layer):
            self.refresh_numbers()
            return True
        return False

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

    # ------------------------------------------------------------------
    # IPL depth
    # ------------------------------------------------------------------

    def draw_ipl_boundaries(self):
        """Add the boundary layers and put the outer one in drawing mode."""
        from pygor.gui.ipl import ensure_boundary_layers

        ensure_boundary_layers(self.viewer)
        self._set_status(
            "Draw the 0% boundary, then the 100% boundary, then compute depths"
        )

    def compute_ipl_depths(self):
        """Measure ROI depths between the two drawn boundaries."""
        from pygor.gui.ipl import compute_depths

        depths, message = compute_depths(self.recording, self.viewer)
        self._set_status(message)
        if depths is not None:
            self._store_depths(depths)
        return depths

    def estimate_ipl_depths(self):
        """Estimate depths from ROI positions, without drawing boundaries."""
        from pygor.gui.ipl import estimate_depths

        depths, message = estimate_depths(self.recording)
        self._set_status(message)
        if depths is not None:
            self._store_depths(depths)
        return depths

    def _store_depths(self, depths):
        self.recording.update_ipl_depths(depths)
        if self.on_depths_updated is not None:
            self.on_depths_updated()

    def _params_widget(self):
        @magicgui(call_button="Edit parameters", layout="vertical")
        def edit_params():
            self.open_param_editor()

        return edit_params

    def open_param_editor(self, section=None, floating=True):
        """Dock the parameter editor, or raise it if already open.

        The table lists every parameter there is, which is more than the
        per-step tabs ask anyone to read; it opens floating, on request,
        as the way to see or set something the tabs do not cover.

        ``params.edit(blocking=False)`` returns a top-level widget that the
        caller has to keep alive; dropping it let Python collect the window
        the moment it appeared. Docking hands ownership to Qt instead.
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
            bus=self.bus,
        )
        self._param_dock = self.viewer.window.add_dock_widget(
            widget, name=name, area="right"
        )
        self._param_dock.setFloating(floating)
        self._set_status("Parameter editor opened")
        return self._param_dock
