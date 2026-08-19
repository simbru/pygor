"""Action buttons that run pygor analysis steps from inside napari.

Long-running calls go through ``thread_worker`` so the viewer stays
responsive; results are written back onto the recording and the relevant
layers are refreshed on the main thread.
"""

import numpy as np
from magicgui import magicgui
from napari.qt.threading import thread_worker
from qtpy.QtWidgets import QLabel, QVBoxLayout, QWidget

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
    ):
        super().__init__()
        self.recording = recording
        self.viewer = viewer
        self._labels_layer = labels_layer
        self.on_traces_changed = on_traces_changed
        self.on_layer_restored = on_layer_restored
        self.viewer.layers.events.removing.connect(self._on_layer_removing)
        self.viewer.layers.events.removed.connect(self._on_layer_removed)

        self.status = QLabel("Idle")
        self.status.setWordWrap(True)

        layout = QVBoxLayout()
        layout.addWidget(self.status)
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
        """Warn when the ROI layer is deleted, rather than failing quietly."""
        if self._labels_layer is None or self.labels_layer is not None:
            return
        self._set_status(
            "ROI layer removed. The mask is still on the recording — "
            "press Restore ROI layer to bring it back."
        )

    def restore_roi_layer(self):
        """Rebuild the ROI layer from the recording and rebind the docks."""
        from pygor.gui.launch import ensure_roi_layer

        layer = ensure_roi_layer(self.viewer, self.recording)
        if layer is None:
            self._set_status("Recording has no ROIs to restore")
            return None
        self._labels_layer = layer
        if self.on_layer_restored is not None:
            self.on_layer_restored(layer)
        return layer

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

    def _segment_widget(self):
        @magicgui(
            call_button="Segment ROIs",
            mode={"choices": SEGMENTATION_MODES},
            layout="vertical",
        )
        def segment(mode: str = "blob", overwrite: bool = True):
            self._set_status(f"Segmenting ({mode})...")

            @thread_worker
            def job():
                return self.recording.segment_rois(mode=mode, overwrite=overwrite)

            worker = job()
            worker.returned.connect(lambda _=None: self._refresh_labels())
            self._run(worker, f"Segmented: {self.recording.num_rois} ROIs")

        return segment

    def _traces_widget(self):
        @magicgui(call_button="Extract traces", layout="vertical")
        def extract(baseline_dur: float = 10.0):
            # Extraction reads recording.rois, so ROIs drawn in the layer
            # are invisible to it until they are written back.
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

        return extract

    def _projection_widget(self):
        @magicgui(call_button="Correlation projection", layout="vertical")
        def project(
            include_diagonals: bool = True,
            timecompress: int = 1,
            binpix: int = 1,
        ):
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
        @magicgui(call_button="Restore ROI layer", layout="vertical")
        def restore():
            existed = self.labels_layer is not None
            layer = self.restore_roi_layer()
            if layer is None:
                return
            self._set_status(
                "ROI layer already present"
                if existed
                else f"Restored ROI layer with {self.recording.num_rois} ROIs"
            )

        return restore

    def _params_widget(self):
        @magicgui(call_button="Edit parameters", layout="vertical")
        def edit_params():
            self.recording.params.edit(blocking=False)
            self._set_status("Parameter editor opened")

        return edit_params
