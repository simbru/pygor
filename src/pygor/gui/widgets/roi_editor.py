"""Pick ROIs on the canvas and delete, merge or reshape them.

napari's Labels tools paint and erase, but have no notion of selecting
ROIs and acting on several at once. This panel adds that: with it open,
click an ROI to select it, shift-click to add or drop more, then delete
or merge the selection. Every edit goes through the layer's undo
history, so Ctrl+Z on the canvas takes it back.

Edits land in the layer only. Analysis reads ``recording.rois``, so they
reach the recording on Push, or when traces are next extracted. Either way
:func:`pygor.core.roi_alignment.apply_mask` carries every per-ROI array
across: an ROI whose pixels are untouched keeps its trace, RF and IPL depth;
a deleted one takes them with it; a new, merged or reshaped one gets NaN
until it is computed.
"""

import numpy as np
from napari.utils.colormaps import DirectLabelColormap
from qtpy.QtWidgets import (
    QGridLayout,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

SELECTION_LAYER_NAME = "ROI selection"

#: A press that moves further than this, in screen pixels, is a pan
_CLICK_TOLERANCE = 3

_OUTLINE = DirectLabelColormap(
    color_dict={None: (0, 0, 0, 0), 0: (0, 0, 0, 0), 1: (1, 1, 1, 1)}
)


class RoiEditorDock(QWidget):
    """Select ROIs by clicking them, then delete, merge or redraw them.

    Clicks only select while the panel is on show and the ROI layer is in
    pan/zoom mode, so painting and panning elsewhere are left alone.
    """

    def __init__(
        self, recording, viewer, labels_layer=None, auto_new_box=None, on_push=None
    ):
        super().__init__()
        self.recording = recording
        self.viewer = viewer
        self.auto_new_box = auto_new_box
        self.on_push = on_push
        self.selection = set()
        self.active = False
        self._labels_layer = None

        self.status = QLabel()
        self.status.setWordWrap(True)
        hint = QLabel(
            "Click an ROI to select it, shift-click to add or remove more. "
            "Edits can be undone with Ctrl+Z on the canvas."
        )
        hint.setWordWrap(True)

        buttons = QGridLayout()
        self.select_button = self._button("Select", self.start_selecting)
        self.clear_button = self._button("Clear selection", self.clear_selection)
        self.delete_button = self._button("Delete (Del)", self.delete_selected)
        self.merge_button = self._button("Merge", self.merge_selected)
        self.paint_button = self._button("Paint into", lambda: self.edit_selected("paint"))
        self.erase_button = self._button("Erase from", lambda: self.edit_selected("erase"))
        self.push_button = self._button("Push to recording", self.push)
        for i, button in enumerate(
            (
                self.select_button,
                self.clear_button,
                self.delete_button,
                self.merge_button,
                self.paint_button,
                self.erase_button,
                self.push_button,
            )
        ):
            buttons.addWidget(button, i // 2, i % 2)

        layout = QVBoxLayout()
        layout.addWidget(hint)
        layout.addLayout(buttons)
        layout.addWidget(self.status)
        layout.addStretch(1)
        self.setLayout(layout)

        self.bind_layer(labels_layer)
        viewer.mouse_drag_callbacks.append(self._on_mouse)
        viewer.bind_key("Delete", self._on_delete_key, overwrite=True)
        self._update()

    def _button(self, text, callback):
        button = QPushButton(text)
        button.clicked.connect(lambda *_: callback())
        return button

    # -- layer plumbing ---------------------------------------------------

    @property
    def labels_layer(self):
        """The ROI layer, or None once it has left the viewer."""
        layer = self._labels_layer
        if layer is None or layer not in self.viewer.layers:
            return None
        return layer

    def bind_layer(self, layer):
        """Follow an ROI layer, dropping the previous one's events."""
        old = self._labels_layer
        if old is not None:
            for event in (old.events.paint, old.events.data, old.events.set_data):
                event.disconnect(self._on_labels_changed)
        self._labels_layer = layer
        if layer is not None:
            # paint covers edits, data covers a mask replaced by push or
            # segmentation, and set_data the redraw undo ends with
            for event in (layer.events.paint, layer.events.data, layer.events.set_data):
                event.connect(self._on_labels_changed)
        self._on_labels_changed()

    def rebind(self, layer):
        """Point the panel at a rebuilt ROI layer."""
        self.bind_layer(layer)

    def disconnect_events(self):
        """Let go of the viewer's mouse and key hooks."""
        if self._on_mouse in self.viewer.mouse_drag_callbacks:
            self.viewer.mouse_drag_callbacks.remove(self._on_mouse)
        self.viewer.bind_key("Delete", None)

    def showEvent(self, event):
        super().showEvent(event)
        self.set_active(True)

    def hideEvent(self, event):
        super().hideEvent(event)
        self.set_active(False)

    def set_active(self, active):
        """Turn click-to-select and the selection outline on or off.

        Hidden, including when another tab is in front, the panel stops
        picking up clicks so it cannot change anything out of sight.
        """
        self.active = bool(active)
        if self.active:
            self.start_selecting()
        else:
            self._draw_outline()

    # -- selecting ----------------------------------------------------------

    def start_selecting(self):
        """Put the ROI layer in pan/zoom, the mode clicks select in."""
        layer = self.labels_layer
        if layer is None:
            self._update("No ROI layer to edit")
            return
        self.viewer.layers.selection.active = layer
        layer.mode = "pan_zoom"
        self._update()

    def _picking(self):
        layer = self.labels_layer
        return self.active and layer is not None and str(layer.mode) == "pan_zoom"

    def _on_mouse(self, viewer, event):
        """Select on a click; a drag is a pan and is left to napari."""
        if event.button != 1 or not self._picking():
            return
        start = np.asarray(event.pos, dtype=float)
        add = "Shift" in event.modifiers
        yield
        while event.type == "mouse_move":
            yield
        if np.abs(np.asarray(event.pos, dtype=float) - start).max() > _CLICK_TOLERANCE:
            return
        self.pick(self.label_at(event.position), add=add)

    def label_at(self, position):
        """The ROI label under a world position, 0 for background."""
        layer = self.labels_layer
        if layer is None:
            return 0
        data = np.asarray(layer.data)
        row, col = np.round(layer.world_to_data(position)[-2:]).astype(int)
        if not (0 <= row < data.shape[-2] and 0 <= col < data.shape[-1]):
            return 0
        return int(data[..., row, col].max())

    def pick(self, label, add=False):
        """Select an ROI; with ``add``, toggle it in the current selection.

        A plain click on background clears the selection, the way clicking
        empty space does in most editors.
        """
        label = int(label)
        if add:
            if label:
                self.selection ^= {label}
        else:
            self.selection = {label} if label else set()
        if len(self.selection) == 1 and self.labels_layer is not None:
            # The plot follows selected_label, so a single pick shows its trace
            self.labels_layer.selected_label = next(iter(self.selection))
        self._update()

    def clear_selection(self):
        self.selection = set()
        self._update()

    # -- editing --------------------------------------------------------------

    def delete_selected(self):
        """Remove every selected ROI from the mask."""
        labels = self._selected_present()
        if not labels:
            self._update("Nothing selected to delete")
            return
        data = np.asarray(self.labels_layer.data)
        self._set_where(np.isin(data, labels), 0)
        self.selection = set()
        self._update(f"Deleted {_describe(labels)}")

    def merge_selected(self):
        """Fold the selected ROIs into the lowest-numbered of them."""
        labels = self._selected_present()
        if len(labels) < 2:
            self._update("Select two or more ROIs to merge")
            return
        target = min(labels)
        data = np.asarray(self.labels_layer.data)
        self._set_where(np.isin(data, labels), target)
        self.selection = {target}
        self.labels_layer.selected_label = target
        self._update(f"Merged {_describe(labels)} into ROI {target}")

    def edit_selected(self, mode):
        """Paint into or erase from the one selected ROI.

        The layer preserves labels, so painting only fills background and
        erasing only removes the selected ROI's pixels: neighbours are
        safe either way. Auto-new would hop to a fresh label after the
        first stroke, so it is switched off for painting.
        """
        layer = self.labels_layer
        labels = self._selected_present()
        if layer is None or len(labels) != 1:
            self._update("Select exactly one ROI to edit")
            return
        label = labels[0]
        note = ""
        if mode == "paint" and self.auto_new_box is not None:
            if self.auto_new_box.isChecked():
                self.auto_new_box.setChecked(False)
                note = " (auto-new ROI turned off)"
        self.viewer.layers.selection.active = layer
        layer.selected_label = label
        layer.mode = mode
        verb = "Painting into" if mode == "paint" else "Erasing from"
        self._update(f"{verb} ROI {label}{note}. Press Select to pick again.")

    def push(self):
        """Write the edited mask to the recording.

        Every per-ROI array follows its ROI's pixels, and ids are compacted
        to 1..N, so the numbers on screen change; the selection is cleared
        rather than left pointing at whichever ROI took a number over.
        """
        if self.on_push is None:
            return None
        from pygor.gui.widgets.segmentation import describe_change

        applied = self.on_push()
        self.selection = set()
        if not applied:
            self._update("ROIs already match the recording")
            return applied
        self._update(describe_change("Pushed", applied, self.recording))
        return applied

    def _set_where(self, where, value):
        """Overwrite pixels as one undoable step."""
        layer = self.labels_layer
        with layer.block_history():
            layer.data_setitem(np.nonzero(where), value)

    def _on_delete_key(self, viewer):
        if self._picking() and self.selection:
            self.delete_selected()

    # -- display --------------------------------------------------------------

    def _selected_present(self):
        """Selected labels that still have pixels, sorted."""
        layer = self.labels_layer
        if layer is None or not self.selection:
            return []
        present = np.unique(np.asarray(layer.data))
        return sorted(label for label in self.selection if label in present)

    def _on_labels_changed(self, event=None):
        """Drop selected ROIs that no longer exist, then redraw."""
        if self.selection:
            self.selection = set(self._selected_present())
        self._draw_outline()

    def _outline_layer(self):
        if SELECTION_LAYER_NAME in self.viewer.layers:
            return self.viewer.layers[SELECTION_LAYER_NAME]
        return None

    def _draw_outline(self):
        """Outline the selected ROIs in white over their colours."""
        layer = self.labels_layer
        outline = self._outline_layer()
        show = self.active and layer is not None and bool(self.selection)
        if not show:
            if outline is not None:
                outline.visible = False
            return
        mask = np.isin(np.asarray(layer.data), sorted(self.selection)).astype(np.uint8)
        if outline is not None:
            outline.data = mask
            outline.visible = True
            return
        # Adding a layer makes it the active one, which would move clicks
        # and Ctrl+Z off the ROI layer
        active = self.viewer.layers.selection.active
        outline = self.viewer.add_labels(
            mask,
            name=SELECTION_LAYER_NAME,
            colormap=_OUTLINE,
            opacity=1.0,
            blending="translucent_no_depth",
        )
        # napari 0.7 takes contour as a property, not a constructor argument
        outline.contour = 1
        if active is not None:
            self.viewer.layers.selection.active = active

    def _update(self, message=None):
        labels = self._selected_present()
        has_layer = self.labels_layer is not None
        self.delete_button.setEnabled(bool(labels))
        self.merge_button.setEnabled(len(labels) >= 2)
        self.paint_button.setEnabled(len(labels) == 1)
        self.erase_button.setEnabled(len(labels) == 1)
        for button in (self.select_button, self.push_button):
            button.setEnabled(has_layer)
        self.clear_button.setEnabled(bool(labels))
        selected = f"Selected: {_describe(labels)}" if labels else "Nothing selected"
        self.status.setText(f"{message}\n{selected}" if message else selected)
        self._draw_outline()


def _describe(labels):
    labels = list(labels)
    if len(labels) == 1:
        return f"ROI {labels[0]}"
    return f"{len(labels)} ROIs ({', '.join(str(label) for label in labels)})"
