"""Draw ROIs by hand: a disk, or a lasso, on the projection.

Reached from the inspector with ``d``, and like re-segmentation it is a screen
of its own, so the inspector stays read-only. Edits are made to a copy of the
mask; the recording is not touched until save is pressed and confirmed, and
escape with unsaved edits asks first.

The terminal reports the pointer one character cell at a time, which is about
two pixels on a 64-row scan. So every tool has a keyboard form that is exact to
the pixel: the arrows move the cursor, the disk is centred on it, and ``space``
drops a lasso vertex on it. The mouse is for getting close quickly.
"""

from __future__ import annotations

from textual import work
from textual.binding import Binding
from textual.containers import Horizontal
from textual.markup import escape
from textual.screen import Screen
from textual.widgets import Footer, Header, Static

from pygor.tui import roi_edit
from pygor.tui.browse_screen import ImagePointer, post_if_current
from pygor.tui.reprocess_screen import ConfirmSave

HELP = """[b]draw ROIs[/b]

[b]c[/b]  disk tool
[b]l[/b]  lasso tool
[b]+ -[/b]  disk radius

click / arrows  cursor
[dim]shift+arrows move ten[/dim]
drag  lasso stroke
click / space  lasso vertex
backspace  drop vertex

[b]enter[/b]  add the ROI
[b]x[/b]  delete ROI under cursor
[b]u[/b]  undo
[b]p[/b]  mean / correlation

[b]S[/b]  save
[b]esc[/b]  cancel lasso, or leave

[dim]Existing ROIs keep their
pixels: the shape is added
where it is green, not
where it is red. Delete a
cell first to redraw it.[/dim]"""


class DrawScreen(ImagePointer, Screen):
    """Add and delete ROIs on one recording's mask, then save or walk away."""

    PREVIEW_ID = "#draw-preview"
    REDRAW_INTERVAL = 0.08
    BACKGROUNDS = ("mean", "correlation")

    BINDINGS = [
        Binding("escape", "back", "back"),
        Binding("c", "tool('disk')", "disk"),
        Binding("l", "tool('lasso')", "lasso"),
        Binding("plus,equals_sign", "radius(1)", "radius +"),
        Binding("minus", "radius(-1)", "radius -"),
        Binding("enter", "commit", "add ROI"),
        Binding("space", "vertex", "vertex", show=False),
        Binding("backspace", "unvertex", "drop vertex", show=False),
        Binding("x", "delete", "delete ROI"),
        Binding("u", "undo", "undo"),
        Binding("p", "background", "background"),
        Binding("S", "save", "save"),
        # Arrows only: the inspector's hjkl would put the lasso key under a
        # cursor direction.
        Binding("left", "move(0, -1)", "cursor", show=False),
        Binding("right", "move(0, 1)", "cursor", show=False),
        Binding("up", "move(1, 0)", "cursor", show=False),
        Binding("down", "move(-1, 0)", "cursor", show=False),
        Binding("shift+left", "move(0, -10)", "cursor", show=False),
        Binding("shift+right", "move(0, 10)", "cursor", show=False),
        Binding("shift+up", "move(10, 0)", "cursor", show=False),
        Binding("shift+down", "move(-10, 0)", "cursor", show=False),
    ]

    def __init__(self, recording, *, caps, save, describe_save, cursor=None):
        """``save(mask)`` commits a mask to the recording and to disk;
        ``describe_save(mask)`` is the sentence the confirmation shows."""
        super().__init__()
        import numpy as np

        self.recording = recording
        self.caps = caps
        self.save_fn = save
        self.describe_save = describe_save
        images = recording.images
        shape = images.shape[1:]
        rois = getattr(recording, "rois", None)
        # An unsegmented recording starts from an empty mask in the IGOR
        # convention, background 1.
        self.rois = (np.array(rois, copy=True) if rois is not None
                     else np.ones(shape, dtype=np.int16))
        self.history = []
        self.tool = "disk"
        self.radius = 2
        self.cursor = cursor or (shape[0] // 2, shape[1] // 2)
        self.path = []
        self._stroke = None
        self.background_index = 0
        self._images = {}
        self._redraw_timer = None
        self._panel_region = None
        self.saved = False
        # True while a save runs in its worker. Edits are refused meanwhile:
        # the save ends by reloading the mask it wrote, which would silently
        # discard anything drawn in the few seconds trace extraction takes.
        self.saving = False

    # -- layout -------------------------------------------------------------

    def compose(self):
        from pygor.tui.app import PanelView

        yield Header()
        yield Horizontal(Static(HELP, id="draw-help"),
                         PanelView(self.caps, id="draw-preview"))
        yield Static("", id="status")
        yield Footer()

    def on_mount(self):
        self.sub_title = f"{getattr(self.recording, 'name', '')}  ·  draw"
        self.set_status()
        self.redraw()

    def image_shape(self):
        return self.rois.shape

    @property
    def dirty(self):
        return bool(self.history)

    @property
    def background(self):
        return self.BACKGROUNDS[self.background_index % len(self.BACKGROUNDS)]

    def pending(self):
        """The pixels the current tool would add, or None."""
        if self.tool == "disk":
            return roi_edit.disk(self.rois.shape, self.cursor, self.radius)
        if len(self.path) >= 3:
            return roi_edit.polygon(self.rois.shape, self.path)
        return None

    def set_status(self, text=None):
        parts = [f"{self.tool}" + (f" r={self.radius}" if self.tool == "disk" else
                                   f" {len(self.path)} vertices"),
                 f"cursor ({self.cursor[0]}, {self.cursor[1]})",
                 f"{roi_edit.count(self.rois)} ROIs"]
        region = self.pending()
        if region is not None:
            free = int((region & (self.rois >= 0)).sum())
            blocked = int((region & (self.rois < 0)).sum())
            parts.append(f"adds {free} px" + (f", {blocked} kept by others" if blocked else ""))
        if self.dirty:
            parts.append(f"{len(self.history)} unsaved")
        if text:
            parts.append(text)
        self.query_one("#status", Static).update("  ·  ".join(parts))

    # -- drawing ------------------------------------------------------------

    def schedule_redraw(self):
        if self._redraw_timer is None:
            self._redraw_timer = self.set_timer(self.REDRAW_INTERVAL, self._redraw_due)

    def _redraw_due(self):
        self._redraw_timer = None
        self.redraw()

    def redraw(self):
        self.render_edit(self.background, self.rois, self.pending(), list(self.path),
                         self.cursor)

    def background_image(self, which):
        """The picture to draw on, and its display range; cached per kind."""
        import numpy as np

        if which not in self._images:
            if which == "correlation":
                image = getattr(self.recording, "correlation_projection", None)
                if image is None:
                    image = self.recording.compute_correlation_projection()
            else:
                image = np.mean(self.recording.images, axis=0)
            image = np.asarray(image, dtype=np.float32)
            low, high = np.nanpercentile(image, [1, 99.5])
            self._images[which] = (image, (float(low), float(max(high, low + 1e-6))))
        return self._images[which]

    @work(thread=True, exclusive=True, group="draw")
    def render_edit(self, which, rois, pending, path, cursor):
        from pygor.tui.app import PanelView
        from pygor.tui.imaging import edit_preview

        view = self.query_one("#draw-preview", PanelView)
        width, height = view.size_px()
        try:
            image, limits = self.background_image(which)
            panel = edit_preview(image, limits, rois, width, height, pending=pending,
                                 path=path, marker=cursor,
                                 name=getattr(self.recording, "name", ""))
        except Exception as error:
            import traceback

            self.log(traceback.format_exc())
            post_if_current(self.app, view.show_message,
                            escape(f"preview failed: {type(error).__name__}: {error}"))
            return
        post_if_current(self.app, view.show, panel)

    def changed(self, text=None, *, now=False):
        self.set_status(text)
        if now:
            self.redraw()
        else:
            self.schedule_redraw()

    # -- cursor and tools ---------------------------------------------------

    def set_cursor(self, position):
        if position == self.cursor:
            return
        self.cursor = position
        self.changed()

    def action_move(self, d_row, d_col):
        rows, cols = self.rois.shape
        row, col = self.cursor
        self.set_cursor((min(max(row + d_row, 0), rows - 1),
                         min(max(col + d_col, 0), cols - 1)))

    def action_tool(self, tool):
        self.tool = tool
        self.changed(now=True)

    def action_radius(self, delta):
        self.radius = min(max(self.radius + delta, 0), max(self.rois.shape) // 2)
        self.changed(now=True)

    def action_background(self):
        self.background_index += 1
        self.changed(f"background: {self.background}", now=True)

    def action_vertex(self):
        if self.tool != "lasso":
            return
        if not self.path or self.path[-1] != self.cursor:
            self.path.append(self.cursor)
        self.changed(now=True)

    def action_unvertex(self):
        if self.path:
            self.path.pop()
            self.changed(now=True)

    # -- the mouse ----------------------------------------------------------

    def on_mouse_down(self, event):
        found = self.pixel_under(event.screen_x, event.screen_y)
        if found is None:
            return
        event.stop()
        self.cursor = found
        if self.tool == "lasso":
            self._stroke = [found]
        self.changed(now=True)

    def on_mouse_move(self, event):
        if self._stroke is None:
            return
        # Textual runs the terminal in any-motion mode, where a move reports
        # the button held and 0 when none is. A move with nothing held means
        # the release happened where this screen could not see it -- outside
        # the window, or on another widget that kept it -- so the stroke ends
        # here rather than following a pointer that is no longer dragging.
        if not event.button:
            self.end_stroke()
            return
        found = self.pixel_under(event.screen_x, event.screen_y)
        if found is None or found == self._stroke[-1]:
            return
        self._stroke.append(found)
        # The stroke is the path while it is being drawn, so the reader sees
        # the line they are making.
        self.path = list(self._stroke)
        self.cursor = found
        self.changed()

    def on_mouse_up(self, event):
        self.end_stroke()

    def end_stroke(self):
        stroke, self._stroke = self._stroke, None
        if stroke is None:
            return
        # A drag is a freehand outline and replaces the path; a click without
        # moving adds one vertex to it, so a polygon can be clicked out too.
        if len(stroke) > 1:
            self.path = stroke
        elif not self.path or self.path[-1] != stroke[0]:
            self.path = [*self.path, stroke[0]]
        self.changed(now=True)

    # -- edits --------------------------------------------------------------

    def busy(self):
        if self.saving:
            self.app.notify("still saving", severity="warning")
        return self.saving

    def action_commit(self):
        if self.busy():
            return
        region = self.pending()
        if region is None:
            self.app.notify("a lasso needs at least three vertices", severity="warning")
            return
        added = roi_edit.add(self.rois, region)
        if added.roi_id is None:
            self.app.notify(f"all {added.blocked} px already belong to other ROIs",
                            severity="warning")
            return
        self.history.append(self.rois)
        self.rois = added.rois
        self.path = []
        note = f"added ROI {added.roi_id}: {added.taken} px"
        if added.blocked:
            note += f" ({added.blocked} kept by existing ROIs)"
        self.changed(note, now=True)

    def action_delete(self):
        if self.busy():
            return
        roi_id = int(self.rois[self.cursor])
        if roi_id >= 0:
            self.app.notify("the cursor is not on an ROI", severity="warning")
            return
        self.history.append(self.rois)
        self.rois = roi_edit.remove(self.rois, roi_id)
        self.changed(f"deleted ROI {roi_id}", now=True)

    def action_undo(self):
        if self.busy():
            return
        if not self.history:
            self.app.notify("nothing to undo", severity="warning")
            return
        self.rois = self.history.pop()
        self.changed("undone", now=True)

    # -- leaving ------------------------------------------------------------

    def action_save(self):
        if self.busy():
            return
        if not self.dirty:
            self.app.notify("no edits to save", severity="warning")
            return
        self.app.push_screen(ConfirmSave(self.describe_save(self.rois)), self._save_confirmed)

    def _save_confirmed(self, yes):
        if yes:
            self.saving = True
            self.set_status("saving: re-extracting traces…")
            self.save_worker(self.rois)

    @work(thread=True, exclusive=True, group="save")
    def save_worker(self, rois):
        try:
            self.save_fn(rois)
        except Exception as error:
            import traceback

            self.log(traceback.format_exc())
            self.app.call_from_thread(self._save_failed, f"{type(error).__name__}: {error}")
            return
        self.app.call_from_thread(self._saved)

    def _save_failed(self, message):
        self.saving = False
        self.app.notify(f"save failed: {message}", severity="error", timeout=12,
                        markup=False)
        self.set_status("save failed; edits kept")

    def _saved(self):
        # Ids were compacted on the way out; show what is on disk now.
        self.rois = self.recording.rois.copy()
        self.history = []
        self.saving = False
        self.saved = True
        self.app.notify("saved")
        self.changed("saved; esc to go back", now=True)

    def action_back(self):
        if self.busy():
            return
        if self.path or self._stroke:
            self.path, self._stroke = [], None
            self.changed("lasso cleared", now=True)
            return
        if self.dirty:
            self.app.push_screen(
                ConfirmSave(f"Discard {len(self.history)} unsaved edits?", yes="discard"),
                self._discard_confirmed)
            return
        self.dismiss(self._outcome())

    def _discard_confirmed(self, yes):
        if yes:
            self.dismiss(self._outcome())

    def _outcome(self):
        return "saved" if self.saved else None
