"""Find a recording, then look at it -- without deciding anything about it.

Two screens. The browser lists what :mod:`pygor.tui.reader` is willing to open
and nothing else, and costs nothing to move around in: no file is read until
one is chosen. The inspector then loads that file and shows what is in it,
read-only. Re-segmenting is one key away but is a different screen, so that
"check this recording" and "change this recording" never share a keystroke.
"""

from __future__ import annotations

import pathlib

from textual import work
from textual.binding import Binding
from textual.containers import Horizontal, Vertical
from textual.markup import escape
from textual.screen import Screen
from textual.widgets import DirectoryTree, Footer, Header, Static

from pygor.tui import reader


def post_if_current(app, callback, *args) -> bool:
    """Hand a thread worker's result to the interface, unless it is stale.

    ``exclusive=True`` only marks the superseded worker cancelled: a thread
    cannot be stopped, so it runs on and posts its result anyway. Two renders
    in flight -- a held arrow key, a zoom pressed twice while a figure draws --
    can then finish out of order, and the older one lands last and stays: a
    crosshair or trace for a place the status line says you have left.
    Checked here, the older one drops its result instead.
    """
    from textual.worker import get_current_worker

    if get_current_worker().is_cancelled:
        return False
    app.call_from_thread(callback, *args)
    return True


class RecordingTree(DirectoryTree):
    """A directory tree showing only directories and openable recordings."""

    def filter_paths(self, paths):
        keep = [
            path for path in paths
            if not path.name.startswith(".")
            and (
                (path.is_dir() and path.name != "__pycache__")
                or reader.is_recording(path)
            )
        ]
        # Directories first, then recordings, each alphabetically: the layout
        # of an imaging session is date directories with files under them.
        return sorted(keep, key=lambda path: (not path.is_dir(), path.name.lower()))


def _file_facts(path: pathlib.Path) -> str:
    import datetime

    # File names are escaped: a "[copy]" in one is otherwise read as a style
    # and silently vanishes, and a "[/" raises.
    name = escape(path.name)
    try:
        stat = path.stat()
    except OSError as error:
        return f"{name}\n\n[red]{escape(str(error))}[/red]"
    when = datetime.datetime.fromtimestamp(stat.st_mtime).strftime("%Y-%m-%d %H:%M")
    return (
        f"[b]{name}[/b]\n\n"
        f"{reader.KINDS.get(reader.kind(path), 'directory')}\n"
        f"{stat.st_size / 1e6:.1f} MB\n"
        f"modified {when}\n\n"
        f"[dim]{escape(str(path.parent))}[/dim]\n\n"
        "enter to open"
    )


class BrowseScreen(Screen):
    """Pick a recording. Nothing here reads a file's contents."""

    BINDINGS = [
        Binding("escape", "app.quit", "quit"),
        Binding("f", "focus_tree", "tree"),
    ]

    def __init__(self, root, *, caps, previews, n_colours=None, reprocess=None, draw=None):
        super().__init__()
        self.root = pathlib.Path(root).expanduser().resolve()
        self.caps = caps
        self.previews = previews
        self.n_colours = n_colours
        self.reprocess = reprocess
        self.draw = draw

    def compose(self):
        yield Header()
        yield Horizontal(
            RecordingTree(str(self.root), id="tree"),
            Vertical(Static(id="file-facts"), id="browse-side"),
        )
        yield Footer()

    def on_mount(self):
        self.sub_title = str(self.root)
        tree = self.query_one("#tree", RecordingTree)
        tree.focus()
        self.query_one("#file-facts", Static).update(
            "move with the arrows, enter to open a recording"
        )

    def action_focus_tree(self):
        self.query_one("#tree", RecordingTree).focus()

    def on_tree_node_highlighted(self, event):
        entry = event.node.data
        if entry is None:
            return
        self.query_one("#file-facts", Static).update(_file_facts(entry.path))

    def on_directory_tree_file_selected(self, event):
        event.stop()
        self.app.push_screen(
            InspectScreen(
                event.path, caps=self.caps, previews=self.previews,
                n_colours=self.n_colours, reprocess=self.reprocess, draw=self.draw,
            )
        )


class TracePane(Vertical):
    """The trace under the probe, as braille text or as a rendered panel.

    Braille is the default because it is the only one that can keep up with a
    moving cursor; see :mod:`pygor.tui.sparkline`.
    """

    def __init__(self, caps, **kwargs):
        super().__init__(**kwargs)
        self.caps = caps

    def show_text(self, text) -> None:
        # A class, never an id: ``remove_children`` completes asynchronously,
        # so the outgoing Static is still mounted when the replacement goes
        # in, and two widgets sharing an id wedge the message pump rather than
        # raising anywhere visible.
        self.remove_children()
        # Never markup: this is a plot, an ROI label or an exception message.
        self.mount(Static(text, classes="trace-text", markup=False))

    def show_panel(self, panel_image) -> None:
        from pygor.tui.imaging import describe_panel, fit_cells, make_widget

        self.remove_children()
        widget = make_widget(panel_image, self.caps)
        if widget is None:
            self.mount(Static(describe_panel(panel_image)))
            return
        cols, rows = fit_cells(panel_image.width, panel_image.height,
                               self.size.width or 80, self.size.height or 10, self.caps)
        widget.styles.width = cols
        widget.styles.height = rows
        self.mount(widget)

    def size_px(self):
        from pygor.tui.imaging import pixels_for_cells

        return pixels_for_cells(self.size.width or 80, self.size.height or 10, self.caps)


class ImagePointer:
    """Turns a screen position into an image pixel, for a screen with a picture.

    The screen names its :class:`PanelView` in ``PREVIEW_ID`` and provides
    ``image_shape()``, and sets ``_panel_region = None`` in its constructor.
    """

    PREVIEW_ID = ""

    def image_widget(self):
        """The mounted panel widget, or None in text mode or mid-remount."""
        from pygor.tui.app import PanelView

        view = self.query_one(self.PREVIEW_ID, PanelView)
        for child in view.children:
            if not isinstance(child, Static) and child.is_attached:
                return child
        return None

    def pixel_under(self, screen_x, screen_y):
        """The image pixel at a screen position, or None if that is not on the picture.

        Textual reports the pointer in cells, and the panel is centred inside
        its pane, so the mapping is taken from the image widget's own region
        rather than from the pane's.

        Redrawing the crosshair re-mounts that widget, and for a moment there
        is no attached one to ask. A click in that moment used to be dropped,
        which is a click that did nothing every so often and no way for the
        reader to tell why, so the last known geometry stands in.
        """
        shape = self.image_shape()
        if shape is None:
            return None
        widget = self.image_widget()
        region = widget.region if widget is not None else None
        if region is not None and region.width and region.height:
            self._panel_region = region
        else:
            region = self._panel_region
        if region is None or not region.contains(screen_x, screen_y):
            return None
        from pygor.tui import probe as probing

        return probing.pixel_at(screen_x - region.x, screen_y - region.y,
                                region.width, region.height, shape)


class InspectScreen(ImagePointer, Screen):
    """One recording, as it is on disk: facts, a picture, and the signal under the cursor."""

    PREVIEW_ID = "#inspect-preview"

    # How long a burst of pointer movement is allowed to coalesce into one
    # redraw. Every repaint re-transmits the whole PNG to the terminal, so a
    # crosshair that followed the mouse event-for-event would saturate a remote
    # connection; the braille trace is not throttled because it is only text.
    REDRAW_INTERVAL = 0.12

    BINDINGS = [
        Binding("escape", "back", "back"),
        Binding("p", "next_preview", "view"),
        Binding("t", "toggle_trace", "trace style"),
        Binding("plus,equals_sign", "zoom(0.5)", "zoom in"),
        Binding("minus", "zoom(2.0)", "zoom out"),
        Binding("left_square_bracket", "pan(-0.5)", "pan", show=False),
        Binding("right_square_bracket", "pan(0.5)", "pan", show=False),
        Binding("0", "unzoom", "whole trace", show=False),
        Binding("v", "napari", "napari"),
        Binding("r", "reprocess", "re-segment"),
        Binding("d", "draw", "draw ROIs"),
        Binding("comma", "step(-1)", "frame -"),
        Binding("full_stop", "step(1)", "frame +"),
        Binding("less_than_sign", "step(-25)", "frame -25", show=False),
        Binding("greater_than_sign", "step(25)", "frame +25", show=False),
        Binding("left,h", "move(0, -1)", "probe", show=False),
        Binding("right,l", "move(0, 1)", "probe", show=False),
        Binding("up,k", "move(1, 0)", "probe", show=False),
        Binding("down,j", "move(-1, 0)", "probe", show=False),
        Binding("H", "move(0, -10)", "probe", show=False),
        Binding("L", "move(0, 10)", "probe", show=False),
        Binding("K", "move(10, 0)", "probe", show=False),
        Binding("J", "move(-10, 0)", "probe", show=False),
    ]

    def __init__(self, path, *, caps, previews, n_colours=None, reprocess=None, draw=None):
        super().__init__()
        self.path = pathlib.Path(path)
        self.draw = draw
        self.caps = caps
        # The stack view is the inspector's own: the reprocess screen has no
        # use for a frame scrubber, so it is appended here rather than added
        # to the shared PREVIEWS.
        self.previews = (*previews, "stack")
        self.preview_index = 0
        self.n_colours = n_colours
        self.reprocess = reprocess
        self.recording = None
        self.frame = 0
        self.probe = None
        # The stretch of the trace on screen, as fractions of it; see
        # probe.clamp_window for why fractions.
        self.window = (0.0, 1.0)
        # What the trace pane last drew and where, for turning a pointer
        # position back into a time: (slice, start, end, text offset, cols).
        self._trace_view = None
        self.trace_mode = "braille"
        self.limits = None
        self.projection = None
        self._redraw_timer = None
        self._dirty = set()
        self._panel_region = None

    @property
    def view(self):
        return self.previews[self.preview_index % len(self.previews)]

    def compose(self):
        from pygor.tui.app import PanelView

        yield Header()
        yield Horizontal(
            Static(id="meta"),
            Vertical(
                PanelView(self.caps, id="inspect-preview"),
                TracePane(self.caps, id="trace"),
                id="inspect-right",
            ),
        )
        yield Static("", id="status")
        yield Footer()

    def on_mount(self):
        self.sub_title = self.path.name
        self.query_one("#meta", Static).update("loading…")
        self.set_status(f"reading {self.path.name}")
        self.load_worker()

    def set_status(self, text=None):
        """The status line always says which view, which frame and which probe."""
        if self.recording is None:
            self.query_one("#status", Static).update(text or "")
            return
        parts = [f"view: {self.view}"]
        if self.view == "stack":
            total = len(self.recording.images)
            hz = getattr(self.recording, "frame_hz", None)
            at = f" ({self.frame / hz:.1f}s)" if hz else ""
            parts.append(f"frame {self.frame + 1}/{total}{at}")
        if self.probe is not None:
            parts.append(f"probe ({self.probe[0]}, {self.probe[1]})")
        if text:
            parts.append(text)
        self.query_one("#status", Static).update("  ·  ".join(parts))

    # -- loading ----------------------------------------------------------

    @work(thread=True, exclusive=True, group="load")
    def load_worker(self):
        try:
            recording = reader.load_recording(self.path, self.n_colours)
        except Exception as error:
            import traceback

            self.log(traceback.format_exc())
            self.app.call_from_thread(self.load_failed, f"{type(error).__name__}: {error}")
            return
        self.app.call_from_thread(self.loaded, recording)

    def load_failed(self, message):
        from pygor.tui.app import PanelView

        self.query_one("#meta", Static).update(
            f"[red]could not open[/red]\n\n{escape(message)}")
        self.query_one("#inspect-preview", PanelView).show_message(
            "nothing to draw: the recording did not load"
        )
        self.set_status("escape to go back and pick another")

    def show_meta(self):
        rows = reader.summarise(self.recording)
        width = max(len(label) for label, _ in rows)
        self.query_one("#meta", Static).update(
            "\n".join(f"[dim]{label.rjust(width)}[/dim]  {escape(value)}"
                      for label, value in rows)
        )

    def loaded(self, recording):
        self.recording = recording
        self.show_meta()
        # Centre of the field of view, so there is a trace on screen before
        # the pointer has been anywhere: the pane is otherwise an empty
        # rectangle that gives no clue it is waiting to be pointed at.
        images = getattr(recording, "images", None)
        if images is not None and images.ndim == 3:
            self.probe = (images.shape[1] // 2, images.shape[2] // 2)
        self.set_status("click or arrows to probe · + - or scroll to zoom the trace"
                        " · , . scrub · p view")
        self.refresh_preview()
        self.refresh_trace()

    def image_shape(self):
        images = getattr(self.recording, "images", None)
        if images is None or images.ndim != 3:
            return None
        return images.shape[1], images.shape[2]

    # -- actions ----------------------------------------------------------

    def action_back(self):
        self.dismiss(None)

    def action_next_preview(self):
        if self.recording is None:
            return
        self.preview_index += 1
        self.set_status()
        self.refresh_preview()

    def action_toggle_trace(self):
        self.trace_mode = "figure" if self.trace_mode == "braille" else "braille"
        self.refresh_trace()

    def action_step(self, delta):
        """Scrub the stack. Only the stack view has frames, so switch to it."""
        if self.recording is None or self.image_shape() is None:
            return
        if self.view != "stack":
            self.preview_index = self.previews.index("stack")
        total = len(self.recording.images)
        self.frame = min(max(self.frame + delta, 0), total - 1)
        self.set_status()
        self.refresh_preview()

    def action_move(self, d_row, d_col):
        shape = self.image_shape()
        if shape is None:
            return
        row, col = self.probe or (shape[0] // 2, shape[1] // 2)
        self.set_probe((min(max(row + d_row, 0), shape[0] - 1),
                        min(max(col + d_col, 0), shape[1] - 1)))

    def trace_length(self):
        # Frames rather than the trace's own length: it only sets how narrow
        # a zoom may go, and every trace here is one sample per frame.
        images = getattr(self.recording, "images", None)
        return len(images) if images is not None else 0

    def set_window(self, window):
        if window == self.window:
            return
        self.window = window
        self.refresh_trace()

    def action_zoom(self, factor, anchor=0.5):
        from pygor.tui import probe as probing

        if self.recording is None:
            return
        self.set_window(probing.zoom(self.window, factor, anchor, self.trace_length()))

    def action_pan(self, by):
        from pygor.tui import probe as probing

        if self.recording is None:
            return
        self.set_window(probing.pan(self.window, by, self.trace_length()))

    def action_unzoom(self):
        self.set_window((0.0, 1.0))

    def action_napari(self):
        args = ["--path", str(self.path)]
        if self.n_colours:
            args += ["--n-colours", str(self.n_colours)]
        self.app.spawn_napari(args, label=self.path.stem)

    # -- the probe --------------------------------------------------------

    def on_mouse_down(self, event):
        """Click a pixel to probe it. The probe stays there until the next click,
        so the pointer is free to go and read the trace."""
        if self.recording is None:
            return
        found = self.pixel_under(event.screen_x, event.screen_y)
        if found is None:
            return
        event.stop()
        self.set_probe(found)

    # -- reading the trace --------------------------------------------------

    def trace_fraction(self, screen_x, screen_y):
        """How far across the plotted trace a screen position is, 0 to 1, or None."""
        pane = self.query_one("#trace", TracePane)
        if self._trace_view is None or not pane.region.contains(screen_x, screen_y):
            return None
        _, _, _, text_offset, cols = self._trace_view
        if self.trace_mode == "braille":
            text = next((child for child in pane.children
                         if isinstance(child, Static) and child.is_attached), None)
            if text is None:
                return None
            fraction = (screen_x - text.content_region.x - text_offset + 0.5) / cols
        else:
            widget = next((child for child in pane.children
                           if not isinstance(child, Static) and child.is_attached), None)
            if widget is None or not widget.region.width:
                return None
            # The axes span 0.1 to 0.98 of the figure; see trace_figure.
            fraction = ((screen_x - widget.region.x) / widget.region.width - 0.1) / 0.88
        if not 0.0 <= fraction <= 1.0:
            return None
        return fraction

    def on_mouse_move(self, event):
        """Over the trace, say what is under the pointer: when, and how much."""
        fraction = self.trace_fraction(event.screen_x, event.screen_y)
        if fraction is None:
            return
        part, start, end, _, _ = self._trace_view
        values = part.values
        if not len(values):
            return
        value = values[min(int(fraction * len(values)), len(values) - 1)]
        unit = "s" if part.seconds else " fr"
        self.set_status(f"t {start + fraction * (end - start):.2f}{unit}  ·  {value:.3g}")

    def _scroll_zoom(self, event, factor):
        if self.recording is None:
            return
        fraction = self.trace_fraction(event.screen_x, event.screen_y)
        if fraction is None:
            return
        event.stop()
        self.action_zoom(factor, anchor=fraction)

    def on_mouse_scroll_up(self, event):
        self._scroll_zoom(event, 0.8)

    def on_mouse_scroll_down(self, event):
        self._scroll_zoom(event, 1.25)

    def set_probe(self, position):
        if position == self.probe:
            return
        self.probe = position
        self.set_status()
        # The braille trace is text and can be redrawn on every event; the
        # crosshair is a PNG and cannot.
        if self.trace_mode == "braille":
            self.refresh_trace()
        else:
            self._dirty.add("trace")
        self._dirty.add("image")
        self.schedule_redraw()

    def schedule_redraw(self):
        if self._redraw_timer is None:
            self._redraw_timer = self.set_timer(self.REDRAW_INTERVAL, self.redraw_due)

    def redraw_due(self):
        self._redraw_timer = None
        dirty, self._dirty = self._dirty, set()
        if "image" in dirty:
            self.refresh_preview()
        if "trace" in dirty:
            self.refresh_trace()

    def action_reprocess(self):
        if self.recording is None:
            self.app.notify("still loading", severity="warning")
            return
        if self.reprocess is None:
            self.app.notify("re-segmentation is not available here", severity="warning")
            return
        self.app.push_screen(
            self.reprocess(self.recording, self.path), self._reprocessed
        )

    def action_draw(self):
        if self.recording is None:
            self.app.notify("still loading", severity="warning")
            return
        if self.draw is None or self.image_shape() is None:
            self.app.notify("ROI drawing is not available here", severity="warning")
            return
        screen = self.draw(self.recording, self.path)
        # Start where the reader was looking.
        if self.probe is not None:
            screen.cursor = self.probe
        self.app.push_screen(screen, self._reprocessed)

    def _reprocessed(self, outcome):
        # Both the reprocess and the draw screen work on this same recording
        # object, so a save shows up here without re-reading the file.
        if outcome == "saved":
            self.set_status("saved")
            self.show_meta()
        # Cheap insurance: re-segmentation only rewrites the mask today, but a
        # cached projection outliving the images it was made from would be a
        # picture of the wrong recording.
        self.projection = None
        self.limits = None
        self.refresh_preview()
        self.refresh_trace()

    # -- preview ----------------------------------------------------------

    def refresh_preview(self):
        if self.recording is None:
            return
        self.render_preview(self.view, self.frame, self.probe)

    @work(thread=True, exclusive=True, group="preview")
    def render_preview(self, which, frame, marker):
        """Render the current view; a frame superseded mid-render is dropped
        (see :func:`post_if_current`)."""
        from pygor.tui.app import PanelView
        from pygor.tui.imaging import frame_preview, recording_preview
        from pygor.tui.probe import display_range

        view = self.query_one("#inspect-preview", PanelView)
        width, height = view.size_px()
        try:
            if which == "stack":
                if self.limits is None:
                    self.limits = display_range(self.recording.images)
                image = frame_preview(self.recording.images, frame, width, height,
                                      self.limits, marker=marker,
                                      name=getattr(self.recording, "name", ""))
            else:
                if self.projection is None:
                    import numpy as np

                    self.projection = np.mean(self.recording.images, axis=0)
                image = recording_preview(self.recording, which, width, height,
                                          marker=marker, projection=self.projection)
        except Exception as error:
            import traceback

            self.log(traceback.format_exc())
            post_if_current(self.app, view.show_message,
                            escape(f"preview failed: {type(error).__name__}: {error}"))
            return
        post_if_current(self.app, view.show, image)

    # -- trace ------------------------------------------------------------

    def refresh_trace(self):
        if self.recording is None or self.probe is None:
            return
        self.render_trace(self.probe, self.trace_mode, self.window)

    @work(thread=True, exclusive=True, group="trace")
    def render_trace(self, position, mode, window):
        from dataclasses import replace

        from pygor.tui import probe as probing
        from pygor.tui.imaging import trace_figure

        pane = self.query_one("#trace", TracePane)
        cols = max(20, (pane.size.width or 60) - 10)
        # Two rows go to the axis and its labels, one to the trigger strip and
        # one to its summary, so the trace itself gets what is left.
        rows = max(3, (pane.size.height or 8) - 5)
        try:
            trace = probing.at_pixel(self.recording, *position)
            triggers = probing.triggers_of(self.recording)
            # The statistics stay whole-recording while zoomed: they are the
            # diagnostic for the trigger train, and a window of it would
            # report a clean train around a dropout just off screen.
            note = probing.trigger_summary(triggers) if triggers is not None else ""
            part, start, end = probing.windowed(trace, window)
            # Triggers are in seconds; a recording without a frame rate is
            # plotted against frames and shown without them.
            events = triggers if trace.seconds else None
            unit = "s" if trace.seconds else " fr"
            label = trace.label
            if window != (0.0, 1.0):
                label += (f"  ·  {start:.2f}-{end:.2f}{unit}"
                          f"  ({1 / (window[1] - window[0]):.0f}x, 0 resets)")
            if mode == "braille":
                from pygor.tui.sparkline import framed

                body = framed(part.values, cols=cols, rows=rows, x_min=start, x_max=end,
                              x_unit=unit, events=events, event_note=note)
                first = body.splitlines()[0]
                view = (part, start, end, first.index("│") + 1, cols)
                post_if_current(self.app, self._trace_ready, view,
                                pane.show_text, f"{label}\n{body}")
            else:
                width, height = pane.size_px()
                panel = trace_figure(replace(part, label=label), width=width, height=height,
                                     triggers=events, note=note, offset=start)
                post_if_current(self.app, self._trace_ready, (part, start, end, 0, cols),
                                pane.show_panel, panel)
        except Exception as error:
            import traceback

            self.log(traceback.format_exc())
            post_if_current(self.app, self._trace_ready, None, pane.show_text,
                            f"trace failed: {type(error).__name__}: {error}")

    def _trace_ready(self, view, show, content):
        """Show a rendered trace and record its geometry, together.

        On the main thread, so the pointer readout never maps a position with
        one trace's geometry onto a pane showing another -- and after a
        failure it has nothing to map, rather than the last good trace's.
        """
        self._trace_view = view
        show(content)
