"""The cockpit itself: browse fields of view, look at the evidence, judge.

Three screens, one per thing being decided. The index is where you choose what
to work on; the field-of-view screen answers "did the ROIs transfer onto the
right cells"; the cell screen answers "is this receptive field real". Verdict
keys are the same on all of them and always apply to whatever the status bar
says they apply to, because a verdict recorded against the wrong subject is
worse than no verdict.

Panels are rendered in a worker thread and arrive as PNG bytes. Nothing in here
loads a recording on a navigation path.
"""

from __future__ import annotations

import subprocess
import sys
import threading

from textual import on, work
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical
from textual.screen import Screen
from textual.widgets import DataTable, Footer, Header, Input, Static

from pygor.review.session import ReviewSession
from pygor.tui.capabilities import napari_availability
from pygor.tui.imaging import (
    clear_terminal_images,
    describe_panel,
    fit_cells,
    make_widget,
    pixels_for_cells,
)


def embed_shell(header, namespace) -> None:
    """Open IPython on the terminal the interface just released.

    On its own thread, because suspending Textual hands back the terminal but
    leaves its asyncio loop running on this one, and IPython's prompt starts a
    loop of its own -- which raises ``asyncio.run() cannot be called from a
    running event loop``. A fresh thread has no loop, so the prompt can make one.
    The caller blocks on the join, which is what makes this feel like stepping
    out of the interface rather than alongside it.
    """
    try:
        from IPython import embed
    except ImportError:
        input("IPython is not installed. Press enter to return.")
        return

    failure = []

    def run():
        try:
            embed(header=header, user_ns=namespace)
        except BaseException as error:  # noqa: BLE001 - reported, not swallowed
            failure.append(error)

    thread = threading.Thread(target=run, name="pygor-review-repl")
    thread.start()
    thread.join()
    if failure:
        print(f"\nThe embedded shell failed: {failure[0]!r}")
        input("Press enter to return to the cockpit.")


class PanelView(Vertical):
    """A pane holding one rendered panel, swapped as the reviewer moves."""

    def __init__(self, app_caps, **kwargs):
        super().__init__(**kwargs)
        self.caps = app_caps
        self._current = None

    def show(self, panel_image) -> None:
        self.remove_children()
        if panel_image is None:
            self.mount(Static("no panel", classes="dim"))
            return
        widget = make_widget(panel_image, self.caps)
        if widget is None:
            self.mount(Static(describe_panel(panel_image)))
            self._current = panel_image
            return
        # Size it here rather than in CSS: a percentage fills both dimensions
        # and stretches the panel out of shape.
        cols, rows = fit_cells(panel_image.width, panel_image.height,
                               self.size.width or 80, self.size.height or 20,
                               self.caps)
        widget.styles.width = cols
        widget.styles.height = rows
        self.mount(widget)
        self._current = panel_image

    def show_message(self, text) -> None:
        self.remove_children()
        self.mount(Static(text, classes="dim"))

    def size_px(self):
        return pixels_for_cells(self.size.width or 80, self.size.height or 20, self.caps)


class ReviewScreen(Screen):
    """Shared verdict handling, so the three screens cannot disagree about it.

    The verdict keys are declared as bindings rather than handled in ``on_key``
    so that the footer and the help panel list them. A key the interface never
    mentions may as well not exist.
    """

    BINDINGS = [
        Binding("a", "verdict('keep')", "keep"),
        Binding("x", "verdict('reject')", "reject"),
        Binding("f", "verdict('flag')", "flag"),
        Binding("s", "verdict('uncertain')", "uncertain"),
        Binding("full_stop", "repeat", "repeat last"),
        Binding("w", "flush", "save"),
    ]

    def action_flush(self) -> None:
        """Sync the log and rewrite the readable view.

        Verdicts are already on disk -- each one is appended as it is made -- so
        this is for durability against a power cut and for refreshing the CSV
        that everything downstream reads.
        """
        self.app.session.store.flush()
        path = self.app.session.store.materialise()
        self.app.notify(f"synced; wrote {path.name}")

    def action_verdict(self, verdict) -> None:
        # Reject and flag say something is wrong; without a reason, the record
        # is unreadable by the time anyone acts on it.
        if verdict in ("reject", "flag"):
            self.app.push_screen(ReasonPrompt(verdict), self._with_reason)
        else:
            self.record(verdict)

    def action_repeat(self) -> None:
        if self.app.last_verdict:
            self.record(*self.app.last_verdict)
        else:
            self.app.notify("nothing to repeat yet", severity="warning")

    def verdict_target(self):
        """(subject_type, subject_uid, check, role, channel) for the verdict keys."""
        raise NotImplementedError

    def current_bundle(self):
        raise NotImplementedError

    def record(self, verdict, reason=""):
        bundle = self.current_bundle()
        if bundle is None:
            self.app.notify("nothing selected", severity="warning")
            return
        subject_type, subject_uid, check, role, channel = self.verdict_target()
        self.app.session.judge(
            bundle,
            subject_type=subject_type,
            subject_uid=subject_uid,
            check=check,
            verdict=verdict,
            role=role,
            channel=channel,
            reason=reason,
        )
        self.app.last_verdict = (verdict, reason)
        self.app.notify(f"{verdict}: {subject_uid} [{check}]")
        self.refresh_status()

    def refresh_status(self) -> None:
        pass

    def _with_reason(self, result) -> None:
        if result is None:
            return
        verdict, reason = result
        self.record(verdict, reason)


class ReasonPrompt(Screen):
    """One line of free text, because a reject with no reason is unusable later."""

    BINDINGS = [Binding("escape", "cancel", "cancel")]

    def __init__(self, verdict):
        super().__init__()
        self.verdict = verdict

    def compose(self) -> ComposeResult:
        yield Vertical(
            Static(f"reason for [b]{self.verdict}[/b] (enter to accept, esc to cancel)"),
            Input(placeholder="what is wrong with it", id="reason"),
            id="reason-box",
        )

    def on_mount(self) -> None:
        self.query_one("#reason", Input).focus()

    @on(Input.Submitted)
    def submit(self, event) -> None:
        self.dismiss((self.verdict, event.value))

    def action_cancel(self) -> None:
        self.dismiss(None)


class IndexScreen(ReviewScreen):
    """Fields of view, worst first, with what has been decided about each."""

    BINDINGS = [
        Binding("enter", "open", "open"),
        Binding("t", "strf_page", "more RFs"),
        Binding("r", "reload", "reload"),
        Binding("q", "quit", "quit"),
    ]

    COLUMNS = ("", "fov_uid", "roles", "cells", "shift", "corr", "lost", "bad")

    GLYPHS = {"keep": "OK", "reject": "NO", "flag": "!!", "uncertain": "??", "": ""}

    def compose(self) -> ComposeResult:
        yield Header()
        yield Horizontal(
            DataTable(id="fovs"),
            Vertical(
                PanelView(self.app.caps, id="preview"),
                PanelView(self.app.caps, id="strfs"),
                id="previews",
            ),
        )
        yield Static("", id="status")
        yield Footer()

    def on_mount(self) -> None:
        table = self.query_one("#fovs", DataTable)
        table.cursor_type = "row"
        table.zebra_stripes = True
        for column in self.COLUMNS:
            table.add_column(column, key=column)
        self._strf_timer = None
        self.strf_page = 0
        self.load_rows()

    def load_rows(self) -> None:
        table = self.query_one("#fovs", DataTable)
        table.clear()
        self.frame = self.app.session.overview()
        for row in self.frame.itertuples():
            corr = "" if row.worst_correlation != row.worst_correlation else f"{row.worst_correlation:.2f}"
            shift = "" if row.max_shift_px != row.max_shift_px else f"{row.max_shift_px:.1f}"
            table.add_row(
                self.GLYPHS.get(row.verdict, ""), row.fov_uid, row.roles,
                str(row.n_cells), shift, corr, str(row.n_lost),
                str(row.bad_status + row.unusable),
            )
        self.refresh_status()

    def refresh_glyphs(self) -> None:
        """Repaint the verdict column from the store.

        Judging is meant to feel like marking a list, so a decision has to show
        up on the row that was decided rather than at the next reload. Cheap
        enough to do wholesale: the store answers from memory.
        """
        from textual.coordinate import Coordinate

        table = self.query_one("#fovs", DataTable)
        if not table.row_count or self.frame.empty:
            return
        for index, row in enumerate(self.frame.itertuples()):
            decided = self.app.session.store.get(
                "fov", row.fov_uid, "alignment", condition=row.condition
            )
            glyph = self.GLYPHS.get(decided.verdict if decided else "", "")
            if table.get_cell_at(Coordinate(index, 0)) != glyph:
                table.update_cell_at(Coordinate(index, 0), glyph)

    def on_screen_resume(self) -> None:
        """Coming back from a field of view: show what was decided in there."""
        self.refresh_glyphs()

    def current_bundle(self):
        table = self.query_one("#fovs", DataTable)
        if not table.row_count:
            return None
        row = self.frame.iloc[table.cursor_row]
        return self.app.session.find(row.fov_uid, row.condition)

    def verdict_target(self):
        bundle = self.current_bundle()
        return "fov", bundle.fov_uid, "alignment", "", -1

    @on(DataTable.RowSelected)
    def open_selected(self, event) -> None:
        """The focused table consumes Enter, so the screen binding never sees it."""
        self.action_open()

    @on(DataTable.RowHighlighted)
    def preview(self, event) -> None:
        bundle = self.current_bundle()
        self.app.render_into(self.query_one("#preview", PanelView), "fov_alignment",
                             bundle)
        # The receptive fields need the recording in memory, which is seconds.
        # Only worth starting once the cursor has settled, so holding a movement
        # key does not queue a load per row.
        if self._strf_timer is not None:
            self._strf_timer.stop()
        self.query_one("#strfs", PanelView).show_message("receptive fields…")
        self._strf_timer = self.set_timer(1.2, self.load_strfs)
        self.refresh_status()

    def load_strfs(self) -> None:
        self.app.render_into(self.query_one("#strfs", PanelView), "fov_strf_sheet",
                             self.current_bundle(), role=None, page=self.strf_page)

    def action_strf_page(self) -> None:
        self.strf_page += 1
        self.load_strfs()

    def refresh_status(self) -> None:
        self.refresh_glyphs()
        progress = self.app.session.progress()
        bundle = self.current_bundle()
        target = f"  target: FOV {bundle.fov_uid}" if bundle else ""
        self.query_one("#status", Static).update(
            f"{progress['fovs_judged']}/{progress['fovs']} fields judged  ·  "
            f"{progress['verdicts']} verdicts{target}"
        )

    def action_open(self) -> None:
        bundle = self.current_bundle()
        if bundle is not None:
            self.app.push_screen(FovScreen(bundle))

    def action_reload(self) -> None:
        """Re-read the dataset from disk, for after a reprocess."""
        self.app.session.rescan()
        self.load_rows()
        self.app.notify("rescanned")


class FovScreen(ReviewScreen):
    """One field of view: is the ROI transfer trustworthy."""

    BINDINGS = [
        Binding("escape", "back", "back"),
        Binding("p", "next_panel", "panel"),
        Binding("greater_than_sign", "next_role", "partner"),
        Binding("enter", "cells", "cells"),
        Binding("R", "reprocess", "reprocess"),
        Binding("v", "napari", "napari"),
    ]

    def __init__(self, bundle):
        super().__init__()
        self.bundle = bundle
        self.panels = list(self.app.binding.PANEL_SETS["fov"])
        self.panel_index = 0
        self.role_index = 0

    def compose(self) -> ComposeResult:
        yield Header()
        yield PanelView(self.app.caps, id="panel")
        yield Static("", id="detail")
        yield Static("", id="status")
        yield Footer()

    def on_mount(self) -> None:
        self.redraw()

    def current_bundle(self):
        return self.bundle

    @property
    def role(self):
        roles = [r for r in self.bundle.roles if r != self.bundle.master_role]
        return roles[self.role_index % len(roles)] if roles else None

    def verdict_target(self):
        return "fov", self.bundle.fov_uid, "alignment", "", -1

    def redraw(self) -> None:
        panel = self.panels[self.panel_index % len(self.panels)]
        self.app.render_into(self.query_one("#panel", PanelView), panel, self.bundle,
                             role=self.role)
        lines = []
        for name, info in self.bundle.alignments().items():
            ref = self.bundle.peek(name)
            line = f"{name:5s} {ref.stem[:38]:38s} {ref.num_rois:4d} ROIs  {info.method}"
            if info.transferred:
                line += f"  r={info.correlation:.3f}  lost={info.n_lost}"
            if not info.usable:
                line += "  PROVENANCE INCONSISTENT"
            lines.append(line)
        for name, why in self.bundle.missing().items():
            lines.append(f"{name:5s} MISSING — {why}")
        self.query_one("#detail", Static).update("\n".join(lines))
        self.refresh_status()

    def refresh_status(self) -> None:
        current = self.app.session.verdict_for(self.bundle)
        decided = f"{current.verdict} ({current.reason})" if current else "undecided"
        self.query_one("#status", Static).update(
            f"panel {self.panels[self.panel_index % len(self.panels)]}  ·  "
            f"partner {self.role}  ·  target: FOV alignment  ·  {decided}"
        )

    def action_next_panel(self) -> None:
        self.panel_index += 1
        self.redraw()

    def action_next_role(self) -> None:
        self.role_index += 1
        self.redraw()

    def action_back(self) -> None:
        self.dismiss()

    def action_cells(self) -> None:
        self.app.push_screen(CellScreen(self.bundle))

    def action_napari(self) -> None:
        self.app.open_napari(self.bundle, self.role)

    def action_reprocess(self) -> None:
        """Edit the recipe for this field of view, re-run it, keep or discard."""
        binding = self.app.binding
        if not hasattr(binding, "run_reprocess"):
            self.app.notify("this dataset binding has no reprocess()", severity="warning")
            return
        from pygor.tui.imaging import PREVIEWS, preview_image, recording_preview
        from pygor.tui.reprocess_screen import ReprocessScreen, segmentation_gating

        bundle = self.bundle
        master = bundle.master_role

        def run(overrides):
            return binding.run_reprocess(bundle.fov_uid, overrides)

        def preview(result, which, width, height):
            if result is None:
                # Nothing re-run yet: draw what is on disk. The small datasets
                # cover the mean and the mask; a correlation projection is not
                # saved, so that view loads the recording and computes one.
                arrays = bundle.light(master)
                correlation = arrays.get("correlation_projection")
                if which == "correlation" and correlation is None:
                    correlation = bundle.full(master).compute_correlation_projection()
                return preview_image(
                    arrays["rois"], arrays["average_stack"], which, width, height,
                    name=f"ON DISK  {bundle.peek(master).stem}",
                    correlation=correlation,
                )
            recording = result.get(master) or next(iter(result.values()))
            return recording_preview(recording, which, width, height,
                                     name=f"RE-RUN, UNSAVED  {recording.name}")

        def save(result, overrides):
            binding.save_reprocessed(bundle.fov_uid, result, overrides)
            self.app.session.rescan()

        values = binding.recipe_values()
        choices, gates = segmentation_gating(values)
        # Say what a re-run touches: the master is re-segmented and its ROIs
        # transferred onto every partner, so all of them are recomputed.
        partners = [r for r in bundle.roles if r != master]
        scope = f"{master} → {', '.join(partners)}" if partners else master
        self.app.push_screen(
            ReprocessScreen(
                title=f"reprocess {bundle.fov_uid}   [{scope}]",
                values=values,
                sections=getattr(binding, "REPROCESS_SECTIONS", ("segmentation",)),
                run=run, preview=preview, save=save, caps=self.app.caps,
                previews=PREVIEWS, choices=choices, gates=gates,
                save_text=(f"Overwrite {len(bundle.roles)} processed recording(s) for "
                           f"{bundle.fov_uid} ({', '.join(bundle.roles)}) and its rows in "
                           "the aggregate CSV? The old files are kept as .prereprocess."),
            ),
            self._after_reprocess,
        )

    def _after_reprocess(self, outcome) -> None:
        if outcome == "saved":
            # This screen's bundle describes the files that were just replaced.
            self.dismiss()


class CellScreen(ReviewScreen):
    """One cell at a time: is this receptive field real."""

    # Textual names printable keys after the character, so ']' arrives as
    # right_square_bracket. PageUp/PageDown are bound alongside because the
    # brackets are awkward on some layouts.
    BINDINGS = [
        Binding("escape", "back", "back"),
        Binding("pagedown", "next_cell", "next cell"),
        Binding("pageup", "prev_cell", "prev cell"),
        Binding("right_square_bracket", "next_cell", "next cell", show=False),
        Binding("left_square_bracket", "prev_cell", "prev cell", show=False),
        Binding("n", "next_cell", "next", show=False),
        Binding("N", "prev_cell", "prev", show=False),
        Binding("p", "next_panel", "panel"),
    ]

    def __init__(self, bundle):
        super().__init__()
        self.bundle = bundle
        self.roi = 0
        self.panels = list(self.app.binding.PANEL_SETS["cell"])
        self.panel_index = 0

    def compose(self) -> ComposeResult:
        yield Header()
        yield PanelView(self.app.caps, id="panel")
        yield Static("", id="status")
        yield Footer()

    def on_mount(self) -> None:
        self.app.warm_bundle(self.bundle, self.bundle.strf_role)
        self.redraw()

    def current_bundle(self):
        return self.bundle

    def verdict_target(self):
        return ("cell", f"{self.bundle.fov_uid}#{self.roi}", "rf_quality",
                self.bundle.strf_role, -1)

    def redraw(self) -> None:
        panel = self.panels[self.panel_index % len(self.panels)]
        self.app.render_into(self.query_one("#panel", PanelView), panel, self.bundle,
                             roi=self.roi, role=self.bundle.strf_role)
        self.refresh_status()

    def refresh_status(self) -> None:
        current = self.app.session.verdict_for(
            self.bundle, subject_type="cell",
            subject_uid=f"{self.bundle.fov_uid}#{self.roi}",
            check="rf_quality", role=self.bundle.strf_role,
        )
        decided = f"{current.verdict} ({current.reason})" if current else "undecided"
        self.query_one("#status", Static).update(
            f"cell {self.roi + 1}/{self.bundle.n_cells}  ·  "
            f"panel {self.panels[self.panel_index % len(self.panels)]}  ·  "
            f"target: cell RF quality  ·  {decided}"
        )

    def action_next_cell(self) -> None:
        self.roi = min(self.roi + 1, self.bundle.n_cells - 1)
        self.redraw()

    def action_prev_cell(self) -> None:
        self.roi = max(self.roi - 1, 0)
        self.redraw()

    def action_next_panel(self) -> None:
        self.panel_index += 1
        self.redraw()

    def action_back(self) -> None:
        self.dismiss()


class ProofreadApp(App):
    """The cockpit."""

    CSS = """
    Screen { background: $surface; }
    #fovs { width: 3fr; }
    #previews { width: 2fr; padding: 0 1; }
    #preview { height: 2fr; }
    #strfs { height: 3fr; }
    /* The image's own size is set in PanelView.show, preserving its aspect;
       a percentage here would fill both dimensions and stretch it. */
    PanelView { height: 1fr; overflow: hidden; align: center middle; }
    #detail { height: auto; padding: 0 1; color: $text-muted; }
    #status { height: 1; padding: 0 1; background: $panel; }
    #reason-box { padding: 1 2; width: 60%; height: auto; background: $panel; }
    #value-box, #confirm-box { padding: 1 2; width: 70%; height: auto; background: $panel; }
    /* Its own id: #preview belongs to the index screen, whose 2fr height rule
       would otherwise make this pane twice the screen and push the image off it. */
    #params { width: 2fr; height: 1fr; }
    #reprocess-preview { width: 3fr; height: 1fr; padding: 0 1; }
    .error { color: $error; }
    .dim { color: $text-muted; }
    """

    BINDINGS = [
        Binding("question_mark", "help", "help"),
        Binding("ctrl+l", "unscramble", "redraw"),
        Binding("o", "repl", "ipython"),
        Binding("q", "quit", "quit"),
    ]

    def __init__(self, session: ReviewSession, binding, caps, goto=None):
        super().__init__()
        self.session = session
        self.binding = binding
        self.caps = caps
        self.goto = goto
        self.last_verdict = None
        self._napari_procs = []

    def on_mount(self) -> None:
        self.push_screen(IndexScreen())
        if self.goto:
            try:
                self.push_screen(FovScreen(self.session.find(self.goto)))
            except KeyError as error:
                self.notify(str(error), severity="error", timeout=10)

    # -- rendering --------------------------------------------------------

    @work(thread=True, exclusive=True, group="render")
    def render_into(self, view: PanelView, panel, bundle, **kwargs) -> None:
        """Render off the UI thread; a fast cursor must not block on a panel."""
        if bundle is None:
            return
        width, height = view.size_px()
        try:
            image = self.session.render(panel, bundle, width=width, height=height,
                                        **kwargs)
        except Exception as error:
            self.call_from_thread(self.notify, f"{panel}: {error}", severity="error")
            return
        self.call_from_thread(view.show, image)

    @work(thread=True, exclusive=True, group="warm")
    def warm_bundle(self, bundle, role=None) -> None:
        """Pay a field of view's derived costs once, before walking its cells."""
        self.call_from_thread(self.notify, f"loading {bundle.fov_uid}…")
        try:
            bundle.warm(role or bundle.strf_role)
        except Exception as error:
            self.call_from_thread(self.notify, f"warm failed: {error}",
                                  severity="warning")
            return
        self.call_from_thread(self.notify, "ready")

    # -- hand-offs --------------------------------------------------------

    def open_napari(self, bundle, role=None) -> None:
        """Launch napari as a detached subprocess, never in this process.

        Qt cannot share a process with Textual: it would block the event loop or
        write to the terminal and shred the display. Output goes to a log file
        for the same reason.
        """
        ok, why = napari_availability()
        if not ok:
            self.notify(why, severity="error", timeout=10)
            return
        import pathlib
        import tempfile

        log = pathlib.Path(tempfile.gettempdir()) / f"pygor-napari-{bundle.prefix}.log"
        handle = log.open("w")
        process = subprocess.Popen(
            [sys.executable, "-m", "pygor.tui.napari_launcher",
             "--path", str(bundle.peek(role or bundle.master_role).path),
             "--master", str(bundle.peek(bundle.master_role).path)],
            stdin=subprocess.DEVNULL, stdout=handle, stderr=handle,
            start_new_session=True,
        )
        self._napari_procs.append(process)
        self.notify(f"napari starting (log: {log})", timeout=6)

    def action_repl(self) -> None:
        """Drop to IPython with the current objects bound, then come back."""
        screen = self.screen
        bundle = screen.current_bundle() if isinstance(screen, ReviewScreen) else None
        from functools import partial

        from pygor.tui.imaging import show, show_rois

        namespace = {
            "session": self.session,
            "bundle": bundle,
            "store": self.session.store,
            "app": self,
            "binding": self.binding,
            # show() is what makes pygor's matplotlib API usable over SSH:
            # any figure it returns is drawn inline instead of needing a window.
            "show": partial(show, caps=self.caps),
            "show_rois": partial(show_rois, caps=self.caps),
        }
        lines = [
            "pygor review.  Ctrl-D returns to the cockpit.",
            "  session  ReviewSession      bundle  this field of view",
            "  store    verdicts           binding dataset rules",
            "  show(x)  draw a figure / (fig, ax) / 2-D array inline",
            "  show_rois(rec, labels=False)  ROI outlines over a stretched projection",
        ]
        if hasattr(self.binding, "reprocess"):
            namespace["reprocess"] = self.binding.reprocess
            lines.append(
                "  reprocess(fov_uid, {'segmentation.blob': {'threshold': .02}}, save=False)"
            )
        if bundle is not None:
            # Bind the recording only when it is already in memory. Loading one
            # here would stall for seconds on a keypress that is supposed to be
            # a quick look at what is in front of you.
            role = bundle.strf_role
            if role in bundle.refs and self.session.bundle_cache.loaded(bundle.peek(role)):
                namespace["rec"] = bundle.full(role)
                lines.append(f"  rec      the loaded {role} recording")
            else:
                lines.append(f"  rec      not loaded; bundle.full({role!r}) to load it")
            if isinstance(screen, CellScreen):
                namespace["roi"] = screen.roi
                lines.append(f"  roi      {screen.roi}, the cell on screen")

        with self.suspend():
            clear_terminal_images()
            embed_shell("\n".join(lines), namespace)

    def action_unscramble(self) -> None:
        clear_terminal_images()
        self.refresh(repaint=True, layout=True)
        if isinstance(self.screen, ReviewScreen):
            getattr(self.screen, "redraw", lambda: None)()

    def action_help(self) -> None:
        """Textual's own key panel, so it lists the real bindings.

        A hand-written cheat sheet goes stale the moment a binding changes; this
        one is generated from what is actually bound.
        """
        self.action_show_help_panel()

    def on_unmount(self) -> None:
        self.session.store.flush()
        clear_terminal_images()
