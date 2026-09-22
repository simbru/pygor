"""Edit parameters, re-run, look at the result, decide whether to keep it.

The loop the napari workflow was built for, in a terminal. The screen owns
none of the processing: it is handed a ``run`` that turns overrides into a
result, a ``preview`` that turns a result into a picture, and a ``save`` that
commits a result to disk. That is what lets the same screen sit inside the
proofreading cockpit, bound to a dataset's full pipeline, and inside a
standalone tool bound to one recording's ``segment_rois``.

Nothing touches disk until ``save`` is pressed and confirmed.
"""

from __future__ import annotations

from textual import work
from textual.binding import Binding
from textual.containers import Horizontal, Vertical
from textual.screen import Screen
from textual.widgets import Footer, Header, Static

from pygor.tui.params_table import ParamTable


def segmentation_gating(values):
    """Choices and gates that make the segmentation mode a drop-down.

    The mode row picks from the segmentation sub-sections present in
    ``values``, and every other mode's parameters are hidden. Derived from the
    dict rather than listed, so a mode pygor gains later appears by itself.
    """
    section = values.get("segmentation", {})
    modes = tuple(sorted(
        key for key in section
        if isinstance(section[key], dict) and key != "general"
        and not key.endswith("_postprocess")
    ))
    if not modes:
        return {}, {}
    return {"segmentation.general.mode": modes}, {"segmentation.general.mode": "segmentation"}


class ConfirmSave(Screen):
    BINDINGS = [Binding("y", "yes", "yes"), Binding("n,escape", "no", "no")]

    def __init__(self, text):
        super().__init__()
        self.text = text

    def compose(self):
        yield Vertical(Static(self.text), Static("[b]y[/b] save   [b]n[/b] cancel"),
                       id="confirm-box")

    def action_yes(self):
        self.dismiss(True)

    def action_no(self):
        self.dismiss(False)


class ReprocessScreen(Screen):
    """Parameter table on the left, preview on the right, verdict in the keys."""

    BINDINGS = [
        Binding("escape", "back", "back"),
        Binding("R", "run", "re-run"),
        Binding("S", "save", "save result"),
        Binding("p", "next_preview", "preview"),
    ]

    def __init__(self, *, title, values, sections, run, preview, save, caps,
                 previews=("rois",), save_text=None, choices=None, gates=None,
                 default_master=None):
        super().__init__()
        self.title_text = title
        self.values = values
        self.sections = sections
        self.choices = choices
        self.gates = gates
        self.default_master = default_master
        self.run_fn = run
        self.preview_fn = preview
        self.save_fn = save
        self.caps = caps
        self.previews = tuple(previews)
        self.preview_index = 0
        self.save_text = save_text or "Write the re-run result to disk, replacing what is there?"
        self.result = None
        self.result_overrides = None
        self.running = False
        self.saved = False

    def compose(self):
        from pygor.tui.app import PanelView

        yield Header()
        yield Horizontal(
            ParamTable(self.values, sections=self.sections, choices=self.choices,
                       gates=self.gates, id="params"),
            PanelView(self.caps, id="reprocess-preview"),
        )
        yield Static("", id="status")
        yield Footer()

    def chosen_master(self):
        """The role the table currently names as master, or the default.

        The preview asks for this rather than the field of view's saved
        master: choosing a different recording to segment should change the
        picture straight away, not only once the re-run finishes.
        """
        table = self.query_one("#params", ParamTable)
        for path in table.choices:
            if path.endswith("master_role"):
                value = table.current_value(path)
                if value:
                    return str(value)
        return self.default_master

    def on_mount(self):
        self.sub_title = self.title_text
        self.query_one("#params", ParamTable).focus()
        self.set_status("edit values with enter, then R to re-run")
        self.refresh_preview()

    def set_status(self, text):
        table = self.query_one("#params", ParamTable)
        edits = len(table.overrides)
        state = "result: unsaved re-run" if self.result is not None else "result: on disk"
        master = self.chosen_master()
        scope = f"master: {master}  ·  " if master else ""
        self.query_one("#status", Static).update(
            f"{scope}{edits} edit{'s' if edits != 1 else ''}  ·  {state}  ·  {text}"
        )

    # -- actions ----------------------------------------------------------

    def action_run(self):
        if self.running:
            self.app.notify("already running", severity="warning")
            return
        overrides = self.query_one("#params", ParamTable).nested_overrides()
        self.running = True
        self.set_status("running…")
        # Replace the picture, not just the status line: over a remote
        # connection the status line is the easiest thing to miss, and an
        # unchanged image reads as "nothing happened".
        from pygor.tui.app import PanelView

        self.query_one("#reprocess-preview", PanelView).show_message(
            f"running with {overrides or 'the recipe as-is'} …\n"
            "the result replaces this when it finishes"
        )
        self.run_worker_thread(overrides)

    @work(thread=True, exclusive=True, group="reprocess")
    def run_worker_thread(self, overrides):
        try:
            result = self.run_fn(overrides)
        except Exception as error:
            self.app.call_from_thread(self.finished, None, overrides, f"{type(error).__name__}: {error}")
            return
        self.app.call_from_thread(self.finished, result, overrides, "")

    def finished(self, result, overrides, error):
        self.running = False
        if error:
            self.set_status(f"failed: {error}")
            self.app.notify(error, severity="error", timeout=12)
            return
        self.result = result
        self.result_overrides = overrides
        self.set_status("done — p to switch preview, S to save")
        self.refresh_preview()

    def action_save(self):
        if self.result is None:
            self.app.notify("nothing to save: re-run first", severity="warning")
            return
        self.app.push_screen(ConfirmSave(self.save_text), self._save_confirmed)

    def _save_confirmed(self, yes):
        if not yes:
            return
        try:
            self.save_fn(self.result, self.result_overrides)
        except Exception as error:
            self.app.notify(f"save failed: {error}", severity="error", timeout=12)
            return
        self.app.notify("saved")
        self.result = None
        self.saved = True
        self.set_status("saved; esc to go back")

    def on_param_table_changed(self, event=None):
        """A parameter changed; the master may have, so redraw."""
        self.refresh_preview()
        self.set_status("edit values with enter, then R to re-run")

    def action_next_preview(self):
        self.preview_index += 1
        self.refresh_preview()

    def action_back(self):
        """Dismisses with what happened: 'saved', 'discarded' or 'unchanged'."""
        if self.result is not None:
            self.app.notify("unsaved re-run discarded", severity="warning")
            self.dismiss("discarded")
        else:
            self.dismiss("saved" if self.saved else "unchanged")

    # -- preview ----------------------------------------------------------

    def refresh_preview(self):
        which = self.previews[self.preview_index % len(self.previews)]
        self.render_preview(which)

    @work(thread=True, exclusive=True, group="preview")
    def render_preview(self, which):
        from pygor.tui.app import PanelView

        view = self.query_one("#reprocess-preview", PanelView)
        width, height = view.size_px()
        try:
            image = self._call_preview(self.result, which, width, height)
        except Exception as error:
            import traceback

            self.log(traceback.format_exc())
            self.app.call_from_thread(
                view.show_message,
                f"preview failed: {type(error).__name__}: {error}")
            return
        self.app.call_from_thread(view.show, image)

    def _call_preview(self, result, which, width, height):
        """Hand the chosen master to previews that take one.

        A preview that does not accept ``role`` is still valid -- the
        standalone tool has a single recording and nothing to choose.
        """
        import inspect

        try:
            takes_role = "role" in inspect.signature(self.preview_fn).parameters
        except (TypeError, ValueError):
            takes_role = False
        if takes_role:
            return self.preview_fn(result, which, width, height,
                                   role=self.chosen_master())
        return self.preview_fn(result, which, width, height)
