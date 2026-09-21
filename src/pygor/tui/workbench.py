"""``pygor-workbench`` -- a terminal analysis workflow for one recording.

Proof of concept for running the pipeline that the dataset modules under
``Analysis_scripts/analyses/datasets/`` hard-code -- register, preprocess,
segment, extract traces, snippets, STA -- interactively, with the parameters
of the current step in an editable table and a picture of the result next to
it. The pieces are the ones the proofreading cockpit already uses
(:class:`~pygor.tui.params_table.ParamTable`, :class:`~pygor.tui.app.PanelView`,
the imaging helpers); what is new is the step list, the log and the views.

    pygor-workbench examples/strf_demo_data.h5
    pygor-workbench recording.smp --n-colours 4 --config analyses/recipes/swn.toml

Layout::

    ┌ steps ──────┬ plot ─────────────────────────┬ parameters ─────────┐
    │ ✓ register  │                               │ segmentation.blob…  │
    │ ✓ preprocess│        current view           │ threshold  0.025    │
    │ › segment   │                               │ …                   │
    │   traces    ├ log ──────────────────────────┤                     │
    │   snippets  │ stdout of the running step    │                     │
    │   STA       │                               │                     │
    │ views…      │                               │                     │
    └─────────────┴───────────────────────────────┴─────────────────────┘

Edits in the table are overrides on the recording's parameters; they are
written into ``rec.params`` when a step is run, so the steps pick them up the
same way they pick up a recipe TOML. Nothing is written to disk until save is
chosen from the menu and confirmed.

Deliberately single-recording. The FOV-level pattern (master segmentation,
``transfer_rois_from`` onto partners) is what a dataset binding knows; the
natural next step is a recipe object that this screen drives per member.
"""

from __future__ import annotations

import argparse
import contextlib
import dataclasses
import io
import os
import pathlib
import shutil
import sys
import threading
from functools import partial
from typing import Callable

import numpy as np

# -- recipe -----------------------------------------------------------------

# Values a step needs that are not in the recording's config (trigger_mode,
# noise array, ...) live under a synthetic ``workbench`` section of the
# parameter table -- see WorkbenchState.values(); apply_overrides() routes them
# back to the state rather than into rec.params.


@dataclasses.dataclass
class Step:
    key: str
    label: str
    # Dotted prefixes of the parameter table rows shown for this step.
    sections: tuple[str, ...]
    run: Callable
    done: Callable
    applies: Callable = lambda rec: True
    hint: str = ""


def _is_array(value, ndim=None) -> bool:
    if value is None or isinstance(value, float):  # np.nan sentinel
        return False
    array = np.asarray(value)
    return array.size > 0 and (ndim is None or array.ndim == ndim)


def _run_register(rec, state):
    stats = rec.register(plot=False, verbose=True)
    shift = stats.get("mean_shift") if isinstance(stats, dict) else None
    return f"registered; mean shift {shift}" if shift is not None else "registered"


def _run_preprocess(rec, state):
    rec.preprocess()
    return "preprocessed"


def _run_segment(rec, state):
    rec.segment_rois(plot=False)
    return f"{rec.num_rois} ROIs"


def _run_traces(rec, state):
    rec.extract_traces_from_rois(baseline_dur=state.baseline_dur)
    return f"traces {np.shape(rec.traces_znorm)}"


def _run_snippets(rec, state):
    rec.trigger_mode = int(state.trigger_mode)
    rec.compute_snippets_and_averages()
    return f"averages {np.shape(rec.averages)} (trigger_mode={rec.trigger_mode})"


def _run_strf(rec, state):
    if not state.noise_array:
        raise ValueError("workbench.noise_array is empty: set the path of the noise .npy/.h5")
    noise = load_noise(state.noise_array)
    kwargs = {"noise_array": noise}
    if state.n_colours:
        kwargs["n_colours"] = int(state.n_colours)
    rec.calculate_strf(**kwargs)
    rec.params.mark_step("calculate_strf", {"n_colours": state.n_colours,
                                            "noise_array": str(state.noise_array)})
    return f"strfs {np.shape(rec.strfs)}"


def _has_traces(rec):
    return _is_array(getattr(rec, "traces_znorm", None), ndim=2)


STEPS = (
    Step("register", "register", ("registration",), _run_register,
         lambda rec: bool(rec.params.registered), hint="motion correction, in place"),
    Step("preprocess", "preprocess", ("preprocessing",), _run_preprocess,
         lambda rec: bool(rec.params.preprocessed), hint="artifact cut, flip, detrend"),
    Step("segment_rois", "segment ROIs", ("segmentation",), _run_segment,
         lambda rec: bool(getattr(rec, "num_rois", 0)), hint="mode from segmentation.general"),
    Step("extract_traces", "extract traces", ("workbench.baseline_dur",), _run_traces,
         _has_traces, hint="needs ROIs"),
    Step("snippets", "snippets + averages", ("workbench.trigger_mode", "triggers"),
         _run_snippets, lambda rec: _is_array(getattr(rec, "averages", None)),
         hint="trigger_mode must match the stimulus"),
    Step("calculate_strf", "calculate STRF",
         ("workbench.noise_array", "workbench.n_colours", "strf.general", "strf.calculate"),
         _run_strf, lambda rec: _is_array(getattr(rec, "strfs", None), ndim=4),
         applies=lambda rec: hasattr(rec, "calculate_strf"), hint="needs the noise array"),
)


def load_noise(path):
    path = pathlib.Path(path).expanduser()
    if path.suffix == ".npy":
        return np.load(path)
    import h5py

    with h5py.File(path, "r") as handle:
        keys = list(handle.keys())
        if len(keys) != 1:
            raise ValueError(f"{path.name} has {len(keys)} datasets; expected one: {keys}")
        return handle[keys[0]][()]


# -- views ------------------------------------------------------------------


@dataclasses.dataclass
class View:
    key: str
    label: str
    render: Callable  # (rec, state, width) -> matplotlib Figure
    applies: Callable = lambda rec: True
    per_roi: bool = False


def _projection(rec, state, width):
    from pygor.tui.imaging import roi_figure

    projection = np.mean(rec.images, axis=0)
    mask = getattr(rec, "rois", None)
    if mask is None or not np.any(np.asarray(mask) < 0):
        mask = np.zeros_like(projection, dtype=int)
    return roi_figure(mask=mask, projection=projection, width=width,
                      title=f"{rec.name} mean")


def _correlation(rec, state, width):
    from pygor.tui.imaging import CORRELATION_CMAP, roi_figure

    correlation = getattr(rec, "correlation_projection", None)
    if correlation is None:
        # Single process: the default fans out loky workers, which inherit
        # Textual's redirected stdio and die with a bad file descriptor.
        correlation = rec.compute_correlation_projection(n_jobs=1)
    mask = getattr(rec, "rois", None)
    if mask is None or not np.any(np.asarray(mask) < 0):
        mask = np.zeros_like(correlation, dtype=int)
    return roi_figure(mask=mask, projection=correlation, width=width,
                      title=f"{rec.name} correlation", cmap=CORRELATION_CMAP)


def _labels(rec, state, width):
    from pygor.tui.imaging import roi_figure

    return roi_figure(rec, width=width, labels=True, title=f"{rec.name} numbered")


def _traces(rec, state, width):
    return rec.plot_traces()


def _averages(rec, state, width):
    return rec.plot_averages()


def _strf_space(rec, state, width):
    return rec.plot_strfs_space()


def _strf_roi(rec, state, width):
    return rec.plot_strfs_space(roi=state.roi)


def _chromatic(rec, state, width):
    return rec.plot_chromatic_overview(roi=state.roi)


def _has_rois(rec):
    return bool(getattr(rec, "num_rois", 0))


def _has_strfs(rec):
    return _is_array(getattr(rec, "strfs", None), ndim=4)


VIEWS = (
    View("projection", "mean + ROIs", _projection),
    View("correlation", "correlation", _correlation),
    View("labels", "numbered ROIs", _labels, applies=_has_rois),
    View("traces", "traces", _traces, applies=_has_traces),
    View("averages", "averages", _averages,
         applies=lambda rec: _is_array(getattr(rec, "averages", None))),
    View("strf_space", "STRF space (all)", _strf_space, applies=_has_strfs),
    View("strf_roi", "STRF space (roi)", _strf_roi, applies=_has_strfs, per_roi=True),
    View("chromatic", "chromatic overview (roi)", _chromatic,
         applies=lambda rec: _has_strfs(rec) and hasattr(rec, "plot_chromatic_overview"),
         per_roi=True),
)

# pyplot keeps global state; two views rendering at once would draw into each
# other's figures.
_PLOT_LOCK = threading.Lock()


def in_process_parallelism():
    """joblib on threads for the duration.

    pygor's registration, correlation projection and segmentation fan out over
    loky worker processes by default. Under Textual, stdio is a redirector
    without a file descriptor and the workers fail with a bad-fd error, so
    while the interface owns the terminal everything stays in this process.
    """
    import joblib

    try:
        return joblib.parallel_config(backend="threading")
    except AttributeError:  # joblib < 1.3
        return joblib.parallel_backend("threading")


def render_view(view, rec, state, width, height, dpi=100):
    """A view as a PanelImage, sized to fit ``width`` x ``height`` px.

    Figures keep their own aspect and are scaled to the pane, because the pane
    scales the PNG the same way and a stretched receptive field is a lie.
    """
    from matplotlib.figure import Figure

    from pygor.review.rasterise import PanelImage, PanelKey, figure_to_png, png_size

    with _PLOT_LOCK, in_process_parallelism():
        import matplotlib.pyplot as plt

        plt.close("all")
        result = view.render(rec, state, width)
        if isinstance(result, tuple):
            result = result[0]
        figure = result if isinstance(result, Figure) else plt.gcf()
        fig_w, fig_h = figure.get_size_inches()
        scale = min(width / (fig_w * dpi), height / (fig_h * dpi))
        out_w, out_h = max(int(fig_w * dpi * scale), 1), max(int(fig_h * dpi * scale), 1)
        png = figure_to_png(figure, width=out_w, height=out_h, dpi=dpi)
        plt.close("all")
    actual = png_size(png)
    key = PanelKey(panel=f"view:{view.key}", fov_uid=getattr(rec, "name", ""),
                   condition="", role="", roi=state.roi if view.per_roi else None,
                   channel=-1, width=out_w, height=out_h, dpi=dpi, params_hash="",
                   sources=())
    return PanelImage(png=png, width=actual[0] or out_w, height=actual[1] or out_h,
                      key=key, meta={"panel": f"view:{view.key}"})


# -- state ------------------------------------------------------------------


@dataclasses.dataclass
class WorkbenchState:
    """What the screen knows beyond the recording itself."""

    path: pathlib.Path
    recording: object = None
    roi: int = 0
    trigger_mode: int = 1
    n_colours: int = 1
    noise_array: str = ""
    baseline_dur: float = 10.0
    dirty: bool = False

    def sync_from_recording(self):
        rec = self.recording
        self.trigger_mode = int(getattr(rec, "trigger_mode", 1) or 1)
        n_colours = getattr(rec, "n_colours", None)
        if n_colours:
            self.n_colours = int(n_colours)

    def values(self) -> dict:
        """The parameter table's source: config plus the workbench section."""
        from pygor.tui.standalone import effective_values

        values = effective_values(self.recording)
        values["workbench"] = {
            "trigger_mode": self.trigger_mode,
            "n_colours": self.n_colours,
            "noise_array": self.noise_array,
            "baseline_dur": self.baseline_dur,
        }
        return values


def apply_overrides(state, overrides: dict) -> list[str]:
    """Write table edits into the recording (or the state) and say what changed."""
    from pygor.core.gui.param_values import flatten

    applied = []
    for path, value in flatten(overrides):
        if path.startswith("workbench."):
            setattr(state, path.split(".", 1)[1], value)
        else:
            try:
                state.recording.params[path] = value
            except KeyError:
                # A key today's pygor has and the saved object does not: create it.
                node = state.recording.params._defaults
                *parents, leaf = path.split(".")
                for part in parents:
                    node = node.setdefault(part, {})
                node[leaf] = value
        applied.append(f"{path} = {value!r}")
    if applied:
        state.dirty = True
    return applied


def apply_recipe(state, toml_path) -> list[str]:
    """Overlay a recipe TOML on the recording's parameters, like ``config=``."""
    import tomllib

    with open(toml_path, "rb") as handle:
        recipe = tomllib.load(handle)
    return apply_overrides(state, recipe)


# -- screens ----------------------------------------------------------------


def build_app(state, caps, log_lines=None):
    """Construct the Textual app; imported lazily so ``--help`` needs no Textual."""
    from textual import on, work
    from textual.app import App, ComposeResult
    from textual.binding import Binding
    from textual.command import Hit, Hits, Provider
    from textual.containers import Horizontal, Vertical
    from textual.screen import ModalScreen, Screen
    from textual.widgets import (
        Footer,
        Header,
        Input,
        Label,
        ListItem,
        ListView,
        OptionList,
        RichLog,
        Static,
    )
    from textual.widgets.option_list import Option

    from pygor.tui.app import PanelView, embed_shell
    from pygor.tui.imaging import clear_terminal_images, show, show_rois
    from pygor.tui.params_table import ParamTable
    from pygor.tui.reprocess_screen import ConfirmSave, segmentation_gating

    GLYPH_DONE, GLYPH_TODO, GLYPH_NA = "✓", "·", "–"

    class PathPrompt(ModalScreen):
        BINDINGS = [Binding("escape", "cancel", "cancel")]

        def __init__(self, title, value=""):
            super().__init__()
            self.title_text = title
            self.value = value

        def compose(self) -> ComposeResult:
            yield Vertical(Static(f"[b]{self.title_text}[/b]  (enter to accept, esc to cancel)"),
                           Input(value=self.value, id="path"), id="value-box")

        def on_mount(self):
            self.query_one("#path", Input).focus()

        @on(Input.Submitted)
        def submit(self, event):
            self.dismiss(event.value.strip() or None)

        def action_cancel(self):
            self.dismiss(None)

    class Menu(ModalScreen):
        """The menu bar's stand-in: one list, keyed by first letter."""

        BINDINGS = [Binding("escape", "cancel", "cancel")]
        ITEMS = (
            ("open", "Open recording…"),
            ("recipe", "Apply recipe TOML…"),
            ("run_all", "Run all remaining steps"),
            ("save", "Save recording (.recording.h5)"),
            ("repl", "IPython with rec bound"),
            ("help", "Key bindings"),
            ("quit", "Quit"),
        )

        def compose(self) -> ComposeResult:
            yield Vertical(Static("[b]pygor workbench[/b]"),
                           OptionList(*[Option(label, id=key) for key, label in self.ITEMS],
                                      id="menu"),
                           id="menu-box")

        def on_mount(self):
            self.query_one("#menu", OptionList).focus()

        def on_option_list_option_selected(self, event):
            self.dismiss(event.option.id)

        def action_cancel(self):
            self.dismiss(None)

    class WorkbenchCommands(Provider):
        """Command palette entries: every step and every view, by name."""

        async def search(self, query: str) -> Hits:
            matcher = self.matcher(query)
            screen = self.app.screen
            if not isinstance(screen, WorkbenchScreen):
                return
            rec = state.recording
            for step in STEPS:
                if not step.applies(rec):
                    continue
                text = f"run: {step.label}"
                score = matcher.match(text)
                if score > 0:
                    yield Hit(score, matcher.highlight(text),
                              partial(screen.run_step, step), help=step.hint)
            for view in VIEWS:
                if not view.applies(rec):
                    continue
                text = f"view: {view.label}"
                score = matcher.match(text)
                if score > 0:
                    yield Hit(score, matcher.highlight(text), partial(screen.show_view, view))
            for key, label in Menu.ITEMS:
                score = matcher.match(label)
                if score > 0:
                    yield Hit(score, matcher.highlight(label), partial(screen.menu_action, key))

    class WorkbenchScreen(Screen):
        BINDINGS = [
            Binding("m", "menu", "menu"),
            Binding("r", "run", "run step"),
            Binding("R", "run_all", "run all"),
            Binding("v", "next_view", "view"),
            # Priority, or the focused list scrolls on PageDown instead.
            Binding("right_square_bracket", "next_roi", "roi +", show=False, priority=True),
            Binding("left_square_bracket", "prev_roi", "roi -", show=False, priority=True),
            Binding("pagedown", "next_roi", "roi +", priority=True),
            Binding("pageup", "prev_roi", "roi -", priority=True),
            Binding("ctrl+s", "save", "save"),
            Binding("o", "repl", "ipython"),
            Binding("question_mark", "help", "help"),
            # app.quit: the action lives on the App, and a Screen binding does
            # not fall back to it.
            Binding("q", "app.quit", "quit"),
        ]

        def __init__(self):
            super().__init__()
            self.view_index = 0
            self.running = False

        # -- layout ---------------------------------------------------------

        def compose(self) -> ComposeResult:
            values = state.values()
            choices, gates = segmentation_gating(values)
            yield Header()
            yield Horizontal(
                Vertical(
                    Static("", id="recording"),
                    Label("steps", classes="section"),
                    ListView(*[ListItem(Label(s.label), name=s.key) for s in STEPS], id="steps"),
                    Label("views", classes="section"),
                    ListView(*[ListItem(Label(v.label), name=v.key) for v in VIEWS], id="views"),
                    id="sidebar",
                ),
                Vertical(
                    PanelView(caps, id="plot"),
                    RichLog(id="log", wrap=True, markup=False, highlight=False),
                    id="centre",
                ),
                Vertical(
                    Label("parameters", classes="section"),
                    ParamTable(values, sections=STEPS[0].sections, choices=choices,
                               gates=gates, id="params"),
                    id="right",
                ),
            )
            yield Static("", id="status")
            yield Footer()

        def on_mount(self):
            self.sub_title = state.path.name
            log = self.query_one("#log", RichLog)
            for line in log_lines or ():
                log.write(line)
            self.rebuild_steps()
            self.rebuild_views()
            self.query_one("#steps", ListView).focus()
            # After the first layout: the table needs its columns and the plot
            # pane its real size before either is asked to draw.
            self.call_after_refresh(self._initial_draw)

        def _initial_draw(self):
            self.select_step(self.first_undone())
            self.show_view(self.current_view())
            self.refresh_status()

        # -- lists ----------------------------------------------------------

        # The lists are composed once and repainted in place: ListView.clear()
        # and append() both take effect asynchronously, so a rebuild between
        # them either duplicates or queries children that are not there yet.
        def rebuild_steps(self):
            rec = state.recording
            steps = self.query_one("#steps", ListView)
            for item, step in zip(steps.children, STEPS):
                if not step.applies(rec):
                    glyph, classes = GLYPH_NA, "na"
                elif step.done(rec):
                    glyph, classes = GLYPH_DONE, "done"
                else:
                    glyph, classes = GLYPH_TODO, "todo"
                item.query_one(Label).update(f"{glyph} {step.label}")
                item.set_classes(classes)
            self.query_one("#recording", Static).update(self.describe_recording())

        def rebuild_views(self):
            rec = state.recording
            views = self.query_one("#views", ListView)
            for item, view in zip(views.children, VIEWS):
                item.set_classes("" if view.applies(rec) else "na")

        def describe_recording(self) -> str:
            rec = state.recording
            images = getattr(rec, "images", None)
            shape = "x".join(str(s) for s in images.shape) if images is not None else "?"
            dirty = "  [yellow]unsaved[/]" if state.dirty else ""
            return (f"[b]{rec.name}[/b]{dirty}\n{type(rec).__name__}  {shape}\n"
                    f"{getattr(rec, 'num_rois', 0)} ROIs  ·  trig {state.trigger_mode}  ·  "
                    f"{state.n_colours} colour(s)")

        def first_undone(self) -> int:
            rec = state.recording
            for index, step in enumerate(STEPS):
                if step.applies(rec) and not step.done(rec):
                    return index
            return 0

        def current_step(self) -> Step:
            return STEPS[self.query_one("#steps", ListView).index or 0]

        def current_view(self) -> View:
            return VIEWS[self.view_index % len(VIEWS)]

        def select_step(self, index):
            steps = self.query_one("#steps", ListView)
            steps.index = index
            self.show_step_params(STEPS[index])

        def show_step_params(self, step):
            table = self.query_one("#params", ParamTable)
            table.sections = step.sections
            table.reload()
            self.refresh_status(step.hint)

        @on(ListView.Highlighted, "#steps")
        def step_highlighted(self, event):
            if event.item is not None:
                step = next(s for s in STEPS if s.key == event.item.name)
                self.show_step_params(step)

        @on(ListView.Selected, "#steps")
        def step_selected(self, event):
            self.action_run()

        @on(ListView.Highlighted, "#views")
        def view_highlighted(self, event):
            if event.item is not None:
                self.view_index = next(i for i, v in enumerate(VIEWS) if v.key == event.item.name)

        @on(ListView.Selected, "#views")
        def view_selected(self, event):
            self.show_view(self.current_view())

        # -- status ---------------------------------------------------------

        def refresh_status(self, text=""):
            table = self.query_one("#params", ParamTable)
            edits = len(table.overrides)
            view = self.current_view()
            roi = f"  ·  roi {state.roi}" if view.per_roi else ""
            self.query_one("#status", Static).update(
                f"step: {self.current_step().label}  ·  view: {view.label}{roi}  ·  "
                f"{edits} pending edit{'s' if edits != 1 else ''}"
                + (f"  ·  {text}" if text else "")
            )

        # Not ``log``: that name is Textual's own logger on every widget.
        def write_log(self, text):
            self.query_one("#log", RichLog).write(text)

        # -- running steps --------------------------------------------------

        def action_run(self):
            self.run_step(self.current_step())

        def run_step(self, step):
            if self.running:
                self.app.notify("a step is already running", severity="warning")
                return
            if not step.applies(state.recording):
                self.app.notify(f"{step.label} does not apply to {type(state.recording).__name__}",
                                severity="warning")
                return
            self.commit_edits()
            self.running = True
            self.refresh_status("running…")
            self.write_log(f"── {step.label} ──")
            self.run_steps_thread([step])

        def action_run_all(self):
            rec = state.recording
            todo = [s for s in STEPS if s.applies(rec) and not s.done(rec)]
            if not todo:
                self.app.notify("nothing left to run")
                return
            if self.running:
                self.app.notify("a step is already running", severity="warning")
                return
            self.commit_edits()
            self.running = True
            self.write_log(f"── run all: {', '.join(s.label for s in todo)} ──")
            self.run_steps_thread(todo)

        def commit_edits(self):
            table = self.query_one("#params", ParamTable)
            applied = apply_overrides(state, table.nested_overrides())
            for line in applied:
                self.write_log(f"set {line}")
            table.overrides.clear()
            table.source = state.values()
            table.reload()

        @work(thread=True, exclusive=True, group="steps")
        def run_steps_thread(self, steps):
            writer = _LogWriter(lambda text: self.app.call_from_thread(self.write_log,text))
            for step in steps:
                if len(steps) > 1:
                    self.app.call_from_thread(self.write_log,f"── {step.label} ──")
                try:
                    with contextlib.redirect_stdout(writer), contextlib.redirect_stderr(writer), \
                            in_process_parallelism():
                        summary = step.run(state.recording, state)
                except Exception as error:  # shown in the log, not swallowed
                    writer.flush()
                    self.app.call_from_thread(self.step_finished, step,
                                              f"{type(error).__name__}: {error}")
                    return
                writer.flush()
                self.app.call_from_thread(self.step_finished, step, "", summary)
            self.app.call_from_thread(self.all_finished)

        def step_finished(self, step, error, summary=""):
            if error:
                self.running = False
                self.write_log(f"✗ {step.label}: {error}")
                self.app.notify(error, severity="error", timeout=12)
                self.refresh_status(f"failed: {error}")
                return
            state.dirty = True
            self.write_log(f"✓ {step.label}: {summary}")
            self.rebuild_steps()
            self.rebuild_views()
            table = self.query_one("#params", ParamTable)
            table.source = state.values()
            table.reload()

        def all_finished(self):
            self.running = False
            self.select_step(self.first_undone())
            self.show_view(self.current_view())
            self.refresh_status("done")

        # -- views ----------------------------------------------------------

        def action_next_view(self):
            rec = state.recording
            for _ in range(len(VIEWS)):
                self.view_index = (self.view_index + 1) % len(VIEWS)
                if VIEWS[self.view_index].applies(rec):
                    break
            self.query_one("#views", ListView).index = self.view_index
            self.show_view(self.current_view())

        def show_view(self, view):
            self.view_index = next(i for i, v in enumerate(VIEWS) if v.key == view.key)
            pane = self.query_one("#plot", PanelView)
            if not view.applies(state.recording):
                pane.show_message(f"{view.label}: not available yet")
                self.refresh_status()
                return
            pane.show_message(f"rendering {view.label}…")
            self.refresh_status()
            self.render_thread(view, pane)

        @work(thread=True, exclusive=True, group="render")
        def render_thread(self, view, pane):
            width, height = pane.size_px()
            try:
                image = render_view(view, state.recording, state, width, height)
            except Exception as error:
                self.app.call_from_thread(pane.show_message,
                                          f"{view.label} failed: {type(error).__name__}: {error}")
                return
            self.app.call_from_thread(pane.show, image)

        def action_next_roi(self):
            n = getattr(state.recording, "num_rois", 0) or 1
            state.roi = min(state.roi + 1, n - 1)
            if self.current_view().per_roi:
                self.show_view(self.current_view())
            self.refresh_status()

        def action_prev_roi(self):
            state.roi = max(state.roi - 1, 0)
            if self.current_view().per_roi:
                self.show_view(self.current_view())
            self.refresh_status()

        # -- menu -----------------------------------------------------------

        def action_menu(self):
            self.app.push_screen(Menu(), self.menu_action)

        def menu_action(self, key):
            if key is None:
                return
            {
                "open": self.action_open,
                "recipe": self.action_recipe,
                "run_all": self.action_run_all,
                "save": self.action_save,
                "repl": self.action_repl,
                "help": self.action_help,
                "quit": self.app.exit,
            }[key]()

        def action_open(self):
            self.app.push_screen(PathPrompt("open recording (.recording.h5, .h5, .smp)",
                                            str(state.path)), self._open)

        def _open(self, path):
            if not path:
                return
            self.write_log(f"loading {path}…")
            self.load_thread(path)

        @work(thread=True, exclusive=True, group="steps")
        def load_thread(self, path):
            from pygor.tui.standalone import load_recording

            try:
                recording = load_recording(path, state.n_colours if state.n_colours > 1 else None)
            except Exception as error:
                self.app.call_from_thread(self.app.notify, f"load failed: {error}",
                                          severity="error", timeout=12)
                return
            self.app.call_from_thread(self._loaded, pathlib.Path(path), recording)

        def _loaded(self, path, recording):
            state.path = path
            state.recording = recording
            state.roi = 0
            state.dirty = False
            state.sync_from_recording()
            self.sub_title = path.name
            self.write_log(f"loaded {recording.name}")
            self.rebuild_steps()
            self.rebuild_views()
            table = self.query_one("#params", ParamTable)
            table.overrides.clear()
            table.source = state.values()
            self.select_step(self.first_undone())
            self.show_view(self.current_view())

        def action_recipe(self):
            self.app.push_screen(PathPrompt("recipe TOML to overlay on the parameters"),
                                 self._recipe)

        def _recipe(self, path):
            if not path:
                return
            try:
                applied = apply_recipe(state, path)
            except Exception as error:
                self.app.notify(f"recipe failed: {error}", severity="error", timeout=12)
                return
            self.write_log(f"recipe {path}: {len(applied)} value(s)")
            table = self.query_one("#params", ParamTable)
            table.source = state.values()
            table.reload()
            self.refresh_status(f"recipe applied ({len(applied)} values)")

        def action_save(self):
            target = save_target(state.path)
            text = (f"Write {target.name}?" + ("  The existing file is kept as .preworkbench."
                                                 if target.exists() else ""))
            self.app.push_screen(ConfirmSave(text), partial(self._save, target))

        def _save(self, target, yes):
            if not yes:
                return
            try:
                save_recording(state.recording, target)
            except Exception as error:
                self.app.notify(f"save failed: {error}", severity="error", timeout=12)
                return
            state.dirty = False
            self.write_log(f"saved {target}")
            self.app.notify(f"saved {target.name}")
            self.rebuild_steps()

        def action_repl(self):
            namespace = {
                "rec": state.recording,
                "state": state,
                "app": self.app,
                "show": partial(show, caps=caps),
                "show_rois": partial(show_rois, caps=caps),
                "np": np,
            }
            header = "\n".join([
                "pygor workbench.  Ctrl-D returns.",
                "  rec      the recording          state   workbench state (roi, trigger_mode…)",
                "  show(x)  draw a figure / (fig, ax) / 2-D array inline",
                "  show_rois(rec, labels=True)",
            ])
            with self.app.suspend():
                clear_terminal_images()
                embed_shell(header, namespace)
            self.rebuild_steps()
            self.rebuild_views()

        def action_help(self):
            self.app.action_show_help_panel()

    class WorkbenchApp(App):
        TITLE = "pygor workbench"
        COMMANDS = App.COMMANDS | {WorkbenchCommands}
        CSS = """
        Screen { background: $surface; }
        #sidebar { width: 30; padding: 0 1; border-right: solid $panel; }
        #recording { height: auto; padding: 0 0 1 0; color: $text-muted; }
        .section { color: $text-muted; text-style: bold; padding: 0 0 0 0; }
        #steps { height: auto; max-height: 10; }
        #views { height: auto; max-height: 12; }
        ListItem.done Label { color: $success; }
        ListItem.todo Label { color: $text; }
        ListItem.na Label { color: $text-muted; }
        #centre { width: 3fr; padding: 0 1; }
        #plot { height: 3fr; }
        #log { height: 1fr; min-height: 5; border-top: solid $panel; }
        #right { width: 2fr; }
        #params { height: 1fr; }
        PanelView { height: 1fr; overflow: hidden; align: center middle; }
        #status { height: 1; padding: 0 1; background: $panel; }
        #value-box, #confirm-box, #menu-box { padding: 1 2; width: 60%; height: auto; background: $panel; }
        PathPrompt, Menu, ConfirmSave { align: center middle; }
        .error { color: $error; }
        .dim { color: $text-muted; }
        """
        BINDINGS = [Binding("ctrl+l", "unscramble", "redraw", show=False)]

        def on_mount(self):
            self.push_screen(WorkbenchScreen())

        def action_unscramble(self):
            clear_terminal_images()
            self.refresh(repaint=True, layout=True)
            screen = self.screen
            if isinstance(screen, WorkbenchScreen):
                screen.show_view(screen.current_view())

        def on_unmount(self):
            clear_terminal_images()

    return WorkbenchApp()


class _LogWriter(io.TextIOBase):
    """A file-like that hands complete lines to a callback."""

    def __init__(self, emit):
        super().__init__()
        self.emit = emit
        self.buffer = ""

    def writable(self):
        return True

    def write(self, text):
        self.buffer += text
        while "\n" in self.buffer:
            line, self.buffer = self.buffer.split("\n", 1)
            if line.strip():
                self.emit(line)
        return len(text)

    def flush(self):
        if self.buffer.strip():
            self.emit(self.buffer)
        self.buffer = ""


# -- saving -----------------------------------------------------------------


def save_target(source: pathlib.Path) -> pathlib.Path:
    if source.name.endswith(".recording.h5"):
        return source
    return source.with_name(source.stem + ".recording.h5")


def save_recording(recording, target: pathlib.Path):
    if target.exists():
        shutil.copy2(target, target.with_suffix(target.suffix + ".preworkbench"))
    recording.save_object(target.with_name(target.name.removesuffix(".recording.h5")),
                          overwrite=True)


# -- entry point ------------------------------------------------------------


def main(argv=None) -> int:
    os.environ["MPLBACKEND"] = "Agg"
    parser = argparse.ArgumentParser(prog="pygor-workbench",
                                     description=__doc__.splitlines()[0])
    parser.add_argument("recording", help=".recording.h5, IGOR .h5, or ScanM .smp/.smh")
    parser.add_argument("--n-colours", type=int)
    parser.add_argument("--config", help="recipe TOML applied on top of the recording's parameters")
    parser.add_argument("--noise", help="noise array (.npy or single-dataset .h5) for the STA step")
    parser.add_argument("--trigger-mode", type=int)
    parser.add_argument("--graphics", default="auto",
                        choices=("auto", "tgp", "sixel", "halfcell", "unicode", "none"))
    args = parser.parse_args(argv)

    from pygor.tui.capabilities import probe

    caps = probe(args.graphics)

    try:
        import textual  # noqa: F401
    except ImportError as error:
        raise SystemExit(
            "pygor-workbench needs the [tui] extra:  uv pip install 'pygor[tui]'\n" f"({error})"
        ) from error

    from pygor.tui.standalone import load_recording

    path = pathlib.Path(args.recording)
    state = WorkbenchState(path=path)
    state.recording = load_recording(path, args.n_colours)
    state.sync_from_recording()
    log_lines = [f"loaded {state.recording.name} ({type(state.recording).__name__})"]
    if args.n_colours:
        state.n_colours = args.n_colours
    if args.trigger_mode is not None:
        state.trigger_mode = args.trigger_mode
    if args.noise:
        state.noise_array = args.noise
    if args.config:
        applied = apply_recipe(state, args.config)
        state.dirty = False
        log_lines.append(f"recipe {args.config}: {len(applied)} value(s)")

    build_app(state, caps, log_lines).run()
    return 0


if __name__ == "__main__":
    sys.exit(main())
