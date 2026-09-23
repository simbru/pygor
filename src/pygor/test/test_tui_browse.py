"""Tests for the reader and the browse/inspect screens.

The reader is what three entry points agree on, so its classification is
tested directly rather than through a screen. The screens are tested through
Textual's headless driver, which is the only way to find out whether a worker
thread's result reaches the widget it was rendered for.
"""

from __future__ import annotations

import asyncio
import pathlib
import re

import numpy as np
import pytest

from pygor.test.test_review_index import write_recording


class TestKinds:
    def test_saved_object_is_not_mistaken_for_an_igor_export(self):
        from pygor.tui import reader

        assert reader.kind("0_0_SWN_200.recording.h5") == "saved"
        assert reader.kind("0_0_SWN_200.h5") == "igor"

    def test_scanm_lists_the_data_half_only(self):
        from pygor.tui import reader

        assert reader.kind("scan.smp") == "scanm"
        assert reader.kind("scan.smh") is None

    def test_everything_else_is_not_a_recording(self):
        from pygor.tui import reader

        for name in ("notes.txt", "figure.png", "recipe.toml", "recording.h5.presegment"):
            assert reader.kind(name) is None
            assert not reader.is_recording(name)


class TestLoadDispatch:
    def test_saved_suffix_goes_to_load_object(self, monkeypatch, tmp_path):
        import pygor.load
        from pygor.tui import reader

        seen = {}
        monkeypatch.setattr(pygor.load.Core, "load_object",
                            classmethod(lambda cls, path: seen.setdefault("saved", path)))
        reader.load_recording(tmp_path / "a.recording.h5")
        assert seen["saved"].name == "a.recording.h5"

    def test_raw_file_goes_to_strf_with_n_colours(self, monkeypatch, tmp_path):
        import pygor.load
        from pygor.tui import reader

        seen = {}

        def fake(path, **kwargs):
            seen["args"] = (path, kwargs)

        monkeypatch.setattr(pygor.load, "STRF", fake)
        reader.load_recording(tmp_path / "a.smp", n_colours=4)
        assert seen["args"][1] == {"n_colours": 4}

    def test_n_colours_is_omitted_when_not_given(self, monkeypatch, tmp_path):
        """Passing n_colours=None would override the name-derived value."""
        import pygor.load
        from pygor.tui import reader

        seen = {}
        monkeypatch.setattr(pygor.load, "STRF",
                            lambda path, **kwargs: seen.setdefault("kwargs", kwargs))
        reader.load_recording(tmp_path / "a.h5")
        assert seen["kwargs"] == {}


class Fake:
    """The shape of a recording, with everything the summary reads."""

    name = "0_0_SWN_200"
    metadata = {"exp_date": "2024-01-01", "exp_time": "12:00:00"}
    images = np.zeros((5, 8, 8), dtype=np.float32)
    frame_hz = 10.0
    rois = np.full((8, 8), 1, dtype=np.int16)
    num_rois = 4
    triggertimes = np.arange(12, dtype=float)
    trigger_mode = 2
    is_registered = True
    ipl_depths = None
    averages = np.nan
    snippets = np.nan
    strfs = np.zeros((16, 6, 4, 4), dtype=np.float32)


def summary_of(recording) -> dict:
    from pygor.tui import reader

    return dict(reader.summarise(recording))


class TestSummarise:
    def test_reports_shapes_and_counts(self):
        rows = summary_of(Fake())
        assert rows["class"] == "Fake"
        assert rows["images"].startswith("5x8x8")
        assert rows["frame rate"] == "10.00 Hz"
        assert rows["duration"] == "0.5 s"
        assert rows["ROIs"] == "4 on 8x8"
        assert rows["triggers"] == "12"
        assert rows["trigger mode"] == "2"
        assert rows["STRFs"] == "16x6x4x4"

    def test_nan_sentinels_read_as_not_computed(self):
        """averages/snippets default to np.nan on Core, not to None."""
        rows = summary_of(Fake())
        assert rows["averages"] == "not computed"
        assert rows["snippets"] == "not computed"

    def test_unsegmented_recording_says_so_instead_of_raising(self):
        class NoRois(Fake):
            rois = None
            num_rois = 0

        assert summary_of(NoRois())["ROIs"] == "none segmented"

    def test_a_property_that_raises_does_not_lose_the_other_rows(self):
        class Broken(Fake):
            @property
            def is_registered(self):
                raise RuntimeError("no params")

        rows = summary_of(Broken())
        assert rows["registered"] == "unknown"
        assert rows["ROIs"] == "4 on 8x8"


class TestTreeFilter:
    @pytest.fixture
    def tree(self, tmp_path):
        pytest.importorskip("textual")
        from pygor.tui.browse_screen import RecordingTree

        # Not instantiated: a DirectoryTree outside a running app schedules a
        # watcher coroutine nothing ever awaits. The filter reads only its
        # argument, so an uninitialised instance is the whole subject here.
        return RecordingTree.__new__(RecordingTree)

    def test_keeps_recordings_and_directories_only(self, tree, tmp_path):
        names = ["a.recording.h5", "b.h5", "c.smp", "c.smh", "notes.txt",
                 "d.recording.h5.presegment"]
        for name in names:
            (tmp_path / name).touch()
        (tmp_path / "2024-01-01").mkdir()
        (tmp_path / "__pycache__").mkdir()
        (tmp_path / ".hidden").mkdir()

        kept = [p.name for p in tree.filter_paths(sorted(tmp_path.iterdir()))]
        assert kept == ["2024-01-01", "a.recording.h5", "b.h5", "c.smp"]

    def test_directories_sort_before_files(self, tree, tmp_path):
        (tmp_path / "a.recording.h5").touch()
        (tmp_path / "zzz").mkdir()
        kept = [p.name for p in tree.filter_paths(sorted(tmp_path.iterdir()))]
        assert kept == ["zzz", "a.recording.h5"]


def run_app(app, steps):
    """Drive ``app`` headless; ``steps`` is an async function of the pilot."""

    async def body():
        async with app.run_test(size=(160, 45)) as pilot:
            await pilot.pause()
            await steps(pilot)

    asyncio.run(body())


async def settle(pilot, app, predicate, limit=200):
    """Wait for a worker thread's result to land, or give up and let the assert fail."""
    for _ in range(limit):
        await pilot.pause()
        if predicate():
            return
        await asyncio.sleep(0.05)


@pytest.fixture
def one_recording(tmp_path):
    pytest.importorskip("textual")
    return write_recording(tmp_path, stem="0_0_SWN_200")


def centre(widget):
    """An offset in the middle of a widget.

    The far corner is not usable: the panel can extend past the visible screen
    region and the pilot refuses to click outside it.
    """
    return (widget.size.width // 2, widget.size.height // 2)


async def live_widget(pilot, app):
    """The image widget as it is now.

    Every redraw re-mounts it, so a reference held across one is detached and
    reports a zero region -- clicking "it" then lands at screen (0, 0), which
    is the header icon, which opens the command palette. The app re-queries on
    every event for the same reason.
    """
    await settle(pilot, app, lambda: app.screen.image_widget() is not None)
    return app.screen.image_widget()


def text_of(screen, selector):
    from textual.widgets import Static

    return str(screen.query_one(selector, Static).content)


class TestScreens:
    def _app(self, target):
        from pygor.tui.capabilities import probe
        from pygor.tui.standalone import build_app

        return build_app(target, probe("none"))

    def test_browser_opens_on_a_directory_and_inspects_a_file(self, one_recording):
        from pygor.tui.browse_screen import BrowseScreen, InspectScreen

        app = self._app(one_recording.parent)
        seen = {}

        async def steps(pilot):
            assert isinstance(app.screen, BrowseScreen)
            await pilot.press("down")
            await pilot.press("enter")
            await settle(pilot, app, lambda: isinstance(app.screen, InspectScreen)
                         and app.screen.recording is not None)
            seen["meta"] = text_of(app.screen, "#meta")
            await pilot.press("escape")
            await pilot.pause()
            seen["back"] = type(app.screen).__name__

        run_app(app, steps)
        assert "STRF" in seen["meta"]
        assert "4 on 8x8" in seen["meta"]
        assert seen["back"] == "BrowseScreen"

    def test_a_file_target_lands_straight_on_the_inspector(self, one_recording):
        from pygor.tui.browse_screen import InspectScreen

        app = self._app(one_recording)
        seen = {}

        async def steps(pilot):
            await settle(pilot, app, lambda: isinstance(app.screen, InspectScreen)
                         and app.screen.recording is not None)
            seen["meta"] = text_of(app.screen, "#meta")

        run_app(app, steps)
        assert "STRF" in seen["meta"]

    def test_r_reaches_the_reprocess_screen_and_comes_back(self, one_recording):
        from pygor.tui.browse_screen import InspectScreen

        app = self._app(one_recording)
        seen = {}

        async def steps(pilot):
            await settle(pilot, app, lambda: isinstance(app.screen, InspectScreen)
                         and app.screen.recording is not None)
            await pilot.press("r")
            await settle(pilot, app,
                         lambda: type(app.screen).__name__ == "ReprocessScreen")
            seen["on"] = type(app.screen).__name__
            await pilot.press("escape")
            await pilot.pause()
            seen["back"] = type(app.screen).__name__

        run_app(app, steps)
        assert seen["on"] == "ReprocessScreen"
        assert seen["back"] == "InspectScreen"

    def test_scrubbing_switches_to_the_stack_view_and_moves_the_frame(self, one_recording):
        from pygor.tui.browse_screen import InspectScreen

        app = self._app(one_recording)
        seen = {}

        async def steps(pilot):
            await settle(pilot, app, lambda: isinstance(app.screen, InspectScreen)
                         and app.screen.recording is not None)
            assert app.screen.view != "stack"
            await pilot.press("full_stop")
            await pilot.press("full_stop")
            await settle(pilot, app, lambda: app.screen.frame == 2)
            seen["view"] = app.screen.view
            seen["frame"] = app.screen.frame
            # The write_recording fixture has five frames, so this clamps.
            for _ in range(10):
                await pilot.press("greater_than_sign")
            await settle(pilot, app, lambda: app.screen.frame == 4)
            seen["clamped"] = app.screen.frame
            for _ in range(10):
                await pilot.press("less_than_sign")
            await settle(pilot, app, lambda: app.screen.frame == 0)
            seen["floor"] = app.screen.frame

        run_app(app, steps)
        assert seen["view"] == "stack"
        assert seen["frame"] == 2
        assert seen["clamped"] == 4
        assert seen["floor"] == 0

    def test_arrow_keys_move_the_probe_and_the_trace_follows(self, one_recording):
        from pygor.tui.browse_screen import InspectScreen

        app = self._app(one_recording)
        seen = {}

        async def steps(pilot):
            await settle(pilot, app, lambda: isinstance(app.screen, InspectScreen)
                         and app.screen.recording is not None)
            seen["initial"] = app.screen.probe
            await settle(pilot, app, lambda: bool(app.screen.query("#trace Static")))
            await pilot.press("left")
            await settle(pilot, app, lambda: app.screen.probe != seen["initial"])
            seen["moved"] = app.screen.probe
            statics = app.screen.query("#trace Static")
            seen["trace"] = str(statics[0].content) if statics else ""

        run_app(app, steps)
        # The fixture's images are 8x8, so the probe starts in the middle.
        assert seen["initial"] == (4, 4)
        assert seen["moved"] == (4, 3)
        assert "pixel (4, 3)" in seen["trace"]

    def test_the_probe_stops_at_the_edge_of_the_field(self, one_recording):
        from pygor.tui.browse_screen import InspectScreen

        app = self._app(one_recording)
        seen = {}

        async def steps(pilot):
            await settle(pilot, app, lambda: isinstance(app.screen, InspectScreen)
                         and app.screen.recording is not None)
            for _ in range(20):
                await pilot.press("left")
                await pilot.press("down")
            await settle(pilot, app, lambda: app.screen.probe == (0, 0))
            seen["corner"] = app.screen.probe

        run_app(app, steps)
        assert seen["corner"] == (0, 0)

    def test_t_swaps_the_trace_between_braille_and_a_panel(self, one_recording):
        from pygor.tui.browse_screen import InspectScreen

        app = self._app(one_recording)
        seen = {}

        async def steps(pilot):
            await settle(pilot, app, lambda: isinstance(app.screen, InspectScreen)
                         and app.screen.recording is not None)
            await settle(pilot, app, lambda: bool(app.screen.query("#trace Static")))
            seen["before"] = app.screen.trace_mode
            await pilot.press("t")
            await settle(pilot, app, lambda: app.screen.trace_mode == "figure")
            seen["after"] = app.screen.trace_mode
            await pilot.press("t")
            await settle(pilot, app, lambda: app.screen.trace_mode == "braille")
            seen["back"] = app.screen.trace_mode

        run_app(app, steps)
        assert (seen["before"], seen["after"], seen["back"]) == (
            "braille", "figure", "braille")

    def test_an_unreadable_file_reports_itself_instead_of_crashing(self, tmp_path):
        pytest.importorskip("textual")
        from pygor.tui.browse_screen import InspectScreen

        broken = tmp_path / "broken.recording.h5"
        broken.write_bytes(b"not an h5 file")
        app = self._app(broken)
        seen = {}

        async def steps(pilot):
            await settle(pilot, app, lambda: isinstance(app.screen, InspectScreen)
                         and "could not open" in text_of(app.screen, "#meta"))
            seen["meta"] = text_of(app.screen, "#meta")

        run_app(app, steps)
        assert "could not open" in seen["meta"]


class TestUnsegmentedPreview:
    def test_a_recording_with_no_rois_still_draws_its_projection(self):
        """A raw ScanM file has no ROIs yet, and the inspector opens those."""
        import os

        os.environ["MPLBACKEND"] = "Agg"
        from pygor.tui.imaging import preview_image

        image = preview_image(None, np.zeros((8, 8)), "rois", 400, 300, name="raw")
        assert image.png.startswith(b"\x89PNG")


class TestPixelAt:
    def test_top_cell_is_the_last_row_because_views_draw_origin_lower(self):
        from pygor.tui.probe import pixel_at

        assert pixel_at(0, 0, 10, 10, (64, 128)) == (63, 0)
        # The bottom-right cell covers the last tenth of each axis, so it is
        # the start of that band, not the final pixel.
        assert pixel_at(9, 9, 10, 10, (64, 128)) == (6, 115)

    def test_off_the_widget_is_no_pixel(self):
        from pygor.tui.probe import pixel_at

        assert pixel_at(-1, 4, 10, 10, (64, 128)) is None
        assert pixel_at(4, 10, 10, 10, (64, 128)) is None

    def test_a_widget_with_no_size_yet_is_no_pixel(self):
        from pygor.tui.probe import pixel_at

        assert pixel_at(0, 0, 0, 0, (64, 128)) is None


class TestTraceSource:
    def _recording(self, rois, traces=None):
        class Rec:
            images = np.arange(5 * 4 * 4, dtype=np.float32).reshape(5, 4, 4)
            frame_hz = 10.0

        rec = Rec()
        rec.rois = rois
        rec.traces_znorm = traces
        rec.traces_raw = None
        return rec

    def test_a_pixel_inside_an_roi_gives_that_rois_trace(self):
        from pygor.tui.probe import at_pixel

        rois = np.ones((4, 4), dtype=np.int16)
        rois[1, 1] = -1
        rois[2, 2] = -2
        traces = np.array([[1.0, 2, 3, 4, 5], [9.0, 8, 7, 6, 5]])
        trace = at_pixel(self._recording(rois, traces), 2, 2)
        assert trace.source == "roi"
        assert "ROI -2" in trace.label
        assert list(trace.values) == [9, 8, 7, 6, 5]

    def test_a_gap_in_the_roi_ids_does_not_shift_the_trace(self):
        """extract_traces writes one row per present id, so -4 is row 2 of 3."""
        from pygor.tui.probe import at_pixel

        rois = np.ones((4, 4), dtype=np.int16)
        rois[0, 0], rois[1, 1], rois[2, 2] = -1, -2, -4
        traces = np.array([[1.0] * 5, [2.0] * 5, [3.0] * 5])
        trace = at_pixel(self._recording(rois, traces), 2, 2)
        assert "ROI -4" in trace.label
        assert list(trace.values) == [3.0] * 5

    def test_off_an_roi_falls_back_to_the_pixel_neighbourhood(self):
        from pygor.tui.probe import at_pixel

        rois = np.ones((4, 4), dtype=np.int16)
        rois[0, 0] = -1
        trace = at_pixel(self._recording(rois, np.zeros((1, 5))), 2, 2)
        assert trace.source == "pixel"
        assert "pixel (2, 2)" in trace.label
        assert len(trace.values) == 5

    def test_an_unsegmented_recording_gives_the_pixel(self):
        from pygor.tui.probe import at_pixel

        trace = at_pixel(self._recording(None), 1, 1)
        assert trace.source == "pixel"
        assert trace.seconds == 0.5

    def test_an_roi_with_no_extracted_traces_falls_back_rather_than_lying(self):
        from pygor.tui.probe import at_pixel

        rois = np.ones((4, 4), dtype=np.int16)
        rois[2, 2] = -1
        trace = at_pixel(self._recording(rois, None), 2, 2)
        assert trace.source == "pixel"


class TestSparkline:
    def test_plot_fills_the_requested_grid(self):
        from pygor.tui.sparkline import plot

        lines = plot(np.sin(np.linspace(0, 10, 300)), cols=20, rows=4)
        assert len(lines) == 4
        assert all(len(line) == 20 for line in lines)

    def test_every_character_is_braille(self):
        from pygor.tui.sparkline import BRAILLE_BASE, plot

        for line in plot(np.arange(50.0), cols=12, rows=3):
            assert all(BRAILLE_BASE <= ord(ch) < BRAILLE_BASE + 256 for ch in line)

    def test_a_flat_trace_does_not_divide_by_zero(self):
        from pygor.tui.sparkline import framed

        assert framed(np.full(20, 3.0), cols=10, rows=3)

    def test_an_empty_trace_renders_a_blank_pane(self):
        from pygor.tui.sparkline import BLANK, plot

        assert plot(np.zeros(0), cols=5, rows=2) == [BLANK * 5] * 2

    def test_nans_do_not_break_the_line(self):
        from pygor.tui.sparkline import framed

        values = np.arange(40.0)
        values[10:15] = np.nan
        assert framed(values, cols=20, rows=4)


class TestTriggers:
    def test_sparse_triggers_are_drawn_one_by_one(self):
        from pygor.tui.sparkline import BRAILLE_BASE, event_row

        row, resolved = event_row(np.arange(0, 200, 4.0), cols=100, x_max=200)
        assert resolved
        assert any(ord(ch) > BRAILLE_BASE for ch in row)

    def test_dense_triggers_become_a_density_bar(self):
        """15616 ticks across 200 dot columns is a solid rule, not a picture."""
        from pygor.tui.sparkline import BLOCKS, event_row

        row, resolved = event_row(np.arange(0, 3000, 0.2), cols=100, x_max=3000)
        assert not resolved
        assert set(row) <= set(BLOCKS)

    def test_a_dropout_leaves_a_hole_in_the_density_bar(self):
        from pygor.tui.sparkline import event_row

        times = np.arange(0, 3000, 0.2)
        times = times[(times < 1200) | (times > 1240)]
        row, _ = event_row(times, cols=100, x_max=3000)
        assert " " in row

    def test_no_triggers_is_a_blank_row_not_a_crash(self):
        from pygor.tui.sparkline import event_row

        row, resolved = event_row(np.zeros(0), cols=20, x_max=100)
        assert row == " " * 20
        assert resolved

    def test_triggers_past_the_end_of_the_trace_are_dropped(self):
        from pygor.tui.sparkline import event_row

        row, _ = event_row(np.array([-5.0, 50.0, 500.0]), cols=20, x_max=100)
        assert row.strip()

    def test_the_event_row_lines_up_with_the_plot(self):
        from pygor.tui.sparkline import framed

        text = framed(np.arange(100.0), cols=30, rows=4, x_max=100,
                      events=np.arange(0, 100, 10.0))
        lines = text.splitlines()
        bars = [line.index("│") for line in lines if "│" in line]
        assert len(set(bars)) == 1
        assert any("trig" in line for line in lines)

    def test_a_clean_train_says_no_gaps(self):
        from pygor.tui.probe import trigger_summary

        summary = trigger_summary(np.arange(0, 100, 0.2))
        assert "500 trig" in summary
        assert "0.200s" in summary
        assert "no gaps" in summary

    def test_a_dropped_trigger_is_reported_with_its_size(self):
        from pygor.tui.probe import trigger_summary

        times = np.arange(0, 100, 0.2)
        times = times[(times < 40) | (times > 45)]
        summary = trigger_summary(times)
        assert "1 gap" in summary
        assert "max 5." in summary

    def test_a_doubled_trigger_is_reported_separately_from_a_gap(self):
        from pygor.tui.probe import trigger_summary

        times = np.sort(np.append(np.arange(0, 100, 0.2), 20.05))
        summary = trigger_summary(times)
        assert "1 short" in summary

    def test_degenerate_trigger_counts_do_not_raise(self):
        from pygor.tui.probe import trigger_summary

        assert trigger_summary(np.zeros(0)) == "no triggers"
        assert "1 trigger" in trigger_summary(np.array([3.0]))

    def test_triggers_of_treats_an_empty_array_as_none(self):
        from pygor.tui.probe import triggers_of

        class Rec:
            triggertimes = np.zeros(0)

        assert triggers_of(Rec()) is None

        class WithSome:
            triggertimes = np.arange(5.0)

        assert triggers_of(WithSome()) is not None


class TestFramePreview:
    def test_a_frame_renders_with_a_crosshair(self):
        from pygor.tui.imaging import frame_preview

        images = np.random.default_rng(0).random((5, 8, 8)).astype(np.float32)
        panel = frame_preview(images, 2, 400, 400, (0.0, 1.0), marker=(3, 4))
        assert panel.png.startswith(b"\x89PNG")
        assert panel.meta["panel"] == "frame:2"

    def test_the_crosshair_lands_where_origin_lower_puts_the_pixel(self):
        """Row 0 is drawn at the bottom, matching every other view."""
        import io

        from PIL import Image

        from pygor.tui.imaging import frame_preview

        images = np.zeros((2, 8, 8), dtype=np.float32)
        panel = frame_preview(images, 0, 400, 400, (0.0, 1.0), marker=(0, 0))
        picture = np.asarray(Image.open(io.BytesIO(panel.png)).convert("RGB"))
        cyan = np.argwhere((picture[:, :, 1] > 200) & (picture[:, :, 0] < 50))
        assert len(cyan) > 0
        rows, cols = cyan[:, 0], cyan[:, 1]
        # The horizontal arm sits in the bottom eighth, the vertical in the
        # leftmost eighth.
        assert rows.max() > picture.shape[0] * 0.8
        assert cols.min() < picture.shape[1] * 0.2

    def test_the_marker_colour_is_honoured(self):
        import io

        from PIL import Image

        from pygor.tui.imaging import frame_preview

        images = np.zeros((2, 8, 8), dtype=np.float32)

        def inks(colour):
            panel = frame_preview(images, 0, 400, 400, (0.0, 1.0), marker=(4, 4),
                                  marker_colour=colour)
            picture = np.asarray(Image.open(io.BytesIO(panel.png)).convert("RGB"))
            return {tuple(px) for px in picture.reshape(-1, 3) if tuple(px) != (0, 0, 0)}

        assert (0, 255, 255) in inks("cyan")
        assert (255, 0, 255) in inks("magenta")

    def test_the_display_range_clips_outliers_rather_than_stretching_to_them(self):
        """One hot pixel would otherwise wash out every frame it appears in."""
        from pygor.tui.probe import display_range

        images = np.zeros((40, 20, 20), dtype=np.float32)
        images[:, 5:15, 5:15] = 100.0
        images[:, 0, 0] = 5000.0  # 1 pixel in 400, under the 99.5th percentile
        low, high = display_range(images)
        assert high < 5000.0
        assert high > low


class TestHover:
    """The pointer path, which needs a terminal that draws images.

    ``probe()`` reports mode "none" whenever stdout is not a tty, which it
    never is under pytest, and in that mode there is no image widget to point
    at. So the capabilities are built rather than probed.
    """

    def _app(self, target):
        from pygor.tui.capabilities import Capabilities
        from pygor.tui.standalone import build_app

        caps = Capabilities(mode="halfcell", cell_width=10, cell_height=20,
                            is_tty=True, tmux=False, term="xterm-256color")
        return build_app(target, caps)

    def test_clicking_the_picture_moves_the_probe(self, one_recording):
        pytest.importorskip("textual_image")
        from pygor.tui.browse_screen import InspectScreen

        app = self._app(one_recording)
        seen = {}

        async def steps(pilot):
            await settle(pilot, app, lambda: isinstance(app.screen, InspectScreen)
                         and app.screen.recording is not None)
            widget = await live_widget(pilot, app)
            seen["widget"] = type(widget).__name__
            # Top-left cell of the panel: the last row of the array, column 0.
            await pilot.click(widget, offset=(0, 0))
            await settle(pilot, app, lambda: app.screen.probe == (7, 0))
            seen["top_left"] = app.screen.probe

        run_app(app, steps)
        assert seen["widget"].endswith("Image")
        assert seen["top_left"] == (7, 0)

    def test_pointing_without_clicking_leaves_the_probe_where_it_is(self, one_recording):
        pytest.importorskip("textual_image")
        from pygor.tui.browse_screen import InspectScreen

        app = self._app(one_recording)
        seen = {}

        async def steps(pilot):
            await settle(pilot, app, lambda: isinstance(app.screen, InspectScreen)
                         and app.screen.recording is not None)
            widget = await live_widget(pilot, app)
            await pilot.click(widget, offset=(0, 0))
            await settle(pilot, app, lambda: app.screen.probe == (7, 0))
            widget = await live_widget(pilot, app)
            await pilot.hover(widget, offset=centre(widget))
            for _ in range(10):
                await pilot.pause()
            seen["after_hover"] = app.screen.probe
            widget = await live_widget(pilot, app)
            await pilot.click(widget, offset=centre(widget))
            await settle(pilot, app, lambda: app.screen.probe != (7, 0))
            seen["after_click"] = app.screen.probe

        run_app(app, steps)
        assert seen["after_hover"] == (7, 0)
        assert seen["after_click"] != (7, 0)

    def test_arrow_keys_move_the_probe(self, one_recording):
        from pygor.tui.browse_screen import InspectScreen

        app = self._app(one_recording)
        seen = {}

        async def steps(pilot):
            await settle(pilot, app, lambda: isinstance(app.screen, InspectScreen)
                         and app.screen.recording is not None)
            await pilot.press("left")
            await settle(pilot, app, lambda: app.screen.probe == (4, 3))
            seen["moved"] = app.screen.probe

        run_app(app, steps)
        assert seen["moved"] == (4, 3)


@pytest.fixture
def long_recording(one_recording):
    """``one_recording`` with 400 frames: its own five are fewer than a zoom
    may narrow to, so every zoom on it is correctly refused."""
    import h5py

    with h5py.File(one_recording, "r+") as handle:
        group = handle["recording_000"]
        del group["images"]
        group.create_dataset(
            "images",
            data=np.random.default_rng(0).random((400, 8, 8)).astype(np.float32))
    return one_recording


class TestZoom:
    def _app(self, target):
        return TestHover()._app(target)

    def test_keys_zoom_pan_and_reset_the_window(self, long_recording):
        from pygor.tui.browse_screen import InspectScreen

        app = self._app(long_recording)
        seen = {}

        async def steps(pilot):
            await settle(pilot, app, lambda: isinstance(app.screen, InspectScreen)
                         and app.screen.recording is not None)
            await pilot.press("plus")
            seen["in"] = app.screen.window
            await pilot.press("right_square_bracket")
            seen["panned"] = app.screen.window
            await settle(pilot, app, lambda: "0 resets" in trace_text(app.screen))
            seen["label"] = trace_text(app.screen)
            await pilot.press("0")
            seen["reset"] = app.screen.window

        run_app(app, steps)
        assert seen["in"] == pytest.approx((0.25, 0.75))
        assert seen["panned"] == pytest.approx((0.5, 1.0))
        assert "2x" in seen["label"]
        assert seen["reset"] == (0.0, 1.0)

    def test_the_window_survives_moving_the_probe(self, long_recording):
        from pygor.tui.browse_screen import InspectScreen

        app = self._app(long_recording)
        seen = {}

        async def steps(pilot):
            await settle(pilot, app, lambda: isinstance(app.screen, InspectScreen)
                         and app.screen.recording is not None)
            await pilot.press("plus")
            await pilot.press("left")
            await settle(pilot, app, lambda: app.screen.probe == (4, 3))
            seen["window"] = app.screen.window

        run_app(app, steps)
        assert seen["window"] == pytest.approx((0.25, 0.75))

    def test_scrolling_over_the_trace_zooms_and_elsewhere_does_not(self, long_recording):
        from pygor.tui.browse_screen import InspectScreen, TracePane

        app = self._app(long_recording)
        seen = {}

        async def steps(pilot):
            await settle(pilot, app, lambda: isinstance(app.screen, InspectScreen)
                         and app.screen._trace_view is not None)
            pane = app.screen.query_one("#trace", TracePane)
            await scroll_up(pilot, pane, offset=centre(pane))
            await settle(pilot, app, lambda: app.screen.window != (0.0, 1.0))
            seen["zoomed"] = app.screen.window
            await scroll_up(pilot, "#meta")
            for _ in range(5):
                await pilot.pause()
            seen["after_meta"] = app.screen.window

        run_app(app, steps)
        low, high = seen["zoomed"]
        assert high - low == pytest.approx(0.8)
        assert seen["after_meta"] == seen["zoomed"]

    def test_pointing_at_the_trace_reads_out_time_and_value(self, long_recording):
        from pygor.tui.browse_screen import InspectScreen, TracePane

        app = self._app(long_recording)
        seen = {}

        async def steps(pilot):
            await settle(pilot, app, lambda: isinstance(app.screen, InspectScreen)
                         and app.screen._trace_view is not None)
            pane = app.screen.query_one("#trace", TracePane)
            await pilot.hover(pane, offset=centre(pane))
            await settle(pilot, app, lambda: "·  t " in text_of(app.screen, "#status"))
            seen["status"] = text_of(app.screen, "#status")

        run_app(app, steps)
        assert re.search(r"·  t [\d.]+s  ·  -?[\d.e+-]+", seen["status"])


async def scroll_up(pilot, widget, offset=(0, 0)):
    """The wheel, which the pilot has no public method for in Textual 8.

    ``_post_mouse_events`` is what ``click`` and ``hover`` are built on, so the
    event goes through the same dispatch a real one would.
    """
    from textual.events import MouseScrollUp

    await pilot._post_mouse_events([MouseScrollUp], widget, offset=offset)


def trace_text(screen):
    from textual.widgets import Static

    found = screen.query("#trace Static")
    return str(found.first(Static).content) if found else ""


class TestWindow:
    def test_zoom_keeps_the_anchor_point_still(self):
        from pygor.tui.probe import zoom

        low, high = zoom((0.0, 1.0), 0.5, 0.25, 1000)
        assert (low, high) == pytest.approx((0.125, 0.625))
        # The point a quarter of the way across is 0.25 before and after.
        assert low + 0.25 * (high - low) == pytest.approx(0.25)

    def test_zoom_stops_at_a_minimum_number_of_samples(self):
        from pygor.tui.probe import MIN_WINDOW, zoom

        window = (0.0, 1.0)
        for _ in range(40):
            window = zoom(window, 0.5, 0.5, 100)
        assert window[1] - window[0] == pytest.approx(MIN_WINDOW / 100)

    def test_zooming_out_past_the_whole_trace_is_the_whole_trace(self):
        from pygor.tui.probe import zoom

        assert zoom((0.4, 0.6), 100.0, 0.9, 1000) == pytest.approx((0.0, 1.0))

    def test_pan_stops_at_the_ends(self):
        from pygor.tui.probe import pan

        assert pan((0.0, 0.5), -1.0, 1000) == pytest.approx((0.0, 0.5))
        assert pan((0.25, 0.75), 1.0, 1000) == pytest.approx((0.5, 1.0))

    def test_windowed_slices_and_places_the_trace(self):
        from pygor.tui.probe import Trace, windowed

        trace = Trace("x", np.arange(100.0), 10.0, "pixel")
        part, start, end = windowed(trace, (0.5, 0.75))
        assert part.values[0] == 50.0
        assert len(part.values) == 25
        assert (start, end) == pytest.approx((5.0, 7.5))
        assert part.seconds == pytest.approx(2.5)

    def test_windowed_without_a_frame_rate_counts_frames(self):
        from pygor.tui.probe import Trace, windowed

        part, start, end = windowed(Trace("x", np.arange(100.0), None, "pixel"),
                                    (0.1, 0.2))
        assert (start, end) == (10.0, 20.0)
        assert part.seconds is None

    def test_a_window_narrower_than_the_pane_still_draws_a_line(self):
        """Ten samples over forty dot columns must not be ten isolated dots."""
        from pygor.tui.sparkline import BLANK, plot

        lines = plot(np.linspace(0, 1, 10), cols=20, rows=4)
        # Every character column has ink somewhere.
        assert all(any(line[i] != BLANK for line in lines) for i in range(20))

    def test_the_axis_is_labelled_from_the_start_of_the_window(self):
        from pygor.tui.sparkline import framed

        text = framed(np.arange(30.0), cols=30, rows=3, x_min=12.5, x_max=15.5)
        assert "12.5" in text.splitlines()[-1]
        assert "15.5s" in text.splitlines()[-1]

    def test_events_are_placed_within_the_window(self):
        from pygor.tui.sparkline import BRAILLE_BASE, event_row

        # One trigger at 105 in a 100-110 window lands mid-row; one at 5 is off it.
        row, _ = event_row(np.array([5.0, 105.0]), cols=20, x_min=100, x_max=110)
        inked = [i for i, ch in enumerate(row) if ord(ch) > BRAILLE_BASE]
        assert inked and 8 <= inked[0] <= 11


class TestNapariHandoff:
    def test_unsegmented_recording_contributes_no_labels_layer(self):
        from pygor.tui.napari_launcher import _add_rois

        class Viewer:
            def __init__(self):
                self.layers = []

            def add_labels(self, data, name, **kwargs):
                self.layers.append(name)

        class NoRois:
            rois = None

        viewer = Viewer()
        _add_rois(viewer, NoRois(), "ROIs")
        assert viewer.layers == []

        class WithRois:
            rois = np.array([[1, -1], [-2, 1]])

        _add_rois(viewer, WithRois(), "ROIs")
        assert viewer.layers == ["ROIs"]

    def test_spawn_refuses_before_starting_a_process_without_a_display(self, monkeypatch):
        from pygor.tui import napari_launcher

        monkeypatch.setattr("pygor.tui.capabilities.napari_availability",
                            lambda: (False, "no display on this session"))
        process, why = napari_launcher.spawn(["--path", "x"], label="x")
        assert process is None
        assert "no display" in why


def test_standalone_still_imports_without_textual_at_module_level():
    import importlib

    import pygor.tui.standalone as standalone

    importlib.reload(standalone)
    assert callable(standalone.load_recording)
    assert isinstance(pathlib.Path(standalone.APP_CSS or "."), pathlib.Path)
