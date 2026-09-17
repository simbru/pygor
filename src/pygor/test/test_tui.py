"""Tests for the terminal cockpit.

Two things matter here. The package must import with neither extra installed and
under an offscreen Qt, because the test suite runs that way and because someone
on a plain SSH connection has no napari. And the app must actually compose and
record a verdict, which Textual's headless driver can check without a terminal.
"""

from __future__ import annotations

import types

import pytest

from pygor.test.test_review_index import write_recording


class TestImportsWithoutExtras:
    def test_package_import_is_light(self):
        """Importing must not need textual: the entry point reports its absence."""
        import pygor.tui

        assert pygor.tui.__version__

    def test_capabilities_imports(self):
        import pygor.tui.capabilities  # noqa: F401

    def test_launcher_imports_without_napari(self):
        import pygor.tui.napari_launcher  # noqa: F401

    def test_main_module_imports(self):
        import pygor.tui.__main__  # noqa: F401


class TestCapabilities:
    def test_non_tty_degrades_rather_than_failing(self):
        from pygor.tui.capabilities import probe

        caps = probe("auto")
        assert caps.mode in ("none", "tgp", "sixel", "halfcell", "unicode")
        assert caps.cell_width > 0 and caps.cell_height > 0

    def test_explicit_none(self):
        from pygor.tui.capabilities import probe

        caps = probe("none")
        assert caps.mode == "none"
        assert not caps.graphical

    def test_unknown_mode_is_rejected(self):
        from pygor.tui.capabilities import probe

        with pytest.raises(ValueError):
            probe("magic")

    def test_no_widget_class_in_text_mode(self):
        from pygor.tui.capabilities import image_widget_class

        assert image_widget_class("none") is None

    def test_napari_availability_reports_a_reason(self, monkeypatch):
        from pygor.tui import capabilities

        monkeypatch.delenv("DISPLAY", raising=False)
        monkeypatch.delenv("WAYLAND_DISPLAY", raising=False)
        ok, why = capabilities.napari_availability()
        assert not ok
        assert why  # a sentence, not a bare False

    def test_headless_qt_is_refused(self, monkeypatch):
        from pygor.tui import capabilities

        monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
        ok, why = capabilities.napari_availability()
        assert not ok
        assert "headless" in why


class TestImaging:
    def test_pixels_for_cells(self):
        from pygor.tui.capabilities import Capabilities
        from pygor.tui.imaging import pixels_for_cells

        caps = Capabilities(mode="tgp", cell_width=10, cell_height=20, is_tty=True)
        assert pixels_for_cells(41, 11, caps) == (400, 200)

    def test_pixels_never_go_to_zero(self):
        from pygor.tui.capabilities import Capabilities
        from pygor.tui.imaging import pixels_for_cells

        caps = Capabilities(mode="tgp", cell_width=10, cell_height=20, is_tty=True)
        width, height = pixels_for_cells(0, 0, caps)
        assert width > 0 and height > 0

    def test_no_widget_in_text_mode(self):
        from pygor.review.rasterise import PanelImage, PanelKey
        from pygor.tui.capabilities import Capabilities
        from pygor.tui.imaging import describe_panel, make_widget

        key = PanelKey(panel="p", fov_uid="f", condition="c", role="", roi=None,
                       channel=-1, width=10, height=10, dpi=100, params_hash="",
                       sources=())
        image = PanelImage(png=b"x", width=10, height=10, key=key, meta={"panel": "p"})
        caps = Capabilities(mode="none", cell_width=10, cell_height=20, is_tty=False)
        assert make_widget(image, caps) is None
        assert "no graphics" in describe_panel(image)

    def test_clear_terminal_images_is_safe_off_a_terminal(self):
        from pygor.tui.imaging import clear_terminal_images

        clear_terminal_images()


@pytest.fixture
def session(tmp_path):
    """A ReviewSession over two synthetic recordings and no CSV."""
    from pygor.review.session import ReviewSession

    write_recording(tmp_path, stem="0_0_ColourSWN_200", num_rois=4)
    write_recording(
        tmp_path, stem="0_0_FFF_RGBUV", num_rois=3,
        roi_origin={"method": "transferred", "source": "0_0_ColourSWN_200",
                    "correlation": 0.5, "error": 0.1, "shift": [1.0, 0.0],
                    "expected_roi_ids": [-1, -2, -3, -4], "lost_roi_ids": [-2]},
    )

    def classify(stem):
        low = stem.lower()
        return "swn" if "swn" in low else "fff" if "fff" in low else None

    binding = types.SimpleNamespace(
        DATASET="Synthetic",
        ROOT=tmp_path,
        CSV=tmp_path / "does_not_exist.csv",
        STATUS=tmp_path / "Processed" / "status.csv",
        PANEL_SETS={"fov": ("fov_alignment", "fov_overlay"),
                    "cell": ("cell_rf_metrics",)},
        classify=classify,
        prefix_of=lambda stem: "0_0",
        fov_uid_of=lambda c, s, p: f"{s}::{c}::{p}",
        pick=lambda refs: sorted(refs, key=lambda r: len(r.stem))[0],
    )
    return ReviewSession(binding, reviewer="tester"), binding


class TestSession:
    def test_overview_ranks_the_worst_first(self, session):
        review, _ = session
        frame = review.overview()
        assert len(frame) == 1
        assert frame.iloc[0].n_lost == 1
        assert frame.iloc[0].roles == "fff+swn"

    def test_missing_csv_is_not_an_error(self, session):
        review, _ = session
        assert review.cells.empty
        assert review.progress()["cells"] == 0

    def test_find_by_partial_uid(self, session):
        review, _ = session
        assert review.find("0_0").prefix == "0_0"

    def test_find_unknown_raises(self, session):
        review, _ = session
        with pytest.raises(KeyError):
            review.find("nope")

    def test_judge_records_identity(self, session):
        review, _ = session
        bundle = review.find("0_0")
        entry = review.judge(bundle, subject_type="fov", check="alignment",
                             verdict="reject", reason="ROIs off the cells")
        assert entry.condition == bundle.condition
        assert entry.fov_uid == bundle.fov_uid
        assert entry.source_digest  # so staleness can be detected later
        assert review.verdict_for(bundle).verdict == "reject"

    def test_progress_counts_verdicts(self, session):
        review, _ = session
        review.judge(review.find("0_0"), subject_type="fov", check="alignment",
                     verdict="keep")
        assert review.progress()["fovs_judged"] == 1


def drive(app, *keys, after=None):
    """Run the app headlessly, press keys, and hand back the app.

    ``asyncio.run`` rather than an async-pytest plugin: the suite has no async
    tests otherwise, and this keeps it from needing one.
    """
    import asyncio

    async def body():
        async with app.run_test() as pilot:
            await pilot.pause()
            for key in keys:
                await pilot.press(key)
                await pilot.pause()
            if after is not None:
                after(app)

    asyncio.run(body())
    return app


class TestApp:
    """Headless drive of the real app, via Textual's test driver."""

    @pytest.fixture
    def app(self, session):
        pytest.importorskip("textual")
        from pygor.tui.app import ProofreadApp
        from pygor.tui.capabilities import probe

        review, binding = session
        return ProofreadApp(review, binding, probe("none"))

    def test_starts_and_lists_fovs(self, app):
        from textual.widgets import DataTable

        seen = {}
        drive(app, after=lambda a: seen.update(
            rows=a.screen.query_one("#fovs", DataTable).row_count))
        assert seen["rows"] == 1

    def test_verdict_key_records_a_verdict(self, app):
        drive(app, "a")
        stored = app.session.store.read_raw()
        assert len(stored) == 1
        assert stored[0].verdict == "keep"
        assert stored[0].subject_type == "fov"

    def test_reject_asks_for_a_reason(self, app):
        """A reject with no reason is unusable when read back months later."""
        from pygor.tui.app import ReasonPrompt

        seen = {}
        drive(app, "x", after=lambda a: seen.update(screen=a.screen))
        assert isinstance(seen["screen"], ReasonPrompt)

    def test_enter_opens_the_fov_screen(self, app):
        from pygor.tui.app import FovScreen

        seen = {}
        drive(app, "enter", after=lambda a: seen.update(screen=a.screen))
        assert isinstance(seen["screen"], FovScreen)

    def test_fov_screen_cycles_panels(self, app):
        seen = {}
        drive(app, "enter", "p", after=lambda a: seen.update(index=a.screen.panel_index))
        assert seen["index"] == 1

    @pytest.mark.parametrize("key", ["pagedown", "right_square_bracket", "n"])
    def test_cell_screen_walks_cells(self, app, key):
        from pygor.tui.app import CellScreen

        seen = {}
        drive(app, "enter", "enter", key,
              after=lambda a: seen.update(screen=a.screen, roi=a.screen.roi))
        assert isinstance(seen["screen"], CellScreen)
        assert seen["roi"] == 1

    def test_embedded_shell_survives_a_running_event_loop(self):
        """The cockpit suspends but its asyncio loop keeps running.

        IPython's prompt starts a loop of its own, so calling it on this thread
        raises 'asyncio.run() cannot be called from a running event loop'. The
        shell has to go on a thread without one.
        """
        import asyncio

        from pygor.tui.app import embed_shell

        calls = []

        def fake_embed(header=None, user_ns=None):
            # Mimics prompt_toolkit: refuse if a loop is already running here.
            asyncio.get_event_loop_policy()
            try:
                asyncio.get_running_loop()
            except RuntimeError:
                calls.append("clean thread")
                return
            raise RuntimeError("asyncio.run() cannot be called from a running event loop")

        import IPython

        original = IPython.embed
        IPython.embed = fake_embed
        try:

            async def body():
                embed_shell("header", {"x": 1})

            asyncio.run(body())
        finally:
            IPython.embed = original

        assert calls == ["clean thread"]

    def test_verdict_keys_are_documented(self):
        """Handling a key in on_key leaves it out of the footer and help panel.

        That is how the verdict keys came to be invisible: they worked, but
        nothing in the interface said they existed.
        """
        from pygor.tui.app import ReviewScreen

        described = {
            binding.key: binding.description
            for binding in ReviewScreen.BINDINGS
        }
        for key in ("a", "x", "f", "s"):
            assert key in described, f"{key} is not a declared binding"
            assert described[key], f"{key} has no description to show"

    def test_every_binding_uses_a_real_key_name(self):
        """A misspelled key silently never fires, and a test that presses the
        same misspelling passes anyway. So check the names themselves."""
        import string

        from textual.keys import Keys, _character_to_key

        from pygor.tui.app import CellScreen, FovScreen, IndexScreen, ProofreadApp

        known = {k.value for k in Keys}
        known |= {_character_to_key(c) for c in string.printable}
        bad = []
        for owner in (ProofreadApp, IndexScreen, FovScreen, CellScreen):
            for binding in getattr(owner, "BINDINGS", []):
                keys = binding.key if isinstance(binding.key, str) else ""
                for key in keys.split(","):
                    key = key.strip()
                    if key and key not in known and len(key) > 1:
                        bad.append(f"{owner.__name__}: {key}")
        assert not bad, f"bindings that will never fire: {bad}"

    def test_cell_verdict_targets_the_cell_not_the_fov(self, app):
        """The status bar says what a verdict applies to; it must be true."""
        drive(app, "enter", "enter", "a")
        stored = [v for v in app.session.store.read_raw() if v.subject_type == "cell"]
        assert len(stored) == 1
        assert stored[0].subject_uid.endswith("#0")
        assert stored[0].check == "rf_quality"

    def test_napari_without_a_display_notifies_rather_than_hangs(self, app, monkeypatch):
        monkeypatch.delenv("DISPLAY", raising=False)
        monkeypatch.delenv("WAYLAND_DISPLAY", raising=False)
        drive(app, "enter", "v")
        assert app._napari_procs == []
