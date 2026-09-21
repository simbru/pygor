"""The terminal parameter table and the reprocess screen.

The screen owns no processing, so it is tested against fake run/preview/save
callables and the assertions are about the loop: an edit becomes a nested
override, re-run hands it to run(), save hands the result to save() and only
after confirmation, and back reports what happened.
"""

from __future__ import annotations

import asyncio

import pytest

from pygor.core.gui.param_values import flatten, nest, parse_value, type_name


class TestParamValues:
    def test_parse_by_original_type(self):
        assert parse_value("0.02", 0.025) == 0.02
        assert parse_value("7", 3) == 7 and isinstance(parse_value("7", 3), int)
        assert parse_value("yes", False) is True
        assert parse_value("[1, 2]", [0]) == [1, 2]
        assert parse_value("blob", "watershed") == "blob"

    def test_fractional_edit_of_an_int_promotes_rather_than_truncates(self):
        """max_sigma = 2 in a TOML is a float quantity written without a point.

        The parser used to do int(float("2.5")) and hand back 2 -- an edit the
        user made, silently thrown away. Now it becomes 2.5; a whole number
        stays an int.
        """
        assert parse_value("2.5", 2) == 2.5
        assert isinstance(parse_value("2.5", 2), float)
        assert parse_value("3", 2) == 3 and isinstance(parse_value("3", 2), int)
        assert parse_value("3.0", 2) == 3 and isinstance(parse_value("3.0", 2), int)

    def test_bad_text_raises_not_coerces(self):
        with pytest.raises(ValueError):
            parse_value("maybe", True)
        with pytest.raises(ValueError):
            parse_value("x", 1.0)

    def test_none_original_takes_a_literal_or_a_string(self):
        assert parse_value("3", None) == 3
        assert parse_value("otsu", None) == "otsu"

    def test_flatten_and_nest_round_trip(self):
        tree = {"a": {"b": 1, "c": {"d": True}}, "e": "x"}
        flat = dict(flatten(tree))
        assert flat == {"a.b": 1, "a.c.d": True, "e": "x"}
        assert nest(flat) == tree

    def test_type_names(self):
        assert type_name(True) == "bool"  # before int, since bool is an int
        assert type_name(1) == "int"
        assert type_name(None) == "NoneType"


def drive(app, *keys, after=None):
    async def body():
        async with app.run_test(size=(160, 45)) as pilot:
            await pilot.pause()
            for key in keys:
                if callable(key):
                    key(app)
                else:
                    await pilot.press(key)
                await pilot.pause()
            if after:
                after(app)

    asyncio.run(body())
    return app


@pytest.fixture
def harness():
    pytest.importorskip("textual")
    from textual.app import App

    from pygor.tui.capabilities import probe
    from pygor.tui.reprocess_screen import ReprocessScreen

    calls = {"run": [], "save": [], "preview": []}

    def run(overrides):
        calls["run"].append(overrides)
        return {"result": overrides}

    def preview(result, which, width, height):
        calls["preview"].append((result, which))
        from pygor.review.rasterise import PanelImage, PanelKey

        key = PanelKey(panel="p", fov_uid="f", condition="", role="", roi=None,
                       channel=-1, width=10, height=10, dpi=100, params_hash="",
                       sources=())
        return PanelImage(png=b"\x89PNG\r\n\x1a\n" + b"\0" * 16, width=10, height=10, key=key)

    def save(result, overrides):
        calls["save"].append((result, overrides))

    values = {"segmentation": {"blob": {"threshold": 0.025, "min_sigma": 0.8},
                               "general": {"mode": "blob"}},
              "preprocessing": {"artifact_width": 3, "detrend": True}}
    outcomes = []

    class Harness(App):
        def on_mount(self):
            self.push_screen(
                ReprocessScreen(title="t", values=values,
                                sections=("segmentation", "preprocessing.artifact_width"),
                                run=run, preview=preview, save=save, caps=probe("none")),
                outcomes.append,
            )

    return Harness(), calls, outcomes


def _set_threshold(app, text):
    from pygor.tui.params_table import ParamTable

    table = app.screen.query_one("#params", ParamTable)
    index = next(i for i, (p, _) in enumerate(table._rows) if p == "segmentation.blob.threshold")
    table.move_cursor(row=index)


class TestReprocessScreen:
    def test_sections_filter_the_rows(self, harness):
        from pygor.tui.params_table import ParamTable

        app, _, _ = harness
        seen = {}
        drive(app, after=lambda a: seen.update(
            paths=[p for p, _ in a.screen.query_one("#params", ParamTable)._rows]))
        assert "segmentation.blob.threshold" in seen["paths"]
        assert "preprocessing.artifact_width" in seen["paths"]
        assert "preprocessing.detrend" not in seen["paths"]  # outside the sections

    def test_edit_becomes_a_nested_override(self, harness):
        from pygor.tui.params_table import ParamTable

        app, _, _ = harness
        seen = {}

        def type_value(a):
            a.screen.query_one("#value").value = "0.015"

        drive(app, lambda a: _set_threshold(a, None), "enter", type_value, "enter",
              after=lambda a: seen.update(
                  nested=a.screen.query_one("#params", ParamTable).nested_overrides()))
        assert seen["nested"] == {"segmentation": {"blob": {"threshold": 0.015}}}

    def test_bad_edit_is_refused_and_kept_open(self, harness):
        from pygor.tui.params_table import ValuePrompt

        app, _, _ = harness
        seen = {}

        def type_value(a):
            a.screen.query_one("#value").value = "not a number"

        drive(app, lambda a: _set_threshold(a, None), "enter", type_value, "enter",
              after=lambda a: seen.update(screen=a.screen))
        assert isinstance(seen["screen"], ValuePrompt)

    def test_run_passes_overrides_and_previews_result(self, harness):
        app, calls, _ = harness

        def type_value(a):
            a.screen.query_one("#value").value = "0.015"

        drive(app, lambda a: _set_threshold(a, None), "enter", type_value, "enter", "R")
        assert calls["run"] == [{"segmentation": {"blob": {"threshold": 0.015}}}]
        # First preview is of what's on disk (None), the one after the run of the result.
        assert calls["preview"][0][0] is None
        assert calls["preview"][-1][0] == {"result": {"segmentation": {"blob": {"threshold": 0.015}}}}

    def test_save_needs_a_result_and_a_confirmation(self, harness):
        app, calls, _ = harness
        drive(app, "S")  # nothing run yet
        assert calls["save"] == []

    def test_save_after_run_and_confirm(self, harness):
        app, calls, _ = harness
        drive(app, "R", "S", "y")
        assert len(calls["save"]) == 1
        result, overrides = calls["save"][0]
        assert result == {"result": {}}

    def test_save_declined_saves_nothing(self, harness):
        app, calls, _ = harness
        drive(app, "R", "S", "n")
        assert calls["save"] == []

    def test_preview_stays_inside_the_screen(self, harness):
        """The preview pane once inherited another screen's 2fr height rule
        through a shared id and ran to twice the screen; the image was centred
        in it and half of it was below the footer."""
        app, _, _ = harness
        seen = {}

        def measure(a):
            screen = a.screen
            pane = screen.query_one("#reprocess-preview")
            seen["screen_h"] = screen.size.height
            seen["pane"] = pane.region
            seen["children"] = [c.region for c in pane.children]

        drive(app, after=measure)
        pane = seen["pane"]
        assert pane.y + pane.height <= seen["screen_h"]
        for child in seen["children"]:
            assert child.y >= pane.y
            assert child.y + child.height <= pane.y + pane.height

    @pytest.mark.parametrize("keys,outcome", [
        (("escape",), "unchanged"),
        (("R", "escape"), "discarded"),
        (("R", "S", "y", "escape"), "saved"),
    ])
    def test_back_reports_what_happened(self, harness, keys, outcome):
        app, _, outcomes = harness
        drive(app, *keys)
        assert outcomes == [outcome]


class TestModeGating:
    def _table_app(self):
        from textual.app import App

        from pygor.tui.params_table import ParamTable
        from pygor.tui.reprocess_screen import segmentation_gating

        values = {"segmentation": {"general": {"mode": "blob", "roi_order": "LR"},
                                   "blob": {"threshold": 0.02},
                                   "watershed": {"threshold": 0.05, "min_distance": 1},
                                   "cellpose": {"diameter": 0},
                                   "cellpose_postprocess": {"split_large": True}}}
        choices, gates = segmentation_gating(values)

        class Harness(App):
            def compose(self):
                yield ParamTable(values, choices=choices, gates=gates, id="params")

        return Harness(), choices

    def test_choices_are_derived_from_the_sections(self):
        _, choices = self._table_app()
        assert choices == {"segmentation.general.mode": ("blob", "cellpose", "watershed")}

    def test_only_the_chosen_modes_parameters_show(self):
        from pygor.tui.params_table import ParamTable

        app, _ = self._table_app()
        seen = {}
        drive(app, after=lambda a: seen.update(rows=[p for p, _ in a.query_one(ParamTable)._rows]))
        assert "segmentation.blob.threshold" in seen["rows"]
        assert "segmentation.watershed.threshold" not in seen["rows"]
        assert "segmentation.cellpose.diameter" not in seen["rows"]

    def test_picking_a_mode_swaps_the_rows_and_records_it(self):
        from pygor.tui.params_table import ChoicePrompt, ParamTable

        app, _ = self._table_app()
        seen = {}

        def go_to_mode(a):
            table = a.query_one(ParamTable)
            table.move_cursor(row=[p for p, _ in table._rows].index("segmentation.general.mode"))

        def pick_cellpose(a):
            assert isinstance(a.screen, ChoicePrompt)
            a.screen.query_one("#choices").highlighted = 1  # cellpose

        drive(app, go_to_mode, "enter", pick_cellpose, "enter",
              after=lambda a: seen.update(rows=[p for p, _ in a.query_one(ParamTable)._rows],
                                          overrides=a.query_one(ParamTable).overrides))
        assert seen["overrides"] == {"segmentation.general.mode": "cellpose"}
        assert "segmentation.cellpose.diameter" in seen["rows"]
        assert "segmentation.cellpose_postprocess.split_large" in seen["rows"]  # travels with it
        assert "segmentation.blob.threshold" not in seen["rows"]

    def test_reverting_the_mode_restores_the_rows(self):
        from pygor.tui.params_table import ParamTable

        app, _ = self._table_app()
        seen = {}

        def go_to_mode(a):
            table = a.query_one(ParamTable)
            table.move_cursor(row=[p for p, _ in table._rows].index("segmentation.general.mode"))

        def pick_watershed(a):
            a.screen.query_one("#choices").highlighted = 2

        drive(app, go_to_mode, "enter", pick_watershed, "enter", "u",
              after=lambda a: seen.update(rows=[p for p, _ in a.query_one(ParamTable)._rows]))
        assert "segmentation.blob.threshold" in seen["rows"]
        assert "segmentation.watershed.threshold" not in seen["rows"]


class TestStandaloneKwargs:
    def test_recipe_shape_becomes_segment_rois_arguments(self):
        from pygor.tui.standalone import segmentation_kwargs

        class Params:
            @staticmethod
            def get_defaults(section):
                return {"general": {"mode": "blob"}}

        class Rec:
            params = Params()

        mode, kwargs = segmentation_kwargs(
            Rec(), {"segmentation": {"blob": {"threshold": 0.015, "edge_margin": 5}},
                    "preprocessing": {"artifact_width": 4}})
        assert mode == "blob"
        assert kwargs == {"threshold": 0.015, "edge_margin": 5, "artifact_width": 4}

    def test_mode_override_selects_that_modes_parameters(self):
        from pygor.tui.standalone import segmentation_kwargs

        class Params:
            @staticmethod
            def get_defaults(section):
                return {"general": {"mode": "blob"}}

        class Rec:
            params = Params()

        mode, kwargs = segmentation_kwargs(
            Rec(), {"segmentation": {"general": {"mode": "watershed"},
                                     "watershed": {"threshold": 0.05},
                                     "blob": {"threshold": 0.01}}})
        assert mode == "watershed"
        assert kwargs == {"threshold": 0.05}

    def test_standalone_imports_without_textual_at_module_level(self):
        import pygor.tui.standalone  # noqa: F401


class TestConfigMergeTypes:
    def test_int_in_config_over_float_default_stays_float(self):
        """swn.toml writes max_sigma = 2 over pygor's 2.0; the table labelled
        it int and the parser then treated edits as ints."""
        from pygor.config import _deep_merge

        merged = _deep_merge({"blob": {"max_sigma": 2.0, "n": 3, "flag": True}},
                             {"blob": {"max_sigma": 2, "n": 4, "flag": False}})
        assert merged["blob"]["max_sigma"] == 2.0
        assert isinstance(merged["blob"]["max_sigma"], float)
        assert merged["blob"]["n"] == 4 and isinstance(merged["blob"]["n"], int)
        assert merged["blob"]["flag"] is False  # a bool is not promoted

    def test_recipe_values_carry_float_for_max_sigma(self):
        binding = pytest.importorskip("analyses.review_datasets.chromatic_swn")
        blob = binding.recipe_values()["segmentation"]["blob"]
        assert isinstance(blob["max_sigma"], float)


class TestReplaceFovRows:
    """The ghost-row bug: a re-run that finds fewer cells left the rest behind."""

    def _csv(self, tmp_path):
        import pandas as pd

        rows = []
        for fov, cond, n in (("s::control::0_0", "control", 176), ("s::control::0_1", "control", 40),
                             ("s::acblock::0_0", "acblock", 20)):  # same prefix, other condition
            rows += [{"cell_uid": f"{fov}#{i}", "fov_uid": fov, "condition": cond,
                      "roi_id": i, "value": 1.0} for i in range(n)]
        path = tmp_path / "rois.csv"
        pd.DataFrame(rows).to_csv(path, index=False)
        return path

    def test_fewer_cells_leaves_no_ghosts(self, tmp_path):
        import pandas as pd

        binding = pytest.importorskip("analyses.review_datasets.chromatic_swn")
        path = self._csv(tmp_path)
        new = [{"cell_uid": f"s::control::0_0#{i}", "fov_uid": "s::control::0_0",
                "condition": "control", "roi_id": i, "value": 2.0} for i in range(117)]
        dropped, written = binding.replace_fov_rows(path, "s::control::0_0", "control", new)
        assert (dropped, written) == (176, 117)
        frame = pd.read_csv(path)
        mine = frame[(frame.fov_uid == "s::control::0_0") & (frame.condition == "control")]
        assert len(mine) == 117
        assert (mine.value == 2.0).all()
        assert len(frame[frame.fov_uid == "s::control::0_1"]) == 40  # untouched
        assert len(frame[frame.condition == "acblock"]) == 20  # colliding uid, other condition

    def test_missing_csv_just_writes(self, tmp_path):
        binding = pytest.importorskip("analyses.review_datasets.chromatic_swn")
        path = tmp_path / "new.csv"
        dropped, written = binding.replace_fov_rows(
            path, "f", "c", [{"cell_uid": "f#0", "fov_uid": "f", "condition": "c"}])
        assert (dropped, written) == (0, 1) and path.exists()


class TestRestoreTerminal:
    def test_off_a_tty_it_is_a_no_op(self):
        from pygor.tui.imaging import restore_terminal

        restore_terminal()  # must not raise or write

    def test_main_restores_even_when_run_raises(self, monkeypatch, tmp_path):
        """An exit through an exception used to leave the next session's keys broken."""
        import io
        import sys

        import pygor.tui.__main__ as entry

        written = io.StringIO()

        class FakeStdout(io.StringIO):
            def isatty(self):
                return True

        fake = FakeStdout()
        monkeypatch.setattr(sys, "__stdout__", fake)
        monkeypatch.setattr(sys, "__stdin__", io.StringIO())  # termios path fails safely

        class Boom(Exception):
            pass

        class FakeApp:
            def __init__(self, *a, **k):
                pass

            def run(self):
                raise Boom()

        import types
        monkeypatch.setitem(sys.modules, "pygor.tui.app", types.SimpleNamespace(ProofreadApp=FakeApp))
        monkeypatch.setitem(sys.modules, "analyses.review_datasets.fake",
                            types.SimpleNamespace(ROOT=tmp_path, DATASET="x", CSV=tmp_path / "c",
                                                  STATUS=tmp_path / "s", classify=lambda s: None,
                                                  prefix_of=lambda s: None, fov_uid_of=lambda *a: "",
                                                  PANEL_SETS={}))
        with pytest.raises(Boom):
            entry.main(["--binding", "analyses.review_datasets.fake", "--graphics", "none"])
        assert "\x1b[<u" in fake.getvalue()
        assert "\x1b[?1049l" in fake.getvalue()
