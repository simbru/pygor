"""Unit tests for pygor.review.panels and .rasterise.

The leak check is the important one. Several pygor plotting functions build
their figures through pyplot, which keeps a global reference, so a review
session rendering a few hundred panels would grow until it died. Every panel is
asserted to leave the pyplot registry empty.
"""

import matplotlib
import numpy as np
import pytest

matplotlib.use("Agg", force=True)

import matplotlib.pyplot as plt
from matplotlib.figure import Figure

from pygor.review import index as review_index
from pygor.review import panels as review_panels
from pygor.review.bundle import FovBundle
from pygor.review.panels import PANELS, panels_for, render, roi_outlines
from pygor.review.rasterise import (
    PanelCache,
    PanelKey,
    figure_to_png,
    hash_params,
    size_figure,
    trim_uniform_border,
)
from pygor.test.test_review_index import write_recording


def make_bundle(tmp_path, *, with_partner=True, lost=()):
    write_recording(tmp_path, stem="0_0_ColourSWN_200", num_rois=4)
    refs_by_stem = {}
    if with_partner:
        write_recording(
            tmp_path,
            stem="0_0_FFF_RGBUV",
            num_rois=4 - len(lost),
            roi_origin={
                "method": "transferred",
                "source": "0_0_ColourSWN_200",
                "correlation": 0.55,
                "error": 0.1,
                "shift": [1.0, -2.0],
                "expected_roi_ids": [-1, -2, -3, -4],
                "lost_roi_ids": list(lost),
            },
        )
    for ref in review_index.scan_processed(tmp_path, cache=False):
        refs_by_stem[ref.stem] = ref
    refs = {"swn": refs_by_stem["0_0_ColourSWN_200"]}
    if with_partner:
        refs["fff"] = refs_by_stem["0_0_FFF_RGBUV"]
    return FovBundle(
        fov_uid="240101 test::control::0_0",
        condition="control",
        session="240101 test",
        prefix="0_0",
        refs=refs,
        master_role="swn",
        expected={"swn": True, "fff": True, "osds": False},
    )


@pytest.fixture
def bundle(tmp_path):
    return make_bundle(tmp_path, lost=[-2])


LIGHT_PANELS = sorted(name for name, spec in PANELS.items() if spec.light)


class TestRasterise:
    def test_exact_pixel_size(self):
        figure = Figure(figsize=(1, 1), dpi=100)
        png = figure_to_png(figure, width=640, height=480, dpi=100)
        decoded = matplotlib.image.imread(__import__("io").BytesIO(png))
        assert decoded.shape[1] == 640
        assert decoded.shape[0] == 480

    def test_closes_the_figure(self):
        figure = plt.figure()
        assert plt.get_fignums()
        figure_to_png(figure, width=100, height=100)
        assert plt.get_fignums() == []

    def test_closes_the_figure_even_when_saving_fails(self):
        figure = plt.figure()

        def boom(*args, **kwargs):
            raise RuntimeError("nope")

        figure.savefig = boom
        with pytest.raises(RuntimeError):
            figure_to_png(figure, width=100, height=100)
        assert plt.get_fignums() == []

    def test_size_figure_sets_inches_from_pixels(self):
        figure = Figure()
        size_figure(figure, 800, 400, dpi=100)
        assert figure.get_size_inches() == pytest.approx([8.0, 4.0])

    def test_trim_removes_uniform_edges(self):
        canvas = np.zeros((20, 20, 4), dtype=np.uint8)
        canvas[5:15, 5:15] = 255
        trimmed = trim_uniform_border(canvas)
        assert trimmed.shape[:2] == (10, 10)

    def test_trim_keeps_an_image_with_no_uniform_edge(self):
        canvas = np.random.randint(0, 255, (10, 10, 4), dtype=np.uint8)
        assert trim_uniform_border(canvas).shape == canvas.shape

    def test_trim_of_a_blank_image_is_a_no_op(self):
        canvas = np.zeros((8, 8, 4), dtype=np.uint8)
        assert trim_uniform_border(canvas).shape == canvas.shape

    def test_hash_params_is_order_independent(self):
        assert hash_params({"a": 1, "b": 2}) == hash_params({"b": 2, "a": 1})
        assert hash_params({}) == ""
        assert hash_params({"a": 1}) != hash_params({"a": 2})


class TestPanelCache:
    def _key(self, panel="p", sources=(("swn", 1.0, 10),)):
        return PanelKey(panel=panel, fov_uid="f", condition="c", role="swn", roi=1,
                        channel=-1, width=10, height=10, dpi=100, params_hash="",
                        sources=sources)

    def test_miss_then_hit(self):
        from pygor.review.rasterise import PanelImage

        cache = PanelCache()
        key = self._key()
        assert cache.get(key) is None
        cache.put(PanelImage(png=b"x", width=10, height=10, key=key))
        assert cache.get(key).png == b"x"

    def test_changed_source_is_a_different_key(self):
        """Reprocessing a recording must not leave a stale picture cached."""
        assert self._key().digest() != self._key(sources=(("swn", 2.0, 10),)).digest()

    def test_evicts_oldest(self):
        from pygor.review.rasterise import PanelImage

        cache = PanelCache(max_items=2)
        for name in "abc":
            key = self._key(panel=name)
            cache.put(PanelImage(png=b"x", width=1, height=1, key=key))
        assert cache.stats()["items"] == 2
        assert cache.get(self._key(panel="a")) is None


class TestRegistry:
    def test_every_panel_declares_a_known_level_and_check(self):
        from pygor.review.verdicts import CHECKS, SUBJECT_TYPES

        for spec in PANELS.values():
            assert spec.level in SUBJECT_TYPES
            assert spec.check in CHECKS

    def test_panels_for_filters(self):
        assert all(s.level == "fov" for s in panels_for(level="fov"))
        assert all(s.check == "alignment" for s in panels_for(check="alignment"))

    def test_unknown_panel_names_what_exists(self, bundle):
        with pytest.raises(KeyError, match="no panel named"):
            render("not_a_panel", bundle)


class TestLightPanels:
    @pytest.mark.parametrize("name", LIGHT_PANELS)
    def test_renders_without_loading_anything(self, name, bundle):
        image = render(name, bundle, width=400, height=200, roi=0, role="swn")
        assert image.png[:8] == b"\x89PNG\r\n\x1a\n"
        assert bundle.cache.stats()["items"] == 0

    @pytest.mark.parametrize("name", LIGHT_PANELS)
    def test_leaves_no_open_figures(self, name, bundle):
        render(name, bundle, width=400, height=200, roi=0, role="swn")
        assert plt.get_fignums() == []

    @pytest.mark.parametrize("name", LIGHT_PANELS)
    def test_output_is_the_requested_size(self, name, bundle):
        import io

        image = render(name, bundle, width=400, height=200, roi=0, role="swn")
        decoded = matplotlib.image.imread(io.BytesIO(image.png))
        if PANELS[name].trim:
            assert decoded.shape[1] <= 400 and decoded.shape[0] <= 200
        else:
            assert (decoded.shape[1], decoded.shape[0]) == (400, 200)

    def test_cache_returns_the_same_image(self, bundle):
        cache = PanelCache()
        first = render("fov_alignment", bundle, width=400, height=200, cache=cache)
        second = render("fov_alignment", bundle, width=400, height=200, cache=cache)
        assert first is second
        assert cache.stats()["hits"] == 1

    def test_single_recording_fov_says_so_rather_than_failing(self, tmp_path):
        """135 of the dataset's fields of view have no partner to align against."""
        alone = make_bundle(tmp_path, with_partner=False)
        image = render("fov_overlay", alone, width=300, height=150)
        assert image.png[:4] == b"\x89PNG"

    def test_lost_roi_panel_runs_when_nothing_was_lost(self, tmp_path):
        clean = make_bundle(tmp_path, lost=[])
        assert render("fov_lost_rois", clean, width=300, height=150).png[:4] == b"\x89PNG"


class TestCellPanels:
    def test_missing_cell_renders_an_explanation_not_an_error(self, bundle):
        """Cell 1 was dropped when ROIs were transferred to the partner."""
        image = render("cell_rf_chroma", bundle, roi=1, role="fff",
                       width=300, height=150)
        assert image.png[:4] == b"\x89PNG"
        assert plt.get_fignums() == []

    def test_metrics_panel_without_a_csv_row(self, bundle):
        image = render("cell_rf_metrics", bundle, roi=0, width=300, height=150)
        assert image.png[:4] == b"\x89PNG"

    def test_metrics_panel_reads_the_csv_row(self, bundle):
        import pandas as pd

        bundle.attach_cells(
            pd.DataFrame(
                {
                    "roi_id": [0],
                    "strf_pass_bool_ch0": [True],
                    "strf_magnitudes_ch0": [3.4],
                    "strf_categorysimple": ["on"],
                }
            )
        )
        assert render("cell_rf_metrics", bundle, roi=0, width=400, height=200)

    def test_cell_panels_need_a_roi(self, bundle):
        with pytest.raises(ValueError, match="needs a roi"):
            render("cell_rf_metrics", bundle, width=200, height=100)


class TestOutlines:
    def test_one_outline_per_roi(self):
        mask = np.ones((10, 10), dtype=int)
        mask[2:4, 2:4] = -1
        mask[6:8, 6:8] = -2
        assert len(roi_outlines(mask)) == 2

    def test_background_only_mask_has_no_outlines(self):
        assert roi_outlines(np.ones((5, 5), dtype=int)) == []


class TestStretch:
    def test_flat_image_does_not_divide_by_zero(self):
        assert np.all(review_panels._stretch(np.ones((4, 4))) == 0)

    def test_all_nan_image_is_handled(self):
        assert np.all(review_panels._stretch(np.full((4, 4), np.nan)) == 0)

    def test_range_is_clipped_to_unit(self):
        stretched = review_panels._stretch(np.arange(100).reshape(10, 10).astype(float))
        assert stretched.min() >= 0 and stretched.max() <= 1
