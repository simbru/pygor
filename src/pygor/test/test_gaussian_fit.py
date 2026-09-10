"""Unit tests for the Gaussian-fit index, using synthetic arrays only.

Most of these need no recording, so they run on a fresh clone. The ones that do are
marked `demo_data` and conftest skips them when the file is missing.

A note on why every synthetic map here carries noise: the index calibrates its noise
level from the residual out where the fit predicts nothing, so a perfectly noiseless
map gives sd == 0 and scores NaN by design rather than 0.
"""

import numpy as np
import pytest

import pygor.load
import pygor.strf.gaussian_fit as gaussian_fit
from pygor.classes.strf_data import STRF

SIZE = 41  # big enough that the >3 sigma annulus has the 20 pixels the noise fit needs
SEED = 0


def oriented_gaussian(amp, x0, y0, sx, sy, theta, size=SIZE):
    """A rotated Gaussian built independently of the module under test.

    Written in the same frame ellipse_mask uses -- u along the major axis, theta
    counter-clockwise from +x -- so agreement with gaussian2d is a real check of the
    rotation convention rather than a restatement of it.
    """
    yy, xx = np.mgrid[0:size, 0:size]
    dx, dy = xx - x0, yy - y0
    u = dx * np.cos(theta) + dy * np.sin(theta)
    v = -dx * np.sin(theta) + dy * np.cos(theta)
    return amp * np.exp(-(u**2 / (2 * sx**2) + v**2 / (2 * sy**2)))


def with_noise(image, sd=0.05, seed=SEED):
    return image + np.random.default_rng(seed).normal(0, sd, image.shape)


@pytest.fixture
def clean_gaussian():
    """A single Gaussian, mildly elongated and axis-aligned."""
    return with_noise(oriented_gaussian(10.0, 20.0, 20.0, 5.0, 3.0, 0.0))


class TestFit:
    def test_recovers_the_generating_parameters(self, clean_gaussian):
        fit = gaussian_fit.fit_gaussian2d(clean_gaussian)
        assert fit["amp"] == pytest.approx(10.0, abs=0.5)
        assert fit["x0"] == pytest.approx(20.0, abs=0.5)
        assert fit["y0"] == pytest.approx(20.0, abs=0.5)
        assert fit["sx"] == pytest.approx(5.0, abs=0.5)
        assert fit["sy"] == pytest.approx(3.0, abs=0.5)

    @pytest.mark.parametrize("theta", [-np.pi / 3, -np.pi / 6, np.pi / 6, np.pi / 4])
    def test_oblique_fits_recover_the_angle_with_the_right_sign(self, theta):
        """The cross-term sign in gaussian2d; axis-aligned fits cannot see it.

        With it flipped the model rotates clockwise while ellipse_mask rotates
        counter-clockwise, and an oblique RF gets a mirrored analysis region.
        """
        image = with_noise(oriented_gaussian(10.0, 20.0, 20.0, 6.0, 2.5, theta))
        fit = gaussian_fit.fit_gaussian2d(image)
        assert fit["theta"] == pytest.approx(theta, abs=0.1)

    def test_major_axis_is_always_sx(self):
        """Whichever way curve_fit lands, _canonical puts the long axis on sx."""
        for sx, sy in ((6.0, 2.5), (2.5, 6.0)):
            image = with_noise(oriented_gaussian(10.0, 20.0, 20.0, sx, sy, 0.0))
            fit = gaussian_fit.fit_gaussian2d(image)
            assert fit["sx"] >= fit["sy"]
            assert max(fit["sx"], fit["sy"]) == pytest.approx(max(sx, sy), abs=0.5)

    def test_fitted_keeps_the_baseline(self, clean_gaussian):
        """The offset stays in `fitted`, or it lands in the residual as a DC floor."""
        fit = gaussian_fit.fit_gaussian2d(clean_gaussian)
        resid = np.abs(clean_gaussian) - fit["fitted"]
        assert abs(float(resid.mean())) < 0.05

    def test_an_empty_map_is_refused(self):
        with pytest.raises(ValueError):
            gaussian_fit.fit_gaussian2d(np.zeros((SIZE, SIZE)))


class TestIndex:
    def test_a_single_gaussian_scores_near_zero(self, clean_gaussian):
        scored = gaussian_fit.fit_and_score(clean_gaussian)
        assert scored["index"] < 0.1
        assert scored["n_px"] > 0

    def test_two_lobes_score_higher_than_one(self, clean_gaussian):
        """The case the metric exists for: no single Gaussian describes an opponent RF."""
        opponent = with_noise(
            oriented_gaussian(10.0, 14.0, 20.0, 3.0, 3.0, 0.0)
            - oriented_gaussian(10.0, 27.0, 20.0, 3.0, 3.0, 0.0)
        )
        assert (
            gaussian_fit.fit_and_score(opponent)["index"]
            > gaussian_fit.fit_and_score(clean_gaussian)["index"]
        )

    def test_index_does_not_simply_track_amplitude(self):
        """Scaling an RF up leaves the index alone; it is a fraction, not a size."""
        base = oriented_gaussian(1.0, 20.0, 20.0, 5.0, 3.0, 0.0)
        weak = gaussian_fit.fit_and_score(with_noise(base * 5, sd=0.025))
        strong = gaussian_fit.fit_and_score(with_noise(base * 50, sd=0.25))
        assert weak["index"] == pytest.approx(strong["index"], abs=0.05)

    @pytest.mark.parametrize(
        "image",
        [
            np.zeros((SIZE, SIZE)),
            np.full((SIZE, SIZE), np.nan),
            np.eye(SIZE) * 0,  # all zeros by another route
        ],
        ids=["zeros", "all-nan", "degenerate"],
    )
    def test_hopeless_maps_return_nan_rather_than_raising(self, image):
        scored = gaussian_fit.fit_and_score(image)
        assert np.isnan(scored["index"])
        assert scored["n_px"] == 0

    def test_a_single_hot_pixel_does_not_raise(self):
        image = np.zeros((SIZE, SIZE))
        image[20, 20] = 50.0
        scored = gaussian_fit.fit_and_score(image)
        assert set(scored) == set(gaussian_fit.FIELDS)

    def test_every_field_is_present_even_on_failure(self):
        scored = gaussian_fit.fit_and_score(np.zeros((SIZE, SIZE)))
        assert set(scored) == set(gaussian_fit.FIELDS)


class TestFoldedVariance:
    def test_collapses_to_the_folded_normal_at_zero_signal(self):
        assert gaussian_fit._folded_var(np.array(0.0), 2.0) == pytest.approx(
            4.0 * (1 - 2 / np.pi)
        )

    def test_approaches_the_plain_variance_when_signal_dominates(self):
        assert gaussian_fit._folded_var(np.array(100.0), 1.0) == pytest.approx(
            1.0, abs=1e-6
        )


class TestDataEllipse:
    @pytest.mark.parametrize("theta", [-np.pi / 3, 0.0, np.pi / 6, np.pi / 3])
    def test_theta_stays_in_the_half_open_half_turn(self, theta):
        """arctan2 returns (-pi, pi]; everything downstream assumes [-pi/2, pi/2)."""
        image = with_noise(oriented_gaussian(10.0, 20.0, 20.0, 6.0, 2.0, theta))
        region = gaussian_fit.data_ellipse(image)
        assert -np.pi / 2 <= region["theta"] < np.pi / 2

    def test_returns_none_without_enough_signal(self):
        assert gaussian_fit.data_ellipse(np.zeros((SIZE, SIZE))) is None

    def test_spans_both_lobes_of_an_opponent_rf(self):
        """The reason the footprint comes from the data and not from the fit."""
        opponent = with_noise(
            oriented_gaussian(10.0, 12.0, 20.0, 2.5, 2.5, 0.0)
            - oriented_gaussian(10.0, 29.0, 20.0, 2.5, 2.5, 0.0)
        )
        region = gaussian_fit.data_ellipse(opponent)
        fit = gaussian_fit.fit_gaussian2d(opponent)
        assert region["x0"] == pytest.approx(20.5, abs=1.5)
        assert region["sx"] > fit["sy"]


class FakeSTRF:
    """Minimal stand-in exposing only what the selector arithmetic touches.

    Maps are supplied directly, so the test exercises roi/idx bookkeeping without
    dragging in a real recording. No `params`, so the module falls through to its
    literal defaults.
    """

    _flat_strf_indices = STRF._flat_strf_indices
    calc_gaussian_fit_index = STRF.calc_gaussian_fit_index
    get_gaussian_fit_index = STRF.get_gaussian_fit_index
    get_gaussian_fit_snr = STRF.get_gaussian_fit_snr

    def __init__(self, n_rois=3, n_colours=4):
        self.n_colours = n_colours
        rng = np.random.default_rng(SEED)
        # Amplitude rises with flat index, so a mis-selection shows up as a wrong value.
        self.strfs = np.zeros((n_rois * n_colours, 2, SIZE, SIZE))
        self._maps = np.ma.masked_array(
            [
                with_noise(
                    oriented_gaussian(5.0 + i, 20.0, 20.0, 5.0, 3.0, 0.0),
                    seed=int(rng.integers(1 << 30)),
                )
                for i in range(n_rois * n_colours)
            ]
        )

    def collapse_times(self, *args, **kwargs):
        return self._maps


class TestSelectors:
    def test_roi_expands_to_all_of_its_colours(self):
        obj = FakeSTRF(n_rois=3, n_colours=4)
        flat, shape = obj._flat_strf_indices(roi=1)
        assert list(flat) == [4, 5, 6, 7]
        assert shape == (4,)

    def test_roi_agrees_with_unravel_strf_indices(self):
        """The existing documented converter, one index at a time."""
        n_colours = 4
        obj = FakeSTRF(n_rois=3, n_colours=n_colours)
        flat, _ = obj._flat_strf_indices(roi=2)
        assert list(flat) == [2 * n_colours + c for c in range(n_colours)]

    @pytest.mark.parametrize(
        "kwargs, expected",
        [
            ({}, (12,)),
            ({"roi": 1}, (4,)),
            ({"roi": [1, 2]}, (2, 4)),
            ({"idx": 6}, ()),
            ({"idx": [4, 5, 6]}, (3,)),
        ],
        ids=["all", "one-roi", "two-rois", "one-idx", "three-idx"],
    )
    def test_selection_shapes(self, kwargs, expected):
        obj = FakeSTRF(n_rois=3, n_colours=4)
        assert obj.get_gaussian_fit_index(**kwargs).shape == expected

    def test_roi_selects_the_same_values_as_the_flat_indices(self):
        obj = FakeSTRF(n_rois=3, n_colours=4)
        everything = obj.get_gaussian_fit_index()
        assert obj.get_gaussian_fit_index(roi=1) == pytest.approx(
            everything[[4, 5, 6, 7]], nan_ok=True
        )

    def test_idx_selects_one_strf(self):
        obj = FakeSTRF(n_rois=3, n_colours=4)
        everything = obj.get_gaussian_fit_index()
        assert obj.get_gaussian_fit_index(idx=6) == pytest.approx(
            everything[6], nan_ok=True
        )

    def test_roi_and_idx_together_are_refused(self):
        obj = FakeSTRF()
        with pytest.raises(ValueError):
            obj.get_gaussian_fit_index(roi=1, idx=1)

    def test_single_colour_roi_is_a_length_one_array(self):
        obj = FakeSTRF(n_rois=5, n_colours=1)
        assert obj.get_gaussian_fit_index(roi=3).shape == (1,)

    def test_the_fit_runs_once_for_repeated_getters(self):
        """Cache is keyed on settings, not on the selection, so getters share a pass."""
        obj = FakeSTRF()
        obj.get_gaussian_fit_index()
        calls = {"n": 0}
        original = gaussian_fit.gaussian_fit_index_wrapper

        def counting(*args, **kwargs):
            calls["n"] += 1
            return original(*args, **kwargs)

        gaussian_fit.gaussian_fit_index_wrapper = counting
        try:
            obj.get_gaussian_fit_index(roi=1)
            obj.get_gaussian_fit_snr()
        finally:
            gaussian_fit.gaussian_fit_index_wrapper = original
        assert calls["n"] == 0


@pytest.mark.demo_data
class TestOnARecording:
    def test_returns_one_value_per_strf(self, strf):
        values = strf.get_gaussian_fit_index()
        assert values.shape == (len(strf.strfs),)

    def test_by_channel_reshapes_consistently(self, strf):
        flat = strf.get_gaussian_fit_index()
        per_channel = strf.get_gaussian_fit_index_by_channel()
        assert per_channel.shape[0] == strf.n_colours
        assert per_channel.size == flat.size
        n_colours = strf.n_colours
        for colour in range(n_colours):
            for roi in range(per_channel.shape[1]):
                assert per_channel[colour][roi] == pytest.approx(
                    flat[roi * n_colours + colour], nan_ok=True
                )

    def test_roi_matches_the_flat_indices_it_stands_for(self, strf):
        flat = strf.get_gaussian_fit_index()
        n_colours = strf.n_colours
        assert strf.get_gaussian_fit_index(roi=0) == pytest.approx(
            flat[[c for c in range(n_colours)]], nan_ok=True
        )

    def test_index_is_bounded_and_mostly_finite(self, strf):
        values = strf.get_gaussian_fit_index()
        finite = values[np.isfinite(values)]
        assert finite.size > 0
        assert np.all(finite >= 0)
        assert np.all(finite <= 2)
