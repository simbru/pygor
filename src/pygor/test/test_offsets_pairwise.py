"""Unit tests for pairwise colour-channel offsets, using synthetic centres only.

These need no recording, so they run on a fresh clone. The point of the pairwise
estimator is that a pair's separation does not depend on how many other channels
were segmented, which is what most of these check.
"""

import numpy as np
import pytest

from pygor.classes.strf_data import STRF


class FakeSTRF:
    """Minimal stand-in exposing only what the pairwise method touches.

    Centres are supplied directly, so the test exercises the offset arithmetic
    without dragging in segmentation or a real recording.
    """

    calc_colour_channel_offsets_pairwise = STRF.calc_colour_channel_offsets_pairwise
    calc_colour_channel_offsets = STRF.calc_colour_channel_offsets
    get_colour_channel_offsets_pairwise_distances = (
        STRF.get_colour_channel_offsets_pairwise_distances
    )

    def __init__(self, centres, stim_size=1.0):
        # centres: (n_cells, n_colours, 2) in (y, x)
        self._centres = np.asarray(centres, dtype=float)
        self.n_colours = self._centres.shape[1]
        self.multicolour = self.n_colours > 1
        self.stim_size = stim_size

    def get_seg_centres(self, channel_reshape=False, **kwargs):
        if channel_reshape:
            return self._centres.copy()
        return self._centres.reshape(-1, 2).copy()

    def get_seg_centres_by_channel(self, **kwargs):
        return np.transpose(self._centres, (1, 0, 2)).copy()


@pytest.fixture
def three_apart():
    """One cell: Ch0 at the origin, Ch1 three pixels along x, Ch2/Ch3 on top of Ch0."""
    return FakeSTRF([[[0.0, 0.0], [0.0, 3.0], [0.0, 0.0], [0.0, 0.0]]])


class TestPairwiseDistances:
    def test_separation_is_the_distance_between_the_two_centres(self, three_apart):
        res = three_apart.calc_colour_channel_offsets_pairwise()
        d = dict(zip(res["pair_names"], res["distances"][:, 0]))
        assert d["Ch0-Ch1"] == pytest.approx(3.0)
        assert d["Ch0-Ch2"] == pytest.approx(0.0)
        assert d["Ch2-Ch3"] == pytest.approx(0.0)

    def test_pair_order_and_names_line_up(self, three_apart):
        res = three_apart.calc_colour_channel_offsets_pairwise()
        assert res["pairs"] == [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]
        assert res["pair_names"][0] == "Ch0-Ch1"
        assert res["distances"].shape == (6, 1)

    def test_scaled_into_visual_angle(self):
        obj = FakeSTRF([[[0.0, 0.0], [0.0, 2.0]]], stim_size=2.371)
        res = obj.calc_colour_channel_offsets_pairwise()
        assert res["distances"][0, 0] == pytest.approx(2 * 2.371)

    def test_missing_centre_gives_nan_only_for_its_own_pairs(self):
        obj = FakeSTRF([[[0.0, 0.0], [0.0, 3.0], [np.nan, np.nan]]])
        res = obj.calc_colour_channel_offsets_pairwise()
        d = dict(zip(res["pair_names"], res["distances"][:, 0]))
        assert d["Ch0-Ch1"] == pytest.approx(3.0)
        assert np.isnan(d["Ch0-Ch2"])
        assert np.isnan(d["Ch1-Ch2"])

    def test_channel_count_does_not_change_a_pair(self):
        """The whole reason this estimator exists.

        The same two channels sit 3 px apart in both objects; the second just has
        two extra channels stacked on Ch0. calc_colour_channel_offsets measures
        against the mean of all of them, so its magnitudes move; pairwise does not.
        """
        two = FakeSTRF([[[0.0, 0.0], [0.0, 3.0]]])
        four = FakeSTRF([[[0.0, 0.0], [0.0, 3.0], [0.0, 0.0], [0.0, 0.0]]])

        def pair01(obj):
            res = obj.calc_colour_channel_offsets_pairwise()
            return res["distances"][res["pair_names"].index("Ch0-Ch1"), 0]

        assert pair01(two) == pytest.approx(pair01(four))
        assert pair01(two) == pytest.approx(3.0)

        # The channel-mean estimator, for contrast: Ch1 keeps half the separation
        # with two channels and three quarters with four.
        assert two.calc_colour_channel_offsets()["magnitudes"][1, 0] == pytest.approx(1.5)
        assert four.calc_colour_channel_offsets()["magnitudes"][1, 0] == pytest.approx(2.25)


class TestPairwiseAngles:
    def test_direction_runs_from_the_first_channel_to_the_second(self):
        obj = FakeSTRF([[[0.0, 0.0], [0.0, 3.0]]])
        res = obj.calc_colour_channel_offsets_pairwise()
        assert res["angles"][0, 0] == pytest.approx(0.0)   # +x
        res = obj.calc_colour_channel_offsets_pairwise(angle_range_360=False)
        assert res["angles"][0, 0] == pytest.approx(0.0)

    def test_perpendicular_pair(self):
        obj = FakeSTRF([[[0.0, 0.0], [4.0, 0.0]]])
        assert obj.calc_colour_channel_offsets_pairwise()["angles"][0, 0] == pytest.approx(90.0)

    def test_negative_angles_wrap_when_asked(self):
        obj = FakeSTRF([[[0.0, 0.0], [-4.0, 0.0]]])
        assert obj.calc_colour_channel_offsets_pairwise()["angles"][0, 0] == pytest.approx(270.0)
        assert obj.calc_colour_channel_offsets_pairwise(angle_range_360=False)[
            "angles"
        ][0, 0] == pytest.approx(-90.0)

    def test_coincident_centres_have_no_direction(self, three_apart):
        res = three_apart.calc_colour_channel_offsets_pairwise()
        assert np.isnan(res["angles"][res["pair_names"].index("Ch0-Ch2"), 0])


class TestSelectionAndErrors:
    def test_channels_argument_restricts_the_pairs(self):
        obj = FakeSTRF([[[0.0, 0.0], [0.0, 3.0], [0.0, 9.0], [0.0, 27.0]]])
        res = obj.calc_colour_channel_offsets_pairwise(channels=[0, 1])
        assert res["pair_names"] == ["Ch0-Ch1"]
        assert res["distances"][0, 0] == pytest.approx(3.0)

    def test_pair_indices_are_positions_in_the_channels_list(self):
        obj = FakeSTRF([[[0.0, 0.0], [0.0, 3.0], [0.0, 9.0]]])
        res = obj.calc_colour_channel_offsets_pairwise(channels=[1, 2])
        assert res["pair_names"] == ["Ch0-Ch1"]
        assert res["distances"][0, 0] == pytest.approx(6.0)

    def test_roi_selection_keeps_the_pair_axis(self):
        obj = FakeSTRF([[[0.0, 0.0], [0.0, 1.0]], [[0.0, 0.0], [0.0, 5.0]]])
        res = obj.calc_colour_channel_offsets_pairwise(roi=1)
        assert res["distances"].shape == (1, 1)
        assert res["distances"][0, 0] == pytest.approx(5.0)

    def test_out_of_range_roi_raises(self):
        obj = FakeSTRF([[[0.0, 0.0], [0.0, 1.0]]])
        with pytest.raises(ValueError, match="out of range"):
            obj.calc_colour_channel_offsets_pairwise(roi=4)

    def test_out_of_range_channel_raises(self):
        obj = FakeSTRF([[[0.0, 0.0], [0.0, 1.0]]])
        with pytest.raises(ValueError, match="out of range"):
            obj.calc_colour_channel_offsets_pairwise(channels=[0, 7])

    def test_single_channel_selection_raises(self):
        obj = FakeSTRF([[[0.0, 0.0], [0.0, 1.0]]])
        with pytest.raises(ValueError, match="at least 2 channels"):
            obj.calc_colour_channel_offsets_pairwise(channels=[0])

    def test_monochrome_object_raises(self):
        obj = FakeSTRF([[[0.0, 0.0]]])
        with pytest.raises(ValueError, match="multicolour"):
            obj.calc_colour_channel_offsets_pairwise()

    def test_getter_returns_the_distances(self, three_apart):
        got = three_apart.get_colour_channel_offsets_pairwise_distances()
        assert got.shape == (6, 1)
        assert got[0, 0] == pytest.approx(3.0)


class TestAgreementWithChannelMeanEstimator:
    """Both estimators read the same centres, so they must agree where comparable."""

    def test_two_channels_split_the_separation_evenly(self):
        obj = FakeSTRF([[[0.0, 0.0], [0.0, 4.0]]])
        pair = obj.calc_colour_channel_offsets_pairwise()["distances"][0, 0]
        mags = obj.calc_colour_channel_offsets()["magnitudes"][:, 0]
        assert mags[0] == pytest.approx(mags[1])
        assert pair == pytest.approx(mags.sum())

    def test_reconstructing_pairwise_from_channel_mean_vectors_agrees(self):
        """The route used to rescue pairwise separations from an existing export.

        Offset vectors about the shared centre subtract to the pairwise vector, so
        |v_i - v_j| has to match the direct pairwise distance.
        """
        rng = np.random.default_rng(0)
        obj = FakeSTRF(rng.normal(size=(6, 4, 2)) * 5)
        direct = obj.calc_colour_channel_offsets_pairwise()
        mean_based = obj.calc_colour_channel_offsets()
        mags, angles = mean_based["magnitudes"], np.radians(mean_based["angles"])
        vec = np.stack([mags * np.cos(angles), mags * np.sin(angles)], axis=-1)
        for pair_idx, (i, j) in enumerate(direct["pairs"]):
            expected = np.linalg.norm(vec[j] - vec[i], axis=-1)
            np.testing.assert_allclose(direct["distances"][pair_idx], expected, atol=1e-9)
