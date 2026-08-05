"""Integration tests for the Core class against the demo recording."""

import numpy as np
import pytest

import pygor.load
import pygor.utils.helpinfo
from pygor.test.helpers import callable_without_arguments

pytestmark = pytest.mark.demo_data

# Errors that mean an internal name went stale, as opposed to a method
# deliberately refusing the arguments it was given.
PLUMBING_ERRORS = (AttributeError, NameError, UnboundLocalError)

# Methods that write next to the source file, so calling them blind would
# clobber the demo recording. They are covered by the export tests instead.
WRITES_TO_SOURCE_DIR = {"export_to_h5", "save", "save_object"}

SMOKE_METHODS = callable_without_arguments(
    pygor.load.Core, exclude=WRITES_TO_SOURCE_DIR
)


@pytest.mark.parametrize("method_name", SMOKE_METHODS)
def test_method_is_wired_up(scratch_core, method_name):
    """Every no-argument method either returns or raises deliberately.

    This does not check the answer is right, only that the call reaches its own
    code. It is what catches an attribute renamed in one place and not another,
    which is the failure mode that keeps showing up here.
    """
    try:
        getattr(scratch_core, method_name)()
    except PLUMBING_ERRORS as e:
        pytest.fail(f"{method_name}() hit a stale name: {type(e).__name__}: {e}")
    except Exception as e:
        assert str(e), f"{method_name}() raised {type(e).__name__} with no message"


def test_attributes_readable(core):
    for name in pygor.utils.helpinfo.get_attribute_list(core, with_types=False):
        getattr(core, name)


def test_required_attributes_present(core):
    for attr in ("filename", "metadata", "rois", "type", "frame_hz", "num_rois"):
        assert hasattr(core, attr), f"Missing required attribute: {attr}"


def test_metadata_is_dict(core):
    assert isinstance(core.metadata, dict)


def test_num_rois_matches_mask(core):
    """num_rois should count the ROI labels in the mask.

    ROI labels are negative integers and the background is one further label,
    hence the -1 (see the ROI convention in CLAUDE.md).
    """
    labels = np.unique(core.rois)
    labels = labels[~np.isnan(labels)]
    assert len(labels) - 1 == core.num_rois


def test_rois_use_negative_labels(core):
    labels = np.unique(core.rois)
    labels = labels[~np.isnan(labels)]
    assert (labels < 0).sum() == core.num_rois, (
        "ROIs are identified by negative integers; background is non-negative"
    )


def test_images_are_a_time_stack(core):
    if core.images is None:
        pytest.skip("recording has no image stack")
    assert core.images.ndim == 3, "images should be [time, y, x]"
    assert core.images.shape[0] > 0


def test_averages_appear_after_snippets(fresh_core):
    """A freshly loaded recording has no averages until snippets are computed."""
    assert fresh_core.averages is None
    fresh_core.compute_snippets_and_averages()
    assert fresh_core.averages.shape[0] == fresh_core.num_rois


def test_frame_hz_is_plausible(core):
    assert 0 < core.frame_hz < 1000


def test_triggertimes_increase(core):
    if core.triggertimes is None or len(core.triggertimes) < 2:
        pytest.skip("recording has fewer than two triggers")
    assert np.all(np.diff(core.triggertimes) >= 0)


def test_triggertimes_frame_are_whole_numbers(core):
    if core.triggertimes_frame is None:
        pytest.skip("recording has no frame-indexed triggers")
    frames = core.triggertimes_frame
    assert np.all(frames == frames.astype(int))


def test_get_help_writes_to_stdout(core, capsys):
    core.get_help(hints=True, types=True)
    assert capsys.readouterr().out.strip(), "get_help() printed nothing"
