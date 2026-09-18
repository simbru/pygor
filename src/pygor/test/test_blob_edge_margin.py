"""The blob segmenter's response to the masked-artefact edge.

The artefact region is zero-filled before detection, which leaves a hard step
from zero to signal. A difference of Gaussians responds to a step as strongly
as to a cell, so a column of false blobs forms just inside the mask boundary;
on one field of view they were 15 of 72 ROIs. Widening the mask moves the
column rather than removing it. ``edge_margin`` drops blobs within a band of
the boundary instead.
"""

import numpy as np
import pytest

from pygor.segmentation.blob.segment import segment as blob_segment_fn
from pygor.segmentation.blob.segment import _detect_blobs


def centroid_columns(masks):
    """x of each ROI's centroid; the filter acts on centres, not painted pixels."""
    return [np.nonzero(masks == label)[1].mean() for label in np.unique(masks) if label > 0]


@pytest.fixture
def stepped_image():
    """Signal everywhere except a zeroed strip, plus one real cell.

    The zeroed strip is what artefact masking produces; the step it creates is
    what DoG mistakes for cells.
    """
    rng = np.random.default_rng(0)
    image = 0.4 + 0.02 * rng.standard_normal((40, 80))
    yy, xx = np.mgrid[:40, :80]
    image += 0.6 * np.exp(-((yy - 20) ** 2 + (xx - 50) ** 2) / (2 * 1.5**2))
    image[:, :5] = 0.0
    return np.clip(image, 0, 1).astype(np.float32)


class TestEdgeMargin:
    def test_the_mask_edge_produces_false_blobs(self, stepped_image):
        blobs = _detect_blobs(stepped_image, min_sigma=1, max_sigma=2, threshold=0.05,
                              overlap=1.0, artifact_width=5)
        assert (blobs[:, 1] < 9).sum() > 0, "expected DoG to respond to the step"

    def test_margin_removes_them_and_keeps_the_cell(self, stepped_image):
        blobs = _detect_blobs(stepped_image, min_sigma=1, max_sigma=2, threshold=0.05,
                              overlap=1.0, artifact_width=5, edge_margin=5)
        assert (blobs[:, 1] < 10).sum() == 0
        assert any(abs(x - 50) < 3 for x in blobs[:, 1]), "the real cell must survive"

    def test_margin_is_ignored_without_an_artefact(self, stepped_image):
        """No masked region means no edge to guard, so nothing near x=0 is dropped."""
        image = stepped_image.copy()
        image[:, :5] = 0.4
        with_margin = _detect_blobs(image, min_sigma=1, max_sigma=2, threshold=0.05,
                                    overlap=1.0, artifact_width=0, edge_margin=5)
        without = _detect_blobs(image, min_sigma=1, max_sigma=2, threshold=0.05,
                                overlap=1.0, artifact_width=0)
        assert len(with_margin) == len(without)

    def test_segment_accepts_edge_margin(self, stepped_image):
        masks = blob_segment_fn(stepped_image, threshold=0.05, artifact_width=5,
                                     edge_margin=5, anatomy_threshold=None,
                                     verbose=False)
        assert masks.max() >= 1
        assert min(centroid_columns(masks)) >= 10

    def test_artifact_width_override_reaches_a_data_object(self, stepped_image):
        """The kwarg used to be honoured only for raw arrays."""

        class FakeRecording:
            images = np.repeat(stepped_image[None], 3, axis=0)
            average_stack = None
            correlation_projection = None

            class params:
                artifact_width = 0

                @staticmethod
                def get_defaults(section):
                    return {}

        masks = blob_segment_fn(FakeRecording(), input_mode="average",
                                     threshold=0.05, artifact_width=8,
                                     anatomy_threshold=None, verbose=False)
        assert min(centroid_columns(masks)) >= 9
