"""Unit tests for pygor.review.index, using synthetic saved recordings.

The files are written by hand rather than through ``Core.save_object`` so these
run on a fresh clone with no demo recording, and so a test can construct the
awkward cases (a lossy ROI transfer, a missing dataset) directly.
"""

import json
import pathlib

import h5py
import numpy as np
import pandas as pd
import pytest

from pygor.review import index as review_index


def write_recording(
    root,
    condition="control",
    session="240101 test",
    stem="0_0_SWN_200",
    *,
    num_rois=4,
    n_colours=4,
    roi_origin=None,
    with_strfs=True,
    n_triggers=12,
):
    """A minimal ``.recording.h5`` with the attrs the scanner reads."""
    directory = pathlib.Path(root) / "Processed" / condition / session
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{stem}.recording.h5"
    if roi_origin is None:
        roi_origin = {"method": "segmented", "mode": "blob", "source": stem}

    with h5py.File(path, "w") as handle:
        handle.attrs["__num_recordings__"] = 1
        group = handle.create_group("recording_000")
        group.attrs["__class_name__"] = "STRF"
        group.attrs["num_rois"] = num_rois
        group.attrs["n_colours"] = n_colours
        group.attrs["num_strfs"] = num_rois * n_colours
        group.attrs["frame_hz"] = 15.625
        group.attrs["trigger_mode"] = 1
        group.attrs["name"] = stem
        group.attrs["filename"] = f"/raw/{condition}/{session}/{stem}.smp"
        group.attrs["_type_roi_origin"] = "json_dict"
        group.attrs["roi_origin"] = json.dumps(roi_origin)
        group.attrs["_type_params"] = "AnalysisParams"
        group.attrs["params"] = json.dumps(
            {"registration": {"mean_error": 0.35, "max_shift": [5.2, 3.6]}}
        )
        group.attrs["_type_metadata"] = "json_dict"
        group.attrs["metadata"] = json.dumps(
            {"filename": stem, "exp_date": {"__date__": "2024-01-01"}}
        )

        group.create_dataset("rois", data=np.full((8, 8), 1, dtype=np.int16))
        group.create_dataset("average_stack", data=np.zeros((8, 8), dtype=np.float32))
        group.create_dataset("images", data=np.zeros((5, 8, 8), dtype=np.float32))
        group.create_dataset("triggertimes", data=np.arange(n_triggers, dtype=float))
        if with_strfs:
            strfs = group.create_group("strfs")
            strfs.create_dataset(
                "data", data=np.zeros((num_rois * n_colours, 6, 4, 4), dtype=np.float32)
            )
    return path


@pytest.fixture
def dataset(tmp_path):
    write_recording(tmp_path, stem="0_0_SWN_200")
    write_recording(
        tmp_path,
        stem="0_0_OSDS_2x_vel",
        roi_origin={
            "method": "transferred",
            "source": "0_0_SWN_200",
            "correlation": 0.41,
            "expected_roi_ids": [-1, -2, -3, -4],
            "lost_roi_ids": [-3],
        },
        num_rois=3,
    )
    return tmp_path


class TestScanRecording:
    def test_reads_identity_and_provenance(self, dataset):
        ref = review_index.scan_recording(
            dataset / "Processed" / "control" / "240101 test" / "0_0_SWN_200.recording.h5"
        )
        assert ref.recording_uid == "240101 test::0_0_SWN_200"
        assert ref.condition == "control"
        assert ref.session == "240101 test"
        assert ref.stem == "0_0_SWN_200"
        assert ref.class_name == "STRF"
        assert ref.num_rois == 4
        assert ref.n_colours == 4
        assert ref.n_triggers == 12
        assert ref.exp_date == "2024-01-01"
        assert ref.registration["mean_error"] == pytest.approx(0.35)
        assert ref.shapes["strfs"] == [16, 6, 4, 4]

    def test_transfer_fields(self, dataset):
        ref = review_index.scan_recording(
            dataset / "Processed" / "control" / "240101 test" / "0_0_OSDS_2x_vel.recording.h5"
        )
        assert ref.transferred
        assert ref.transfer_correlation == pytest.approx(0.41)
        assert ref.lost_roi_ids == [-3]

    def test_segmented_transfer_correlation_is_nan(self, dataset):
        ref = review_index.scan_recording(
            dataset / "Processed" / "control" / "240101 test" / "0_0_SWN_200.recording.h5"
        )
        assert not ref.transferred
        assert np.isnan(ref.transfer_correlation)

    def test_never_reads_array_data(self, dataset, monkeypatch):
        """The whole point of the scanner: shapes and attrs only.

        If this starts failing, scanning a 270-file dataset has quietly gone
        from seconds to minutes.
        """

        def forbidden(self, key):
            raise AssertionError(f"scanner read array data from {self.name}")

        monkeypatch.setattr(h5py.Dataset, "__getitem__", forbidden)
        ref = review_index.scan_recording(
            dataset / "Processed" / "control" / "240101 test" / "0_0_SWN_200.recording.h5"
        )
        assert ref.num_rois == 4

    def test_malformed_json_attr_warns_rather_than_raising(self, tmp_path):
        path = write_recording(tmp_path)
        with h5py.File(path, "r+") as handle:
            handle["recording_000"].attrs["roi_origin"] = "{not json"
        with pytest.warns(UserWarning, match="roi_origin"):
            ref = review_index.scan_recording(path)
        assert ref.roi_origin == {}

    def test_missing_recording_group_raises(self, tmp_path):
        path = tmp_path / "empty.recording.h5"
        with h5py.File(path, "w"):
            pass
        with pytest.raises(ValueError, match="no recording group"):
            review_index.scan_recording(path)


class TestScanProcessed:
    def test_finds_every_recording(self, dataset):
        refs = review_index.scan_processed(dataset, cache=False)
        assert {r.stem for r in refs} == {"0_0_SWN_200", "0_0_OSDS_2x_vel"}

    def test_cache_is_reused_and_invalidated(self, dataset):
        first = review_index.scan_processed(dataset)
        cache = dataset / "review" / review_index.INDEX_CACHE_NAME
        assert cache.exists()

        # A second scan must not re-read the files it already knows about.
        def forbidden(*args, **kwargs):
            raise AssertionError("rescanned an unchanged file")

        original = review_index.scan_recording
        review_index.scan_recording = forbidden
        try:
            second = review_index.scan_processed(dataset)
        finally:
            review_index.scan_recording = original
        assert {r.recording_uid for r in second} == {r.recording_uid for r in first}

        # Touching a file must invalidate just that row.
        target = first[0].path
        stat = target.stat()
        import os

        os.utime(target, (stat.st_atime, stat.st_mtime + 10))
        third = review_index.scan_processed(dataset)
        changed = [r for r in third if r.path == target][0]
        assert changed.mtime != first[0].mtime

    def test_empty_root_is_empty_not_an_error(self, tmp_path):
        assert review_index.scan_processed(tmp_path, cache=False) == []


class TestStatus:
    def _ledger(self, root, rows):
        path = root / "Processed" / "status.csv"
        path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(rows).to_csv(path, index=False)
        return path

    def test_attach_status_merges_on_condition_and_uid(self, dataset):
        path = self._ledger(
            dataset,
            [
                {
                    "condition": "control",
                    "recording_uid": "240101 test::0_0_SWN_200",
                    "status": "ok",
                    "reason": np.nan,
                },
                {
                    "condition": "control",
                    "recording_uid": "240101 test::0_0_OSDS_2x_vel",
                    "status": "skipped",
                    "reason": "RoiCountMismatch: master=4 fff=3",
                },
            ],
        )
        refs = review_index.attach_status(
            review_index.scan_processed(dataset, cache=False), path
        )
        by_uid = {r.recording_uid: r for r in refs}
        assert by_uid["240101 test::0_0_SWN_200"].status == "ok"
        assert by_uid["240101 test::0_0_SWN_200"].status_reason == ""
        assert by_uid["240101 test::0_0_OSDS_2x_vel"].status == "skipped"
        assert "RoiCountMismatch" in by_uid["240101 test::0_0_OSDS_2x_vel"].status_reason

    def test_unlisted_recording_stays_unknown(self, dataset):
        path = self._ledger(
            dataset,
            [{"condition": "control", "recording_uid": "other::x", "status": "ok", "reason": ""}],
        )
        refs = review_index.attach_status(
            review_index.scan_processed(dataset, cache=False), path
        )
        assert all(r.status == "unknown" for r in refs)

    def test_missing_from_disk_finds_ledger_only_rows(self, dataset):
        """A recording that failed processing has no file, so a scan cannot see it."""
        path = self._ledger(
            dataset,
            [
                {
                    "condition": "control",
                    "recording_uid": "240101 test::0_0_SWN_200",
                    "status": "ok",
                    "reason": "",
                },
                {
                    "condition": "control",
                    "recording_uid": "240101 test::never_made_it",
                    "status": "failed",
                    "reason": "boom",
                },
            ],
        )
        refs = review_index.scan_processed(dataset, cache=False)
        missing = review_index.missing_from_disk(refs, path)
        assert list(missing.recording_uid) == ["240101 test::never_made_it"]

    def test_no_ledger_is_tolerated(self, dataset):
        refs = review_index.scan_processed(dataset, cache=False)
        assert review_index.attach_status(refs, dataset / "nope.csv") == refs
        assert review_index.missing_from_disk(refs, dataset / "nope.csv").empty


class TestReadArrays:
    def test_reads_small_arrays(self, dataset):
        path = dataset / "Processed" / "control" / "240101 test" / "0_0_SWN_200.recording.h5"
        arrays = review_index.read_arrays(path, keys=("rois", "average_stack"))
        assert arrays["rois"].shape == (8, 8)
        assert arrays["average_stack"].shape == (8, 8)

    def test_refuses_large_arrays(self, dataset):
        """`images` here would undo the reason this module exists."""
        path = dataset / "Processed" / "control" / "240101 test" / "0_0_SWN_200.recording.h5"
        with pytest.raises(ValueError, match="SMALL_ARRAYS"):
            review_index.read_arrays(path, keys=("images",))

    def test_absent_dataset_is_omitted_not_an_error(self, dataset):
        path = dataset / "Processed" / "control" / "240101 test" / "0_0_SWN_200.recording.h5"
        arrays = review_index.read_arrays(path, keys=("rois", "ipl_depths"))
        assert "rois" in arrays
        assert "ipl_depths" not in arrays


class TestAsDataFrame:
    def test_flattens_nested_provenance(self, dataset):
        frame = review_index.as_dataframe(
            review_index.scan_processed(dataset, cache=False)
        )
        assert set(frame.roi_method) == {"segmented", "transferred"}
        transferred = frame[frame.roi_method == "transferred"].iloc[0]
        assert transferred.transfer_correlation == pytest.approx(0.41)
        assert transferred.n_lost_rois == 1
        assert transferred.reg_max_shift == pytest.approx(5.2)

    def test_empty_is_empty_frame(self):
        assert review_index.as_dataframe([]).empty
