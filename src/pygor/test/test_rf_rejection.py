"""Rejected RFs: stored on the recording, and verdict staleness kept to its role."""

import pathlib
import types

import numpy as np
import pytest

from pygor.review.exclusions import covered_verdict
from pygor.review.verdicts import Verdict
from pygor.strf.rejection import is_rejected, rejected, set_rejected


class TestRejection:
    def test_a_recording_without_the_attribute_rejects_nothing(self):
        assert rejected(types.SimpleNamespace()).size == 0

    def test_reject_and_restore(self):
        rec = types.SimpleNamespace()
        set_rejected(rec, 41)
        set_rejected(rec, 3)
        set_rejected(rec, 41)  # twice is once
        assert rejected(rec).tolist() == [3, 41]
        assert is_rejected(rec, 3) and not is_rejected(rec, 4)
        set_rejected(rec, 3, reject=False)
        assert rejected(rec).tolist() == [41]


FOV = "s::gb_ablation::8_2"
SAVED = {"osds": "1:10", "swn": "1:20", "fff": "1:30"}


def fov_keep():
    return Verdict(subject_type="fov", subject_uid=FOV, check="alignment",
                   verdict="keep", condition="gb_ablation", fov_uid=FOV,
                   source_digest=dict(SAVED))


def digests(**changed):
    now = {**SAVED, **changed}
    return {(FOV, "gb_ablation", role): digest for role, digest in now.items()}


class TestDirectionVerdictStaleness:
    def test_unchanged_files_keep_the_verdict(self):
        assert covered_verdict(fov_keep(), None, "osds", digests()) == "keep"

    def test_resaving_the_noise_recording_leaves_direction_qc_alone(self):
        # e.g. after rejecting single RFs in the SWN recording
        assert covered_verdict(fov_keep(), None, "osds", digests(swn="2:21")) == "keep"

    def test_a_reprocessed_direction_recording_still_invalidates_it(self):
        assert covered_verdict(fov_keep(), None, "osds", digests(osds="2:11")) == ""


def test_gui_labels_map_through_the_mask_not_by_one():
    from pygor.strf.rejection import label_to_roi, roi_to_label

    # IGOR-style ids -1, -2, -5: label 5 is the third ROI, not the fifth
    rois = np.ones((4, 4), dtype=int)
    rois[0, 0], rois[1, 1], rois[2, 2] = -1, -2, -5
    rec = types.SimpleNamespace(rois=rois, strfs=np.zeros((3, 2, 2, 2)), n_colours=1)
    assert label_to_roi(rec, 5) == 2 and roi_to_label(rec, 2) == 5
    with pytest.raises(ValueError):
        label_to_roi(rec, 3)


SAVED_FILE = pathlib.Path(
    "/mnt/data/BC paper data/Achromatic SWN/Processed/gb_ablation/261001 GB abl/"
    "8_2_SWN_200_Achrom.recording.h5"
)


@pytest.mark.skipif(not SAVED_FILE.exists(), reason="BC paper data not mounted")
def test_command_line_rejects_by_label_saves_and_draws(tmp_path, capsys):
    import shutil

    from pygor.classes.core_data import Core
    from pygor.strf.rejection import main, roi_to_label

    # A copy: the real file holds the curated rejections
    path = tmp_path / SAVED_FILE.name
    shutil.copy(SAVED_FILE, path)
    before = rejected(Core.load_object(path)).tolist()
    free = [r for r in range(5) if r not in before][:2]
    labels = [roi_to_label(Core.load_object(path), r) for r in free]

    main([str(path), "--reject", *map(str, labels), "--sheet", str(tmp_path / "s.png")])
    assert rejected(Core.load_object(path)).tolist() == sorted(before + free)
    assert (tmp_path / "s.png").stat().st_size > 0

    main([str(path), "--restore", str(labels[0])])
    assert rejected(Core.load_object(path)).tolist() == sorted(before + free[1:])

    # A bad label changes nothing
    with pytest.raises(SystemExit):
        main([str(path), "--reject", str(labels[1]), "99999"])
    assert rejected(Core.load_object(path)).tolist() == sorted(before + free[1:])


@pytest.mark.parametrize("saved", [[], [3, 41]])
def test_rejections_survive_save_and_load(tmp_path, saved):
    from pygor.classes.core_data import Core

    rec = Core.__new__(Core)
    rec.__dict__.update(rf_rejected=np.array(saved, dtype=int))
    # Through the same generic state writer every saved attribute goes through
    import h5py

    from pygor.persistence import read_group

    with h5py.File(tmp_path / "x.h5", "w") as handle:
        group = handle.create_group("recording_000")
        rec._save_state(group)
    with h5py.File(tmp_path / "x.h5", "r") as handle:
        state = read_group(handle["recording_000"])
    assert rejected(types.SimpleNamespace(**state)).tolist() == saved
