"""ScanM header metadata kept on Core, and z-stack detection built on it."""

import h5py
import numpy as np
import pytest

import pygor.load
from pygor.classes.core_data import Core
from pygor.preproc import scanm, wparams
from pygor.test.helpers import DEMO_DATA

# Keys from a real ScanM z-stack header, with the frame shrunk to keep it small.
STACK_HEADER = {
    "String": {
        "ComputerName": "cssd901260",
        "UserName": "Main",
        "DateStamp": "2019-4-19",
        "TimeStamp": "14-54-12-647",
        "ScanPathFunc": "XYScan2|1024|32|16|0|0|0|1",
        "Comment": "n/a",
    },
    "UINT32": {
        "InputChannelMask": 1,
        "PixelBuffer_#0_Length": 32 * 16,
        "NumberOfFrames": 4,
        "FrameCounter": 0,
        "FrameWidth": 32,
        "FrameHeight": 16,
        "PixRetraceLen": 0,
        "XPixLineOffs": 0,
        "ScanMode": 0,
        "ScanType": 11,
        "NFrPerStep": 3,
        "StimBufPerFr": 1,
        "dZPixels": 1,
    },
    "REAL32": {
        "RealPixelDuration_µs": 3.0,
        "TargetedPixelDuration_µs": 3.0,
        "XCoord_um": -4349.0,
        "YCoord_um": -2816.2,
        "ZCoord_um": 2222.8,
        "ZStep_um": 1.0,
        "Zoom": 3.2,
        "Angle_deg": 0.0,
        "zoomFactorZ": 1.0,
    },
}


def write_scanm_pair(tmp_path, name="rec", overrides=None, drop=()):
    """Write a minimal .smh/.smp pair; returns the .smh path.

    ``overrides`` maps header keys to new values, ``drop`` removes keys.
    The .smp holds NumberOfFrames * NFrPerStep raw frames for a stack, so the
    loader's per-step averaging yields NumberOfFrames planes.
    """
    sections = {kind: dict(entries) for kind, entries in STACK_HEADER.items()}
    for kind in sections:
        for key, val in (overrides or {}).items():
            if key in sections[kind]:
                sections[kind][key] = val
        for key in drop:
            sections[kind].pop(key, None)

    lines = [
        f"{kind}, {key} = {val};"
        for kind, entries in sections.items()
        for key, val in entries.items()
    ]
    smh = tmp_path / f"{name}.smh"
    smh.write_bytes(b"\x00" * 64 + ("\r\n" + "\r\n".join(lines)).encode("utf-16-le"))

    ints = sections["UINT32"]
    per_step = ints.get("NFrPerStep", 1) if ints.get("ScanType") == 11 else 1
    n_raw = ints["NumberOfFrames"] * per_step
    h, w = ints["FrameHeight"], ints["FrameWidth"]
    # Each logical frame gets its own constant value so planes are distinguishable.
    frames = np.repeat(np.arange(n_raw // per_step, dtype=np.uint16) * 100, per_step)
    pixels = np.broadcast_to(frames[:, None, None], (n_raw, h, w))
    smh.with_suffix(".smp").write_bytes(np.ascontiguousarray(pixels).tobytes())
    return smh


@pytest.fixture
def stack_smh(tmp_path):
    return write_scanm_pair(tmp_path, "stack")


@pytest.fixture
def sweep_smh(tmp_path):
    # A time-lapse recording still carries a non-zero ZStep_um, as on the rig.
    return write_scanm_pair(tmp_path, "sweep", {"ScanType": 10, "NFrPerStep": 1})


# -- header -> metadata --------------------------------------------------------


def test_stack_metadata_keeps_z_fields(stack_smh):
    rec = Core(stack_smh)
    wpn = rec.metadata["wParamsNum"]
    assert wpn["User_ScanType"] == 11
    assert wpn["ZStep_um"] == 1.0
    assert wpn["User_NFrPerStep"] == 3
    assert wpn["User_zoomZ"] == 1.0
    assert rec.metadata["scanm_header"]["NFrPerStep"] == 3


def test_header_names_resolve(stack_smh):
    """Top-level keys that used to be looked up under names ScanM doesn't write."""
    meta = Core(stack_smh).metadata
    assert meta["PixelDuration_us"] == 3.0
    assert meta["RetracePixels"] == 0
    assert meta["LineOffset"] == 0
    assert meta["Angle"] == 0.0
    assert meta["User"] == "Main"


def test_wparamsstr_from_header(stack_smh):
    wps = Core(stack_smh).metadata["wParamsStr"]
    assert wps["DateStamp_d_m_y"] == "2019-4-19"
    assert wps["User_ScanPathFunc"].startswith("XYScan2")


def test_from_scanm_matches_constructor(stack_smh):
    """Both ScanM entry points build metadata the same way."""
    assert Core.from_scanm(stack_smh).metadata == Core(stack_smh).metadata


def test_every_mapped_label_exists_in_wparamsnum():
    labels = set(wparams.WPARAMSNUM_LABELS)
    assert set(wparams.SCANM_HEADER_TO_WPARAMSNUM.values()) <= labels
    assert set(wparams.SCANM_HEADER_TO_WPARAMSSTR.values()) <= set(
        wparams.WPARAMSSTR_LABELS
    )


# -- z-stack detection ---------------------------------------------------------


def test_stack_is_zstack(stack_smh):
    rec = Core(stack_smh)
    assert rec.is_zstack
    assert rec.images.shape[0] == 4  # NFrPerStep raw frames averaged per plane
    np.testing.assert_allclose(rec.z_positions_um, 2222.8 + np.arange(4) * 1.0)


def test_sweep_is_not_zstack(sweep_smh):
    rec = Core(sweep_smh)
    assert not rec.is_zstack
    assert rec.z_positions_um is None
    assert rec.images.shape[0] == 4


def test_missing_scantype_is_not_zstack(tmp_path):
    rec = Core(write_scanm_pair(tmp_path, "noscantype", drop=("ScanType",)))
    assert "User_ScanType" not in rec.metadata["wParamsNum"]
    assert not rec.is_zstack


def test_zero_zstep_has_no_spacing_but_still_a_stack(tmp_path):
    rec = Core(write_scanm_pair(tmp_path, "flat", {"ZStep_um": 0.0}))
    assert rec.is_zstack
    np.testing.assert_allclose(rec.z_positions_um, np.full(4, 2222.8))


# -- persistence ---------------------------------------------------------------


def test_stack_round_trips(stack_smh, tmp_path):
    rec = Core(stack_smh)
    path = rec.save_object(tmp_path / "out")
    loaded = Core.load_object(path)
    assert loaded.is_zstack
    assert loaded.metadata["wParamsNum"] == rec.metadata["wParamsNum"]
    assert loaded.metadata["wParamsStr"] == rec.metadata["wParamsStr"]
    assert loaded.metadata["scanm_header"] == rec.metadata["scanm_header"]
    np.testing.assert_allclose(loaded.z_positions_um, rec.z_positions_um)


def test_file_saved_without_wparams_loads(stack_smh, tmp_path, recwarn):
    """A .recording.h5 from before wParamsNum was kept: loads quietly, not a stack."""
    rec = Core(stack_smh)
    for key in ("wParamsNum", "wParamsStr", "scanm_header"):
        del rec.metadata[key]
    path = rec.save_object(tmp_path / "old")
    loaded = Core.load_object(path)
    assert not loaded.is_zstack
    assert loaded.z_positions_um is None
    assert not [w for w in recwarn if "wParams" in str(w.message)]


def test_reloaded_scanm_exports_real_wparamsnum(stack_smh, tmp_path):
    """export_to_h5 after a reload uses the saved wParamsNum, not the demo defaults."""
    loaded = Core.load_object(Core(stack_smh).save_object(tmp_path / "out"))
    out = tmp_path / "export.h5"
    loaded.export_to_h5(out, overwrite=True)
    with h5py.File(out, "r") as f:
        ds = f["wParamsNum"]
        exported = wparams.wparamsnum_to_dict(
            ds[()], ds.attrs["IGORWaveDimensionLabels"]
        )
    assert exported["User_ScanType"] == 11
    assert exported["User_NFrPerStep"] == 3


# -- IGOR exports --------------------------------------------------------------


@pytest.mark.demo_data
def test_igor_h5_metadata_has_labelled_wparams():
    rec = pygor.load.Core(DEMO_DATA)
    wpn = rec.metadata["wParamsNum"]
    assert wpn["User_dxPix"] * wpn["RealPixDur"] * 1e-6 == pytest.approx(rec.linedur_s)
    assert "DateStamp_d_m_y" in rec.metadata["wParamsStr"]
    assert rec.is_zstack == (wpn.get("User_ScanType") == 11)


def test_wparamsnum_to_dict_without_labels():
    data = np.arange(len(wparams.WPARAMSNUM_LABELS) - 1, dtype=float)
    out = wparams.wparamsnum_to_dict(data)
    assert out["HdrLenInValuePairs"] == 0.0
    assert out["User_ScanType"] == wparams.WPARAMSNUM_LABELS.index("User_ScanType") - 1
    assert wparams.wparamsnum_to_dict(np.zeros(5)) == {}
