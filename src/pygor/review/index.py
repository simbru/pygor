"""Cheap views of saved pygor objects, for deciding what to review.

A ``.recording.h5`` is a few hundred megabytes and ``Core.load_object`` reads all
of it eagerly, so building a work queue by loading recordings is not an option.
Everything a queue needs -- ROI counts, transfer provenance, registration
summary, trigger counts -- is in the group's HDF5 attributes, and dataset
*shapes* are metadata too. Reading only those takes a few milliseconds per file
instead of a few seconds, which is what makes a 270-file dataset scannable.

So this module never touches array data. :func:`read_arrays` is the one exception
and is restricted to the small 2-D datasets a panel needs (``rois``,
``average_stack``); anything larger belongs behind an explicit full load.
"""

from __future__ import annotations

import dataclasses
import json
import pathlib
import warnings

import h5py
import numpy as np
import pandas as pd

# Datasets small enough to read without thinking about it. `images` and `strfs`
# are deliberately absent: they are the reason a full load is expensive.
SMALL_ARRAYS = ("rois", "average_stack", "correlation_projection", "triggertimes",
                "triggertimes_frame", "ipl_depths", "quality_indices", "roi_sizes")

INDEX_CACHE_NAME = "index.jsonl"

# Columns carrying nested dicts, which need JSON round-tripping in the cache.
_JSON_FIELDS = ("roi_origin", "registration", "shapes")


@dataclasses.dataclass(frozen=True)
class RecordingRef:
    """What one ``.recording.h5`` says about itself, without reading arrays.

    Deliberately carries no ``role``: which recording is the SWN and which the
    OSDS master is a statement about a field of view, not about a file, so it
    belongs to the bundle that groups them.
    """

    path: pathlib.Path
    stem: str
    session: str
    condition: str
    recording_uid: str
    class_name: str
    num_rois: int
    n_colours: int | None
    num_strfs: int
    frame_hz: float | None
    trigger_mode: int | None
    n_triggers: int
    roi_origin: dict
    registration: dict
    exp_date: str
    source_path: str
    shapes: dict
    mtime: float
    size: int
    status: str = "unknown"
    status_reason: str = ""

    @property
    def transferred(self) -> bool:
        return self.roi_origin.get("method") == "transferred"

    @property
    def transfer_correlation(self) -> float:
        """Projection correlation of the ROI transfer, or NaN if not transferred."""
        value = self.roi_origin.get("correlation")
        return float("nan") if value is None else float(value)

    @property
    def lost_roi_ids(self) -> list[int]:
        """Master ROI ids dropped because they fell outside the shifted frame."""
        return [int(x) for x in self.roi_origin.get("lost_roi_ids", [])]

    def as_row(self) -> dict:
        row = dataclasses.asdict(self)
        row["path"] = str(self.path)
        return row


def _first_recording_group(h5file):
    """The single recording in a ``.recording.h5``.

    ``save_object`` writes one ``recording_000`` group; ``Experiment.save``
    writes many. Taking the first keeps this readable for both without
    pretending to support multi-recording files properly.
    """
    names = [k for k in h5file.keys() if k.startswith("recording_")]
    if not names:
        raise ValueError(f"no recording group in {h5file.filename}")
    return h5file[sorted(names)[0]]


def _json_attr(group, key) -> dict:
    """A ``_type_*=json_dict`` attribute as a dict, or {} when absent or broken.

    A malformed attribute must not take down a scan of 270 files, so it warns
    and yields an empty dict; the caller sees a recording with no provenance
    rather than a traceback.
    """
    raw = group.attrs.get(key)
    if raw is None:
        return {}
    if isinstance(raw, bytes):
        raw = raw.decode()
    if not isinstance(raw, str):
        return {}
    try:
        value = json.loads(raw)
    except json.JSONDecodeError:
        warnings.warn(f"could not parse {key!r} in {group.file.filename}", stacklevel=3)
        return {}
    return value if isinstance(value, dict) else {}


def _scalar(group, key, cast, default=None):
    value = group.attrs.get(key)
    if value is None:
        return default
    try:
        return cast(value)
    except (TypeError, ValueError):
        return default


def _strip_recording_suffix(path: pathlib.Path) -> str:
    """'1_0_SWN.recording.h5' -> '1_0_SWN'. The suffix is compound."""
    stem = path.name
    for suffix in (".recording.h5", ".h5"):
        if stem.endswith(suffix):
            return stem[: -len(suffix)]
    return path.stem


def scan_recording(path, *, condition=None, session=None) -> RecordingRef:
    """Read one saved recording's attributes. No array data is touched.

    ``condition`` and ``session`` default to the two directories above the file,
    which is how ``process.h5_path_for`` lays ``Processed/`` out.
    """
    path = pathlib.Path(path)
    stat = path.stat()
    stem = _strip_recording_suffix(path)
    if session is None:
        session = path.parent.name
    if condition is None:
        condition = path.parent.parent.name

    with h5py.File(path, "r") as h5file:
        group = _first_recording_group(h5file)
        roi_origin = _json_attr(group, "roi_origin")
        params = _json_attr(group, "params")
        metadata = _json_attr(group, "metadata")
        # Shapes are HDF5 metadata, so this stays free; a missing dataset is
        # itself worth recording (no `strfs` means STRFs were never calculated).
        shapes = {
            name: list(group[name].shape)
            for name in ("images", "rois", "average_stack", "traces_znorm")
            if name in group and isinstance(group[name], h5py.Dataset)
        }
        if "strfs" in group and isinstance(group["strfs"], h5py.Group):
            if "data" in group["strfs"]:
                shapes["strfs"] = list(group["strfs"]["data"].shape)
        n_triggers = int(group["triggertimes"].shape[0]) if "triggertimes" in group else 0
        exp_date = metadata.get("exp_date")
        if isinstance(exp_date, dict):  # persistence stores dates as {"__date__": ...}
            exp_date = exp_date.get("__date__", "")

        return RecordingRef(
            path=path,
            stem=stem,
            session=session,
            condition=condition,
            recording_uid=f"{session}::{stem}",
            class_name=str(group.attrs.get("__class_name__", "")),
            num_rois=_scalar(group, "num_rois", int, 0),
            n_colours=_scalar(group, "n_colours", int),
            num_strfs=_scalar(group, "num_strfs", int, 0),
            frame_hz=_scalar(group, "frame_hz", float),
            trigger_mode=_scalar(group, "trigger_mode", int),
            n_triggers=n_triggers,
            roi_origin=roi_origin,
            registration=params.get("registration") or {},
            exp_date=str(exp_date or ""),
            source_path=str(group.attrs.get("filename", "")),
            shapes=shapes,
            mtime=stat.st_mtime,
            size=stat.st_size,
        )


def _read_cache(path) -> dict[str, dict]:
    """Cached rows keyed by path string. A corrupt line is skipped, not fatal."""
    rows: dict[str, dict] = {}
    if not path.exists():
        return rows
    with path.open() as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if "path" in row:
                rows[row["path"]] = row
    return rows


def _write_cache(path, rows) -> None:
    """Replace the cache atomically, so an interrupted scan leaves the old one."""
    tmp = path.with_suffix(path.suffix + ".tmp")
    path.parent.mkdir(parents=True, exist_ok=True)
    with tmp.open("w") as handle:
        for row in rows:
            handle.write(json.dumps(row, separators=(",", ":"), default=str) + "\n")
    tmp.replace(path)


def _row_to_ref(row: dict) -> RecordingRef:
    row = dict(row)
    row["path"] = pathlib.Path(row["path"])
    for field in _JSON_FIELDS:
        if isinstance(row.get(field), str):
            row[field] = json.loads(row[field])
    known = {f.name for f in dataclasses.fields(RecordingRef)}
    return RecordingRef(**{k: v for k, v in row.items() if k in known})


def scan_processed(root, *, cache=True, pattern="*/*/*.recording.h5") -> list[RecordingRef]:
    """Every saved recording under ``<root>/Processed``, as refs.

    Cached to ``<root>/review/index.jsonl`` and invalidated per file on
    ``(mtime, size)``, so re-scanning after reprocessing one recording costs one
    file read rather than all of them.
    """
    root = pathlib.Path(root)
    processed = root / "Processed"
    paths = sorted(processed.glob(pattern))

    cache_path = root / "review" / INDEX_CACHE_NAME
    cached = _read_cache(cache_path) if cache else {}

    refs, rows, reused = [], [], 0
    for path in paths:
        stat = path.stat()
        row = cached.get(str(path))
        if row is not None and row.get("mtime") == stat.st_mtime and row.get("size") == stat.st_size:
            try:
                refs.append(_row_to_ref(row))
                rows.append(row)
                reused += 1
                continue
            except (TypeError, KeyError, json.JSONDecodeError):
                pass  # stale schema; fall through and rescan the file
        try:
            ref = scan_recording(path)
        except (OSError, ValueError) as error:
            warnings.warn(f"could not scan {path}: {error}", stacklevel=2)
            continue
        refs.append(ref)
        rows.append(ref.as_row())

    if cache and len(rows) != reused:
        _write_cache(cache_path, rows)
    return refs


def attach_status(refs, status_path) -> list[RecordingRef]:
    """Fold the pipeline's ledger into the refs.

    ``Processed/status.csv`` is the existing record of what processing made of
    each recording, and its failures are the first thing a reviewer should see.
    Merged on ``(condition, recording_uid)``, matching ``_common.write_status``.
    A recording with no ledger row keeps ``status='unknown'``.
    """
    status_path = pathlib.Path(status_path)
    if not status_path.exists():
        return list(refs)
    ledger = pd.read_csv(status_path)
    if ledger.empty:
        return list(refs)
    lookup = {
        (str(row.get("condition")), str(row.get("recording_uid"))): row
        for _, row in ledger.iterrows()
    }
    out = []
    for ref in refs:
        row = lookup.get((ref.condition, ref.recording_uid))
        if row is None:
            out.append(ref)
            continue
        reason = row.get("reason")
        out.append(
            dataclasses.replace(
                ref,
                status=str(row.get("status", "unknown")),
                status_reason="" if pd.isna(reason) else str(reason),
            )
        )
    return out


def missing_from_disk(refs, status_path) -> pd.DataFrame:
    """Ledger rows with no file on disk.

    A recording the pipeline claims to have processed but which is not there is
    invisible to a scan of ``Processed/``, so it has to be found from the other
    direction or it never reaches the reviewer.
    """
    status_path = pathlib.Path(status_path)
    if not status_path.exists():
        return pd.DataFrame()
    ledger = pd.read_csv(status_path)
    present = {(ref.condition, ref.recording_uid) for ref in refs}
    mask = [
        (str(row.condition), str(row.recording_uid)) not in present
        for row in ledger.itertuples()
    ]
    return ledger[pd.Series(mask, index=ledger.index)]


def as_dataframe(refs) -> pd.DataFrame:
    """Refs as a frame, with the nested dicts flattened into useful columns."""
    if not refs:
        return pd.DataFrame()
    frame = pd.DataFrame([ref.as_row() for ref in refs])
    frame["roi_method"] = [ref.roi_origin.get("method", "") for ref in refs]
    frame["transfer_correlation"] = [ref.transfer_correlation for ref in refs]
    frame["n_lost_rois"] = [len(ref.lost_roi_ids) for ref in refs]
    frame["reg_mean_error"] = [
        ref.registration.get("mean_error", float("nan")) for ref in refs
    ]
    frame["reg_max_shift"] = [
        max(ref.registration.get("max_shift") or [float("nan")]) for ref in refs
    ]
    return frame


def read_arrays(path, keys=("rois", "average_stack")) -> dict[str, np.ndarray]:
    """Read named small datasets from a saved recording.

    Restricted to :data:`SMALL_ARRAYS`: the point of this module is that a panel
    can draw a field of view without paying for a full load, and allowing
    ``images`` here would quietly undo that.
    """
    path = pathlib.Path(path)
    unknown = [k for k in keys if k not in SMALL_ARRAYS]
    if unknown:
        raise ValueError(
            f"{unknown} not in SMALL_ARRAYS; read large arrays through an explicit "
            "full load instead"
        )
    out = {}
    with h5py.File(path, "r") as h5file:
        group = _first_recording_group(h5file)
        for key in keys:
            if key in group and isinstance(group[key], h5py.Dataset):
                out[key] = group[key][()]
    return out


def _main(argv=None) -> int:
    import argparse
    import time

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("root", help="dataset root holding Processed/")
    parser.add_argument("--no-cache", action="store_true")
    args = parser.parse_args(argv)

    root = pathlib.Path(args.root)
    start = time.perf_counter()
    refs = scan_processed(root, cache=not args.no_cache)
    refs = attach_status(refs, root / "Processed" / "status.csv")
    elapsed = time.perf_counter() - start

    frame = as_dataframe(refs)
    print(f"{len(refs)} recordings scanned in {elapsed:.2f} s")
    if frame.empty:
        return 0
    print()
    print(frame["status"].value_counts().to_string())
    print()
    print(frame["roi_method"].value_counts().to_string())
    transferred = frame[frame.roi_method == "transferred"]
    if not transferred.empty:
        corr = transferred.transfer_correlation
        print()
        print(
            f"transfer correlation: min {corr.min():.3f} median {corr.median():.3f} "
            f"| {int((corr < 0.8).sum())} below 0.8 "
            f"| {int((transferred.n_lost_rois > 0).sum())} with lost ROIs"
        )
    missing = missing_from_disk(refs, root / "Processed" / "status.csv")
    if not missing.empty:
        print(f"\n{len(missing)} ledger rows with no file on disk")
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
