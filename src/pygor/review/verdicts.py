"""Non-destructive record of what a human decided about the data.

Verdicts live beside the dataset, never inside the ``.recording.h5`` files.
Writing one into a recording would mean rewriting several hundred megabytes to
store a sentence, would give a judgement about the *relationship* between three
recordings no obvious owner, and would be destroyed by the next reprocessing
run. A sidecar survives all three.

The write path is an append-only JSONL file: one self-contained line per
judgement, so a crash costs at most the line being written, two reviewers can
append concurrently, and the history of who decided what and when is kept rather
than overwritten. ``verdicts.csv`` is a materialised latest-wins view of it for
reading in a spreadsheet, and is regenerated rather than edited.

A judgement is keyed by more than a cell id. One cell has up to three
recordings behind it and up to five colour channels within the noise recording,
and its receptive field can be fine in one channel and junk in another. So the
key is (subject, condition, role, channel, check), and ``condition`` is on every
row whatever the subject: ``fov_uid`` is not unique across conditions in the
Chromatic SWN dataset, where one plane was recorded before and after injection.
"""

from __future__ import annotations

import dataclasses
import datetime
import fcntl
import getpass
import json
import os
import pathlib
import uuid

import pandas as pd

SCHEMA_VERSION = 1

SUBJECT_TYPES = ("fov", "recording", "cell", "cell_channel")
CHECKS = ("alignment", "rf_quality", "trace_quality", "registration", "segmentation")
VERDICTS = ("keep", "reject", "flag", "uncertain")

# What makes two judgements the same judgement. `condition` is in here because
# fov_uid alone is ambiguous; `role` and `channel` because a cell is judged once
# per recording and once per colour.
LATEST_KEY = (
    "dataset",
    "subject_type",
    "subject_uid",
    "condition",
    "role",
    "channel",
    "check",
)

NO_CHANNEL = -1

VERDICTS_NAME = "verdicts.jsonl"
MATERIALISED_NAME = "verdicts.csv"


class SchemaTooNew(RuntimeError):
    """A record written by a newer pygor.review than this one."""


# Applied in order on read, so the JSONL may hold mixed versions forever and no
# migration ever rewrites history. Each entry upgrades v(key) -> v(key+1).
_UPGRADES: dict[int, callable] = {}


def _upgrade(record: dict) -> dict:
    version = int(record.get("schema_version", 1))
    if version > SCHEMA_VERSION:
        raise SchemaTooNew(
            f"record schema_version {version} > supported {SCHEMA_VERSION}; "
            "upgrade pygor rather than reading it partially"
        )
    while version < SCHEMA_VERSION:
        record = _UPGRADES[version](record)
        version = int(record.get("schema_version", version + 1))
    return record


@dataclasses.dataclass(frozen=True)
class Verdict:
    """One judgement about one subject."""

    subject_type: str
    subject_uid: str
    check: str
    verdict: str
    condition: str
    fov_uid: str = ""
    recording_uid: str = ""
    role: str = ""
    channel: int = NO_CHANNEL
    reason: str = ""
    reviewer: str = ""
    dataset: str = ""
    bulk: bool = False
    retracts: str = ""
    source_digest: dict = dataclasses.field(default_factory=dict)
    tool_version: str = f"pygor.review/{SCHEMA_VERSION}"
    verdict_id: str = ""
    timestamp: str = ""
    schema_version: int = SCHEMA_VERSION

    def __post_init__(self):
        if self.subject_type not in SUBJECT_TYPES:
            raise ValueError(f"subject_type {self.subject_type!r} not in {SUBJECT_TYPES}")
        if self.check not in CHECKS:
            raise ValueError(f"check {self.check!r} not in {CHECKS}")
        if self.verdict not in VERDICTS:
            raise ValueError(f"verdict {self.verdict!r} not in {VERDICTS}")
        # An int, never None: one NaN turns the pandas column to float64 and
        # then -1.0 != -1 breaks every downstream merge silently.
        object.__setattr__(self, "channel", int(self.channel))
        if not self.verdict_id:
            object.__setattr__(self, "verdict_id", uuid.uuid4().hex)
        if not self.timestamp:
            object.__setattr__(
                self,
                "timestamp",
                datetime.datetime.now().astimezone().isoformat(timespec="seconds"),
            )
        if not self.reviewer:
            object.__setattr__(self, "reviewer", _default_reviewer())

    @property
    def key(self) -> tuple:
        return tuple(getattr(self, field) for field in LATEST_KEY)

    def to_line(self) -> str:
        return json.dumps(dataclasses.asdict(self), separators=(",", ":"), default=str)

    @classmethod
    def from_line(cls, line: str) -> "Verdict":
        record = _upgrade(json.loads(line))
        known = {f.name for f in dataclasses.fields(cls)}
        return cls(**{k: v for k, v in record.items() if k in known})


def _default_reviewer() -> str:
    try:
        return getpass.getuser()
    except (KeyError, OSError):
        return "unknown"


class VerdictStore:
    """Append-only verdict log for one dataset, with an in-memory latest view.

    The in-memory view exists because a terminal front-end asks "what did I
    decide about this?" for every visible row on every repaint; a file read per
    row would be unusable.
    """

    def __init__(self, root, dataset, reviewer=None):
        self.root = pathlib.Path(root)
        self.dir = self.root / "review"
        self.path = self.dir / VERDICTS_NAME
        self.dataset = dataset
        self.reviewer = reviewer or _default_reviewer()
        self._latest: dict[tuple, Verdict] | None = None
        self._retracted: set[str] = set()

    # -- writing ----------------------------------------------------------

    def make(self, **kwargs) -> Verdict:
        """A verdict with this store's dataset and reviewer filled in."""
        kwargs.setdefault("dataset", self.dataset)
        kwargs.setdefault("reviewer", self.reviewer)
        return Verdict(**kwargs)

    def append(self, *verdicts: Verdict, fsync=False) -> None:
        """Append verdicts and update the in-memory view.

        ``fsync`` is off by default: a judgement is one short line, and paying a
        disk sync per keystroke would stall the UI. Call :meth:`flush` (or pass
        ``fsync=True``) at a natural pause to make them durable; a hard kill
        costs at most the unsynced tail.
        """
        if not verdicts:
            return
        self.dir.mkdir(parents=True, exist_ok=True)
        payload = "".join(v.to_line() + "\n" for v in verdicts).encode()
        fd = os.open(self.path, os.O_APPEND | os.O_WRONLY | os.O_CREAT, 0o644)
        try:
            # Exclusive lock so two reviewers cannot interleave a partial line.
            fcntl.flock(fd, fcntl.LOCK_EX)
            os.write(fd, payload)
            if fsync:
                os.fsync(fd)
        finally:
            fcntl.flock(fd, fcntl.LOCK_UN)
            os.close(fd)

        if self._latest is not None:
            for verdict in verdicts:
                self._apply(verdict)

    def flush(self) -> None:
        """Make everything appended so far durable."""
        if not self.path.exists():
            return
        fd = os.open(self.path, os.O_RDONLY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)

    def retract(self, verdict_id, reason="") -> Verdict:
        """Withdraw an earlier judgement by appending a tombstone.

        History is never rewritten, so a mistaken verdict stays visible as
        something that was made and then withdrawn.
        """
        target = self.by_id(verdict_id)
        if target is None:
            raise KeyError(f"no verdict with id {verdict_id!r}")
        tombstone = self.make(
            subject_type=target.subject_type,
            subject_uid=target.subject_uid,
            check=target.check,
            verdict="uncertain",
            condition=target.condition,
            fov_uid=target.fov_uid,
            recording_uid=target.recording_uid,
            role=target.role,
            channel=target.channel,
            reason=reason or f"retracts {verdict_id}",
            retracts=verdict_id,
        )
        self.append(tombstone)
        return tombstone

    # -- reading ----------------------------------------------------------

    def read_raw(self) -> list[Verdict]:
        """Every line, schema-upgraded, in file order."""
        if not self.path.exists():
            return []
        out = []
        with self.path.open() as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                out.append(Verdict.from_line(line))
        return out

    def _apply(self, verdict: Verdict) -> None:
        assert self._latest is not None
        if verdict.retracts:
            self._retracted.add(verdict.retracts)
            previous = self._latest.get(verdict.key)
            if previous is not None and previous.verdict_id == verdict.retracts:
                del self._latest[verdict.key]
            return
        if verdict.verdict_id in self._retracted:
            return
        self._latest[verdict.key] = verdict

    def _load(self) -> dict[tuple, Verdict]:
        if self._latest is None:
            self._latest = {}
            self._retracted = set()
            for verdict in self.read_raw():
                self._apply(verdict)
        return self._latest

    def get(self, subject_type, subject_uid, check, *, condition,
            role="", channel=NO_CHANNEL) -> Verdict | None:
        """The current verdict for one subject, or None. O(1)."""
        key = (self.dataset, subject_type, subject_uid, condition, role, int(channel), check)
        return self._load().get(key)

    def by_id(self, verdict_id) -> Verdict | None:
        for verdict in self.read_raw():
            if verdict.verdict_id == verdict_id:
                return verdict
        return None

    def latest(self) -> pd.DataFrame:
        """Latest-wins view as a frame, one row per distinct judgement."""
        records = list(self._load().values())
        if not records:
            return pd.DataFrame(columns=[f.name for f in dataclasses.fields(Verdict)])
        frame = pd.DataFrame([dataclasses.asdict(v) for v in records])
        # Keep the key columns typed so merges behave; see the channel note above.
        frame["channel"] = frame["channel"].astype("int64")
        return frame.sort_values(["subject_type", "subject_uid", "check"]).reset_index(drop=True)

    def for_fov(self, fov_uid, condition) -> pd.DataFrame:
        frame = self.latest()
        if frame.empty:
            return frame
        return frame[(frame.fov_uid == fov_uid) & (frame.condition == condition)]

    def materialise(self, path=None) -> pathlib.Path:
        """Write the latest-wins view to CSV, replacing it atomically.

        Never truncates in place: this file is meant to be opened in a
        spreadsheet while the JSONL keeps being appended to.
        """
        path = pathlib.Path(path) if path else self.dir / MATERIALISED_NAME
        path.parent.mkdir(parents=True, exist_ok=True)
        frame = self.latest()
        tmp = path.with_suffix(path.suffix + ".tmp")
        frame.to_csv(tmp, index=False)
        tmp.replace(path)
        return path

    # -- progress ---------------------------------------------------------

    def coverage(self, plan: pd.DataFrame) -> pd.DataFrame:
        """``plan`` left-joined onto the latest view, so progress is derived.

        There is no separate progress file to fall out of step: the log is the
        progress, and what remains is whatever the plan lists and the log lacks.
        """
        if plan.empty:
            return plan
        frame = self.latest()
        key = [c for c in LATEST_KEY if c in plan.columns and c != "dataset"]
        if frame.empty:
            out = plan.copy()
            out["verdict"] = pd.NA
            out["reason"] = pd.NA
            return out
        return plan.merge(
            frame[key + ["verdict", "reason", "reviewer", "timestamp", "verdict_id"]],
            on=key,
            how="left",
        )

    def stale(self, refs) -> pd.DataFrame:
        """Verdicts whose source recording changed since the judgement.

        Reprocessing a recording invalidates what a human concluded from it, and
        without this nothing would say so -- the verdict would sit beside data it
        was never about.
        """
        digests = {
            ref.recording_uid: f"{ref.mtime}:{ref.size}" for ref in refs
        }
        rows = []
        for verdict in self._load().values():
            for role, digest in (verdict.source_digest or {}).items():
                current = digests.get(verdict.recording_uid)
                if current is not None and current != digest:
                    rows.append(
                        {
                            **dataclasses.asdict(verdict),
                            "stale_role": role,
                            "recorded_digest": digest,
                            "current_digest": current,
                        }
                    )
                    break
        return pd.DataFrame(rows)
