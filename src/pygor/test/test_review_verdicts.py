"""Unit tests for pygor.review.verdicts. No recording needed.

The store is the only thing in the review stack that a reviewer's work cannot be
recovered from if it is wrong, so these lean on durability and on the key
semantics rather than on happy-path round trips.
"""

import concurrent.futures
import json
import multiprocessing
import pathlib

import pandas as pd
import pytest

from pygor.review.verdicts import (
    LATEST_KEY,
    SCHEMA_VERSION,
    SchemaTooNew,
    Verdict,
    VerdictStore,
)

DATASET = "Test Dataset"


def _store(root) -> VerdictStore:
    return VerdictStore(root, DATASET, reviewer="tester")


def _cell(store, uid="s::0_0#3", verdict="keep", **kwargs):
    kwargs.setdefault("condition", "control")
    kwargs.setdefault("fov_uid", "s::0_0")
    kwargs.setdefault("recording_uid", "s::0_0_SWN")
    kwargs.setdefault("role", "swn")
    kwargs.setdefault("check", "rf_quality")
    return store.make(subject_type="cell", subject_uid=uid, verdict=verdict, **kwargs)


# Module level so ProcessPoolExecutor can pickle it.
def _append_many(args):
    root, reviewer, count = args
    store = VerdictStore(root, DATASET, reviewer=reviewer)
    for i in range(count):
        store.append(
            store.make(
                subject_type="cell",
                subject_uid=f"{reviewer}#{i}",
                check="rf_quality",
                verdict="keep",
                condition="control",
            )
        )
    return count


class TestVerdictRecord:
    def test_defaults_are_filled_in(self):
        verdict = Verdict(
            subject_type="cell",
            subject_uid="s::0_0#1",
            check="rf_quality",
            verdict="keep",
            condition="control",
        )
        assert verdict.verdict_id
        assert verdict.timestamp
        assert verdict.channel == -1
        assert verdict.schema_version == SCHEMA_VERSION

    def test_channel_is_coerced_to_int(self):
        """A float channel would turn the pandas column to float64 later."""
        verdict = Verdict(
            subject_type="cell_channel",
            subject_uid="s::0_0#1",
            check="rf_quality",
            verdict="keep",
            condition="control",
            channel=2.0,
        )
        assert verdict.channel == 2
        assert isinstance(verdict.channel, int)

    @pytest.mark.parametrize(
        "kwargs",
        [
            {"subject_type": "nonsense"},
            {"check": "vibes"},
            {"verdict": "maybe-ish"},
        ],
    )
    def test_rejects_unknown_vocabulary(self, kwargs):
        base = dict(
            subject_type="cell",
            subject_uid="s::0_0#1",
            check="rf_quality",
            verdict="keep",
            condition="control",
        )
        with pytest.raises(ValueError):
            Verdict(**{**base, **kwargs})

    def test_line_round_trip(self):
        verdict = Verdict(
            subject_type="cell_channel",
            subject_uid="s::0_0#1",
            check="rf_quality",
            verdict="reject",
            condition="acblock",
            channel=3,
            reason="no structure",
        )
        assert Verdict.from_line(verdict.to_line()) == verdict

    def test_key_matches_latest_key_order(self):
        verdict = Verdict(
            subject_type="cell",
            subject_uid="s::0_0#1",
            check="rf_quality",
            verdict="keep",
            condition="control",
            dataset=DATASET,
        )
        assert verdict.key == tuple(getattr(verdict, f) for f in LATEST_KEY)

    def test_future_schema_is_refused_loudly(self):
        line = json.dumps(
            {
                "schema_version": SCHEMA_VERSION + 1,
                "subject_type": "cell",
                "subject_uid": "x",
                "check": "rf_quality",
                "verdict": "keep",
                "condition": "control",
            }
        )
        with pytest.raises(SchemaTooNew):
            Verdict.from_line(line)


class TestAppendAndRead:
    def test_append_then_read(self, tmp_path):
        store = _store(tmp_path)
        store.append(_cell(store))
        assert len(store.read_raw()) == 1
        assert store.path == tmp_path / "review" / "verdicts.jsonl"

    def test_latest_wins_on_rejudgement(self, tmp_path):
        store = _store(tmp_path)
        store.append(_cell(store, verdict="keep", reason="looks fine"))
        store.append(_cell(store, verdict="reject", reason="changed my mind"))
        assert len(store.read_raw()) == 2  # history kept
        frame = store.latest()
        assert len(frame) == 1
        assert frame.iloc[0].verdict == "reject"
        assert frame.iloc[0].reason == "changed my mind"

    def test_role_and_channel_are_part_of_the_key(self, tmp_path):
        """One cell judged per recording and per colour is several judgements."""
        store = _store(tmp_path)
        store.append(_cell(store, role="swn", verdict="keep"))
        store.append(_cell(store, role="fff", verdict="reject"))
        store.append(
            store.make(
                subject_type="cell_channel",
                subject_uid="s::0_0#3",
                check="rf_quality",
                verdict="reject",
                condition="control",
                role="swn",
                channel=2,
            )
        )
        assert len(store.latest()) == 3

    def test_condition_separates_colliding_fov_uids(self, tmp_path):
        """fov_uid is not unique across conditions, so a verdict must say which."""
        store = _store(tmp_path)
        for condition in ("control", "acblock"):
            store.append(
                store.make(
                    subject_type="fov",
                    subject_uid="240124 inj::1_0",
                    check="alignment",
                    verdict="keep" if condition == "control" else "reject",
                    condition=condition,
                    fov_uid="240124 inj::1_0",
                )
            )
        frame = store.latest()
        assert len(frame) == 2
        assert set(frame.condition) == {"control", "acblock"}

    def test_get_returns_current_verdict(self, tmp_path):
        store = _store(tmp_path)
        store.append(_cell(store, verdict="flag", reason="odd"))
        found = store.get("cell", "s::0_0#3", "rf_quality", condition="control", role="swn")
        assert found is not None
        assert found.verdict == "flag"
        assert store.get("cell", "nope", "rf_quality", condition="control") is None

    def test_get_reflects_appends_without_rereading(self, tmp_path):
        store = _store(tmp_path)
        store._load()  # warm the in-memory view first
        store.append(_cell(store, verdict="keep"))
        found = store.get("cell", "s::0_0#3", "rf_quality", condition="control", role="swn")
        assert found is not None and found.verdict == "keep"

    def test_channel_column_stays_integer(self, tmp_path):
        """A float64 channel silently breaks every downstream merge."""
        store = _store(tmp_path)
        store.append(_cell(store))
        store.append(
            store.make(
                subject_type="cell_channel",
                subject_uid="s::0_0#3",
                check="rf_quality",
                verdict="keep",
                condition="control",
                channel=0,
            )
        )
        assert store.latest().channel.dtype == "int64"

    def test_empty_store_reads_as_empty(self, tmp_path):
        store = _store(tmp_path)
        assert store.read_raw() == []
        assert store.latest().empty
        assert store.get("cell", "x", "rf_quality", condition="control") is None

    def test_blank_lines_are_skipped(self, tmp_path):
        store = _store(tmp_path)
        store.append(_cell(store))
        with store.path.open("a") as handle:
            handle.write("\n\n")
        assert len(store.read_raw()) == 1


class TestRetract:
    def test_retract_appends_and_clears_the_verdict(self, tmp_path):
        store = _store(tmp_path)
        verdict = _cell(store, verdict="reject")
        store.append(verdict)
        assert len(store.latest()) == 1
        store.retract(verdict.verdict_id, reason="misread the panel")
        assert len(store.read_raw()) == 2  # nothing rewritten
        assert store.latest().empty

    def test_retract_unknown_id_raises(self, tmp_path):
        store = _store(tmp_path)
        with pytest.raises(KeyError):
            store.retract("nope")

    def test_retraction_survives_a_reload(self, tmp_path):
        store = _store(tmp_path)
        verdict = _cell(store, verdict="reject")
        store.append(verdict)
        store.retract(verdict.verdict_id)
        assert _store(tmp_path).latest().empty


class TestDurability:
    def test_concurrent_appends_do_not_interleave(self, tmp_path):
        """Two reviewers on one dataset must not corrupt each other's lines."""
        root = str(tmp_path)
        # spawn, not the default fork: pytest is multi-threaded by the time this
        # runs, and forking out of a threaded process is deprecated.
        context = multiprocessing.get_context("spawn")
        with concurrent.futures.ProcessPoolExecutor(max_workers=2, mp_context=context) as pool:
            list(pool.map(_append_many, [(root, "alice", 200), (root, "bob", 200)]))
        store = _store(tmp_path)
        lines = [ln for ln in store.path.read_text().splitlines() if ln.strip()]
        assert len(lines) == 400
        for line in lines:  # every line parses, so none was written half-way
            json.loads(line)
        assert len(store.latest()) == 400

    def test_flush_on_empty_store_is_harmless(self, tmp_path):
        _store(tmp_path).flush()

    def test_append_with_fsync(self, tmp_path):
        store = _store(tmp_path)
        store.append(_cell(store), fsync=True)
        store.flush()
        assert len(store.read_raw()) == 1


class TestMaterialise:
    def test_writes_latest_view_and_leaves_no_temp_file(self, tmp_path):
        store = _store(tmp_path)
        store.append(_cell(store, verdict="keep"))
        store.append(_cell(store, verdict="reject"))
        path = store.materialise()
        assert path.exists()
        assert not list(path.parent.glob("*.tmp"))
        frame = pd.read_csv(path)
        assert len(frame) == 1
        assert frame.iloc[0].verdict == "reject"

    def test_replaces_rather_than_truncates(self, tmp_path):
        """The CSV is meant to be readable in a spreadsheet while appends continue."""
        store = _store(tmp_path)
        store.append(_cell(store))
        first = store.materialise()
        inode = first.stat().st_ino
        store.append(_cell(store, uid="s::0_0#4"))
        second = store.materialise()
        assert second.stat().st_ino != inode

    def test_empty_store_materialises_with_headers(self, tmp_path):
        path = _store(tmp_path).materialise()
        frame = pd.read_csv(path)
        assert frame.empty
        assert "subject_uid" in frame.columns


class TestCoverage:
    def _plan(self):
        return pd.DataFrame(
            [
                {
                    "subject_type": "cell",
                    "subject_uid": "s::0_0#3",
                    "condition": "control",
                    "role": "swn",
                    "channel": -1,
                    "check": "rf_quality",
                },
                {
                    "subject_type": "cell",
                    "subject_uid": "s::0_0#4",
                    "condition": "control",
                    "role": "swn",
                    "channel": -1,
                    "check": "rf_quality",
                },
            ]
        )

    def test_coverage_marks_what_is_done(self, tmp_path):
        store = _store(tmp_path)
        store.append(_cell(store, uid="s::0_0#3", verdict="keep"))
        covered = store.coverage(self._plan())
        done = covered[covered.subject_uid == "s::0_0#3"].iloc[0]
        todo = covered[covered.subject_uid == "s::0_0#4"].iloc[0]
        assert done.verdict == "keep"
        assert pd.isna(todo.verdict)

    def test_coverage_of_empty_store_is_all_pending(self, tmp_path):
        covered = _store(tmp_path).coverage(self._plan())
        assert covered.verdict.isna().all()

    def test_coverage_does_not_duplicate_plan_rows(self, tmp_path):
        """A re-judged subject must not fan the plan out into two rows."""
        store = _store(tmp_path)
        store.append(_cell(store, uid="s::0_0#3", verdict="keep"))
        store.append(_cell(store, uid="s::0_0#3", verdict="reject"))
        assert len(store.coverage(self._plan())) == 2


class TestStale:
    def test_flags_verdicts_whose_source_changed(self, tmp_path):
        class Ref:
            def __init__(self, uid, mtime, size):
                self.recording_uid = uid
                self.mtime = mtime
                self.size = size

        store = _store(tmp_path)
        store.append(
            _cell(store, source_digest={"swn": "100.0:500"})
        )
        assert store.stale([Ref("s::0_0_SWN", 100.0, 500)]).empty
        stale = store.stale([Ref("s::0_0_SWN", 999.0, 500)])
        assert len(stale) == 1
        assert stale.iloc[0].stale_role == "swn"

    def test_verdict_without_digest_is_never_stale(self, tmp_path):
        store = _store(tmp_path)
        store.append(_cell(store))
        assert store.stale([]).empty



class TestExclusions:
    """Verdicts become a vetted export, never an edit of the aggregate table."""

    def _cells(self):
        rows = []
        for fov, cond, n in (("s::control::0_0", "control", 3), ("s::control::0_1", "control", 2),
                             ("s::acblock::0_0", "acblock", 2), ("s::control::0_2", "control", 1)):
            rows += [{"cell_uid": f"{fov}#{i}", "fov_uid": fov, "condition": cond, "roi_id": i}
                     for i in range(n)]
        return pd.DataFrame(rows)

    def _store(self, tmp_path):
        store = _store_for(tmp_path)
        for fov, cond, verdict in (("s::control::0_0", "control", "keep"),
                                   ("s::control::0_1", "control", "reject"),
                                   ("s::acblock::0_0", "acblock", "uncertain")):
            store.append(store.make(subject_type="fov", subject_uid=fov, check="alignment",
                                    verdict=verdict, condition=cond, fov_uid=fov,
                                    reason="because"))
        # s::control::0_2 is never judged
        store.append(store.make(subject_type="cell", subject_uid="s::control::0_0#1",
                                check="rf_quality", verdict="reject", condition="control",
                                fov_uid="s::control::0_0", role="swn"))
        return store

    def test_vetted_keeps_only_kept_fovs(self, tmp_path):
        from pygor.review import exclusions

        table = exclusions.vetted(self._store(tmp_path), self._cells())
        assert set(table.fov_uid) == {"s::control::0_0"}
        assert len(table) == 3
        assert (table.qc_fov_verdict == "keep").all()

    def test_unjudged_fovs_are_left_out(self, tmp_path):
        """A vetted pool is one where every recording was looked at."""
        from pygor.review import exclusions

        table = exclusions.vetted(self._store(tmp_path), self._cells())
        assert "s::control::0_2" not in set(table.fov_uid)

    def test_include_uncertain(self, tmp_path):
        from pygor.review import exclusions

        table = exclusions.vetted(self._store(tmp_path), self._cells(),
                                  include=("keep", "uncertain"))
        assert set(table.fov_uid) == {"s::control::0_0", "s::acblock::0_0"}

    def test_cell_verdicts_are_a_column_not_a_filter(self, tmp_path):
        from pygor.review import exclusions

        table = exclusions.vetted(self._store(tmp_path), self._cells())
        assert len(table) == 3  # the rejected cell is still there
        assert dict(zip(table.cell_uid, table.qc_cell_verdict))["s::control::0_0#1"] == "reject"

    def test_excluded_pairs_carry_condition(self, tmp_path):
        from pygor.review import exclusions

        assert exclusions.excluded_fovs(self._store(tmp_path)) == {("control", "s::control::0_1")}

    def test_export_writes_beside_the_csv(self, tmp_path):
        from pygor.review import exclusions

        csv = tmp_path / "data" / "rois.csv"
        csv.parent.mkdir()
        self._cells().to_csv(csv, index=False)
        paths = exclusions.export(self._store(tmp_path), self._cells(), csv)
        assert paths["vetted"] == csv.with_name("rois_qc_keep.csv")
        assert paths["vetted"].exists() and paths["excluded"].exists()
        assert not list(csv.parent.glob("*.tmp"))
        excluded = pd.read_csv(paths["excluded"])
        assert set(excluded.verdict) == {"reject", "uncertain"}

    def test_literal_is_valid_python(self, tmp_path):
        from pygor.review import exclusions

        namespace = {}
        exec(exclusions.exclude_literal(self._store(tmp_path)), namespace)
        assert namespace["EXCLUDE"] == {("control", "s::control::0_1")}

    def test_empty_store(self, tmp_path):
        from pygor.review import exclusions

        store = _store_for(tmp_path)
        assert exclusions.vetted(store, self._cells()).empty
        assert exclusions.excluded_fovs(store) == set()


def _store_for(root):
    return VerdictStore(root, DATASET, reviewer="tester")
