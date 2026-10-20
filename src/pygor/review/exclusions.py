"""Turn verdicts into things the rest of the analysis can consume.

The policy here is the one the pipeline already states: QC is non-destructive
and happens downstream, on the population table. So nothing in this module
edits the aggregate CSV or a saved recording. It produces a *vetted export* --
the aggregate table restricted to the fields of view a reviewer kept -- plus a
list of what was excluded and why, which is what a methods section and a
version registry both need.

A field of view verdict is a verdict about the recording, so it applies to
every cell in it. Cell verdicts, where they exist, are applied on top as a
column, not as a filter: which cells to drop is an analysis-time decision,
and dropping them here would make it invisible.
"""

from __future__ import annotations

import pathlib

import pandas as pd

from pygor.review.verdicts import VerdictStore

KEPT = ("keep",)
DROPPED = ("reject",)


def fov_verdicts(store: VerdictStore) -> pd.DataFrame:
    """One row per (fov_uid, condition) with its current alignment verdict."""
    latest = store.latest()
    if latest.empty:
        return pd.DataFrame(columns=["fov_uid", "condition", "verdict", "reason",
                                     "reviewer", "timestamp"])
    fovs = latest[latest.subject_type == "fov"]
    return fovs[["fov_uid", "condition", "verdict", "reason", "reviewer", "timestamp"]]


def excluded_fovs(store: VerdictStore) -> set[tuple[str, str]]:
    """(condition, fov_uid) pairs a reviewer rejected."""
    fovs = fov_verdicts(store)
    return {(r.condition, r.fov_uid)
            for r in fovs[fovs.verdict.isin(DROPPED)].itertuples()}


def cell_verdict_column(store: VerdictStore, cells: pd.DataFrame,
                        check="rf_quality") -> pd.Series:
    """The current cell-level verdict per row of the aggregate table, or ''."""
    latest = store.latest()
    if latest.empty or cells.empty:
        return pd.Series("", index=cells.index, dtype=str)
    mine = latest[(latest.subject_type == "cell") & (latest.check == check)]
    lookup = {(r.subject_uid, r.condition): r.verdict for r in mine.itertuples()}
    return pd.Series(
        [lookup.get((uid, cond), "") for uid, cond in zip(cells.cell_uid, cells.condition)],
        index=cells.index, dtype=str,
    )


def vetted(store: VerdictStore, cells: pd.DataFrame, *, include=KEPT) -> pd.DataFrame:
    """The aggregate table restricted to fields of view whose verdict is in ``include``.

    Fields of view with no verdict at all are dropped too: a vetted pool is one
    where every recording was looked at, not one where the unlooked-at ones
    slipped through. Two provenance columns are added -- ``qc_fov_verdict`` and
    ``qc_cell_verdict`` -- so a row can say why it is here.
    """
    fovs = fov_verdicts(store)
    keep = fovs[fovs.verdict.isin(include)][["fov_uid", "condition", "verdict"]]
    keep = keep.rename(columns={"verdict": "qc_fov_verdict"})
    # copy() defragments the wide merge result, which pandas warns about otherwise.
    out = cells.merge(keep, on=["fov_uid", "condition"], how="inner").copy()
    # assign() rather than item-setting on a 200-column frame that pandas
    # considers fragmented; same result, without the performance warning.
    return out.assign(qc_cell_verdict=cell_verdict_column(store, out))


def current(verdict, digests):
    """``verdict`` if none of the recordings it was made from has changed since, else None.

    ``digests`` maps (fov_uid, condition, role) to the recording's current "mtime:size",
    as ReviewSession.source_digests gives. A reprocess rewrites the recordings, so a
    verdict on the old cells stops counting rather than silently vouching for new ones.
    """
    if verdict is None or digests is None:
        return verdict
    for role, digest in (verdict.source_digest or {}).items():
        now = digests.get((verdict.fov_uid, verdict.condition, role))
        if now is not None and now != digest:
            return None
    return verdict


def covered_verdict(fov, own, role, digests=None) -> str:
    """The verdict that applies to one direction recording, or '' if none does.

    Its own recording verdict, if it has one. Otherwise a field-of-view 'keep' covers
    it, but only if the recording was part of the field of view when that verdict was
    made (``source_digest`` lists every recording it saw). A field of view that was not
    kept says nothing about whether its direction recording is usable on its own, and
    one judged before a recording was added never saw it. With ``digests``, a verdict
    made before the recordings were reprocessed does not count.
    """
    fov, own = current(fov, digests), current(own, digests)
    if own is not None:
        return own.verdict
    if fov is not None and fov.verdict == "keep" and role in (fov.source_digest or {}):
        return "keep"
    return ""


def direction_roles(cells: pd.DataFrame) -> list[str]:
    """The extra direction recordings carried as column blocks, 'osds[<stem>]'."""
    return sorted({c[: c.index("]") + 1] for c in cells.columns if c.startswith("osds[")})


def qc_columns(store: VerdictStore, cells: pd.DataFrame, digests=None) -> pd.DataFrame:
    """Per-row QC at every level, to filter on rather than to have filtered.

    qc_fov             the field-of-view verdict: RF, direction tuning and full field
                       can be paired cell for cell
    qc_osds            the master direction recording is usable on its own
    qc_osds[<stem>]    the same for each extra speed or repeat

    '' means nothing covers it. A rejected field of view still carries a 'keep' in
    qc_osds when its direction recording was judged clean on its own -- the receptive
    field failed, the direction tuning did not.
    """
    out = pd.DataFrame(index=cells.index)
    if cells.empty:
        return out.assign(qc_fov="", qc_osds="")
    roles = [("osds", "osds_recording_uid", "segmentation")] + [
        (role, f"{role}_recording_uid", "alignment") for role in direction_roles(cells)]
    values = {"qc_fov": {}, **{f"qc_{role}": {} for role, _c, _k in roles}}
    for (fov_uid, condition), group in cells.groupby(["fov_uid", "condition"], sort=False):
        fov = store.get("fov", fov_uid, "alignment", condition=condition)
        values["qc_fov"][(fov_uid, condition)] = fov.verdict if fov is not None else ""
        for role, uid_column, check in roles:
            uids = group[uid_column].dropna() if uid_column in group else pd.Series([])
            if uids.empty:
                verdict = ""
            else:
                own = store.get("recording", uids.iloc[0], check, condition=condition,
                                role=role)
                verdict = covered_verdict(fov, own, role, digests)
            values[f"qc_{role}"][(fov_uid, condition)] = verdict
    keys = list(zip(cells.fov_uid, cells.condition))
    for column, lookup in values.items():
        out[column] = [lookup.get(key, "") for key in keys]
    return out


def summary(store: VerdictStore, cells: pd.DataFrame, n_fovs: int) -> dict:
    fovs = fov_verdicts(store)
    counts = fovs.verdict.value_counts().to_dict() if not fovs.empty else {}
    kept_cells = 0
    if not cells.empty and not fovs.empty:
        kept = fovs[fovs.verdict.isin(KEPT)][["fov_uid", "condition"]]
        kept_cells = len(cells.merge(kept, on=["fov_uid", "condition"], how="inner"))
    return {
        "fovs": n_fovs,
        "judged": int(fovs.fov_uid.nunique()) if not fovs.empty else 0,
        "by_verdict": counts,
        "cells_total": int(len(cells)),
        "cells_kept": int(kept_cells),
    }


def export(store: VerdictStore, cells: pd.DataFrame, csv_path, *, include=KEPT,
           suffix="_qc_keep", digests=None) -> dict:
    """Write the vetted table beside the aggregate CSV, plus the exclusion list.

    Returns the paths written. The vetted file is what a version registry such
    as ``bc_rf_paper/dataset.py`` points at; the exclusion list is what a
    methods section cites.
    """
    csv_path = pathlib.Path(csv_path)
    vetted_path = csv_path.with_name(csv_path.stem + suffix + csv_path.suffix)
    review_dir = csv_path.parent / "review"
    review_dir.mkdir(parents=True, exist_ok=True)
    excluded_path = review_dir / "excluded_fovs.csv"
    verdicts_path = review_dir / "fov_verdicts.csv"

    table = vetted(store, cells, include=include)
    tmp = vetted_path.with_suffix(vetted_path.suffix + ".tmp")
    table.to_csv(tmp, index=False)
    tmp.replace(vetted_path)

    # Every row, with the QC of every level as columns. The vetted file above is the
    # RF view; this one is what direction analyses filter on qc_osds.
    full_path = csv_path.with_name(csv_path.stem + "_qc" + csv_path.suffix)
    full = pd.concat([cells, qc_columns(store, cells, digests)], axis=1).copy()
    full = full.assign(qc_cell_verdict=cell_verdict_column(store, full))
    tmp = full_path.with_suffix(full_path.suffix + ".tmp")
    full.to_csv(tmp, index=False)
    tmp.replace(full_path)

    fovs = fov_verdicts(store).sort_values(["condition", "fov_uid"])
    fovs.to_csv(verdicts_path, index=False)
    fovs[~fovs.verdict.isin(include)].to_csv(excluded_path, index=False)

    return {"vetted": vetted_path, "full": full_path, "excluded": excluded_path,
            "fov_verdicts": verdicts_path, "rows": len(table), "full_rows": len(full)}


def exclude_literal(store: VerdictStore) -> str:
    """A Python literal of rejected (condition, fov_uid) pairs, for pasting into
    a dataset module's EXCLUDE when a run must be reproducible without the sidecar."""
    pairs = sorted(excluded_fovs(store))
    lines = ",\n".join(f"    ({c!r}, {f!r})" for c, f in pairs)
    return "EXCLUDE = {\n" + lines + ("\n}" if pairs else "}")
