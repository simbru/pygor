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
    out = cells.merge(keep, on=["fov_uid", "condition"], how="inner")
    # assign() rather than item-setting on a 200-column frame that pandas
    # considers fragmented; same result, without the performance warning.
    return out.assign(qc_cell_verdict=cell_verdict_column(store, out))


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
           suffix="_qc_keep") -> dict:
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

    fovs = fov_verdicts(store).sort_values(["condition", "fov_uid"])
    fovs.to_csv(verdicts_path, index=False)
    fovs[~fovs.verdict.isin(include)].to_csv(excluded_path, index=False)

    return {"vetted": vetted_path, "excluded": excluded_path, "fov_verdicts": verdicts_path,
            "rows": len(table)}


def exclude_literal(store: VerdictStore) -> str:
    """A Python literal of rejected (condition, fov_uid) pairs, for pasting into
    a dataset module's EXCLUDE when a run must be reproducible without the sidecar."""
    pairs = sorted(excluded_fovs(store))
    lines = ",\n".join(f"    ({c!r}, {f!r})" for c, f in pairs)
    return "EXCLUDE = {\n" + lines + ("\n}" if pairs else "}")
