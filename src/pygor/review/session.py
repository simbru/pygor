"""One review session: the dataset, its fields of view, and the verdicts so far.

This is the seam every front-end sits on. The command line drives it now and a
terminal interface will drive the same methods later, so anything a reviewer can
do has to be a method here rather than logic living in a screen.

A dataset is described by a *binding* -- a small module supplying the things
pygor cannot know: where the data is, which stem is which stimulus, how a field
of view is named. Everything else is shared.
"""

from __future__ import annotations

import pathlib

import pandas as pd

from pygor.review.bundle import BundleCache, group_refs
from pygor.review.index import attach_status, missing_from_disk, scan_processed
from pygor.review.panels import render as render_panel
from pygor.review.rasterise import PanelCache
from pygor.review.verdicts import VerdictStore


class ReviewSession:
    """Everything a reviewer works against, loaded once and shared."""

    def __init__(self, binding, *, reviewer=None, cache_bytes=None, use_index_cache=True):
        self.binding = binding
        self.root = pathlib.Path(binding.ROOT)
        self.dataset = binding.DATASET
        self.store = VerdictStore(self.root, self.dataset, reviewer=reviewer)
        self.bundle_cache = BundleCache(**({"max_bytes": cache_bytes} if cache_bytes else {}))
        self.panel_cache = PanelCache()
        self._use_index_cache = use_index_cache
        self._refs = None
        self._bundles = None
        self._cells = None

    # -- loading ----------------------------------------------------------

    @property
    def refs(self):
        if self._refs is None:
            refs = scan_processed(self.root, cache=self._use_index_cache)
            self._refs = attach_status(refs, self.binding.STATUS)
        return self._refs

    @property
    def bundles(self):
        if self._bundles is None:
            self._bundles = group_refs(
                self.refs,
                prefix_of=self.binding.prefix_of,
                classify=self.binding.classify,
                fov_uid_of=self.binding.fov_uid_of,
                pick=getattr(self.binding, "pick", None),
            )
            for bundle in self._bundles:
                bundle.cache = self.bundle_cache
        return self._bundles

    @property
    def cells(self) -> pd.DataFrame:
        """The aggregate CSV, or an empty frame when it is not there yet.

        Absent is a normal state, not an error: the tool is useful before the
        table has been built, and the table is sometimes mid-rebuild.
        """
        if self._cells is None:
            path = pathlib.Path(self.binding.CSV)
            if path.exists():
                self._cells = pd.read_csv(path, low_memory=False)
            else:
                self._cells = pd.DataFrame()
        return self._cells

    def rescan(self) -> None:
        """Forget everything read from disk and read it again.

        For after a recording has been reprocessed: the index cache notices
        the changed file on its own, but the bundles, the loaded objects and
        the CSV rows were all read at startup and would otherwise keep
        describing the old one.
        """
        self._refs = None
        self._bundles = None
        self._cells = None
        self.bundle_cache.clear()
        self.panel_cache.clear()

    def missing_recordings(self) -> pd.DataFrame:
        """Ledger rows with no file, which a scan of Processed/ cannot see."""
        return missing_from_disk(self.refs, self.binding.STATUS)

    # -- navigation -------------------------------------------------------

    def find(self, fov_uid, condition=None):
        """The bundle for one field of view, with its CSV rows attached."""
        matches = [b for b in self.bundles if b.fov_uid == fov_uid]
        if condition is not None:
            matches = [b for b in matches if b.condition == condition]
        if not matches:
            partial = [b for b in self.bundles if fov_uid in b.fov_uid]
            if len(partial) == 1:
                matches = partial
            elif len(partial) > 1:
                raise KeyError(
                    f"{fov_uid!r} matches {len(partial)} fields of view; "
                    f"be more specific: {[b.fov_uid for b in partial[:5]]}"
                )
        if not matches:
            raise KeyError(f"no field of view matching {fov_uid!r}")
        bundle = matches[0]
        self._attach_cells(bundle)
        return bundle

    def _attach_cells(self, bundle) -> None:
        if bundle.cells is not None or self.cells.empty:
            return
        frame = self.cells
        rows = frame[frame.fov_uid == bundle.fov_uid]
        if "condition" in frame.columns:
            rows = rows[rows.condition == bundle.condition]
        bundle.attach_cells(rows)

    def render(self, panel, bundle, **kwargs):
        return render_panel(panel, bundle, cache=self.panel_cache, **kwargs)

    # -- judging ----------------------------------------------------------

    def judge(self, bundle, *, subject_type, check, verdict, subject_uid=None,
              role="", channel=-1, reason="", bulk=False):
        """Record one verdict, with the field of view's identity filled in."""
        if subject_uid is None:
            subject_uid = bundle.fov_uid
        recording_uid = bundle.peek(role).recording_uid if role in bundle.refs else ""
        entry = self.store.make(
            subject_type=subject_type,
            subject_uid=subject_uid,
            check=check,
            verdict=verdict,
            condition=bundle.condition,
            fov_uid=bundle.fov_uid,
            recording_uid=recording_uid,
            role=role,
            channel=channel,
            reason=reason,
            bulk=bulk,
            source_digest={
                r: f"{ref.mtime}:{ref.size}" for r, ref in bundle.refs.items()
            },
        )
        self.store.append(entry)
        return entry

    def verdict_for(self, bundle, *, subject_type="fov", check="alignment",
                    subject_uid=None, role="", channel=-1):
        return self.store.get(
            subject_type,
            subject_uid if subject_uid is not None else bundle.fov_uid,
            check,
            condition=bundle.condition,
            role=role,
            channel=channel,
        )

    # -- overview ---------------------------------------------------------

    def overview(self) -> pd.DataFrame:
        """One row per field of view: what it holds and what has been decided."""
        rows = []
        for bundle in self.bundles:
            alignments = bundle.alignments()
            transferred = [a for a in alignments.values() if a.transferred]
            worst = min((a.correlation for a in transferred), default=float("nan"))
            # Shift is the alignment-specific signal. Correlation is not: it is
            # taken between activity projections, so two recordings under
            # different stimuli drive different cells and correlate poorly while
            # being perfectly registered. Observed at r=0.364 with a 0.7 px
            # shift, against r=0.97 where the same cells are bright in both.
            shift = max(
                (max(abs(float(s)) for s in a.shift) for a in transferred if a.shift),
                default=float("nan"),
            )
            decided = self.verdict_for(bundle)
            rows.append(
                {
                    "fov_uid": bundle.fov_uid,
                    "condition": bundle.condition,
                    "session": bundle.session,
                    "prefix": bundle.prefix,
                    "roles": "+".join(sorted(bundle.roles)),
                    "master": bundle.master_role,
                    "n_cells": bundle.n_cells,
                    "worst_correlation": worst,
                    "max_shift_px": shift,
                    "n_lost": sum(a.n_lost for a in alignments.values()),
                    "unusable": sum(0 if a.usable else 1 for a in alignments.values()),
                    "bad_status": sum(
                        1 for r in bundle.refs.values() if r.status not in ("ok", "unknown")
                    ),
                    "verdict": decided.verdict if decided else "",
                }
            )
        frame = pd.DataFrame(rows)
        if frame.empty:
            return frame
        # Worst first: the point of an ordering here is that a reviewer can stop
        # once the list turns clean rather than walking all of them.
        return frame.sort_values(
            ["unusable", "bad_status", "n_lost", "max_shift_px", "worst_correlation"],
            ascending=[False, False, False, False, True],
        ).reset_index(drop=True)

    def progress(self) -> dict:
        latest = self.store.latest()
        total = len(self.bundles)
        judged = 0 if latest.empty else latest[latest.subject_type == "fov"].shape[0]
        return {
            "fovs": total,
            "fovs_judged": judged,
            "cells": int(self.cells.shape[0]) if not self.cells.empty else 0,
            "verdicts": 0 if latest.empty else len(latest),
            "by_check": {}
            if latest.empty
            else latest.groupby("check").size().to_dict(),
            "by_verdict": {}
            if latest.empty
            else latest.groupby("verdict").size().to_dict(),
        }
