"""Receptive fields judged not real, recorded on the recording itself.

A noise recording can hold a handful of STRFs that are clearly artifacts while
the rest of its cells are fine. Rejecting those RFs here keeps the judgement in
the ``.recording.h5`` -- the unit of distribution -- so everything built from
the file sees it. The STRFs themselves are left untouched, so a rejection is
undone by restoring it and nothing is recomputed either way.

Stored as ``rf_rejected``: sorted ROI indices in the STRF array's own ROI space
(``strfs[roi * n_colours + colour]``), which for a recording that inherited
its ROIs is the master's index, ``roi_id`` in the tables. A rejection covers
every colour channel of the ROI. Nothing in pygor's own getters reads it; it
is for the code that turns recordings into tables, and for the GUI.

Without a display (over SSH), the same judgement from the shell::

    python -m pygor.strf.rejection FILE --sheet rfs.png     # every RF, numbered
    python -m pygor.strf.rejection FILE --reject 12 40      # and saves FILE
    python -m pygor.strf.rejection FILE --restore 12
    python -m pygor.strf.rejection FILE --list

Numbers are the labels pygor-gui shows on the ROIs and RF tiles, not the
stored indices: the two differ by one at least, and by more when the mask's
ids are not contiguous, so the conversion is left to the GUI's own mapping.
"""

import argparse
import pathlib
import sys

import numpy as np

ATTR = "rf_rejected"


def rejected(recording):
    """The ROI indices whose RF is rejected, as a sorted array (empty if none)."""
    values = getattr(recording, ATTR, None)
    if values is None:
        return np.array([], dtype=int)
    return np.unique(np.asarray(values, dtype=int))


def is_rejected(recording, roi):
    return int(roi) in set(rejected(recording).tolist())


def set_rejected(recording, roi, reject=True):
    """Reject (or restore) one ROI's RF. Returns the new sorted array.

    Changes the object only; save the recording to keep it.
    """
    current = set(rejected(recording).tolist())
    if reject:
        current.add(int(roi))
    else:
        current.discard(int(roi))
    values = np.array(sorted(current), dtype=int)
    setattr(recording, ATTR, values)
    return values


# -- labels, as pygor-gui numbers ROIs ------------------------------------------


def label_to_roi(recording, label):
    """The STRF ROI index behind a GUI label. Raises on a label with no STRF."""
    from pygor.gui.roi_bridge import label_to_trace_index

    roi = label_to_trace_index(label, getattr(recording, "rois", None))
    n_rois = _n_rois(recording)
    if roi < 0 or roi >= n_rois:
        raise ValueError(f"no ROI with label {label} has an STRF "
                         f"(labels run over {n_rois} ROIs)")
    return roi


def roi_to_label(recording, roi):
    from pygor.gui.roi_bridge import trace_index_to_label

    return trace_index_to_label(roi, getattr(recording, "rois", None))


def _n_rois(recording):
    n_colours = int(getattr(recording, "n_colours", 1) or 1)
    return len(recording.strfs) // n_colours


# -- a numbered sheet of every RF -------------------------------------------------


def draw_sheet(recording, path, channel=0, n_cols=10, fits=False, title=None):
    """Every ROI's time-collapsed RF as one numbered mosaic, saved to ``path``.

    The RFs panel of pygor-gui, as a file: one tile per ROI in ROI order, each
    scaled to its own peak, numbered with its GUI label, and crossed out if its
    RF is rejected. Needs no display.
    """
    from matplotlib.figure import Figure
    from matplotlib.patches import Rectangle

    from pygor.gui.rf import RFData, draw_fit, mosaic, normalised, tile_origin

    data = RFData(recording).refresh()
    if not 0 <= channel < data.n_colours:
        raise ValueError(f"channel {channel}: the recording has {data.n_colours}")
    rows = np.arange(data.n_rois) * data.n_colours + channel
    tiles = normalised(data.maps[rows])
    n_cols = max(1, min(n_cols, len(tiles)))
    grid = mosaic(tiles, n_cols)
    h, w = tiles.shape[-2:]

    tile_in = 1.1  # inches per tile across
    n_rows = int(np.ceil(len(tiles) / n_cols))
    fig = Figure(figsize=(n_cols * tile_in, n_rows * tile_in * h / w + 0.6))
    ax = fig.add_axes((0, 0, 1, 1 - 0.6 / fig.get_figheight()))
    import matplotlib

    cmap = matplotlib.colormaps["RdBu_r"].copy()
    cmap.set_bad("white")
    ax.imshow(grid, cmap=cmap, vmin=-1, vmax=1, interpolation="nearest")
    ax.set_axis_off()

    gone = set(rejected(recording).tolist())
    for roi in range(len(tiles)):
        x, y = tile_origin(roi, (h, w), n_cols)
        ax.text(x + 0.3, y + 0.3, str(roi_to_label(recording, roi)), fontsize=7,
                va="top", ha="left", color="tab:red" if roi in gone else "k",
                fontweight="bold" if roi in gone else "normal")
        if fits:
            box = Rectangle((x - 0.5, y - 0.5), w, h, transform=ax.transData)
            for patch in draw_fit(ax, data, rows[roi], offset=(x, y)):
                patch.set_clip_path(box)
        if roi in gone:
            xs, ys = [x - 0.5, x + w - 0.5], [y - 0.5, y + h - 0.5]
            ax.plot(xs, ys, color="k", lw=1.2)
            ax.plot(xs, ys[::-1], color="k", lw=1.2)
    fig.suptitle(
        (title or "") + f"  ch{channel}, {len(tiles)} ROIs, each scaled to its own peak. "
        "Numbers are pygor-gui labels; crossed out = RF rejected.",
        fontsize=9, x=0.01, ha="left",
    )
    fig.savefig(path, dpi=150)
    return pathlib.Path(path)


# -- command line -------------------------------------------------------------------


def main(argv=None):
    parser = argparse.ArgumentParser(
        prog="python -m pygor.strf.rejection",
        description="Draw, reject or restore RFs in a saved .recording.h5, "
                    "by the labels pygor-gui shows.",
    )
    parser.add_argument("file", help="a saved .recording.h5")
    parser.add_argument("--sheet", metavar="PNG", help="draw every RF, numbered")
    parser.add_argument("--channel", type=int, default=0)
    parser.add_argument("--cols", type=int, default=10, help="tiles per row")
    parser.add_argument("--fits", action="store_true", help="draw the Gaussian fits")
    parser.add_argument("--reject", type=int, nargs="+", default=[], metavar="LABEL")
    parser.add_argument("--restore", type=int, nargs="+", default=[], metavar="LABEL")
    parser.add_argument("--list", action="store_true", help="print the rejected labels")
    args = parser.parse_args(argv)

    from pygor.classes.core_data import Core

    path = pathlib.Path(args.file)
    if not args.file or not path.is_file():
        parser.error(f"not a file: {args.file!r} (an unset shell variable passes '')")
    recording = Core.load_object(path)
    if getattr(recording, "strfs", None) is None:
        parser.error(f"{path.name} has no STRFs")

    def labels():
        return [roi_to_label(recording, r) for r in rejected(recording)]

    if args.reject or args.restore:
        try:
            # Every label checked before anything changes
            changes = ([(label_to_roi(recording, l), True) for l in args.reject]
                       + [(label_to_roi(recording, l), False) for l in args.restore])
        except ValueError as exc:
            parser.error(f"{exc}; nothing was changed")
        before = labels()
        for roi, reject in changes:
            set_rejected(recording, roi, reject)
        recording.save_object(path, overwrite=True)
        print(f"{path.name}: {len(before)} -> {len(labels())} rejected RFs; saved")
        print("Rebuild the tables for this field (review --resync-fov, --export-qc, "
              "assemble) to see it in the paper data.")
    if args.list or not (args.sheet or args.reject or args.restore):
        print(f"rejected RFs, as GUI labels: {labels() or 'none'}")
    if args.sheet:
        out = draw_sheet(recording, args.sheet, channel=args.channel,
                         n_cols=args.cols, fits=args.fits, title=path.name)
        print(f"sheet: {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
