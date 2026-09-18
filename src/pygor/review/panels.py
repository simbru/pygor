"""Named renderers that draw the evidence behind a judgement.

A panel takes a field of view and returns a matplotlib figure at a requested
pixel size. Which recording, cell or colour it draws is passed in; what it needs
loaded is declared, so a caller can tell a panel that reads a projection in a
millisecond from one that needs a recording in memory.

Two rules that are easy to break silently:

* build figures through ``matplotlib.figure.Figure``, not pyplot. Panels are
  rendered from worker threads, and pyplot's global state does not survive that.
  A panel that wraps an existing pygor plotting function cannot honour this --
  several of them build figures through pyplot internally -- so it declares
  ``threadsafe=False`` and gets serialised instead of corrupting a neighbour.
* never render a whole recording at once. A full-recording receptive-field
  overview comes out around six inches by eighty-three, which no terminal can
  show and no reviewer can read. Panels page instead.

Panels draw outlines rather than filled ROI overlays, because the thing being
judged is whether a mask sits on the cells underneath it, and a filled overlay
hides exactly that.
"""

from __future__ import annotations

import dataclasses

import numpy as np
from matplotlib.figure import Figure

from pygor.review.index import read_arrays
from pygor.review.rasterise import (
    DEFAULT_DPI,
    PYPLOT_LOCK,
    PanelCache,
    PanelImage,
    PanelKey,
    figure_to_png,
    hash_params,
    png_size,
)

PANELS: dict[str, "PanelSpec"] = {}

# Colour per role, so a recording looks the same wherever it is drawn.
ROLE_COLOURS = {"osds": "#ff7f0e", "swn": "#1f77b4", "fff": "#2ca02c"}

# How far a trimmed panel's shape may differ from the requested one before it is
# worth drawing again at its own shape. Below this the second render costs more
# than the pixels it wins back.
REFIT_TOLERANCE = 0.25

# Temporal kernels get this much of a spatial map's height, and this many pixels
# are set aside for titles and axis labels.
KERNEL_HEIGHT_RATIO = 0.8
PANEL_CHROME_PX = 70


@dataclasses.dataclass(frozen=True)
class PanelSpec:
    name: str
    level: str  # fov | recording | cell | cell_channel
    check: str  # alignment | rf_quality | trace_quality | registration | segmentation
    needs: tuple  # roles that must be fully loaded; () means light tier only
    fn: callable
    doc: str = ""
    threadsafe: bool = True
    #: Crop uniform margins after rasterising. Needed by panels that wrap a
    #: pygor plotting function laid out for a page rather than for a pane.
    trim: bool = False

    @property
    def light(self) -> bool:
        return not self.needs


def panel(*, name, level, check, needs=(), threadsafe=True, trim=False):
    """Register a renderer under ``name``."""

    def decorator(fn):
        PANELS[name] = PanelSpec(
            name=name,
            level=level,
            check=check,
            needs=tuple(needs),
            fn=fn,
            doc=(fn.__doc__ or "").strip().splitlines()[0] if fn.__doc__ else "",
            threadsafe=threadsafe,
            trim=trim,
        )
        return fn

    return decorator


def panels_for(level=None, check=None) -> list[PanelSpec]:
    return [
        spec
        for spec in PANELS.values()
        if (level is None or spec.level == level) and (check is None or spec.check == check)
    ]


# ---------------------------------------------------------------------------
# Drawing helpers
# ---------------------------------------------------------------------------


def new_figure(width, height, dpi=DEFAULT_DPI) -> Figure:
    """A figure at an exact pixel size, built without pyplot."""
    figure = Figure(figsize=(width / dpi, height / dpi), dpi=dpi)
    figure.set_facecolor("black")
    return figure


def _stretch(image, low=1, high=99):
    """Percentile-stretch for display. Flat images would otherwise divide by zero."""
    finite = np.asarray(image, dtype=float)
    valid = finite[np.isfinite(finite)]
    if valid.size == 0:
        return np.zeros_like(finite)
    lo, hi = np.percentile(valid, [low, high])
    if hi <= lo:
        return np.zeros_like(finite)
    return np.clip((finite - lo) / (hi - lo), 0, 1)


def roi_outlines(roi_mask):
    """Longest contour per ROI, in (row, col) order.

    ROI masks are IGOR convention -- background 1, ROIs -1, -2, -3 -- so the
    negative values are the cells.
    """
    from skimage import measure

    outlines = []
    mask = np.asarray(roi_mask)
    for roi_id in np.unique(mask):
        if roi_id >= 0:
            continue
        contours = measure.find_contours((mask == roi_id).astype(float), 0.5)
        if contours:
            outlines.append(max(contours, key=len))
    return outlines


def draw_outlines(axis, roi_mask, colour="yellow", linewidth=0.6, alpha=0.9):
    for contour in roi_outlines(roi_mask):
        axis.plot(contour[:, 1], contour[:, 0], color=colour, linewidth=linewidth,
                  alpha=alpha)


def square_pixels(figure):
    """Make every image axis keep its data's shape.

    A receptive field is a small pixel grid, and a figure sized to fill a
    terminal pane is whatever shape that pane is. Without this the images are
    stretched to the pane's aspect, which makes a round field look elliptical --
    the exact property a reviewer is judging. Trimming afterwards removes the
    margin this leaves behind.
    """
    for axis in figure.axes:
        if axis.images:
            axis.set_aspect("equal", adjustable="box")
    _match_band(figure)
    return figure


def _match_band(figure):
    """Pull line axes down to the band the images actually occupy.

    Square pixels shrink an image axis inside the box it was given, so the
    images end up as a short strip while a neighbouring trace axis still spans
    the whole figure. The empty band between them is not uniform -- the trace
    crosses it -- so trimming cannot remove it, and the panel arrives mostly
    black. Matching the extents first is what lets the trim work.
    """
    figure.canvas.draw()
    image_axes = [a for a in figure.axes if a.images]
    line_axes = [a for a in figure.axes if not a.images and a.lines]
    if not image_axes or not line_axes:
        return
    transform = figure.transFigure.inverted()
    extents = [transform.transform(a.get_window_extent().extents.reshape(2, 2))
               for a in image_axes]
    bottom = min(e[0][1] for e in extents)
    top = max(e[1][1] for e in extents)
    for axis in line_axes:
        position = axis.get_position()
        axis.set_position([position.x0, bottom, position.width, top - bottom])


def _bare(axis):
    axis.set_xticks([])
    axis.set_yticks([])
    for spine in axis.spines.values():
        spine.set_visible(False)


def message_figure(text, width, height, dpi=DEFAULT_DPI, colour="#ff6b6b") -> Figure:
    """A panel that says why it could not be drawn.

    A missing partner or an unreadable recording is itself something the
    reviewer needs to see, so it gets a panel rather than an exception.
    """
    figure = new_figure(width, height, dpi)
    axis = figure.add_subplot(111)
    axis.set_facecolor("black")
    _bare(axis)
    axis.text(0.5, 0.5, text, ha="center", va="center", color=colour, fontsize=9,
              wrap=True, transform=axis.transAxes)
    return figure


# ---------------------------------------------------------------------------
# Field-of-view panels: is the ROI transfer trustworthy
# ---------------------------------------------------------------------------


@panel(name="fov_alignment", level="fov", check="alignment")
def fov_alignment(bundle, *, width, height, dpi=DEFAULT_DPI, role=None, roi=None,
                  channel=-1, **params):
    """Each recording's projection with its own ROI outlines on top.

    Its own, not the master's: ``transfer_rois_from`` shifts the mask into the
    partner's frame, so the partner already holds a shifted copy. Drawing the
    master's unshifted mask here would render every successful transfer as an
    apparent misalignment. The question worth answering is whether the
    transferred outlines sit on cells in *this* recording, which is what
    ``use_master_mask`` lets you check against when a transfer looks wrong.
    """
    roles = list(bundle.roles)
    figure = new_figure(width, height, dpi)
    axes = figure.subplots(1, len(roles), squeeze=False)[0]
    master_mask = bundle.roi_mask(bundle.master_role) if params.get("use_master_mask") else None

    for axis, name in zip(axes, roles):
        axis.set_facecolor("black")
        _bare(axis)
        try:
            projection = bundle.projection(name)
        except (KeyError, OSError) as error:
            axis.text(0.5, 0.5, f"{name}\n{error}", ha="center", va="center",
                      color="#ff6b6b", fontsize=7, transform=axis.transAxes)
            continue
        axis.imshow(_stretch(projection), cmap="Greys_r", origin="lower")
        mask = master_mask if master_mask is not None else bundle.roi_mask(name)
        draw_outlines(axis, mask, colour=ROLE_COLOURS.get(name, "yellow"))

        info = bundle.alignment(name)
        if info.transferred:
            label = f"{name}  r={info.correlation:.3f}"
            if info.n_lost:
                label += f"  −{info.n_lost} ROI"
            colour = "#ff6b6b" if info.correlation < 0.8 else "white"
        else:
            label = f"{name}  (master)"
            colour = "white"
        axis.set_title(label, color=colour, fontsize=8, pad=3)
        if not info.usable:
            axis.text(0.5, 0.04, "provenance inconsistent", ha="center",
                      color="#ff6b6b", fontsize=7, transform=axis.transAxes)

    figure.suptitle(f"{bundle.fov_uid}  ·  {bundle.n_cells} cells",
                    color="white", fontsize=9)
    figure.tight_layout()
    return figure


@panel(name="fov_overlay", level="fov", check="alignment")
def fov_overlay(bundle, *, width, height, dpi=DEFAULT_DPI, role=None, roi=None,
                channel=-1, **params):
    """Master and partner projections as one red/green image; yellow is aligned.

    Same recipe as the napari ROI-transfer comparison, so the terminal and the
    viewer show the same picture at different resolutions.
    """
    partner = role or next((r for r in bundle.roles if r != bundle.master_role), None)
    if partner is None:
        return message_figure(
            f"{bundle.fov_uid}\nonly one recording — nothing to align against",
            width, height, dpi, colour="#999999",
        )

    figure = new_figure(width, height, dpi)
    axis = figure.add_subplot(111)
    axis.set_facecolor("black")
    _bare(axis)

    master = _stretch(bundle.projection(bundle.master_role))
    other = _stretch(bundle.projection(partner))
    if master.shape != other.shape:
        return message_figure(
            f"frame sizes differ: {master.shape} vs {other.shape}",
            width, height, dpi,
        )
    rgb = np.dstack([master, other, np.zeros_like(master)])
    axis.imshow(rgb, origin="lower")
    draw_outlines(axis, bundle.roi_mask(bundle.master_role), colour="white",
                  linewidth=0.4, alpha=0.5)

    info = bundle.alignment(partner)
    shift = ", ".join(f"{s:.2f}" for s in info.shift) if info.shift else "n/a"
    axis.set_title(
        f"{bundle.master_role} (red) vs {partner} (green)   "
        f"r={info.correlation:.3f}  shift=({shift})  lost={info.n_lost}",
        color="#ff6b6b" if info.correlation < 0.8 else "white", fontsize=8, pad=3,
    )
    figure.tight_layout()
    return figure


@panel(name="fov_lost_rois", level="fov", check="alignment")
def fov_lost_rois(bundle, *, width, height, dpi=DEFAULT_DPI, role=None, roi=None,
                  channel=-1, **params):
    """Which cells a transfer dropped, marked on the master projection."""
    partner = role or next((r for r in bundle.roles if r != bundle.master_role), None)
    if partner is None:
        return message_figure(f"{bundle.fov_uid}\nno partner recording",
                              width, height, dpi, colour="#999999")

    info = bundle.alignment(partner)
    figure = new_figure(width, height, dpi)
    axis = figure.add_subplot(111)
    axis.set_facecolor("black")
    _bare(axis)
    axis.imshow(_stretch(bundle.projection(bundle.master_role)), cmap="Greys_r",
                origin="lower")

    from skimage import measure

    mask = bundle.roi_mask(bundle.master_role)
    lost = {abs(int(i)) - 1 for i in info.lost_roi_ids}
    for roi_id in np.unique(mask):
        if roi_id >= 0:
            continue
        dropped = abs(int(roi_id)) - 1 in lost
        contours = measure.find_contours((mask == roi_id).astype(float), 0.5)
        if not contours:
            continue
        contour = max(contours, key=len)
        axis.plot(contour[:, 1], contour[:, 0],
                  color="#ff3333" if dropped else "#444444",
                  linewidth=1.2 if dropped else 0.4)

    axis.set_title(
        f"{partner}: {len(lost)} of {bundle.n_cells} cells lost in transfer",
        color="#ff6b6b" if lost else "white", fontsize=8, pad=3,
    )
    figure.tight_layout()
    return figure


# ---------------------------------------------------------------------------
# Recording panels: did this recording come out at all
# ---------------------------------------------------------------------------


@panel(name="rec_triggers", level="recording", check="registration")
def rec_triggers(bundle, *, width, height, dpi=DEFAULT_DPI, role=None, roi=None,
                 channel=-1, **params):
    """Trigger intervals, which catch the silent trigger-mode failures."""
    name = role or bundle.master_role
    ref = bundle.peek(name)
    arrays = read_arrays(ref.path, keys=("triggertimes",))
    times = arrays.get("triggertimes")
    if times is None or len(times) < 2:
        return message_figure(f"{name}: {0 if times is None else len(times)} triggers",
                              width, height, dpi)

    intervals = np.diff(np.asarray(times, dtype=float))
    figure = new_figure(width, height, dpi)
    axis = figure.add_subplot(111)
    axis.set_facecolor("black")
    axis.hist(intervals, bins=60, color=ROLE_COLOURS.get(name, "#cccccc"))
    axis.set_xlabel("inter-trigger interval (s)", color="white", fontsize=8)
    axis.set_ylabel("count", color="white", fontsize=8)
    axis.tick_params(colors="white", labelsize=7)
    for spine in axis.spines.values():
        spine.set_color("#666666")
    axis.set_title(
        f"{name}: {len(times)} triggers, mode={ref.trigger_mode}, "
        f"median {np.median(intervals):.3f} s  (cv {np.std(intervals)/np.mean(intervals):.3f})",
        color="white", fontsize=8, pad=3,
    )
    figure.tight_layout()
    return figure


@panel(name="rec_segmentation", level="recording", check="segmentation")
def rec_segmentation(bundle, *, width, height, dpi=DEFAULT_DPI, role=None, roi=None,
                     channel=-1, **params):
    """The recording's own ROI mask over its projection, outlines and numbered."""
    name = role or bundle.master_role
    arrays = bundle.light(name)
    figure = new_figure(width, height, dpi)
    axis = figure.add_subplot(111)
    axis.set_facecolor("black")
    _bare(axis)
    projection = arrays.get("average_stack")
    if projection is None:
        return message_figure(f"{name} has no average_stack", width, height, dpi)
    axis.imshow(_stretch(projection), cmap="Greys_r", origin="lower")

    mask = arrays.get("rois")
    if mask is not None:
        draw_outlines(axis, mask, colour="yellow", linewidth=0.5)
        if params.get("labels", True):
            for roi_id in np.unique(mask):
                if roi_id >= 0:
                    continue
                ys, xs = np.nonzero(mask == roi_id)
                axis.text(xs.mean(), ys.mean(), str(abs(int(roi_id)) - 1),
                          color="white", fontsize=4, ha="center", va="center")
    ref = bundle.peek(name)
    axis.set_title(f"{name}: {ref.num_rois} ROIs ({ref.roi_origin.get('method', '?')})",
                   color="white", fontsize=8, pad=3)
    figure.tight_layout()
    return figure


# ---------------------------------------------------------------------------
# Cell panels: is this receptive field real
# ---------------------------------------------------------------------------


@panel(name="cell_rf_chroma", level="cell", check="rf_quality", needs=("swn",),
       threadsafe=False, trim=True)
def cell_rf_chroma(bundle, *, width, height, dpi=DEFAULT_DPI, role="swn", roi=None,
                   channel=-1, **params):
    """One cell's receptive field in every colour, with its temporal kernels.

    Wraps ``strf.plotting.advanced.chroma_overview``. Its figure is taken as
    returned rather than by passing ``ax``: a non-None ``ax`` dispatches to an
    older renderer that draws something different.
    """
    import pygor.strf.plotting.advanced as advanced

    if roi is None:
        raise ValueError("cell_rf_chroma needs a roi")
    name = role or bundle.strf_role
    recording = bundle.full(name)
    n_colours = int(recording.n_colours or 1)
    # STRFs live in whichever index space the pipeline left them in; on a
    # transferred recording that is the master's, NaN-padded for lost cells.
    row = bundle.row_in(name, roi, recording.num_strfs // n_colours)
    if row is None:
        return message_figure(
            f"cell {roi} is not present in the {name} recording\n"
            "(dropped when ROIs were transferred)",
            width, height, dpi,
        )
    result = advanced.chroma_overview(
        recording,
        specify_rois=row,
        with_times=params.get("with_times", True),
        contours=params.get("contours", False),
        figsize=(width / dpi, height / dpi),
    )
    figure = result[0] if isinstance(result, tuple) else result
    figure.set_facecolor("black")
    square_pixels(figure)
    return figure


@panel(name="cell_rf", level="cell", check="rf_quality", needs=("swn",), trim=True)
def cell_rf(bundle, *, width, height, dpi=DEFAULT_DPI, role=None, roi=None,
            channel=-1, **params):
    """One cell's receptive field per colour, with the temporal kernels below.

    Laid out for a pane rather than for a page: ``chroma_overview`` puts every
    panel in a single row, which is a nine-to-one strip and comes out tiny once
    it is fitted into a terminal. A row of spatial maps over one shared kernel
    axis is close to three-to-one, so the same pane gives each map several times
    the area.
    """
    if roi is None:
        raise ValueError("cell_rf needs a roi")
    name = role or bundle.strf_role
    recording = bundle.full(name)
    n_colours = int(recording.n_colours or 1)
    row = bundle.row_in(name, roi, recording.num_strfs // n_colours)
    if row is None:
        return message_figure(
            f"cell {roi} is not in the {name} recording\n(dropped in ROI transfer)",
            width, height, dpi,
        )

    spatial = np.asarray(recording.collapse_times_chroma(roi=row))
    n_colours = spatial.shape[0]
    timecourses = np.asarray(recording.get_timecourses())
    if np.all(np.isnan(spatial)):
        return message_figure(
            f"cell {roi}: STRF is empty in the {name} recording",
            width, height, dpi,
        )

    from pygor.plotting.custom import fish_palette, maps_concat

    # Height follows from the receptive fields' own shape rather than from the
    # pane's. Accepting the pane's height and then squaring the pixels leaves the
    # maps floating in a band of empty figure, because the row keeps its
    # allocation whatever shape the image inside it ends up.
    rf_h, rf_w = spatial.shape[1], spatial.shape[2]
    image_height = (width / n_colours) * (rf_h / rf_w)
    kernel_height = image_height * KERNEL_HEIGHT_RATIO
    height = min(int(image_height + kernel_height + PANEL_CHROME_PX), height)

    figure = new_figure(width, height, dpi)
    grid = figure.add_gridspec(2, n_colours,
                               height_ratios=[image_height, kernel_height],
                               hspace=0.12, wspace=0.04)
    # Per channel, and robust. One scale for all four hides the weaker
    # channels in the middle grey of the colour map, and a single hot pixel
    # does the same to the channel it sits in. The reviewer needs to see
    # structure in each map; relative amplitude is in the kernels below.
    scale = params.get("scale", "channel")
    clip = float(params.get("clip_percentile", 99.5))
    shared = np.nanpercentile(np.abs(spatial), clip) if scale == "global" else None

    for colour in range(n_colours):
        axis = figure.add_subplot(grid[0, colour])
        axis.set_facecolor("black")
        _bare(axis)
        cmap = maps_concat[colour] if colour < len(maps_concat) else "bwr"
        limit = shared if shared else np.nanpercentile(np.abs(spatial[colour]), clip)
        limit = float(limit) if np.isfinite(limit) and limit > 0 else 1.0
        axis.imshow(spatial[colour], cmap=cmap, vmin=-limit, vmax=limit,
                    origin="lower", aspect="equal", interpolation="nearest")
        pass_col = f"strf_pass_bool_ch{colour}"
        cell_row = bundle.cell_row(roi)
        label = f"ch{colour}"
        colour_ok = None
        if cell_row is not None and pass_col in cell_row.index:
            colour_ok = bool(cell_row[pass_col])
            label += "  pass" if colour_ok else "  —"
        axis.set_title(label, fontsize=7, pad=2,
                       color="white" if colour_ok is not False else "#888888")

    kernel = figure.add_subplot(grid[1, :])
    kernel.set_facecolor("black")
    kernel.tick_params(colors="#888888", labelsize=6)
    for spine in kernel.spines.values():
        spine.set_color("#444444")
    times = np.linspace(-(recording.strf_dur_ms or 0) / 1000, 0, timecourses.shape[-1])
    for colour in range(n_colours):
        flat = row * n_colours + colour
        if flat >= len(timecourses):
            continue
        pair = timecourses[flat]
        shade = fish_palette[colour] if colour < len(fish_palette) else "white"
        # Negative and positive lobes are stored separately; both belong on the
        # same axis or a biphasic kernel reads as two different cells.
        for lobe in np.atleast_2d(pair):
            kernel.plot(times, lobe, color=shade, linewidth=0.9)
    kernel.axhline(0, color="#555555", linewidth=0.5)
    kernel.set_xlabel("time before spike (s)", color="#888888", fontsize=6)

    figure.suptitle(f"{bundle.fov_uid}#{roi}  ·  {name}", color="white", fontsize=8)
    return figure


@panel(name="fov_strf_sheet", level="fov", check="rf_quality", needs=("swn",),
       threadsafe=False, trim=True)
def fov_strf_sheet(bundle, *, width, height, dpi=DEFAULT_DPI, role="swn", roi=None,
                   channel=-1, **params):
    """Time-collapsed receptive fields for a page of this field of view's cells.

    A page, never the whole recording: a full overview lays out at roughly six
    inches by eighty-three, which is unreadable anywhere and impossible in a
    terminal. ``page`` steps through the cells a screenful at a time.
    """
    name = role or "swn"
    recording = bundle.full(name)
    per_page = int(params.get("per_page", 12))
    page = int(params.get("page", 0))

    if getattr(recording, "strfs", None) is None or not recording.num_strfs:
        return message_figure(f"{name} has no calculated STRFs", width, height, dpi)

    # Page over the cells that are really there. A transferred recording's
    # STRF array is NaN-padded for the cells it lost, and a fully masked row
    # makes the colour scaling refuse the whole page.
    n_colours = int(recording.n_colours or 1)
    n_rows = recording.num_strfs // n_colours
    present = [
        row
        for master in range(bundle.n_cells)
        if (row := bundle.row_in(name, master, n_rows)) is not None
    ]
    if not present:
        return message_figure(f"{name}: every cell was lost in transfer", width, height, dpi)
    n_cells = len(present)
    start = (page * per_page) % n_cells
    rois = present[start : start + per_page]
    result = recording.plot_strfs_space(
        roi=rois,
        channel=None if channel < 0 else channel,
        show_cbar=False,
        max_x=int(params.get("max_x", 4)),
    )
    figure = result[0] if isinstance(result, tuple) else result
    figure.set_facecolor("black")
    for axis in figure.axes:
        axis.set_title(axis.get_title(), color="white", fontsize=7)
    figure.suptitle(f"{n_cells} cells · page {page % max(-(-n_cells // per_page), 1) + 1}",
                    color="white", fontsize=8)
    square_pixels(figure)
    return figure


@panel(name="cell_rf_metrics", level="cell", check="rf_quality")
def cell_rf_metrics(bundle, *, width, height, dpi=DEFAULT_DPI, role=None, roi=None,
                    channel=-1, **params):
    """The CSV's own numbers for one cell, read rather than recomputed.

    Recomputing these would cost seconds per cell -- the pass flag alone is a
    four-second call -- and they are already columns in the aggregate table.
    """
    if roi is None:
        raise ValueError("cell_rf_metrics needs a roi")
    row = bundle.cell_row(roi)
    if row is None:
        return message_figure(f"cell {roi} has no row in the aggregate CSV",
                              width, height, dpi, colour="#999999")

    families = params.get(
        "families",
        ("strf_pass_bool", "strf_magnitudes", "strf_gaussian_fit_snr",
         "strf_areas", "strf_polarities_simple"),
    )
    n_colours = int(bundle.peek(bundle.master_role).n_colours or 4)

    figure = new_figure(width, height, dpi)
    axis = figure.add_subplot(111)
    axis.set_facecolor("black")
    _bare(axis)

    lines = [f"{bundle.fov_uid}#{roi}"]
    for family in families:
        values = []
        for ch in range(n_colours):
            value = row.get(f"{family}_ch{ch}")
            if value is None or (isinstance(value, float) and np.isnan(value)):
                values.append("  ·  ")
            elif isinstance(value, (bool, np.bool_)):
                values.append(" yes " if value else "  no ")
            elif isinstance(value, (int, float, np.number)):
                values.append(f"{float(value):5.2f}")
            else:
                values.append(str(value)[:5].rjust(5))
        lines.append(f"{family.removeprefix('strf_'):<22s} " + " ".join(values))
    for scalar in ("strf_categorysimple", "strf_spatial_corr_mean", "ipl"):
        if scalar in row.index:
            lines.append(f"{scalar.removeprefix('strf_'):<22s} {row[scalar]}")

    axis.text(0.02, 0.98, "\n".join(lines), ha="left", va="top", color="white",
              fontsize=7, family="monospace", transform=axis.transAxes)
    figure.tight_layout()
    return figure


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------


def _sources_for(bundle) -> tuple:
    return tuple(
        (role, ref.mtime, ref.size) for role, ref in sorted(bundle.refs.items())
    )


def render(name, bundle, *, width=1000, height=600, dpi=DEFAULT_DPI, roi=None,
           channel=-1, role=None, cache: PanelCache | None = None,
           **params) -> PanelImage:
    """Render a named panel, via the cache when one is supplied."""
    try:
        spec = PANELS[name]
    except KeyError as error:
        raise KeyError(f"no panel named {name!r}; have {sorted(PANELS)}") from error

    key = PanelKey(
        panel=name,
        fov_uid=bundle.fov_uid,
        condition=bundle.condition,
        role=role or "",
        roi=roi,
        channel=int(channel),
        width=width,
        height=height,
        dpi=dpi,
        params_hash=hash_params(params),
        sources=_sources_for(bundle),
    )
    if cache is not None:
        hit = cache.get(key)
        if hit is not None:
            return hit

    def build(w, h):
        return spec.fn(bundle, width=w, height=h, dpi=dpi, roi=roi,
                       channel=channel, role=role, **params)

    def rasterise(w, h):
        if spec.threadsafe:
            figure = build(w, h)
        else:
            with PYPLOT_LOCK:
                figure = build(w, h)
        # No size forced here: the panel was given the target and may have
        # chosen a shorter figure to suit its content. Overriding that would
        # stretch the layout back out and undo the choice.
        return figure_to_png(figure, dpi=dpi, trim=spec.trim)

    png = rasterise(width, height)

    # A trimmed panel reports the shape its content actually wants, which is
    # rarely the shape of the pane it was asked for. Drawing a wide strip into a
    # tall figure leaves the content in a thin band and wastes most of the
    # pixels, so measure once and draw again at the right shape.
    if spec.trim:
        actual = png_size(png)
        if actual[0] and actual[1]:
            natural = actual[0] / actual[1]
            if abs(natural - width / height) / natural > REFIT_TOLERANCE:
                refit_height = max(int(round(width / natural)), 1)
                if refit_height > height:
                    png = rasterise(max(int(round(height * natural)), 1), height)
                else:
                    png = rasterise(width, refit_height)
    # The real size, not the requested one: trimming changes it, and a caller
    # scaling this into a pane needs the true aspect or the panel comes out
    # stretched.
    actual_width, actual_height = png_size(png)
    image = PanelImage(
        png=png,
        width=actual_width or width,
        height=actual_height or height,
        key=key,
        meta={"panel": name, "level": spec.level, "check": spec.check,
              "trimmed": spec.trim, "requested": (width, height)},
    )
    return cache.put(image) if cache is not None else image
