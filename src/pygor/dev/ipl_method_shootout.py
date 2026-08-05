# %%
"""Run every IPL-depth method pygor has on one recording and compare the output.

Three ways to get boundaries currently exist, and they disagree about which
argument is 0 %:

    rois     estimate_ipl_boundaries()          percentiles of the ROI cloud
    anatomy  estimate_ipl_boundaries_anatomy()  intensity band in average_stack
    manual   NapariDepthPrompt                  two hand-drawn polylines

All three feed calculate_ipl_depths(centroids, upper_boundary, lower_boundary),
which computes (roi - upper) / (lower - upper), i.e. upper_boundary = 0 %.
Three committed call sites assume the opposite (test_ipl.py:86-88,
NapariDepthPrompt.process_data, plot_ipl_estimation's tick labels), so at least
one of them is inverted and nothing currently proves which.

Lab convention this has to end up matching:

    0 %    ON layer,  proximal, GCL side
    100 %  OFF layer, distal,   INL side (toward photoreceptors)
    ON/OFF border around 40 %

Works either as a plain script or as VS Code cells. As a script, the napari
cell blocks until you close the window; in cells under %gui qt it does not, so
draw with the window left open and run the next cell. NapariDepthPrompt.run()
is avoided in both cases because it deadlocks on a second QEventLoop (see the
comment on that cell). Later cells draw the histograms.

Nothing is written back into the .recording.h5.
"""

import pathlib

import matplotlib.pyplot as plt
import numpy as np

import pygor.load
from pygor.anatomy.ipl import (
    calculate_ipl_depths,
    estimate_ipl_boundaries,
    estimate_ipl_boundaries_anatomy,
    interp_boundary,
)
from pygor.core.gui.methods import _import_napari

REC_PATH = pathlib.Path(
    "/mnt/data/BC paper data/SWN BC aggregation/Processed/Control/"
    "260605_3_1_SWN_200_RGBUVAll.recording.h5"
)
# Hand-drawn boundaries are cached here so the plotting cells can be re-run
# without redrawing them every time.
MANUAL_CACHE = pathlib.Path("ipl_shootout_manual.npz")
# Run as a plain script there is no inline rendering and no plt.show(), so the
# figures are written here too.
FIG_DIR = pathlib.Path("ipl_shootout_figures")

N_BINS = 10
DEPTH_RANGE = (0.0, 100.0)

ROI_KWARGS = {"n_bins": 8, "upper_percentile": 0.0, "lower_percentile": 100.0}
ANATOMY_KWARGS = {"n_bins": 16, "k_sigma": 2.0, "border_crop": 6}


# %%
rec = pygor.load.STRF.load_object(REC_PATH)
centroids = np.asarray(rec.roi_centroids)
image = np.asarray(rec.average_stack, dtype=float)

print(f"{rec.name}: {rec.num_rois} ROIs, average_stack {image.shape}")
print(f"centroid spread  y {np.ptp(centroids[:, 0]):.0f} px   x {np.ptp(centroids[:, 1]):.0f} px")
print(f"ipl_depths in file: {rec.ipl_depths if rec.ipl_depths is None else 'set'}")

# Every method lands in here as name -> (upper_boundary, lower_boundary, orientation).
# Keeping the boundaries rather than just the depths is the point: the depths
# follow from them plus a polarity choice, and the polarity choice is what is
# in dispute.
boundaries = {}


# %%
# Method 1 -- ROI cloud percentiles.
# Expected to be poor here (non-signal ROIs were culled, so the extremes that
# define the band are missing) and it may raise outright if too few ROIs land
# in each bin. Included anyway so the failure mode is on the record.
try:
    upper, lower = estimate_ipl_boundaries(centroids, **ROI_KWARGS)
    boundaries["rois"] = (upper, lower, None)
    print("rois: ok")
except Exception as exc:
    print(f"rois: FAILED -- {type(exc).__name__}: {exc}")


# %%
# Method 2 -- intensity band in the anatomy.
upper, lower, orientation = estimate_ipl_boundaries_anatomy(image, **ANATOMY_KWARGS)
boundaries["anatomy"] = (upper, lower, orientation)
depth_axis = 0 if orientation == "horizontal" else 1
print(f"anatomy: orientation={orientation}, depth along {'y' if depth_axis == 0 else 'x'}")
print(f"  upper_boundary mean depth-coord {upper[:, depth_axis].mean():.1f}")
print(f"  lower_boundary mean depth-coord {lower[:, depth_axis].mean():.1f}")


# %%
# Method 3a -- open the viewer and draw.
#
# NapariDepthPrompt.run() is deliberately NOT used here: it calls napari.run()
# (which blocks until the window closes) and then event_loop.exec_() (a second
# QEventLoop). The close handler calls event_loop.quit() while that loop has
# never been started, so the quit is a no-op and exec_() then blocks with no
# window left to close it. It hangs immediately after printing "Processing user
# selection...". It only survives under %gui qt, where napari.run() returns
# straight away and the ordering happens to work out.
#
# A single napari.run() and no QEventLoop works in both contexts. As a script
# it blocks here until you close the window, and the polylines are snapshotted
# the moment it returns. Under %gui qt it returns immediately, the window stays
# open, and the next cell does the reading instead.
if MANUAL_CACHE.exists():
    cached = np.load(MANUAL_CACHE)
    drawn_0pct, drawn_100pct = cached["drawn_0pct"], cached["drawn_100pct"]
    print(f"loaded hand-drawn boundaries from {MANUAL_CACHE} (delete it to redraw)")
    viewer = None
else:
    napari, _, _ = _import_napari()
    viewer = napari.Viewer()
    viewer.add_image(image, name="Average stack", colormap="Greys_r")
    viewer.add_image(rec.rois_alt, name="ROIs", colormap="rainbow", opacity=0.25)
    viewer.add_points(centroids, name="ROI centroids", opacity=1,
                      face_color="orange", size=1.5)
    # Same layer names as NapariDepthPrompt so what you draw here transfers.
    viewer.add_shapes(name="100% boundary", edge_color="red").mode = "add_polyline"
    viewer.add_shapes(name="0% boundary", edge_color="blue").mode = "add_polyline"
    print("Draw '0% boundary' (ON/GCL side) and '100% boundary' (OFF/INL side).")
    # As a plain script this blocks until the window is closed, and the drawn
    # data stays on the viewer object afterwards. Under %gui qt (VS Code cells
    # with an active Qt loop) it returns straight away instead, so there the
    # window stays open and the next cell does the reading -- both routes end
    # up at the same harvest below.
    napari.run()
    # Snapshot straight after the blocking run returns, while the layers are
    # certainly still alive -- don't rely on the viewer surviving the window.
    if len(viewer.layers["0% boundary"].data) and len(viewer.layers["100% boundary"].data):
        raw_0pct = np.squeeze(viewer.layers["0% boundary"].data[-1])
        raw_100pct = np.squeeze(viewer.layers["100% boundary"].data[-1])
        viewer = None


# %%
# Method 3b -- read the drawn polylines back.
if viewer is not None:
    if not len(viewer.layers["0% boundary"].data) or not len(viewer.layers["100% boundary"].data):
        raise RuntimeError(
            "One or both boundaries are empty. As a script: draw both, then close "
            "the napari window to continue. In cells: draw both with the window "
            "left open, then re-run this cell."
        )
    raw_0pct = np.squeeze(viewer.layers["0% boundary"].data[-1])
    raw_100pct = np.squeeze(viewer.layers["100% boundary"].data[-1])

if not MANUAL_CACHE.exists():
    # interp_boundary is what NapariDepthPrompt applies on layer switch; doing it
    # explicitly makes the result independent of which layer happened to be
    # active when you stopped drawing.
    drawn_0pct = interp_boundary(raw_0pct)
    drawn_100pct = interp_boundary(raw_100pct)
    np.savez(MANUAL_CACHE, drawn_0pct=drawn_0pct, drawn_100pct=drawn_100pct)
    print(f"captured and cached to {MANUAL_CACHE}")

print(f"0% boundary   {drawn_0pct.shape}, mean y {drawn_0pct[:, 0].mean():.1f}")
print(f"100% boundary {drawn_100pct.shape}, mean y {drawn_100pct[:, 0].mean():.1f}")

# Passed the way the layer names read: the "0% boundary" layer becomes the 0 %
# argument. This is NOT what NapariDepthPrompt does today.
boundaries["manual"] = (drawn_0pct, drawn_100pct, None)


# %%
# Depths for every method through the one shared endpoint, unclipped so
# out-of-range values stay visible rather than piling onto the rails.
depths = {}
for name, (upper, lower, orientation) in boundaries.items():
    d = np.asarray(
        calculate_ipl_depths(centroids, upper, lower, orientation=orientation),
        dtype=float,
    )
    depths[name] = d
    n_out = int(np.sum((d < 0) | (d > 100)))
    print(f"{name:8s} min {np.nanmin(d):7.1f}  median {np.nanmedian(d):6.1f}  "
          f"max {np.nanmax(d):7.1f}  {n_out}/{len(d)} outside [0,100]")

# What NapariDepthPrompt actually returns today, for the record: it passes the
# "100% boundary" layer as upper_boundary, so its output is the mirror of
# depths["manual"] above.
depths_gui_asis = np.asarray(
    calculate_ipl_depths(centroids, drawn_100pct, drawn_0pct),
    dtype=float,
)
print(f"\nNapariDepthPrompt as-implemented median: {np.nanmedian(depths_gui_asis):.1f} "
      f"(hand-drawn labels give {np.nanmedian(depths['manual']):.1f})")


# %%
def depth_histogram(ax, values, title):
    """10-bin depth histogram, depth on Y with 0 % at the bottom."""
    values = np.asarray(values, dtype=float)
    finite = values[np.isfinite(values)]
    counts, edges = np.histogram(finite, bins=N_BINS, range=DEPTH_RANGE)
    centres = (edges[:-1] + edges[1:]) / 2
    ax.barh(centres, counts, height=np.diff(edges), color="steelblue",
            edgecolor="white", linewidth=0.5)
    # The ON/OFF border, as the anatomical anchor to read the shape against.
    ax.axhline(40, color="k", ls="--", lw=1)
    n_out = int(np.sum((finite < 0) | (finite > 100)))
    ax.set_ylim(*DEPTH_RANGE)
    ax.set_title(f"{title}\nmedian {np.nanmedian(finite):.0f} %, {n_out} outside range",
                 fontsize=9)
    ax.set_xlabel("ROIs")


order = [k for k in ("rois", "anatomy", "manual") if k in depths]

fig, axes = plt.subplots(1, len(order), figsize=(3.2 * len(order), 4), sharey=True)
axes = np.atleast_1d(axes)
for ax, name in zip(axes, order):
    depth_histogram(ax, depths[name], name)
axes[0].set_ylabel("IPL depth (%)   0 = ON/GCL, 100 = OFF/INL")
fig.suptitle(f"{rec.name} -- IPL depth by method", fontsize=11)
fig.tight_layout()
FIG_DIR.mkdir(exist_ok=True)
fig.savefig(FIG_DIR / "1_depth_by_method.png", dpi=130, bbox_inches="tight")


# %%
# Same three, mirrored. If one of these is the shape you expect rather than its
# counterpart above, the polarity is inverted somewhere and this says where.
fig, axes = plt.subplots(1, len(order), figsize=(3.2 * len(order), 4), sharey=True)
axes = np.atleast_1d(axes)
for ax, name in zip(axes, order):
    depth_histogram(ax, 100.0 - depths[name], f"{name} (mirrored)")
axes[0].set_ylabel("IPL depth (%)   0 = ON/GCL, 100 = OFF/INL")
fig.suptitle(f"{rec.name} -- polarity mirror, for reference", fontsize=11)
fig.tight_layout()
fig.savefig(FIG_DIR / "2_polarity_mirror.png", dpi=130, bbox_inches="tight")


# %%
# Where each method actually put its boundaries. The scatter is coloured by
# depth, so the 0 % end is whichever side the dark points sit on.
fig, axes = plt.subplots(1, len(order), figsize=(4.2 * len(order), 3.4))
axes = np.atleast_1d(axes)
for ax, name in zip(axes, order):
    upper, lower, _ = boundaries[name]
    ax.imshow(image, cmap="Greys_r", origin="lower")
    ax.plot(upper[:, 1], upper[:, 0], color="tab:blue", lw=2, label="0 % arg")
    ax.plot(lower[:, 1], lower[:, 0], color="tab:red", lw=2, label="100 % arg")
    sc = ax.scatter(centroids[:, 1], centroids[:, 0], c=depths[name],
                    cmap="coolwarm", vmin=0, vmax=100, s=16,
                    edgecolors="k", linewidths=0.3)
    ax.set_title(name, fontsize=9)
    ax.set_xlim(0, image.shape[1])
    ax.set_ylim(0, image.shape[0])
    ax.set_xticks([])
    ax.set_yticks([])
axes[0].legend(fontsize=7, loc="upper left")
fig.colorbar(sc, ax=axes, label="depth (%)", fraction=0.02)
fig.suptitle(f"{rec.name} -- boundaries as passed (origin='lower')", fontsize=11)
fig.savefig(FIG_DIR / "3_boundaries.png", dpi=130, bbox_inches="tight")
print(f"figures written to {FIG_DIR.resolve()}")
plt.show()
