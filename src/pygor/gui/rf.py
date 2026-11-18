"""Receptive-field data and drawing shared by the RF view and the RF panel.

Everything here reads what the STRF object already holds or computes with its
own getters: the border-cropped STRFs, their time-collapsed maps, and the
Gaussian fit. The fit is drawn as two ellipses for assessing it by eye; no
call is made here on whether an RF is Gaussian.
"""

import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Ellipse


class RFData:
    """One recording's maps, movies and fits, refetched when its STRFs change.

    Fitting a recording takes a second or more, so it is done once, on first
    use, rather than per ROI click. Call :meth:`refresh` before reading.
    """

    def __init__(self, recording):
        self.recording = recording
        self._key = None

    def _load(self):
        strfs = self.recording.strfs
        if self._key is strfs:
            return
        from pygor.strf import gaussian_fit

        rec = self.recording
        self.n_colours = int(getattr(rec, "n_colours", 1) or 1)
        # Border removed, like the collapsed maps, so the fit's coordinates
        # hold on every movie frame
        self.movies = np.asarray(rec.strfs_no_border, dtype=float)
        self.maps = np.ma.filled(np.ma.asarray(rec.collapse_times()).astype(float), np.nan)
        self.fits = rec.calc_gaussian_fit_index()
        self.edge = np.asarray(rec.get_rf_peak_edge_distance(), dtype=float)
        self.n_sigma = gaussian_fit._resolve(rec, None, None, None)[0]
        self._key = strfs

    def refresh(self):
        """Load, or refetch if the STRFs were recomputed; call before reading."""
        self._load()
        return self

    @property
    def n_rois(self):
        return len(self.movies) // self.n_colours

    def kernels(self, i):
        """Centre and surround temporal kernels of flat STRF ``i``, (2, n_frames).

        From the recording's own centre-surround segmentation
        (``get_timecourses``), computed for every STRF on first use and kept,
        so only the first ROI shown waits for it. None if it cannot be had.
        """
        if getattr(self, "_kernels_key", None) is not self._key:
            try:
                self._kernels = np.asarray(self.recording.get_timecourses(), dtype=float)
            except Exception:  # an STRF the segmentation cannot handle
                self._kernels = None
            self._kernels_key = self._key
        if self._kernels is None or i >= len(self._kernels):
            return None
        return np.atleast_2d(self._kernels[i])

    def frame_times_ms(self, n_frames=None):
        """Each STRF frame's time relative to the response, in ms, or None.

        The same mapping as :meth:`frame_label`, so a kernel and the movie
        agree on when a frame is.
        """
        n_frames = n_frames or self.movies.shape[1]
        dur_ms = getattr(self.recording, "strf_dur_ms", None)
        try:
            past_s = self.recording.params.get_defaults("strf")["calculate"][
                "sta_past_window"
            ]
        except (AttributeError, KeyError, TypeError, ValueError):
            return None
        if not dur_ms:
            return None
        return np.arange(n_frames) * dur_ms / n_frames - past_s * 1000

    def flat(self, roi, colour):
        """Flat STRF index of an ROI's colour channel."""
        return roi * self.n_colours + colour

    def frame_label(self, frame):
        """Time of a movie frame relative to the response, when it is known.

        The saved object does not record the STA window it was computed
        with, so the past window comes from the config.
        """
        n_frames = self.movies.shape[1]
        dur_ms = getattr(self.recording, "strf_dur_ms", None)
        try:
            past_s = self.recording.params.get_defaults("strf")["calculate"][
                "sta_past_window"
            ]
        except (AttributeError, KeyError, TypeError, ValueError):
            past_s = None
        if not dur_ms or past_s is None:
            return f"frame {frame + 1}/{n_frames}"
        return f"{frame * dur_ms / n_frames - past_s * 1000:+.0f} ms"

    def frame_interval_ms(self):
        """Real-time playback interval, or about 15 frames a second."""
        n_frames = self.movies.shape[1]
        dur_ms = getattr(self.recording, "strf_dur_ms", None) or n_frames * 66
        return max(30, int(dur_ms / n_frames))


# Pixels of blank between the tiles of a mosaic
GAP = 1


def normalised(stack):
    """Scale each ROI's map or movie by its own largest absolute value.

    ``stack`` is (n_rois, ...); every ROI's values end up in [-1, 1]. An ROI
    with nothing finite or all zeros is left as it is.
    """
    stack = np.asarray(stack, dtype=float)
    flat = np.abs(np.nan_to_num(stack)).reshape(len(stack), -1)
    peak = flat.max(axis=1) if flat.size else np.ones(len(stack))
    peak[peak == 0] = 1.0
    return stack / peak.reshape(-1, *([1] * (stack.ndim - 1)))


def mosaic(tiles, n_cols):
    """Lay (n, h, w) tiles out in rows of ``n_cols``, NaN in the gaps."""
    n, h, w = tiles.shape
    n_rows = int(np.ceil(n / n_cols))
    padded = np.full((n_rows * n_cols, h + GAP, w + GAP), np.nan)
    padded[:n, :h, :w] = tiles
    grid = padded.reshape(n_rows, n_cols, h + GAP, w + GAP)
    grid = grid.transpose(0, 2, 1, 3).reshape(n_rows * (h + GAP), n_cols * (w + GAP))
    return grid[:-GAP, :-GAP]


def tile_origin(roi, tile_shape, n_cols):
    """Top-left (x, y) of an ROI's tile in a :func:`mosaic`, in image pixels."""
    h, w = tile_shape
    row, col = divmod(roi, n_cols)
    return col * (w + GAP), row * (h + GAP)


def symmetric_limit(values):
    """Largest absolute finite value, for a colour scale centred on zero."""
    values = np.asarray(values, dtype=float)
    finite = np.isfinite(values)
    limit = float(np.abs(values[finite]).max()) if finite.any() else 0.0
    return limit or 1.0


def draw_map(ax, image, limit):
    """A signed map on a symmetric diverging scale, without ticks."""
    artist = ax.imshow(image, cmap="RdBu_r", vmin=-limit, vmax=limit,
                       interpolation="nearest")
    ax.set_xticks([])
    ax.set_yticks([])
    return artist


def draw_fit(ax, data, i, offset=(0, 0)):
    """Fitted Gaussian (solid) and scored footprint (dashed), at n_sigma.

    ``offset`` places them on a tile of a mosaic. Returns the patches drawn.
    """
    fits = data.fits
    patches = []
    for prefix, style in (("", "-"), ("footprint_", "--")):
        sx, sy = fits[f"{prefix}sx"][i], fits[f"{prefix}sy"][i]
        if not (np.isfinite(sx) and np.isfinite(sy)):
            continue
        patches.append(ax.add_patch(Ellipse(
            (fits[f"{prefix}x0"][i] + offset[0], fits[f"{prefix}y0"][i] + offset[1]),
            2 * data.n_sigma * sx,
            2 * data.n_sigma * sy,
            angle=np.degrees(fits[f"{prefix}theta"][i]),
            fill=False, lw=1, ls=style, color="k",
        )))
    return patches


def fit_legend(n_sigma):
    """What the two ellipses are."""
    return [
        Line2D([], [], color="k", lw=1, ls="-",
               label=f"Gaussian fitted to |RF|, {n_sigma:g} σ"),
        Line2D([], [], color="k", lw=1, ls="--",
               label=f"region the fit index is scored over, {n_sigma:g} σ"),
    ]
