# %%
import pygor
import pygor.load

import matplotlib.pyplot as plt
import numpy as np

%load_ext autoreload
%autoreload 2

# %%
stimulus_array = np.load("/home/simen/Noise_npy_arrs/9deg_200_SINGLEcolour_30000x6x10_0.25_1.npy")
example_recording_path = "/mnt/data/Igor analyses/OSDS_FF_RF/260306 OSDS RF FF/0_1_SWN_200_RGBUVAll.recording.h5"
example_recording_path = "/mnt/data/Igor analyses/OSDS/251104 OSDS/0_1_SWN_200_White.recording.h5"
example_recording_path = "/mnt/data/Igor analyses/OSDS/251103 OSDS/2_0_SWN_200_White.recording.h5"
obj = pygor.load.STRF.load_object(example_recording_path)
# %%
obj.plot_strfs_space()


# %% ---------------------------------------------------------------------------
# Event-triggered snippet analysis (prototype).
#
# Goal: for a chosen ROI, detect response events, find the STRF peak pixel, and
# pull out the stimulus history at that pixel for every event -> an
# (n_events, n_lags) "SnippetLog" matrix. This is the raw material the STA
# averages over; the point is to keep the events *separate* so later we can
# split co-incident event types instead of collapsing to one linear filter.
#
# CORRECTNESS: the stimulus->imaging-frame mapping is copied verbatim from
# pygor.strf.calculate_strf (lines ~471-513) so events and pixel-history land on
# exactly the frames the STRF was built from. Do not reinvent this alignment.
#
# Single white colour channel only (v1). The `# COLOUR-SEAM` comments mark where
# n_colours / colour_lookup handling (per calculate_strf) would slot in later.


def build_frame_to_pattern(
    obj,
    noise_array,
    skip_first_triggers: int = 0,
    skip_last_triggers: int = 0,
    max_frames_per_trigger: int = 100,
):
    """Frame-precise stimulus mapping, in 'relevant-frame' coordinates.

    Returns
    -------
    frame_to_pattern : (n_f_relevant,) int
        Pattern index (column into noise_array[..., k]) held at each relevant
        frame. -1 means no pattern was written (STRF held these at the 0.5
        baseline).
    trigger_start : int
        Absolute imaging frame of the first used trigger. Relevant frame f
        corresponds to absolute frame f + trigger_start.
    n_f_relevant : int

    This walks triggertimes_frame exactly like calculate_strf: hold each noise
    pattern from one trigger to the next, cap the gap at max_frames_per_trigger,
    and cycle trigger_counter through the pattern stack (incl. the +=1 on the
    out-of-bounds `continue`, which matters for phase).
    """
    ttf = obj.triggertimes_frame.copy()
    nan_mask = np.isnan(ttf)
    n_triggers = int(np.argmax(nan_mask)) if np.any(nan_mask) else len(ttf)

    trigger_start = int(ttf[skip_first_triggers])
    n_f_relevant = int(
        ttf[n_triggers - skip_last_triggers - 1] - ttf[skip_first_triggers]
    )
    n_patterns = noise_array.shape[2]

    frame_to_pattern = np.full(n_f_relevant, -1, dtype=np.int64)

    trigger_counter = 0
    for tt in range(skip_first_triggers, n_triggers - skip_last_triggers - 1):
        current_start_frame = int(ttf[tt]) - trigger_start
        current_end_frame = int(ttf[tt + 1]) - trigger_start

        if current_start_frame < 0 or current_end_frame >= n_f_relevant:
            trigger_counter += 1
            if trigger_counter >= n_patterns:
                trigger_counter = 0
            continue

        if current_end_frame - current_start_frame > max_frames_per_trigger:
            current_end_frame = current_start_frame + max_frames_per_trigger

        if trigger_counter < n_patterns:
            frame_range = np.arange(current_start_frame, current_end_frame + 1)
            frame_range = frame_range[frame_range < n_f_relevant]
            if len(frame_range) > 0:
                frame_to_pattern[frame_range] = trigger_counter

        trigger_counter += 1
        if trigger_counter >= n_patterns:
            trigger_counter = 0

    return frame_to_pattern, trigger_start, n_f_relevant


def detect_events(
    obj,
    roi: int,
    threshold: float = 1.75,
    sign: str = "pos",
    trigger_start: int = 0,
    n_f_relevant: int | None = None,
    use_znorm: bool = True,
):
    """Boss's event detector: z-scored temporal derivative crossings.

    Baseline mean/std taken from the first ~100 frames of the relevant window,
    matching the event-count logic in calculate_strf._process_single_roi.

    Parameters
    ----------
    sign : {'pos', 'neg'}
        'pos' -> depolarising events (derivative > +threshold).
        'neg' -> hyperpolarising events (derivative < -threshold). Bipolars are
        bidirectional; 'neg' is here for that, not yet scientifically validated.

    Returns
    -------
    event_frames : (n_events,) int
        Event locations in *relevant-frame* coordinates.
    event_amps : (n_events,) float
        z-scored derivative amplitude at each event (signed).
    """
    traces = obj.traces_znorm if use_znorm else obj.traces_raw  # (rois, frames)
    trace = np.asarray(traces[roi], dtype=float)
    if n_f_relevant is None:
        n_f_relevant = trace.shape[0] - trigger_start
    seg = trace[trigger_start : trigger_start + n_f_relevant].copy()

    dif = np.diff(seg, prepend=seg[0])
    baseline_points = min(100, n_f_relevant)
    base = dif[:baseline_points]
    if np.std(base) == 0:
        return np.array([], dtype=int), np.array([], dtype=float)
    dif = (dif - np.mean(base)) / np.std(base)

    if sign == "pos":
        mask = dif > threshold
    elif sign == "neg":
        mask = dif < -threshold
    else:
        raise ValueError("sign must be 'pos' or 'neg'")

    event_frames = np.flatnonzero(mask)
    return event_frames, dif[event_frames]


def peak_pixel_neighbourhood(obj, roi: int, n_neighbours: int = 4, edge_crop: int = 2):
    """Peak pixel of the linear STRF plus its 4-connected neighbours.

    Peak = spatial argmax of |STRF| across all lags. Returns pixel indices in
    STRF spatial coords (i_x in [0, n_x_noise), i_y in [0, n_y_noise)), which map
    to noise_array as noise_array[i_y, i_x, pattern].

    n_neighbours : arms of the cross to include beyond the centre (default 4 =
    up/down/left/right, i.e. the boss's 5-px cross). Out-of-bounds / edge_crop
    border arms are dropped.
    """
    strf = np.asarray(obj.strfs[roi])            # (time, x, y)
    proj = np.max(np.abs(strf), axis=0)          # (x, y) peak-|amp| over lags
    px, py = (int(v) for v in np.unravel_index(np.argmax(proj), proj.shape))  # i_x, i_y

    nx, ny = proj.shape
    offsets = [(-1, 0), (1, 0), (0, -1), (0, 1)][:n_neighbours]
    pixels = [(px, py)]
    for dx, dy in offsets:
        qx, qy = px + dx, py + dy
        if edge_crop <= qx < nx - edge_crop and edge_crop <= qy < ny - edge_crop:
            pixels.append((qx, qy))
    return (px, py), pixels


def extract_snippets(
    noise_array,
    frame_to_pattern,
    event_frames,
    pixels,
    n_f_filter_past: int,
    n_f_filter: int,
    baseline: float = 0.5,
):
    """Per-event pixel history: (n_events, n_lags) contrast matrix ('SnippetLog').

    For each event at relevant-frame f and each STRF lag tau, take the stimulus
    at frame f+tau, averaged over the peak-pixel neighbourhood, minus `baseline`.
    Frames with no pattern (or off the ends of the relevant window) sit at the
    baseline -> contrast 0, matching the STRF's 0.5 hold value.

    Lag convention copied from calculate_strf: tau = (1 - n_f_filter_past) + j,
    so column 0 is the deepest past lag and negative lags = past.
    """
    taus = np.arange(n_f_filter) + (1 - n_f_filter_past)
    n_rel = len(frame_to_pattern)

    # neighbourhood pixel values across ALL patterns: (n_pix, n_patterns)
    # COLOUR-SEAM: noise_array would gain a colour axis; select per colour_lookup.
    pix_vals = np.stack(
        [noise_array[iy, ix, :].astype(np.float64) for (ix, iy) in pixels], axis=0
    )
    pix_mean_per_pattern = pix_vals.mean(axis=0)  # (n_patterns,)

    snippets = np.full((len(event_frames), n_f_filter), baseline, dtype=np.float64)
    for e, f in enumerate(event_frames):
        frames = f + taus
        valid = (frames >= 0) & (frames < n_rel)
        pats = frame_to_pattern[frames[valid]]
        has_pat = pats >= 0
        col = np.full(n_f_filter, baseline)
        idx = np.flatnonzero(valid)[has_pat]
        col[idx] = pix_mean_per_pattern[pats[has_pat]]
        snippets[e] = col

    return snippets - baseline, taus


def strf_window_from_obj(obj):
    """Derive the STA lag window from the object, no hardcoded 2s/2s.

    Returns
    -------
    frame_duration : float   (s)  = linedur_s * n_lines
    n_f_filter : int              = obj.strfs.shape[1] (definitive lag count)
    n_f_filter_past : int         past-lag count, so tau = (1 - past) + j

    The object stores the *total* window (strf_dur_ms) but not the past/future
    split. pygor's default window is symmetric, so past = total/2 recovers the
    split; n_f_filter itself comes straight from strfs.shape (no assumption).
    """
    frame_duration = obj.linedur_s * obj.images.shape[1]
    n_f_filter = int(obj.strfs.shape[1])
    total_s = getattr(obj, "strf_dur_ms", n_f_filter * frame_duration * 1000) / 1000.0
    n_f_filter_past = max(1, int(np.floor((total_s / 2.0) / frame_duration)))
    n_f_filter_past = min(n_f_filter_past, n_f_filter - 1)  # keep >=1 future lag
    return frame_duration, n_f_filter, n_f_filter_past


def plot_snippet_log(
    snippets, taus, frame_duration, roi, sign="pos",
    sort="pre_event", pre_window_s=0.3, ignore_frac=0.0, split_at="median", ax=None,
):
    """Heatmap of the SnippetLog.

    Layout: events along x (there are always many more events than lags), STRF
    lag along y. Greyscale contrast: dark = OFF (below 0.5), light = ON.

    sort : {'incidence', 'pre_event'}
        'incidence'  -> natural/chronological order (drift over recording).
        'pre_event'  -> sort by mean contrast in the pre-event window
                        [-pre_window_s, 0) s, descending, so 'up' (ON-driven)
                        events sit on the LEFT and 'down' (OFF-driven) on the
                        RIGHT. This is the boss's up/down split as a sort key.
    pre_window_s : float
        Width of the pre-event averaging window for the 'pre_event' sort. The
        linear RF peaks ~-0.26 s, so ~0.3 s captures "just before the line".
    ignore_frac : float
        ALL events are always drawn. When >0 and sort='pre_event', the central
        `ignore_frac` of events (the uncorrelated middle that split_updown drops)
        is shaded blue, so you can see exactly which events the up/down split
        discards. Matches split_updown's median-split geometry.
    """
    lag_s = taus * frame_duration
    S = snippets
    key_sorted = None
    if sort == "pre_event":
        pre = (lag_s < 0) & (lag_s >= -pre_window_s)
        if not pre.any():
            pre = lag_s < 0
        key = S[:, pre].mean(axis=1)
        order = np.argsort(key)[::-1]                 # up (high key) left, down right
        S = S[order]
        key_sorted = key[order]
    elif sort in (None, "incidence"):
        pass
    else:
        raise ValueError("sort must be 'incidence' or 'pre_event'")
    n_events = S.shape[0]

    if ax is None:
        fig, ax = plt.subplots(figsize=(25, 5))
    vmax = np.max(np.abs(S)) if S.size else 1.0
    im = ax.imshow(
        S.T, aspect="auto", cmap="gray", vmin=-vmax, vmax=vmax, origin="lower",
        extent=[0, n_events, lag_s[0], lag_s[-1]],
    )
    # Mark the split at its VALUE location (not the 50% centre) + the ignored
    # window around it. up|down are the flanks; the band sits where key crosses V.
    if sort == "pre_event" and key_sorted is not None:
        v = _split_point(key_sorted, split_at)
        cross = int(np.sum(key_sorted > v))           # position where key crosses V
        ign = np.zeros(n_events, dtype=bool)
        if ignore_frac > 0:
            pos = np.argsort(np.abs(key_sorted - v))[:int(round(ignore_frac * n_events))]
            ign[pos] = True
            ax.axvspan(pos.min(), pos.max() + 1, color="tab:blue", alpha=0.28,
                       label=f"ignored {ignore_frac:.0%} around split")
        up_kept = int(np.sum((key_sorted > v) & ~ign))
        down_kept = int(np.sum((key_sorted < v) & ~ign))
        ax.axvline(cross, color="tab:orange", lw=1.4, ls="--",
                   label=f"split @ key={v:.2f}  (up={up_kept}, down={down_kept})")
    ax.axhline(0.0, color="tab:red", lw=0.7, ls="--")
    xlab = {
        "pre_event": "event  (<- 'up'/ON-driven   |   'down'/OFF-driven ->)",
        "incidence": "event (chronological / by incidence)",
    }.get(sort, "event")
    ax.set_xlabel(xlab)
    ax.set_ylabel("STRF lag (s)  [negative = past]")
    ax.set_title(f"ROI {roi}  |  {n_events} events  |  sign={sign}  |  sort={sort}")
    if sort == "pre_event":
        ax.legend(loc="upper right", fontsize=8, framealpha=0.7)
    plt.colorbar(im, ax=ax, label="stimulus contrast (value - 0.5)")
    return ax


def _split_point(key, split_at):
    """The key VALUE at which up/down are divided.

    'median'   -> 50th percentile (equal event counts; the old behaviour).
    'midrange' -> halfway of the key's value range (min+max)/2 -- the 'halfway
                  point' between most-OFF and most-ON; ~0 for balanced contrast.
    'zero'     -> 0 (the true ON/OFF contrast boundary).
    'mean'     -> mean key.  numeric -> that value.
    """
    if split_at in ("median", None):
        return float(np.median(key))
    if split_at == "midrange":
        return 0.5 * (float(np.min(key)) + float(np.max(key)))
    if split_at == "zero":
        return 0.0
    if split_at == "mean":
        return float(np.mean(key))
    return float(split_at)


def split_updown(snippets, taus, frame_duration, pre_window_s=0.3, ignore_frac=0.0,
                 split_at="median"):
    """Split events into down/up by the pre-event contrast, at a chosen VALUE.

    key = mean contrast in [-pre_window_s, 0) per event. Split at `split_at`
    (see _split_point); events whose key falls in the `ignore_frac` fraction
    CLOSEST to the split value are dropped (the ambiguous middle). The rest:
      down_idx : key < split value  (OFF-driven / 'down')
      up_idx   : key > split value  (ON-driven  / 'up')

    Unlike the old median split, a value split (e.g. 'midrange'/'zero') yields
    ASYMMETRIC group sizes reflecting the cell's polarity bias -- e.g. an
    OFF-driven cell has many 'down' and few 'up' events.
    Returns (down_idx, up_idx, key).
    """
    lag_s = taus * frame_duration
    pre = (lag_s < 0) & (lag_s >= -pre_window_s)
    if not pre.any():
        pre = lag_s < 0
    key = snippets[:, pre].mean(axis=1)
    v = _split_point(key, split_at)
    n = len(key)
    ignored = np.zeros(n, dtype=bool)
    n_ignore = int(round(ignore_frac * n))
    if n_ignore > 0:
        ignored[np.argsort(np.abs(key - v))[:n_ignore]] = True
    down_idx = np.flatnonzero((key < v) & ~ignored)
    up_idx = np.flatnonzero((key > v) & ~ignored)
    return down_idx, up_idx, key


def plot_updown_means(
    snippets, taus, frame_duration, roi, sta_at_peak=None,
    pre_window_s=0.3, ignore_frac=0.5, split_at="median", ax=None,
):
    """Group-mean snippet for up vs down events (pre-event split).

    down/OFF -> black, up/ON -> lightgrey, optional linear STRF@peak -> red.
    The two curves differ in the pre-event window BY CONSTRUCTION (that's the
    sort key); watch for differences AWAY from it -- that's the nonlinear signal.
    """
    lag_s = taus * frame_duration
    down_idx, up_idx, _ = split_updown(
        snippets, taus, frame_duration, pre_window_s, ignore_frac, split_at
    )
    down_mean = snippets[down_idx].mean(axis=0)
    up_mean = snippets[up_idx].mean(axis=0)

    if ax is None:
        fig, ax = plt.subplots(figsize=(6.5, 3.5))
    ax.set_facecolor("0.5")  # mid-grey so the lightgrey 'up' curve is visible
    ax.plot(lag_s, down_mean, color="black", lw=1.8, label=f"down/OFF (n={len(down_idx)})")
    ax.plot(lag_s, up_mean, color="lightgrey", lw=1.8, label=f"up/ON (n={len(up_idx)})")
    ax.axhline(0.0, color="0.7", lw=0.5)
    ax.axvline(0.0, color="k", lw=0.7, ls="--")
    ax.axvspan(-pre_window_s, 0.0, color="tab:blue", alpha=0.06)  # sort window
    ax.set_xlabel("lag (s)")
    ax.set_ylabel("mean contrast")
    ax.set_title(f"ROI {roi}: up/down split means  (ignore {ignore_frac:.0%} middle)")
    if sta_at_peak is not None:
        ax2 = ax.twinx()
        ax2.plot(lag_s, sta_at_peak, color="tab:red", alpha=0.6, lw=1.2)
        ax2.set_ylabel("STRF amp", color="tab:red")
    ax.legend(loc="upper left", fontsize=8)
    return ax


def _crop_box(px, py, crop, n_x, n_y):
    """Clipped [x0,x1,y0,y1] box of half-width `crop` around (px,py). None -> full."""
    if crop is None:
        return 0, n_x, 0, n_y
    return (max(0, px - crop), min(n_x, px + crop + 1),
            max(0, py - crop), min(n_y, py + crop + 1))


def extract_snippets_spatial(
    noise_array, frame_to_pattern, event_frames,
    n_f_filter_past, n_f_filter, center=None, crop: int | None = 6, baseline=0.5, max_gib=6,
):
    """Per-event SPATIOTEMPORAL stimulus history cropped around the RF.

    The spatial generalisation of extract_snippets: instead of collapsing the
    peak-pixel neighbourhood to one timecourse, keep the whole cropped patch.

    Returns
    -------
    snips : (n_events, n_lags, nx_c, ny_c) float32   contrast (value - baseline)
        Spatial axes are (x, y) to match obj.strfs[roi] = (time, x, y).
    taus : (n_lags,) int
    box : (x0, x1, y0, y1) crop bounds in strf/noise pixel coords.

    center=(px,py) strf pixel (i_x,i_y); crop=half-width (box=2*crop+1); crop=None
    -> full frame. Memory-guarded (raises above max_gib) since n_events can be huge
    at low thresholds -- raise the event threshold or shrink crop if it trips.
    """
    n_y, n_x = noise_array.shape[0], noise_array.shape[1]   # noise[iy, ix, pat]
    px, py = (n_x // 2, n_y // 2) if center is None else center
    x0, x1, y0, y1 = _crop_box(px, py, crop, n_x, n_y)
    nx_c, ny_c = x1 - x0, y1 - y0

    taus = np.arange(n_f_filter) + (1 - n_f_filter_past)
    n_rel = len(frame_to_pattern)
    n_ev = len(event_frames)

    nbytes = n_ev * n_f_filter * nx_c * ny_c * 4
    if nbytes > max_gib * 2**30:
        raise MemoryError(
            f"spatial snippets would be {nbytes / 2**30:.1f} GiB "
            f"({n_ev} events x {n_f_filter} lags x {nx_c}x{ny_c}); "
            "raise the event threshold or reduce crop."
        )

    # crop patch across all patterns, reordered noise[iy,ix,pat] -> (x, y, pat).
    # COLOUR-SEAM: noise_array gains a colour axis; select per colour_lookup here.
    crop_vals = np.transpose(
        noise_array[y0:y1, x0:x1, :].astype(np.float32), (1, 0, 2)
    )  # (nx_c, ny_c, n_pat)

    snips = np.full((n_ev, n_f_filter, nx_c, ny_c), baseline, dtype=np.float32)
    for e, f in enumerate(event_frames):
        frames = f + taus
        vi = np.flatnonzero((frames >= 0) & (frames < n_rel))
        pats = frame_to_pattern[frames[vi]]
        has = pats >= 0
        snips[e, vi[has]] = np.transpose(crop_vals[:, :, pats[has]], (2, 0, 1))
    snips -= baseline
    return snips, taus, (x0, x1, y0, y1)


def updown_spatial(
    snips_spatial, snippets_1d, taus, frame_duration,
    pre_window_s=0.3, ignore_frac=0.0, split_at="median",
):
    """Up/down group-mean spatiotemporal kernels.

    Split is defined on the 1-D peak-pixel `snippets_1d` (the anchor), then
    applied to the spatial snippets. Returns (down_k, up_k, down_idx, up_idx),
    each kernel (n_lags, nx_c, ny_c).
    """
    down_idx, up_idx, _ = split_updown(
        snippets_1d, taus, frame_duration, pre_window_s, ignore_frac, split_at
    )
    return (snips_spatial[down_idx].mean(0), snips_spatial[up_idx].mean(0),
            down_idx, up_idx)


def plot_spatial_kernels(
    down_k, up_k, taus, frame_duration, roi, strf_crop=None, n_show=7,
):
    """Filmstrip of the up/down spatiotemporal kernels + their difference.

    Rows: [linear STRF crop (if given)], down/OFF, up/ON, (up-down) split axis.
    Columns: lags spread around the kernel's peak-energy lag.

    NOTE: up and down are equal-size halves, so (up+down)/2 IS the event STA --
    the sum carries no information beyond ordinary reverse correlation. The
    DIFFERENCE (up-down) is the axis along which the split separates events (the
    informative object). It is still a linear conditional quantity; proving any
    of it is *non-linear* requires a linear-model null (not yet built).
    """
    lag_s = taus * frame_duration
    energy = np.abs(down_k).mean((1, 2)) + np.abs(up_k).mean((1, 2))
    pk = int(energy.argmax())
    idxs = np.unique(np.clip(
        np.linspace(pk - 3, pk + 3, n_show).round().astype(int), 0, len(taus) - 1
    ))

    diff = up_k - down_k       # the split axis (what varies); sum would be 2*STA
    rows = []
    if strf_crop is not None:
        rows.append(("linear STRF", np.asarray(strf_crop), "gray"))
    rows += [("down/OFF", down_k, "gray"), ("up/ON", up_k, "gray"),
             ("up-down\n(split axis)", diff, "RdBu_r")]

    vmax = float(np.max(np.abs(np.stack([down_k, up_k])))) or 1.0
    dmax = float(np.max(np.abs(diff))) or vmax

    fig, axs = plt.subplots(
        len(rows), len(idxs), figsize=(1.7 * len(idxs), 2.0 * len(rows)),
        squeeze=False,
    )
    for ri, (name, K, cmap) in enumerate(rows):
        if name == "linear STRF":
            vm = float(np.max(np.abs(K))) or 1.0
        elif "split axis" in name:
            vm = dmax
        else:
            vm = vmax
        for ci, ti in enumerate(idxs):
            ax = axs[ri][ci]
            ax.imshow(K[ti].T, origin="lower", cmap=cmap, vmin=-vm, vmax=vm)
            ax.set_xticks([]); ax.set_yticks([])
            if ri == 0:
                ax.set_title(f"{lag_s[ti] * 1000:.0f} ms", fontsize=8)
            if ci == 0:
                ax.set_ylabel(name, fontsize=9)
    fig.suptitle(
        f"ROI {roi}: up/down spatiotemporal kernels  "
        f"(sum = STA; difference = split axis)", y=1.0,
    )
    fig.tight_layout()
    return fig


def _peak_energy_lag(K):
    """Lag index where the kernel has most spatial energy."""
    return int(np.abs(K).mean((1, 2)).argmax())


def _peak_signed_lag(K, sign):
    """Lag index where the kernel's POSITIVE (sign>0) or NEGATIVE (sign<0) energy peaks."""
    signed = np.clip(K, 0, None) if sign > 0 else np.clip(-K, 0, None)
    return int(signed.mean((1, 2)).argmax())


def _centroid(frame2d, thresh_frac=0.5):
    """Amplitude-weighted centroid of a non-negative map, over pixels >= thresh_frac*max."""
    m = np.asarray(frame2d, float)
    if m.max() <= 0:
        return np.nan, np.nan
    m = np.where(m >= thresh_frac * m.max(), m, 0.0)
    ii, jj = np.indices(m.shape)
    tot = m.sum()
    return (ii * m).sum() / tot, (jj * m).sum() / tot


def plot_onoff_overlay(
    down_k, up_k, taus, frame_duration, roi=None,
    on_lag=None, off_lag=None, edge_crop=0, ax=None,
):
    """Overlay the ON (up) and OFF (down) subfields on ONE spatial axis.

    THE discriminator: are the ON-preceded and OFF-preceded portions CO-LOCATED
    (temporal opponency at one spot -> likely a single biphasic *linear* filter)
    or SPATIALLY OFFSET (distinct subfields -> candidate two-feature coincidence)?

    ON = positive part of the up kernel at its peak lag (reds); OFF = negative
    part of the down kernel at its peak lag (blues); alpha ~ |amplitude|.
    Centroids marked, ON->OFF displacement drawn + reported in pixels. Each
    subfield shown at its own peak-energy lag unless on_lag/off_lag (s) given.
    """
    D = np.array(down_k, float)
    U = np.array(up_k, float)
    if edge_crop > 0:
        for A in (D, U):
            A[:, :edge_crop, :] = 0.0; A[:, -edge_crop:, :] = 0.0
            A[:, :, :edge_crop] = 0.0; A[:, :, -edge_crop:] = 0.0

    lag_s = taus * frame_duration
    # ON = where up's POSITIVE energy peaks; OFF = where down's NEGATIVE energy
    # peaks (not the |energy| lag, which for up is its shared OFF blob).
    li_off = _peak_signed_lag(D, -1) if off_lag is None else int(np.argmin(np.abs(lag_s - off_lag)))
    li_on = _peak_signed_lag(U, +1) if on_lag is None else int(np.argmin(np.abs(lag_s - on_lag)))

    on_map = np.clip(U[li_on], 0, None).T     # ON portion (transpose -> display x horiz)
    off_map = np.clip(-D[li_off], 0, None).T  # OFF portion (magnitude)

    cy_on, cx_on = _centroid(on_map)
    cy_off, cx_off = _centroid(off_map)
    disp = float(np.hypot(cx_on - cx_off, cy_on - cy_off))

    if ax is None:
        fig, ax = plt.subplots(figsize=(5, 4))
    ax.imshow(off_map, origin="lower", cmap="Blues",
              alpha=(off_map / (off_map.max() or 1)), interpolation="none")
    ax.imshow(on_map, origin="lower", cmap="Reds",
              alpha=(on_map / (on_map.max() or 1)), interpolation="none")
    ax.plot(cx_off, cy_off, "o", mfc="none", mec="tab:blue", ms=12, mew=2,
            label=f"OFF @ {lag_s[li_off]*1000:.0f} ms")
    ax.plot(cx_on, cy_on, "o", mfc="none", mec="tab:red", ms=12, mew=2,
            label=f"ON @ {lag_s[li_on]*1000:.0f} ms")
    ax.annotate("", xy=(cx_off, cy_off), xytext=(cx_on, cy_on),
                arrowprops=dict(arrowstyle="->", color="k", lw=1.5))
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title(f"ROI {roi}: ON/OFF spatial overlay  |  centroid offset = {disp:.2f} px")
    ax.legend(loc="upper right", fontsize=8, framealpha=0.6)
    return ax, disp


def play_components(
    down_k, up_k, third_k, taus, frame_duration, roi=None,
    dur_s=None, edge_crop=0, aspect="equal",
    third_label="up - down (split axis)", third_cmap="RdBu_r",
    sta_k=None, sta_label="STA",
):
    """Synchronised side-by-side movie of down / up / [STA] / (third) kernels.

    Unified time axis a la pygor.plotting.play_movie_4d: ONE FuncAnimation drives
    every panel off the same frame index, so you see the same lag in each
    simultaneously, with a live lag-in-ms readout. Grey-scale panels (down/OFF,
    up/ON, STA) share one scale; the third panel uses its own scale + cmap.
    Orientation matches obj.play_strf (no transpose of (x, y)).

    third_k : the third array to show. Pass the DIFFERENCE (up-down = the split
        axis) -- NOT the sum, since (up+down)/2 is just the event STA.
    sta_k : optional STA panel (typically (up+down)/2). Shown grey, for reference
        that the "common" component is just ordinary reverse correlation.
    edge_crop : zero a border on all panels before display -- the noise stimulus
        does not tile the FOV edge, so border pixels are degenerate and would
        otherwise dominate a panel's colour scale.
    """
    import matplotlib.animation

    lag_ms = taus * frame_duration * 1000
    D, U, R = (np.array(a, dtype=float) for a in (down_k, up_k, third_k))
    S = None if sta_k is None else np.array(sta_k, dtype=float)
    to_mask = [D, U, R] + ([] if S is None else [S])
    if edge_crop > 0:
        for A in to_mask:
            A[:, :edge_crop, :] = 0.0
            A[:, -edge_crop:, :] = 0.0
            A[:, :, :edge_crop] = 0.0
            A[:, :, -edge_crop:] = 0.0
    n_frames = D.shape[0]
    if dur_s is None:
        dur_s = n_frames * frame_duration  # ~real time

    # Force-scale S to the same range as D/U, so the grey panel is visually comparable.
    if S is not None:
        S *= np.max(np.abs(D)) / (np.max(np.abs(S)) or 1.0)
    
    grey_clim = float(np.max(np.abs([D, U]))) or 1.0
    r_clim = float(np.max(np.abs(R))) or grey_clim
    # STA panel gets its OWN scale: it may be the pygor STRF array (units ~100x
    # the contrast of up/down), so it can't share the grey clim.
    s_clim = (float(np.max(np.abs(S))) or grey_clim) if S is not None else None

    # Panel order: down, up, [STA], third (split axis).
    panels = [
        ("down / OFF", D, "Greys_r", grey_clim),
        ("up / ON", U, "Greys_r", grey_clim),
    ]
    if S is not None:
        panels.append((sta_label, S, "Greys_r", s_clim))
    panels.append((third_label, R, third_cmap, r_clim))

    plt.rcParams["animation.html"] = "jshtml"
    fig, axs = plt.subplots(1, len(panels), figsize=(3.0 * len(panels), 3.4))
    ims = []
    for ax, (name, A, cmap, vm) in zip(axs, panels):
        im = ax.imshow(A[0], origin="lower", cmap=cmap, vmin=-vm, vmax=vm,
                       interpolation="none", aspect=aspect)
        ax.set_title(name, fontsize=10)
        ax.set_xticks([]); ax.set_yticks([])
        ims.append(im)
    sup = fig.suptitle("", fontsize=11)
    fig.tight_layout()

    def video(frame):
        for im, (_, A, _, _) in zip(ims, panels):
            im.set_array(A[frame])
        tag = f"ROI {roi}  |  " if roi is not None else ""
        sup.set_text(f"{tag}lag = {lag_ms[frame]:+.0f} ms   (0 = event)")
        return ims

    anim = matplotlib.animation.FuncAnimation(
        fig, video, frames=n_frames, interval=dur_s / n_frames * 1000,
        repeat_delay=500,
    )
    # Close the figure so the inline backend doesn't ALSO render a static
    # duplicate below the jshtml player; anim keeps the fig ref, so it still plays.
    plt.close(fig)
    return anim


def calcium_kernel(frame_duration, tau_decay_s=0.150, tau_rise_s=0.01):
    """Causal GCaMP-like impulse response (double-exponential), unit sum.

    tau_rise_s -> 0 gives a pure decay exponential. Length ~5*tau_decay.
    """
    dt = frame_duration
    K = max(1, int(np.ceil(5 * tau_decay_s / dt)))
    k = np.arange(K + 1)
    h = np.exp(-k * dt / tau_decay_s)
    if tau_rise_s and tau_rise_s > 0:
        h = h - np.exp(-k * dt / tau_rise_s)
    h = np.clip(h, 0.0, None)
    s = h.sum()
    return h / s if s > 0 else h


def simulate_linear_trace(
    filter_txy, noise_array, frame_to_pattern, taus, frame_duration=None,
    baseline=0.5, noise_sigma=1.0, calcium_tau_decay_s=0.150,
    calcium_tau_rise_s=0.01, seed=0,
):
    """Synthetic CALCIUM trace of a purely LINEAR neuron driven by the same stimulus.

    Linear drive  r(t) = sum_{tau,px} filter(tau,px) * stim(t+tau, px), then
    convolved with a GCaMP-like calcium kernel (so the null has the SAME
    low-pass + delay as real GCaMP8m -- without this the null's event timing is
    wrong), z-scored, + Gaussian measurement noise (std = noise_sigma).

    filter_txy is (n_lags, nx, ny) in the (time, x, y) kernel layout -- pass the
    real event STA so the null cell has the same linear RF. Set
    calcium_tau_decay_s=0 to get the raw (instantaneous) linear drive.
    """
    n_rel = len(frame_to_pattern)
    _, nx, ny = filter_txy.shape
    stim = np.zeros((n_rel, nx, ny), dtype=np.float32)
    valid = frame_to_pattern >= 0
    noise_xy = np.transpose(noise_array.astype(np.float32), (1, 0, 2))  # (nx, ny, npat)
    stim[valid] = np.transpose(noise_xy[:, :, frame_to_pattern[valid]], (2, 0, 1)) - baseline

    K = filter_txy.astype(np.float32)
    r = np.zeros(n_rel, dtype=np.float64)
    for j, tau in enumerate(taus):
        proj = np.tensordot(stim, K[j], axes=([1, 2], [0, 1]))  # (n_rel,)
        if tau > 0:
            r[: n_rel - tau] += proj[tau:]
        elif tau < 0:
            r[-tau:] += proj[: n_rel + tau]
        else:
            r += proj

    # Calcium low-pass + delay (matches the real GCaMP trace the events come from).
    if calcium_tau_decay_s and calcium_tau_decay_s > 0 and frame_duration:
        h = calcium_kernel(frame_duration, calcium_tau_decay_s, calcium_tau_rise_s)
        r = np.convolve(r, h)[:n_rel]

    r = (r - r.mean()) / (r.std() or 1.0)
    rng = np.random.default_rng(seed)
    return r + noise_sigma * rng.standard_normal(n_rel)


def linear_null_kernels(
    filter_txy, noise_array, frame_to_pattern, taus,
    n_f_filter_past, n_f_filter, peak_pixels, frame_duration,
    threshold=2.0, pre_window_s=0.3, ignore_frac=0.1, noise_sigma=1.0,
    calcium_tau_decay_s=0.150, calcium_tau_rise_s=0.01, seed=0,
):
    """Full detect -> split -> spatial-kernel pipeline on a LINEAR-null trace.

    Returns dict with down, up, diff (=up-down), sta (=(up+down)/2), n_events.
    Compare its 'diff' to the real split axis: if the cell is linear they match.
    """
    trace = simulate_linear_trace(
        filter_txy, noise_array, frame_to_pattern, taus, frame_duration=frame_duration,
        noise_sigma=noise_sigma, calcium_tau_decay_s=calcium_tau_decay_s,
        calcium_tau_rise_s=calcium_tau_rise_s, seed=seed,
    )
    dif = np.diff(trace, prepend=trace[0])
    base = dif[: min(100, len(dif))]
    dif = (dif - base.mean()) / (base.std() or 1.0)
    ev = np.flatnonzero(dif > threshold)  # sign='pos', matches detect_events

    snips1d, _ = extract_snippets(
        noise_array, frame_to_pattern, ev, peak_pixels, n_f_filter_past, n_f_filter
    )
    full, _, _ = extract_snippets_spatial(
        noise_array, frame_to_pattern, ev, n_f_filter_past, n_f_filter, crop=None
    )
    down, up, _, _ = updown_spatial(
        full, snips1d, taus, frame_duration, pre_window_s=pre_window_s, ignore_frac=ignore_frac
    )
    return {"down": down, "up": up, "diff": up - down, "sta": (up + down) / 2,
            "n_events": len(ev)}


def plot_diff_compare(
    diff_real, diff_null, taus, frame_duration, roi=None, edge_crop=0, n_show=7,
):
    """Compare the real split-axis (up-down) to the LINEAR-null split-axis.

    Rows: real, linear-null, (real - null). A linear cell -> real ~ null -> the
    bottom row is ~0. Structure surviving in (real - null) is the non-linear
    signal. Title reports Frobenius-norm power and the excess fraction
    ||real - null|| / ||real||.
    """
    R = np.array(diff_real, float)
    N = np.array(diff_null, float)
    if edge_crop > 0:
        for A in (R, N):
            A[:, :edge_crop, :] = 0.0; A[:, -edge_crop:, :] = 0.0
            A[:, :, :edge_crop] = 0.0; A[:, :, -edge_crop:] = 0.0
    resid = R - N
    lag_s = taus * frame_duration
    en = np.abs(R).mean((1, 2)) + np.abs(N).mean((1, 2))
    pk = int(en.argmax())
    idxs = np.unique(np.clip(
        np.linspace(pk - 3, pk + 3, n_show).round().astype(int), 0, len(taus) - 1))

    rows = [("real (up-down)", R), ("linear null", N), ("real - null", resid)]
    vm = float(np.max(np.abs([R, N]))) or 1.0
    p_real = float(np.linalg.norm(R)); p_null = float(np.linalg.norm(N))
    p_exc = float(np.linalg.norm(resid))

    fig, axs = plt.subplots(3, len(idxs), figsize=(1.7 * len(idxs), 6), squeeze=False)
    for ri, (name, K) in enumerate(rows):
        for ci, ti in enumerate(idxs):
            ax = axs[ri][ci]
            ax.imshow(K[ti].T, origin="lower", cmap="RdBu_r", vmin=-vm, vmax=vm)
            ax.set_xticks([]); ax.set_yticks([])
            if ri == 0:
                ax.set_title(f"{lag_s[ti] * 1000:.0f} ms", fontsize=8)
            if ci == 0:
                ax.set_ylabel(name, fontsize=9)
    fig.suptitle(
        f"ROI {roi}: split axis vs linear null  |  ||real||={p_real:.2f}  "
        f"||null||={p_null:.2f}  excess={p_exc / (p_real or 1):.2f}", y=1.0)
    fig.tight_layout()
    return fig, {"power_real": p_real, "power_null": p_null,
                 "excess_frac": p_exc / (p_real or 1)}


# ============================================================================
# Spike-triggered covariance (STC) -- sort-free 2nd-order analysis.
# The event/median split is a 1-D shadow of this; STC eigenvectors give the
# extra filters a non-linear cell is sensitive to, with no tautological sort.
# ============================================================================


def time_crop_lags(taus, frame_duration, lo_s=-1.0, hi_s=0.2):
    """Boolean lag mask restricting to the causal RF window [lo_s, hi_s] (s).

    Keeps the STC/clustering covariance dimensionality tractable.
    """
    lag_s = taus * frame_duration
    return (lag_s >= lo_s) & (lag_s <= hi_s)


def _flatten_snips(spatial_snips, lag_mask):
    """(n, n_lags, nx, ny) -> (n, n_lags_c*nx*ny) over kept lags; returns (P, shape)."""
    sub = spatial_snips[:, lag_mask]
    n, nl, nx, ny = sub.shape
    return sub.reshape(n, nl * nx * ny).astype(np.float64), (nl, nx, ny)


def stimulus_prior_basis(
    noise_array, frame_to_pattern, center, crop, taus, lag_mask,
    n_f_filter_past, n_f_filter, n_pca=120, n_sample=10000, seed=0,
):
    """Truncated-PCA whitening basis of the stimulus prior (cropped+time-cropped).

    Sampled over random relevant frames (NOT events). STC runs in this n_pca-dim
    prior-whitened subspace, so the eigendecomposition drops from D^3 to n_pca^3
    (and the truncation regularizes). The 'shifting' noise is not white in the
    snippet basis, so this empirical prior is required.
    Returns dict {mean (D,), components (M, D), variance (M,), shape}.
    """
    from sklearn.decomposition import PCA
    rng = np.random.default_rng(seed)
    n_rel = len(frame_to_pattern)
    lo = max(1, int(-taus.min()) + 1)
    hi = n_rel - int(taus.max()) - 1
    frames = rng.integers(lo, hi, size=min(n_sample, max(1, hi - lo)))
    snips, _, _ = extract_snippets_spatial(
        noise_array, frame_to_pattern, frames, n_f_filter_past, n_f_filter,
        center=center, crop=crop,
    )
    P, shape = _flatten_snips(snips, lag_mask)
    m = min(n_pca, P.shape[1], P.shape[0] - 1)
    pca = PCA(n_components=m, svd_solver="randomized", random_state=seed).fit(P)
    return {"mean": pca.mean_, "components": pca.components_,
            "variance": pca.explained_variance_, "shape": shape}


def compute_stc(snips_flat, prior_basis, weights=None, n_keep=6, eps=1e-6):
    """STC eigen-decomposition in the prior-whitened PCA subspace.

    Project snippets onto the prior PCA basis, whiten by prior variances (prior
    becomes the identity there), then eigen-decompose (event covariance - I) --
    an n_pca x n_pca problem, not D x D. Eigen-filters are un-whitened back to
    stimulus space; the STA is the exact full-D weighted mean.

    weights : None -> event-triggered; array -> continuous rectified-trace weighted.
    Returns dict: eigvals (M, desc), keep_idx, keep_eigvals,
    filters (n_keep, *shape), sta (*shape).
    """
    U = np.asarray(prior_basis["components"], float)      # (M, D)
    var = np.asarray(prior_basis["variance"], float)      # (M,)
    mean = np.asarray(prior_basis["mean"], float)         # (D,)
    shape = prior_basis["shape"]
    X = np.asarray(snips_flat, float)
    n = len(X)
    w = np.ones(n) if weights is None else np.clip(np.asarray(weights, float), 0, None)
    wsum = w.sum() or 1.0
    vs = np.sqrt(var + eps * var.max())

    sta = (w[:, None] * X).sum(0) / wsum                  # exact STA (full D)
    Z = ((X - mean) @ U.T) / vs                           # (n, M) prior-whitened coords
    zbar = (w[:, None] * Z).sum(0) / wsum
    Zc = Z - zbar
    C = (Zc.T * w) @ Zc / wsum
    dC = 0.5 * (C + C.T) - np.eye(len(var))               # prior = I in this space
    lam, Vz = np.linalg.eigh(dC)                          # (M,), (M, M) -- cheap
    order = np.argsort(lam)[::-1]
    lam, Vz = lam[order], Vz[:, order]
    keep = np.argsort(np.abs(lam))[::-1][:n_keep]
    keep = keep[np.argsort(lam[keep])[::-1]]              # exc first, supp last
    filters = np.stack([                                  # un-whiten to stimulus space
        (U.T @ (vs * Vz[:, i])).reshape(shape) for i in keep
    ])
    return {"eigvals": lam, "keep_idx": keep, "keep_eigvals": lam[keep],
            "filters": filters, "sta": sta.reshape(shape)}


def stc_event(spatial_snips, lag_mask, prior_basis, n_keep=6):
    """Event-triggered STC: covariance of the detected-event snippet ensemble."""
    P, _ = _flatten_snips(spatial_snips, lag_mask)
    return compute_stc(P, prior_basis, weights=None, n_keep=n_keep)


def stc_continuous(
    obj, roi, noise_array, frame_to_pattern, trigger_start, taus, lag_mask,
    center, crop, n_f_filter_past, n_f_filter, prior_basis, n_keep=6,
    use_znorm=True, stride=2, seed=0,
):
    """Continuous response-weighted STC (2nd-order Wiener): all frames weighted
    by the (rectified) calcium trace, instead of thresholded events."""
    n_rel = len(frame_to_pattern)
    lo = max(1, int(-taus.min()) + 1)
    hi = n_rel - int(taus.max()) - 1
    frames = np.arange(lo, hi, stride)
    trace = (obj.traces_znorm if use_znorm else obj.traces_raw)[roi]
    weights = np.asarray(trace[trigger_start + frames], float)
    snips, _, _ = extract_snippets_spatial(
        noise_array, frame_to_pattern, frames, n_f_filter_past, n_f_filter,
        center=center, crop=crop,
    )
    P, _ = _flatten_snips(snips, lag_mask)
    return compute_stc(P, prior_basis, weights=weights, n_keep=n_keep)


def stc_null_eigenvalues(
    filter_txy, noise_array, frame_to_pattern, taus, lag_mask,
    n_f_filter_past, n_f_filter, center, crop, prior_basis,
    threshold=2.0, noise_sigma=1.5, calcium_tau_decay_s=0.08, n_boot=25,
    n_keep=6, seed=0,
):
    """Null STC eigenvalue band: event-triggered STC on LINEAR-null ensembles.

    A linear cell has NO covariance features beyond the prior, so its eigenvalues
    define the band a real eigenvalue must exceed to count as genuine. The
    deterministic linear drive is computed ONCE (noise_sigma=0) and only the
    measurement noise is re-drawn per bootstrap -- avoids re-projecting the
    stimulus n_boot times. Returns (top_eigvals (n_boot, n_keep), all_eigvals).
    """
    det = simulate_linear_trace(
        filter_txy, noise_array, frame_to_pattern, taus,
        noise_sigma=0.0, calcium_tau_decay_s=calcium_tau_decay_s, seed=0,
    )
    rng = np.random.default_rng(seed)
    tops, alls = [], []
    for _ in range(n_boot):
        trace = det + noise_sigma * rng.standard_normal(len(det))
        dif = np.diff(trace, prepend=trace[0])
        base = dif[: min(100, len(dif))]
        dif = (dif - base.mean()) / (base.std() or 1.0)
        ev = np.flatnonzero(dif > threshold)
        if len(ev) < 20:
            continue
        snips, _, _ = extract_snippets_spatial(
            noise_array, frame_to_pattern, ev, n_f_filter_past, n_f_filter,
            center=center, crop=crop,
        )
        res = stc_event(snips, lag_mask, prior_basis, n_keep=n_keep)
        alls.append(res["eigvals"]); tops.append(res["keep_eigvals"])
    return np.array(tops), np.array(alls)


# ============================================================================
# Clustering-based event split -- data-driven replacement for the 50/50 median
# split. cluster_split: a GMM on the RF-focused split key (pre-event contrast)
# places the boundaries and flags the uncertain middle (posterior-gated), for a
# variable number of groups. cluster_spatial builds per-cluster mean kernels.
# ============================================================================


def cluster_spatial(spatial_snips, labels):
    """Per-cluster mean spatiotemporal kernel. Returns (k, n_lags, nx, ny)."""
    return np.stack([spatial_snips[labels == c].mean(0) for c in np.unique(labels)])


def cluster_split(
    snippets_1d, spatial_snips, taus, frame_duration, pre_window_s=0.3,
    feature="key", k=None, k_range=(1, 4), min_posterior=0.6, seed=0,
):
    """Data-driven up/down split: same RF-focused axis as split_updown, but a GMM
    finds the natural number of groups + the boundaries, instead of a hard cut at
    0 and a fixed ignored fraction.

    This is the user's "above/below 0.5, drop the middle" logic, made data-driven:
    the split feature is the pre-event contrast at the peak pixel (exactly the
    split_updown key), a GMM (n_components by BIC over k_range) determines the
    split points, and the 'uncertain middle' = events whose max GMM posterior is
    below min_posterior (ambiguous between groups) get dropped.

    feature : 'key'     -> scalar pre-event contrast (the split_updown key).
              'peakpix' -> the peak-pixel timecourse over [-pre_window_s, 0) (richer).
    Clusters are relabelled by mean key (0 = most OFF/'down', last = most ON/'up').
    Returns dict: labels, kept (bool mask), k, kernels (k, n_lags, nx, ny) from KEPT
    events, boundaries (k-1 key thresholds), sizes{c}, key, posterior.

    CAVEAT: the binary stimulus makes `key` DISCRETE (mean of ~15 +/-0.5 samples ->
    ~16 levels), so auto-k (BIC) tends to fit the stimulus quantization grid
    (equally-spaced boundaries) rather than biology. Use a FIXED small k (2-3) and
    treat this as a principled soft-boundary / confidence-gated version of the
    up/down split -- NOT as discovery of neural event types (STC + cluster
    stability are the type-discovery tests).
    """
    from sklearn.mixture import GaussianMixture

    lag_s = taus * frame_duration
    pre = (lag_s < 0) & (lag_s >= -pre_window_s)
    if not pre.any():
        pre = lag_s < 0
    key = snippets_1d[:, pre].mean(1)                       # pre-event contrast per event
    F = key[:, None] if feature == "key" else snippets_1d[:, pre]

    if k is None:
        models = {kk: GaussianMixture(kk, random_state=seed).fit(F)
                  for kk in range(k_range[0], k_range[1] + 1)}
        k = min(models, key=lambda kk: models[kk].bic(F))
        gmm = models[k]
    else:
        gmm = GaussianMixture(k, random_state=seed).fit(F)

    post = gmm.predict_proba(F)
    raw = post.argmax(1)
    kept = post.max(1) >= min_posterior
    # relabel clusters by ascending mean key (down -> up)
    order = np.argsort([key[raw == c].mean() if (raw == c).any() else np.inf
                        for c in range(k)])
    remap = {old: new for new, old in enumerate(order)}
    labels = np.array([remap[c] for c in raw])

    means = np.array([key[labels == c].mean() for c in range(k)])
    boundaries = (means[:-1] + means[1:]) / 2 if k > 1 else np.array([])
    kernels = np.stack([
        spatial_snips[(labels == c) & kept].mean(0) if ((labels == c) & kept).any()
        else np.zeros(spatial_snips.shape[1:]) for c in range(k)
    ])
    sizes = {c: int(((labels == c) & kept).sum()) for c in range(k)}
    return {"labels": labels, "kept": kept, "k": k, "kernels": kernels,
            "boundaries": boundaries, "sizes": sizes, "key": key, "posterior": post}


# ============================================================================
# Plots -- clustering and STC equivalents of the up/down snippet plots.
# ============================================================================


def _filmstrip_lags(energy_profile, taus, n_show):
    """Column lags for a filmstrip: n_show lags spread around the energy peak."""
    pk = int(energy_profile.argmax())
    return np.unique(np.clip(
        np.linspace(pk - 3, pk + 3, n_show).round().astype(int), 0, len(taus) - 1))


def _mask_edges(K, edge_crop):
    if edge_crop > 0:
        K[..., :edge_crop, :] = 0.0; K[..., -edge_crop:, :] = 0.0
        K[..., :, :edge_crop] = 0.0; K[..., :, -edge_crop:] = 0.0
    return K


def _peak_var_ij(K, edge_crop=2):
    """Pixel (i,j) of max temporal variance (lags = axis -3), averaged over any
    leading axes, edge-masked. Uses variance not |mean| so a near-constant border
    pixel (large |mean|, ~zero variance) can't win the peak-pixel search."""
    v = _mask_edges(np.asarray(K, float).copy(), edge_crop).var(axis=-3)
    while v.ndim > 2:
        v = v.mean(0)
    return tuple(int(x) for x in np.unravel_index(v.argmax(), v.shape))


def plot_cluster_kernels(kernels, taus, frame_duration, sizes=None, jaccard=None,
                         n_show=7, edge_crop=0):
    """Filmstrip of per-cluster mean spatiotemporal kernels (rows=clusters)."""
    K = _mask_edges(np.array(kernels, float), edge_crop)
    lag_s = taus * frame_duration
    idxs = _filmstrip_lags(np.abs(K).mean((0, 2, 3)), taus, n_show)
    vm = float(np.max(np.abs(K))) or 1.0
    nk = K.shape[0]
    fig, axs = plt.subplots(nk, len(idxs), figsize=(1.7 * len(idxs), 2.0 * nk), squeeze=False)
    for r in range(nk):
        for c, ti in enumerate(idxs):
            ax = axs[r][c]
            ax.imshow(K[r, ti].T, origin="lower", cmap="Greys_r", vmin=-vm, vmax=vm)
            ax.set_xticks([]); ax.set_yticks([])
            if r == 0:
                ax.set_title(f"{lag_s[ti] * 1000:.0f} ms", fontsize=8)
            if c == 0:
                lab = f"clust {r}"
                if sizes is not None:
                    lab += f"\nn={sizes.get(r, '')}"
                if jaccard is not None and r in jaccard:
                    lab += f"\nJ={jaccard[r].mean():.2f}"
                ax.set_ylabel(lab, fontsize=8)
    fig.suptitle("Cluster mean spatiotemporal kernels", y=1.0)
    fig.tight_layout()
    return fig


def plot_cluster_means_1d(kernels, taus, frame_duration, peak_ij=None, sizes=None,
                          edge_crop=2, ax=None):
    """Per-cluster peak-pixel timecourse (cluster analogue of plot_updown_means)."""
    K = np.array(kernels, float)
    lag_s = taus * frame_duration
    if peak_ij is None:
        peak_ij = _peak_var_ij(K, edge_crop)
    i, j = peak_ij
    if ax is None:
        _, ax = plt.subplots(figsize=(6.5, 3.5))
    ax.set_facecolor("0.5")
    cols = plt.cm.viridis(np.linspace(0, 1, K.shape[0]))
    for c in range(K.shape[0]):
        lab = f"clust {c}" + (f" (n={sizes.get(c, '')})" if sizes else "")
        ax.plot(lag_s, K[c][:, i, j], color=cols[c], lw=1.8, label=lab)
    ax.axvline(0, color="k", lw=0.7, ls="--"); ax.axhline(0, color="0.75", lw=0.5)
    ax.set_xlabel("lag (s)"); ax.set_ylabel("contrast @ peak px")
    ax.set_title("Cluster peak-pixel timecourses"); ax.legend(fontsize=8)
    return ax


def plot_snippet_log_clustered(snippets, taus, frame_duration, csplit, roi=None, ax=None):
    """SnippetLog heatmap with cluster_split membership as coloured column bands.

    Events on x sorted by the split key (up/ON left -> down/OFF right), STRF lag
    on y (greyscale contrast). Full-height axvspan bands tint each event column by
    its GMM cluster; the dropped 'uncertain middle' events are grey. Sorted by the
    key, clusters appear as contiguous colour blocks -- handy for eyeballing the
    split as you raise k.
    """
    key = np.asarray(csplit["key"], float)
    labels = np.asarray(csplit["labels"])
    kept = np.asarray(csplit["kept"])
    k = int(csplit["k"])
    order = np.argsort(key)[::-1]                 # high key (up/ON) left -> low (down) right
    S = snippets[order]
    band = np.where(kept[order], labels[order], -1)
    lag_s = taus * frame_duration
    n = len(S)

    if ax is None:
        _, ax = plt.subplots(figsize=(14, 4.5))
    vmax = float(np.max(np.abs(S))) or 0.5
    ax.imshow(S.T, aspect="auto", cmap="gray", vmin=-vmax, vmax=vmax, origin="lower",
              extent=[0, n, lag_s[0], lag_s[-1]])
    ax.axhline(0.0, color="tab:red", lw=0.7, ls="--")

    cmap = plt.get_cmap("tab10")
    seen = set()
    x0 = 0
    for x in range(1, n + 1):
        if x == n or band[x] != band[x0]:
            b = int(band[x0])
            if b < 0:
                lbl = "dropped (middle)" if "d" not in seen else None
                seen.add("d")
                ax.axvspan(x0, x, color="0.4", alpha=0.30, lw=0, label=lbl)
            else:
                lbl = f"cluster {b} (n={int((band == b).sum())})" if b not in seen else None
                seen.add(b)
                ax.axvspan(x0, x, color=cmap(b % 10), alpha=0.16, lw=0, label=lbl)
            x0 = x
    ax.set_xlim(0, n)
    ax.set_xlabel("event  (sorted by split key: up/ON left  ->  down/OFF right)")
    ax.set_ylabel("STRF lag (s)  [negative = past]")
    ax.set_title(f"ROI {roi}  |  {n} events  |  cluster_split k={k}")
    ax.legend(loc="upper right", fontsize=8, ncol=min(k + 1, 4), framealpha=0.7)
    return ax


def plot_stc_filters(stc_res, taus_c, frame_duration, n_show=7, edge_crop=0):
    """Filmstrip of the top STC eigen-filters (rows=eigenvectors, RdBu_r)."""
    F = _mask_edges(np.array(stc_res["filters"], float), edge_crop)
    lam = stc_res["keep_eigvals"]
    lag_s = taus_c * frame_duration
    idxs = _filmstrip_lags(np.abs(F).mean((0, 2, 3)), taus_c, n_show)
    vm = float(np.max(np.abs(F))) or 1.0
    nf = F.shape[0]
    fig, axs = plt.subplots(nf, len(idxs), figsize=(1.7 * len(idxs), 2.0 * nf), squeeze=False)
    for r in range(nf):
        for c, ti in enumerate(idxs):
            ax = axs[r][c]
            ax.imshow(F[r, ti].T, origin="lower", cmap="RdBu_r", vmin=-vm, vmax=vm)
            ax.set_xticks([]); ax.set_yticks([])
            if r == 0:
                ax.set_title(f"{lag_s[ti] * 1000:.0f} ms", fontsize=8)
            if c == 0:
                ax.set_ylabel(f"e{r}\nλ={lam[r]:.2f}", fontsize=8)
    fig.suptitle("STC eigen-filters (top by |eigenvalue|)", y=1.0)
    fig.tight_layout()
    return fig


def plot_stc_timecourses(stc_res, taus_c, frame_duration, peak_ij=None, ax=None):
    """Peak-pixel timecourse of each top STC eigen-filter."""
    F = np.array(stc_res["filters"], float)
    lam = stc_res["keep_eigvals"]
    lag_s = taus_c * frame_duration
    if peak_ij is None:
        peak_ij = _peak_var_ij(F, edge_crop=1)
    i, j = peak_ij
    if ax is None:
        _, ax = plt.subplots(figsize=(6.5, 3.5))
    cols = plt.cm.coolwarm(np.linspace(0, 1, F.shape[0]))
    for r in range(F.shape[0]):
        ax.plot(lag_s, F[r][:, i, j], color=cols[r], lw=1.6, label=f"e{r} λ={lam[r]:.2f}")
    ax.axvline(0, color="k", lw=0.7, ls="--"); ax.axhline(0, color="0.75", lw=0.5)
    ax.set_xlabel("lag (s)"); ax.set_ylabel("filter @ peak px")
    ax.set_title("STC eigen-filter timecourses"); ax.legend(fontsize=8)
    return ax


def plot_stc_spectrum(stc_res, null_tops=None, ax=None):
    """STC eigenvalue spectrum with the linear-null 95% band overlaid.

    Eigenvalues outside the band are candidate genuine (non-linear) features.
    """
    lam = np.sort(stc_res["eigvals"])[::-1]
    if ax is None:
        _, ax = plt.subplots(figsize=(6.5, 3.5))
    ax.plot(np.arange(len(lam)), lam, "o-", ms=2.5, lw=0.8, label="real eigenvalues")
    if null_tops is not None and len(null_tops):
        hi = float(np.percentile(null_tops, 97.5))
        lo = float(np.percentile(null_tops, 2.5))
        ax.axhspan(lo, hi, color="tab:red", alpha=0.15,
                   label=f"linear-null 95% band [{lo:.1f}, {hi:.1f}]")
    ax.axhline(0, color="0.6", lw=0.5)
    ax.set_xlabel("eigenvalue rank"); ax.set_ylabel("whitened variance")
    ax.set_title("STC spectrum vs linear null"); ax.legend(fontsize=8)
    return ax


def _collapse_spatial(kernel_txy, edge_crop=0):
    """[t,x,y] -> [x,y]: signed frame at the peak-energy lag (edge-masked).

    (collapse_3d gave near-uniform maps for these small kernels; the peak-lag
    frame is the cleaner, more interpretable spatial summary here.)
    """
    K = _mask_edges(np.asarray(kernel_txy, float).copy(), edge_crop)
    return K[int(np.abs(K).mean((1, 2)).argmax())]


def _peak_tc(kernel_txy, edge_crop=0):
    """Peak-pixel timecourse of a [t,x,y] kernel; picks the max-temporal-variance
    pixel (edge-masked) so a near-constant border pixel can't win."""
    i, j = _peak_var_ij(kernel_txy, edge_crop)
    return np.asarray(kernel_txy, float)[:, i, j]


def plot_three_way(obj, roi, snippet_kernels, stc_res, taus, frame_duration,
                   taus_c=None, edge_crop=2):
    """Visual 3-way comparison: pygor STRF vs snippet-based vs STC.

    Each recovered filter shown as [collapsed spatial map | peak-pixel timecourse].
    Rows grouped by method. snippet_kernels = list of (label, kernel_txy) using the
    full `taus`; STC filters use the time-cropped `taus_c`.
    """
    if taus_c is None:
        taus_c = taus
    rows = [("pygor", "linear STRF", np.asarray(obj.strfs[roi], float), taus)]
    for lab, K in snippet_kernels:
        rows.append(("snippet", lab, np.asarray(K, float), taus))
    rows.append(("STC", "STA (0th)", stc_res["sta"], taus_c))
    for r in range(stc_res["filters"].shape[0]):
        rows.append(("STC", f"e{r} λ={stc_res['keep_eigvals'][r]:.1f}",
                     stc_res["filters"][r], taus_c))

    n = len(rows)
    fig, axs = plt.subplots(n, 2, figsize=(7, 1.5 * n), squeeze=False,
                            gridspec_kw={"width_ratios": [1, 2]})
    for r, (meth, lab, K, tt) in enumerate(rows):
        ec = edge_crop if K.shape[1] > 2 * edge_crop else 0
        spat = _collapse_spatial(K, edge_crop=ec)
        vm = float(np.max(np.abs(spat))) or 1.0
        axs[r][0].imshow(spat.T, origin="lower", cmap="RdBu_r", vmin=-vm, vmax=vm)
        axs[r][0].set_xticks([]); axs[r][0].set_yticks([])
        axs[r][0].set_ylabel(f"{meth}\n{lab}", fontsize=7)
        axs[r][1].plot(tt * frame_duration, _peak_tc(K, edge_crop=ec), color="k", lw=1.2)
        axs[r][1].axvline(0, color="r", lw=0.5, ls="--")
        axs[r][1].axhline(0, color="0.75", lw=0.5)
        if r < n - 1:
            axs[r][1].set_xticks([])
        else:
            axs[r][1].set_xlabel("lag (s)")
    fig.suptitle(f"ROI {roi}: 3-way comparison (pygor STRF / snippet / STC)", y=1.0)
    fig.tight_layout()
    return fig


# %% ---------------------------------------------------------------------------
# Demo: run end-to-end on the real recording.

noise_array = stimulus_array

# STA window derived from the object (no hardcoded 2s/2s).
frame_duration, n_f_filter, n_f_filter_past = strf_window_from_obj(obj)
print(f"derived: frame_duration={frame_duration:.4f}s  n_f_filter={n_f_filter}  "
      f"n_f_filter_past={n_f_filter_past}")

frame_to_pattern, trigger_start, n_f_relevant = build_frame_to_pattern(obj, noise_array)
print(f"frame_duration={frame_duration:.4f}s  n_f_relevant={n_f_relevant}")
print(f"mapped frames: {(frame_to_pattern >= 0).sum()} / {n_f_relevant}")

# Pick a representative ROI: strongest linear STRF.
strf_strength = np.nanmax(np.abs(obj.strfs), axis=(1, 2, 3))
# demo_roi = int(np.argmax(strf_strength))

# non linear candidates: 8, 20, 26, 28, 38, 61
# idks: 36

demo_roi = 44 # 4, 8 is good
thresh = .5
# Shared split params -> heatmap shading and split-means use the SAME geometry.
pre_window_s = .25
ignore_frac = 0.2
# split_at: where up/down divide. 'midrange' = the VALUE halfway point (min+max)/2
# ~ contrast 0, the true ON/OFF boundary -> ASYMMETRIC groups reflecting the
# cell's polarity bias. 'median' = old 50/50-by-count. Also 'zero'/'mean'/number.
split_at = "midrange"

print(f"demo ROI = {demo_roi}  (|STRF| max = {strf_strength[demo_roi]:.3f})")

(px, py), pixels = peak_pixel_neighbourhood(obj, demo_roi, n_neighbours=4)
print(f"peak pixel (i_x, i_y) = ({px}, {py});  neighbourhood = {pixels}")

event_frames, event_amps = detect_events(
    obj, demo_roi, threshold=thresh, sign="pos",
    trigger_start=trigger_start, n_f_relevant=n_f_relevant,
)
print(f"n_events (pos) = {len(event_frames)}")

snippets, taus = extract_snippets(
    noise_array, frame_to_pattern, event_frames, pixels,
    n_f_filter_past, n_f_filter,
)
print(f"SnippetLog shape = {snippets.shape}")


# Heatmap: all events drawn; split marked at its VALUE location + ignored window.
plot_snippet_log(
    snippets, taus, frame_duration, demo_roi, sign="pos",
    sort="pre_event", pre_window_s=pre_window_s, ignore_frac=ignore_frac,
    split_at=split_at,
)
plt.tight_layout()
plt.show()

# Event-mean snippet vs linear STRF, now SPLIT into up/down partitions:
#   down/OFF -> black, up/ON -> lightgrey, linear STRF@peak -> red.
# The two curves differ inside the shaded pre-event window BY CONSTRUCTION;
# any difference AWAY from it is the (non-linear) signal of interest.
sta_at_peak = obj.strfs[demo_roi][:, px, py]
plot_updown_means(
    snippets, taus, frame_duration, demo_roi, sta_at_peak=sta_at_peak,
    pre_window_s=pre_window_s, ignore_frac=ignore_frac, split_at=split_at,
)
ax = plt.gca()
ax.text(0.02, 0.2, f"threshold = {thresh}", color="blue",
        transform=ax.transAxes, fontsize=12, va="top", ha="left")
plt.tight_layout()
plt.show()

# # %% Spatial kernels ----------------------------------------------------------
# Project the events into full space: crop the stimulus history around the RF
# per event, average within the up/down groups -> two spatiotemporal kernels.
# The residual row (up+down) is where non-linear / co-incident structure shows.
crop = 8  # half-width px -> (2*crop+1) box around the peak pixel
spatial_snips, taus_sp, box = extract_snippets_spatial(
    noise_array, frame_to_pattern, event_frames,
    n_f_filter_past, n_f_filter, center=(px, py), crop=crop,
)
print(f"spatial snippets = {spatial_snips.shape}  box(x0,x1,y0,y1) = {box}")

down_k, up_k, di, ui = updown_spatial(
    spatial_snips, snippets, taus, frame_duration,
    pre_window_s=pre_window_s, ignore_frac=ignore_frac, split_at=split_at,
)
x0, x1, y0, y1 = box
strf_crop = obj.strfs[demo_roi][:, x0:x1, y0:y1]  # (time, x, y), same box
plot_spatial_kernels(down_k, up_k, taus, frame_duration, demo_roi, strf_crop=strf_crop)
plt.show()

# ON/OFF spatial relationship on a unified axis ----------------------------
# THE discriminator: co-located ON/OFF (offset ~0 px) = temporal opponency at
# one spot (likely single biphasic linear filter); spatially offset = distinct
# subfields (candidate two-feature coincidence).
_, onoff_offset = plot_onoff_overlay(
    down_k, up_k, taus, frame_duration, roi=demo_roi, edge_crop=2,
)
print(f"ON/OFF centroid offset = {onoff_offset:.2f} px")
plt.show()

# %% STRF-like component movies -----------------------------------------------
# Full-frame (uncropped) up/down kernels + their difference as STRF-shaped arrays
# (time, x, y) -- same layout as obj.strfs[roi] -- so each plays with pygor's
# movie player exactly like obj.play_strf.
#   down = OFF-preceded events, up = ON-preceded events.
#   sum (up+down)/2 == event STA (redundant, not shown); the DIFFERENCE
#   (up-down) is the split axis = the informative object.
import pygor.plotting

spatial_full, _, _ = extract_snippets_spatial(
    noise_array, frame_to_pattern, event_frames,
    n_f_filter_past, n_f_filter, crop=None,  # whole field, not cropped
)
down_full, up_full, _, _ = updown_spatial(
    spatial_full, snippets, taus, frame_duration,
    pre_window_s=pre_window_s, ignore_frac=ignore_frac, split_at=split_at,
)
diff_full = up_full - down_full          # split axis (the informative object)
event_sta_full = (up_full + down_full) / 2   # real event STA = linear-null's filter
sta_full = obj.strfs[demo_roi]           # == event STA (ordinary reverse corr)
comp_clim = float(np.max(np.abs([down_full, up_full])))  # shared scale for up/down
movie_dur = obj.strf_dur_ms / 1000
print(f"components (time,x,y): down {down_full.shape}  up {up_full.shape}  "
      f"diff {diff_full.shape}  |  comp_clim={comp_clim:.3f}")

# Unified side-by-side movie: down | up | STA | (up-down) on one shared time axis
vid = play_components(
    down_full, up_full, diff_full, taus, frame_duration * 2,
    roi=demo_roi, edge_crop=2,  # mask the degenerate FOV border
    sta_k=sta_full,             # 4th panel: the STA (=(up+down)/2)
    sta_label="STA CCA (scaled)"
)
vid
# vid.save("demo_updown_components.mp4", fps=15/2, dpi=150, bitrate=4000)

# %% Data-driven up/down split: GMM on the RF-focused key --------------------
# Same axis as split_updown (pre-event contrast at peak pixel), but a GMM places
# the boundary (data-driven -- lands ~0, confirming the hard 0.5 cut) and the
# 'uncertain middle' = low-posterior events. Use a FIXED small k: auto-k (BIC)
# fits the binary-stimulus discretization of the key, not biology. Bump k to
# experiment with more clusters.
csplit = cluster_split(
    snippets, spatial_snips, taus, frame_duration,
    pre_window_s=pre_window_s, feature="key", k=4, min_posterior=0.7,
)
print(f"GMM boundary(s)={np.round(csplit['boundaries'], 3)}  sizes={csplit['sizes']}  "
      f"dropped_middle={int((~csplit['kept']).sum())}")
# SnippetLog heatmap, events sorted by key, coloured column bands = clusters:
plot_snippet_log_clustered(snippets, taus, frame_duration, csplit, roi=demo_roi); plt.show()
plot_cluster_kernels(csplit["kernels"], taus, frame_duration, sizes=csplit["sizes"],
                     edge_crop=1); plt.show()
plot_cluster_means_1d(csplit["kernels"], taus, frame_duration, sizes=csplit["sizes"]); plt.show()

# %% STC (spike-triggered covariance) -- both ensembles + null band ----------
# Sort-free 2nd-order analysis. Eigen-filters = the extra features a non-linear
# cell is sensitive to; eigenvalues outside the linear-null band are genuine.
#
# lag_mask MUST stay tight and n_sample MUST stay >> D (=n_lags_kept*nx_c*ny_c):
# PCA fit on a prior sample smaller than D overestimates its own top-variance
# directions (eigenvalue selection bias), so ANY other ensemble projected onto
# them reads out an artificial, near-uniform NEGATIVE shift across most of the
# spectrum -- gibberish eigen-filters with no real structure, not a null result.
# Measured on ROI 44: wide lags (D=17918, n_sample=6000) -> median eigenvalue
# -0.19 (should be ~0); narrow lags (D=5491) + n_sample=20000 -> -0.02.
lag_mask = time_crop_lags(taus, frame_duration, lo_s=-1.0, hi_s=0.2)
taus_c = taus[lag_mask]
prior_basis = stimulus_prior_basis(
    noise_array, frame_to_pattern, (px, py), crop, taus, lag_mask,
    n_f_filter_past, n_f_filter, n_pca=120, n_sample=20000,
)
stc_evt = stc_event(spatial_snips, lag_mask, prior_basis, n_keep=6)           # event-triggered
stc_cont = stc_continuous(                                                    # continuous weighted
    obj, demo_roi, noise_array, frame_to_pattern, trigger_start, taus, lag_mask,
    (px, py), crop, n_f_filter_past, n_f_filter, prior_basis, n_keep=6, stride=2,
)
null_tops, null_all = stc_null_eigenvalues(                                   # linear-null band
    event_sta_full, noise_array, frame_to_pattern, taus, lag_mask,
    n_f_filter_past, n_f_filter, (px, py), crop, prior_basis,
    threshold=thresh,  # MUST match real detect_events threshold or null n_events
    noise_sigma=1.5, n_boot=20,          # diverges wildly (was default 2.0 -> 11x too few)
)
# Two-sided band from null_tops (each boot's own top-6-by-|eig|), NOT null_all
# (every dim pooled): keep_eigvals is an extreme-value/order statistic (max of
# 120), so it must be compared against the null's extreme-value distribution,
# not its bulk/typical-eigenvalue distribution -- the latter is far too narrow
# (no multiple-comparisons correction) and falsely flags ~6/6 on every ROI.
null_band = float(np.percentile(np.abs(null_tops), 97.5))
above = np.abs(stc_evt["keep_eigvals"]) > null_band
print(f"event STC top6 eig: {np.round(stc_evt['keep_eigvals'], 2)}")
print(f"cont  STC top6 eig: {np.round(stc_cont['keep_eigvals'], 2)}")
print(f"null two-sided 97.5% band = +/-{null_band:.2f}  ->  "
      f"{int(above.sum())}/{len(above)} kept eigenvalues exceed it")

plot_stc_spectrum(stc_evt, null_tops=null_tops); plt.show()
plot_stc_filters(stc_evt, taus_c, frame_duration, edge_crop=1); plt.show()
plot_stc_timecourses(stc_evt, taus_c, frame_duration); plt.show()

# %% k-panel synced movies (via pygor.plotting.play_movie_4d) ----------------
pygor.plotting.play_movie_4d(csplit["kernels"])          # per-cluster kernel movie

# %% STC eigen-filter movie
pygor.plotting.play_movie_4d(stc_evt["filters"])

# %% Three-way comparison: pygor STRF | snippet (STA + clusters) | STC -------
# event_sta_full defined earlier (STRF-like component movies cell) -- reused here.
null = linear_null_kernels(
    event_sta_full, noise_array, frame_to_pattern, taus,
    n_f_filter_past, n_f_filter, pixels, frame_duration,
    threshold=thresh, pre_window_s=pre_window_s, ignore_frac=ignore_frac,
    noise_sigma=1.0, seed=0,
)
snippet_kernels = [("event STA", event_sta_full)] + [
    (f"clust{c}", csplit["kernels"][c]) for c in range(csplit["k"])
]
plot_three_way(obj, demo_roi, snippet_kernels, stc_evt, taus, frame_duration, taus_c=taus_c)
plt.show()

# %%
# %% Linear-model null: does the split axis exceed a linear cell? -------------
# Null cell = a purely linear neuron whose RF is the REAL event STA. Feed its
# trace through the identical detect -> split pipeline and compare split axes.
# Linear cell -> real ~ null -> (real - null) ~ 0. Surviving structure = non-linear.
# Tune noise_sigma so null n_events ~ real n_events (same selectivity).

print(f"real n_events={len(event_frames)}   null n_events={null['n_events']}")
_, null_stats = plot_diff_compare(
    diff_full, null["diff"], taus, frame_duration, roi=demo_roi, edge_crop=2,
)
print("null comparison:", {k: round(v, 3) for k, v in null_stats.items()})
plt.show()

# %%
