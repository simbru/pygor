# %%
import pygor
import pygor.load

from dataclasses import dataclass, field

import matplotlib.pyplot as plt
import numpy as np

%load_ext autoreload
%autoreload 2

# %% ---------------------------------------------------------------------------
# v2: four ON/OFF-splitting strategies behind ONE shared result interface, so
# they can be plotted/played/null-tested identically. Builds on
# dev/snippet_analysis.py (v1) -- see that file for the exploratory derivation
# of the anchor-based split, the linear-null test, and STC. v1 is NOT imported
# here (its bottom ~300 lines are top-level executing demo code, not
# import-safe); the specific machinery reused below is copied verbatim instead
# (each function says so).
#
# STRF AXIS CONVENTION (checked empirically, don't trust docstrings blindly):
# CLAUDE.md says STRF arrays are [cell, time, y, x]. v1's own docstring for
# peak_pixel_neighbourhood claims (time, x, y). These disagree. Checked against
# pygor.strf.calculate_strf source directly (not just shapes):
#   strfs_output = np.zeros((n_colours, n_roi, n_f_filter, n_x_noise, n_y_noise))
#   n_x_noise = noise_array.shape[1]; n_y_noise = noise_array.shape[0]
# i.e. obj.strfs[roi] IS (time, x, y) -- v1's docstring was right, CLAUDE.md's
# axis-order note is wrong for the raw array (at least for this dataset/loader
# path). Confirmed on the demo data: noise_array.shape = (40, 24, 30000) ->
# n_y_noise=40, n_x_noise=24, and obj.strfs.shape = (110, 62, 24, 40) matches
# (time=62, x=24, y=40) exactly, not (time, y=24, x=40). Every function below
# treats obj.strfs[roi] and all derived kernels as (n_lags, nx, ny) with
# nx == noise_array.shape[1], ny == noise_array.shape[0] -- same as v1.

# %% Load demo data (same dataset v1's demo cell actually runs against)
stimulus_array = np.load("/home/simen/Noise_npy_arrs/9deg_200_SINGLEcolour_30000x6x10_0.25_1.npy")
example_recording_path = "/mnt/data/Igor analyses/OSDS/251103 OSDS/2_0_SWN_200_White.recording.h5"
obj = pygor.load.STRF.load_object(example_recording_path)
noise_array = stimulus_array
print(f"noise_array.shape = {noise_array.shape}  (n_y, n_x, n_patterns)")
print(f"obj.strfs.shape = {obj.strfs.shape}  (n_roi, n_lags, nx, ny)")


# %% ---------------------------------------------------------------------------
# Machinery copied from v1 (dev/snippet_analysis.py) -- unchanged unless noted.
# ============================================================================


def strf_window_from_obj(obj):
    """Derive the STA lag window from the object. Copied from v1 verbatim."""
    frame_duration = obj.linedur_s * obj.images.shape[1]
    n_f_filter = int(obj.strfs.shape[1])
    total_s = getattr(obj, "strf_dur_ms", n_f_filter * frame_duration * 1000) / 1000.0
    n_f_filter_past = max(1, int(np.floor((total_s / 2.0) / frame_duration)))
    n_f_filter_past = min(n_f_filter_past, n_f_filter - 1)
    return frame_duration, n_f_filter, n_f_filter_past


def build_frame_to_pattern(
    obj, noise_array, skip_first_triggers=0, skip_last_triggers=0, max_frames_per_trigger=100,
):
    """Stimulus->imaging-frame mapping, copied verbatim from v1 (itself copied
    from pygor.strf.calculate_strf lines ~471-513) -- do not reinvent this."""
    ttf = obj.triggertimes_frame.copy()
    nan_mask = np.isnan(ttf)
    n_triggers = int(np.argmax(nan_mask)) if np.any(nan_mask) else len(ttf)

    trigger_start = int(ttf[skip_first_triggers])
    n_f_relevant = int(ttf[n_triggers - skip_last_triggers - 1] - ttf[skip_first_triggers])
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
    obj, roi, threshold=1.75, sign="pos", trigger_start=0, n_f_relevant=None, use_znorm=True,
):
    """Z-scored temporal derivative crossings. Copied verbatim from v1."""
    traces = obj.traces_znorm if use_znorm else obj.traces_raw
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


def pixel_neighbourhood(px, py, n_neighbours=4, edge_crop=2, nx=None, ny=None):
    """Cross-shaped neighbour offsets, edge-clipped. Copied verbatim from v1."""
    offsets = [(-1, 0), (1, 0), (0, -1), (0, 1)][:n_neighbours]
    pixels = [(px, py)]
    for dx, dy in offsets:
        qx, qy = px + dx, py + dy
        if edge_crop <= qx < nx - edge_crop and edge_crop <= qy < ny - edge_crop:
            pixels.append((qx, qy))
    return pixels


def _crop_box(px, py, crop, n_x, n_y):
    """Clipped [x0,x1,y0,y1] box of half-width `crop`. Copied verbatim from v1."""
    if crop is None:
        return 0, n_x, 0, n_y
    return (max(0, px - crop), min(n_x, px + crop + 1),
            max(0, py - crop), min(n_y, py + crop + 1))


def extract_snippets_spatial(
    noise_array, frame_to_pattern, event_frames,
    n_f_filter_past, n_f_filter, center=None, crop=6, baseline=0.5, max_gib=6,
):
    """Per-event spatiotemporal stimulus history, cropped around the RF.
    Copied verbatim from v1 -- see v1 for the full docstring/rationale."""
    n_y, n_x = noise_array.shape[0], noise_array.shape[1]
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


def neighbourhood_timecourse(spatial_snips, pixels):
    """Mean stimulus-contrast timecourse over `pixels`. Copied verbatim from v1."""
    vals = np.stack([spatial_snips[:, :, ix, iy] for ix, iy in pixels], axis=0)
    return vals.mean(axis=0)


def key_from_timecourse(timecourse, taus, frame_duration, pre_window_s=0.3):
    """Pre-event split key: mean contrast in [-pre_window_s, 0). Copied from v1."""
    lag_s = taus * frame_duration
    pre = (lag_s < 0) & (lag_s >= -pre_window_s)
    if not pre.any():
        pre = lag_s < 0
    return timecourse[:, pre].mean(axis=1)


def _split_point(key, split_at):
    """The key VALUE at which up/down are divided. Copied verbatim from v1."""
    if split_at in ("median", None):
        return float(np.median(key))
    if split_at == "midrange":
        return 0.5 * (float(np.min(key)) + float(np.max(key)))
    if split_at == "zero":
        return 0.0
    if split_at == "mean":
        return float(np.mean(key))
    return float(split_at)


def _split_indices(key, split_at="median", ignore_frac=0.0):
    """down_idx/up_idx from a precomputed key. Copied verbatim from v1."""
    v = _split_point(key, split_at)
    n = len(key)
    ignored = np.zeros(n, dtype=bool)
    n_ignore = int(round(ignore_frac * n))
    if n_ignore > 0:
        ignored[np.argsort(np.abs(key - v))[:n_ignore]] = True
    down_idx = np.flatnonzero((key < v) & ~ignored)
    up_idx = np.flatnonzero((key > v) & ~ignored)
    return down_idx, up_idx


def _mask_edges(K, edge_crop):
    """Zero a border on a (..., nx, ny) kernel. Copied verbatim from v1."""
    if edge_crop > 0:
        K[..., :edge_crop, :] = 0.0; K[..., -edge_crop:, :] = 0.0
        K[..., :, :edge_crop] = 0.0; K[..., :, -edge_crop:] = 0.0
    return K


def _peak_var_ij(K, edge_crop=2):
    """Pixel of max temporal variance, edge-masked. Copied verbatim from v1."""
    v = _mask_edges(np.asarray(K, float).copy(), edge_crop).var(axis=-3)
    while v.ndim > 2:
        v = v.mean(0)
    return tuple(int(x) for x in np.unravel_index(v.argmax(), v.shape))


def _filmstrip_lags(energy_profile, taus, n_show):
    """Column lags for a filmstrip: n_show lags spread around the energy peak.
    Copied verbatim from v1."""
    pk = int(energy_profile.argmax())
    return np.unique(np.clip(
        np.linspace(pk - 3, pk + 3, n_show).round().astype(int), 0, len(taus) - 1))


def calcium_kernel(frame_duration, tau_decay_s=0.150, tau_rise_s=0.01):
    """Causal GCaMP-like impulse response. Copied verbatim from v1."""
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
    """Synthetic calcium trace of a purely linear neuron. Copied verbatim from v1."""
    n_rel = len(frame_to_pattern)
    _, nx, ny = filter_txy.shape
    stim = np.zeros((n_rel, nx, ny), dtype=np.float32)
    valid = frame_to_pattern >= 0
    noise_xy = np.transpose(noise_array.astype(np.float32), (1, 0, 2))  # (nx, ny, npat)
    stim[valid] = np.transpose(noise_xy[:, :, frame_to_pattern[valid]], (2, 0, 1)) - baseline

    K = filter_txy.astype(np.float32)
    r = np.zeros(n_rel, dtype=np.float64)
    for j, tau in enumerate(taus):
        proj = np.tensordot(stim, K[j], axes=([1, 2], [0, 1]))
        if tau > 0:
            r[: n_rel - tau] += proj[tau:]
        elif tau < 0:
            r[-tau:] += proj[: n_rel + tau]
        else:
            r += proj

    if calcium_tau_decay_s and calcium_tau_decay_s > 0 and frame_duration:
        h = calcium_kernel(frame_duration, calcium_tau_decay_s, calcium_tau_rise_s)
        r = np.convolve(r, h)[:n_rel]

    r = (r - r.mean()) / (r.std() or 1.0)
    rng = np.random.default_rng(seed)
    return r + noise_sigma * rng.standard_normal(n_rel)


# %% ---------------------------------------------------------------------------
# Shared result interface -- every strategy below populates the SAME shape, so
# one plotting/video routine can iterate over a list of these and render them
# consistently (same colour scale conventions, same panel layout, same lag axis).
# ============================================================================


@dataclass
class MethodResult:
    """Common output of all four ON/OFF-split strategies.

    up_kernel/down_kernel/diff/sta : (n_lags, nx, ny), FULL FOV (crop=None),
        edge-masked at the source (see _mask_edges) -- display crops happen
        downstream, never baked into the stored arrays, so every method's
        kernels stay directly comparable and diff/null-comparisons never hit a
        shape mismatch from different crop boxes.
    n_up/n_down : event counts behind each half of the split. NaN for method 1
        (rectified_linear_baseline), which isn't event-based at all.
    taus : (n_lags,) int, shared lag convention across all methods:
        tau = (1 - n_f_filter_past) + j (same as pygor.strf.calculate_strf).
    meta : method-specific extras (peak pixel, split key, PCA scores, ...).
    """

    method_name: str
    up_kernel: np.ndarray
    down_kernel: np.ndarray
    taus: np.ndarray
    n_up: float
    n_down: float
    diff: np.ndarray = None
    sta: np.ndarray = None
    meta: dict = field(default_factory=dict)

    def __post_init__(self):
        if self.diff is None:
            self.diff = self.up_kernel - self.down_kernel
        if self.sta is None:
            self.sta = (self.up_kernel + self.down_kernel) / 2.0


# %% ---------------------------------------------------------------------------
# Method 1: rectified-linear baseline. NOT event-based -- the null hypothesis
# every other method should be judged against: what a purely linear cell's
# STRF looks like when naively split by sign.
# ============================================================================


def method_rectified_linear(obj, roi, taus):
    """Split pygor's own linear STRF by sign: positive -> up, |negative| -> down.

    diff (= up - down) reduces exactly to the raw STRF by construction -- this
    method carries NO information beyond the ordinary linear STA, which is the
    point: it's the reference every event-based method's "excess" is judged
    against.
    """
    strf = np.asarray(obj.strfs[roi], dtype=np.float64)  # (n_lags, nx, ny)
    assert strf.shape[0] == len(taus), "obj.strfs lag count doesn't match shared taus"
    up_kernel = np.clip(strf, 0, None)
    down_kernel = -np.clip(strf, None, 0)
    return MethodResult(
        method_name="rectified_linear_baseline",
        up_kernel=up_kernel, down_kernel=down_kernel, taus=taus,
        n_up=np.nan, n_down=np.nan,
        meta={"note": "not event-based; null-hypothesis reference only. "
                       "diff == raw STRF by construction; excluded from the "
                       "linear-null comparison (there's nothing to null-test)."},
    )


# %% ---------------------------------------------------------------------------
# Method 2: response-polarity STA. Reuses v1's polarity_sta logic verbatim --
# TWO independently detected event sets (response sign), each a plain
# full-field STA. Answers "STA split by response type", a DIFFERENT question
# than methods 3/4 (which split ONE event set by stimulus history).
# ============================================================================


def method_polarity_sta(
    noise_array, frame_to_pattern, up_frames, down_frames,
    n_f_filter_past, n_f_filter, edge_crop=2, max_gib=6,
):
    spatial_up, taus, _ = extract_snippets_spatial(
        noise_array, frame_to_pattern, up_frames,
        n_f_filter_past, n_f_filter, center=None, crop=None, max_gib=max_gib,
    )
    spatial_down, _, _ = extract_snippets_spatial(
        noise_array, frame_to_pattern, down_frames,
        n_f_filter_past, n_f_filter, center=None, crop=None, max_gib=max_gib,
    )
    up_k = _mask_edges(spatial_up.mean(axis=0), edge_crop)
    down_k = _mask_edges(spatial_down.mean(axis=0), edge_crop)
    return MethodResult(
        method_name="response_polarity_sta",
        up_kernel=up_k, down_kernel=down_k, taus=taus,
        n_up=len(up_frames), n_down=len(down_frames),
        meta={"note": "splits by RESPONSE polarity (independent pos/neg "
                       "detect_events calls), not stimulus history -- a "
                       "different question than methods 3/4; cheap reference."},
    )


# %% ---------------------------------------------------------------------------
# Method 3: anchor-based split. Reuses v1's emergent_updown_sta logic -- ONE
# event set (calcium-up only), full-field STA -> data-found peak pixel -> that
# pixel's own pre-event contrast as the split key -> two sub-STAs.
# ============================================================================


def method_anchor_split(
    noise_array, frame_to_pattern, event_frames, frame_duration,
    n_f_filter_past, n_f_filter, n_neighbours=4, edge_crop=2,
    pre_window_s=0.25, ignore_frac=0.2, split_at="midrange",
    peak_method="var", max_gib=6,
):
    spatial_snips, taus, _ = extract_snippets_spatial(
        noise_array, frame_to_pattern, event_frames,
        n_f_filter_past, n_f_filter, center=None, crop=None, max_gib=max_gib,
    )
    sta = _mask_edges(spatial_snips.mean(axis=0), edge_crop)
    nx, ny = sta.shape[1], sta.shape[2]

    if peak_method == "var":
        px, py = _peak_var_ij(sta, edge_crop=edge_crop)
    elif peak_method == "amp":
        proj = np.abs(sta).max(axis=0)
        px, py = (int(v) for v in np.unravel_index(proj.argmax(), proj.shape))
    else:
        raise ValueError("peak_method must be 'var' or 'amp'")

    pixels = pixel_neighbourhood(px, py, n_neighbours, edge_crop, nx, ny)
    timecourse = neighbourhood_timecourse(spatial_snips, pixels)
    key = key_from_timecourse(timecourse, taus, frame_duration, pre_window_s)
    down_idx, up_idx = _split_indices(key, split_at, ignore_frac)

    down_k = _mask_edges(spatial_snips[down_idx].mean(axis=0), edge_crop)
    up_k = _mask_edges(spatial_snips[up_idx].mean(axis=0), edge_crop)

    return MethodResult(
        method_name="anchor_split",
        up_kernel=up_k, down_kernel=down_k, taus=taus,
        n_up=len(up_idx), n_down=len(down_idx),
        meta={"peak_px": px, "peak_py": py, "pixels": pixels, "key": key,
              "down_idx": down_idx, "up_idx": up_idx},
    )


# %% ---------------------------------------------------------------------------
# Method 4 (NEW): PCA-key split. Same calcium-up-only event set as method 3,
# but the split key comes from data instead of a hand-picked anchor pixel: top
# eigenvector of the event-triggered residual covariance (plain PCA, no
# prior-whitening, no STC machinery -- deliberately cheap, one eigenvector).
#
# CAVEAT (check empirically, don't oversell): PC1 is the direction of max
# variance across events, which only aligns with a genuine ON/OFF divide if
# that divide really is the dominant source of variance. If the projected
# score histogram is unimodal/Gaussian-ish rather than bimodal, the "split" is
# just cutting a continuous distribution in half, not separating two real
# populations -- report this plainly via _bimodality_check.
# ============================================================================


def _bimodality_check(scores, seed=0):
    """Two cheap bimodality proxies (NOT a formal dip test):

    - Sarle's bimodality coefficient (skew, kurtosis based); > 0.555 (the
      value for a uniform distribution) is a common rule-of-thumb flag.
    - BIC comparison of a 1- vs 2-component 1-D GMM fit to the scores; GMM(2)
      winning is a second, independent signal.

    Both are heuristics, reported as-is -- neither proves a real 2-population
    split, they just flag whether the score histogram LOOKS bimodal.
    """
    from scipy.stats import kurtosis, skew
    from sklearn.mixture import GaussianMixture

    x = np.asarray(scores, float)
    n = len(x)
    g = skew(x)
    k = kurtosis(x, fisher=False)  # Pearson kurtosis, normal = 3
    bc = (g**2 + 1) / (k + 3 * (n - 1) ** 2 / ((n - 2) * (n - 3)))
    gmm1 = GaussianMixture(1, random_state=seed).fit(x[:, None])
    gmm2 = GaussianMixture(2, random_state=seed).fit(x[:, None])
    bic1, bic2 = gmm1.bic(x[:, None]), gmm2.bic(x[:, None])
    return {
        "n": n, "skew": float(g), "kurtosis": float(k),
        "bimodality_coeff": float(bc), "bic_1comp": float(bic1),
        "bic_2comp": float(bic2), "gmm2_preferred": bool(bic2 < bic1),
    }


def method_pca_split(
    noise_array, frame_to_pattern, event_frames, frame_duration,
    n_f_filter_past, n_f_filter, crop=8, edge_crop=2,
    pca_lag_window_s=(-1.0, 0.2), split_at="midrange", ignore_frac=0.2,
    max_gib=6, seed=0,
):
    """PCA-key split.

    crop + pca_lag_window_s restrict the DIMENSIONALITY of the PCA fit only
    (spatial crop around the data-found peak pixel, time crop to the
    physiologically relevant window) -- same tractability trick v1's STC
    section uses (see v1's stimulus_prior_basis note: D must stay << n_events
    or the top eigenvector is dominated by sampling noise, not signal). The
    returned up_kernel/down_kernel are still full-field, full-lag (event
    membership from the split is just applied back to the full ensemble).

    edge_crop matters here for the same reason v1's emergent_updown_sta flags:
    a noisy/high-variance border pixel (stimulus doesn't tile the FOV edge)
    could dominate the residual covariance and hijack PC1 -- the peak-pixel
    search and the crop box are both taken from the edge-masked STA.
    """
    from sklearn.decomposition import PCA

    spatial_snips, taus, _ = extract_snippets_spatial(
        noise_array, frame_to_pattern, event_frames,
        n_f_filter_past, n_f_filter, center=None, crop=None, max_gib=max_gib,
    )
    sta_full = _mask_edges(spatial_snips.mean(axis=0), edge_crop)
    nx_full, ny_full = sta_full.shape[1], sta_full.shape[2]
    px, py = _peak_var_ij(sta_full, edge_crop=edge_crop)
    x0, x1, y0, y1 = _crop_box(px, py, crop, nx_full, ny_full)

    lag_s = taus * frame_duration
    lo_s, hi_s = pca_lag_window_s
    lag_mask = (lag_s >= lo_s) & (lag_s <= hi_s)
    if not lag_mask.any():
        lag_mask = np.ones_like(lag_s, dtype=bool)

    snips_c = spatial_snips[:, lag_mask][:, :, x0:x1, y0:y1]
    sta_c = snips_c.mean(axis=0)
    n_ev = len(event_frames)
    resid = (snips_c - sta_c[None]).reshape(n_ev, -1).astype(np.float64)

    pca = PCA(n_components=1, svd_solver="randomized", random_state=seed).fit(resid)
    scores = pca.transform(resid)[:, 0]
    down_idx, up_idx = _split_indices(scores, split_at, ignore_frac)

    down_k = _mask_edges(spatial_snips[down_idx].mean(axis=0), edge_crop)
    up_k = _mask_edges(spatial_snips[up_idx].mean(axis=0), edge_crop)
    bimodality = _bimodality_check(scores, seed=seed)

    return MethodResult(
        method_name="pca_key_split",
        up_kernel=up_k, down_kernel=down_k, taus=taus,
        n_up=len(up_idx), n_down=len(down_idx),
        meta={"peak_px": px, "peak_py": py, "box": (x0, x1, y0, y1),
              "pca_lag_window_s": pca_lag_window_s, "scores": scores,
              "explained_variance_ratio": float(pca.explained_variance_ratio_[0]),
              "bimodality": bimodality, "down_idx": down_idx, "up_idx": up_idx},
    )



# %% ---------------------------------------------------------------------------
# Linear-null validation, applied to every event-based method (2-4). Reuses
# v1's simulate_linear_trace + linear_null_kernels pattern: build a synthetic
# linear-null cell using the method's OWN recovered sta as the filter, run it
# through the SAME detect->split pipeline, compare diff_real vs diff_null.
# ============================================================================


def null_diff_for_event_method(
    method_fn, sta_filter, noise_array, frame_to_pattern, taus,
    n_f_filter_past, n_f_filter, frame_duration, threshold,
    two_sided=False, seed=0,
):
    """Re-run an event-based method on a linear-null trace built from its own STA.

    method_fn(pos_frames) -> MethodResult             for anchor/PCA methods.
    method_fn(pos_frames, neg_frames) -> MethodResult  for polarity_sta (two_sided=True).
    """
    trace = simulate_linear_trace(
        sta_filter, noise_array, frame_to_pattern, taus,
        frame_duration=frame_duration, seed=seed,
    )
    dif = np.diff(trace, prepend=trace[0])
    base = dif[: min(100, len(dif))]
    dif = (dif - base.mean()) / (base.std() or 1.0)
    pos_frames = np.flatnonzero(dif > threshold)
    if two_sided:
        neg_frames = np.flatnonzero(dif < -threshold)
        return method_fn(pos_frames, neg_frames)
    return method_fn(pos_frames)


def excess_over_null(diff_real, diff_null, edge_crop=0):
    """||real||, ||null||, and the excess fraction ||real-null|| / ||real||.
    Same metric as v1's plot_diff_compare, factored out for reuse without the plot."""
    R = np.array(diff_real, float)
    N = np.array(diff_null, float)
    if edge_crop > 0:
        R = _mask_edges(R.copy(), edge_crop)
        N = _mask_edges(N.copy(), edge_crop)
    resid = R - N
    p_real = float(np.linalg.norm(R))
    p_null = float(np.linalg.norm(N))
    p_exc = float(np.linalg.norm(resid))
    return {"power_real": p_real, "power_null": p_null, "excess_frac": p_exc / (p_real or 1.0)}


# %% ---------------------------------------------------------------------------
# Driver: run all four methods + null tests for one ROI.
# ============================================================================


def run_all_methods(
    obj, noise_array, roi, frame_to_pattern, frame_duration,
    n_f_filter_past, n_f_filter, taus, trigger_start, n_f_relevant,
    thresh=1.0, pre_window_s=0.25, ignore_frac=0.2, split_at="midrange",
    crop=10, edge_crop=2,
):
    event_frames, _ = detect_events(
        obj, roi, threshold=thresh, sign="pos",
        trigger_start=trigger_start, n_f_relevant=n_f_relevant,
    )
    neg_frames, _ = detect_events(
        obj, roi, threshold=thresh, sign="neg",
        trigger_start=trigger_start, n_f_relevant=n_f_relevant,
    )

    res1 = method_rectified_linear(obj, roi, taus)
    res2 = method_polarity_sta(
        noise_array, frame_to_pattern, event_frames, neg_frames,
        n_f_filter_past, n_f_filter, edge_crop=edge_crop,
    )
    res3 = method_anchor_split(
        noise_array, frame_to_pattern, event_frames, frame_duration,
        n_f_filter_past, n_f_filter, edge_crop=edge_crop,
        pre_window_s=pre_window_s, ignore_frac=ignore_frac, split_at=split_at,
    )

    null2 = null_diff_for_event_method(
        lambda pos, neg: method_polarity_sta(
            noise_array, frame_to_pattern, pos, neg,
            n_f_filter_past, n_f_filter, edge_crop=edge_crop,
        ),
        res2.sta, noise_array, frame_to_pattern, taus,
        n_f_filter_past, n_f_filter, frame_duration, thresh, two_sided=True,
    )
    null3 = null_diff_for_event_method(
        lambda pos: method_anchor_split(
            noise_array, frame_to_pattern, pos, frame_duration,
            n_f_filter_past, n_f_filter, edge_crop=edge_crop,
            pre_window_s=pre_window_s, ignore_frac=ignore_frac, split_at=split_at,
        ),
        res3.sta, noise_array, frame_to_pattern, taus,
        n_f_filter_past, n_f_filter, frame_duration, thresh,
    )

    excess = {
        "method2_polarity": excess_over_null(res2.diff, null2.diff, edge_crop=edge_crop),
        "method3_anchor": excess_over_null(res3.diff, null3.diff, edge_crop=edge_crop),
    }
    baseline_power = float(np.linalg.norm(_mask_edges(res1.diff.copy(), edge_crop)))

    return {
        "roi": roi, "results": [res1, res2, res3],
        "nulls": {"method2": null2, "method3": null3},
        "excess": excess, "baseline_power": baseline_power,
        "n_events_pos": len(event_frames), "n_events_neg": len(neg_frames),
    }


def print_summary(summary):
    r = summary
    print(f"\n=== ROI {r['roi']} ===")
    print(f"n_events: pos(up)={r['n_events_pos']}  neg(down)={r['n_events_neg']}")
    print(f"method 1 (rectified-linear baseline, reference only): "
          f"||diff||={r['baseline_power']:.3f}")
    for key, label in [("method2_polarity", "method 2 (response-polarity STA)"),
                       ("method3_anchor", "method 3 (anchor split)")]:
        e = r["excess"][key]
        print(f"{label}: ||real||={e['power_real']:.3f}  ||null||={e['power_null']:.3f}  "
              f"excess_frac={e['excess_frac']:.3f}")


# %% ---------------------------------------------------------------------------
# Shared plotting: filmstrip grid, one row-block (down/up/diff) per method, at
# the SAME lag columns and the SAME shared display crop -- generalises v1's
# plot_spatial_kernels/plot_onoff_overlay to an arbitrary list of MethodResult.
# ============================================================================


def _display_edge_mask(K, edge_crop):
    """Border of width edge_crop -> NaN (display-only; analysis kernels stay
    zero-masked via _mask_edges). NaN renders as flat grey via cmap.set_bad,
    so the SWN "shifting noise" edge artifact reads as INVALID, not as a real
    near-zero pixel value blending into the colour scale."""
    K = np.array(K, float)
    if edge_crop > 0:
        K[..., :edge_crop, :] = np.nan; K[..., -edge_crop:, :] = np.nan
        K[..., :, :edge_crop] = np.nan; K[..., :, -edge_crop:] = np.nan
    return K


def _cmap_bad(name, bad="0.5"):
    """Copy of a named colormap with NaN pixels forced to flat grey."""
    cmap = plt.get_cmap(name).copy()
    cmap.set_bad(bad)
    return cmap


def _method_display_arrays(result, crop_box, edge_crop):
    """down/up/diff for one method: border NaN-masked on the FULL array first
    (so the true FOV edge is what gets flagged, not whatever happens to sit at
    a crop window's edge), then cropped -- crop_box=None keeps the full FOV.
    Never transposed: arrays stay (n_lags, nx, ny) all the way to imshow."""
    D = _display_edge_mask(result.down_kernel, edge_crop)
    U = _display_edge_mask(result.up_kernel, edge_crop)
    if crop_box is not None:
        x0, x1, y0, y1 = crop_box
        D = D[:, x0:x1, y0:y1]
        U = U[:, x0:x1, y0:y1]
    diff = U - D
    return D, U, diff


def _build_display_rows(results, crop_box, edge_crop, reference_kernel=None, reference_label=None):
    """(name, D, U, diff) per row, reference row (if given) prepended first.

    Shared by plot_methods_grid and play_methods_grid so a raw, UNSPLIT filter
    (e.g. obj.strfs[roi]) can be shown identically in all three columns/panels
    -- same image, own colour scale -- ahead of the split methods, without
    pretending it has an up/down decomposition of its own.
    """
    rows = [(r.method_name, *_method_display_arrays(r, crop_box, edge_crop)) for r in results]
    if reference_kernel is not None:
        ref = _display_edge_mask(reference_kernel, edge_crop)
        if crop_box is not None:
            x0, x1, y0, y1 = crop_box
            ref = ref[:, x0:x1, y0:y1]
        rows = [(reference_label, ref, ref, ref)] + rows
    return rows


def plot_methods_grid(
    results, taus, frame_duration, roi=None, crop_box=None, edge_crop=2, n_show=7,
    reference_kernel=None, reference_label="pygor STRF (raw, from obj.strfs)",
):
    """Filmstrip: rows = [down, up, diff] per method (stacked), columns = lags.

    crop_box = (x0,x1,y0,y1) in full-FOV pixel coords, shared across ALL
    methods (None -> full FOV, no cropping) -- so spatial position is directly
    comparable; methods differ in HOW the split is made, not in what patch of
    the FOV is displayed. Lag columns are also shared (picked from the summed
    energy across all methods' diff), so every row lines up at the same
    physical lag. Border pixels are NaN-masked -> flat grey (SWN edge
    artifact), display-only. reference_kernel (optional), e.g. obj.strfs[roi]
    -- shown as its own row ahead of `results` (see _build_display_rows).
    """
    lag_s = taus * frame_duration
    per_method = _build_display_rows(results, crop_box, edge_crop, reference_kernel, reference_label)
    energy_total = np.zeros(len(taus))
    for _, _, _, diff in per_method:
        energy_total += np.nanmean(np.abs(diff), axis=(1, 2))
    idxs = _filmstrip_lags(energy_total, taus, n_show)

    n_methods = len(per_method)
    fig, axs = plt.subplots(
        n_methods * 3, len(idxs), figsize=(1.7 * len(idxs), 1.7 * n_methods * 3), squeeze=False,
    )
    grey_cmap, diff_cmap = _cmap_bad("gray"), _cmap_bad("RdBu_r")
    for mi, (name, D, U, diff) in enumerate(per_method):
        vmax = float(np.nanmax(np.abs([D, U]))) or 1.0
        dmax = float(np.nanmax(np.abs(diff))) or vmax
        rows = [("down/OFF", D, grey_cmap, vmax), ("up/ON", U, grey_cmap, vmax),
                ("diff\n(up-down)", diff, diff_cmap, dmax)]
        for ri, (label, K, cmap, vm) in enumerate(rows):
            row = mi * 3 + ri
            for ci, ti in enumerate(idxs):
                ax = axs[row][ci]
                ax.imshow(K[ti], origin="lower", cmap=cmap, vmin=-vm, vmax=vm)
                ax.set_xticks([]); ax.set_yticks([])
                if row == 0:
                    ax.set_title(f"{lag_s[ti] * 1000:.0f} ms", fontsize=8)
                if ci == 0:
                    ax.set_ylabel(f"{name}\n{label}", fontsize=7)
    title = f"ROI {roi}: method comparison" if roi is not None else "method comparison"
    fig.suptitle(title, y=1.0)
    fig.tight_layout()
    return fig


def play_methods_grid(
    results, taus, frame_duration, roi=None, crop_box=None, edge_crop=2, dur_s=None,
    reference_kernel=None, reference_label="pygor STRF (raw, from obj.strfs)",
):
    """Synchronised grid movie: rows = methods, columns = [down, up, diff],
    ONE shared FuncAnimation time axis -- generalises v1's play_components
    (hardcoded to a fixed few side-by-side panels) to an arbitrary method list.

    crop_box=None -> full FOV, no display crop. Border pixels NaN-masked ->
    flat grey (SWN edge artifact), same as plot_methods_grid. Never
    transposed. reference_kernel (optional), e.g. obj.strfs[roi] -- a raw,
    UNSPLIT filter shown as its own row (same image in all three columns, own
    colour scale) so the plain linear STRF is visible for comparison without
    pretending it has an up/down decomposition of its own.
    """
    import matplotlib.animation

    lag_ms = taus * frame_duration * 1000
    arrays = _build_display_rows(results, crop_box, edge_crop, reference_kernel, reference_label)

    n_frames = arrays[0][1].shape[0]
    if dur_s is None:
        dur_s = n_frames * frame_duration

    # PER-ROW clim (not shared across rows): rectified_linear_baseline and the
    # raw-STRF reference row are both in native STRF units, which dwarf the
    # event-based methods' ~+-0.5 contrast averages (same issue v1's
    # play_components flags for its STA panel). A single global clim would peg
    # every row's colour scale to whichever one has the largest amplitude and
    # wash the rest out to flat grey.
    grey_clims = [float(np.nanmax(np.abs([D, U]))) or 1.0 for _, D, U, _ in arrays]
    diff_clims = [float(np.nanmax(np.abs(diff))) or gc for gc, (_, _, _, diff) in zip(grey_clims, arrays)]

    n_methods = len(arrays)
    plt.rcParams["animation.html"] = "jshtml"
    fig, axs = plt.subplots(n_methods, 3, figsize=(3 * 1.8, n_methods * 1.8), squeeze=False)
    grey_cmap, diff_cmap = _cmap_bad("Greys_r"), _cmap_bad("RdBu_r")
    ims = []
    for mi, (name, D, U, diff) in enumerate(arrays):
        gc, dc = grey_clims[mi], diff_clims[mi]
        panels = [("down/OFF", D, grey_cmap, gc), ("up/ON", U, grey_cmap, gc),
                  ("diff", diff, diff_cmap, dc)]
        row_ims = []
        for ci, (label, A, cmap, vm) in enumerate(panels):
            ax = axs[mi][ci]
            im = ax.imshow(A[0], origin="lower", cmap=cmap, vmin=-vm, vmax=vm, interpolation="none")
            ax.set_xticks([]); ax.set_yticks([])
            if mi == 0:
                ax.set_title(label, fontsize=8)
            if ci == 0:
                ax.set_ylabel(name, fontsize=7)
            row_ims.append(im)
        ims.append(row_ims)
    sup = fig.suptitle("", fontsize=9, y=0.999)
    fig.tight_layout(pad=0.3)

    def video(frame):
        artists = []
        for row_ims, (name, D, U, diff) in zip(ims, arrays):
            row_ims[0].set_array(D[frame]); row_ims[1].set_array(U[frame]); row_ims[2].set_array(diff[frame])
            artists.extend(row_ims)
        tag = f"ROI {roi}  |  " if roi is not None else ""
        sup.set_text(f"{tag}lag = {lag_ms[frame]:+.0f} ms   (0 = event)")
        return artists

    anim = matplotlib.animation.FuncAnimation(
        fig, video, frames=n_frames, interval=dur_s / n_frames * 1000, repeat_delay=500,
    )
    plt.close(fig)
    return anim


# %% ---------------------------------------------------------------------------
# PCA component reconstruction: pca_key_split (method 4) uses only PC1 as a
# split key; this looks at what the top N components actually LOOK like,
# spatially and over lag. Same tractability trick as method_pca_split (crop +
# lag-window restrict the PCA FIT dimensionality only) -- peak-pixel search,
# crop box and lag mask are copied verbatim from method_pca_split.
# ============================================================================


def _bin_spatial(arr, bin_factor):
    """Block-average the last two (nx, ny) axes of `arr` by bin_factor.
    Trims any remainder pixels that don't fill a whole block. bin_factor<=1 is
    a no-op -- lets callers pass it through unconditionally."""
    if bin_factor <= 1:
        return arr
    *lead, nx, ny = arr.shape
    nx_t, ny_t = (nx // bin_factor) * bin_factor, (ny // bin_factor) * bin_factor
    arr = arr[..., :nx_t, :ny_t]
    new_shape = (*lead, nx_t // bin_factor, bin_factor, ny_t // bin_factor, bin_factor)
    return arr.reshape(new_shape).mean(axis=(-3, -1))


def pca_component_reconstruction(
    noise_array, frame_to_pattern, event_frames, frame_duration,
    n_f_filter_past, n_f_filter, crop=8, edge_crop=2,
    pca_lag_window_s=(-1.0, 0.2), n_fit=5, bin_factor=1, max_gib=6, seed=0,
):
    """Fit PCA(n_components=n_fit) on the event-triggered residual ensemble
    (same cropped ensemble method_pca_split fits, but keep every component
    instead of just PC1). Returns each component's own eigenvector reshaped
    back into the (n_lags_c, nx_c, ny_c) box it was fit in -- directly
    viewable as a filmstrip/movie, not just usable as a scalar split key.

    bin_factor=1 (default): unchanged behaviour, crop around the data-found
    peak pixel (same D-control knob as method_pca_split -- see its docstring:
    D must stay << n_events or the components chase sampling noise, not
    signal). bin_factor>1: fit over the FULL FOV instead, block-averaged
    spatially by bin_factor -- trades spatial resolution for dimensionality so
    D stays tractable while covering the whole FOV rather than a local patch.
    `crop` is ignored when bin_factor>1.
    """
    from sklearn.decomposition import PCA

    spatial_snips, taus, _ = extract_snippets_spatial(
        noise_array, frame_to_pattern, event_frames,
        n_f_filter_past, n_f_filter, center=None, crop=None, max_gib=max_gib,
    )
    sta_full = _mask_edges(spatial_snips.mean(axis=0), edge_crop)
    nx_full, ny_full = sta_full.shape[1], sta_full.shape[2]
    px, py = _peak_var_ij(sta_full, edge_crop=edge_crop)

    lag_s = taus * frame_duration
    lo_s, hi_s = pca_lag_window_s
    lag_mask = (lag_s >= lo_s) & (lag_s <= hi_s)
    if not lag_mask.any():
        lag_mask = np.ones_like(lag_s, dtype=bool)
    lag_idx = np.flatnonzero(lag_mask)
    n_lags_c = len(lag_idx)

    if bin_factor > 1:
        x0, x1, y0, y1 = 0, nx_full, 0, ny_full
        snips_c = _bin_spatial(spatial_snips[:, lag_mask], bin_factor)
    else:
        x0, x1, y0, y1 = _crop_box(px, py, crop, nx_full, ny_full)
        snips_c = spatial_snips[:, lag_mask][:, :, x0:x1, y0:y1]
    nx_c, ny_c = snips_c.shape[2], snips_c.shape[3]

    sta_c = snips_c.mean(axis=0)
    n_ev = len(event_frames)
    resid = (snips_c - sta_c[None]).reshape(n_ev, -1).astype(np.float64)

    n_fit = min(n_fit, n_ev - 1, resid.shape[1])
    pca = PCA(n_components=n_fit, svd_solver="randomized", random_state=seed).fit(resid)
    scores = pca.transform(resid)
    components = pca.components_.reshape(n_fit, n_lags_c, nx_c, ny_c)

    return {
        "components": components, "explained_variance_ratio": pca.explained_variance_ratio_,
        "scores": scores, "taus_c": taus[lag_idx], "box": (x0, x1, y0, y1),
        "peak_px": px, "peak_py": py, "n_events": n_ev, "n_dims": resid.shape[1],
        "bin_factor": bin_factor,
    }


def plot_pca_components(components, taus_c, frame_duration, explained_variance_ratio, n_show, n_show_lags=7):
    """Filmstrip: one row per shown PC (signed spatial pattern per lag),
    columns = lags -- picked PER-PC from that PC's own |value| energy (unlike
    plot_methods_grid's shared columns; different PCs need not peak at the
    same lag)."""
    n_show = min(n_show, components.shape[0])
    lag_s = taus_c * frame_duration
    diff_cmap = _cmap_bad("RdBu_r")

    idxs_per_pc = [_filmstrip_lags(np.abs(components[i]).mean(axis=(1, 2)), taus_c, n_show_lags)
                   for i in range(n_show)]
    n_cols = max(len(idxs) for idxs in idxs_per_pc)
    fig, axs = plt.subplots(n_show, n_cols, figsize=(1.7 * n_cols, 1.7 * n_show), squeeze=False)
    for pi in range(n_show):
        K = components[pi]
        vmax = float(np.abs(K).max()) or 1.0
        idxs = idxs_per_pc[pi]
        for ci in range(n_cols):
            ax = axs[pi][ci]
            if ci < len(idxs):
                ti = idxs[ci]
                ax.imshow(K[ti], origin="lower", cmap=diff_cmap, vmin=-vmax, vmax=vmax)
                if pi == 0:
                    ax.set_title(f"{lag_s[ti] * 1000:.0f} ms", fontsize=8)
            else:
                ax.axis("off")
            ax.set_xticks([]); ax.set_yticks([])
            if ci == 0:
                ax.set_ylabel(f"PC{pi + 1}\n({explained_variance_ratio[pi] * 100:.1f}% var)", fontsize=7)
    fig.suptitle("PCA components -- spatial reconstruction", y=1.0)
    fig.tight_layout()
    return fig


def play_pca_components(components, taus_c, frame_duration, explained_variance_ratio, n_show, dur_s=None):
    """Synchronised movie: one panel per shown PC, own signed colour scale,
    shared lag axis (taus_c). No down/up/diff triplet -- a PC has no natural
    up/down split, just its own signed spatial pattern per lag."""
    import matplotlib.animation

    n_show = min(n_show, components.shape[0])
    lag_ms = taus_c * frame_duration * 1000
    n_frames = components.shape[1]
    if dur_s is None:
        dur_s = n_frames * frame_duration

    plt.rcParams["animation.html"] = "jshtml"
    fig, axs = plt.subplots(1, n_show, figsize=(1.8 * n_show, 2.0), squeeze=False)
    diff_cmap = _cmap_bad("RdBu_r")
    ims = []
    for pi in range(n_show):
        K = components[pi]
        vmax = float(np.abs(K).max()) or 1.0
        ax = axs[0][pi]
        im = ax.imshow(K[0], origin="lower", cmap=diff_cmap, vmin=-vmax, vmax=vmax, interpolation="none")
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(f"PC{pi + 1} ({explained_variance_ratio[pi] * 100:.1f}% var)", fontsize=8)
        ims.append(im)
    sup = fig.suptitle("", fontsize=9, y=0.999)
    fig.tight_layout(pad=0.3)

    def video(frame):
        for im, pi in zip(ims, range(n_show)):
            im.set_array(components[pi, frame])
        sup.set_text(f"lag = {lag_ms[frame]:+.0f} ms   (0 = event)")
        return ims

    anim = matplotlib.animation.FuncAnimation(
        fig, video, frames=n_frames, interval=dur_s / n_frames * 1000, repeat_delay=500,
    )
    plt.close(fig)
    return anim


# %% ---------------------------------------------------------------------------
# Demo: run end-to-end on the real recording, ROI 44 (v1's own demo ROI).
# ============================================================================

frame_duration, n_f_filter, n_f_filter_past = strf_window_from_obj(obj)
frame_to_pattern, trigger_start, n_f_relevant = build_frame_to_pattern(obj, noise_array)
taus = np.arange(n_f_filter) + (1 - n_f_filter_past)  # shared lag axis, methods 1-3
print(f"frame_duration={frame_duration:.4f}s  n_f_filter={n_f_filter}  "
      f"n_f_filter_past={n_f_filter_past}  n_f_relevant={n_f_relevant}")

demo_roi = 8 # 44, 8
thresh = 1.0
pre_window_s = 0.25
ignore_frac = 0.2
split_at = "midrange"
crop = 10       # shared DISPLAY crop half-width
edge_crop = 2

summary = run_all_methods(
    obj, noise_array, demo_roi, frame_to_pattern, frame_duration,
    n_f_filter_past, n_f_filter, taus, trigger_start, n_f_relevant,
    thresh=thresh, pre_window_s=pre_window_s, ignore_frac=ignore_frac,
    split_at=split_at, crop=crop, edge_crop=edge_crop,
)
print_summary(summary)

# Summary comparison plot -- shared crop centred on method 3's anchor peak
# (data-driven, not pygor's STRF) so pygor's raw STRF + all split methods are
# shown at the same patch of the FOV.
res3 = summary["results"][2]
px, py = res3.meta["peak_px"], res3.meta["peak_py"]
nx_full, ny_full = obj.strfs.shape[2], obj.strfs.shape[3]
crop_box = _crop_box(px, py, crop, nx_full, ny_full)
print(f"shared display crop centred on anchor peak ({px},{py}) -> box={crop_box}")

# Row order: pygor STRF (raw reference) -> rectified-linear baseline ->
# response-polarity STA -> anchor split.
fig = plot_methods_grid(
    summary["results"], taus, frame_duration, roi=demo_roi,
    crop_box=crop_box, edge_crop=edge_crop, n_show=7,
    reference_kernel=obj.strfs[demo_roi],
    reference_label="pygor STRF (raw, from obj.strfs)",
)
plt.show()

# %% Summary comparison video -- full FOV (no crop), same row order as the
# static grid above (pygor STRF first, from obj.strfs, not re-derived).
vid = play_methods_grid(
    summary["results"], taus, frame_duration, roi=demo_roi,
    crop_box=None, edge_crop=edge_crop,
    reference_kernel=obj.strfs[demo_roi],
    reference_label="pygor STRF (raw, from obj.strfs)",
)
vid

# %% PCA component reconstruction -- own cell, decoupled from run_all_methods
# (pca is no longer part of the combined summary/null pipeline; its
# interesting content is the top components' spatial-temporal structure, not
# an up/down split). Recomputes its own event frames rather than reusing
# `summary`.
pca_lag_window_s = (-1.0, 0.2)
pca_crop = 8            # spatial crop half-width around the data-found peak (ignored if bin_factor>1)
pca_bin_factor = 2      # >1 -> full FOV, block-averaged by this factor, instead of a local crop
n_fit_components = 5    # PCs to actually fit
n_show_components = n_fit_components   # PCs to plot/play (<= n_fit_components)

pca_event_frames, _ = detect_events(
    obj, demo_roi, threshold=thresh, sign="pos",
    trigger_start=trigger_start, n_f_relevant=n_f_relevant,
)
pca_recon = pca_component_reconstruction(
    noise_array, frame_to_pattern, pca_event_frames, frame_duration,
    n_f_filter_past, n_f_filter, crop=pca_crop, edge_crop=edge_crop,
    pca_lag_window_s=pca_lag_window_s, n_fit=n_fit_components, bin_factor=pca_bin_factor,
)
print(f"PCA fit on {pca_recon['n_events']} events, peak pixel ({pca_recon['peak_px']},{pca_recon['peak_py']})")
print(f"D (fit dims) = {pca_recon['n_dims']}  vs n_events = {pca_recon['n_events']}  "
      f"(D << n_events wanted, e.g. D < n_events/5 -- else components chase sampling noise)")
print("explained variance ratio per PC:", np.round(pca_recon["explained_variance_ratio"], 3))

fig = plot_pca_components(
    pca_recon["components"], pca_recon["taus_c"], frame_duration,
    pca_recon["explained_variance_ratio"], n_show=n_show_components,
)
plt.show()

pca_vid = play_pca_components(
    pca_recon["components"], pca_recon["taus_c"], frame_duration,
    pca_recon["explained_variance_ratio"], n_show=n_show_components,
)
pca_vid

# %% Spot-check a couple of v1's flagged "nonlinear candidate" ROIs (8, 20, 26,
# 28, 38, 61) -- numbers only, no plots, to see whether excess-over-null looks
# any different from the ROI 44 demo.
for spot_roi in [28, 26]:
    spot_summary = run_all_methods(
        obj, noise_array, spot_roi, frame_to_pattern, frame_duration,
        n_f_filter_past, n_f_filter, taus, trigger_start, n_f_relevant,
        thresh=thresh, pre_window_s=pre_window_s, ignore_frac=ignore_frac,
        split_at=split_at, crop=crop, edge_crop=edge_crop,
    )
    print_summary(spot_summary)

# %%
