# %%
import pygor
import pygor.load

import matplotlib.pyplot as plt
import numpy as np

%load_ext autoreload
%autoreload 2

# %% ---------------------------------------------------------------------------
# stc_analysis: spike/event-triggered covariance (STC), split out of
# dev/snippet_analysis.py (v1) into its own file because it answers a
# DIFFERENT kind of question than snippet_analysis_v2.py's up/down-split
# methods and needs its own demo/validation flow.
#
# MOTIVATION (from snippet_analysis_v2.py's method 5): "does the cell respond
# to a single pixel's ON state AND its OFF state" turned out to be
# information-theoretically untestable using that ONE pixel's own statistics,
# for a binary (+-0.5) noise stimulus -- a single bit supports exactly one
# linear sensitivity parameter, full stop, regardless of method (confirmed via
# joint regression, conditional means, AND single-pixel variance -- see v2's
# method_neighbourhood_energy section note). The fix there was to average a
# small NEIGHBOURHOOD of pixels first, giving real graded magnitude to test a
# rectification hypothesis against -- but that only tests ONE hand-picked
# neighbourhood/lag-window combination at a time.
#
# STC is the general version of that fix: it looks at the FULL covariance of
# the event-triggered stimulus ensemble (across ALL pixel-lags in the fit
# window jointly, not just one hand-picked neighbourhood average), and finds
# the eigenvectors along which that covariance departs from the stimulus
# prior. A pixel (or small patch) that drives the response via EITHER
# polarity shows up as inflated variance along some direction in that joint
# space -- exactly the multi-dimensional generalisation of the
# neighbourhood-energy idea, found automatically instead of by hand-picking a
# crop box in advance.
#
# STRF AXIS CONVENTION: same as v2 -- obj.strfs[roi] is (n_lags, nx, ny), with
# nx == noise_array.shape[1], ny == noise_array.shape[0]. Checked directly
# against pygor.strf.calculate_strf's construction (see v2's header comment
# for the full derivation); not re-derived here.

# %% Load demo data (same dataset as v1/v2's demo cells)
# Load the TRUE base pattern (grid_shape [6,10]), not the shuffled/jittered
# 4x-upsampled array (.npy file, noise_jitter in the h5 below) v1/v2/earlier
# v3 all used. The 24x40 "pixels" the .npy exposes are ~960 nominal values
# but only 60 are independent -- each base cell is jitter-shifted by a
# per-frame sub-pixel offset (h5 'shift' dataset, shift_ratio=0.25 = 1/4px
# steps -> exactly the 4x upsample). Every STC dimensionality estimate this
# session (D=5491 at crop=8, D=18240 at crop=None) was computed against a
# spatial resolution 16x finer than the stimulus actually supports -- v1's
# own old comment ("the shifting noise is not white in the snippet basis")
# was already pointing at this. Working at native 6x10 resolution trades
# spatial resolution for a real fix: D drops ~16x, oversample ratio goes
# from 14.6x to ~70x at the same n_sample.
import h5py
with h5py.File("/mnt/data/IGOR arrays/jitternoise_SINGLEcolour_30000x6x10_0.25_1.h5", "r") as _f:
    noise_base = _f["noise"][:]  # (n_patterns, ny, nx) = (30000, 6, 10)
noise_array = np.transpose(noise_base, (1, 2, 0))  # -> (n_y, n_x, n_patterns), matches convention below
example_recording_path = "/mnt/data/Igor analyses/OSDS/251103 OSDS/2_0_SWN_200_White.recording.h5"
obj = pygor.load.STRF.load_object(example_recording_path)
print(f"noise_array.shape = {noise_array.shape}  (n_y, n_x, n_patterns) -- TRUE base resolution")
print(f"obj.strfs.shape = {obj.strfs.shape}  (n_roi, n_lags, nx, ny)")


# %% ---------------------------------------------------------------------------
# Machinery copied verbatim from v1/v2 (dev/snippet_analysis.py,
# dev/snippet_analysis_v2.py) -- unchanged unless noted. Not imported (v1's
# module has top-level executing demo code; v2's does too) -- copied instead,
# same convention both files already use.
# ============================================================================


def strf_window_from_obj(obj):
    """Derive the STA lag window from the object. Copied verbatim from v1/v2."""
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


def _suppress_refractory(event_frames, min_gap):
    """Greedy min-gap de-dup: keep the first crossing, drop any subsequent one
    within `min_gap` frames of the last KEPT one (not the last raw crossing).

    Real calcium-trace noise is autocorrelated (the indicator's own rise/decay
    smears raw shot noise across adjacent frames), unlike the i.i.d. Gaussian
    noise stc_null_eigenvalues adds to its synthetic linear neuron -- so a
    naive threshold crossing lets ONE real transient register as several
    adjacent "events" whose STC snippets overlap almost completely (the fit
    lag window spans ~1.2s, many frames), inflating real covariance relative
    to a null built from genuinely-independent synthetic events. min_gap=0
    (default) reproduces the old unsuppressed behaviour exactly.
    """
    if min_gap <= 0 or len(event_frames) < 2:
        return event_frames
    kept = [event_frames[0]]
    for f in event_frames[1:]:
        if f - kept[-1] >= min_gap:
            kept.append(f)
    return np.asarray(kept, dtype=event_frames.dtype)


def detect_events(
    obj, roi, threshold=1.75, sign="pos", trigger_start=0, n_f_relevant=None,
    use_znorm=True, min_gap=0, traces=None,
):
    """Z-scored temporal derivative crossings. Adapted from v1/v2 (added
    min_gap refractory suppression, see _suppress_refractory; added `traces`
    override, see module note on GCaMP-nonlinearity below).

    traces : optional (n_roi, n_frames) array overriding obj.traces_znorm/raw
    -- pass obj.traces_deconvolved here. GCaMP's own saturating dose-response
    is a static output nonlinearity on top of the true (possibly linear)
    signal; thresholding its derivative selects exactly the high-activity
    epochs where that saturation is worst, which the 2P calcium-imaging
    literature (Nauhaus, Nielsen & Callaway 2012, J Neurophysiol -- indicator
    saturation distorts tuning measurements, esp. at high activity) shows can
    manufacture spurious second-order structure indistinguishable from real
    synaptic rectification. Deconvolving first (pygor.strf.deconvolution,
    already used for obj.strfs elsewhere in this codebase, just never wired
    into this file) removes that confound before any covariance analysis.
    """
    if traces is None:
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
    event_frames = _suppress_refractory(event_frames, min_gap)
    return event_frames, dif[event_frames]


def _crop_box(px, py, crop, n_x, n_y):
    """Clipped [x0,x1,y0,y1] box of half-width `crop`. Copied verbatim from v1/v2."""
    if crop is None:
        return 0, n_x, 0, n_y
    return (max(0, px - crop), min(n_x, px + crop + 1),
            max(0, py - crop), min(n_y, py + crop + 1))


def extract_snippets_spatial(
    noise_array, frame_to_pattern, event_frames,
    n_f_filter_past, n_f_filter, center=None, crop=6, baseline=0.5, max_gib=6,
    taus_override=None,
):
    """Per-event spatiotemporal stimulus history, cropped around the RF.
    Adapted from v1 -- see v1 for the full docstring/rationale.

    taus_override : optional explicit tau array, e.g. taus[lag_mask]. When STC
    only needs a causal SUBSET of lags (usually true -- the fit lag_mask is
    almost always narrower than the full n_f_filter range), passing the
    already-restricted taus here avoids ever allocating the FULL n_f_filter
    lags just to discard most of them afterward -- the difference between
    tractable and not once crop=None (full FOV).
    """
    n_y, n_x = noise_array.shape[0], noise_array.shape[1]
    px, py = (n_x // 2, n_y // 2) if center is None else center
    x0, x1, y0, y1 = _crop_box(px, py, crop, n_x, n_y)
    nx_c, ny_c = x1 - x0, y1 - y0

    taus = np.arange(n_f_filter) + (1 - n_f_filter_past) if taus_override is None else np.asarray(taus_override)
    n_lags = len(taus)
    n_rel = len(frame_to_pattern)
    n_ev = len(event_frames)

    nbytes = n_ev * n_lags * nx_c * ny_c * 4
    if nbytes > max_gib * 2**30:
        raise MemoryError(
            f"spatial snippets would be {nbytes / 2**30:.1f} GiB "
            f"({n_ev} events x {n_lags} lags x {nx_c}x{ny_c}); "
            "raise the event threshold, reduce crop, or pass taus_override to "
            "a narrower lag set."
        )

    crop_vals = np.transpose(
        noise_array[y0:y1, x0:x1, :].astype(np.float32), (1, 0, 2)
    )  # (nx_c, ny_c, n_pat)

    snips = np.full((n_ev, n_lags, nx_c, ny_c), baseline, dtype=np.float32)
    for e, f in enumerate(event_frames):
        frames = f + taus
        vi = np.flatnonzero((frames >= 0) & (frames < n_rel))
        pats = frame_to_pattern[frames[vi]]
        has = pats >= 0
        snips[e, vi[has]] = np.transpose(crop_vals[:, :, pats[has]], (2, 0, 1))
    snips -= baseline
    return snips, taus, (x0, x1, y0, y1)


def _mask_edges(K, edge_crop):
    """Zero a border on a (..., nx, ny) kernel. Copied verbatim from v1/v2."""
    if edge_crop > 0:
        K[..., :edge_crop, :] = 0.0; K[..., -edge_crop:, :] = 0.0
        K[..., :, :edge_crop] = 0.0; K[..., :, -edge_crop:] = 0.0
    return K


def _peak_var_ij(K, edge_crop=2):
    """Pixel of max temporal variance, edge-masked. Copied verbatim from v1/v2."""
    v = _mask_edges(np.asarray(K, float).copy(), edge_crop).var(axis=-3)
    while v.ndim > 2:
        v = v.mean(0)
    return tuple(int(x) for x in np.unravel_index(v.argmax(), v.shape))


def _filmstrip_lags(energy_profile, taus, n_show):
    """Column lags for a filmstrip: n_show lags spread around the energy peak.
    Copied verbatim from v1/v2. Kept for other callers; plot_stc_filters uses
    _lags_leading_to_zero instead (see there for why)."""
    pk = int(energy_profile.argmax())
    return np.unique(np.clip(
        np.linspace(pk - 3, pk + 3, n_show).round().astype(int), 0, len(taus) - 1))


def _lags_spanning_zero(taus, n_show):
    """n_show lag indices evenly spread across the FULL available lag window
    -- both the causal build-up before zero AND what happens after it -- not
    whichever narrow window happens to contain the peak-energy lag (which is
    what _filmstrip_lags picks, and can land almost entirely on one side of
    zero depending on where a given eigenfilter's energy concentrates). The
    zero lag itself is always included as one of the shown columns.
    """
    zero_idx = int(np.argmin(np.abs(taus)))
    idxs = np.linspace(0, len(taus) - 1, n_show).round().astype(int)
    if zero_idx not in idxs:
        idxs[np.argmin(np.abs(idxs - zero_idx))] = zero_idx
    return np.unique(idxs)


def calcium_kernel(frame_duration, tau_decay_s=0.150, tau_rise_s=0.01):
    """Causal GCaMP-like impulse response. Copied verbatim from v1/v2."""
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
    """Synthetic calcium trace of a purely linear neuron. Copied verbatim from v1/v2."""
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


def find_anchor_peak(
    noise_array, frame_to_pattern, event_frames, n_f_filter_past, n_f_filter,
    edge_crop=2, peak_method="var", max_gib=6,
):
    """Data-found peak pixel from a plain full-field event-triggered STA --
    the same anchor-finding step as v2's method_anchor_split, factored out on
    its own since STC doesn't need the rest of that method's up/down split
    machinery, just a place to centre the prior/fit crop.
    """
    spatial_snips, taus, _ = extract_snippets_spatial(
        noise_array, frame_to_pattern, event_frames,
        n_f_filter_past, n_f_filter, center=None, crop=None, max_gib=max_gib,
    )
    sta = _mask_edges(spatial_snips.mean(axis=0), edge_crop)
    if peak_method == "var":
        px, py = _peak_var_ij(sta, edge_crop=edge_crop)
    elif peak_method == "amp":
        proj = np.abs(sta).max(axis=0)
        px, py = (int(v) for v in np.unravel_index(proj.argmax(), proj.shape))
    else:
        raise ValueError("peak_method must be 'var' or 'amp'")
    return px, py, sta, taus


# %% ---------------------------------------------------------------------------
# STC core -- copied verbatim from v1 (dev/snippet_analysis.py's STC section).
# See v1 for the original derivation notes; key correctness constraint
# (unchanged): lag_mask MUST stay tight and n_sample MUST stay >> D
# (=n_lags_kept*nx_c*ny_c), else the prior PCA basis overestimates its own
# top-variance directions and every projected ensemble reads out a spurious,
# near-uniform negative eigenvalue shift -- not a null result, an artifact.
# ============================================================================


def time_crop_lags(taus, frame_duration, lo_s=-1.0, hi_s=0.2):
    """Boolean lag mask restricting to the causal RF window [lo_s, hi_s] (s)."""
    lag_s = taus * frame_duration
    return (lag_s >= lo_s) & (lag_s <= hi_s)


def _flatten_snips(spatial_snips, lag_mask):
    """(n, n_lags, nx, ny) -> (n, n_lags_c*nx*ny) over kept lags; returns (P, shape)."""
    sub = spatial_snips[:, lag_mask]
    n, nl, nx, ny = sub.shape
    return sub.reshape(n, nl * nx * ny).astype(np.float64), (nl, nx, ny)


def stimulus_prior_basis(
    noise_array, frame_to_pattern, center, crop, taus, lag_mask,
    n_f_filter_past, n_f_filter, n_pca=120, n_sample=10000, seed=0, max_gib=6,
):
    """Truncated-PCA whitening basis of the stimulus prior (cropped+time-cropped).
    Sampled over random relevant frames (NOT events) -- see module note above
    on why n_sample must stay >> D.

    Extracts ONLY the lag_mask-kept lags directly (taus_override) rather than
    the full n_f_filter range then discarding most of it -- the difference
    between tractable and not once crop=None (full FOV).
    """
    from sklearn.decomposition import PCA
    rng = np.random.default_rng(seed)
    n_rel = len(frame_to_pattern)
    lo = max(1, int(-taus.min()) + 1)
    hi = n_rel - int(taus.max()) - 1
    frames = rng.integers(lo, hi, size=min(n_sample, max(1, hi - lo)))
    taus_kept = taus[lag_mask]
    snips, _, _ = extract_snippets_spatial(
        noise_array, frame_to_pattern, frames, n_f_filter_past, n_f_filter,
        center=center, crop=crop, taus_override=taus_kept, max_gib=max_gib,
    )
    P, shape = _flatten_snips(snips, np.ones(len(taus_kept), dtype=bool))
    m = min(n_pca, P.shape[1], P.shape[0] - 1)
    pca = PCA(n_components=m, svd_solver="randomized", random_state=seed).fit(P)
    return {"mean": pca.mean_, "components": pca.components_,
            "variance": pca.explained_variance_, "shape": shape}


def compute_stc(snips_flat, prior_basis, weights=None, n_keep=6, eps=1e-6):
    """STC eigen-decomposition in the prior-whitened PCA subspace. Project
    snippets onto the prior PCA basis, whiten by prior variances, eigen-
    decompose (event covariance - I), un-whiten the kept eigenvectors back to
    stimulus space. weights=None -> event-triggered; array -> continuous
    rectified-trace weighted."""
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
        (U.T @ (Vz[:, i] / vs)).reshape(shape) for i in keep
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
    use_znorm=True, stride=2, seed=0, max_gib=6, traces=None,
):
    """Continuous response-weighted STC (2nd-order Wiener): all frames weighted
    by the (rectified) calcium trace, instead of thresholded events.

    traces : optional (n_roi, n_frames) override, same rationale as
    detect_events's `traces` param -- pass obj.traces_deconvolved to avoid
    the GCaMP-saturation confound (applies here too, arguably more so: this
    uses every frame, not just high-activity threshold crossings).
    """
    n_rel = len(frame_to_pattern)
    lo = max(1, int(-taus.min()) + 1)
    hi = n_rel - int(taus.max()) - 1
    frames = np.arange(lo, hi, stride)
    if traces is None:
        traces = obj.traces_znorm if use_znorm else obj.traces_raw
    trace = traces[roi]
    weights = np.asarray(trace[trigger_start + frames], float)
    taus_kept = taus[lag_mask]
    snips, _, _ = extract_snippets_spatial(
        noise_array, frame_to_pattern, frames, n_f_filter_past, n_f_filter,
        center=center, crop=crop, taus_override=taus_kept, max_gib=max_gib,
    )
    P, _ = _flatten_snips(snips, np.ones(len(taus_kept), dtype=bool))
    return compute_stc(P, prior_basis, weights=weights, n_keep=n_keep)


def stc_null_eigenvalues(
    filter_txy, noise_array, frame_to_pattern, taus, lag_mask,
    n_f_filter_past, n_f_filter, center, crop, prior_basis,
    threshold=2.0, noise_sigma=1.5, calcium_tau_decay_s=0.08, n_boot=25,
    n_keep=6, seed=0, max_gib=6, min_gap=0, deconv_kernel=None, deconv_lambd=3e-3,
):
    """Null STC eigenvalue band: event-triggered STC on LINEAR-null ensembles.
    A linear cell has NO covariance features beyond the prior, so its
    eigenvalues define the band a real eigenvalue must exceed to count as
    genuine. Deterministic linear drive computed ONCE (noise_sigma=0); only
    measurement noise re-drawn per bootstrap. filter_txy must be FULL-lag
    (simulate_linear_trace needs the whole filter to simulate correctly);
    only the per-bootstrap event extraction is lag-restricted.

    min_gap : passed through to the SAME refractory suppression detect_events
    applies (see _suppress_refractory) -- this loop reimplements the z-score/
    threshold logic inline rather than calling detect_events, so the
    suppression has to be duplicated here explicitly, or real (suppressed)
    and null (unsuppressed) event ensembles would no longer be comparable.

    deconv_kernel : optional, from pygor.strf.deconvolution.calcium_kernel --
    if given, each bootstrap's noisy synthetic trace is Wiener-deconvolved
    with this SAME kernel before thresholding, mirroring whatever the real
    detect_events() call did (see its docstring re: GCaMP saturation). Must
    match the real pipeline's kernel or the two ensembles aren't comparable.
    """
    det = simulate_linear_trace(
        filter_txy, noise_array, frame_to_pattern, taus,
        noise_sigma=0.0, calcium_tau_decay_s=calcium_tau_decay_s, seed=0,
    )
    rng = np.random.default_rng(seed)
    taus_kept = taus[lag_mask]
    kept_mask = np.ones(len(taus_kept), dtype=bool)
    tops, alls = [], []
    for _ in range(n_boot):
        trace = det + noise_sigma * rng.standard_normal(len(det))
        if deconv_kernel is not None:
            from pygor.strf.deconvolution import wiener_deconvolve
            trace = wiener_deconvolve(trace, deconv_kernel, lambd=deconv_lambd)
        dif = np.diff(trace, prepend=trace[0])
        base = dif[: min(100, len(dif))]
        dif = (dif - base.mean()) / (base.std() or 1.0)
        ev = np.flatnonzero(dif > threshold)
        ev = _suppress_refractory(ev, min_gap)
        if len(ev) < 20:
            continue
        snips, _, _ = extract_snippets_spatial(
            noise_array, frame_to_pattern, ev, n_f_filter_past, n_f_filter,
            center=center, crop=crop, taus_override=taus_kept, max_gib=max_gib,
        )
        res = stc_event(snips, kept_mask, prior_basis, n_keep=n_keep)
        alls.append(res["eigvals"]); tops.append(res["keep_eigvals"])
    return np.array(tops), np.array(alls)


# %% ---------------------------------------------------------------------------
# NEW: same-location vs distinct-location diagnostic. Ties STC's output back
# to the motivating question ("does the cell respond to both ON and OFF of
# roughly the SAME spot, or to two genuinely different spots") -- a top
# eigenfilter concentrated right at the anchor pixel supports the former
# (same-location rectification, exactly what method_neighbourhood_energy in
# v2 targets with a hand-picked crop); an eigenfilter peaking somewhere else
# in the fit crop supports a genuinely distinct second subfield.
# ============================================================================


def eigenfilter_peak_offset(stc_res, anchor_px, anchor_py, box, edge_crop=2):
    """For each kept STC eigenfilter, its own peak-variance pixel (in FULL-FOV
    coords) and its Euclidean offset from the anchor pixel.

    box = (x0,x1,y0,y1), the crop the filters live in (local coords -> add
    x0,y0 to get full-FOV coords, matching anchor_px/anchor_py's frame).
    """
    x0, _, y0, _ = box
    offsets = []
    for i, F in enumerate(stc_res["filters"]):
        lx, ly = _peak_var_ij(F, edge_crop=edge_crop)
        fx, fy = lx + x0, ly + y0
        dist = float(np.hypot(fx - anchor_px, fy - anchor_py))
        offsets.append({"eig_rank": i, "eigval": float(stc_res["keep_eigvals"][i]),
                        "peak_px": fx, "peak_py": fy, "offset_px": dist})
    return offsets


# %% ---------------------------------------------------------------------------
# Plots -- copied verbatim from v1.
# ============================================================================


def plot_stc_filters(stc_res, taus_c, frame_duration, n_show=7, edge_crop=2):
    """Filmstrip of the top STC eigen-filters (rows=eigenvectors, RdBu_r).

    Columns span the FULL available lag window -- the causal build-up before
    zero AND the response after it -- not centred on the peak-energy lag
    (with a biphasic/rectifying filter the peak can sit right near zero,
    which used to crop the filmstrip down to mostly one side and skip the
    rest). Zero itself is always one of the shown columns.

    edge_crop defaults to 2, not 1: pixels near the fit crop's OWN boundary
    (not just the true sensor edge) are less well-represented by the
    truncated n_pca-component prior basis, so their whitening divisor is
    underestimated and they come out over-amplified -- a masking-1-ring-only
    crop lets that sit one ring further in, visible as a spurious streak that
    (unlike real structure) recurs identically across otherwise-independent
    eigenvectors. Confirmed NOT a stimulus artifact (raw per-pixel contrast
    variance checked directly -- uniform ~0.25 everywhere in the fit crop).
    """
    F = _mask_edges(np.array(stc_res["filters"], float), edge_crop)
    lam = stc_res["keep_eigvals"]
    lag_s = taus_c * frame_duration
    idxs = _lags_spanning_zero(taus_c, n_show)
    vm = float(np.max(np.abs(F))) or 1.0
    nf = F.shape[0]
    fig, axs = plt.subplots(nf, len(idxs), figsize=(1.7 * len(idxs), 2.0 * nf), squeeze=False)
    for r in range(nf):
        for c, ti in enumerate(idxs):
            ax = axs[r][c]
            ax.imshow(F[r, ti], origin="lower", cmap="RdBu_r", vmin=-vm, vmax=vm)
            ax.set_xticks([]); ax.set_yticks([])
            if r == 0:
                ax.set_title(f"{lag_s[ti] * 1000:.0f} ms", fontsize=8)
            if c == 0:
                ax.set_ylabel(f"e{r}\nlambda={lam[r]:.2f}", fontsize=8)
    fig.suptitle("STC eigen-filters (top by |eigenvalue|)", y=1.0)
    fig.tight_layout()
    return fig


def play_stc_filters(stc_res, taus_c, frame_duration, edge_crop=2, dur_s=None, roi=None):
    """Synchronised movie of the top STC eigen-filters, one panel per
    eigenvector, all sharing ONE FuncAnimation time axis across the FULL
    lag window (both before and after zero) -- the STC analogue of v1's
    play_components / v2's play_methods_grid.

    Each panel gets its OWN colour scale (an eigenfilter's amplitude has no
    fixed relationship to another's -- they're different directions in
    stimulus space, not comparable quantities the way up/down kernels from
    the same method are).
    """
    import matplotlib.animation

    F = _mask_edges(np.array(stc_res["filters"], float), edge_crop)
    lam = stc_res["keep_eigvals"]
    lag_ms = taus_c * frame_duration * 1000
    n_frames = F.shape[1]
    if dur_s is None:
        dur_s = n_frames * frame_duration

    clims = [float(np.max(np.abs(F[i]))) or 1.0 for i in range(F.shape[0])]

    plt.rcParams["animation.html"] = "jshtml"
    nf = F.shape[0]
    fig, axs = plt.subplots(1, nf, figsize=(2.2 * nf, 2.4), squeeze=False)
    axs = axs[0]
    ims = []
    for i, ax in enumerate(axs):
        im = ax.imshow(F[i, 0], origin="lower", cmap="RdBu_r",
                       vmin=-clims[i], vmax=clims[i], interpolation="none")
        ax.set_title(f"e{i}  lambda={lam[i]:.2f}", fontsize=8)
        ax.set_xticks([]); ax.set_yticks([])
        ims.append(im)
    sup = fig.suptitle("", fontsize=9, y=1.02)
    fig.tight_layout()

    def video(frame):
        for im, i in zip(ims, range(nf)):
            im.set_array(F[i, frame])
        tag = f"ROI {roi}  |  " if roi is not None else ""
        sup.set_text(f"{tag}lag = {lag_ms[frame]:+.0f} ms   (0 = event)")
        return ims

    anim = matplotlib.animation.FuncAnimation(
        fig, video, frames=n_frames, interval=dur_s / n_frames * 1000, repeat_delay=500,
    )
    plt.close(fig)
    return anim


def plot_stc_timecourses(stc_res, taus_c, frame_duration, peak_ij=None, ax=None):
    """Peak-pixel timecourse of each top STC eigen-filter."""
    F = np.array(stc_res["filters"], float)
    lam = stc_res["keep_eigvals"]
    lag_s = taus_c * frame_duration
    if peak_ij is None:
        peak_ij = _peak_var_ij(F, edge_crop=2)
    i, j = peak_ij
    if ax is None:
        _, ax = plt.subplots(figsize=(6.5, 3.5))
    cols = plt.cm.coolwarm(np.linspace(0, 1, F.shape[0]))
    for r in range(F.shape[0]):
        ax.plot(lag_s, F[r][:, i, j], color=cols[r], lw=1.6, label=f"e{r} lambda={lam[r]:.2f}")
    ax.axvline(0, color="k", lw=0.7, ls="--"); ax.axhline(0, color="0.75", lw=0.5)
    ax.set_xlabel("lag (s)"); ax.set_ylabel("filter @ peak px")
    ax.set_title("STC eigen-filter timecourses"); ax.legend(fontsize=8)
    return ax


def plot_stc_spectrum(stc_res, null_tops=None, ax=None):
    """STC eigenvalue spectrum with the linear-null 95% band overlaid.
    Eigenvalues outside the band are candidate genuine (non-linear) features."""
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


# %% ---------------------------------------------------------------------------
# Demo: run end-to-end on the real recording.
# ============================================================================

frame_duration, n_f_filter, n_f_filter_past = strf_window_from_obj(obj)
frame_to_pattern, trigger_start, n_f_relevant = build_frame_to_pattern(obj, noise_array)
taus = np.arange(n_f_filter) + (1 - n_f_filter_past)
print(f"frame_duration={frame_duration:.4f}s  n_f_filter={n_f_filter}  "
      f"n_f_filter_past={n_f_filter_past}  n_f_relevant={n_f_relevant}")

# Deconvolve traces before event detection (see detect_events docstring):
# GCaMP's own saturating dose-response is a static output nonlinearity that
# thresholding-on-raw-derivative selects for HARDEST at exactly the
# high-activity epochs used as "events" -- Nauhaus/Nielsen/Callaway 2012
# show this can manufacture spurious apparent nonlinear structure out of a
# linear cell. pygor already ships Wiener deconvolution for this (used
# elsewhere for obj.strfs, never wired into this file before now).
from pygor.strf.deconvolution import calcium_kernel as pygor_calcium_kernel
obj.deconvolve_traces(verbose=True)
_frame_dt_ms = 1000.0 * obj.linedur_s * obj.images.shape[1]
_, deconv_kernel = pygor_calcium_kernel(_frame_dt_ms)

demo_roi = 32
thresh = .5
crop = None            # full FOV -- now genuinely feasible: at native 6x10
                      # resolution the whole FOV is only 60 px (vs 960 at
                      # the jittered 24x40 resolution), so there's no longer
                      # a dimensionality reason to restrict to an
                      # anchor-centred window. See the data-load cell for why
                      # (true stimulus resolution is 6x10, not 24x40).
edge_crop = 1          # 2 was calibrated for the old 24x40 grid; at 6x10,
                      # edge_crop=2 would mask 2 of only 6 rows on each
                      # side, leaving almost nothing in the middle.
n_pca = 150
n_sample = 80000     # MUST stay >> D=n_lags_kept*(2*crop+1)^2 -- see module note
max_gib = 10         # prior_basis's n_sample draw is the big one; bump the guard to match
min_gap = int(5 * 0.150 / frame_duration)   # refractory suppression, ~12 frames
                      # (~0.77s) -- real calcium-trace noise is autocorrelated,
                      # unlike the null's i.i.d. Gaussian noise, so a single
                      # real transient crosses threshold on several adjacent
                      # frames whose STC snippets almost fully overlap,
                      # inflating real covariance vs a null built from
                      # genuinely-independent synthetic events. NOT arbitrary:
                      # matches 5*calcium_tau_decay_s (calcium_kernel's own
                      # support length, same 0.150s default simulate_linear_trace
                      # uses). Empirically swept on ROI 58 (clean-linear
                      # control) from 0 to 32 frames: 0/4/12/16/24/32 frames gave
                      # 6/4/1/0/1/1/0 of 6 eigenvalues spuriously exceeding the
                      # null band -- below ~8 frames the clean control fails
                      # robustly and systematically (not noise); at 12+ frames
                      # it's consistent with the ~2.5%-per-eigenvalue chance
                      # rate expected under a true null. Applied to BOTH real
                      # detect_events() calls AND stc_null_eigenvalues's own
                      # inline event detection, so the two stay comparable.

event_frames, _ = detect_events(
    obj, demo_roi, threshold=thresh, sign="pos", trigger_start=trigger_start,
    n_f_relevant=n_f_relevant, min_gap=min_gap, traces=obj.traces_deconvolved,
)
print(f"n_events (pos) = {len(event_frames)}")

anchor_px, anchor_py, anchor_sta, _ = find_anchor_peak(
    noise_array, frame_to_pattern, event_frames, n_f_filter_past, n_f_filter, edge_crop=edge_crop,
)
anchor_center = (anchor_px, anchor_py)
print(f"anchor peak (i_x, i_y) = ({anchor_px}, {anchor_py})  -- the fit below is "
      f"cropped and centred on this point")

lag_mask = time_crop_lags(taus, frame_duration, lo_s=-1.0, hi_s=0.2)
taus_c = taus[lag_mask]
n_lags_kept = len(taus_c)

# Dimensionality check BEFORE running anything -- this is the "n_sample must
# stay >> D" constraint from the module note, now measured against the
# ACTUAL cropped fit box (not the full FOV) so the printed ratio is the one
# that governs the eigenvalue bias. If this ratio looks thin, shrink crop,
# widen n_sample, or narrow lo_s/hi_s above rather than trusting a
# borderline fit.
x0, x1, y0, y1 = _crop_box(anchor_px, anchor_py, crop, noise_array.shape[1], noise_array.shape[0])
nx_c, ny_c = x1 - x0, y1 - y0
D = n_lags_kept * nx_c * ny_c
print(f"fit dimensionality D = n_lags_kept*nx_c*ny_c = {n_lags_kept}*{nx_c}*{ny_c} "
      f"= {D}  (n_sample={n_sample}, ratio={n_sample / D:.1f}x)")

prior_basis = stimulus_prior_basis(
    noise_array, frame_to_pattern, anchor_center, crop, taus, lag_mask,
    n_f_filter_past, n_f_filter, n_pca=n_pca, n_sample=n_sample, max_gib=max_gib,
)
event_snips, _, box = extract_snippets_spatial(
    noise_array, frame_to_pattern, event_frames, n_f_filter_past, n_f_filter,
    center=anchor_center, crop=crop, taus_override=taus_c, max_gib=max_gib,
)
kept_mask = np.ones(n_lags_kept, dtype=bool)

stc_evt = stc_event(event_snips, kept_mask, prior_basis, n_keep=6)
stc_cont = stc_continuous(
    obj, demo_roi, noise_array, frame_to_pattern, trigger_start, taus, lag_mask,
    anchor_center, crop, n_f_filter_past, n_f_filter, prior_basis, n_keep=6, stride=2, max_gib=max_gib,
    traces=obj.traces_deconvolved,
)

null_tops, null_all = stc_null_eigenvalues(
    anchor_sta, noise_array, frame_to_pattern, taus, lag_mask,
    n_f_filter_past, n_f_filter, anchor_center, crop, prior_basis,
    threshold=thresh, noise_sigma=1.5, n_boot=20, max_gib=max_gib, min_gap=min_gap,
    deconv_kernel=deconv_kernel,
)
null_band = float(np.percentile(np.abs(null_tops), 97.5))
above = np.abs(stc_evt["keep_eigvals"]) > null_band
print(f"event STC top6 eig: {np.round(stc_evt['keep_eigvals'], 2)}")
print(f"cont  STC top6 eig: {np.round(stc_cont['keep_eigvals'], 2)}")
print(f"null two-sided 97.5% band = +/-{null_band:.2f}  ->  "
      f"{int(above.sum())}/{len(above)} kept eigenvalues exceed it")
print(f"null median(all eigvals) = {np.median(null_all):.3f}  (want ~0 -- if this drifts "
      f"negative, the prior-oversample ratio above is still too thin)")

# %% ---------------------------------------------------------------------------
# Synthetic-linear-neuron validation of compute_stc's un-whiten fix (line 381:
# was `vs * Vz[:, i]`, now `Vz[:, i] / vs` -- multiply vs divide inverted the
# un-whitening, amplifying low-prior-variance directions instead of the
# informative ones). A KNOWN-linear neuron (anchor_sta as ground-truth filter,
# driven through the identical anchor-crop/thresh/n_pca/n_sample config as the
# real ROI above) must show eigenvalues inside the null band and structureless
# (RdBu noise, no blob) eigenfilters -- anything else means the fix didn't
# resolve it, or issues 2-4 (deferred: event refractory gap, null-model-for-
# flat-STA caveat) also need addressing.
# ============================================================================
sim_trace = simulate_linear_trace(
    anchor_sta, noise_array, frame_to_pattern, taus, frame_duration=frame_duration,
    baseline=0.5, noise_sigma=1.5, seed=1,
)
sim_trace_padded = np.zeros(trigger_start + n_f_relevant)
sim_trace_padded[trigger_start:trigger_start + len(sim_trace)] = sim_trace
from pygor.strf.deconvolution import wiener_deconvolve
sim_trace_deconv = wiener_deconvolve(sim_trace_padded, deconv_kernel)

class _SimObj:
    pass

sim_ev, _ = detect_events(_SimObj(), 0, threshold=thresh, sign="pos",
                           trigger_start=trigger_start, n_f_relevant=n_f_relevant,
                           min_gap=min_gap, traces=np.asarray([sim_trace_deconv]))
sim_snips, _, _ = extract_snippets_spatial(
    noise_array, frame_to_pattern, sim_ev, n_f_filter_past, n_f_filter,
    center=anchor_center, crop=crop, taus_override=taus_c, max_gib=max_gib,
)
sim_stc = stc_event(sim_snips, kept_mask, prior_basis, n_keep=6)
sim_above = np.abs(sim_stc["keep_eigvals"]) > null_band
print(f"[SYNTHETIC LINEAR] n_events={len(sim_ev)}  top6 eig: {np.round(sim_stc['keep_eigvals'], 2)}")
print(f"[SYNTHETIC LINEAR] median(all eigvals) = {np.median(sim_stc['eigvals']):.3f}  (want ~0)")
print(f"[SYNTHETIC LINEAR] {int(sim_above.sum())}/{len(sim_above)} kept eigenvalues exceed "
      f"null band +/-{null_band:.2f}  (want 0/6 -- any hits = fix incomplete)")
plot_stc_filters(sim_stc, taus_c, frame_duration, edge_crop=2)
plt.show()

offsets = eigenfilter_peak_offset(stc_evt, anchor_px, anchor_py, box, edge_crop=2)
print("eigenfilter peak offsets from anchor (px):")
for o in offsets:
    flag = "  <-- EXCEEDS NULL BAND" if abs(o["eigval"]) > null_band else ""
    print(f"  e{o['eig_rank']} lambda={o['eigval']:+.2f}  peak=({o['peak_px']},{o['peak_py']})  "
          f"offset={o['offset_px']:.2f}px{flag}")
print("small offset -> same-location rectification (what method 5 in v2 targets);")
print("large offset -> genuinely distinct second subfield, not just one rectified spot.")

plot_stc_spectrum(stc_evt, null_tops=null_tops)
plt.show()
plot_stc_filters(stc_evt, taus_c, frame_duration, edge_crop=2)
plt.show()
plot_stc_timecourses(stc_evt, taus_c, frame_duration)
plt.show()

# %% STC eigen-filter movie -- all kept eigenvectors, synced on one lag axis
vid_stc = play_stc_filters(stc_evt, taus_c, frame_duration, edge_crop=2, roi=demo_roi)
vid_stc

# %% Spot-check v1's flagged "nonlinear candidate" ROIs, plus a clean-linear
# negative control (see the STRF-quality scan below for how that ROI was
# picked). Each ROI gets its OWN anchor pixel, so -- unlike the shared
# full-FOV prior_basis the old crop=None version could get away with --
# the prior basis has to be refit per ROI, centred on that ROI's own anchor.
# This is now cheap (D=5491 at crop=8, not 18240) so it's not a real cost.
spot_rois = [44, 8, 20, 26, 28, 38, 61, 58, 32, 27]  # 58 added as a clean-linear negative
# control -- highest edge-masked peak/median temporal-variance ratio (222)
# of any un-flagged ROI, single compact monophasic blob in obj.strfs (visually
# confirmed), picked precisely because it's the kind of ROI STC should come
# back near-null on if the pipeline is trustworthy.
spot_results = {}
for spot_roi in spot_rois:
    ev, _ = detect_events(
        obj, spot_roi, threshold=thresh, sign="pos", trigger_start=trigger_start,
        n_f_relevant=n_f_relevant, min_gap=min_gap, traces=obj.traces_deconvolved,
    )
    px, py, sta, _ = find_anchor_peak(
        noise_array, frame_to_pattern, ev, n_f_filter_past, n_f_filter, edge_crop=edge_crop,
    )
    spot_center = (px, py)
    spot_prior = stimulus_prior_basis(
        noise_array, frame_to_pattern, spot_center, crop, taus, lag_mask,
        n_f_filter_past, n_f_filter, n_pca=n_pca, n_sample=n_sample, max_gib=max_gib,
    )
    snips, _, spot_box = extract_snippets_spatial(
        noise_array, frame_to_pattern, ev, n_f_filter_past, n_f_filter,
        center=spot_center, crop=crop, taus_override=taus_c, max_gib=max_gib,
    )
    res = stc_event(snips, kept_mask, spot_prior, n_keep=6)
    ntop, nall = stc_null_eigenvalues(
        sta, noise_array, frame_to_pattern, taus, lag_mask, n_f_filter_past, n_f_filter,
        spot_center, crop, spot_prior, threshold=thresh, noise_sigma=1.5, n_boot=20,
        max_gib=max_gib, min_gap=min_gap, deconv_kernel=deconv_kernel,
    )
    band = float(np.percentile(np.abs(ntop), 97.5))
    offs = eigenfilter_peak_offset(res, px, py, spot_box, edge_crop=2)
    print(f"\n=== ROI {spot_roi} ===  anchor=({px},{py})  n_events={len(ev)}  "
          f"null_band=+/-{band:.2f}  null_median={np.median(nall):.3f}")
    print(f"  top6 eig: {np.round(res['keep_eigvals'], 2)}")
    for o in offs:
        flag = "  <-- EXCEEDS NULL BAND" if abs(o["eigval"]) > band else ""
        print(f"  e{o['eig_rank']} lambda={o['eigval']:+.2f}  offset={o['offset_px']:.2f}px{flag}")

    # Rough eyeball classification -- 1.5px is a starting threshold (~one RF
    # diameter at 9deg pitch), not derived; sanity-check against
    # plot_stc_filters per ROI before trusting the auto-label, same as
    # null_band is a review aid, not a hard cutoff.
    sig = [o for o in offs if abs(o["eigval"]) > band]
    if not sig:
        verdict = "NO RECOVERABLE STRUCTURE (all eigvals within null band)"
    elif max(o["offset_px"] for o in sig) < 1.5:
        verdict = "SAME-LOCATION RECTIFICATION (small offset)"
    else:
        verdict = "DISTINCT SUBFIELD / GABOR-LIKE (large offset)"
    print(f"  verdict: {verdict}")
    spot_results[spot_roi] = {"res": res, "band": band, "verdict": verdict}

    fig = plot_stc_filters(res, taus_c, frame_duration, edge_crop=2)
    fig.suptitle(f"ROI {spot_roi}: {verdict}", y=1.02)
    plt.show()

# %%
