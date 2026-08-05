# %%
import numpy as np
import matplotlib.pyplot as plt

%load_ext autoreload
%autoreload 2

# %% ---------------------------------------------------------------------------
# stc_modelling: synthetic ground-truth testbed for the STC pipeline validated
# in dev/stc_analysis.py. That pipeline's real-data results (all ROIs tested,
# including the strongest "candidate" by raw eigenvalue) failed split-half
# reproducibility -- consistent with "no reproducible 2nd-order structure at
# current event counts" but leaves one question real data alone can't answer:
# does this pipeline actually have the STATISTICAL POWER to recover a KNOWN
# nonlinear feature at realistic sample sizes at all? Real data can't answer
# that (we don't know the true answer to compare against). This file builds
# neuron models with an ANALYTICALLY KNOWN STC outcome, generates synthetic
# data from them at both an idealized (spike-count) and a realistic
# (GCaMP-convolved, noisy, deconvolved) fidelity, and runs the SAME estimator
# (copied from stc_analysis.py, not reinvented) against ground truth.
#
# Two questions this answers that dev/stc_analysis.py's real-data run cannot:
#   1. At the real recording's ~2500-17000 effective sample count, would a
#      real effect of plausible size even be recoverable by this estimator?
#      (If no: today's null result is a power problem, not evidence of
#      absence.) If yes at much lower N: today's null result is more likely
#      genuine absence of structure in that data.
#   2. Does the realistic-2P fidelity tier (indicator convolution + noise +
#      deconvolution + event detection) lose recoverable information relative
#      to the idealized spike-count tier? Isolates "is GCaMP/2P imaging itself
#      the bottleneck" from "is the raw estimator underpowered".
#
# Design choice: SYNTHETIC stimulus generated fresh per-frame (genuinely i.i.d.
# binary noise, no sample-and-hold/jitter structure) -- deliberately simpler
# than dev/stc_analysis.py's real "shifting noise" stimulus, to isolate the
# estimator's own power from that stimulus's extra temporal-correlation
# complexity (already handled correctly there; not re-tested here).
# ============================================================================

# %% ---------------------------------------------------------------------------
# Core STC machinery -- copied verbatim/trimmed from dev/stc_analysis.py
# (same convention that file's own header uses: not imported, since real-data
# files there have top-level executing demo code; copied here too even though
# this file has none, for consistency and because these are the ALREADY-
# VALIDATED pieces under test, not things to re-derive).
# ============================================================================


def _crop_box(px, py, crop, n_x, n_y):
    if crop is None:
        return 0, n_x, 0, n_y
    return (max(0, px - crop), min(n_x, px + crop + 1),
            max(0, py - crop), min(n_y, py + crop + 1))


def _mask_edges(K, edge_crop):
    if edge_crop > 0:
        K[..., :edge_crop, :] = 0.0; K[..., -edge_crop:, :] = 0.0
        K[..., :, :edge_crop] = 0.0; K[..., :, -edge_crop:] = 0.0
    return K


def extract_snippets_spatial(
    noise_array, event_frames, n_f_filter_past, n_f_filter,
    center=None, crop=None, baseline=0.5, taus_override=None,
):
    """Trimmed from stc_analysis.py: no frame_to_pattern indirection (this
    file's synthetic stimulus is generated 1:1 with frame index -- no
    trigger-driven sample-and-hold, no jitter -- so pattern index == frame
    index directly), no max_gib guard (synthetic runs are small by
    construction)."""
    n_y, n_x = noise_array.shape[0], noise_array.shape[1]
    n_rel = noise_array.shape[2]
    px, py = (n_x // 2, n_y // 2) if center is None else center
    x0, x1, y0, y1 = _crop_box(px, py, crop, n_x, n_y)
    nx_c, ny_c = x1 - x0, y1 - y0

    taus = np.arange(n_f_filter) + (1 - n_f_filter_past) if taus_override is None else np.asarray(taus_override)
    n_lags = len(taus)
    n_ev = len(event_frames)

    crop_vals = np.transpose(noise_array[y0:y1, x0:x1, :].astype(np.float32), (1, 0, 2))

    snips = np.full((n_ev, n_lags, nx_c, ny_c), baseline, dtype=np.float32)
    for e, f in enumerate(event_frames):
        frames = f + taus
        vi = np.flatnonzero((frames >= 0) & (frames < n_rel))
        snips[e, vi] = np.transpose(crop_vals[:, :, frames[vi]], (2, 0, 1))
    snips -= baseline
    return snips, taus, (x0, x1, y0, y1)


def _flatten_snips(spatial_snips, lag_mask):
    sub = spatial_snips[:, lag_mask]
    n, nl, nx, ny = sub.shape
    return sub.reshape(n, nl * nx * ny).astype(np.float64), (nl, nx, ny)


def stimulus_prior_basis(noise_array, center, crop, taus, lag_mask, n_f_filter_past,
                          n_f_filter, n_pca=150, n_sample=80000, seed=0):
    from sklearn.decomposition import PCA
    rng = np.random.default_rng(seed)
    n_rel = noise_array.shape[2]
    lo = max(1, int(-taus.min()) + 1)
    hi = n_rel - int(taus.max()) - 1
    frames = rng.integers(lo, hi, size=min(n_sample, max(1, hi - lo)))
    taus_kept = taus[lag_mask]
    snips, _, _ = extract_snippets_spatial(noise_array, frames, n_f_filter_past, n_f_filter,
                                           center=center, crop=crop, taus_override=taus_kept)
    P, shape = _flatten_snips(snips, np.ones(len(taus_kept), dtype=bool))
    m = min(n_pca, P.shape[1], P.shape[0] - 1)
    pca = PCA(n_components=m, svd_solver="randomized", random_state=seed).fit(P)
    return {"mean": pca.mean_, "components": pca.components_,
            "variance": pca.explained_variance_, "shape": shape}


def compute_stc(snips_flat, prior_basis, weights=None, n_keep=6, eps=1e-6):
    """Identical to stc_analysis.py's compute_stc (post un-whiten-sign-fix)."""
    U = np.asarray(prior_basis["components"], float)
    var = np.asarray(prior_basis["variance"], float)
    mean = np.asarray(prior_basis["mean"], float)
    shape = prior_basis["shape"]
    X = np.asarray(snips_flat, float)
    n = len(X)
    w = np.ones(n) if weights is None else np.clip(np.asarray(weights, float), 0, None)
    wsum = w.sum() or 1.0
    vs = np.sqrt(var + eps * var.max())

    sta = (w[:, None] * X).sum(0) / wsum
    Z = ((X - mean) @ U.T) / vs
    zbar = (w[:, None] * Z).sum(0) / wsum
    Zc = Z - zbar
    C = (Zc.T * w) @ Zc / wsum
    dC = 0.5 * (C + C.T) - np.eye(len(var))
    lam, Vz = np.linalg.eigh(dC)
    order = np.argsort(lam)[::-1]
    lam, Vz = lam[order], Vz[:, order]
    keep = np.argsort(np.abs(lam))[::-1][:n_keep]
    keep = keep[np.argsort(lam[keep])[::-1]]
    filters = np.stack([(U.T @ (Vz[:, i] / vs)).reshape(shape) for i in keep])
    return {"eigvals": lam, "keep_idx": keep, "keep_eigvals": lam[keep],
            "filters": filters, "sta": sta.reshape(shape)}


def stc_event(spatial_snips, lag_mask, prior_basis, n_keep=6):
    P, _ = _flatten_snips(spatial_snips, lag_mask)
    return compute_stc(P, prior_basis, weights=None, n_keep=n_keep)


def time_crop_lags(taus, frame_duration, lo_s=-1.0, hi_s=0.2):
    lag_s = taus * frame_duration
    return (lag_s >= lo_s) & (lag_s <= hi_s)


# %% ---------------------------------------------------------------------------
# Synthetic stimulus + neuron models
# ============================================================================


def make_noise_stimulus(n_frames, ny=6, nx=10, seed=0):
    """i.i.d. binary noise, matching the real {0,1}-value convention
    (baseline=0.5 subtracted downstream, same as stc_analysis.py)."""
    rng = np.random.default_rng(seed)
    return rng.integers(0, 2, size=(ny, nx, n_frames)).astype(np.uint8)


def make_filter(n_lags, ny, nx, center, sigma=1.1, peak_lag=6, width=2.5, biphasic=0.35):
    """Localized spatial Gaussian x biphasic temporal kernel -- a plausible,
    clean ground-truth RF, nothing fancy."""
    yy, xx = np.mgrid[0:ny, 0:nx]
    cx, cy = center
    spatial = np.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * sigma ** 2))
    spatial /= np.linalg.norm(spatial)
    t = np.arange(n_lags)
    temporal = np.exp(-((t - peak_lag) ** 2) / (2 * width ** 2))
    temporal -= biphasic * np.exp(-((t - peak_lag - 7) ** 2) / (2 * (width * 1.4) ** 2))
    temporal /= np.abs(temporal).max()
    return temporal[:, None, None] * spatial[None, :, :]  # (n_lags, ny, nx)


def project_filter(noise_array, filt, taus, baseline=0.5):
    """Same tensordot-over-lags convention as stc_analysis.py's
    simulate_linear_trace, but 1:1 frame<->pattern (no frame_to_pattern)."""
    ny, nx, n_frames = noise_array.shape
    stim = np.transpose(noise_array.astype(np.float32), (2, 0, 1)) - baseline  # (n_frames, ny, nx)
    n_lags = filt.shape[0]
    r = np.zeros(n_frames, dtype=np.float64)
    for j, tau in enumerate(taus):
        proj = np.tensordot(stim, filt[j], axes=([1, 2], [0, 1]))
        if tau > 0:
            r[: n_frames - tau] += proj[tau:]
        elif tau < 0:
            r[-tau:] += proj[: n_frames + tau]
        else:
            r += proj
    return r


class LinearModel:
    """Negative control: purely linear, no 2nd-order structure by
    construction. STC on this (correctly implemented) should recover nothing
    -- any eigenvalue exceeding the null band here is a pipeline false
    positive, full stop."""
    def __init__(self, filt):
        self.filt = filt
        self.true_filters = []  # nothing for STC to find

    def response(self, noise_array, taus):
        return project_filter(noise_array, self.filt, taus)


class StaticNonlinearityModel:
    """Negative control #2: linear filter through a MONOTONIC static output
    nonlinearity (sigmoid). Tests the specific worry from dev/stc_analysis.py
    (GCaMP's own saturating dose-response) in isolation: for a white/Gaussian
    stimulus, a monotonic output nonlinearity should NOT create new STC
    structure (Bussgang-type argument) -- if this model shows a "significant"
    eigenvalue, that's evidence the pipeline itself manufactures false
    positives from output nonlinearity alone, independent of real 2P data."""
    def __init__(self, filt, gain=3.0):
        self.filt = filt
        self.gain = gain
        self.true_filters = []

    def response(self, noise_array, taus):
        lin = project_filter(noise_array, self.filt, taus)
        lin = (lin - lin.mean()) / (lin.std() or 1.0)
        return 1.0 / (1.0 + np.exp(-self.gain * lin))  # sigmoid, monotonic


class EnergyModel:
    """Classic STC textbook case (Rust et al. 2005 / Schwartz et al. 2006):
    response = (f1.stim)^2 + (f2.stim)^2. STC should recover f1, f2 as two
    excitatory eigenvectors above the null -- the standard sanity check that
    STC CAN find real structure when it's genuinely there."""
    def __init__(self, filt1, filt2):
        self.filt1, self.filt2 = filt1, filt2
        self.true_filters = [filt1, filt2]

    def response(self, noise_array, taus):
        p1 = project_filter(noise_array, self.filt1, taus)
        p2 = project_filter(noise_array, self.filt2, taus)
        return p1 ** 2 + p2 ** 2


class SubunitModel:
    """Two spatially DISTINCT rectified subunits: response =
    ReLU(f1.stim) + ReLU(f2.stim), f1/f2 centred at different locations.
    Direct synthetic stand-in for the motivating real question (ROI 44:
    spatially-distinct ON+OFF, Gabor-like rather than centre-surround) --
    ground truth is f1 and f2, known exactly, unlike any real ROI."""
    def __init__(self, filt1, filt2):
        self.filt1, self.filt2 = filt1, filt2
        self.true_filters = [filt1, filt2]

    def response(self, noise_array, taus):
        p1 = project_filter(noise_array, self.filt1, taus)
        p2 = project_filter(noise_array, self.filt2, taus)
        return np.clip(p1, 0, None) + np.clip(p2, 0, None)


# %% ---------------------------------------------------------------------------
# Two fidelity tiers: idealized spike counts, and realistic 2P (GCaMP
# convolution + noise + Wiener deconvolution + derivative-threshold events).
# ============================================================================


def idealized_spike_events(response, base_rate_hz=0.2, gain_hz=1.0, frame_dt_s=0.064, seed=0):
    """Poisson-thinned spike events straight from the model response, no
    calcium simulation at all -- isolates the ESTIMATOR's power from any
    imaging/indicator confound."""
    rng = np.random.default_rng(seed)
    r = response.copy()
    r = (r - r.mean()) / (r.std() or 1.0)
    rate_hz = np.clip(base_rate_hz + gain_hz * r, 0, None)
    p_spike = 1 - np.exp(-rate_hz * frame_dt_s)
    spikes = rng.random(len(r)) < p_spike
    return np.flatnonzero(spikes)


def realistic_2p_events(response, base_rate_hz=0.2, gain_hz=1.0, frame_dt_s=0.064,
                         noise_sigma=1.5, threshold=0.5, min_gap=12, seed=0,
                         return_intermediate=False):
    """Same underlying spike-generating process, but convolved through a real
    GCaMP kernel, noised, then run through the SAME deconvolve -> z-score
    derivative -> threshold -> refractory-suppress pipeline dev/stc_analysis.py
    uses on real data -- isolates whether the imaging/detection LAYER loses
    recoverable information relative to the idealized tier above.

    return_intermediate : if True, return a dict with spike_train, calcium,
    noisy, deconv, ev instead of just ev -- for visualization only."""
    from pygor.strf.deconvolution import calcium_kernel, wiener_deconvolve
    rng = np.random.default_rng(seed)
    spike_frames = idealized_spike_events(response, base_rate_hz, gain_hz, frame_dt_s, seed=seed)
    spike_train = np.zeros(len(response))
    spike_train[spike_frames] = 1.0
    frame_dt_ms = frame_dt_s * 1000.0
    _, kernel = calcium_kernel(frame_dt_ms)
    calcium = np.convolve(spike_train, kernel)[: len(spike_train)]
    noisy = calcium + noise_sigma * rng.standard_normal(len(calcium)) * (calcium.std() or 1.0) * 0.3
    deconv = wiener_deconvolve(noisy, kernel)
    dif = np.diff(deconv, prepend=deconv[0])
    base = dif[: min(200, len(dif))]
    dif = (dif - base.mean()) / (base.std() or 1.0)
    ev = np.flatnonzero(dif > threshold)
    if min_gap > 0 and len(ev) > 1:
        kept = [ev[0]]
        for f in ev[1:]:
            if f - kept[-1] >= min_gap:
                kept.append(f)
        ev = np.asarray(kept)
    if return_intermediate:
        return {"spike_train": spike_train, "calcium": calcium, "noisy": noisy,
                "deconv": deconv, "ev": ev}
    return ev


# %% ---------------------------------------------------------------------------
# Validation: ground-truth recovery + split-half reliability, at a given N.
# ============================================================================


def eval_recovery(noise_array, event_frames, prior, taus, lag_mask, taus_c, n_f_filter_past,
                   n_f_filter, true_filters, crop, n_keep=6, n_splithalf_reps=3, seed=0):
    """Returns dict: n_events, keep_eigvals, best_gt_corr (top eigenfilter's
    best |correlation| against any true filter -- 0 if model has none),
    splithalf_mean (mean |correlation| of top eigenfilter across random
    halves)."""
    rng = np.random.default_rng(seed)
    kept_mask = np.ones(len(taus_c), dtype=bool)
    snips, _, box = extract_snippets_spatial(noise_array, event_frames, n_f_filter_past, n_f_filter,
                                             center=None, crop=crop, taus_override=taus_c)
    res = stc_event(snips, kept_mask, prior, n_keep=n_keep)

    gt_corr = 0.0
    if true_filters:
        top = res["filters"][0]
        lag_mask_arr = np.asarray(lag_mask)
        for gt in true_filters:
            # extract_snippets_spatial (copied from stc_analysis.py) returns
            # spatial axes as (nx, ny) -- but make_filter/project_filter build
            # and use filters in (ny, nx) order (matching noise_array's own
            # (ny, nx, n_patterns) INPUT layout, which project_filter's
            # tensordot needs). Transpose here, at the ground-truth COMPARISON
            # point only -- response generation itself doesn't need this and
            # was verified correct via a single-pixel/single-lag alignment
            # test. Confirmed empirically: this exact bug was why the first
            # version of this file showed near-zero recovery even at 60k+
            # events on the textbook-easy Energy model (STA peak landed at
            # the exact right lag/pixel; whole-array correlation was ~0
            # because the two arrays were transposed relative to each other).
            gt_c = np.transpose(gt[lag_mask_arr], (0, 2, 1))
            c = float(np.corrcoef(top.ravel(), gt_c.ravel())[0, 1])
            gt_corr = max(gt_corr, abs(c))

    corrs = []
    if len(event_frames) >= 20:
        for rep in range(n_splithalf_reps):
            perm = rng.permutation(len(event_frames))
            half = len(event_frames) // 2
            if half < 10:
                break
            ev_a, ev_b = event_frames[perm[:half]], event_frames[perm[half:2 * half]]
            halves = []
            for ev_h in (ev_a, ev_b):
                s, _, _ = extract_snippets_spatial(noise_array, ev_h, n_f_filter_past, n_f_filter,
                                                    center=None, crop=crop, taus_override=taus_c)
                halves.append(stc_event(s, kept_mask, prior, n_keep=n_keep))
            fa, fb = halves[0]["filters"][0].ravel(), halves[1]["filters"][0].ravel()
            corrs.append(abs(float(np.corrcoef(fa, fb)[0, 1])))
    splithalf_mean = float(np.mean(corrs)) if corrs else float("nan")

    return {"n_events": len(event_frames), "keep_eigvals": res["keep_eigvals"],
            "best_gt_corr": gt_corr, "splithalf_mean": splithalf_mean, "res": res}


# %% ---------------------------------------------------------------------------
# Visualization -- true filters, response/spike generation, simulated 2P
# trace through deconvolution, and the STC outcome with ground truth overlaid.
# ============================================================================


def plot_true_filters(filters, labels, taus, frame_dt_s, n_show=6):
    """Filmstrip of one or more TRUE filters (each (n_lags, ny, nx), the
    NATIVE make_filter/project_filter orientation -- not the transposed
    (nx,ny) convention extract_snippets_spatial's output uses)."""
    lag_s = taus * frame_dt_s
    idxs = np.unique(np.linspace(0, len(taus) - 1, n_show).round().astype(int))
    vm = max(np.abs(f).max() for f in filters) or 1.0
    fig, axs = plt.subplots(len(filters), len(idxs), figsize=(1.6 * len(idxs), 1.8 * len(filters)),
                            squeeze=False)
    for r, (filt, label) in enumerate(zip(filters, labels)):
        for c, ti in enumerate(idxs):
            ax = axs[r][c]
            ax.imshow(filt[ti], origin="lower", cmap="RdBu_r", vmin=-vm, vmax=vm)
            ax.set_xticks([]); ax.set_yticks([])
            if r == 0:
                ax.set_title(f"{lag_s[ti] * 1000:.0f} ms", fontsize=8)
            if c == 0:
                ax.set_ylabel(label, fontsize=9)
    fig.suptitle("True filter(s) -- ground truth", y=1.02)
    fig.tight_layout()
    return fig


def plot_response_and_events(response, event_frames, frame_dt_s, window=(0, 3000), title=""):
    """Model response over a representative window, with detected event
    times marked -- shows what's actually driving/being selected as "spikes"."""
    lo, hi = window
    t = np.arange(lo, hi) * frame_dt_s
    resp_win = response[lo:hi]
    ev_win = event_frames[(event_frames >= lo) & (event_frames < hi)]
    fig, ax = plt.subplots(figsize=(9, 2.5))
    ax.plot(t, resp_win, lw=0.7, color="0.3")
    ymax = resp_win.max() if len(resp_win) else 1.0
    ax.scatter(ev_win * frame_dt_s, np.full(len(ev_win), ymax * 1.08), marker="|",
              color="crimson", s=60, label=f"events (n={len(ev_win)} shown)")
    ax.set_xlabel("time (s)"); ax.set_ylabel("model response")
    ax.set_title(title); ax.legend(fontsize=8, loc="upper right")
    fig.tight_layout()
    return fig


def plot_2p_trace(intermediate, frame_dt_s, window=(0, 3000), title=""):
    """Spike train -> noiseless calcium -> noisy calcium -> deconvolved
    trace with detected events, for a representative window -- the full
    "realistic 2P tier" generative chain, visualized end to end."""
    lo, hi = window
    t = np.arange(lo, hi) * frame_dt_s
    spike_train, calcium = intermediate["spike_train"], intermediate["calcium"]
    noisy, deconv, ev = intermediate["noisy"], intermediate["deconv"], intermediate["ev"]
    fig, axs = plt.subplots(4, 1, figsize=(9, 7), sharex=True)
    spike_t = np.flatnonzero(spike_train[lo:hi]) * frame_dt_s + lo * frame_dt_s
    axs[0].vlines(spike_t, 0, 1, color="k", lw=0.8)
    axs[0].set_ylabel("true spikes"); axs[0].set_yticks([])
    axs[1].plot(t, calcium[lo:hi], color="tab:green", lw=0.8)
    axs[1].set_ylabel("calcium\n(noiseless)")
    axs[2].plot(t, noisy[lo:hi], color="0.4", lw=0.6)
    axs[2].set_ylabel("calcium\n(+ noise)")
    axs[3].plot(t, deconv[lo:hi], color="tab:blue", lw=0.8)
    ev_win = ev[(ev >= lo) & (ev < hi)]
    axs[3].scatter(ev_win * frame_dt_s, deconv[ev_win], color="crimson", zorder=5, s=18,
                   label=f"detected events (n={len(ev_win)} shown)")
    axs[3].set_ylabel("deconvolved"); axs[3].set_xlabel("time (s)")
    axs[3].legend(fontsize=8, loc="upper right")
    fig.suptitle(title, y=1.0)
    fig.tight_layout()
    return fig


def plot_stc_outcome(res, taus_c, frame_dt_s, true_filters_cropped=None, true_labels=None,
                     n_show=6, edge_crop=0, title=""):
    """Filmstrip: TRUE filter(s) in the top row(s) (already lag-mask-cropped
    AND transposed to the (nx,ny) convention -- see eval_recovery's comment),
    recovered top eigenfilters below, same lag columns -- direct visual
    ground-truth-vs-recovered comparison."""
    lag_s = taus_c * frame_dt_s
    idxs = np.unique(np.linspace(0, len(taus_c) - 1, n_show).round().astype(int))
    F = _mask_edges(np.array(res["filters"], float), edge_crop)
    lam = res["keep_eigvals"]
    rows = []
    if true_filters_cropped:
        for gt, label in zip(true_filters_cropped, true_labels):
            rows.append((f"TRUE {label}", gt))
    for i in range(F.shape[0]):
        rows.append((f"e{i}\nlambda={lam[i]:+.2f}", F[i]))
    vm = max(np.abs(arr).max() for _, arr in rows) or 1.0
    fig, axs = plt.subplots(len(rows), len(idxs), figsize=(1.6 * len(idxs), 1.6 * len(rows)),
                            squeeze=False)
    for r, (label, arr) in enumerate(rows):
        for c, ti in enumerate(idxs):
            ax = axs[r][c]
            ax.imshow(arr[ti], origin="lower", cmap="RdBu_r", vmin=-vm, vmax=vm)
            ax.set_xticks([]); ax.set_yticks([])
            if r == 0:
                ax.set_title(f"{lag_s[ti] * 1000:.0f} ms", fontsize=8)
            if c == 0:
                ax.set_ylabel(label, fontsize=7)
    fig.suptitle(title, y=1.01)
    fig.tight_layout()
    return fig


def plot_spectrum_compare(results_by_label):
    """Sorted eigenvalue spectra overlaid for several models/conditions --
    shows whether one has a clear outlier (real structure) vs a flat/uniform
    spread (null-consistent)."""
    fig, ax = plt.subplots(figsize=(6.5, 3.5))
    for label, res in results_by_label.items():
        lam = np.sort(res["eigvals"])[::-1]
        ax.plot(np.arange(len(lam)), lam, "o-", ms=2, lw=0.8, label=label)
    ax.axhline(0, color="0.7", lw=0.5)
    ax.set_xlabel("eigenvalue rank"); ax.set_ylabel("whitened variance")
    ax.set_title("STC spectrum comparison"); ax.legend(fontsize=8)
    fig.tight_layout()
    return fig


def plot_tier_comparison(true_filters_cropped, true_labels, res_spike, res_2p, n_eigs=4,
                         edge_crop=0, title=""):
    """Same underlying model, two columns (idealized spike tier vs realistic
    2P tier), rows = TRUE filter(s) then top eigenvectors -- one representative
    lag per cell (the lag where TRUE filter 1 peaks, shared across the whole
    grid so every cell is directly comparable at the same time point).
    Answers "what did each tier's eigen-solution actually converge to" side
    by side, not just as summary numbers."""
    tf0 = true_filters_cropped[0]
    peak_lag = int(np.argmax(np.abs(tf0).reshape(tf0.shape[0], -1).max(axis=1)))

    F_spike = _mask_edges(np.array(res_spike["filters"], float), edge_crop)
    F_2p = _mask_edges(np.array(res_2p["filters"], float), edge_crop)
    lam_spike, lam_2p = res_spike["keep_eigvals"], res_2p["keep_eigvals"]

    n_rows = len(true_filters_cropped) + n_eigs
    vm = max(np.abs(tf).max() for tf in true_filters_cropped)
    vm = max(vm, np.abs(F_spike[:n_eigs]).max(), np.abs(F_2p[:n_eigs]).max()) or 1.0

    fig, axs = plt.subplots(n_rows, 2, figsize=(4.4, 1.7 * n_rows), squeeze=False)
    for i, (tf, label) in enumerate(zip(true_filters_cropped, true_labels)):
        for c in range(2):
            axs[i][c].imshow(tf[peak_lag], origin="lower", cmap="RdBu_r", vmin=-vm, vmax=vm)
            axs[i][c].set_xticks([]); axs[i][c].set_yticks([])
        axs[i][0].set_ylabel(f"TRUE {label}", fontsize=8)
    for j in range(n_eigs):
        r = len(true_filters_cropped) + j
        axs[r][0].imshow(F_spike[j][peak_lag], origin="lower", cmap="RdBu_r", vmin=-vm, vmax=vm)
        axs[r][1].imshow(F_2p[j][peak_lag], origin="lower", cmap="RdBu_r", vmin=-vm, vmax=vm)
        axs[r][0].set_ylabel(f"e{j}\nlambda={lam_spike[j]:+.2f}", fontsize=7)
        axs[r][1].set_ylabel(f"lambda={lam_2p[j]:+.2f}", fontsize=7)
        for c in range(2):
            axs[r][c].set_xticks([]); axs[r][c].set_yticks([])
    axs[0][0].set_title("Idealized spike tier", fontsize=9)
    axs[0][1].set_title("Realistic 2P tier", fontsize=9)
    fig.suptitle(title, y=1.0)
    fig.tight_layout()
    return fig


# %% ---------------------------------------------------------------------------
# Demo: sanity check at generous N, then sweep N to find the recovery
# threshold, for the model most relevant to the motivating question
# (SubunitModel: two spatially distinct rectified subfields).
# ============================================================================

ny, nx = 6, 10
n_f_filter, n_f_filter_past = 25, 12
frame_dt_s = 0.064
taus = np.arange(n_f_filter) + (1 - n_f_filter_past)
lag_mask = time_crop_lags(taus, frame_dt_s, lo_s=-1.0, hi_s=0.2)
taus_c = taus[lag_mask]
crop = None  # full 6x10 FOV -- same reasoning as stc_analysis.py's post-fix config

f1 = make_filter(n_f_filter, ny, nx, center=(2, 2))
f2 = make_filter(n_f_filter, ny, nx, center=(7, 3))
subunit_model = SubunitModel(f1, f2)
linear_model = LinearModel(f1)
static_nl_model = StaticNonlinearityModel(f1)
energy_model = EnergyModel(f1, f2)

# %% Generate one big stimulus/response pool (2M frames), used throughout by
# truncating -- NOT by cranking gain per target N. gain/base_rate are FIXED at
# a sparse, realistic rate (~3%, close to real data's ~7%); cranking gain to
# hit a large target_n on a short recording instead saturates the spike
# process (most frames spike regardless of stimulus), which destroys the very
# stimulus-locked timing STC depends on -- confirmed empirically: the first,
# uncalibrated version of this file did exactly that (rates up to 36%) and
# every model, including the textbook-positive-control Energy model, showed
# near-zero ground-truth recovery even at 55k-85k "events".
BASE_RATE_HZ, GAIN_HZ = 0.2, 1.0   # ~3% event rate at these settings (checked directly)
n_frames_big = 2_000_000
noise_big = make_noise_stimulus(n_frames_big, ny, nx, seed=1)

# n_pca near-FULL RANK (D = n_lags_kept*ny*nx = 900 here), not the n_pca=150
# copied uncritically from stc_analysis.py's real-data config. That default
# made sense there: the real "shifting noise" stimulus has near-degenerate
# jittered structure, so more PCA components mean amplifying near-zero-
# variance noise directions (confirmed: increasing n_pca made REAL recovery
# WORSE). This synthetic stimulus is genuinely i.i.d. white -- no such
# degeneracy -- and truncating it anyway was confirmed EMPIRICALLY to
# suppress real, reproducible structure: Energy model gt_corr went
# 0.39->0.78 and Subunit gt_corr went 0.06->0.68 (both split-half-confirmed
# real, not noise) purely from raising n_pca 150->900. Don't copy this
# n_pca=900 default back to stc_analysis.py -- confirmed separately (full
# 150->1140 sweep on real ROI 32/58) that real data does NOT show the same
# improvement; the two stimuli have genuinely different covariance structure.
N_PCA = 900
D = int(lag_mask.sum()) * ny * nx
print(f"D={D}  N_PCA={N_PCA} ({100*N_PCA/D:.0f}% of full rank)")
prior_big = stimulus_prior_basis(noise_big, None, crop, taus, lag_mask, n_f_filter_past, n_f_filter,
                                  n_pca=N_PCA, n_sample=60000, seed=1)

# %% Basic sanity gate BEFORE trusting anything at the STC level: does a
# plain STA recover the LinearModel's own filter? If this fails, nothing
# downstream can be trusted either -- check this first, always.
resp_lin = linear_model.response(noise_big, taus)
ev_lin = idealized_spike_events(resp_lin, BASE_RATE_HZ, GAIN_HZ, frame_dt_s, seed=2)
snips_lin, _, _ = extract_snippets_spatial(noise_big, ev_lin, n_f_filter_past, n_f_filter,
                                           center=None, crop=crop, taus_override=taus_c)
sta_lin = snips_lin.mean(0)
f1_c = np.transpose(f1[lag_mask], (0, 2, 1))  # (nx,ny) vs (ny,nx) -- see eval_recovery's comment
sta_corr = float(np.corrcoef(sta_lin.ravel(), f1_c.ravel())[0, 1])
print(f"[STA SANITY GATE] n_events={len(ev_lin)}  STA-vs-true-filter corr={sta_corr:.3f}  "
      f"(must be high, e.g. >0.6-0.7, before trusting anything below)")

# %% Sanity check: all 4 models at generous N (~60k events), idealized tier.
sanity_results = {}
sanity_events = {}
for name, model in [("Linear (neg ctrl)", linear_model),
                    ("StaticNonlin (neg ctrl)", static_nl_model),
                    ("Energy (pos ctrl)", energy_model),
                    ("Subunit (motivating case)", subunit_model)]:
    resp = model.response(noise_big, taus)
    ev = idealized_spike_events(resp, BASE_RATE_HZ, GAIN_HZ, frame_dt_s, seed=2)
    rate_pct = 100 * len(ev) / len(resp)
    out = eval_recovery(noise_big, ev, prior_big, taus, lag_mask, taus_c, n_f_filter_past, n_f_filter,
                        model.true_filters, crop, seed=3)
    sanity_results[name] = out
    sanity_events[name] = (resp, ev)
    print(f"[SANITY, idealized, N={out['n_events']} ({rate_pct:.1f}% rate)] {name}: "
          f"top6={np.round(out['keep_eigvals'], 2)}  "
          f"best_gt_corr={out['best_gt_corr']:.2f}  splithalf={out['splithalf_mean']:.2f}")

# %% VISUALIZATION 1: the true filters (ground truth f1, f2) used by every
# model above -- Energy and Subunit both use these same two, just combine
# them differently (squared-sum vs rectified-sum).
fig = plot_true_filters([f1, f2], ["f1 (center 2,2)", "f2 (center 7,3)"], taus, frame_dt_s)
plt.show()

# %% VISUALIZATION 2: model response + detected events, for a representative
# ~3s window -- Energy (works) vs Subunit (doesn't), side by side in spirit.
resp_energy, ev_energy = sanity_events["Energy (pos ctrl)"]
resp_subunit, ev_subunit = sanity_events["Subunit (motivating case)"]
fig = plot_response_and_events(resp_energy, ev_energy, frame_dt_s, window=(0, 3000),
                               title="Energy model: response + detected events")
plt.show()
fig = plot_response_and_events(resp_subunit, ev_subunit, frame_dt_s, window=(0, 3000),
                               title="Subunit model: response + detected events")
plt.show()

# %% VISUALIZATION 3: the full realistic-2P generative chain (spikes ->
# calcium -> noisy -> deconvolved+detected), for the Subunit model, same
# representative window.
interm = realistic_2p_events(resp_subunit, BASE_RATE_HZ, GAIN_HZ, frame_dt_s=frame_dt_s, seed=6,
                             return_intermediate=True)
fig = plot_2p_trace(interm, frame_dt_s, window=(0, 3000),
                    title="Subunit model: simulated 2P generative chain")
plt.show()

# %% VISUALIZATION 4: STC outcome, ground truth vs recovered, for the
# success case (Energy) and the failure case (Subunit) -- the direct visual
# answer to "does recovery actually work".
lag_mask_arr = np.asarray(lag_mask)
f1_c = np.transpose(f1[lag_mask_arr], (0, 2, 1))
f2_c = np.transpose(f2[lag_mask_arr], (0, 2, 1))

fig = plot_stc_outcome(sanity_results["Energy (pos ctrl)"]["res"], taus_c, frame_dt_s,
                       true_filters_cropped=[f1_c, f2_c], true_labels=["f1", "f2"],
                       title=f"Energy model: gt_corr={sanity_results['Energy (pos ctrl)']['best_gt_corr']:.2f}  "
                             f"splithalf={sanity_results['Energy (pos ctrl)']['splithalf_mean']:.2f}")
plt.show()

fig = plot_stc_outcome(sanity_results["Subunit (motivating case)"]["res"], taus_c, frame_dt_s,
                       true_filters_cropped=[f1_c, f2_c], true_labels=["f1", "f2"],
                       title=f"Subunit model: gt_corr={sanity_results['Subunit (motivating case)']['best_gt_corr']:.2f}  "
                             f"splithalf={sanity_results['Subunit (motivating case)']['splithalf_mean']:.2f}")
plt.show()

# %% VISUALIZATION 5: eigenvalue spectra overlaid.
fig = plot_spectrum_compare({name: out["res"] for name, out in sanity_results.items()})
plt.show()

# %% VISUALIZATION 6: idealized spike tier vs realistic 2P tier, SAME model,
# side by side -- what each tier's eigen-solution actually converges to,
# at the largest N each tier can offer from this pool. This is the direct
# answer to "what's going on" that summary gt_corr/splithalf numbers don't
# show on their own: it's not that the 2P tier finds a WORSE version of the
# same structure -- watch whether it finds anything spatially coherent at all.
kept_mask_full = np.ones(len(taus_c), dtype=bool)

for model_name, model, resp in [("Energy", energy_model, resp_energy),
                                ("Subunit", subunit_model, resp_subunit)]:
    ev_spike_big = idealized_spike_events(resp, BASE_RATE_HZ, GAIN_HZ, frame_dt_s, seed=2)
    ev_2p_big = realistic_2p_events(resp, BASE_RATE_HZ, GAIN_HZ, frame_dt_s=frame_dt_s, seed=6)
    print(f"{model_name}: spike tier N={len(ev_spike_big)}   2P tier N={len(ev_2p_big)}")

    snips_spike, _, _ = extract_snippets_spatial(noise_big, ev_spike_big, n_f_filter_past, n_f_filter,
                                                 center=None, crop=crop, taus_override=taus_c)
    res_spike = stc_event(snips_spike, kept_mask_full, prior_big, n_keep=6)
    snips_2p, _, _ = extract_snippets_spatial(noise_big, ev_2p_big, n_f_filter_past, n_f_filter,
                                              center=None, crop=crop, taus_override=taus_c)
    res_2p = stc_event(snips_2p, kept_mask_full, prior_big, n_keep=6)

    fig = plot_tier_comparison([f1_c, f2_c], ["f1", "f2"], res_spike, res_2p, n_eigs=4,
                               title=f"{model_name} model: spike (N={len(ev_spike_big)}) vs "
                                     f"2P (N={len(ev_2p_big)})")
    plt.show()

# %% Sweep event count, both fidelity tiers, SubunitModel only (the
# motivating case) -- find the N at which ground-truth recovery becomes
# reliable (gt_corr high AND splithalf high together), at each tier. Fixed
# sparse gain throughout; N varies by how much of the (long) recording is
# used, matching how N would actually grow with real recording duration.
target_ns = [500, 1000, 2500, 5000, 10000, 20000, 50000]
resp_big = subunit_model.response(noise_big, taus)

print("\n--- Idealized spike-count tier ---")
ev_pool = idealized_spike_events(resp_big, BASE_RATE_HZ, GAIN_HZ, frame_dt_s, seed=4)
print(f"(pool: {len(ev_pool)} events available, {100*len(ev_pool)/len(resp_big):.1f}% rate)")
for target_n in target_ns:
    ev = ev_pool[:target_n]
    out = eval_recovery(noise_big, ev, prior_big, taus, lag_mask, taus_c, n_f_filter_past, n_f_filter,
                        subunit_model.true_filters, crop, seed=5)
    print(f"N~{target_n:6d} (actual {out['n_events']:6d}): "
          f"best_gt_corr={out['best_gt_corr']:.2f}  splithalf={out['splithalf_mean']:.2f}  "
          f"top6={np.round(out['keep_eigvals'], 2)}")

print("\n--- Realistic 2P tier (GCaMP conv + noise + deconv + threshold) ---")
ev_pool_2p = realistic_2p_events(resp_big, BASE_RATE_HZ, GAIN_HZ, frame_dt_s=frame_dt_s, seed=6)
print(f"(pool: {len(ev_pool_2p)} events available, {100*len(ev_pool_2p)/len(resp_big):.1f}% rate)")
for target_n in target_ns:
    ev = ev_pool_2p[:target_n]
    out = eval_recovery(noise_big, ev, prior_big, taus, lag_mask, taus_c, n_f_filter_past, n_f_filter,
                        subunit_model.true_filters, crop, seed=7)
    print(f"N~{target_n:6d} (actual {out['n_events']:6d}): "
          f"best_gt_corr={out['best_gt_corr']:.2f}  splithalf={out['splithalf_mean']:.2f}  "
          f"top6={np.round(out['keep_eigvals'], 2)}")

print("\nDONE")

# %%
