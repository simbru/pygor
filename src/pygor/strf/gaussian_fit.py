"""How Gaussian is a receptive field?

One number per STRF, ``index``: how far the RF departs from the best-fitting single
Gaussian, as a fraction of the RF's own signal. 0 means a Gaussian describes it to
within noise. Everything else here is the machinery for that number.

The pipeline, per STRF:

1. collapse over time                  -> signed 2D map (:meth:`STRF.collapse_times`)
2. rectify                             -> ``|data|``, the RF envelope
3. fit one elliptical Gaussian to it by least squares
4. measure over a footprint taken from the DATA's second moments, not from the fit --
   a fit that missed a lobe would otherwise exclude that lobe from its own exam
5. ``index`` = noise-corrected ``RMS(|data| - fit) / RMS(|data|)`` over that region

The fit is on ``|data|`` rather than on the signed map, which makes it polarity-blind
by construction: what is left in the residual is the polarity structure itself, i.e.
spatial nonlinearity, rather than a lobe some polarity-picking rule decided to ignore.

Array functions here take a single 2D map and know nothing about pygor;
:func:`gaussian_fit_index_wrapper` is the only thing that touches an STRF object.
"""

import numpy as np
from scipy.optimize import curve_fit
from scipy.stats import norm

# Size of the analysis ellipse, in sigmas of the footprint. 2 sigma looks generous but
# holds only ~76% of the pixels above 3 noise sigmas (91% of their mass), because real
# RFs have heavier shoulders than a Gaussian; 2.5 holds 95% (98% of mass) and leaves the
# ranking alone (spearman 0.993). 3 adds 4 points of coverage and dilutes the index with
# near-noise pixels.
N_SIGMA = 2.5
# Noise sigmas a pixel must clear to count as signal.
THRESHOLD_SD = 3.0
# Radius, in footprint sigmas, beyond which the residual is taken to be noise.
NOISE_SIGMA = 3.0
# Smallest the noise-calibration region may be, as a fraction of the valid map.
# On a map with no real RF the >k sigma pixels are scattered everywhere, so the
# data ellipse swallows most of it and the "far" annulus collapses to a thin rim
# at the edge -- 30-50 px out of 612 in the cases this was calibrated on. sd
# estimated there comes out 3-6x too small, which inflates BOTH chi2 and snr, so
# an SNR filter cannot catch it: amp and sd are each tiny and their ratio looks
# healthy. Such maps landed 8 of the top 25 of a pooled ranking. A fraction
# rather than a pixel count, so it holds for any map size. 0.25 sits in a clean
# empirical gap: the pathological maps measured 0.05-0.18, visually-confirmed
# real RFs 0.49-0.93, with one large-amplitude borderline at 0.37 that is kept.
MIN_FAR_FRAC = 0.25

# Every key :func:`fit_and_score` returns, and so every array the wrapper returns.
FIELDS = (
    "index",
    "chi2",
    "snr",
    "n_px",
    "amp",
    "x0",
    "y0",
    "sx",
    "sy",
    "theta",
    "offset",
    "footprint_x0",
    "footprint_y0",
    "footprint_sx",
    "footprint_sy",
    "footprint_theta",
)


def gaussian2d(coords, amp, x0, y0, sx, sy, theta, offset):
    """Elliptical Gaussian, theta counter-clockwise from +x with sx along it.

    The cross-term sign matters: with it flipped the blob rotates clockwise while
    :func:`ellipse_mask` and anything drawing the ellipse rotate counter-clockwise, so
    an oblique fit gets a mirrored analysis region. Axis-aligned fits hide it, which is
    why the test suite fits an oblique Gaussian specifically.
    """
    x, y = coords
    a = np.cos(theta) ** 2 / (2 * sx**2) + np.sin(theta) ** 2 / (2 * sy**2)
    b = np.sin(2 * theta) / (4 * sx**2) - np.sin(2 * theta) / (4 * sy**2)
    c = np.sin(theta) ** 2 / (2 * sx**2) + np.cos(theta) ** 2 / (2 * sy**2)
    dx, dy = x - x0, y - y0
    return (offset + amp * np.exp(-(a * dx**2 + 2 * b * dx * dy + c * dy**2))).ravel()


def _canonical(fit):
    """Put sx on the major axis so sigmas are comparable between ROIs.

    sx/sy swapped with theta rotated 90 deg is the same ellipse, and curve_fit lands on
    either version depending on the starting guess.
    """
    if fit["sy"] > fit["sx"]:
        fit["sx"], fit["sy"] = fit["sy"], fit["sx"]
        fit["theta"] = (fit["theta"] + np.pi) % np.pi - np.pi / 2
    return fit


def fit_gaussian2d(space):
    """Fit one elliptical Gaussian to the RF envelope, ``|space|``.

    Parameters
    ----------
    space : 2D array or np.ma.MaskedArray
        A time-collapsed spatial map. Masked and non-finite pixels are excluded from
        the fit rather than treated as zeros.

    Returns
    -------
    dict
        ``amp``, ``x0``, ``y0``, ``sx``, ``sy``, ``theta``, ``offset``, and ``fitted``,
        the model evaluated on the pixel grid.

    Raises
    ------
    RuntimeError, ValueError
        Straight from :func:`scipy.optimize.curve_fit` when it cannot converge.
    """
    space = np.ma.masked_invalid(space)
    h, w = space.shape
    yy, xx = np.mgrid[0:h, 0:w]
    target = np.ma.filled(np.abs(space), 0)

    total = target.sum()
    if total == 0:
        raise ValueError("map is empty, nothing to fit")
    x0 = (target * xx).sum() / total
    y0 = (target * yy).sum() / total
    sx = max(np.sqrt((target * (xx - x0) ** 2).sum() / total), 1)
    sy = max(np.sqrt((target * (yy - y0) ** 2).sum() / total), 1)

    valid = ~np.ma.getmaskarray(space)
    popt, _ = curve_fit(
        gaussian2d,
        np.vstack([xx[valid], yy[valid]]),
        target[valid].ravel(),
        p0=[target.max(), x0, y0, sx, sy, 0, 0],
        bounds=(
            [0, 0, 0, 0.5, 0.5, -np.pi / 2, -np.inf],
            [np.inf, w, h, w, h, np.pi / 2, np.inf],
        ),
        maxfev=10000,
    )
    fit = dict(zip(["amp", "x0", "y0", "sx", "sy", "theta", "offset"], popt))
    # Baseline kept in the map: abs() rectifies the noise, so the fit sits on a floor of
    # ~mean|noise|. Subtracting a model without it puts that floor into the residual.
    fit["fitted"] = gaussian2d((xx, yy), *popt).reshape(h, w)
    return _canonical(fit)


def data_ellipse(space, k=THRESHOLD_SD):
    """Ellipse from the data's own second moments over the >k sigma pixels.

    Independent of how well the Gaussian fitted, which is the point: when the fit lands
    on one lobe of two, the moments still span both. Measured SNR-independent (spearman
    -0.04) where the fitted sigma is not (-0.26).

    Returns None when fewer than four pixels clear the threshold.
    """
    space = np.ma.masked_invalid(space)
    flat = np.ma.filled(space, 0)
    a = np.abs(flat)
    sigma = 1.4826 * np.median(np.abs(flat - np.median(flat)))
    sig = (a > k * sigma) & ~np.ma.getmaskarray(space)
    if sig.sum() < 4:
        return None

    h, w = a.shape
    yy, xx = np.mgrid[0:h, 0:w]
    mass = a[sig].sum()
    cx = (a * xx)[sig].sum() / mass
    cy = (a * yy)[sig].sum() / mass
    vxx = (a * (xx - cx) ** 2)[sig].sum() / mass
    vyy = (a * (yy - cy) ** 2)[sig].sum() / mass
    vxy = (a * (xx - cx) * (yy - cy))[sig].sum() / mass
    evals, evecs = np.linalg.eigh([[vxx, vxy], [vxy, vyy]])
    major = evecs[:, int(np.argmax(evals))]
    # Wrapped into [-pi/2, pi/2) like the fit's theta: an ellipse axis has no direction,
    # and arctan2 returns (-pi, pi], outside the range the rest of this module assumes.
    theta = (np.arctan2(major[1], major[0]) + np.pi / 2) % np.pi - np.pi / 2
    return {
        "x0": float(cx),
        "y0": float(cy),
        "sx": float(np.sqrt(max(evals.max(), 0.25))),
        "sy": float(np.sqrt(max(evals.min(), 0.25))),
        "theta": float(theta),
    }


def footprint(space, fit, k=THRESHOLD_SD):
    """The ellipse the index is measured over: the data's, falling back to the fit."""
    return data_ellipse(space, k) or {
        key: fit[key] for key in ("x0", "y0", "sx", "sy", "theta")
    }


def ellipse_mask(space, fit, n_sigma=1):
    """Pixels inside the n_sigma ellipse of `fit`, plus the squared radii themselves."""
    h, w = space.shape
    yy, xx = np.mgrid[0:h, 0:w]
    dx, dy = xx - fit["x0"], yy - fit["y0"]
    u = dx * np.cos(fit["theta"]) + dy * np.sin(fit["theta"])
    v = -dx * np.sin(fit["theta"]) + dy * np.cos(fit["theta"])
    rad2 = (u / fit["sx"]) ** 2 + (v / fit["sy"]) ** 2
    return rad2 <= n_sigma**2, rad2


def _folded_var(mu, sd):
    """Var(|X|) for X ~ N(mu, sd^2).

    The residual is taken on ``|data|``, so what noise alone puts in it depends on how
    much signal is under that pixel: sd^2 where signal dominates, only (1-2/pi)sd^2
    where there is none. One closed form covers both ends, which is what lets a single
    metric span strong and weak parts of the same RF.
    """
    z = mu / sd
    e_abs = sd * np.sqrt(2 / np.pi) * np.exp(-(z**2) / 2) + mu * (1 - 2 * norm.cdf(-z))
    return mu**2 + sd**2 - e_abs**2


def gaussianity_index(
    space,
    fit,
    n_sigma=N_SIGMA,
    noise_sigma=NOISE_SIGMA,
    k=THRESHOLD_SD,
    min_far_frac=MIN_FAR_FRAC,
):
    """Departure from a single Gaussian, one number for every ROI. 0 = Gaussian.

    The region is the union of what the data shows and what the model claims: pixels
    above k noise sigmas, OR inside the n_sigma ellipse. Defining it from the fit alone
    is circular -- a fit that misses a lobe also excludes it from its own region, then
    scores well on what is left. With the union, a missed lobe is still measured (the
    model is near zero where the data is not) and a fit spread over empty pedestal is
    too (the model claims signal where the data is noise). Both failures raise the same
    number, no special cases.

    Returns
    -------
    dict
        index
            ``RMS(|data| - fit)`` with the expected noise variance removed, over
            ``RMS(|data|)``. Dimensionless, 0 when the Gaussian explains the RF to
            within noise, ~1 when it explains nothing. Rank on this.
        chi2
            The same misfit over what noise alone predicts. 1 is the null, so this says
            whether a departure is real rather than how big it is.
        snr
            Fitted amplitude over the noise. Neither of the above depends on it
            (spearman -0.07), so it is free to be used as a quality filter.
        n_px
            Pixels in the analysis region.
    """
    empty = {"n_px": 0, "index": np.nan, "chi2": np.nan, "snr": np.nan}
    space = np.ma.masked_invalid(space)
    valid = ~np.ma.getmaskarray(space)
    signed = np.ma.filled(space, 0)
    absdata = np.abs(signed)
    resid = absdata - fit["fitted"]

    region = footprint(space, fit, k)
    ellipse, rad2 = ellipse_mask(space, region, n_sigma)
    far = (rad2 > noise_sigma**2) & valid
    # Both floors: the fraction is what catches a footprint that has swallowed
    # the map, the absolute count guards tiny maps where 25% is still too few
    # pixels to estimate a variance from.
    if far.sum() < max(20, min_far_frac * valid.sum()):
        return empty

    # Noise calibrated on the residual itself, out where the fit predicts nothing. MAD
    # of the signed map overstates it about twofold on real data (0.486 vs 0.236), which
    # is enough to subtract every real departure away.
    sd = float(np.sqrt((resid[far] ** 2).mean() / (1 - 2 / np.pi)))
    mad = 1.4826 * np.median(np.abs(signed - np.median(signed)))
    omega = ((absdata > k * mad) | ellipse) & valid
    n = int(omega.sum())
    if n == 0 or sd == 0:
        return dict(empty, n_px=n)

    var_in = float((resid[omega] ** 2).mean())
    var_noise = float(_folded_var(fit["fitted"][omega], sd).mean())
    denom = float(np.sqrt((absdata[omega] ** 2).mean()))
    if var_noise == 0 or denom == 0:
        return dict(empty, n_px=n)
    return {
        "n_px": n,
        "index": float(np.sqrt(max(var_in - var_noise, 0)) / denom),
        "chi2": float(var_in / var_noise),
        "snr": float(fit["amp"] / sd),
    }


def fit_and_score(
    space,
    n_sigma=N_SIGMA,
    threshold_sd=THRESHOLD_SD,
    noise_sigma=NOISE_SIGMA,
    min_far_frac=MIN_FAR_FRAC,
):
    """Fit one Gaussian to a single collapsed map and score it.

    Returns a flat dict with every key in :data:`FIELDS`. An RF that cannot be fitted
    (empty map, no convergence, too little signal to calibrate noise against) comes back
    all-NaN with ``n_px`` 0 rather than raising, so one bad ROI cannot take down a whole
    recording.
    """
    blank = dict.fromkeys(FIELDS, np.nan)
    blank["n_px"] = 0
    try:
        fit = fit_gaussian2d(space)
    except (RuntimeError, ValueError):  # no convergence / empty map
        return blank

    scored = gaussianity_index(
        space, fit, n_sigma=n_sigma, noise_sigma=noise_sigma, k=threshold_sd,
        min_far_frac=min_far_frac,
    )
    region = footprint(space, fit, threshold_sd)
    out = dict(blank)
    out.update(scored)
    for key in ("amp", "x0", "y0", "sx", "sy", "theta", "offset"):
        out[key] = float(fit[key])
    for key in ("x0", "y0", "sx", "sy", "theta"):
        out[f"footprint_{key}"] = float(region[key])
    return out


def _resolve(strf_obj, n_sigma, threshold_sd, noise_sigma, min_far_frac=None):
    """Argument -> [strf.gaussian_fit] config -> module literal, per value.

    Mirrors the chain calculate_strf uses. Objects with no ``params`` (test stubs) fall
    straight through to the literals.
    """
    params = getattr(strf_obj, "params", None)
    defaults = {}
    if params is not None:
        try:
            defaults = params.get_defaults("strf").get("gaussian_fit", {})
        except (ValueError, AttributeError):
            defaults = {}
    return (
        float(n_sigma if n_sigma is not None else defaults.get("n_sigma", N_SIGMA)),
        float(
            threshold_sd
            if threshold_sd is not None
            else defaults.get("threshold_sd", THRESHOLD_SD)
        ),
        float(
            noise_sigma
            if noise_sigma is not None
            else defaults.get("noise_sigma", NOISE_SIGMA)
        ),
        float(
            min_far_frac
            if min_far_frac is not None
            else defaults.get("min_far_frac", MIN_FAR_FRAC)
        ),
    )


def gaussian_fit_index_wrapper(
    strf_obj, n_sigma=None, threshold_sd=None, noise_sigma=None, min_far_frac=None
):
    """Fit and score every STRF on `strf_obj`.

    Always does the whole recording; selecting ROIs is the caller's job, because the
    class caches this result and slices it. Returns a dict of 1D arrays, each of length
    ``len(strf_obj.strfs)`` in flat STRF order, with the keys in :data:`FIELDS`.
    """
    n_sigma, threshold_sd, noise_sigma, min_far_frac = _resolve(
        strf_obj, n_sigma, threshold_sd, noise_sigma, min_far_frac
    )
    collapsed = strf_obj.collapse_times()
    records = [
        fit_and_score(
            np.squeeze(collapsed[i]),
            n_sigma=n_sigma,
            threshold_sd=threshold_sd,
            noise_sigma=noise_sigma,
            min_far_frac=min_far_frac,
        )
        for i in range(len(collapsed))
    ]
    if not records:
        return {key: np.array([], dtype=float) for key in FIELDS}
    return {
        key: np.array([record[key] for record in records], dtype=float)
        for key in FIELDS
    }
