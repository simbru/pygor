"""Calcium indicator deconvolution for STRF traces.

Provides Wiener deconvolution to remove calcium indicator blur from
fluorescence traces, sharpening temporal dynamics before STRF computation.

The kernel models a causal calcium transient:
    h(t) = exp(-t/tau_decay) * (1 - exp(-t/tau_rise)),  t >= 0

Default time constants target jGCaMP8f (Zhang et al. 2023):
    tau_rise ~ 2.5 ms, tau_decay ~ 75 ms.
"""

import numpy as np


def calcium_kernel(
    frame_dt_ms: float,
    rise_tau_ms: float = 2.5,
    decay_tau_ms: float = 75.0,
    kernel_window_ms: float = 800.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Discretize the calcium indicator kernel by sampling at frame-bin centers.

    Each bin value is h((n + 0.5) * dt). Sampling at bin centers rather than
    bin edges avoids the h(0)=0 artefact that would introduce a fictitious
    one-frame delay.

    Parameters
    ----------
    frame_dt_ms : float
        Frame duration in milliseconds.
    rise_tau_ms : float
        Rise time constant in ms (default 2.5, jGCaMP8f).
    decay_tau_ms : float
        Decay time constant in ms (default 75.0, jGCaMP8f).
    kernel_window_ms : float
        Duration of kernel support in ms.

    Returns
    -------
    t_centers : np.ndarray
        Time of each bin center in ms, shape (n_bins,).
    kernel : np.ndarray
        Normalized kernel values, shape (n_bins,). Sums to 1.
    """
    if frame_dt_ms <= 0:
        raise ValueError("frame_dt_ms must be > 0")

    n_bins = int(np.ceil(kernel_window_ms / frame_dt_ms))
    t_centers = (np.arange(n_bins) + 0.5) * frame_dt_ms

    kernel = np.exp(-t_centers / decay_tau_ms) * (1 - np.exp(-t_centers / rise_tau_ms))

    ksum = kernel.sum()
    if ksum <= 0:
        raise ValueError("Kernel has non-positive sum — check time constants.")
    kernel /= ksum
    return t_centers, kernel


def wiener_deconvolve(
    traces: np.ndarray,
    kernel: np.ndarray,
    lambd: float = 3e-3,
    pad_factor: int = 4,
) -> np.ndarray:
    """Causal Wiener deconvolution, vectorized over ROIs.

    Parameters
    ----------
    traces : np.ndarray, shape (n_rois, n_frames)
        Input traces (e.g. ΔF/F₀ or z-normalized). Each row is one ROI.
    kernel : np.ndarray, shape (n_kernel,)
        Causal impulse-response kernel (e.g. from ``calcium_kernel``).
        Will be normalized to unit sum internally.
    lambd : float
        Wiener regularization parameter. Larger = more smoothing.
    pad_factor : int
        Reflective padding on each side as a multiple of kernel length.

    Returns
    -------
    deconvolved : np.ndarray, shape (n_rois, n_frames)
        Deconvolved traces, same shape as input.
    """
    traces = np.asarray(traces, dtype=np.float64)
    kernel = np.asarray(kernel, dtype=np.float64)

    if traces.ndim == 1:
        traces = traces[np.newaxis, :]
        squeeze = True
    else:
        squeeze = False

    # Normalize kernel
    ksum = kernel.sum()
    if ksum <= 0:
        raise ValueError("Kernel must have positive sum.")
    kernel = kernel / ksum

    # Demean each ROI
    traces = traces - traces.mean(axis=1, keepdims=True)

    # Reflective padding
    pad = pad_factor * len(kernel)
    traces_padded = np.pad(traces, ((0, 0), (pad, pad)), mode="reflect")

    # Batched FFT deconvolution
    n_fft = traces_padded.shape[1] + len(kernel) - 1
    H = np.fft.rfft(kernel, n=n_fft)
    Y = np.fft.rfft(traces_padded, n=n_fft, axis=1)

    X = Y * np.conj(H) / (np.abs(H) ** 2 + lambd)
    deconv_full = np.fft.irfft(X, n=n_fft, axis=1)

    # Causal alignment (kernel onset at index 0) + un-pad
    n_padded = traces_padded.shape[1]
    deconvolved = deconv_full[:, :n_padded][:, pad:-pad]

    if squeeze:
        return deconvolved[0]
    return deconvolved
