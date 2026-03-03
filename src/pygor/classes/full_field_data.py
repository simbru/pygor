from dataclasses import dataclass, field
from pygor.classes.core_data import Core
import numpy as np


@dataclass(kw_only=False, repr=False) #Decide if kw_only is necessary
class FullField(Core):
    ipl_depths: np.ndarray = np.nan
    # Post init attrs
    name: str = field(init=False)
    averages: np.array = field(init=False)
    ms_dur: int = field(init=False)

    def __post_init__(self):
        # Post initialise the contents of Data class to be inherited
        # super().__dict__["data_types"].append(self.type)
        super().__post_init__()

    @staticmethod
    def _chop_epochs(
        traces: np.ndarray,
        mode: int,
        epochs_ms: list[tuple[int, int]],
    ) -> tuple[np.ndarray, list[slice], int]:
        """
        Split traces into stimulus chunks and extract epoch windows.

        Parameters
        ----------
        traces : ndarray, shape (n_rois, n_samples)
        mode : int
            Number of stimulus conditions (chunks to split into).
        epochs_ms : list of (start_ms, end_ms) tuples
            Time windows relative to each chunk start, in samples (ms == samples
            at 1 kHz; adjust caller-side if your sampling rate differs).

        Returns
        -------
        traces_chopped : ndarray, shape (mode, n_rois, samples_per_chunk)
        epoch_slices : list of slice
            One slice per entry in epochs_ms, clamped to chunk length.
        samples_per_chunk : int
        """
        n_samples = traces.shape[1]
        samples_per_chunk = n_samples // mode
        usable_samples = samples_per_chunk * mode
        if usable_samples < n_samples:
            print(f"Truncating {n_samples - usable_samples} samples for even split")

        traces_chopped = np.array(np.split(traces[:, :usable_samples], mode, axis=1))
        # Shape: (mode, n_rois, samples_per_chunk)

        epoch_slices = [
            slice(min(start, samples_per_chunk), min(end, samples_per_chunk))
            for start, end in epochs_ms
        ]

        return traces_chopped, epoch_slices, samples_per_chunk

    def get_chopped_traces(self, mode=None,
        epochs_ms: list[tuple[int, int]] = [(500, 2000), (2000, 3500)],
        baseline_ms: int | None = 50):
        """
        Return traces split by stimulus condition, optionally baseline-subtracted.

        Parameters
        ----------
        mode : int, optional
            Number of stimulus conditions. Defaults to self.trigger_mode.
        epochs_ms : list of (start_ms, end_ms) tuples
            Passed through to _chop_epochs for epoch slice computation.
        baseline_ms : int or None
            Number of samples from the start of each chunk to use as baseline.
            None to skip baseline subtraction.

        Returns
        -------
        traces_chopped : ndarray, shape (mode, n_rois, samples_per_chunk)
            Traces per condition (baseline-subtracted if baseline_ms is not None).
        """
        if mode is None:
            mode = self.trigger_mode
        traces_chopped, _, _ = self._chop_epochs(
            self.averages, mode, epochs_ms)
        if baseline_ms is not None:
            baseline = np.mean(traces_chopped[:, :, :baseline_ms], axis=2, keepdims=True)
            traces_chopped = traces_chopped - baseline
        return traces_chopped

    def get_amplitudes(self, mode=None, return_diff=True,
        epochs_ms: list[tuple[int, int]] = [(500, 2000), (2000, 3500)],
        baseline_ms: int | None = 50):
        """
        Extract mean amplitudes for each stimulus phase and epoch,
        optionally baseline-subtracted per chunk.

        Parameters
        ----------
        mode : int, optional
            Number of stimulus conditions (colors). Defaults to self.trigger_mode.
        return_diff : bool
            If True, return ON - OFF difference instead of per-epoch values.
            Assumes epochs_ms[0] = ON window, epochs_ms[1] = OFF window.
        epochs_ms : list of (start_ms, end_ms) tuples
            Time windows (relative to each stimulus chunk) to average over.
        baseline_ms : int or None
            Number of samples from the start of each chunk to use as baseline.
            None to skip baseline subtraction.

        Returns
        -------
        If return_diff=False:
            amps : ndarray, shape (mode, epochs_num, n_rois)
                Mean amplitudes for each color/epoch/ROI.
        If return_diff=True:
            diff : ndarray, shape (mode, n_rois)
                ON - OFF difference for each color/ROI.
        """
        traces = self.averages
        if mode is None:
            mode = self.trigger_mode

        traces_chopped, epoch_slices, _ = self._chop_epochs(traces, mode, epochs_ms)

        if baseline_ms is not None:
            baseline = np.mean(traces_chopped[:, :, :baseline_ms], axis=2, keepdims=True)
            traces_chopped = traces_chopped - baseline

        n_rois = traces.shape[0]
        amps = np.zeros((mode, len(epoch_slices), n_rois))

        for m in range(mode):
            for e, sl in enumerate(epoch_slices):
                amps[m, e, :] = np.mean(traces_chopped[m, :, sl], axis=1)

        if return_diff:
            return amps[:, 0, :] - amps[:, 1, :]  # Shape: (mode, n_rois)

        return amps

    def get_amplitudes_0(self, mode=None, **kwargs):
        """
        Convenience method to get ON amplitudes only (for compatibility with
        older code). See get_amplitudes with return_diff=False for details.
        """
        amps = self.get_amplitudes(mode=mode, return_diff=False, **kwargs)
        return amps[:, 0, :]  # Shape: (mode, n_rois)

    def get_amplitudes_1(self, mode=None, **kwargs):
        """
        Convenience method to get OFF amplitudes only (for compatibility with
        older code). See get_amplitudes with return_diff=False for details.
        """
        amps = self.get_amplitudes(mode=mode, return_diff=False, **kwargs)
        return amps[:, 1, :]  # Shape: (mode, n_rois)

    def get_opponency(self, amplitude_threshold: float = 0.5) -> np.ndarray:
        """
        Classify ROIs as On, Off, or Opponent based on full-field flicker responses.

        For each color, computes the ON - OFF amplitude difference. ROIs are
        classified based on whether any color exceeds the threshold positively
        or negatively:
        - "On": at least one color positive, none negative
        - "Off": at least one color negative, none positive
        - "Opp": both positive and negative colors present
        - nan: no color passes threshold

        Parameters
        ----------
        amplitude_threshold : float
            Minimum absolute ON-OFF difference to count a color as responsive.

        Returns
        -------
        opponency : ndarray of object, shape (n_rois,)
            "On", "Off", "Opp", or nan for each ROI.
        """
        tuning = self.get_amplitudes(return_diff=True)  # Shape: (mode, n_rois)
        pos = tuning > amplitude_threshold
        neg = tuning < -amplitude_threshold
        has_pos = np.any(pos, axis=0)
        has_neg = np.any(neg, axis=0)
        opponency_arr = np.full(self.num_rois, np.nan, dtype=object)
        opponency_arr[has_pos & ~has_neg] = "On"
        opponency_arr[has_neg & ~has_pos] = "Off"
        opponency_arr[has_pos & has_neg] = "Opp"
        return opponency_arr

    def get_transience(self, mode=None,
        on_window: tuple[int, int] = (500, 1950),
        off_window: tuple[int, int] = (2500, 3950),
        compare_frac: float = 0.33,
        min_amplitude: float = 0.5):
        """
        Measure response transience per color and ROI.

        For each stimulus phase (ON and OFF), compares the mean amplitude
        at the start vs end of the response window. Returns a normalised
        index from -1 to +1:
            -1 = fully transient (strong early, no late response)
             0 = sustained (early == late)
            +1 = ramping (weak early, strong late response)
           nan = no meaningful response in that phase

        Parameters
        ----------
        mode : int, optional
            Number of stimulus conditions. Defaults to self.trigger_mode.
        on_window : (start, end)
            Sample range within each chunk for the ON phase.
        off_window : (start, end)
            Sample range within each chunk for the OFF phase.
        compare_frac : float
            Fraction of each window to use for comparison. E.g. 0.5 compares
            the first half vs last half; 0.33 compares first third vs last third
            (ignoring the middle).
        min_amplitude : float
            Minimum mean absolute amplitude (baseline-subtracted) within the
            phase window for a ROI/color to be scored. Below this, transience
            is set to nan.

        Returns
        -------
        transience : ndarray, shape (2, mode, n_rois)
            Axis 0: [ON, OFF]. Values in [-1, 1], or nan if below min_amplitude.
        """
        chunks = self.get_chopped_traces(mode=mode)
        results = []
        for start, end in [on_window, off_window]:
            phase = chunks[:, :, start:end]  # (mode, n_rois, phase_samples)
            n = phase.shape[-1]
            split = int(n * compare_frac)
            early = np.mean(phase[:, :, :split], axis=-1)   # (mode, n_rois)
            late = np.mean(phase[:, :, -split:], axis=-1)    # (mode, n_rois)
            denom = np.abs(late) + np.abs(early)
            t = np.where(denom > 0, (late - early) / denom, 0.0)
            # Mask out non-responsive phases
            phase_amp = np.mean(np.abs(phase), axis=-1)  # (mode, n_rois)
            t[phase_amp < min_amplitude] = np.nan
            results.append(t)
        return np.array(results)

    def get_transience_dominant(self, **kwargs):
        """
        Return transience from whichever phase (ON/OFF) has the stronger
        response, chosen per ROI and color.

        Uses get_amplitudes to determine which phase dominates, then selects
        the corresponding transience score from get_transience.

        Parameters
        ----------
        **kwargs
            Passed to get_transience (mode, on_window, off_window, etc.).

        Returns
        -------
        transience : ndarray, shape (mode, n_rois)
            Transience score from the dominant phase per ROI.
        """
        on_window = kwargs.get('on_window', (500, 1950))
        off_window = kwargs.get('off_window', (2500, 3950))
        transience = self.get_transience(**kwargs)  # (2, mode, n_rois)
        amps = self.get_amplitudes(
            mode=kwargs.get('mode', None), return_diff=False,
            epochs_ms=[on_window, off_window]
        )  # (mode, 2, n_rois)
        pick_off = np.abs(amps[:, 1, :]) > np.abs(amps[:, 0, :])  # (mode, n_rois)
        return np.where(pick_off, transience[1], transience[0])

    def get_transcience_0(self, **kwargs):
        """
        Convenience method to get ON transience only (for compatibility with
        older code). See get_transience for details.
        """
        return self.get_transience(**kwargs)[0]  # Shape: (mode, n_rois)

    def get_transcience_1(self, **kwargs):
        """
        Convenience method to get OFF transience only (for compatibility with
        older code). See get_transience for details.
        """
        return self.get_transience(**kwargs)[1]  # Shape: (mode, n_rois)