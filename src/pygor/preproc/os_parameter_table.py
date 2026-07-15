"""
Canonical OS_Parameters table as declared in IGOR's OS_ParameterTable() function
(OS_ParameterTable.ipf). Used to synthesize a fresh, IGOR-compatible OS_Parameters
dataset + IGORWaveDimensionLabels attribute when no real IGOR-sourced wave was
captured (i.e. for objects loaded straight from ScanM .smh/.smp files), and to
patch known live values into either a synthesized or a captured-verbatim wave
when exporting back to an IGOR-compatible H5 file.

Index order matches entry_position in the .ipf script exactly (verified by a
regex parse of its SetDimLabel / entry_position control flow, including the
if/else branch around LineDuration). A `None` name marks a reserved/gap slot
(unlabelled, value NaN in the ipf script).
"""

import numpy as np

from typing import NamedTuple, Optional


class OSParamEntry(NamedTuple):
    name: Optional[str]
    default: float


# Authoritative layout transcribed from the CURRENTLY-INSTALLED OS_ParameterTable.ipf
# (~/Documents/WaveMetrics/Igor Pro 9 User Files/__User Procedures/OS/). That script
# is the contract the user's IGOR reads by dimension-label name; the demo file
# strf_demo_data.h5 is an OLDER script version (73 slots, `AverageStack_make`, no
# `AvgStack_SkipTrig`) and is NOT a safe source for the analysis-param names.
#
# Indices 0-67 are the ipf table verbatim (length 68, float64 — `make /o/n=100`,
# redimensioned to entry_position=68, no `/S` → double). Entries 68-70 are appended
# imaging-derived params that IGOR normally computes during RAW import (OS_STRFs.ipf
# reads `%samp_rate_Hz`), which do NOT get re-added on an H5 import — so pygor supplies
# them. A trailing gap (index 70) is deliberate: IGOR's on-disk IGORWaveDimensionLabels
# has length == data length (verified against genuine OS_Parameters, 73/73 — NOT N+1),
# leaving the LAST data point unlabelled, so the last real param must not sit there.
#
# A `None` name marks a reserved/gap slot (value NaN). Any field pygor tracks is patched
# over these defaults at export time via collect_known_os_parameters().
OS_PARAMETER_TABLE: tuple[OSParamEntry, ...] = (
    OSParamEntry("Detrend_Skip", 1),                           # 0
    OSParamEntry("Detrend_nTimeBin", 10),                      # 1
    OSParamEntry("Detrend_smooth_window", 1000),               # 2
    OSParamEntry("LightArtifact_cut", 2),                      # 3
    OSParamEntry("nPlanes", 1),                                # 4
    OSParamEntry(None, float("nan")),                          # 5  (gap)
    OSParamEntry("ROI_corr_min", 1),                           # 6
    OSParamEntry("ROI_GaussSize", 3),                          # 7
    OSParamEntry("ROI_minPx", 5),                              # 8
    OSParamEntry("ROI_maxPx", 15),                             # 9
    OSParamEntry("ROIGap_px", 1),                              # 10
    OSParamEntry("ROI_SD_min", 10),                            # 11
    OSParamEntry("useMask4Corr", 0),                           # 12
    OSParamEntry(None, float("nan")),                          # 13 (gap)
    OSParamEntry("ROI_PxBinning", 1),                          # 14
    OSParamEntry("IncludeDiagonals", 1),                       # 15
    OSParamEntry("TimeCompress", 1),                           # 16
    OSParamEntry(None, float("nan")),                          # 17 (gap)
    OSParamEntry("Skip_First_Triggers", 0),                    # 18
    OSParamEntry("Skip_Last_Triggers", 0),                     # 19
    OSParamEntry("Baseline_nSeconds", 5),                      # 20
    OSParamEntry("Ignore1stXseconds", 1),                      # 21
    OSParamEntry("IgnoreLastXseconds", 0),                     # 22
    OSParamEntry(None, float("nan")),                          # 23 (gap)
    OSParamEntry("Trigger_Mode", 1),                           # 24
    OSParamEntry("Stim_Marker", 0),                            # 25
    OSParamEntry("nLines_lumped", 1),                          # 26
    OSParamEntry(None, float("nan")),                          # 27 (gap)
    OSParamEntry("AvgStack_make", 0),                          # 28
    OSParamEntry("AvgStack_SkipTrig", 1),                      # 29
    OSParamEntry("AvgStack_firstplane", 1),                    # 30
    OSParamEntry("PlotOnlyMeans", 20),                         # 31
    OSParamEntry("PlotOnlyHeatMap", 50),                       # 32
    OSParamEntry(None, float("nan")),                          # 33 (gap)
    OSParamEntry("QCProjection_make", 0),                      # 34
    OSParamEntry("QCProj_TriggersPerStim", 1),                 # 35
    OSParamEntry("QCProjection_binning", 1),                   # 36
    OSParamEntry(None, float("nan")),                          # 37 (gap)
    OSParamEntry("Clustering_nClasses", 10),                   # 38
    OSParamEntry("Clustering_SDplot", 5),                      # 39
    OSParamEntry("Events_nMax", 1000),                         # 40
    OSParamEntry("Events_Threshold", 1),                       # 41
    OSParamEntry("Events_RateBins_s", 0.05),                   # 42
    OSParamEntry(None, float("nan")),                          # 43 (gap)
    OSParamEntry("Noise_EventSD", 0.7),                        # 44
    OSParamEntry("Noise_PxSize_degree", 3),                    # 45
    OSParamEntry("Noise_interval_sec", 0.078),                 # 46
    OSParamEntry("Noise_FilterLength_s", 2),                   # 47
    OSParamEntry("Kernel_SDplot", 30),                         # 48
    OSParamEntry("Noise_Compression", 10),                     # 49
    OSParamEntry("nColourChannels", 4),                        # 50
    OSParamEntry(None, float("nan")),                          # 51 (gap)
    OSParamEntry("LineDuration", 0.001),                       # 52
    OSParamEntry("Data_Channel", 0),                           # 53
    OSParamEntry("Data_Channel2", 1),                          # 54
    OSParamEntry("Trigger_Channel", 2),                        # 55
    OSParamEntry("Display_Stuff", 1),                          # 56
    OSParamEntry(None, float("nan")),                          # 57 (gap)
    OSParamEntry("Detrend_RatiometricData", 0),                # 58
    OSParamEntry("Use_Znorm", 1),                              # 59
    OSParamEntry("Trigger_Threshold", 20000),                  # 60
    OSParamEntry("Trigger_after_skip_s", 0.1),                 # 61
    OSParamEntry("Trigger_DisplayHeight", 6),                  # 62
    OSParamEntry("Trigger_LevelRead_after_lines", 2),          # 63
    OSParamEntry(None, float("nan")),                          # 64 (gap)
    OSParamEntry("Registration_AverageN", 10),                 # 65
    OSParamEntry("Registration_SkipN", 10),                    # 66
    OSParamEntry(None, float("nan")),                          # 67 (gap; end of ipf table)
    # ── appended imaging-derived params (see module note) ──
    OSParamEntry("samp_period", 0.064),                        # 68
    OSParamEntry("samp_rate_Hz", 15.625),                      # 69
    OSParamEntry(None, float("nan")),                          # 70 (trailing gap)
)

# Number of entries transcribed verbatim from OS_ParameterTable.ipf (the rest are
# pygor-appended imaging params). Used by tests to pin the ipf contract.
IPF_TABLE_LENGTH = 68


def collect_known_os_parameters(core) -> dict:
    """
    Inspect a Core (or ScanMData) instance's live attributes / AnalysisParams
    state and return {OS_Parameters_name: value} for every field pygor
    confidently tracks. Only includes entries where correspondence is
    unambiguous; anything not covered here is left untouched (either the
    captured raw value, or the ipf default).
    """
    values = {}

    if getattr(core, "linedur_s", None) is not None:
        values["LineDuration"] = float(core.linedur_s)
    if getattr(core, "n_planes", None) is not None:
        values["nPlanes"] = float(core.n_planes)
    if getattr(core, "trigger_mode", None) is not None:
        values["Trigger_Mode"] = float(core.trigger_mode)

    # Imaging-derived params IGOR normally computes at raw import (OS_STRFs.ipf reads
    # %samp_rate_Hz). frame_hz is the per-frame sampling rate; samp_period its inverse.
    frame_hz = getattr(core, "frame_hz", None)
    if frame_hz is not None and np.isfinite(frame_hz) and frame_hz > 0:
        values["samp_rate_Hz"] = float(frame_hz)
        values["samp_period"] = float(1.0 / frame_hz)

    # name-mangled private attrs set in Core._load_from_h5
    skip_first = getattr(core, "_Core__skip_first_frames", None)
    if skip_first is not None:
        values["Skip_First_Triggers"] = float(skip_first)
    skip_last = getattr(core, "_Core__skip_last_frames", None)
    if skip_last is not None:
        values["Skip_Last_Triggers"] = float(-skip_last)  # stored negated in Core

    params = getattr(core, "params", None)
    if params is not None:
        preprocessing = params.get_defaults("preprocessing")
        if "artifact_width" in preprocessing:
            values["LightArtifact_cut"] = float(preprocessing["artifact_width"])
        if "detrend" in preprocessing:
            values["Detrend_Skip"] = float(not preprocessing["detrend"])
        if "smooth_window_s" in preprocessing:
            values["Detrend_smooth_window"] = float(preprocessing["smooth_window_s"])
        if "time_bin" in preprocessing:
            values["Detrend_nTimeBin"] = float(preprocessing["time_bin"])

        triggers = params.get_defaults("triggers")
        if "threshold" in triggers:
            values["Trigger_Threshold"] = float(triggers["threshold"])
        if "min_gap_seconds" in triggers:
            values["Trigger_after_skip_s"] = float(triggers["min_gap_seconds"])

    return values


def build_os_parameters(live_values, captured_data=None, captured_labels=None):
    """
    Build (data, labels) for an IGOR-compatible OS_Parameters dataset +
    IGORWaveDimensionLabels attribute.

    If captured_data/captured_labels are given (raw arrays captured from a
    genuine IGOR-sourced H5 file at load time), reuse them verbatim except
    for patching named slots found in `live_values` by label lookup.

    If not given, synthesize fresh from OS_PARAMETER_TABLE (ipf defaults),
    patched by `live_values` where applicable.

    Returns
    -------
    data : np.ndarray, shape (N,), dtype float64
        IGOR's OS_ParameterTable() creates this wave via
        ``make /o/n=100 OS_Parameters = NaN`` with no ``/S`` flag, i.e. IGOR's
        default double precision (8 bytes/point). Writing single precision
        (4 bytes/point) here makes IGOR misread adjacent points as one
        double, producing garbage values.
    labels : np.ndarray, shape (N, 1), dtype object (vlen str), labels[0] == ""
        IGOR's on-disk IGORWaveDimensionLabels has exactly as many rows as the
        wave has points (leading blank dimension-name slot, ``labels[i]`` names
        ``data[i-1]``, last data point left unlabelled). Writing N+1 rows makes
        IGOR mis-apply the labels and read garbage.
    """
    if captured_data is not None and captured_labels is not None:
        data = np.array(captured_data, dtype=np.float64, copy=True)
        labels = np.array(captured_labels, copy=True)
        flat_labels = np.asarray(labels).reshape(-1)
        name_to_index = {
            str(name): i - 1  # label[i] describes data[i-1]
            for i, name in enumerate(flat_labels)
            if str(name) != ""
        }
        for name, value in live_values.items():
            if name in name_to_index:
                data[name_to_index[name]] = value
        return data, labels

    n = len(OS_PARAMETER_TABLE)
    data = np.array([entry.default for entry in OS_PARAMETER_TABLE], dtype=np.float64)
    name_to_index = {
        entry.name: i for i, entry in enumerate(OS_PARAMETER_TABLE) if entry.name is not None
    }
    for name, value in live_values.items():
        if name in name_to_index:
            data[name_to_index[name]] = value

    # IGOR writes the label attribute with exactly `n` rows (== data length):
    # a leading blank dimension-name slot, then label[i] naming data[i-1].
    # The final data point (a gap slot) is left unlabelled, so we only fill
    # labels[1 .. n-1] from entries[0 .. n-2].
    labels = np.empty((n, 1), dtype=object)
    labels[0, 0] = ""
    for i in range(1, n):
        name = OS_PARAMETER_TABLE[i - 1].name
        labels[i, 0] = name if name is not None else ""
    return data, labels


def write_os_parameters(h5_group, data, labels):
    """Write the OS_Parameters dataset + IGORWaveDimensionLabels attribute."""
    ds = h5_group.create_dataset("OS_Parameters", data=data, dtype=np.float64)
    ds.attrs["IGORWaveDimensionLabels"] = labels
    return ds
