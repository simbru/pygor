"""
IGOR-compatible ``wParamsNum`` / ``wParamsStr`` reconstruction for ``export_to_h5``.

These two waves carry ScanM acquisition metadata. IGOR reads ``wParamsNum`` **by
dimension label** — the only read across the OS scripts is
``lineDur = wParamsNum[%User_dxPix] * wParamsNum[%RealPixDur] * 1e-6`` in
OS_ParameterTable.ipf, which runs when the OS_Parameters wave is (re)created. A
``wParamsNum`` written without an ``IGORWaveDimensionLabels`` attribute makes
``FindDimLabel(...,"User_dxPix")`` return -1 → ``wParamsNum[-1]`` errors in IGOR.

``wParamsStr`` is read as ``wParamsStr[4]`` (experiment date). IGOR exports it as
fixed-length text (``HDF5SaveData /IGOR=8``); a variable-length string dataset is the
likely cause of the observed ``H5Dread failed`` load error.

Structure (labels, order, N+1 label-attr length for wParamsNum) is transcribed from a
genuine IGOR export (pygor/examples/strf_demo_data.h5). Unlike the OS_Parameters
analysis names, these acquisition fields are ScanM-hardware stable across script
versions, so the demo file is a reliable oracle here.
"""

import numpy as np

__all__ = [
    "build_wparamsnum",
    "write_wparamsnum",
    "build_wparamsstr",
    "write_wparamsstr",
    "WPARAMSSTR_LEN",
]

# 61-row IGORWaveDimensionLabels for the 60-point wParamsNum wave (N+1 convention:
# leading blank dimension-name slot, then label[i] names data[i-1]). Blank strings
# are genuine reserved gaps in the ScanM header.
WPARAMSNUM_LABELS: tuple[str, ...] = (
    "", "HdrLenInValuePairs", "HdrLenInBytes", "MinVolts_AO", "MaxVolts_AO",
    "StimChanMask", "MaxStimBufMapLen", "NumberOfStimBufs", "TargetedPixDur_us",
    "MinVolts_AI", "MaxVolts_AI", "InputChanMask", "NumberOfInputChans",
    "PixSizeInBytes", "NumberOfPixBufsSet", "PixelOffs", "PixBufCounter",
    "User_ScanMode", "User_dxPix", "User_dyPix", "User_nPixRetrace",
    "User_nXPixLineOffs", "User_divFrameBufReq", "User_ScanType",
    "User_nSubPixOversamp", "RealPixDur", "OversampFactor", "XCoord_um", "YCoord_um",
    "ZCoord_um", "ZStep_um", "Zoom", "Angle_deg", "User_NFrPerStep", "User_XOffset_V",
    "User_YOffset_V", "User_dzPix", "", "User_nZPixLineOff", "", "User_SetupID",
    "User_LaserWaveLen_nm", "", "", "User_aspectRatioFr", "User_stimBufPerFr",
    "User_nYPixLineOffs", "User_iChFastScan", "User_noYScan", "User_dxFrDecoded",
    "User_dyFrDecoded", "User_dzFrDecoded", "User_trajDefVRange_V", "User_nTrajParams",
    "User_zoomZ", "User_offsetZ_V", "User_zeroZ_V", "User_ETL_polarity_V",
    "User_ETL_min_V", "User_ETL_max_V", "User_ETL_neutral_V",
)

# 60 fallback defaults (from the genuine file). Recording-specific coordinates are
# zeroed here — they are supplied from live metadata during synthesis when known.
WPARAMSNUM_DEFAULTS: tuple[float, ...] = (
    80.0, 6028.0, -4.0, 4.0, 15.0, 1.0, 4.0, 5.0, -1.0, 5.0, 5.0, 2.0, 2.0, 40000.0,
    0.0, 0.0, 0.0, 200.0, 64.0, 50.0, 22.0, 0.0, 10.0, 10.0, 5.0, 10.0,
    0.0, 0.0, 0.0,          # XCoord_um / YCoord_um / ZCoord_um (filled from metadata)
    1.0, 1.35, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 1.0,
    0.0, 0.0, 0.0, 200.0, 64.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
)

# Genuine wParamsStr length; IGOR reads index 4 (date), pygor also uses 5 (time).
WPARAMSSTR_LEN = 18

# wParamsStr row labels, in order (ScM_FileIO.ipf SetDimLabel on wStrParams).
WPARAMSSTR_LABELS: tuple[str, ...] = (
    "GUID", "ComputerName", "UserName", "OrigPixDataFileName", "DateStamp_d_m_y",
    "TimeStamp_h_m_s_ms", "ScanM_PVer_TargetOS", "CallingProcessPath",
    "CallingProcessVer", "StimBufLenList", "TargetedStimDurList",
    "InChan_PixBufLenList", "User_ScanPathFunc", "IgorGUIVer", "User_Comment",
    "User_Objective", "RealStimDurList",
)

# ScanM header key (as returned by scanm.read_smh_header, type prefix dropped) ->
# wParamsNum label. Transcribed from the SCMIO_key_* table in ScM_FileIO.ipf, which
# is how IGOR fills wParamsNum when it loads an .smh. Header keys and labels differ
# (uFrameWidth -> User_dxPix), so looking labels up in the header directly misses
# most of them.
SCANM_HEADER_TO_WPARAMSNUM: dict[str, str] = {
    "HeaderLengthInValuePairs": "HdrLenInValuePairs",
    "Header_length_in_bytes": "HdrLenInBytes",
    "MinVoltsAO": "MinVolts_AO",
    "MaxVoltsAO": "MaxVolts_AO",
    "StimulusChannelMask": "StimChanMask",
    "MaxStimulusBufferMapLength": "MaxStimBufMapLen",
    "NumberOfStimulusBuffers": "NumberOfStimBufs",
    "TargetedPixelDuration_µs": "TargetedPixDur_us",
    "MinVoltsAI": "MinVolts_AI",
    "MaxVoltsAI": "MaxVolts_AI",
    "InputChannelMask": "InputChanMask",
    "PixelSizeInBytes": "PixSizeInBytes",
    "NumberOfFrames": "NumberOfPixBufsSet",
    "PixelOffset": "PixelOffs",
    "FrameCounter": "PixBufCounter",
    "ScanMode": "User_ScanMode",
    "FrameWidth": "User_dxPix",
    "FrameHeight": "User_dyPix",
    "PixRetraceLen": "User_nPixRetrace",
    "XPixLineOffs": "User_nXPixLineOffs",
    "ChunksPerFrame": "User_divFrameBufReq",
    "ScanType": "User_ScanType",
    "NSubPixOversamp": "User_nSubPixOversamp",
    "RealPixelDuration_µs": "RealPixDur",
    "Oversampling_Factor": "OversampFactor",
    "XCoord_um": "XCoord_um",
    "YCoord_um": "YCoord_um",
    "ZCoord_um": "ZCoord_um",
    "ZStep_um": "ZStep_um",
    "Zoom": "Zoom",
    "Angle_deg": "Angle_deg",
    "NFrPerStep": "User_NFrPerStep",
    "XOffset_V": "User_XOffset_V",
    "YOffset_V": "User_YOffset_V",
    "dZPixels": "User_dzPix",
    "ZPixLineOffs": "User_nZPixLineOff",
    "SetupID": "User_SetupID",
    "LaserWavelength_nm": "User_LaserWaveLen_nm",
    "AspectRatioFrame": "User_aspectRatioFr",
    "StimBufPerFr": "User_stimBufPerFr",
    "YPixLineOffs": "User_nYPixLineOffs",
    "iChFastScan": "User_iChFastScan",
    "dxFrDecoded": "User_dxFrDecoded",
    "dyFrDecoded": "User_dyFrDecoded",
    "dzFrDecoded": "User_dzFrDecoded",
    "trajDefVRange_V": "User_trajDefVRange_V",
    "nTrajParams": "User_nTrajParams",
    "zoomFactorZ": "User_zoomZ",
    "offsetZ_V": "User_offsetZ_V",
    "zeroZ_V": "User_zeroZ_V",
    "ETL_polarity_V": "User_ETL_polarity_V",
    "ETL_min_V": "User_ETL_min_V",
    "ETL_max_V": "User_ETL_max_V",
    "ETL_neutral_V": "User_ETL_neutral_V",
}

# Same for the string header entries that IGOR copies into wParamsStr. The
# list-valued rows (StimBufLenList etc.) are assembled from repeated per-channel
# keys and are left out; the full header keeps those.
SCANM_HEADER_TO_WPARAMSSTR: dict[str, str] = {
    "ComputerName": "ComputerName",
    "UserName": "UserName",
    "OriginalPixelDataFileName": "OrigPixDataFileName",
    "DateStamp": "DateStamp_d_m_y",
    "TimeStamp": "TimeStamp_h_m_s_ms",
    "ScanMproductVersionAndTargetOS": "ScanM_PVer_TargetOS",
    "CallingProcessPath": "CallingProcessPath",
    "CallingProcessVersion": "CallingProcessVer",
    "ScanPathFunc": "User_ScanPathFunc",
    "IgorGUIVer": "IgorGUIVer",
    "Comment": "User_Comment",
    "Objective": "User_Objective",
}


def _labels_index_map() -> dict:
    """{label_name: data_index}. label[i] names data[i-1] (N+1 convention)."""
    return {name: i - 1 for i, name in enumerate(WPARAMSNUM_LABELS) if name}


def wparamsnum_from_header(header: dict) -> dict:
    """``{wParamsNum label: value}`` for every mapped key present in a ScanM header."""
    out = {}
    for key, label in SCANM_HEADER_TO_WPARAMSNUM.items():
        if key in header:
            try:
                out[label] = float(header[key])
            except (TypeError, ValueError):
                pass
    return out


def wparamsstr_from_header(header: dict) -> dict:
    """``{wParamsStr label: value}`` for every mapped key present in a ScanM header."""
    return {
        label: str(header[key])
        for key, label in SCANM_HEADER_TO_WPARAMSSTR.items()
        if key in header
    }


def wparamsnum_to_dict(data, labels=None) -> dict:
    """``{label: value}`` from a wParamsNum wave and its IGORWaveDimensionLabels.

    ``labels`` follows the N+1 convention (label[i] names data[i-1]). Without
    labels, the standard 60-point layout is assumed only if the length matches;
    otherwise an empty dict is returned rather than guessing. Blank labels are
    reserved gaps and are skipped.
    """
    data = np.asarray(data, dtype=np.float64).reshape(-1)
    if labels is not None:
        names = [
            x.decode("utf-8", "replace") if isinstance(x, bytes) else str(x)
            for x in np.asarray(labels).reshape(-1)
        ][1:]
    elif len(data) == len(WPARAMSNUM_LABELS) - 1:
        names = list(WPARAMSNUM_LABELS[1:])
    else:
        return {}
    return {name: float(val) for name, val in zip(names, data) if name}


def wparamsstr_to_dict(arr) -> dict:
    """``{label: value}`` from a wParamsStr wave. Unlabelled rows keep their index."""
    out = {}
    for i, val in enumerate(np.asarray(arr).reshape(-1)):
        if isinstance(val, bytes):
            val = val.decode("utf-8", "replace")
        name = WPARAMSSTR_LABELS[i] if i < len(WPARAMSSTR_LABELS) else str(i)
        out[name] = str(val)
    return out


def build_wparamsnum(core):
    """Return ``(data, labels)`` for an IGOR-compatible ``wParamsNum`` dataset.

    Priority: verbatim captured wave (H5-loaded objects) → full reconstruction from
    ``_scanm_header`` (raw ScanM loads) → minimal fallback that at least makes the
    ``User_dxPix`` × ``RealPixDur`` line-duration read resolve.

    Returns
    -------
    data : np.ndarray (N,), float64
    labels : np.ndarray (N+1, 1), object (vlen str), labels[0] == ""
    """
    raw = getattr(core, "_wparamsnum_raw", None)
    raw_labels = getattr(core, "_wparamsnum_labels_raw", None)
    if raw is not None and raw_labels is not None:
        return np.asarray(raw, dtype=np.float64).copy(), np.asarray(raw_labels).copy()

    data = np.array(WPARAMSNUM_DEFAULTS, dtype=np.float64)
    name_to_index = _labels_index_map()

    # Core stores the ScanM header as _scanm_header; ScanMData as _header. The
    # header is not saved with a .recording.h5, so a reloaded object falls back
    # to the wParamsNum dict kept in its metadata.
    header = getattr(core, "_scanm_header", None) or getattr(core, "_header", None)
    metadata = getattr(core, "metadata", None)
    if header:
        known = wparamsnum_from_header(header)
    elif metadata:
        known = metadata.get("wParamsNum") or {}
    else:
        known = {}
    for name, val in known.items():
        if name in name_to_index:
            data[name_to_index[name]] = val

    if metadata:
        xyz = metadata.get("objectiveXYZ", None)
        if xyz is not None and len(xyz) >= 3:
            for name, val in (("XCoord_um", xyz[0]),
                              ("YCoord_um", xyz[1]),
                              ("ZCoord_um", xyz[2])):
                if name in name_to_index:
                    data[name_to_index[name]] = float(val)

    # Guarantee IGOR's LineDuration recompute (dxPix * RealPixDur * 1e-6) == linedur_s.
    linedur = getattr(core, "linedur_s", None)
    if (linedur is not None and np.isfinite(linedur) and linedur > 0
            and "User_dxPix" in name_to_index and "RealPixDur" in name_to_index):
        dx = data[name_to_index["User_dxPix"]]
        rp = data[name_to_index["RealPixDur"]]
        if not (dx > 0 and rp > 0 and abs(dx * rp * 1e-6 - linedur) <= 1e-9 * linedur):
            rp = rp if rp > 0 else 2.0            # keep a plausible pixel dwell (µs)
            dx = linedur * 1e6 / rp
            data[name_to_index["User_dxPix"]] = dx
            data[name_to_index["RealPixDur"]] = rp

    labels = np.empty((len(WPARAMSNUM_LABELS), 1), dtype=object)
    for i, name in enumerate(WPARAMSNUM_LABELS):
        labels[i, 0] = name
    return data, labels


def write_wparamsnum(h5_group, data, labels):
    """Write the ``wParamsNum`` dataset + its ``IGORWaveDimensionLabels`` attribute."""
    ds = h5_group.create_dataset("wParamsNum", data=data, dtype=np.float64)
    ds.attrs["IGORWaveDimensionLabels"] = labels
    return ds


def build_wparamsstr(core):
    """Return a fixed-length (``S100``) ``wParamsStr`` array.

    IGOR reads index 4 (date "YYYY-M-D"); pygor's data_helpers also reads index 5
    (time "H-M-S-ms"). Prefers a verbatim captured wave for H5-loaded objects.
    """
    raw = getattr(core, "_wparamsstr_raw", None)
    if raw is not None:
        arr = np.asarray(raw)
        if arr.dtype.kind == "S":
            return arr.copy()
        # Captured as object/unicode — coerce to fixed-length bytes.
        return np.array([s.encode("utf-8") if isinstance(s, str)
                         else (s if isinstance(s, bytes) else str(s).encode("utf-8"))
                         for s in arr.reshape(-1)], dtype="S100")

    arr = np.full(WPARAMSSTR_LEN, b"", dtype="S100")
    metadata = getattr(core, "metadata", None)
    if metadata:
        for i, name in enumerate(WPARAMSSTR_LABELS):
            val = (metadata.get("wParamsStr") or {}).get(name)
            if val:
                arr[i] = str(val).encode("utf-8")[:100]
        exp_date = metadata.get("exp_date", None)
        exp_time = metadata.get("exp_time", None)
        if exp_date is not None:
            arr[4] = f"{exp_date.year}-{exp_date.month:02d}-{exp_date.day:02d}".encode()
        if exp_time is not None:
            arr[5] = (f"{exp_time.hour:02d}-{exp_time.minute:02d}-"
                      f"{exp_time.second:02d}-00").encode()
    filename = getattr(core, "filename", None)
    if filename is not None:
        arr[0] = str(filename.stem).encode("utf-8")[:100]
    return arr


def write_wparamsstr(h5_group, arr):
    """Write the ``wParamsStr`` dataset (fixed-length text, matching IGOR /IGOR=8)."""
    return h5_group.create_dataset("wParamsStr", data=arr)
