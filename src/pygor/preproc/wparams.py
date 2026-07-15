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


def _labels_index_map() -> dict:
    """{label_name: data_index}. label[i] names data[i-1] (N+1 convention)."""
    return {name: i - 1 for i, name in enumerate(WPARAMSNUM_LABELS) if name}


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

    # Core stores the ScanM header as _scanm_header; ScanMData as _header.
    header = getattr(core, "_scanm_header", None) or getattr(core, "_header", None)
    if header:
        for name, idx in name_to_index.items():
            if name in header:
                try:
                    data[idx] = float(header[name])
                except (TypeError, ValueError):
                    pass

    metadata = getattr(core, "metadata", None)
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
