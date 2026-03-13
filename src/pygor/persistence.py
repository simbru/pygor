"""
Introspection-based HDF5 persistence for pygor objects.

Walks instance ``__dict__`` and serialises each attribute by type.
Used by ``Core._save_state`` / ``Core._from_saved_state`` and
``Experiment.save`` / ``Experiment.load``.

File convention: ``.pygor.h5`` to distinguish from IGOR-exported H5 files.
"""

import datetime
import json
import pathlib
import warnings

import h5py
import numpy as np

PYGOR_H5_VERSION = 1

# Instance attributes that should NOT be persisted — they are either
# non-serialisable (operator functions), transient caches, or init-only
# parameters that are not needed after construction.
SKIP_ATTRS = frozenset({
    # Core internals (reconstructed on load)
    "_Core__compare_ops_map",
    "_Core__keyword_lables",
    # Image backups (transient, usually None)
    "_original_images",
    "_pre_registration_images",
    # Raw ScanM header (all useful info already extracted)
    "_scanm_header",
    # Init-only parameters
    "do_preprocess",
    "config",
})


# ── JSON helpers ─────────────────────────────────────────────────────────

def _json_default(obj):
    """Fallback encoder for types that json.dumps cannot handle natively."""
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    # datetime — check datetime before date (datetime is a date subclass)
    if isinstance(obj, datetime.datetime):
        return {"__datetime__": obj.isoformat()}
    if isinstance(obj, datetime.date):
        return {"__date__": obj.isoformat()}
    if isinstance(obj, datetime.time):
        return {"__time__": obj.isoformat()}
    if isinstance(obj, datetime.timedelta):
        return {"__timedelta__": obj.total_seconds()}
    if isinstance(obj, pathlib.PurePath):
        return {"__path__": str(obj)}
    if isinstance(obj, (set, frozenset)):
        return list(obj)
    raise TypeError(f"Not JSON serializable: {type(obj).__name__}")


def _json_object_hook(obj):
    """Restore tagged objects produced by :func:`_json_default`."""
    if "__datetime__" in obj:
        return datetime.datetime.fromisoformat(obj["__datetime__"])
    if "__date__" in obj:
        return datetime.date.fromisoformat(obj["__date__"])
    if "__time__" in obj:
        return datetime.time.fromisoformat(obj["__time__"])
    if "__timedelta__" in obj:
        return datetime.timedelta(seconds=obj["__timedelta__"])
    if "__path__" in obj:
        return pathlib.Path(obj["__path__"])
    return obj


# ── write ────────────────────────────────────────────────────────────────

def write_value(group, key, value):
    """Write a single Python value into an HDF5 *group* (dataset or attr)."""
    from pygor.params import AnalysisParams

    if value is None:
        group.attrs[f"_none_{key}"] = True
        return

    # --- numpy masked array (before ndarray — MaskedArray is a subclass) ---
    if isinstance(value, np.ma.MaskedArray):
        sub = group.create_group(key)
        sub.attrs["_pygor_type"] = "masked_array"
        sub.create_dataset("data", data=np.asarray(value.data), compression="gzip")
        if value.mask is not np.ma.nomask and np.any(value.mask):
            sub.create_dataset("mask", data=np.asarray(value.mask))
        return

    # --- numpy array ---
    if isinstance(value, np.ndarray):
        if value.dtype.kind in ("U", "S", "O"):
            # String / object array → JSON list
            group.attrs[key] = json.dumps(value.tolist(), default=_json_default)
            group.attrs[f"_type_{key}"] = "json"
        else:
            group.create_dataset(key, data=value, compression="gzip")
        return

    # --- AnalysisParams ---
    if isinstance(value, AnalysisParams):
        group.attrs[key] = json.dumps(value.to_dict(), default=_json_default)
        group.attrs[f"_type_{key}"] = "AnalysisParams"
        return

    # --- datetime (datetime before date — subclass ordering) ---
    if isinstance(value, datetime.datetime):
        group.attrs[key] = value.isoformat()
        group.attrs[f"_type_{key}"] = "datetime"
        return
    if isinstance(value, datetime.date):
        group.attrs[key] = value.isoformat()
        group.attrs[f"_type_{key}"] = "date"
        return
    if isinstance(value, datetime.time):
        group.attrs[key] = value.isoformat()
        group.attrs[f"_type_{key}"] = "time"
        return
    if isinstance(value, datetime.timedelta):
        group.attrs[key] = value.total_seconds()
        group.attrs[f"_type_{key}"] = "timedelta"
        return

    # --- pathlib ---
    if isinstance(value, pathlib.PurePath):
        group.attrs[key] = str(value)
        group.attrs[f"_type_{key}"] = "path"
        return

    # --- bool (before int — bool is an int subclass) ---
    if isinstance(value, (bool, np.bool_)):
        group.attrs[key] = bool(value)
        group.attrs[f"_type_{key}"] = "bool"
        return

    # --- Python scalars ---
    if isinstance(value, (int, float, str)):
        group.attrs[key] = value
        return
    if isinstance(value, (np.integer,)):
        group.attrs[key] = int(value)
        return
    if isinstance(value, (np.floating,)):
        group.attrs[key] = float(value)
        return

    # --- dict ---
    if isinstance(value, dict):
        try:
            group.attrs[key] = json.dumps(value, default=_json_default)
            group.attrs[f"_type_{key}"] = "json_dict"
            return
        except (TypeError, ValueError, OverflowError):
            pass
        # Fallback: sub-group (for dicts with non-JSON-friendly values)
        sub = group.create_group(key)
        sub.attrs["_pygor_type"] = "dict"
        for k, v in value.items():
            write_value(sub, str(k), v)
        return

    # --- list / tuple ---
    if isinstance(value, (list, tuple)):
        type_tag = "list" if isinstance(value, list) else "tuple"
        # Try as numeric dataset
        if value:
            try:
                arr = np.asarray(value)
                if arr.dtype.kind in ("i", "f", "u", "b") and arr.ndim >= 1:
                    group.create_dataset(key, data=arr, compression="gzip")
                    group.attrs[f"_type_{key}"] = type_tag
                    return
            except (ValueError, TypeError):
                pass
        # Fallback: JSON
        try:
            group.attrs[key] = json.dumps(value, default=_json_default)
            group.attrs[f"_type_{key}"] = f"json_{type_tag}"
            return
        except (TypeError, ValueError):
            pass

    # --- fallback ---
    warnings.warn(
        f"Skipping non-serializable attribute '{key}' ({type(value).__name__})",
        stacklevel=3,
    )


# ── read ─────────────────────────────────────────────────────────────────

def read_group(group):
    """Reconstruct a ``dict`` of Python objects from an HDF5 group."""
    from pygor.params import AnalysisParams

    result = {}
    none_keys = set()
    type_hints = {}
    regular_attrs = {}

    # --- pass 1: categorise attrs ---
    for name, value in group.attrs.items():
        if name.startswith("_none_"):
            none_keys.add(name[6:])
        elif name.startswith("_type_"):
            type_hints[name[6:]] = str(value)
        elif name.startswith("_pygor_"):
            continue  # group-level metadata
        elif name.startswith("__"):
            continue  # system metadata (__class_name__, etc.)
        else:
            regular_attrs[name] = value

    # --- pass 2: decode regular attrs ---
    for key, value in regular_attrs.items():
        hint = type_hints.get(key)
        result[key] = _decode_attr(value, hint)

    # --- pass 3: datasets and sub-groups ---
    for key in group.keys():
        item = group[key]
        if isinstance(item, h5py.Group):
            pygor_type = item.attrs.get("_pygor_type", "")
            if pygor_type == "masked_array":
                data = np.array(item["data"])
                mask = np.array(item["mask"]) if "mask" in item else np.ma.nomask
                result[key] = np.ma.MaskedArray(data, mask=mask)
            elif pygor_type == "dict":
                result[key] = read_group(item)
            else:
                # Unknown sub-group — try as generic dict
                result[key] = read_group(item)
        elif isinstance(item, h5py.Dataset):
            arr = np.array(item)
            hint = type_hints.get(key)
            if hint == "list":
                result[key] = arr.tolist()
            elif hint == "tuple":
                result[key] = tuple(arr.tolist())
            else:
                result[key] = arr

    # --- pass 4: None values ---
    for key in none_keys:
        result[key] = None

    return result


def _decode_attr(value, hint):
    """Decode a single HDF5 attribute, applying *hint* if available."""
    if hint == "date":
        return datetime.date.fromisoformat(value)
    if hint == "time":
        return datetime.time.fromisoformat(value)
    if hint == "datetime":
        return datetime.datetime.fromisoformat(value)
    if hint == "timedelta":
        return datetime.timedelta(seconds=float(value))
    if hint == "path":
        return pathlib.Path(value)
    if hint == "bool":
        return bool(value)
    if hint in ("json", "json_dict"):
        return json.loads(value, object_hook=_json_object_hook)
    if hint == "json_list":
        return json.loads(value, object_hook=_json_object_hook)
    if hint == "json_tuple":
        return tuple(json.loads(value, object_hook=_json_object_hook))
    if hint == "AnalysisParams":
        from pygor.params import AnalysisParams
        data = json.loads(value, object_hook=_json_object_hook)
        return AnalysisParams._from_dict(data)

    # No hint — coerce numpy scalars to native Python types
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return value
