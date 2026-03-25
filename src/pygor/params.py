"""
Analysis parameter management for pygor.

Provides a unified container for tracking analysis parameters across
preprocessing, registration, segmentation, and other pipeline steps.

``_defaults`` is the single source of truth for what parameters will be
used when a processing method is called.  Use bracket syntax to inspect
or modify defaults::

    rec.params["segmentation.blob.threshold"]        # read
    rec.params["segmentation.blob.threshold"] = 0.1  # write

Applied-parameter records (``mark_*`` methods) track what *was* used for
provenance but do not feed back into defaults.
"""

from pathlib import Path
from typing import Optional, Union
import warnings

# Try to import tomllib (Python 3.11+) or tomli as fallback
try:
    import tomllib
    HAS_TOML_READ = True
except ImportError:
    try:
        import tomli as tomllib
        HAS_TOML_READ = True
    except ImportError:
        HAS_TOML_READ = False

# For writing TOML, we need tomli_w or similar
try:
    import tomli_w
    HAS_TOML_WRITE = True
except ImportError:
    HAS_TOML_WRITE = False


class AnalysisParams:
    """Unified parameter management for pygor analysis pipeline.

    Stores defaults from config, tracks applied parameters for each
    processing step, and provides serialization to/from TOML format.

    ``_defaults`` is the single source of truth for configurable
    parameters.  Cross-step values like ``artifact_width`` are exposed
    as properties that read/write through ``_defaults``.

    Attributes
    ----------
    preprocessing : dict or None
        Parameters applied during preprocessing (None if not yet run)
    registration : dict or None
        Parameters and stats from registration (None if not yet run)
    segmentation : dict or None
        Parameters used for ROI segmentation (None if not yet run)
    triggers : dict or None
        Parameters used for trigger detection (None if not yet run)
    preprocessed : bool
        Whether preprocessing has been applied
    registered : bool
        Whether registration has been applied
    segmented : bool
        Whether segmentation has been applied
    artifact_width : int
        Property — reads/writes ``_defaults["preprocessing"]["artifact_width"]``.

    Examples
    --------
    >>> from pygor.params import AnalysisParams
    >>> params = AnalysisParams.from_config()
    >>> print(params)  # Shows defaults and state
    >>> params.to_toml()  # Export as TOML string
    """

    def __init__(
        self,
        *,
        preprocessing: Optional[dict] = None,
        registration: Optional[dict] = None,
        segmentation: Optional[dict] = None,
        triggers: Optional[dict] = None,
        preprocessed: bool = False,
        registered: bool = False,
        segmented: bool = False,
        _config_source: str = "package",
        _defaults: Optional[dict] = None,
        analysis_type: str = "Core",
        steps: Optional[dict] = None,
        # Legacy — accepted for backward compat but ignored (property reads _defaults)
        artifact_width: Optional[int] = None,
    ):
        # Applied-parameter records (provenance)
        self.preprocessing = preprocessing
        self.registration = registration
        self.segmentation = segmentation
        self.triggers = triggers

        # Processing state flags
        self.preprocessed = preprocessed
        self.registered = registered
        self.segmented = segmented

        # Internal
        self._config_source = _config_source
        self._defaults = _defaults if _defaults is not None else {}
        self.analysis_type = analysis_type
        self.steps = steps if steps is not None else {}

        # If caller passed artifact_width explicitly (legacy / _from_dict),
        # seed it into _defaults so the property picks it up.
        if artifact_width is not None:
            self._defaults.setdefault("preprocessing", {})
            # Only seed if _defaults doesn't already have it
            self._defaults["preprocessing"].setdefault("artifact_width", artifact_width)

    # ------------------------------------------------------------------
    # artifact_width — property backed by _defaults (single source of truth)
    # ------------------------------------------------------------------

    @property
    def artifact_width(self) -> int:
        """Light artifact width in pixels.

        Reads from ``_defaults["preprocessing"]["artifact_width"]``.
        Set via this property or via bracket syntax::

            params.artifact_width = 5
            params["preprocessing.artifact_width"] = 5  # equivalent
        """
        return self._defaults.get("preprocessing", {}).get("artifact_width", 2)

    @artifact_width.setter
    def artifact_width(self, value: int):
        self._defaults.setdefault("preprocessing", {})["artifact_width"] = value

    # ------------------------------------------------------------------
    # Construction helpers
    # ------------------------------------------------------------------

    @classmethod
    def from_config(cls, config_path: Union[str, Path, None] = None, analysis_type: str = "Core") -> "AnalysisParams":
        """
        Create AnalysisParams with defaults loaded from config.

        Parameters
        ----------
        config_path : str, Path, or None
            Path to a TOML config file to merge with package defaults.
            If None, only package defaults are used.
        analysis_type : str, optional
            The analysis class name (e.g. "STRF", "CenterSurround", "OSDS").
            Used to load analysis-specific defaults from config and to label
            the analysis section in the repr. Default: "Core".

        Returns
        -------
        AnalysisParams
            New instance with defaults loaded from config

        Examples
        --------
        >>> # Use package defaults
        >>> params = AnalysisParams.from_config()

        >>> # Use a project-specific config
        >>> params = AnalysisParams.from_config("configs/high_zoom.toml")

        >>> # For a specific analysis type
        >>> params = AnalysisParams.from_config(analysis_type="CenterSurround")
        """
        from pygor.config import get_defaults

        config_source = "custom" if config_path else "package"

        # Build the single _defaults dict — all config lives here
        defaults = {
            "preprocessing": get_defaults("preprocessing", config_path),
            "registration": get_defaults("registration", config_path),
            "segmentation": get_defaults("segmentation", config_path),
            "triggers": get_defaults("triggers", config_path),
            "instrument": get_defaults("instrument", config_path),
            "deconvolution": get_defaults("deconvolution", config_path),
        }

        # Load analysis-type-specific defaults using lowercase TOML section name
        config_key = analysis_type.lower()
        analysis_defaults = get_defaults(config_key, config_path)
        if analysis_defaults:
            defaults[config_key] = analysis_defaults

        return cls(
            _config_source=config_source,
            _defaults=defaults,
            analysis_type=analysis_type,
        )

    # ------------------------------------------------------------------
    # Reading defaults
    # ------------------------------------------------------------------

    def get_defaults(self, step: str) -> dict:
        """Get a **copy** of default parameters for a processing step.

        Returns a snapshot of the current defaults.  Modifications to the
        returned dict do **not** affect stored defaults — use bracket
        syntax (``params["key.path"] = value``) to change defaults.

        Parameters
        ----------
        step : str
            A pipeline step name. Common keys: "preprocessing", "registration",
            "segmentation", "triggers", "instrument". The analysis-specific
            key (e.g. "strf", "osds") depends on the analysis_type.
            Use ``list(self._defaults.keys())`` to see available keys.

        Returns
        -------
        dict
            Copy of default parameters for the specified step

        Examples
        --------
        >>> params = AnalysisParams.from_config()
        >>> params.get_defaults("preprocessing")
        {'artifact_width': 3, 'flip_x': True, ...}
        >>> params.get_defaults(params.analysis_type)  # analysis-specific defaults
        {'contouring': {...}, 'spatial': {...}, ...}
        """
        if step not in self._defaults:
            raise ValueError(f"Unknown step: {step}. Must be one of: {list(self._defaults.keys())}")
        return self._defaults.get(step, {}).copy()

    # ------------------------------------------------------------------
    # Modifying defaults
    # ------------------------------------------------------------------

    def load_config(self, config_path: Union[str, Path]) -> None:
        """
        Load and merge a config file into current params.

        Useful for applying project-specific settings after loading data.
        Only affects the defaults stored in this instance, not already-applied
        processing parameters.

        Parameters
        ----------
        config_path : str or Path
            Path to a TOML config file to merge with current defaults.

        Examples
        --------
        >>> data = Core.from_scanm(path)
        >>> data.params.load_config("configs/noisy_recordings.toml")
        >>> print(data.params.artifact_width)  # May have changed
        """
        from pygor.config import load_config as _load_config, _deep_merge

        new_config = _load_config(config_path)
        self._defaults = _deep_merge(self._defaults, new_config)
        self._config_source = "custom"

    # ------------------------------------------------------------------
    # Recording applied parameters (provenance)
    # ------------------------------------------------------------------

    def mark_preprocessing(self, params: dict) -> None:
        """
        Record that preprocessing was applied with the given parameters.

        Parameters
        ----------
        params : dict
            The parameters that were used for preprocessing
        """
        self.preprocessing = params.copy()
        self.preprocessed = True

    def mark_registration(self, params: dict[str, object], stats: dict[str, object] | None = None) -> None:
        """
        Record that registration was applied with the given parameters.

        Parameters
        ----------
        params : dict
            The parameters that were used for registration
        stats : dict, optional
            Registration statistics (mean_shift, errors, etc.)
        """
        result = params.copy()
        if stats:
            result.update(stats)
        self.registration = result
        self.registered = True

    def mark_segmentation(self, params: dict) -> None:
        """
        Record that segmentation was applied with the given parameters.

        Parameters
        ----------
        params : dict
            The parameters that were used for segmentation
        """
        self.segmentation = params.copy()
        self.segmented = True

    def mark_triggers(self, params: dict) -> None:
        """
        Record trigger detection parameters.

        Parameters
        ----------
        params : dict
            The parameters that were used for trigger detection
        """
        self.triggers = params.copy()

    def mark_step(self, step_name: str, params: dict) -> None:
        """
        Record parameters for a named pipeline step.

        For steps that don't have dedicated mark_* methods (e.g.,
        correlation projection, trace extraction, snippet computation).

        Parameters
        ----------
        step_name : str
            Name identifying this step (e.g., "correlation_projection")
        params : dict
            Parameters that were used for this step
        """
        self.steps[step_name] = params.copy()

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------

    def to_dict(self) -> dict:
        """
        Export all parameters as a nested dictionary.

        Returns
        -------
        dict
            All parameters in a serializable format
        """
        result = {
            "state": {
                "preprocessed": self.preprocessed,
                "registered": self.registered,
                "segmented": self.segmented,
            },
            "analysis_type": self.analysis_type,
            "preprocessing": self.preprocessing,
            "registration": self._clean_for_export(self.registration),
            "segmentation": self.segmentation,
            "triggers": self.triggers,
            "_config_source": self._config_source,
            "_defaults": self._clean_for_export_nested(self._defaults),
        }
        if self.steps:
            result["steps"] = {
                k: self._clean_for_export(v) for k, v in self.steps.items()
            }
        return result

    def _clean_for_export(self, params: Optional[dict]) -> Optional[dict]:
        """Remove non-serializable items (like numpy arrays) from params."""
        if params is None:
            return None

        import numpy as np

        cleaned = {}
        for key, value in params.items():
            if isinstance(value, np.ndarray):
                # Convert small arrays to lists, skip large ones
                if value.size <= 100:
                    cleaned[key] = value.tolist()
                else:
                    cleaned[f"{key}_shape"] = list(value.shape)
            elif isinstance(value, (np.floating, np.integer)):
                cleaned[key] = float(value) if isinstance(value, np.floating) else int(value)
            elif isinstance(value, tuple):
                cleaned[key] = list(value)
            else:
                cleaned[key] = value
        return cleaned

    def _clean_for_export_nested(self, params: Optional[dict]) -> Optional[dict]:
        """Recursively clean nested dicts for serialization."""
        if params is None:
            return None
        import numpy as np

        cleaned = {}
        for key, value in params.items():
            if isinstance(value, dict):
                cleaned[key] = self._clean_for_export_nested(value)
            elif isinstance(value, np.ndarray):
                if value.size <= 100:
                    cleaned[key] = value.tolist()
                else:
                    cleaned[f"{key}_shape"] = list(value.shape)
            elif isinstance(value, (np.floating, np.integer)):
                cleaned[key] = float(value) if isinstance(value, np.floating) else int(value)
            elif isinstance(value, tuple):
                cleaned[key] = list(value)
            else:
                cleaned[key] = value
        return cleaned

    def to_toml(self) -> str:
        """
        Export parameters as a TOML string.

        Returns
        -------
        str
            TOML-formatted string of all parameters

        Raises
        ------
        ImportError
            If tomli_w is not installed
        """
        if not HAS_TOML_WRITE:
            raise ImportError(
                "tomli_w is required for TOML export. "
                "Install with: pip install tomli-w"
            )

        data = self.to_dict()
        # Remove None values for cleaner TOML
        data = {k: v for k, v in data.items() if v is not None}
        return tomli_w.dumps(data)

    @classmethod
    def from_toml(cls, toml_str: str) -> "AnalysisParams":
        """
        Create AnalysisParams from a TOML string.

        Parameters
        ----------
        toml_str : str
            TOML-formatted string

        Returns
        -------
        AnalysisParams
            New instance with parameters from TOML
        """
        if not HAS_TOML_READ:
            raise ImportError(
                "tomllib or tomli is required for TOML import. "
                "Install with: pip install tomli (Python <3.11)"
            )

        data = tomllib.loads(toml_str)
        return cls._from_dict(data)

    @classmethod
    def _from_dict(cls, data: dict) -> "AnalysisParams":
        """Create AnalysisParams from a dictionary."""
        state = data.get("state", {})

        # Backward compat: legacy "shared" section stored artifact_width separately
        shared = data.get("shared", {})
        legacy_aw = shared.get("artifact_width")

        instance = cls(
            preprocessing=data.get("preprocessing"),
            registration=data.get("registration"),
            segmentation=data.get("segmentation"),
            triggers=data.get("triggers"),
            preprocessed=state.get("preprocessed", False),
            registered=state.get("registered", False),
            segmented=state.get("segmented", False),
            artifact_width=legacy_aw,  # Seeds into _defaults if present
            _config_source=data.get("_config_source", "loaded"),
            analysis_type=data.get("analysis_type", "Core"),
            _defaults=data.get("_defaults"),
        )
        if "steps" in data:
            instance.steps = data["steps"]
        return instance

    def save_toml(self, path: Union[str, Path]) -> Path:
        """
        Save parameters to a TOML file.

        Parameters
        ----------
        path : str or Path
            Output file path

        Returns
        -------
        Path
            Path to the saved file
        """
        path = Path(path)
        toml_str = self.to_toml()
        path.write_text(toml_str)
        return path

    @classmethod
    def load_toml(cls, path: Union[str, Path]) -> "AnalysisParams":
        """
        Load parameters from a TOML file.

        Parameters
        ----------
        path : str or Path
            Input file path

        Returns
        -------
        AnalysisParams
            New instance with parameters from file
        """
        path = Path(path)
        toml_str = path.read_text()
        return cls.from_toml(toml_str)

    # ------------------------------------------------------------------
    # Display
    # ------------------------------------------------------------------

    def summary(self, show_all: bool = False) -> str:
        """
        Display a tree-style summary of all parameters.

        Shows defaults for each category with markers indicating processing state:
        - ✓ applied (green in terminals that support it)
        - ○ pending
        - Modified values shown as: default → applied

        Parameters
        ----------
        show_all : bool, default False
            If True, show all parameters. If False, collapse unchanged defaults
            to "..." for brevity.

        Returns
        -------
        str
            Tree-formatted parameter summary

        Examples
        --------
        >>> data.params.summary()
        >>> print(data.params.summary(show_all=True))
        """
        lines = []
        lines.append(f"AnalysisParams (source: {self._config_source})")
        lines.append(f"├── artifact_width: {self.artifact_width}")
        lines.append("│")

        # Define categories and their state
        categories = [
            ("instrument", None, None),
            ("preprocessing", self.preprocessing, self.preprocessed),
            ("registration", self.registration, self.registered),
            ("triggers", self.triggers, None),
            ("segmentation", self.segmentation, self.segmented),
        ]
        # Add analysis-specific section if defaults exist for it
        analysis_config_key = self.analysis_type.lower()
        if analysis_config_key in self._defaults:
            categories.append((analysis_config_key, None, None))

        for i, (cat_name, applied, state_flag) in enumerate(categories):
            is_last_category = (i == len(categories) - 1) and not self.steps
            prefix = "└── " if is_last_category else "├── "
            child_prefix = "    " if is_last_category else "│   "

            # Determine status marker
            if state_flag is True:
                status = " ✓ applied"
            elif state_flag is False:
                status = " ○ pending"
            elif applied is not None:
                status = " ✓ recorded"
            else:
                status = ""

            lines.append(f"{prefix}{cat_name}{status}")

            # Get defaults for this category
            defaults = self._defaults.get(cat_name, {})

            if defaults:
                # Separate nested sub-categories from flat params
                nested = {k: v for k, v in defaults.items() if isinstance(v, dict)}
                flat = {k: v for k, v in defaults.items() if not isinstance(v, dict)}

                # Show flat params first (if any)
                if flat:
                    if show_all or applied:
                        self._add_params_to_tree(lines, flat, applied, child_prefix)
                    elif not nested:
                        param_count = len(flat)
                        lines.append(f"{child_prefix}... ({param_count} params, use show_all=True)")

                # Show nested sub-categories
                if nested:
                    sub_cats = list(nested.keys())
                    for j, sub_cat in enumerate(sub_cats):
                        is_last_sub = (j == len(sub_cats) - 1) and not flat
                        sub_prefix = "└── " if is_last_sub else "├── "
                        sub_child_prefix = "    " if is_last_sub else "│   "

                        lines.append(f"{child_prefix}{sub_prefix}{sub_cat}")
                        sub_defaults = nested[sub_cat]
                        if show_all:
                            self._add_params_to_tree(
                                lines, sub_defaults, None,
                                child_prefix + sub_child_prefix
                            )
                        else:
                            param_count = len(sub_defaults)
                            lines.append(f"{child_prefix}{sub_child_prefix}... ({param_count} params)")

            if not is_last_category or self.steps:
                lines.append("│")

        # Show pipeline steps if any have been recorded
        if self.steps:
            lines.append("└── pipeline steps")
            step_items = list(self.steps.items())
            for i, (step_name, step_params) in enumerate(step_items):
                is_last = (i == len(step_items) - 1)
                step_prefix = "    └── " if is_last else "    ├── "
                step_child = "        " if is_last else "    │   "
                lines.append(f"{step_prefix}{step_name} ✓")
                self._add_params_to_tree(lines, step_params, None, step_child)

        return "\n".join(lines)

    def edit(self, section=None, blocking=True):
        """Open an interactive parameter editor (requires ``[gui]`` extra).

        Pops up an IGOR-style editable table. Changes are applied
        immediately when you click away from a cell.

        Parameters
        ----------
        section : str, optional
            Filter to a top-level section (e.g. "segmentation", "strf").
            If None, show all parameters.
        blocking : bool, optional
            If True (default), block until the editor window is closed.
            If False, return immediately after opening the window.

        Returns
        -------
        widget
            The editor widget (keep a reference to prevent garbage collection).

        Examples
        --------
        >>> rec.params.edit()                   # all params
        >>> rec.params.edit("segmentation")     # just segmentation
        >>> rec.params.edit(blocking=False)     # non-blocking window
        """
        try:
            from pygor.core.gui.param_editor import launch_editor
        except ImportError as e:
            raise ImportError(
                "Parameter editor requires Qt. Install with:\n"
                "  uv pip install 'pygor[gui]'"
            ) from e

        title = f"Parameters — {self.analysis_type}"
        if section:
            title += f" [{section}]"
        return launch_editor(self, section=section, title=title, blocking=blocking)

    # ------------------------------------------------------------------
    # Bracket access to _defaults
    # ------------------------------------------------------------------

    def _flatten_defaults(self, prefix="", d=None):
        """Flatten nested _defaults dict into dotted-path key/value pairs.

        Returns
        -------
        list of (str, value)
            Sorted list of (dotted.path, value) pairs for all leaf values.
        """
        if d is None:
            d = self._defaults
        items = []
        for key, value in d.items():
            path = f"{prefix}.{key}" if prefix else key
            if isinstance(value, dict):
                items.extend(self._flatten_defaults(path, value))
            else:
                items.append((path, value))
        return items

    def to_dataframe(self, section=None):
        """Return defaults as a two-column DataFrame (parameter path, value).

        Parameters
        ----------
        section : str, optional
            Filter to a top-level section (e.g. "segmentation", "strf").
            If None, show all defaults.

        Returns
        -------
        pandas.DataFrame
            Columns: ``parameter``, ``value``

        Examples
        --------
        >>> rec.params.to_dataframe()
        >>> rec.params.to_dataframe("segmentation")
        """
        import pandas as pd

        if section is not None:
            if section not in self._defaults:
                raise KeyError(
                    f"Unknown section '{section}'. "
                    f"Available: {list(self._defaults.keys())}"
                )
            flat = self._flatten_defaults(section, self._defaults[section])
        else:
            flat = self._flatten_defaults()
        return pd.DataFrame(flat, columns=["parameter", "value"])

    def __getitem__(self, dotted_key: str):
        """Get a default parameter value by dotted path.

        Parameters
        ----------
        dotted_key : str
            Dotted path into the defaults dict, e.g.
            ``"segmentation.blob.min_sigma"`` or ``"strf.calculate.n_colours"``.

        Returns
        -------
        value
            The parameter value.

        Raises
        ------
        KeyError
            If the path does not exist.

        Examples
        --------
        >>> rec.params["segmentation.blob.min_sigma"]
        0.8
        """
        keys = dotted_key.split(".")
        d = self._defaults
        for i, k in enumerate(keys):
            if not isinstance(d, dict) or k not in d:
                traversed = ".".join(keys[:i])
                if isinstance(d, dict):
                    available = list(d.keys())
                    raise KeyError(
                        f"Key '{k}' not found at '{traversed}'. "
                        f"Available keys: {available}"
                    )
                else:
                    raise KeyError(
                        f"'{traversed}' is a leaf value ({d!r}), "
                        f"cannot descend into '{k}'"
                    )
            d = d[k]
        return d

    def __setitem__(self, dotted_key: str, value):
        """Set a default parameter value by dotted path.

        Only allows setting existing keys to prevent typos from silently
        creating new parameters. The change affects subsequent processing
        steps that read from defaults.

        Parameters
        ----------
        dotted_key : str
            Dotted path into the defaults dict, e.g.
            ``"segmentation.blob.min_sigma"``.
        value
            The new value.

        Raises
        ------
        KeyError
            If the path does not exist (prevents typos).

        Examples
        --------
        >>> rec.params["segmentation.blob.min_sigma"] = 1.2
        >>> rec.params["strf.calculate.n_colours"] = 4
        """
        keys = dotted_key.split(".")
        d = self._defaults
        for i, k in enumerate(keys[:-1]):
            if not isinstance(d, dict) or k not in d:
                traversed = ".".join(keys[:i])
                if isinstance(d, dict):
                    available = list(d.keys())
                    raise KeyError(
                        f"Key '{k}' not found at '{traversed}'. "
                        f"Available keys: {available}"
                    )
                else:
                    raise KeyError(
                        f"'{traversed}' is a leaf value ({d!r}), "
                        f"cannot descend into '{k}'"
                    )
            d = d[k]

        final_key = keys[-1]
        if not isinstance(d, dict) or final_key not in d:
            parent = ".".join(keys[:-1])
            if isinstance(d, dict):
                available = list(d.keys())
                raise KeyError(
                    f"Key '{final_key}' not found at '{parent}'. "
                    f"Available keys: {available}. "
                    f"Only existing parameters can be set (prevents typos)."
                )
            else:
                raise KeyError(
                    f"'{parent}' is a leaf value ({d!r}), "
                    f"cannot set '{final_key}' on it"
                )
        d[final_key] = value

    # ------------------------------------------------------------------
    # Internal display helpers
    # ------------------------------------------------------------------

    def _add_params_to_tree(
        self,
        lines: list,
        defaults: dict,
        applied: Optional[dict],
        prefix: str
    ) -> None:
        """Add parameter lines to the tree, highlighting modifications."""
        items = list(defaults.items())
        for i, (key, default_val) in enumerate(items):
            is_last = (i == len(items) - 1)
            item_prefix = "└── " if is_last else "├── "

            # Check if this value was modified
            if applied and key in applied:
                applied_val = applied[key]
                # Skip large arrays
                if hasattr(applied_val, '__len__') and not isinstance(applied_val, (str, tuple)) and len(applied_val) > 10:
                    applied_str = f"<array len={len(applied_val)}>"
                else:
                    applied_str = repr(applied_val)

                if applied_val != default_val:
                    lines.append(f"{prefix}{item_prefix}{key}: {default_val} → {applied_str}")
                else:
                    lines.append(f"{prefix}{item_prefix}{key}: {applied_str}")
            else:
                lines.append(f"{prefix}{item_prefix}{key}: {default_val}")

    def _repr_html_(self) -> str:
        """
        Rich HTML representation for Jupyter notebooks.

        Returns collapsible sections for each parameter category with
        visual indicators for processing state and modified values.
        """
        html_parts = []

        # CSS styles
        html_parts.append("""
        <style>
        .pygor-params {
            font-family: 'Consolas', 'Monaco', monospace;
            font-size: 13px;
            line-height: 1.4;
            background: #ffffff;
            color: #333;
            padding: 12px;
            border-radius: 6px;
        }
        .pygor-params details {
            margin-left: 20px;
            margin-bottom: 8px;
        }
        .pygor-params summary {
            cursor: pointer;
            font-weight: bold;
            padding: 4px 8px;
            background: #f5f5f5;
            border-radius: 4px;
            user-select: none;
        }
        .pygor-params summary:hover {
            background: #e8e8e8;
        }
        .pygor-params .status-applied {
            color: #28a745;
            font-weight: normal;
        }
        .pygor-params .status-pending {
            color: #6c757d;
            font-weight: normal;
        }
        .pygor-params .param-table {
            margin: 8px 0 8px 20px;
            border-collapse: collapse;
            width: calc(100% - 40px);
        }
        .pygor-params .param-table td {
            padding: 3px 12px 3px 0;
            border-bottom: 1px solid #eee;
        }
        .pygor-params .param-key {
            color: #0366d6;
            font-weight: 500;
        }
        .pygor-params .param-value {
            color: #333;
        }
        .pygor-params .param-modified {
            color: #d73a49;
        }
        .pygor-params .param-arrow {
            color: #6c757d;
            padding: 0 8px;
        }
        .pygor-params .header {
            font-size: 14px;
            font-weight: bold;
            margin-bottom: 12px;
            padding-bottom: 8px;
            border-bottom: 2px solid #0366d6;
        }
        .pygor-params .shared-params {
            margin: 8px 0 16px 20px;
            padding: 8px 12px;
            background: #f8f9fa;
            border-radius: 4px;
            display: inline-block;
        }
        </style>
        """)

        # Container
        html_parts.append('<div class="pygor-params">')
        html_parts.append(f'<div class="header">AnalysisParams (source: {self._config_source})</div>')

        # Shared parameters
        html_parts.append('<div class="shared-params">')
        html_parts.append(f'<span class="param-key">artifact_width</span>: {self.artifact_width}')
        html_parts.append('</div>')

        # Categories
        categories = [
            ("instrument", "Instrument Calibration", None, None),
            ("preprocessing", "Preprocessing", self.preprocessing, self.preprocessed),
            ("registration", "Registration", self.registration, self.registered),
            ("triggers", "Triggers", self.triggers, None),
            ("segmentation", "Segmentation", self.segmentation, self.segmented),
        ]
        # Add analysis-specific section if defaults exist for it
        analysis_config_key = self.analysis_type.lower()
        if analysis_config_key in self._defaults:
            categories.append(
                (analysis_config_key, self.analysis_type, None, None)
            )

        for cat_key, cat_name, applied, state_flag in categories:
            # Status badge
            if state_flag is True:
                status_html = '<span class="status-applied"> ✓ applied</span>'
                open_attr = "open"  # Auto-expand applied sections
            elif state_flag is False:
                status_html = '<span class="status-pending"> ○ pending</span>'
                open_attr = ""
            elif applied is not None:
                status_html = '<span class="status-applied"> ✓ recorded</span>'
                open_attr = "open"
            else:
                status_html = ""
                open_attr = ""

            defaults = self._defaults.get(cat_key, {})

            if defaults:
                # Separate nested sub-categories from flat params
                nested = {k: v for k, v in defaults.items() if isinstance(v, dict)}
                flat = {k: v for k, v in defaults.items() if not isinstance(v, dict)}

                html_parts.append(f'<details {open_attr}>')
                html_parts.append(f'<summary>{cat_name}{status_html}</summary>')

                # Show flat params first (if any)
                if flat:
                    html_parts.append(self._params_to_html_table(flat, applied))

                # Show nested sub-categories
                for sub_key, sub_defaults in nested.items():
                    html_parts.append('<details>')
                    html_parts.append(f'<summary>{sub_key}</summary>')
                    html_parts.append(self._params_to_html_table(sub_defaults, None))
                    html_parts.append('</details>')

                html_parts.append('</details>')

        # Show pipeline steps if any have been recorded
        if self.steps:
            html_parts.append('<details open>')
            html_parts.append('<summary>Pipeline Steps</summary>')
            for step_name, step_params in self.steps.items():
                html_parts.append('<details open>')
                html_parts.append(
                    f'<summary>{step_name}'
                    f'<span class="status-applied"> ✓</span></summary>'
                )
                html_parts.append(self._params_to_html_table(step_params, None))
                html_parts.append('</details>')
            html_parts.append('</details>')

        html_parts.append('</div>')
        return "\n".join(html_parts)

    def _params_to_html_table(self, defaults: dict, applied: Optional[dict]) -> str:
        """Generate an HTML table for parameter display."""
        import html as html_module

        rows = []
        rows.append('<table class="param-table">')

        for key, default_val in defaults.items():
            key_escaped = html_module.escape(str(key))
            default_escaped = html_module.escape(repr(default_val))

            if applied and key in applied:
                applied_val = applied[key]
                # Handle large arrays
                if hasattr(applied_val, '__len__') and not isinstance(applied_val, (str, tuple)) and len(applied_val) > 10:
                    applied_str = f"<array len={len(applied_val)}>"
                else:
                    applied_str = repr(applied_val)
                applied_escaped = html_module.escape(applied_str)

                if applied_val != default_val:
                    rows.append(
                        f'<tr>'
                        f'<td class="param-key">{key_escaped}</td>'
                        f'<td class="param-value">{default_escaped}'
                        f'<span class="param-arrow">→</span>'
                        f'<span class="param-modified">{applied_escaped}</span></td>'
                        f'</tr>'
                    )
                else:
                    rows.append(
                        f'<tr>'
                        f'<td class="param-key">{key_escaped}</td>'
                        f'<td class="param-value">{applied_escaped}</td>'
                        f'</tr>'
                    )
            else:
                rows.append(
                    f'<tr>'
                    f'<td class="param-key">{key_escaped}</td>'
                    f'<td class="param-value">{default_escaped}</td>'
                    f'</tr>'
                )

        rows.append('</table>')
        return "\n".join(rows)

    def __repr__(self) -> str:
        """Pretty-print current state for inspection."""
        lines = [f"AnalysisParams ({self.analysis_type}):"]
        lines.append(f"  Config source: {self._config_source}")
        lines.append(f"  Artifact width: {self.artifact_width}")
        lines.append("")

        # State summary
        lines.append("  Processing state:")
        lines.append(f"    Preprocessed: {self.preprocessed}")
        lines.append(f"    Registered: {self.registered}")
        lines.append(f"    Segmented: {self.segmented}")
        lines.append("")

        # Applied parameters
        if self.preprocessing:
            lines.append("  Preprocessing params:")
            for k, v in self.preprocessing.items():
                lines.append(f"    {k}: {v}")
            lines.append("")

        if self.registration:
            lines.append("  Registration params:")
            for k, v in self.registration.items():
                # Skip large arrays in display
                if hasattr(v, '__len__') and not isinstance(v, (str, tuple)) and len(v) > 10:
                    lines.append(f"    {k}: <array of length {len(v)}>")
                else:
                    lines.append(f"    {k}: {v}")
            lines.append("")

        if self.segmentation:
            lines.append("  Segmentation params:")
            for k, v in self.segmentation.items():
                lines.append(f"    {k}: {v}")
            lines.append("")

        if self.triggers:
            lines.append("  Trigger params:")
            for k, v in self.triggers.items():
                lines.append(f"    {k}: {v}")
            lines.append("")

        if self.steps:
            lines.append("  Pipeline steps:")
            for step_name, step_params in self.steps.items():
                lines.append(f"    {step_name}:")
                for k, v in step_params.items():
                    lines.append(f"      {k}: {v}")

        return "\n".join(lines)
