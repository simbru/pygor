"""
Analysis parameter management for pygor.

Provides a unified container for tracking analysis parameters across
preprocessing, registration, segmentation, and other pipeline steps.
"""

from dataclasses import dataclass, field
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


@dataclass
class AnalysisParams:
    """
    Unified parameter management for pygor analysis pipeline.

    Stores defaults from config, tracks applied parameters for each processing
    step, and provides serialization to/from TOML format.

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
        Light artifact width in pixels (set during preprocessing, used by registration)

    Examples
    --------
    >>> from pygor.params import AnalysisParams
    >>> params = AnalysisParams.from_config()
    >>> print(params)  # Shows defaults and state
    >>> params.to_toml()  # Export as TOML string
    """

    # Processing step parameters (None = not yet applied)
    preprocessing: Optional[dict] = None
    registration: Optional[dict] = None
    segmentation: Optional[dict] = None
    triggers: Optional[dict] = None

    # Processing state
    preprocessed: bool = False
    registered: bool = False
    segmented: bool = False

    # Key values shared between steps
    artifact_width: int = 2  # Set during preprocess, used by registration

    # Internal: track where defaults came from
    _config_source: str = field(default="package", repr=False)

    # Store defaults for reference
    _defaults: dict = field(default_factory=dict, repr=False)

    @classmethod
    def from_config(cls, config_path: Union[str, Path, None] = None) -> "AnalysisParams":
        """
        Create AnalysisParams with defaults loaded from config.

        Parameters
        ----------
        config_path : str, Path, or None
            Path to a TOML config file to merge with package defaults.
            If None, only package defaults are used.

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
        """
        from pygor.config import get_defaults

        # Determine config source
        config_source = "custom" if config_path else "package"

        # Get artifact_width from preprocessing defaults
        preprocess_defaults = get_defaults("preprocessing", config_path)
        artifact_width = preprocess_defaults.get("artifact_width", 2)

        # Store all defaults for reference
        defaults = {
            "preprocessing": preprocess_defaults,
            "registration": get_defaults("registration", config_path),
            "segmentation": get_defaults("segmentation", config_path),
            "triggers": get_defaults("triggers", config_path),
            "instrument": get_defaults("instrument", config_path),
            "strf": get_defaults("strf", config_path),
        }

        return cls(
            artifact_width=artifact_width,
            _config_source=config_source,
            _defaults=defaults,
        )

    def get_defaults(self, step: str) -> dict:
        """
        Get default parameters for a processing step.

        Parameters
        ----------
        step : str
            One of: "preprocessing", "registration", "segmentation", "triggers",
            "instrument", "strf"

        Returns
        -------
        dict
            Default parameters for the specified step

        Examples
        --------
        >>> params = AnalysisParams.from_config()
        >>> params.get_defaults("preprocessing")
        {'artifact_width': 2, 'flip_x': True, ...}
        >>> params.get_defaults("strf")
        {'contouring': {...}, 'spatial': {...}, ...}
        """
        if step not in self._defaults:
            raise ValueError(f"Unknown step: {step}. Must be one of: {list(self._defaults.keys())}")
        return self._defaults.get(step, {}).copy()

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

        # Load new config (merged with package defaults)
        new_config = _load_config(config_path)

        # Merge into our stored defaults
        self._defaults = _deep_merge(self._defaults, new_config)

        # Update artifact_width from new config
        if "preprocessing" in new_config:
            self.artifact_width = self._defaults["preprocessing"].get(
                "artifact_width", self.artifact_width
            )

        self._config_source = "custom"

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
        self.artifact_width = params.get("artifact_width", self.artifact_width)

    def mark_registration(self, params: dict, stats: dict = None) -> None:
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

    def to_dict(self) -> dict:
        """
        Export all parameters as a nested dictionary.

        Returns
        -------
        dict
            All parameters in a serializable format
        """
        return {
            "state": {
                "preprocessed": self.preprocessed,
                "registered": self.registered,
                "segmented": self.segmented,
            },
            "shared": {
                "artifact_width": self.artifact_width,
            },
            "preprocessing": self.preprocessing,
            "registration": self._clean_for_export(self.registration),
            "segmentation": self.segmentation,
            "triggers": self.triggers,
            "_config_source": self._config_source,
        }

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
        shared = data.get("shared", {})

        instance = cls(
            preprocessing=data.get("preprocessing"),
            registration=data.get("registration"),
            segmentation=data.get("segmentation"),
            triggers=data.get("triggers"),
            preprocessed=state.get("preprocessed", False),
            registered=state.get("registered", False),
            segmented=state.get("segmented", False),
            artifact_width=shared.get("artifact_width", 2),
            _config_source=data.get("_config_source", "loaded"),
        )
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
            ("instrument", None, None),  # No applied state for instrument
            ("preprocessing", self.preprocessing, self.preprocessed),
            ("registration", self.registration, self.registered),
            ("triggers", self.triggers, None),  # No state flag for triggers
            ("segmentation", self.segmentation, self.segmented),
            ("strf", None, None),  # STRF has nested structure, no applied state
        ]

        for i, (cat_name, applied, state_flag) in enumerate(categories):
            is_last_category = (i == len(categories) - 1)
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

            # Handle nested structure (strf has sub-categories)
            if cat_name == "strf" and defaults:
                sub_cats = list(defaults.keys())
                for j, sub_cat in enumerate(sub_cats):
                    is_last_sub = (j == len(sub_cats) - 1)
                    sub_prefix = "└── " if is_last_sub else "├── "
                    sub_child_prefix = "    " if is_last_sub else "│   "

                    lines.append(f"{child_prefix}{sub_prefix}{sub_cat}")
                    sub_defaults = defaults[sub_cat]
                    if show_all:
                        self._add_params_to_tree(
                            lines, sub_defaults, None,
                            child_prefix + sub_child_prefix
                        )
                    else:
                        param_count = len(sub_defaults)
                        lines.append(f"{child_prefix}{sub_child_prefix}... ({param_count} params)")
            elif defaults:
                if show_all or applied:
                    self._add_params_to_tree(lines, defaults, applied, child_prefix)
                else:
                    # Collapse to count
                    param_count = len(defaults)
                    lines.append(f"{child_prefix}... ({param_count} params, use show_all=True)")

            if not is_last_category:
                lines.append("│")

        return "\n".join(lines)

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
            ("strf", "STRF Analysis", None, None),
        ]

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

            # Handle nested STRF structure
            if cat_key == "strf" and defaults:
                html_parts.append(f'<details {open_attr}>')
                html_parts.append(f'<summary>{cat_name}{status_html}</summary>')

                for sub_key, sub_defaults in defaults.items():
                    html_parts.append(f'<details>')
                    html_parts.append(f'<summary>{sub_key}</summary>')
                    html_parts.append(self._params_to_html_table(sub_defaults, None))
                    html_parts.append('</details>')

                html_parts.append('</details>')
            elif defaults:
                html_parts.append(f'<details {open_attr}>')
                html_parts.append(f'<summary>{cat_name}{status_html}</summary>')
                html_parts.append(self._params_to_html_table(defaults, applied))
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
        lines = ["AnalysisParams:"]
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

        return "\n".join(lines)
