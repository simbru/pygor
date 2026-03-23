from dataclasses import dataclass, field
from typing import Any, Callable

from matplotlib.axes import Axes

try:
    from collections import Iterable
except ImportError:
    from collections.abc import Iterable
# Local imports
import math

# Dependencies
import operator
import pathlib
import warnings

import h5py
import matplotlib
import matplotlib.patheffects as path_effects
import matplotlib.pyplot as plt
import numpy as np
import scipy.ndimage
import skimage
from mpl_toolkits.axes_grid1 import make_axes_locatable

import pygor.core
import pygor.core.calculations
import pygor.core.gui
import pygor.core.methods
import pygor.core.plot
import pygor.data_helpers
import pygor.plotting.basic
import pygor.strf.contouring
import pygor.strf.spatial
import pygor.strf.temporal
import pygor.utils
import pygor.utils.helpinfo
from pygor.params import AnalysisParams


def try_fetch(file, key):
    try:
        result = file[key]
        result_shape = result.shape
        if result_shape != ():  # if not a scalar
            # reshape to match expected orientation for Python
            result = np.array(result).T
    except KeyError as error:
        result = None
        # # raise KeyError(f"'{key}' not found in {file.filename}, setting to np.nan") from error
        # warnings.warn(
        #     f"'{key}' not found in {file.filename}, setting to np.nan", stacklevel=2
        # )
        error
    return result


def try_fetch_table_params(file, params_key, file_key="OS_Parameters"):
    try:
        """
        Will always default to fetching given key from a IGOR table holding parameters. will default to 'OS_Parameters'
        but can be changed to other tables if needed for specific experiments.
        """
        attr_items = list(file[file_key].attrs.items())
        if not attr_items:
            raise KeyError(f"No attributes found under '{file_key}'")
        keys = np.asarray(attr_items[0][1]).squeeze()[1:]
        key_dict = {key: idx for idx, key in enumerate(keys)}
        return file[file_key][key_dict[params_key]]
    except KeyError as error:
        # # raise KeyError(f"'{key}' not found in {file.filename}, setting to np.nan") from error
        warnings.warn(
            f"'{params_key}' not found in {file.filename}, setting to np.nan",
            stacklevel=2,
        )
        error
        return np.nan


@dataclass
class Core:
    filename: str | pathlib.Path
    config: str | pathlib.Path = None  # Optional path to TOML config file
    do_preprocess: bool | dict = False  # Preprocessing options (for ScanM files)
    metadata: dict = field(init=False)
    rois: dict = field(init=False)
    type: str = field(init=False)
    frame_hz: float = field(init=False)
    averages: np.array = np.nan
    snippets: np.array = np.nan
    ms_dur: int = np.nan
    trigger_mode: int = 1  # defualt value
    num_rois: int = field(init=False)
    params: AnalysisParams = field(init=False)  # Analysis parameters

    def __post_init__(self):
        """initialize Core by auto-detecting file format and loading data."""
        # Ensure path is pathlib compatible
        if isinstance(self.filename, pathlib.Path) is False:
            self.filename = pathlib.Path(self.filename)

        ext = self.filename.suffix.lower()

        # Auto-detect format based on extension
        if ext in [".smp", ".smh"]:
            # ScanM file - delegate to internal loader
            self._load_from_scanm_internal()
        elif ext in [".h5", ".hdf5"]:
            # H5 file - use original loading logic
            if self.do_preprocess:
                warnings.warn(
                    "do_preprocess parameter is ignored for H5 files (typically already preprocessed). Call manually if needed.",
                    stacklevel=2,
                )
            self._load_from_h5()
        else:
            raise ValueError(
                f"Unknown file extension '{ext}'. "
                "Supported formats: .h5/.hdf5 (IGOR export), .smp/.smh (ScanM raw)"
            )

        # Initialize analysis parameters from config (both paths)
        self.params = AnalysisParams.from_config(self.config, analysis_type=self.type)

        # No backup yet — set lazily on first destructive operation
        self._original_images = None
        self._pre_registration_images = None

    def _load_from_h5(self):
        """Load data from IGOR-exported H5 file."""
        # Set type attribute
        self.type = self.__class__.__name__
        # Fetch all relevant data from the HDF5 file (if not in file, gets set to None)
        with h5py.File(self.filename, "r") as HDF5_file:
            # Data
            self.traces_raw = try_fetch(HDF5_file, "Traces0_raw")
            self.traces_znorm = try_fetch(HDF5_file, "Traces0_znorm")
            self.images = try_fetch(HDF5_file, "wDataCh0_detrended")
            self.trigger_images = try_fetch(HDF5_file, "wDataCh2")  # Trigger channel
            # Basic information
            self.metadata = pygor.data_helpers.metadata_dict(HDF5_file)
            self.rois = try_fetch(HDF5_file, "ROIs")
            # check if 0 in rois, if so, set to 1
            if self.rois is not None and np.any(self.rois == 0):
                self.rois[self.rois == 0] = 1
            if self.rois is not None:
                self.num_rois = len(np.unique(self.rois)) - 1
            else:
                self.num_rois = 0
                warnings.warn(
                    f"{self.filename.name}: No 'ROIs' dataset found in H5 file. "
                    "Was this file exported without ROI segmentation?",
                    stacklevel=3,
                )
            self.roi_sizes = try_fetch(HDF5_file, "RoiSizes")
            if self.roi_sizes is not None:
                self.roi_sizes = self.roi_sizes[: self.num_rois]
            # Timing parameters
            self.triggertimes = try_fetch(HDF5_file, "Triggertimes")
            self.triggertimes = self.triggertimes[~np.isnan(self.triggertimes)].astype(
                float
            )
            self.triggertimes_frame = try_fetch(HDF5_file, "Triggertimes_Frame")
            self.__skip_first_frames = int(
                try_fetch_table_params(HDF5_file, "Skip_First_Triggers")
            )  # Note name mangling to prevent accidents if
            self.__skip_last_frames = -int(
                try_fetch_table_params(HDF5_file, "Skip_Last_Triggers")
            )  # private class attrs share names
            self.ipl_depths = try_fetch(HDF5_file, "Positions")
            self.averages = try_fetch(HDF5_file, "Averages0")
            self.snippets = try_fetch(HDF5_file, "Snippets0")
            self.quality_indices = try_fetch(HDF5_file, "QualityCriterion")
            self.correlation_projection = try_fetch(HDF5_file, "correlation_projection")
            self.linedur_s = float(try_fetch_table_params(HDF5_file, "LineDuration"))
            self.trigger_mode = int(try_fetch_table_params(HDF5_file, "Trigger_Mode"))
            self.n_planes = int(try_fetch_table_params(HDF5_file, "nPlanes"))
            self.average_stack = try_fetch(HDF5_file, "Stack_Ave")
            exp_params = try_fetch(HDF5_file, "wExpParams")
            if exp_params is not None:
                self.stage = try_fetch_table_params(
                    HDF5_file, "stage", "wExpParams"
                ).decode("utf-8")
                self.orientation = try_fetch_table_params(
                    HDF5_file, "orientation", "wExpParams"
                ).decode("utf-8")
                self.depth = try_fetch_table_params(
                    HDF5_file, "depth", "wExpParams"
                ).decode("utf-8")
                self.stimulus = try_fetch_table_params(
                    HDF5_file, "stimulus", "wExpParams"
                ).decode("utf-8")
            if self.images is not None:
                self.frame_hz = float(
                    1 / (self.images.shape[1] / self.n_planes * self.linedur_s)
                )
            else:
                self.frame_hz = float(
                    1 / (self.average_stack.shape[0] / self.n_planes * self.linedur_s)
                )
        # Check that trigger mode matches phase number
        # if self.trigger_mode != self.phase_num:
        #     warnings.warn(
        #         f"{self.filename.stem}: Trigger mode {self.trigger_mode} does not match phase number {self.phase_num}",
        #         stacklevel=3,
        #     )
        # Imply from averages the ms_duration of one repeat
        if self.averages is not None:
            self.ms_dur = self.averages.shape[-1]
        else:
            self.ms_dur = None
        # Ensure triggertimes_frame does not include uneccessary nans
        if self.triggertimes_frame is not None:
            self.triggertimes_frame = self.triggertimes_frame[
                ~np.isnan(self.triggertimes_frame)
            ].astype(int)
        # Set name
        self.name = self.filename.stem
        # Set keyword lables
        self.__keyword_lables = {
            "ipl_depths": self.ipl_depths,
        }
        self.__compare_ops_map = {
            "==": operator.eq,
            ">": operator.gt,
            "<": operator.lt,
            ">=": operator.ge,
            "<=": operator.le,
        }

    def _load_from_scanm_internal(self):
        """
        Internal method to load ScanM data when using Core(path) with auto-detection.

        Uses default channel settings. For more control, use Core.from_scanm() directly.
        """
        import pygor.preproc.scanm as scanm_module

        # Default channel settings
        imaging_channel = 0
        trigger_channel = 2
        skip_first_triggers = 0
        skip_last_triggers = 0

        # Load ScanM data
        channels_to_load = list(set([imaging_channel, trigger_channel]))
        header, channel_data = scanm_module.load_scanm(
            self.filename, channels=channels_to_load
        )

        # Get actual number of frames recorded
        n_frames_total = header.get("NumberOfFrames", 0)
        frame_counter = header.get("FrameCounter", 0)
        stim_buf_per_fr = header.get("StimBufPerFr", 1)
        actual_frames = (n_frames_total - frame_counter) * stim_buf_per_fr

        # Get imaging data
        if imaging_channel not in channel_data:
            raise ValueError(f"Imaging channel {imaging_channel} not found in data")
        images = channel_data[imaging_channel][:actual_frames]

        # Compute timing parameters
        timing = scanm_module._compute_timing_params(header, images)

        # Parse datetime
        exp_date, exp_time = scanm_module._parse_scanm_datetime(header)

        # Detect triggers
        if trigger_channel in channel_data:
            trigger_stack = channel_data[trigger_channel][:actual_frames]
            trigger_frames, trigger_times = scanm_module.detect_triggers(
                trigger_stack,
                line_duration=timing["line_duration_s"],
            )

            # Apply skip settings
            if skip_last_triggers > 0:
                trigger_frames = trigger_frames[skip_first_triggers:-skip_last_triggers]
                trigger_times = trigger_times[skip_first_triggers:-skip_last_triggers]
            else:
                trigger_frames = trigger_frames[skip_first_triggers:]
                trigger_times = trigger_times[skip_first_triggers:]
        else:
            trigger_stack = None
            trigger_frames = np.array([], dtype=int)
            trigger_times = np.array([], dtype=float)

        # Set all required attributes
        self.type = self.__class__.__name__
        self.name = self.filename.stem

        # Image data
        self.images = images
        self.trigger_images = trigger_stack
        self.average_stack = images.mean(axis=0)
        self.correlation_projection = None

        # Timing
        self.frame_hz = timing["frame_hz"]
        self.linedur_s = timing["line_duration_s"]
        self.n_planes = timing["n_planes"]

        # Triggers
        self.triggertimes_frame = trigger_frames
        self.triggertimes = trigger_times
        self._Core__skip_first_frames = skip_first_triggers
        self._Core__skip_last_frames = (
            -skip_last_triggers if skip_last_triggers > 0 else 0
        )

        # Metadata
        self.metadata = {
            "filename": str(self.filename),
            "exp_date": exp_date,
            "exp_time": exp_time,
            "objectiveXYZ": (
                header.get("XCoord_um"),
                header.get("YCoord_um"),
                header.get("ZCoord_um"),
            ),
            "PixelDuration_us": header.get("PixelDuration"),
            "RetracePixels": header.get("RtrcLen"),
            "LineOffset": header.get("LineOffSet"),
            "FrameWidth": header.get("FrameWidth"),
            "FrameHeight": header.get("FrameHeight"),
            "NumberOfFrames": header.get("NumberOfFrames"),
            "FrameCounter": header.get("FrameCounter"),
            "StimBufPerFr": header.get("StimBufPerFr"),
            "ScanMode": header.get("ScanMode"),
            "Zoom": header.get("Zoom"),
            "Angle": header.get("Angle"),
            "User": header.get("User"),
            "Comment": header.get("Comment"),
        }

        # ROI-related (initially empty)
        self.rois = None
        self.num_rois = 0
        self.roi_sizes = None
        self.traces_raw = None
        self.traces_znorm = None

        # Other attributes
        self.quality_indices = None
        self.ipl_depths = None

        # Private attributes
        self._Core__keyword_lables = {"ipl_depths": None}
        self._Core__compare_ops_map = {
            "==": operator.eq,
            ">": operator.gt,
            "<": operator.lt,
            ">=": operator.ge,
            "<=": operator.le,
        }

        # Store header for export
        self._scanm_header = header

        # Apply preprocessing if requested
        if self.do_preprocess:
            if isinstance(self.do_preprocess, dict):
                self.preprocess(**self.do_preprocess)
            else:
                self.preprocess()

    @classmethod
    def from_h5(cls, path, config=None):
        """
        Load from IGOR-exported H5 file.

        This is an explicit alternative to `Core(path)` for H5 files.

        Parameters
        ----------
        path : str or Path
            Path to .h5 or .hdf5 file
        config : str or Path, optional
            Path to a TOML config file to merge with package defaults.

        Returns
        -------
        Core
            A fully initialized Core object.

        Examples
        --------
        >>> data = Core.from_h5("recording.h5")
        >>> data = Core.from_h5("recording.h5", config="configs/myconfig.toml")
        """
        return cls(filename=path, config=config)

    @classmethod
    def from_scanm(
        cls,
        path,
        imaging_channel: int = 0,
        trigger_channel: int = 2,
        skip_first_triggers: int = 0,
        skip_last_triggers: int = 0,
        trigger_mode: int = 1,
        preprocess: bool | dict = False,
        config: str | pathlib.Path = None,
    ):
        """
        Create a Core object directly from ScanM SMP/SMH files.

        This alternative constructor bypasses the need for an intermediate H5 file,
        loading data directly from ScanM format and populating all Core attributes.

        Parameters
        ----------
        path : str or Path
            Path to .smp or .smh file
        imaging_channel : int, optional
            Channel index for imaging data (default: 0)
        trigger_channel : int, optional
            Channel index for trigger detection (default: 2)
        skip_first_triggers : int, optional
            Number of initial triggers to skip (default: 0)
        skip_last_triggers : int, optional
            Number of final triggers to skip (default: 0)
        trigger_mode : int, optional
            Trigger detection mode (default: 1)
        preprocess : bool or dict, optional
            If False (default), load raw data without preprocessing.
            If True, apply preprocessing with defaults from config.
            If dict, apply preprocessing with custom parameters.

            Preprocessing parameters:
            - artifact_width (int): Light artifact pixels (default: 2)
            - flip_x (bool): X-flip image (default: True)
            - detrend (bool): Apply detrending (default: True)
            - smooth_window_s (float): Detrend window (default: 1000.0)
            - time_bin (int): Detrend binning (default: 10)
            - fix_first_frame (bool): Fix first frame (default: True)
        config : str or Path, optional
            Path to a TOML config file to merge with package defaults.
            Use this to apply project-specific parameter presets.

        Returns
        -------
        Core
            A fully initialized Core object with all standard methods available.

        Examples
        --------
        >>> from pygor.classes.core_data import Core
        >>> # Load raw data
        >>> data = Core.from_scanm("recording.smp")
        >>>
        >>> # Load with default preprocessing
        >>> data = Core.from_scanm("recording.smp", preprocess=True)
        >>>
        >>> # Load with custom preprocessing
        >>> data = Core.from_scanm("recording.smp", preprocess={"detrend": False})
        >>>
        >>> # Load with a project-specific config
        >>> data = Core.from_scanm("recording.smp", config="configs/high_zoom.toml")
        >>>
        >>> # Load raw, then preprocess later
        >>> data = Core.from_scanm("recording.smp")
        >>> data.preprocess(artifact_width=3, detrend=True)

        Notes
        -----
        To save the data for later use, call `data.export_to_h5("output.h5")`.
        """
        import pygor.preproc.scanm as scanm_module

        path = pathlib.Path(path)

        # Load ScanM data
        channels_to_load = list(set([imaging_channel, trigger_channel]))
        header, channel_data = scanm_module.load_scanm(path, channels=channels_to_load)

        # Get actual number of frames recorded
        n_frames_total = header.get("NumberOfFrames", 0)
        frame_counter = header.get("FrameCounter", 0)
        stim_buf_per_fr = header.get("StimBufPerFr", 1)
        actual_frames = (n_frames_total - frame_counter) * stim_buf_per_fr

        # Get imaging data
        if imaging_channel not in channel_data:
            raise ValueError(f"Imaging channel {imaging_channel} not found in data")
        images = channel_data[imaging_channel][:actual_frames]

        # Compute timing parameters
        timing = scanm_module._compute_timing_params(header, images)

        # Parse datetime
        exp_date, exp_time = scanm_module._parse_scanm_datetime(header)

        # Detect triggers
        if trigger_channel in channel_data:
            trigger_stack = channel_data[trigger_channel][:actual_frames]
            trigger_frames, trigger_times = scanm_module.detect_triggers(
                trigger_stack,
                line_duration=timing["line_duration_s"],
            )

            # Apply skip settings
            if skip_last_triggers > 0:
                trigger_frames = trigger_frames[skip_first_triggers:-skip_last_triggers]
                trigger_times = trigger_times[skip_first_triggers:-skip_last_triggers]
            else:
                trigger_frames = trigger_frames[skip_first_triggers:]
                trigger_times = trigger_times[skip_first_triggers:]
        else:
            trigger_stack = None
            trigger_frames = np.array([], dtype=int)
            trigger_times = np.array([], dtype=float)

        # Create instance without calling __post_init__
        # We use object.__new__ to bypass dataclass __init__, hacky but works
        instance = object.__new__(cls)

        # Set all required attributes manually
        instance.filename = path
        instance.type = cls.__name__
        instance.name = path.stem

        # Image data
        instance.images = images
        instance.trigger_images = (
            trigger_stack  # Store trigger channel for visualization
        )
        instance.average_stack = images.mean(axis=0)
        instance.correlation_projection = None

        # Timing
        instance.frame_hz = timing["frame_hz"]
        instance.linedur_s = timing["line_duration_s"]
        instance.n_planes = timing["n_planes"]

        # Triggers - now with accurate line-precision times
        instance.triggertimes_frame = trigger_frames
        instance.triggertimes = trigger_times
        instance.trigger_mode = trigger_mode
        instance._Core__skip_first_frames = skip_first_triggers
        instance._Core__skip_last_frames = (
            -skip_last_triggers if skip_last_triggers > 0 else 0
        )

        # Metadata - preserve all relevant header info
        instance.metadata = {
            "filename": str(path),
            "exp_date": exp_date,
            "exp_time": exp_time,
            "objectiveXYZ": (
                header.get("XCoord_um"),
                header.get("YCoord_um"),
                header.get("ZCoord_um"),
            ),
            # Preserve additional ScanM-specific metadata
            "PixelDuration_us": header.get("PixelDuration"),
            "RetracePixels": header.get("RtrcLen"),
            "LineOffset": header.get("LineOffSet"),
            "FrameWidth": header.get("FrameWidth"),
            "FrameHeight": header.get("FrameHeight"),
            "NumberOfFrames": header.get("NumberOfFrames"),
            "FrameCounter": header.get("FrameCounter"),
            "StimBufPerFr": header.get("StimBufPerFr"),
            "ScanMode": header.get("ScanMode"),
            "Zoom": header.get("Zoom"),
            "Angle": header.get("Angle"),
            "User": header.get("User"),
            "Comment": header.get("Comment"),
        }

        # ROI-related (initially empty)
        instance.rois = None
        instance.num_rois = 0
        instance.roi_sizes = None
        instance.traces_raw = None
        instance.traces_znorm = None

        # Other attributes
        instance.averages = None
        instance.snippets = None
        instance.ms_dur = None
        instance.quality_indices = None
        instance.ipl_depths = None

        # Private attributes that Core uses
        instance._Core__keyword_lables = {"ipl_depths": None}
        instance._Core__compare_ops_map = {
            "==": operator.eq,
            ">": operator.gt,
            "<": operator.lt,
            ">=": operator.ge,
            "<=": operator.le,
        }

        # Store header for export
        instance._scanm_header = header

        # Initialize analysis parameters from config
        instance.params = AnalysisParams.from_config(config)

        # No backup yet — set lazily on first destructive operation
        instance._original_images = None

        # Apply preprocessing if requested
        if preprocess:
            if isinstance(preprocess, dict):
                instance.preprocess(**preprocess)
            else:
                instance.preprocess()

        return instance

    def preprocess(
        self,
        artifact_width: int = None,
        flip_x: bool = None,
        detrend: bool = None,
        smooth_window_s: float = None,
        time_bin: int = None,
        fix_first_frame: bool = None,
        check_trigger_start: bool = True,
        force: bool = False,
    ) -> None:
        """
        Apply preprocessing to images in-place.

        Handles light artifact correction, X-flip, and optional detrending.
        This matches IGOR's OS_DetrendStack preprocessing pipeline.

        Parameters
        ----------
        artifact_width : int, optional
            Number of pixels affected by light artifact (default: 2).
            IGOR parameter: LightArtifact_cut
        flip_x : bool, optional
            X-flip the image (default: True). Standard for ScanM data.
        detrend : bool, optional
            Apply temporal baseline subtraction (default: True).
        smooth_window_s : float, optional
            Detrending smooth window in seconds (default: 1000.0).
        time_bin : int, optional
            Temporal binning factor for detrending speed (default: 10).
        fix_first_frame : bool, optional
            Copy frame 2 to frame 1 to fix first-frame artifact (default: True).
        check_trigger_start : bool, optional
            Check and correct TTL trigger signal if it starts low (default: True).
            Some recordings start with the trigger signal in LOW state before
            actual stimuli begin, causing a ghost trigger at index 0. This
            detects and corrects that condition.
        force : bool, optional
            If True, re-apply preprocessing even if already done (default: False).

        Raises
        ------
        RuntimeWarning
            If preprocessing was already applied and force=False.

        Examples
        --------
        >>> data = Core.from_scanm("recording.smp")
        >>> data.preprocess()  # Apply with defaults
        >>> data.preprocess(detrend=False)  # Skip detrending
        >>> data.preprocess(artifact_width=3, force=True)  # Re-apply with custom params

        See Also
        --------
        pygor.preproc.scanm.preprocess_stack : Underlying preprocessing function
        pygor.preproc.triggers.correct_ttl_baseline : TTL baseline correction
        pygor.config : Configuration management for defaults
        """
        import pygor.preproc.scanm as scanm_module

        # Check if already preprocessed
        if self.params.preprocessed and not force:
            warnings.warn(
                "Data has already been preprocessed. Use force=True to re-apply. "
                "Note: re-preprocessing already-preprocessed data may produce artifacts.",
                RuntimeWarning,
            )
            return

        # Backup raw images before first destructive operation
        if self._original_images is None:
            self._original_images = self.images.copy()

        # If force=True, restore from backup so we preprocess from raw
        if force and self._original_images is not None:
            self.images = self._original_images.copy()
            self._pre_registration_images = None  # invalidate, preprocessing changed

        # Get defaults from params (loaded from config)
        defaults = self.params.get_defaults("preprocessing")

        # Use current params.artifact_width as the default (can be modified before calling preprocess)
        if artifact_width is None:
            artifact_width = self.params.artifact_width

        # Collect user-provided params (filter out None values)
        user_params = {
            k: v
            for k, v in {
                "artifact_width": artifact_width,
                "flip_x": flip_x,
                "detrend": detrend,
                "smooth_window_s": smooth_window_s,
                "time_bin": time_bin,
                "fix_first_frame": fix_first_frame,
            }.items()
            if v is not None
        }

        # Merge defaults with user overrides
        params = {**defaults, **user_params}

        # Apply preprocessing
        self.images = scanm_module.preprocess_stack(
            self.images,
            frame_rate=self.frame_hz,
            **params,
        )

        # Update average_stack to reflect preprocessed data
        self.average_stack = self.images.mean(axis=0)

        # Check and correct TTL baseline if needed
        if (
            check_trigger_start
            and hasattr(self, "trigger_images")
            and self.trigger_images is not None
        ):
            from pygor.preproc.triggers import correct_ttl_baseline

            self.trigger_images, n_corrected = correct_ttl_baseline(self.trigger_images)
            if n_corrected > 0:
                # Re-detect triggers with corrected signal
                trigger_frames, trigger_times = scanm_module.detect_triggers(
                    self.trigger_images,
                    line_duration=self.linedur_s,
                )
                # The baseline correction may have clipped a trigger at the
                # boundary, shifting the first detected trigger late. Fix by
                # adjusting its time to match the regular interval pattern.
                # We do NOT prepend triggers — the pre-correction window is
                # genuine pre-stimulus baseline.
                if len(trigger_times) >= 3:
                    intervals = np.diff(trigger_times)
                    median_interval = (
                        np.median(intervals[1:]) if len(intervals) > 1 else intervals[0]
                    )
                    tolerance = 0.02 * median_interval
                    if abs(intervals[0] - median_interval) > tolerance:
                        trigger_times[0] = trigger_times[1] - median_interval
                        trigger_frames[0] = int(round(trigger_times[0] * self.frame_hz))
                        print(
                            f"TTL baseline correction: adjusted boundary trigger "
                            f"by {(intervals[0] - median_interval) * 1000:.0f}ms"
                        )
                self.triggertimes_frame = trigger_frames
                self.triggertimes = trigger_times

        # Reduce trigger channel to 2 columns to save memory (matches IGOR)
        if hasattr(self, "trigger_images") and self.trigger_images is not None:
            if self.trigger_images.ndim == 3 and self.trigger_images.shape[-1] > 2:
                self.trigger_images = self.trigger_images[:, :, :2].copy()

        # Record preprocessing in params (sets preprocessed=True and artifact_width)
        self.params.mark_preprocessing(params)

    def register(
        self,
        n_reference_frames: int | None = None,
        batch_size: int | None = None,
        artifact_width: int | None = None,
        upsample_factor: int | None = None,
        normalization: str | None = None,
        order: int | None = None,
        mode: str | None = None,
        force: bool = False,
        plot: bool = False,
        parallel: bool = True,
        n_jobs: int = -1,
        batch_mode: str | None = None,
        reference_mode: str | None = None,
        edge_crop: int | None = None,
        ref_plane: np.ndarray | None = None,
        verbose: bool = False,
    ) -> dict[str, object]:
        """
        Apply motion correction (registration) to images in-place.

        Uses batch-averaged phase cross-correlation to correct for sample
        drift and motion artifacts. This dramatically improves the quality
        of registration for low-SNR calcium imaging data.

        Parameters
        ----------
        n_reference_frames : int, optional
            Number of initial frames to average for stable reference (default: 1000).
        batch_size : int, optional
            Frames per batch for shift computation (default: 10).
            Larger values give better shift estimates but lower temporal resolution.
        upsample_factor : int, optional
            Subpixel precision factor (default: 10).
            Higher values increase precision but slow computation.
        normalization : str or None, optional
            Phase correlation normalization mode (default: None).
            For low-SNR data, None is recommended. Use 'phase' for high-SNR.
        order : int, optional
            Spline interpolation order for shifting (0-5, default: 1).
        mode : str, optional
            Edge handling mode for shifting (default: 'reflect').
                - 'reflect': Reflects at edge, duplicating the edge pixel
                - 'constant': Pads with zeros
                - 'nearest': Extends with the nearest edge pixel value
                - 'mirror': Reflects at edge without duplicating the edge pixel
                - 'wrap': Wraps around to the opposite edge
        force : bool, optional
            If True, re-apply registration even if already done (default: False).
        plot : bool, optional
            If True, display a matplotlib plot of shifts and errors (default: False).
        parallel : bool, optional
            Use parallel processing with FFT-based shifting for ~2x speedup
            (default: True).
        n_jobs : int, optional
            Number of parallel jobs. -1 uses all CPU cores (default: -1).
        batch_mode : str, optional
            Projection mode for batch images (default: "std").
            Options: "mean", "std", "var", "median", "max".
            Std captures morphology better and is less affected by
            temporal brightness fluctuations.
        reference_mode : str, optional
            Projection mode for reference image (default: "mean").
            Mean over many frames gives clean, stable structure.
        edge_crop : int, optional
            Pixels to crop from all edges before cross-correlation (default: 0).
            Useful to exclude edge artifacts from shift computation.
            Does not affect the output dimensions.
        ref_plane : ndarray, optional
            External 2D reference image for registration (default: None).
            If provided, all frames are aligned to this reference instead of
            computing one from the first n_reference_frames. Useful for
            cross-experiment alignment where you want multiple recordings
            in the same coordinate space.

        Returns
        -------
        dict
            Registration statistics with keys:
            - 'mean_shift': (dy, dx) mean shift in pixels
            - 'std_shift': (dy, dx) standard deviation of shifts
            - 'max_shift': (dy, dx) maximum shift in pixels
            - 'mean_error': mean registration error (lower is better)
            - 'shifts': per-batch shifts array (n_batches, 2)
            - 'errors': per-batch errors array (n_batches,)

        Raises
        ------
        RuntimeWarning
            If registration was already applied and force=False.

        Examples
        --------
        >>> data = Core.from_scanm("recording.smp", preprocess=True)
        >>> stats = data.register()  # Apply with defaults
        >>> print(f"Mean drift: {stats['mean_shift']}")
        >>>
        >>> # Custom parameters for faster processing
        >>> stats = data.register(batch_size=20, upsample_factor=5)
        >>>
        >>> # Force re-registration
        >>> stats = data.register(force=True)
        >>>
        >>> # Register and plot results
        >>> stats = data.register(plot=True)

        Notes
        -----
        - Registration should typically be applied AFTER preprocessing
        - For low-SNR calcium imaging, normalization=None is crucial
        - Preprocessing handles artifact removal before registration
        - Registration modifies self.images in-place

        See Also
        --------
        pygor.preproc.registration.register_stack : Underlying registration function
        preprocess : Preprocessing method (should be called first)
        """
        import pygor.preproc.registration as reg_module

        # Check if already registered
        if self.params.registered and not force:
            warnings.warn(
                "Data has already been registered. Use force=True to re-apply. "
                "Note: re-registering already-registered data may produce artifacts.",
                RuntimeWarning,
            )
            return self.params.registration or {}

        # Backup images before first destructive operation
        if self._original_images is None:
            self._original_images = self.images.copy()

        # Backup pre-registration state (preserves preprocessing)
        if self._pre_registration_images is None:
            self._pre_registration_images = self.images.copy()

        # If force=True, restore from pre-registration backup so we don't
        # re-register already-registered data (while preserving preprocessing)
        if force and self._pre_registration_images is not None:
            self.images = self._pre_registration_images.copy()

        # Store pre-registration state for plotting comparison
        original_stack = self.images.copy() if plot else None

        # Get defaults from params (loaded from config)
        defaults = self.params.get_defaults("registration")

        # Collect user-provided params (filter out None values)
        user_params = {
            k: v
            for k, v in {
                "n_reference_frames": n_reference_frames,
                "batch_size": batch_size,
                "upsample_factor": upsample_factor,
                "normalization": normalization,
                "order": order,
                "mode": mode,
                "parallel": parallel,
                "n_jobs": n_jobs,
                "batch_mode": batch_mode,
                "reference_mode": reference_mode,
                "edge_crop": edge_crop,
            }.items()
            if v is not None
        }

        # Merge defaults with user overrides
        params = {**defaults, **user_params}

        # Use artifact_width from params (set during preprocessing) if not explicitly provided
        if artifact_width is None:
            artifact_width = self.params.artifact_width

        # Apply registration
        registered, shifts, errors = reg_module.register_stack(
            self.images,
            return_shifts=True,
            artifact_width=artifact_width,
            ref_plane=ref_plane,
            **params,
        )

        # Include artifact_width in params so it gets stored
        params["artifact_width"] = artifact_width

        # Update images
        self.images = registered

        # Update average_stack to reflect registered data
        self.average_stack = self.images.mean(axis=0)

        # Compute statistics
        stats = {
            "mean_shift": tuple(shifts.mean(axis=0)),
            "std_shift": tuple(shifts.std(axis=0)),
            "max_shift": tuple(shifts.max(axis=0)),
            "mean_error": float(errors.mean()),
            "shifts": shifts,
            "errors": errors,
        }

        # Record registration in params (sets registered=True)
        self.params.mark_registration(params, stats)

        # Plot if requested
        if plot:
            self._plot_registration_results(
                shifts, errors, original_stack, params.get("reference_mode", "std")
            )
            plt.show()
        # if stats["mean_error"] < 0.05:
        if verbose:
            print(
                f"Registration complete.\n"
                f"  Mean error: {stats['mean_error']:.4f}\n"
                f"  Max shift: (y={stats['max_shift'][0]:.2f}, x={stats['max_shift'][1]:.2f})\n"
                f"  Mean shift: (y={stats['mean_shift'][0]:.2f}, x={stats['mean_shift'][1]:.2f})\n"
                f"  Shift SD: (y={stats['std_shift'][0]:.2f}, x={stats['std_shift'][1]:.2f})\n"
                f"  Registration error: {stats['mean_error']:.4f}"
            )
        return stats

    def reset_images(self) -> None:
        """Restore images to the state before any preprocessing or registration.

        Resets `self.images` from the backup taken before the first destructive
        operation, recomputes `self.average_stack`, and clears the preprocessed
        and registered flags so the pipeline can be re-run with new parameters.

        Raises
        ------
        RuntimeError
            If no backup exists (data was never preprocessed or registered).
        """
        if self._original_images is None:
            raise RuntimeError(
                "No original images stored. Cannot reset — data was never "
                "preprocessed or registered (or discard_original() was called)."
            )
        self.images = self._original_images.copy()
        self._pre_registration_images = None
        self.average_stack = self.images.mean(axis=0)
        self.params.preprocessed = False
        self.params.registered = False

    def discard_original(self) -> None:
        """Free the original image backup to reclaim memory.

        After calling this, `reset_images()` will no longer be available.
        """
        self._original_images = None
        self._pre_registration_images = None

    def _plot_registration_results(
        self,
        shifts: np.ndarray,
        errors: np.ndarray,
        original_stack: np.ndarray,
        reference_mode: str,
    ):
        """Plot registration results with images and shift traces."""
        import matplotlib.pyplot as plt

        from pygor.preproc.registration import _compute_projection

        batch_idx = np.arange(len(shifts))

        # Compute projections for comparison
        proj_original = _compute_projection(original_stack, reference_mode)
        proj_registered = _compute_projection(self.images, reference_mode)

        # Layout: 3 images on top, shift plot below
        fig = plt.figure(figsize=(12, 7))
        gs = fig.add_gridspec(2, 3, height_ratios=[1.2, 1], hspace=0.3, wspace=0.3)

        # Top row: images
        ax_ref = fig.add_subplot(gs[0, 0])
        ax_orig = fig.add_subplot(gs[0, 1])
        ax_reg = fig.add_subplot(gs[0, 2])

        # Bottom row: shift plot spanning all columns
        ax_shifts = fig.add_subplot(gs[1, :])

        # Shared colormap limits for original vs registered
        vmin = min(proj_original.min(), proj_registered.min())
        vmax = max(proj_original.max(), proj_registered.max())

        # Reference image: recompute from the original stack to show what was actually used
        n_ref = (
            self.params.registration.get("n_reference_frames", 100)
            if self.params.registration
            else 100
        )
        ref_image = _compute_projection(original_stack[:n_ref], reference_mode)
        ax_ref.imshow(ref_image, cmap="gray", origin="lower")
        ax_ref.set_title(f"Reference ({reference_mode}, n={n_ref})")
        ax_ref.axis("off")

        # Original projection
        ax_orig.imshow(proj_original, cmap="gray", vmin=vmin, vmax=vmax, origin="lower")
        ax_orig.set_title(f"Before ({reference_mode})")
        ax_orig.axis("off")

        # Registered projection
        ax_reg.imshow(
            proj_registered, cmap="gray", vmin=vmin, vmax=vmax, origin="lower"
        )
        ax_reg.set_title(f"After ({reference_mode})")
        ax_reg.axis("off")

        # Shift traces
        ax_shifts.plot(
            batch_idx, shifts[:, 0], "b-", label="Y shift", linewidth=1, alpha=0.8
        )
        ax_shifts.plot(
            batch_idx, shifts[:, 1], "r-", label="X shift", linewidth=1, alpha=0.8
        )
        ax_shifts.axhline(0, color="k", linestyle="--", alpha=0.3)
        ax_shifts.set_xlabel("Batch index")
        ax_shifts.set_ylabel("Shift (pixels)")
        ax_shifts.legend(loc="upper right")
        ax_shifts.grid(True, alpha=0.3)
        ax_shifts.set_title("Registration Shifts Over Time")

        plt.tight_layout()
        plt.show()

    def export_to_h5(
        self,
        output_path=None,
        overwrite: bool = False,
    ) -> pathlib.Path:
        """
        Export Core data to H5 file.

        Useful for saving preprocessed data or data loaded from ScanM files.

        Parameters
        ----------
        output_path : str or Path, optional
            Output H5 file path. If None, uses same name as source with .h5 extension.
        overwrite : bool, optional
            If True, overwrite existing file. Default False.

        Returns
        -------
        Path
            Path to the created H5 file.
        """
        import h5py

        if output_path is None:
            output_path = self.filename.with_suffix(".h5")
        else:
            output_path = pathlib.Path(output_path)

        if output_path.exists() and not overwrite:
            raise FileExistsError(
                f"File already exists: {output_path}. Use overwrite=True to replace."
            )

        with h5py.File(output_path, "w") as f:
            #  Image data
            # H5 expects (width, height, frames) - transposed from our (frames, height, width)
            if self.images is not None:
                images_t = self.images.transpose(2, 1, 0)
                # IGOR stores as uint16 (unsigned), matching raw ADC values
                f.create_dataset("wDataCh0_detrended", data=images_t, dtype=np.uint16)

            # Trigger channel (if available)
            if hasattr(self, "trigger_images") and self.trigger_images is not None:
                trigger_t = self.trigger_images.transpose(2, 1, 0)
                f.create_dataset("wDataCh2", data=trigger_t, dtype=np.int16)

            # Average stack
            if self.average_stack is not None:
                f.create_dataset(
                    "Stack_Ave", data=self.average_stack.T, dtype=np.float32
                )

            #  ROIs
            if self.rois is not None:
                f.create_dataset("ROIs", data=self.rois.T, dtype=np.int16)

            if self.roi_sizes is not None:
                f.create_dataset("RoiSizes", data=self.roi_sizes, dtype=np.int32)

            #  Traces
            if self.traces_raw is not None:
                f.create_dataset(
                    "Traces0_raw", data=self.traces_raw.T, dtype=np.float32
                )

            if self.traces_znorm is not None:
                f.create_dataset(
                    "Traces0_znorm", data=self.traces_znorm.T, dtype=np.float32
                )

            #  Trigger times
            if self.triggertimes is not None or self.triggertimes_frame is not None:
                max_triggers = max(
                    len(self.triggertimes_frame)
                    if self.triggertimes_frame is not None
                    else 0,
                    len(self.triggertimes) if self.triggertimes is not None else 0,
                    1000,
                )
                triggertimes = np.full(max_triggers, np.nan)
                if self.triggertimes is not None and len(self.triggertimes) > 0:
                    triggertimes[: len(self.triggertimes)] = self.triggertimes
                f.create_dataset("Triggertimes", data=triggertimes, dtype=np.float64)

                triggertimes_frame = np.full(max_triggers, np.nan)
                if (
                    self.triggertimes_frame is not None
                    and len(self.triggertimes_frame) > 0
                ):
                    triggertimes_frame[: len(self.triggertimes_frame)] = (
                        self.triggertimes_frame
                    )
                f.create_dataset(
                    "Triggertimes_Frame", data=triggertimes_frame, dtype=np.float64
                )

            #  wParamsStr (date/time metadata)
            if hasattr(self, "metadata") and self.metadata is not None:
                exp_date = self.metadata["exp_date"]
                exp_time = self.metadata["exp_time"]
                date_str = f"{exp_date.year}-{exp_date.month:02d}-{exp_date.day:02d}"
                time_str = f"{exp_time.hour:02d}-{exp_time.minute:02d}-{exp_time.second:02d}-00"

                params_str = [""] * 10
                params_str[4] = date_str
                params_str[5] = time_str
                params_str[0] = str(self.filename.stem)

                dt = h5py.special_dtype(vlen=str)
                params_str_ds = f.create_dataset(
                    "wParamsStr", (len(params_str),), dtype=dt
                )
                for i, s in enumerate(params_str):
                    params_str_ds[i] = s.encode("utf-8")

            #  wParamsNum (XYZ position)
            if hasattr(self, "metadata") and self.metadata is not None:
                params_num = np.zeros(50, dtype=np.float64)
                xyz = self.metadata.get("objectiveXYZ", (0, 0, 0))
                params_num[26] = xyz[0]
                params_num[27] = xyz[2]
                params_num[28] = xyz[1]
                f.create_dataset("wParamsNum", data=params_num, dtype=np.float64)

            #  OS_Parameters
            if hasattr(self, "linedur_s") and self.linedur_s is not None:
                os_params_keys = [
                    "placeholder",
                    "LineDuration",
                    "nPlanes",
                    "Trigger_Mode",
                    "Skip_First_Triggers",
                    "Skip_Last_Triggers",
                ]
                os_params_values = np.array(
                    [
                        0,
                        self.linedur_s,
                        self.n_planes,
                        self.trigger_mode,
                        0,
                        0,
                    ],
                    dtype=np.float64,
                )

                os_params_ds = f.create_dataset("OS_Parameters", data=os_params_values)
                os_params_ds.attrs["OS_Parameters"] = np.array(
                    [b"Keys"] + [k.encode() for k in os_params_keys], dtype=object
                )

            #  Optional data - check for both None and nan
            def _is_valid(attr):
                """Check if attribute is valid (not None and not nan)."""
                if attr is None:
                    return False
                if isinstance(attr, np.ndarray):
                    return attr.size > 0
                try:
                    return not np.isnan(attr).all()
                except (TypeError, ValueError):
                    return True

            if _is_valid(self.averages):
                f.create_dataset("Averages0", data=self.averages.T, dtype=np.float32)

            if _is_valid(self.snippets):
                f.create_dataset("Snippets0", data=self.snippets.T, dtype=np.float32)

            if _is_valid(self.ipl_depths):
                f.create_dataset("Positions", data=self.ipl_depths, dtype=np.float64)

            if _is_valid(self.correlation_projection):
                f.create_dataset(
                    "correlation_projection",
                    data=self.correlation_projection.T,
                    dtype=np.float32,
                )

            if _is_valid(self.quality_indices):
                f.create_dataset(
                    "QualityCriterion", data=self.quality_indices, dtype=np.float64
                )

        print(f"Exported to: {output_path}")
        return output_path

    # ── pygor state persistence (.pygor.h5) ──────────────────────────────

    def _save_state(self, group):
        """Save instance state to an HDF5 group via ``__dict__`` introspection.

        This is the counterpart to :meth:`_from_saved_state`.  Subclasses do
        **not** need to override this — any attribute in ``self.__dict__`` that
        is not in the skip-list is serialised automatically.
        """
        from pygor.persistence import SKIP_ATTRS, write_value

        group.attrs["__class_name__"] = type(self).__name__

        for attr_name, value in self.__dict__.items():
            if attr_name in SKIP_ATTRS:
                continue
            try:
                write_value(group, attr_name, value)
            except Exception as e:
                warnings.warn(
                    f"Failed to save attribute '{attr_name}': {e}", stacklevel=2
                )

    @classmethod
    def _from_saved_state(cls, group):
        """Reconstruct an instance from an HDF5 group, bypassing ``__post_init__``.

        Uses ``object.__new__`` so no file I/O or validation runs.
        """
        from pygor.persistence import read_group

        instance = object.__new__(cls)
        attrs = read_group(group)
        instance.__dict__.update(attrs)
        instance._reconstruct_internals()
        return instance

    def _reconstruct_internals(self):
        """Rebuild non-persisted internal state after loading from saved state.

        Subclasses should call ``super()._reconstruct_internals()`` and then
        restore their own caches / internal objects.
        """
        self._Core__compare_ops_map = {
            "==": operator.eq,
            ">": operator.gt,
            "<": operator.lt,
            ">=": operator.ge,
            "<=": operator.le,
        }
        self._Core__keyword_lables = {
            "ipl_depths": getattr(self, "ipl_depths", None),
        }
        if not hasattr(self, "_original_images"):
            self._original_images = None
        if not hasattr(self, "_pre_registration_images"):
            self._pre_registration_images = None

    def save_object(self, path, overwrite=False):
        """Save this single recording to a ``.pygor.h5`` file.

        Parameters
        ----------
        path : str or Path
            Output file path (recommended extension: ``.pygor.h5``).
        overwrite : bool, optional
            If True, overwrite an existing file.  Default False.

        Returns
        -------
        Path
            Path to the saved file.
        """
        path = pathlib.Path(path)
        if path.exists() and not overwrite:
            raise FileExistsError(
                f"File already exists: {path}. Use overwrite=True to replace."
            )
        with h5py.File(path, "w") as f:
            from pygor.persistence import PYGOR_H5_VERSION

            f.attrs["__pygor_h5_version__"] = PYGOR_H5_VERSION
            f.attrs["__num_recordings__"] = 1
            group = f.create_group("recording_000")
            self._save_state(group)
        print(f"Saved to: {path}")
        return path

    @classmethod
    def load_object(cls, path):
        """Load a single recording from a ``.pygor.h5`` file.

        Parameters
        ----------
        path : str or Path
            Path to a ``.pygor.h5`` file containing one recording.

        Returns
        -------
        Core (or subclass)
            The reconstructed recording object.
        """
        import pygor.load as pygor_load

        path = pathlib.Path(path)
        with h5py.File(path, "r") as f:
            group_names = sorted(k for k in f.keys() if k.startswith("recording_"))
            if not group_names:
                raise ValueError(f"No recording groups found in {path}")
            group = f[group_names[0]]
            class_name = group.attrs["__class_name__"]
            rec_cls = getattr(pygor_load, class_name)
            return rec_cls._from_saved_state(group)

    def __repr__(self):
        # For pretty printing
        date = self.metadata["exp_date"].strftime("%d-%m-%Y")
        return f"{date}:{self.__class__.__name__}:{self.filename.stem}"

    def __str__(self):
        # For pretty printing
        return f"{self.__class__}"

    @property
    def is_registered(self):
        """Whether registration has been applied to the images."""
        return self.params.registered

    @property
    def frametime_ms(self):
        time_arr = np.arange(self.traces_raw.shape[1]) / self.frame_hz
        return time_arr

    def get_help(self, hints=False, types=False) -> None:
        """
        Get help information for the object, including methods and attributes.

        Parameters
        ----------
        hints : bool, optional
            Whether to include hints in the help information (default is False)
        types : bool, optional
            Whether to include types in the help information (default is False)

        Returns
        -------
        None
        """
        # Check if this class has patterns to exclude from help
        exclude_patterns = getattr(self, "_help_exclude_patterns", None)
        method_list = pygor.utils.helpinfo.get_methods_list(
            self, with_returns=types, exclude_patterns=exclude_patterns
        )
        attribute_list = pygor.utils.helpinfo.get_attribute_list(self, with_types=types)
        welcome = pygor.utils.helpinfo.welcome_help(
            self.type, self.metadata, hints=hints
        )
        attrs = pygor.utils.helpinfo.attrs_help(attribute_list, hints=hints)
        meths = pygor.utils.helpinfo.meths_help(method_list, hints=hints)
        pygor.utils.helpinfo.print_help(
            [welcome, attrs, meths, pygor.utils.helpinfo.text_exit()]
        )

    def view_stack_projection(
        self,
        func: Callable[..., Any] = np.mean,
        axis: int = 0,
        cbar: bool = False,
        ax: Axes | None = None,
        figsize: tuple[float | None, float | None] = (None, None),
        figsize_scale: float | None = None,
        zcrop: tuple = None,
        xcrop: tuple = None,
        ycrop: tuple = None,
        alpha: float = 0.5,
        show_axes: bool = False,
        **kwargs: Any,
    ):
        """
        Display a projection of the image stack using the specified function.

        Parameters:
        - func: Callable or str, optional, default: np.mean
            The function used to compute the projection along the specified axis.
            If "average_stack", uses self.average_stack directly.
            If pygor.core.methods.correlation_map, applies correlation mapping.
        - axis: int, optional, default: 0
            The axis along which the projection is computed.
        - cbar: bool, optional, default: False
            Whether to display a colorbar.
        - ax: matplotlib.axes.Axes, optional, default: None
            The matplotlib axes to use for the display. If None, the current axes will be used.
        - figsize: tuple, optional, default: (None, None)
            Figure size. If (None, None), defaults to (10, 10).
        - figsize_scale: float, optional, default: None
            Scale factor for figure size.
        - alpha: float, optional, default: 0.5
            Alpha transparency value (for consistency with view_stack_rois).

        Returns:
        tuple
            (fig, ax) - matplotlib figure and axes objects
        """
        if figsize == (None, None):
            figsize = (10, 10)
        if figsize_scale is not None:
            figsize = np.array(figsize) * np.array(figsize_scale)
        else:
            figsize_scale = 1
        if ax is None:
            fig, ax = plt.subplots(figsize=figsize)
        else:
            fig = plt.gcf()
        if zcrop is None:
            zstart = None
            zstop = None
        else:
            zstart = zcrop[0]
            zstop = zcrop[1]
        if xcrop is None:
            xstart = None
            xstop = None
        else:
            xstart = xcrop[0]
            xstop = xcrop[1]
        if ycrop is None:
            ystart = None
            ystop = None
        else:
            ystart = ycrop[0]
            ystop = ycrop[1]
        if func == "average_stack":
            scanv = ax.imshow(
                self.average_stack, cmap="Greys_r", origin="lower", **kwargs
            )
        elif func == pygor.core.methods.correlation_map:
            correlation_result = func(
                self.images[zstart:zstop, ystart:ystop, xstart:xstop]
            )
            scanv = ax.imshow(
                correlation_result, cmap="Greys_r", origin="lower", **kwargs
            )
        else:
            scanv = ax.imshow(
                func(self.images[zstart:zstop, ystart:ystop, xstart:xstop:], axis=axis),
                cmap="Greys_r",
                origin="lower",
                **kwargs,
            )
        if cbar == True:
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="5%", pad=0.05)
            plt.colorbar(scanv, ax=ax, cax=cax)
        if show_axes == False:
            ax.axis("off")
        return fig, ax

    def view_stack_rois(
        self,
        labels=True,
        func=np.mean,
        axis=0,
        cbar=False,
        ax=None,
        figsize=(None, None),
        figsize_scale=None,
        zcrop: tuple = None,
        xcrop: tuple = None,
        ycrop: tuple = None,
        alpha=0.5,
        outline=True,
        outline_smooth=True,
        outline_width=2,
        roi_indices=None,
        **kwargs: Any,
    ):
        """
        Display a projection of the image stack using the specified function.

        Parameters:
        - func: Callable or str, optional, default: np.mean
            The function used to compute the projection along the specified axis.
            If "average_stack", uses self.average_stack directly.
            If pygor.core.methods.correlation_map, applies correlation mapping.
        - axis: int, optional, default: 0
            The axis along which the projection is computed.
        - cbar: bool, optional, default: False
            Whether to display a colorbar.
        - ax: matplotlib.axes.Axes, optional, default: None
            The matplotlib axes to use for the display. If None, the current axes will be used.
        - outline: bool, optional, default: False
            Whether to show ROIs as outlines instead of filled regions.
        - outline_smooth: bool, optional, default: True
            Whether to smooth the ROI outlines when outline=True.
        - outline_width: int, optional, default: 2
            Line width for ROI outlines when outline=True.
        - roi_indices: list or array-like, optional, default: None
            Indices of specific ROIs to display using 0-based indexing (0, 1, 2, etc.).
            If None, all ROIs are shown.

        Returns:
        tuple
            (fig, ax) - matplotlib figure and axes objects
        """
        if figsize == (None, None):
            figsize = (10, 10)
        if figsize_scale is not None:
            figsize = np.array(figsize) * np.array(figsize_scale)
        else:
            figsize_scale = 1
        if ax is None:
            fig, ax = plt.subplots(figsize=figsize)
        else:
            fig = plt.gcf()
        if "text_scale" in kwargs:
            txt_scl = kwargs["text_scale"]
        else:
            txt_scl = 1
        if zcrop is None:
            zstart = None
            zstop = None
        else:
            zstart = zcrop[0]
            zstop = zcrop[1]
        if xcrop is None:
            xstart = None
            xstop = None
        else:
            xstart = xcrop[0]
            xstop = xcrop[1]
        if ycrop is None:
            ystart = None
            ystop = None
        else:
            ystart = ycrop[0]
            ystop = ycrop[1]

        # Use rois_alt for 0-based indexing
        rois_to_use = self.rois_alt

        # num_rois = int(np.nanmax(rois_to_use)) + 1  # +1 because 0-based indexing
        color = matplotlib.colormaps["jet_r"]

        if func == "average_stack":
            scanv = ax.imshow(
                self.average_stack, cmap="Greys_r", origin="lower", **kwargs
            )
        elif func == pygor.core.methods.correlation_map:
            correlation_result = func(
                self.images[zstart:zstop, ystart:ystop, xstart:xstop]
            )
            scanv = ax.imshow(
                correlation_result, cmap="Greys_r", origin="lower", **kwargs
            )
        else:
            scanv = ax.imshow(
                func(self.images[zstart:zstop, ystart:ystop, xstart:xstop:], axis=axis),
                cmap="Greys_r",
                origin="lower",
                **kwargs,
            )
        if outline:
            # Extract ROI outlines instead of filled regions
            from skimage import measure

            # Get unique ROI values from rois_alt (0-based indexing, background is NaN)
            roi_values = np.unique(rois_to_use)
            roi_values = roi_values[~np.isnan(roi_values)].astype(
                int
            )  # Remove NaN (background)

            # Filter by specific ROI indices if provided
            if roi_indices is not None:
                roi_indices = np.array(roi_indices)
                roi_values = roi_values[np.isin(roi_values, roi_indices)]

            # Check if any ROIs remain after filtering
            if len(roi_values) == 0:
                print(f"Warning: No ROIs found with indices {roi_indices}")
                # Create empty ScalarMappable for consistency
                import matplotlib.cm as cm

                norm = matplotlib.colors.Normalize(vmin=0, vmax=1)
                rois = cm.ScalarMappable(norm=norm, cmap=color)
            else:
                # Apply cropping to ROIs
                rois_cropped = rois_to_use[ystart:ystop, xstart:xstop]

                # Plot each ROI outline individually
                for i, roi_val in enumerate(roi_values):
                    # Create binary mask for current ROI
                    roi_mask = (rois_cropped == roi_val).astype(int)

                    if np.sum(roi_mask) == 0:  # Skip if ROI not in cropped region
                        continue

                    # Find contours
                    contours = measure.find_contours(roi_mask, 0.5)

                    # Get color for this ROI
                    roi_color = color(i / len(roi_values))

                    # Plot each contour
                    for contour in contours:
                        if outline_smooth:
                            # Apply Gaussian smoothing to contour coordinates
                            from scipy.ndimage import gaussian_filter1d

                            contour[:, 0] = gaussian_filter1d(contour[:, 0], sigma=1.5)
                            contour[:, 1] = gaussian_filter1d(contour[:, 1], sigma=1.5)

                            # Close the contour by adding the first point to the end
                            contour = np.vstack([contour, contour[0]])

                        ax.plot(
                            contour[:, 1],
                            contour[:, 0],
                            color=roi_color,
                            linewidth=outline_width,
                            alpha=alpha,
                        )

                # Create dummy mappable for colorbar compatibility
                import matplotlib.cm as cm

                norm = matplotlib.colors.Normalize(
                    vmin=np.min(roi_values), vmax=np.max(roi_values)
                )
                rois = cm.ScalarMappable(norm=norm, cmap=color)
        else:
            # Original filled region display using rois_alt
            rois_display = rois_to_use.copy()

            # Filter by specific ROI indices if provided
            if roi_indices is not None:
                roi_indices = np.array(roi_indices)
                # Create mask for ROIs not in roi_indices (set them to NaN)
                mask = ~np.isin(rois_display, roi_indices) & ~np.isnan(rois_display)
                rois_display[mask] = np.nan

            rois_masked = np.ma.masked_where(np.isnan(rois_display), rois_display)[
                ystart:ystop, xstart:xstop
            ]
            rois = ax.imshow(
                rois_masked, cmap=color, alpha=alpha, origin="lower", **kwargs
            )
        ax.grid(False)
        ax.axis("off")
        if cbar == True:
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="5%", pad=0.05)
            plt.colorbar(rois, ax=ax, cax=cax)
        if labels == True:
            # Get ROI values from rois_alt (0-based indexing)
            label_map = np.unique(rois_to_use)
            label_map = label_map[~np.isnan(label_map)].astype(
                int
            )  # Remove NaN (background)
            label_map = np.sort(label_map)  # Sort in ascending order

            # Filter label_map by roi_indices if provided
            if roi_indices is not None:
                roi_indices = np.array(roi_indices)
                label_map = label_map[np.isin(label_map, roi_indices)]
            if "label_by" in kwargs:
                labels = self.__keyword_lables[kwargs["label_by"]]
                if np.isnan(labels) is True:
                    raise AttributeError(
                        f"Attribute {kwargs['label_by']} not found in object."
                    )
            else:
                labels = label_map  # Use 0-based indexing directly
            for label_loc, label in zip(label_map, labels):
                curr_roi_mask = rois_to_use == label_loc
                curr_roi_centroid = np.mean(np.argwhere(curr_roi_mask == 1), axis=0)
                ax.text(
                    curr_roi_centroid[1],
                    curr_roi_centroid[0],
                    label,
                    ma="center",
                    va="center",
                    ha="center",
                    c="w",
                    size=12 * np.array(figsize_scale) * txt_scl,
                    weight="normal",
                    path_effects=[
                        path_effects.Stroke(
                            linewidth=2 * np.array(figsize_scale) * txt_scl,
                            foreground="k",
                        ),
                        path_effects.Normal(),
                    ],
                )

        return fig, ax

    def view_drift(
        self, frame_num="auto", butterworth_factor=0.5, chan_vese_factor=0.01, ax=None
    ) -> None:
        """
        View drift of images over time.

        Parameters
        ----------
        frame_num : str, optional
            The number of frames to use for the drift calculation. If "auto", it will use approximately 1/3 of the total frames.
        butterworth_factor : float, optional
            The Butterworth filter factor.
        chan_vese_factor : float, optional
            The Chan-Vese segmentation factor.
        ax : None or axis object, optional
            The axis to plot the result on. If None, it will use the current axis.

        Returns
        -------
        None
        """
        if ax is None:
            ax = plt.gca()

        def _prep_img(image: np.array) -> np.ndarray:
            image = skimage.filters.butterworth(image, 0.5, high_pass=False)
            image = skimage.morphology.diameter_closing(
                image, diameter_threshold=image.shape[-1]
            )
            image = skimage.segmentation.chan_vese(image, 0.01).astype(int)
            return image

        # split array at 3 evenly spaced time points
        mid_split = math.floor(self.images.shape[0] / 2)
        # Split in 3
        if frame_num == "auto":
            frame_num = math.floor(self.images.shape[0] / 3)
        base = _prep_img(np.average(self.images[0:frame_num], axis=0))
        base1 = _prep_img(
            np.average(self.images[mid_split : mid_split + frame_num], axis=0)
        )
        base2 = _prep_img(np.average(self.images[-frame_num - 1 : -1], axis=0))
        #
        d3 = pygor.plotting.basic.stack_to_rgb(base)
        d3[:, :, 0] = base
        d3[:, :, 1] = base1
        d3[:, :, 2] = base2
        ax.imshow(pygor.utilities.min_max_norm(d3, 0, 1), origin="lower")

    def get_depth(self):
        """
        Get the depth of the images in the stack.

        Returns
        -------
        int
            The depth of the images in the stack.
        """
        session = pygor.core.gui.methods.NapariDepthPrompt(self)
        return session.run()

    def update_h5_key(self, key, value, overwrite=False):
        """
        Update a specific key in the H5 file with new data.

        Parameters
        ----------
        key : str
            The H5 dataset key to update (e.g., 'Positions' for ipl_depths)
        value : array-like
            The new value to store
        overwrite : bool, optional
            Whether to overwrite existing data (default: False)

        Returns
        -------
        bool
            True if update was successful, False otherwise
        """
        return pygor.core.methods.update_h5_key(self, key, value, overwrite)

    def update_ipl_depths(self, depths=None):
        """
        Update IPL depths on the in-memory object, optionally using interactive depth selection.

        Parameters
        ----------
        depths : array-like, optional
            Pre-calculated depths. If None, launches interactive depth selection.

        Returns
        -------
        bool
            True if update was successful, False otherwise
        """
        if depths is None:
            depths = self.get_depth()
            if depths is None:
                print("Depth calculation was cancelled or failed.")
                return False

        self.ipl_depths = depths
        print(f"Successfully updated ipl_depths for {len(depths)} ROIs")
        return True

    def estimate_ipl_depths(
        self,
        n_bins=8,
        upper_percentile=0.0,
        lower_percentile=100.0,
        orientation=None,
        plot=False,
    ):
        """Automatically estimate IPL depths from ROI positions without GUI.

        Uses percentile-based boundary estimation along the scan axis to
        approximate the outer (0 %) and inner (100 %) IPL boundaries from
        the spatial distribution of ROI centroids.

        Parameters
        ----------
        n_bins : int, optional
            Number of bins along the scan axis (default: 15).
        upper_percentile : float, optional
            Percentile for the outer (0 %) boundary (default: 5.0).
        lower_percentile : float, optional
            Percentile for the inner (100 %) boundary (default: 95.0).
        orientation : str or None, optional
            ``"horizontal"`` or ``"vertical"``. Auto-detected if None.
        plot : bool, optional
            Whether to show a diagnostic plot with image, boundaries,
            centroids, and depth KDE (default: True).

        Returns
        -------
        np.ndarray, shape (n_rois,)
            Estimated IPL depth percentages.
        """
        from pygor.anatomy.ipl import (
            calculate_ipl_depths,
            estimate_ipl_boundaries,
            plot_ipl_estimation,
        )

        upper, lower = estimate_ipl_boundaries(
            self.roi_centroids,
            n_bins=n_bins,
            upper_percentile=upper_percentile,
            lower_percentile=lower_percentile,
            orientation=orientation,
        )
        depths = calculate_ipl_depths(
            self.roi_centroids,
            upper,
            lower,
            orientation=orientation,
        )
        self.ipl_depths = depths
        if plot:
            mean_image = np.average(self.images, axis=0)
            plot_ipl_estimation(
                mean_image,
                self.roi_centroids,
                upper,
                lower,
                depths,
            )
        return depths

    def update_rois(self, roi_mask):
        """
        Update ROIs on the in-memory object.

        To persist changes to an H5 file, use export_to_h5().

        Parameters
        ----------
        roi_mask : array-like
            ROI mask to save (background=1, ROIs=-1,-2,...,-n)
        """
        self.rois = roi_mask
        self.num_rois = len(np.unique(roi_mask)) - 1
        print(f"Successfully updated object.rois: {self.num_rois} ROIs saved")

    def transfer_rois_from(
        self,
        source: "Core",
        *,
        max_shift: int = 20,
        upsample_factor: int = 10,
        projection_mode: str = "mean",
        plot: bool = False,
        overwrite: bool = True,
        extract_traces: bool = False,
    ) -> dict:
        """
        Transfer ROIs from another experiment to this one using image registration.

        This method computes the spatial offset between two recordings of the same
        field of view and applies that offset to transfer ROI masks from a source
        experiment (e.g., with good segmentation) to this experiment.

        Parameters
        ----------
        source : Core
            Source data object containing ROIs to transfer. Must have valid
            `rois` attribute (not None) and `average_stack` or `images`.
        max_shift : int, optional
            Maximum expected shift in pixels (default: 20). A warning is raised
            if the detected shift exceeds this value.
        upsample_factor : int, optional
            Subpixel precision factor for phase cross-correlation (default: 10).
            Higher values increase precision but slow computation.
        projection_mode : str, optional
            How to compute reference images for alignment (default: "mean").
            Options: "mean", "std", "correlation".
        plot : bool, optional
            If True, display a 4-panel comparison figure showing alignment quality
            (default: False).
        overwrite : bool, optional
            If True, save transferred ROIs to H5 file if available (default: True).
        extract_traces : bool, optional
            If True, automatically extract traces from transferred ROIs (default: False).

        Returns
        -------
        dict
            Transform information with keys:
            - 'shift': (dy, dx) shift in pixels
            - 'error': registration error metric (lower is better)
            - 'num_rois': number of ROIs transferred
            - 'source_name': name of source recording

        Raises
        ------
        ValueError
            If source has no ROIs or if image dimensions don't match.
        RuntimeError
            If source has no image data to use for alignment.

        Examples
        --------
        >>> # Transfer ROIs from STRF to OSDS recording
        >>> data_ref = STRF("strf_recording.smp")
        >>> data_ref.preprocess()
        >>> data_ref.segment_rois()
        >>>
        >>> data_dir = OSDS("osds_recording.smp")
        >>> data_dir.preprocess()
        >>> result = data_dir.transfer_rois_from(data_ref, plot=True)
        >>> print(f"Shifted by {result['shift']} pixels")

        See Also
        --------
        pygor.preproc.registration.transfer_rois : Low-level transfer function
        update_rois : Update ROIs with a pre-defined mask
        segment_rois : Automated ROI segmentation
        """
        # Import here to avoid circular imports
        from pygor.preproc.registration import transfer_rois

        # Validate source has ROIs
        if source.rois is None:
            raise ValueError(
                f"Source '{source.name}' has no ROIs. "
                "Run segment_rois() or draw_rois() on source first."
            )

        # Get projections for alignment
        source_proj = self._get_projection_for_alignment(source, projection_mode)
        target_proj = self._get_projection_for_alignment(self, projection_mode)

        # Validate dimensions match
        if source_proj.shape != target_proj.shape:
            raise ValueError(
                f"Image dimensions don't match: source {source_proj.shape} vs "
                f"target {target_proj.shape}. Recordings must have the same field of view size."
            )

        # Call existing transfer_rois function
        shifted_mask, transform = transfer_rois(
            roi_mask=source.rois,
            ref_projection=source_proj,
            target_projection=target_proj,
            max_shift=max_shift,
            upsample_factor=upsample_factor,
        )

        # Update self with transferred ROIs
        self.update_rois(shifted_mask)

        # Optionally extract traces
        if extract_traces and self.images is not None:
            self.extract_traces_from_rois()

        # Build result dict
        result = {
            "shift": transform["shift"],
            "error": transform["error"],
            "num_rois": self.num_rois,
            "source_name": source.name,
        }

        # Plot if requested
        if plot:
            self._plot_roi_transfer(source, shifted_mask, transform, projection_mode)

        # Print summary
        print(
            f"ROI Transfer complete.\n"
            f"  Source: {source.name}\n"
            f"  Shift: (y={result['shift'][0]:.2f}, x={result['shift'][1]:.2f}) pixels\n"
            f"  Registration error: {result['error']:.4f}\n"
            f"  ROIs transferred: {result['num_rois']}"
        )

        return result

    def _get_projection_for_alignment(self, data_obj: "Core", mode: str) -> np.ndarray:
        """Get appropriate projection image for cross-experiment alignment."""
        if mode == "mean":
            if data_obj.average_stack is not None:
                return data_obj.average_stack
            elif data_obj.images is not None:
                return data_obj.images.mean(axis=0)
            else:
                raise RuntimeError(
                    f"'{data_obj.name}' has no image data for alignment. "
                    "Load images or compute average_stack first."
                )
        elif mode == "std":
            if data_obj.images is not None:
                return data_obj.images.std(axis=0)
            else:
                raise RuntimeError(
                    f"'{data_obj.name}' has no image stack. Cannot compute std projection."
                )
        elif mode == "correlation":
            if (
                hasattr(data_obj, "correlation_projection")
                and data_obj.correlation_projection is not None
            ):
                return data_obj.correlation_projection
            else:
                raise RuntimeError(
                    f"'{data_obj.name}' has no correlation projection. "
                    "Call compute_correlation_projection() first."
                )
        else:
            raise ValueError(
                f"Unknown projection_mode: '{mode}'. Options: 'mean', 'std', 'correlation'"
            )

    def _plot_roi_transfer(
        self,
        source: "Core",
        shifted_mask: np.ndarray,
        transform: dict,
        projection_mode: str,
    ) -> None:
        """Plot ROI transfer results with before/after comparison."""
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))

        # Get projections
        source_proj = self._get_projection_for_alignment(source, projection_mode)
        target_proj = self._get_projection_for_alignment(self, projection_mode)

        # Compute pearsons correlation for two projections
        corr = np.corrcoef(source_proj.flatten(), target_proj.flatten())[0, 1]

        # Shared colormap limits
        vmin = min(source_proj.min(), target_proj.min())
        vmax = max(source_proj.max(), target_proj.max())

        # Top-left: Source with original ROIs
        ax = axes[0, 0]
        ax.imshow(source_proj, cmap="gray", vmin=vmin, vmax=vmax, origin="lower")
        self._overlay_rois_on_axis(ax, source.rois, alpha=0.3)
        ax.set_title(f"Source: {source.name}\n(original ROIs)")
        ax.axis("off")

        # Top-right: Target projection (for reference)
        ax = axes[0, 1]
        ax.imshow(target_proj, cmap="gray", vmin=vmin, vmax=vmax, origin="lower")
        ax.set_title(f"Target: {self.name}\n({projection_mode} projection)")
        ax.axis("off")

        # Bottom-left: Overlay showing alignment (red-cyan composite)
        ax = axes[1, 0]
        # Create RGB overlay: source=red, target=cyan
        rgb = np.zeros((*source_proj.shape, 3))
        source_norm = (source_proj - vmin) / (vmax - vmin + 1e-10)
        target_norm = (target_proj - vmin) / (vmax - vmin + 1e-10)
        rgb[:, :, 0] = source_norm  # Red channel = source
        rgb[:, :, 1] = target_norm  # Green channel = target
        rgb[:, :, 2] = target_norm  # Blue channel = target (makes cyan)
        ax.imshow(np.clip(rgb, 0, 1), origin="lower")
        shift = transform["shift"]
        ax.set_title(
            f"Alignment overlay (R=source, C=target)\n"
            f"Shift to apply: dy={shift[0]:.2f}, dx={shift[1]:.2f} px"
        )
        ax.axis("off")

        # Bottom-right: Target with transferred ROIs
        ax = axes[1, 1]
        ax.imshow(target_proj, cmap="gray", vmin=vmin, vmax=vmax, origin="lower")
        self._overlay_rois_on_axis(ax, shifted_mask, alpha=0.3)
        ax.set_title(
            f"Target with transferred ROIs\n"
            f"({self.num_rois} ROIs, error={transform['error']:.4f}, corr={corr:.4f})"
        )
        ax.axis("off")

        plt.tight_layout()
        plt.show()

    def _overlay_rois_on_axis(self, ax, roi_mask: np.ndarray, alpha: float = 0.3):
        """Overlay ROI mask on axes with transparent coloring."""
        # Get unique ROI values (excluding background=1)
        unique_rois = np.unique(roi_mask)
        unique_rois = unique_rois[unique_rois < 0]  # Only negative values (ROIs)

        if len(unique_rois) == 0:
            return

        # Create colored overlay
        colored = np.zeros((*roi_mask.shape, 4))  # RGBA
        cmap = matplotlib.colormaps["jet"]

        for i, roi_id in enumerate(unique_rois):
            color = cmap(i / max(len(unique_rois), 1))
            mask = roi_mask == roi_id
            colored[mask] = (*color[:3], alpha)

        ax.imshow(colored, origin="lower")

    def segment_rois(self, mode=None, overwrite=True, **kwargs: Any) -> np.ndarray:
        """
        Segment ROIs using automated methods.

        Default parameters for each mode are loaded from the config system
        (``[segmentation.blob]``, ``[segmentation.watershed]``, etc. in
        defaults.toml or user config). Pass kwargs to override any parameter.
        To change defaults permanently, edit your config TOML file.

        Parameters
        ----------
        mode : str or None
            Segmentation mode. If None, uses default from config
            (``[segmentation.general] mode``, defaults to "blob").
            Available options:
            - "cellpose+": Cellpose with post-processing heuristics (requires model)
            - "cellpose": Raw Cellpose output only (requires model)
            - "blob": Difference of Gaussian blob detection (no ML required)
            - "watershed": Watershed segmentation (no ML required)
            - "flood_fill": IGOR-style region growing (no ML required)
        overwrite : bool
            If True, overwrite existing ROIs. Default: True
        verbose : bool
            Print progress messages. Default: True
        plot : bool
            Show segmentation result overlaid on input image. Default: False
            Displays 3-panel figure: input image, ROI masks, and overlay.
            Works for all segmentation modes.
        roi_order : str or None
            Spatial ordering for ROI numbering. Default: "LR"
            - "LR": Left-to-right (x primary, y as tiebreaker)
            - "TB": Top-to-bottom (y primary, x as tiebreaker)
            - None: Original detection order

        Cellpose Parameters (mode="cellpose" or "cellpose+")
        ----------------------------------------------------
        model_path : str or Path
            Direct path to a trained Cellpose model file
        model_dir : str or Path
            Directory to search for trained models
        diameter : float
            Expected cell diameter in pixels. None for auto-detect
        flow_threshold : float
            Flow error threshold (default: 0.9)
        cellprob_threshold : float
            Cell probability threshold (default: 0.5)
        min_size : int
            Minimum ROI size in pixels (default: 2)

        Post-processing Parameters (mode="cellpose+" only)
        --------------------------------------------------
        split_large : bool
            Split large ROIs using watershed (default: True)
        size_multiplier : float
            ROIs larger than median * this are split (default: 1.5)
        min_peak_distance : int
            Min distance between peaks when splitting (default: 1)
        min_size_after_split : int
            Remove split fragments smaller than this (default: 4)
        shrink_iterations : int
            Erosion iterations to shrink ROIs (default: 1, 0 to disable)
        shrink_size_threshold : int
            Only shrink ROIs larger than this (default: 30)

        Lightweight Parameters (mode="watershed", "flood_fill", "blob")
        ---------------------------------------------------------------
        input_mode : str
            Which image representation to use:
            - "combined": correlation * average (default, recommended)
            - "correlation": correlation projection only
            - "average": mean image only
            - "std": standard deviation image
            Note: "correlation" and "combined" require compute_correlation_projection() first

        Shared Preprocessing (mode="watershed", "flood_fill", "blob")
        -------------------------------------------------------------
        Image enhancement (optional, enabled by default for blob):
            unsharp_radius : float
                Radius for unsharp masking (default: None for watershed/flood_fill, 1.0 for blob)
            unsharp_amount : float
                Strength of sharpening (default: None for watershed/flood_fill, 2.5 for blob)
                Both must be set to enable enhancement.

        Anatomy masking (excludes background/border regions):
            anatomy_threshold : str or float
                'otsu' for automatic, float for manual (default: None for watershed/flood_fill, 'otsu' for blob)
                Set to 'otsu' to enable automatic anatomy masking.
            anatomy_thresh_mult : float
                Multiply threshold by this, lower = more permissive (default: 0.2)
            erode_iterations : int
                Erode mask to exclude edge regions (default: 1, 0 to disable)

        Watershed Parameters (mode="watershed")
        ---------------------------------------
        threshold : float
            Intensity threshold for foreground (default: 0.1)
        min_distance : int
            Minimum distance between seed peaks (default: 1)
        gap_pixels : int
            Erosion iterations to create gaps between ROIs (default: 2)
        min_size_to_shrink : int
            ROIs smaller than this won't be eroded (default: 10)
        min_roi_size : int
            Remove ROIs smaller than this (default: 3)

        Flood Fill Parameters (mode="flood_fill")
        -----------------------------------------
        threshold : float
            Intensity threshold for foreground (default: 0.15)
        min_distance : int
            Minimum distance between seed peaks (default: 1)
        max_size : int
            Maximum pixels per ROI (default: 20)
        drop_fraction : float
            Stop growing when intensity < peak * this (default: 0.2)
        min_gap : int
            Minimum gap from other ROIs in pixels (default: 0)
        min_roi_size : int
            Remove ROIs smaller than this (default: 3)

        Blob Detection Parameters (mode="blob")
        ---------------------------------------
        Image enhancement:
            unsharp_radius : float
                Radius for unsharp masking (default: 1.0, range: 0.5-2.0)
            unsharp_amount : float
                Strength of sharpening (default: 2.5, range: 1.0-5.0)

        Blob detection:
            min_sigma : float
                Minimum sigma for DoG, controls min blob size (default: 1)
            max_sigma : float
                Maximum sigma for DoG, controls max blob size (default: 2)
            threshold : float
                Detection threshold, lower = more blobs (default: 0.01)
            eliminate_overlap : float
                Overlap fraction before DoG merges (default: 1.0 = no merging)
            merge_overlap : float
                Merge if overlap > this fraction of smaller blob (default: 0.6)

        Anatomy masking (exclude border regions):
            anatomy_threshold : str or float
                'otsu' for automatic, or float for manual (default: 'otsu')
            anatomy_thresh_mult : float
                Multiply threshold by this, lower = more permissive (default: 0.2)
            erode_iterations : int
                Erode mask to exclude edge blobs (default: 1, 0 to disable)

        Mask creation:
            radius_multiplier : float
                Scale factor for blob radius (default: 1.5)
            min_radius : float
                Minimum radius in pixels (default: 1)

        Returns
        -------
        masks : ndarray
            ROI mask in pygor format (background=1, ROIs=-1,-2,-3...)

        Examples
        --------
        >>> # Cellpose with trained model
        >>> data.segment_rois(mode="cellpose+", model_dir="./models/synaptic")

        >>> # Blob detection (no ML) - good for synaptic terminals
        >>> data.compute_correlation_projection()
        >>> data.segment_rois(mode="blob", input_mode="combined", plot=True)

        >>> # Blob with custom parameters
        >>> data.segment_rois(mode="blob", min_sigma=1.5, max_sigma=3, threshold=0.02, plot=True)

        >>> # Watershed segmentation
        >>> data.segment_rois(mode="watershed", input_mode="average", threshold=0.15)

        >>> # Enable anatomy masking for watershed/flood_fill (excludes background)
        >>> data.segment_rois(mode="watershed", anatomy_threshold='otsu')

        >>> # Enable image enhancement for weak signals
        >>> data.segment_rois(mode="flood_fill", unsharp_radius=1.0, unsharp_amount=2.5)

        >>> # Using different image representations
        >>> data.segment_rois(mode="blob", input_mode="correlation")  # correlation only
        >>> data.segment_rois(mode="blob", input_mode="std")  # standard deviation
        """
        from pygor.segmentation import segment_rois as _segment_rois

        # Load default mode and roi_order from config if not specified
        if mode is None or "roi_order" not in kwargs:
            try:
                from pygor.config import get_defaults

                general = get_defaults("segmentation.general")
            except (KeyError, AttributeError, ImportError):
                general = {}
            if mode is None:
                mode = general.get("mode", "blob")
            if "roi_order" not in kwargs:
                roi_order = general.get("roi_order")
                if roi_order is not None:
                    kwargs["roi_order"] = roi_order

        roi_mask = _segment_rois(self, mode=mode, overwrite=overwrite, **kwargs)
        self.update_rois(roi_mask)

        # Record segmentation in params
        self.params.mark_segmentation({"mode": mode, **kwargs})

        return roi_mask

    def view_images_interactive(self, **kwargs: Any) -> None:
        """
        View the image stack interactively using Napari.

        Parameters:
        -----------
        **kwargs : dict
            Additional keyword arguments passed to Napari viewer
        """
        session = pygor.core.gui.methods.NapariViewStack(self, **kwargs)
        session.run()

    def draw_rois(
        self,
        attribute="calculate_image_average",
        style="stacked",
        load_existing_rois=True,
        overwrite=True,
        show_correlation=False,
        **kwargs: Any,
    ) -> None:
        """
        Draw ROIs on the image stack.

        Parameters:
        -----------
        attribute : str
            Method or attribute to use for image data
        style : str
            Trace plotting style ('stacked', 'individual', or 'raster')
        load_existing_rois : bool
            If True and self.rois exists, loads existing ROIs as editable shapes (default: True)
        overwrite : bool
            If True, saves ROIs to H5 file and overwrites existing data (default: False)
        show_correlation : bool
            If True, displays correlation projection in Napari viewer (default: True)
        **kwargs : dict
            Additional keyword arguments passed to NapariRoiPrompt
        """

        def call_method(obj, method_str, *args, **kwargs: Any):
            # Extract method name by stripping trailing parentheses (if present)
            method_name = method_str.split("(")[
                0
            ].strip()  # Handles "method" or "method()"
            method = getattr(obj, method_name)  # Get the method from the object
            print(method)
            if attribute in obj.__dict__:
                return obj.__getattribute__(attribute)
            else:
                return method(*args, **kwargs)  # Call the method with arguments

        target = call_method(self, attribute)

        # If target is None (e.g., no repetitions), fall back to raw images
        if target is None:
            print(
                "No averaged data available (likely no repetitions). Using raw images instead."
            )
            target = self.images

        # Check if existing ROIs should be loaded
        existing_roi_mask = None
        if load_existing_rois and hasattr(self, "rois") and self.rois is not None:
            existing_roi_mask = self.rois
            print("Loading existing ROIs from self.rois")

        # Compute correlation projection if requested and not already available
        correlation_projection = None
        if show_correlation:
            if self.correlation_projection is None:
                print("Computing correlation projection...")
                correlation_projection = self.compute_correlation_projection()
            else:
                correlation_projection = self.correlation_projection

        session = pygor.core.gui.methods.NapariRoiPrompt(
            target,
            traces_plot_style=style,
            existing_roi_mask=existing_roi_mask,
            correlation_projection=correlation_projection,
            **kwargs,
        )
        traces = session.run()

        # Save ROI mask if overwrite is True
        if overwrite:
            # Check if user actually modified ROIs in Napari
            if (
                hasattr(session, "rois_were_modified")
                and not session.rois_were_modified
            ):
                # ROIs were not modified - use original mask to prevent growth
                print("No changes detected - ROIs not overwritten")
            else:
                # ROIs were modified - convert and save
                napari_mask = session.mask
                igor_style_mask = session.convert_napari_mask_to_igor_format(
                    napari_mask
                )
                # Postpone saving to H5 until user executes save operation, but update object attributes

                # success = self.update_h5_key('ROIs', h5_mask, overwrite=True)
                # if success:
                self.rois = igor_style_mask
                self.num_rois = len(
                    np.unique(igor_style_mask)[np.unique(igor_style_mask) < 0]
                )
                print(f"Successfully updated {self.num_rois} ROIs in memory")

                # Recompute dependent data since ROIs changed
                print("\nRecomputing traces, snippets, and averages for new ROIs...")

                # Compute both raw and z-normalized traces
                self.extract_traces_from_rois()

                # Compute snippets and averages
                self.compute_snippets_and_averages()

                # Verify shapes match
                print("\nVerifying data integrity:")
                print(f"  num_rois: {self.num_rois}")
                print(
                    f"  traces_raw shape: {self.traces_raw.shape if self.traces_raw is not None else 'None'}"
                )
                print(
                    f"  averages shape: {self.averages.shape if self.averages is not None else 'None'}"
                )
                print(
                    f"  snippets shape: {self.snippets.shape if self.snippets is not None else 'None'}"
                )

                print("All dependent data recomputed and saved successfully")
            # else:
            #     print("Failed to save ROIs to H5 file")

        # Plot traces if requested
        if kwargs.get("plot", False):
            print("\nGenerating traces plot...")

            if overwrite and self.traces_znorm is not None:
                # Saved mode: plot from computed traces
                self._plot_traces(style=style, session=session)
            elif hasattr(session, "mask") and session.mask is not None:
                # Preview mode: plot from temporary mask
                target_images = self.images if target is self.images else target
                self._plot_traces(
                    style=style,
                    session=session,
                    roi_mask=session.mask,
                    images=target_images,
                )
            else:
                print("No ROI data available for plotting")

        return traces

    def _plot_traces(self, style="stacked", session=None, roi_mask=None, images=None):
        """
        Plot z-normalized traces using matplotlib.

        Can operate in two modes:
        1. Preview mode: Pass roi_mask and images to compute traces on-the-fly
        2. Saved mode: Use self.traces_znorm (already computed and saved)

        Parameters:
        -----------
        style : str
            Plotting style ('stacked', 'individual', or 'raster')
        session : NapariRoiPrompt, optional
            Napari session object for accessing ROI visualization data
        roi_mask : np.ndarray, optional
            ROI mask for preview mode (if provided with images, computes traces on-the-fly)
        images : np.ndarray, optional
            Image stack for preview mode (if provided with roi_mask, computes traces on-the-fly)
        """
        import matplotlib.pyplot as plt

        import pygor.core.gui.methods

        # Determine mode and prepare data
        if roi_mask is not None and images is not None:
            # PREVIEW MODE: Compute traces on-the-fly
            print("Preview mode: computing traces on-the-fly...")

            # Validate ROI mask has valid ROIs
            unique_vals = np.unique(roi_mask)
            unique_vals = unique_vals[~np.isnan(unique_vals)]

            # Handle both H5 format (-1,-2,-3...) and Napari format (0,1,2...)
            if np.any(unique_vals < 0):
                valid_rois = unique_vals[unique_vals < 0]
            else:
                valid_rois = unique_vals[unique_vals >= 0]

            if len(valid_rois) == 0:
                print("No valid ROIs found - skipping plot")
                return

            # Extract traces using vectorized method
            from pygor.core.trace_extraction import extract_traces

            traces_raw = extract_traces(images, roi_mask)
            # Returns (n_rois, n_frames)

            n_rois, n_frames = traces_raw.shape

            # Compute z-scores using same baseline logic as extract_traces_from_rois
            traces_znorm = np.zeros_like(traces_raw)

            # Get baseline parameters
            try:
                ignore_first_seconds = self.os_parameters["Ignore1stXseconds"]
                baseline_seconds = self.os_parameters["Baseline_nSeconds"]
                line_duration = self.os_parameters["LineDuration"]
                n_lines = images.shape[1]  # nY from image dimensions
            except (AttributeError, KeyError, TypeError):
                # Fallback if OS_Parameters not available
                print(
                    "Warning: OS_Parameters not available, using default baseline settings"
                )
                ignore_first_seconds = 0
                baseline_seconds = 2
                line_duration = 0.002  # 2ms default
                n_lines = images.shape[1]

            frame_duration = n_lines * line_duration
            n_frames_ignore = int(ignore_first_seconds / frame_duration)
            n_frames_baseline = int(baseline_seconds / frame_duration)

            # Ensure baseline window is valid
            if n_frames_baseline < 3:
                n_frames_baseline = 3
            if n_frames_ignore + n_frames_baseline > n_frames:
                n_frames_ignore = 0
                n_frames_baseline = min(n_frames // 2, 10)

            baseline_start = n_frames_ignore
            baseline_end = baseline_start + n_frames_baseline

            # Compute z-score for each ROI
            for roi_idx in range(n_rois):
                trace = traces_raw[roi_idx, :]  # Shape (n_rois, n_frames), get one ROI
                baseline = trace[baseline_start:baseline_end]
                baseline_mean = np.mean(baseline)
                baseline_std = np.std(baseline)

                if baseline_std > 0:
                    traces_znorm[roi_idx, :] = (trace - baseline_mean) / baseline_std
                else:
                    traces_znorm[roi_idx, :] = 0

            # Prepare for plotting (already in correct format)
            traces_plot = traces_znorm  # Already (n_rois, n_frames)
            avg_img = np.mean(images, axis=0)

        else:
            # SAVED MODE: Use existing computed traces
            if self.traces_znorm is None:
                print(
                    "No traces available - call extract_traces_from_rois first or provide roi_mask and images for preview"
                )
                return

            # Check if traces shape matches current ROI count (staleness check)
            if self.rois is not None:
                current_roi_count = len(np.unique(self.rois)[np.unique(self.rois) < 0])
                trace_roi_count = self.traces_znorm.shape[
                    0
                ]  # First dimension is n_rois
                if trace_roi_count != current_roi_count:
                    print(
                        f"WARNING: Trace count ({trace_roi_count}) doesn't match ROI count ({current_roi_count})."
                    )
                    print(
                        "Traces may be stale from H5 file. Restart kernel or call extract_traces_from_rois()."
                    )

            traces_plot = (
                self.traces_znorm
            )  # Already (n_rois, n_frames) from IGOR convention
            avg_img = np.mean(self.images, axis=0)

        # Create plot
        fig, ax = plt.subplots(2, 1, figsize=(10, 4))
        colormap = plt.cm.rainbow(np.linspace(0, 1, len(traces_plot)))

        # Top panel: Show average image with ROI overlay
        ax[0].imshow(avg_img, cmap="Greys_r", origin="lower")
        if session is not None and hasattr(session, "mask"):
            ax[0].imshow(session.mask, cmap="rainbow", alpha=0.25, origin="lower")
        elif roi_mask is not None:
            ax[0].imshow(roi_mask, cmap="rainbow", alpha=0.25, origin="lower")
        ax[0].set_title("ROIs")
        ax[0].axis("off")

        # Bottom panel: Plot z-normalized traces
        if style == "stacked":
            for n, trace in enumerate(traces_plot):
                ax[1].plot(trace, color=colormap[-n], alpha=0.7, linewidth=0.5)
            ax[1].set_ylabel("Z-score", fontsize=10)
            ax[1].set_xlabel("Frame", fontsize=10)
            ax[1].set_title("Z-normalized traces (baseline corrected)")

        elif style == "raster":
            im = ax[1].imshow(
                traces_plot,
                aspect="auto",
                cmap="RdBu_r",
                interpolation="none",
                origin="lower",
            )
            ax[1].set_ylabel("ROI #", fontsize=10)
            ax[1].set_xlabel("Frame", fontsize=10)
            ax[1].set_title("Z-normalized traces (baseline corrected)")
            plt.colorbar(im, ax=ax[1], label="Z-score")

        plt.tight_layout()
        plt.show()

    def compute_correlation_projection(
        self,
        include_diagonals: bool = True,
        n_jobs: int = -1,
        timecompress: int = 1,
        binpix: int = 1,
        overwrite: bool = False,
        force: bool = False,
    ) -> np.ndarray:
        """
        Compute pixel-wise temporal correlation with neighboring pixels.

        Creates a correlation map useful for visualizing functional connectivity
        in 2-photon calcium imaging data. Each pixel's correlation is computed as
        the average correlation coefficient with its immediate neighbors.

        Parameters
        ----------
        include_diagonals : bool, optional
            If True, uses 8-neighbor connectivity (including diagonals).
            If False, uses 4-neighbor connectivity (cardinal directions only).
            Default: True
        n_jobs : int, optional
            Number of parallel jobs to run. -1 uses all available CPUs.
            Default: -1
        timecompress : int, optional
            Temporal downsampling factor. Takes every Nth frame before computing
            correlations. This reduces noise and speeds up computation.
            Default: 1 (no downsampling)
        binpix : int, optional
            Spatial binning factor. Pixels are averaged in binpix x binpix
            blocks before computing correlations. The result is expanded back to
            original resolution. Dramatically speeds up computation.
            Default: 1 (no binning)
        overwrite : bool, optional
            If True, saves the result to H5 file, overwriting existing data.
            Default: False
        force : bool, optional
            If True, recomputes even if correlation_projection already exists.
            Default: False

        Returns
        -------
        np.ndarray
            Correlation projection with shape (height, width).
            Values range from -1 to 1, representing average correlation with neighbors.

        See Also
        --------
        pygor.core.calculations.compute_correlation_projection : Standalone function
        """
        # Check if already computed and not forcing recompute
        if not force and not overwrite and self.correlation_projection is not None:
            print(
                "Correlation projection already exists. Use force=True or overwrite=True to recompute."
            )
            return self.correlation_projection

        if self.images is None:
            raise ValueError(
                "No image data available. Cannot compute correlation projection."
            )

        # Call standalone function
        correlation_projection = pygor.core.calculations.compute_correlation_projection(
            images=self.images,
            include_diagonals=include_diagonals,
            n_jobs=n_jobs,
            timecompress=timecompress,
            binpix=binpix,
        )

        # Save to object attribute
        self.correlation_projection = correlation_projection

        # Record step in params
        self.params.mark_step(
            "correlation_projection",
            {
                "include_diagonals": include_diagonals,
                "timecompress": timecompress,
                "binpix": binpix,
            },
        )

        return correlation_projection

    def _compute_baseline_window(
        self, baseline_duration_s: float | None = None, buffer_s: float = 1.0
    ):
        """
        Compute baseline window from pre-stimulus period.

        Works backwards from the first trigger to find a clean baseline period.
        This avoids initial recording artifacts that can skew z-normalization.

        Parameters
        ----------
        baseline_duration_s : float or None, optional
            Duration of baseline window in seconds. If None (default), uses the
            entire pre-stimulus period minus a buffer at the start.
        buffer_s : float, optional
            Buffer in seconds to skip at recording start to avoid artifacts (default: 1.0).
            Only used when baseline_duration_s is None.

        Returns
        -------
        tuple
            (baseline_start_frame, baseline_end_frame) where end is exclusive
        """
        # Check for manual override first
        if hasattr(self, "_baseline_window") and self._baseline_window is not None:
            return self._baseline_window

        # Fallback if no triggers available
        if self.triggertimes_frame is None or len(self.triggertimes_frame) == 0:
            warnings.warn("No triggers found, using first 2s as baseline")
            n_frames = min(int(2.0 * self.frame_hz), self.images.shape[0] // 4)
            return 0, max(3, n_frames)

        first_trigger_frame = int(self.triggertimes_frame[0])
        baseline_end = first_trigger_frame  # exclusive

        if baseline_duration_s is None:
            # Use entire pre-stimulus period minus buffer at start
            buffer_frames = int(buffer_s * self.frame_hz)
            baseline_start = buffer_frames
        else:
            # Work backwards from first trigger for specified duration
            baseline_frames = int(baseline_duration_s * self.frame_hz)
            baseline_start = max(0, baseline_end - baseline_frames)

        # Validate minimum window size (need at least 3 frames for std)
        actual_frames = baseline_end - baseline_start
        if actual_frames < 3:
            warnings.warn(
                f"First trigger at frame {first_trigger_frame} too early for "
                f"requested baseline. Using frames 0-{first_trigger_frame} "
                f"({actual_frames} frames)"
            )
            baseline_start = 0
            baseline_end = max(3, first_trigger_frame)

        return baseline_start, baseline_end

    def set_baseline_window(self, start_frame=None, end_frame=None, duration_s=None):
        """
        Manually set baseline window (overrides automatic pre-stimulus detection).

        Use this to override the automatic baseline calculation if needed.
        Call before extract_traces_from_rois() to take effect.

        Parameters
        ----------
        start_frame : int, optional
            Start frame of baseline window (inclusive)
        end_frame : int, optional
            End frame of baseline window (exclusive)
        duration_s : float, optional
            If provided with only end_frame, calculates start_frame as
            end_frame - (duration_s * frame_hz)

        Examples
        --------
        >>> # Set explicit window
        >>> data.set_baseline_window(start_frame=100, end_frame=150)
        >>>
        >>> # Set window by end frame and duration
        >>> data.set_baseline_window(end_frame=160, duration_s=2.0)
        >>>
        >>> # Clear manual override (revert to automatic)
        >>> data._baseline_window = None
        """
        if start_frame is not None and end_frame is not None:
            self._baseline_window = (int(start_frame), int(end_frame))
        elif end_frame is not None and duration_s is not None:
            start = max(0, int(end_frame - duration_s * self.frame_hz))
            self._baseline_window = (start, int(end_frame))
        else:
            raise ValueError(
                "Provide (start_frame, end_frame) or (end_frame, duration_s)"
            )

        duration = (self._baseline_window[1] - self._baseline_window[0]) / self.frame_hz
        print(
            f"Baseline window set: frames {self._baseline_window[0]}-{self._baseline_window[1]} "
            f"({duration:.2f}s)"
        )

    @property
    def baseline_info(self):
        """
        Get information about the baseline window used for z-normalization.

        Returns
        -------
        dict or None
            Dictionary with baseline window info, or None if not yet computed.
        """
        return getattr(self, "_baseline_used", None)

    def extract_traces_from_rois(
        self, baseline_dur: int | float | None = 10
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Compute ROI traces from images and ROI mask.

        baseline_dur: Optional duration in seconds for baseline window. If None, uses automatic pre-stimulus detection.

        Extracts the average fluorescence signal for each ROI across all frames.
        Always computes BOTH raw and z-normalized traces to ensure consistency.
        Mimics IGOR's OS_TracesAndTriggers functionality for trace extraction.
        Uses vectorized NumPy operations for speed.

        Returns:
        --------
        tuple
            (traces_raw, traces_znorm) both with shape (n_rois, n_frames)
        """
        if self.images is None:
            raise ValueError("No image data available. Cannot compute traces.")
        if self.rois is None:
            raise ValueError("No ROIs defined. Cannot compute traces.")

        from pygor.core.trace_extraction import extract_traces, znorm_traces

        print("Extracting traces...")

        # Extract raw traces using vectorized method
        # Returns shape (n_rois, n_frames)
        traces_raw = extract_traces(self.images, self.rois)

        n_rois, n_frames = traces_raw.shape

        # Compute baseline window from backwards pre-stimulus period (before first trigger)
        baseline_start, baseline_end = self._compute_baseline_window(
            baseline_duration_s=baseline_dur
        )

        baseline_frames = baseline_end - baseline_start
        baseline_duration = baseline_frames / self.frame_hz
        method = (
            "manual"
            if hasattr(self, "_baseline_window") and self._baseline_window
            else "pre-stimulus"
        )

        print(
            f"Using {method} baseline: frames {baseline_start}-{baseline_end} "
            f"({baseline_frames} frames, {baseline_duration:.2f}s)"
        )

        # Compute z-normalized traces using vectorized function
        traces_znorm = znorm_traces(traces_raw, baseline_start, baseline_end)

        # Store baseline info for transparency
        self._baseline_used = {
            "start_frame": baseline_start,
            "end_frame": baseline_end,
            "start_ms": baseline_start / self.frame_hz * 1000,
            "end_ms": baseline_end / self.frame_hz * 1000,
            "n_frames": baseline_frames,
            "duration_s": baseline_duration,
            "method": method,
        }

        # Set both attributes
        self.traces_raw = traces_raw
        self.traces_znorm = traces_znorm

        # Record step in params
        self.params.mark_step(
            "trace_extraction",
            {
                "n_rois": n_rois,
                "n_frames": n_frames,
                "baseline_start": baseline_start,
                "baseline_end": baseline_end,
                "baseline_method": method,
            },
        )

        print(f"Extracted {n_rois} traces ({n_frames} frames each)")

        return self.traces_raw, self.traces_znorm

    def deconvolve_traces(
        self,
        rise_tau_ms: float | None = None,
        decay_tau_ms: float | None = None,
        lambd: float | None = None,
        kernel_window_ms: float | None = None,
        use_znorm: bool = True,
        verbose: bool = False,
    ) -> np.ndarray:
        """Remove calcium indicator blur from traces via Wiener deconvolution.

        Builds a causal calcium kernel (center-point sampled at the frame rate)
        and applies regularized Wiener deconvolution to all ROIs in a single
        vectorized FFT pass. The result is stored as ``self.traces_deconvolved``.

        Parameters fall back to ``[deconvolution]`` in defaults.toml when not
        provided. The shipped defaults target jGCaMP8f (Zhang et al. 2023).

        Parameters
        ----------
        rise_tau_ms : float or None
            Indicator rise time constant in ms. Default from config: 2.5.
        decay_tau_ms : float or None
            Indicator decay time constant in ms. Default from config: 75.0.
        lambd : float or None
            Wiener regularization. Larger = more smoothing. Default from
            config: 3e-3.
        kernel_window_ms : float or None
            Kernel support duration in ms. Default from config: 800.0.
        use_znorm : bool, default True
            If True, deconvolve ``self.traces_znorm``. If False, use
            ``self.traces_raw``.
        verbose : bool, default False
            Print kernel and timing info.

        Returns
        -------
        np.ndarray, shape (n_rois, n_frames)
            Deconvolved traces. Also stored as ``self.traces_deconvolved``.

        Examples
        --------
        >>> obj.deconvolve_traces()
        >>> # For STRF objects, pass directly:
        >>> rf.deconvolve_traces()
        >>> rf.calculate_strf(noise_array, traces=rf.traces_deconvolved)
        """
        from pygor.strf.deconvolution import calcium_kernel, wiener_deconvolve

        # Fall back to defaults.toml [deconvolution] section
        defaults = self.params.get_defaults("deconvolution")
        if rise_tau_ms is None:
            rise_tau_ms = defaults.get("rise_tau_ms", 2.5)
        if decay_tau_ms is None:
            decay_tau_ms = defaults.get("decay_tau_ms", 75.0)
        if lambd is None:
            lambd = defaults.get("lambd", 3e-3)
        if kernel_window_ms is None:
            kernel_window_ms = defaults.get("kernel_window_ms", 800.0)

        frame_dt_ms = 1000.0 * self.linedur_s * self.images.shape[1]
        t_kernel, kernel = calcium_kernel(
            frame_dt_ms,
            rise_tau_ms=rise_tau_ms,
            decay_tau_ms=decay_tau_ms,
            kernel_window_ms=kernel_window_ms,
        )

        if use_znorm:
            traces = self.traces_znorm
        else:
            traces = self.traces_raw

        if traces is None:
            raise ValueError(
                "No traces available. Run extract_traces_from_rois() first."
            )

        if verbose:
            print(f"Frame duration: {frame_dt_ms:.2f} ms")
            print(
                f"Kernel: rise={rise_tau_ms} ms, decay={decay_tau_ms} ms, "
                f"{len(kernel)} bins, λ={lambd:g}"
            )
            print(f"Deconvolving {traces.shape[0]} ROIs × {traces.shape[1]} frames")

        self.traces_deconvolved = wiener_deconvolve(traces, kernel, lambd=lambd)

        self.params.mark_step(
            "deconvolution",
            {
                "rise_tau_ms": rise_tau_ms,
                "decay_tau_ms": decay_tau_ms,
                "lambd": lambd,
                "kernel_window_ms": kernel_window_ms,
                "frame_dt_ms": frame_dt_ms,
                "kernel_bins": len(kernel),
                "use_znorm": use_znorm,
            },
        )

        return self.traces_deconvolved

    def compute_traces_from_rois(self):
        """
        Deprecated method. Use extract_traces_from_rois instead.
        """
        warnings.warn(
            "compute_traces_from_rois is deprecated. Use extract_traces_from_rois instead.",
            DeprecationWarning,
        )
        return self.extract_traces_from_rois()

    def compute_snippets_and_averages(self) -> tuple[np.ndarray, np.ndarray]:
        """
        Compute snippets and averages from ROI traces.

        Mimics IGOR's OS_BasicAveraging functionality. Snippets are individual
        stimulus repetitions, and averages are the mean across all repetitions.
        Always uses raw traces (not z-normalized) as IGOR does.

        Returns:
        --------
        tuple
            (snippets, averages) where:
            - snippets: shape (snippet_length, n_loops, n_rois)
            - averages: shape (snippet_length, n_rois)
        """
        # Check prerequisites
        if self.traces_raw is None:
            print("Traces not available, computing them now...")
            self.extract_traces_from_rois()

        if self.triggertimes is None:
            raise ValueError("Triggertimes not available. Cannot compute snippets.")

        # Use z-normalized traces (always use SD format as IGOR does)
        # Traces are stored as (n_rois, n_frames) - IGOR convention
        # Need to transpose to (n_frames, n_rois) for snippet extraction
        traces = self.traces_znorm.T  # Transpose to (n_frames, n_rois)

        n_frames, n_rois = traces.shape

        # Get parameters
        ignore_first_triggers = self._Core__skip_first_frames
        ignore_last_triggers = self._Core__skip_last_frames
        trigger_mode = self.trigger_mode
        triggertimes = self.triggertimes

        # Line-precision upsampling parameters (matching IGOR)
        n_lines = self.images.shape[1]  # Height of image (number of scan lines)
        line_duration = self.linedur_s
        frame_duration = n_lines * line_duration
        n_lines_lumped = 1  # Default, could be made a parameter
        lines_per_frame = n_lines // n_lines_lumped

        # Calculate valid triggers and snippet parameters (in frame units first)
        n_triggers = len(triggertimes)
        snippet_duration_s = (
            triggertimes[trigger_mode + ignore_first_triggers]
            - triggertimes[ignore_first_triggers]
        )
        snippet_duration_frames = int(snippet_duration_s / frame_duration)

        # Calculate number of complete loops
        valid_triggers = n_triggers - ignore_first_triggers + ignore_last_triggers
        n_loops = valid_triggers // trigger_mode

        print(
            f"Extracting snippets: {n_triggers} triggers, {n_loops} complete loops, snippet duration: {snippet_duration_frames} frames"
        )

        # Initialize output arrays (using frame-level dimensions)
        snippets_frames = np.zeros((snippet_duration_frames, n_loops, n_rois))

        # Extract snippets for each loop from frame-level traces (much faster!)
        for loop_idx in range(n_loops):
            trigger_idx = loop_idx * trigger_mode + ignore_first_triggers
            start_time = triggertimes[trigger_idx]
            # Convert time to frame index
            start_frame = int(start_time / frame_duration)
            end_frame = start_frame + snippet_duration_frames

            # Ensure we don't exceed array bounds
            if end_frame > n_frames:
                print(
                    f"Warning: Loop {loop_idx} exceeds frame array, truncating at {n_loops}"
                )
                n_loops = loop_idx
                break

            # Extract snippet for all ROIs from frame-level traces
            snippets_frames[:, loop_idx, :] = traces[start_frame:end_frame, :]

        # Trim snippets if we had to stop early
        if n_loops < snippets_frames.shape[1]:
            snippets_frames = snippets_frames[:, :n_loops, :]

        # Compute averages across loops (still at frame level)
        averages_frames = np.mean(
            snippets_frames, axis=1
        )  # Shape: (snippet_duration_frames, n_rois)

        print(
            f"Upsampling averages from {snippet_duration_frames} frames to line-precision"
        )

        # NOW upsample only the averages (much more efficient!)
        # Vectorized upsampling using linear interpolation
        weights_next = np.tile(
            np.arange(lines_per_frame) / lines_per_frame, snippet_duration_frames - 1
        )
        weights_curr = 1 - weights_next
        frame_indices = np.repeat(
            np.arange(snippet_duration_frames - 1), lines_per_frame
        )

        # Upsample averages
        averages_upsampled_flat = (
            averages_frames[frame_indices, :] * weights_curr[:, np.newaxis]
            + averages_frames[frame_indices + 1, :] * weights_next[:, np.newaxis]
        )

        # Use the actual length from interpolation (not the calculated target)
        snippet_duration_upsampled = len(averages_upsampled_flat)
        averages = averages_upsampled_flat  # Shape: (snippet_upsampled, n_rois)

        # Upsample snippets using the same interpolation approach
        print(
            f"Upsampling snippets from {snippet_duration_frames} frames to line-precision ({snippet_duration_upsampled} samples)"
        )
        snippets_upsampled = np.zeros((snippet_duration_upsampled, n_loops, n_rois))

        for loop_idx in range(n_loops):
            # Upsample each loop separately
            snippets_upsampled_flat = (
                snippets_frames[frame_indices, loop_idx, :]
                * weights_curr[:, np.newaxis]
                + snippets_frames[frame_indices + 1, loop_idx, :]
                * weights_next[:, np.newaxis]
            )
            snippets_upsampled[:, loop_idx, :] = snippets_upsampled_flat

        # For in-memory use, transpose to match try_fetch behavior
        # Averages: (n_rois, snippet_length)
        averages = averages.T
        # Snippets: (n_rois, n_loops, snippet_length) to match first dimension with averages
        snippets = np.transpose(snippets_upsampled, (2, 1, 0))

        # Compute quality criterion (variance of mean / mean of variance)
        quality_criterion = np.zeros(n_rois)
        for roi_idx in range(n_rois):
            # Variance of the mean (averages has shape (n_rois, snippet_length))
            variance_of_mean = np.var(averages[roi_idx, :])

            # Mean of variances across loops
            # snippets now has shape (n_rois, n_loops, snippet_length)
            variances = np.var(snippets[roi_idx, :, :], axis=1)
            mean_of_variance = np.mean(variances)

            # Quality criterion
            if mean_of_variance > 0:
                quality_criterion[roi_idx] = variance_of_mean / mean_of_variance
            else:
                quality_criterion[roi_idx] = np.nan

        # Save to attributes
        self.snippets = snippets  # Shape: (n_rois, n_loops, snippet_length)
        self.averages = averages  # Shape: (n_rois, snippet_length)
        self.quality_indices = quality_criterion

        # Record step in params
        self.params.mark_step(
            "snippets_and_averages",
            {
                "n_loops": int(n_loops),
                "n_triggers": int(n_triggers),
                "snippet_duration_frames": int(snippet_duration_frames),
                "trigger_mode": int(trigger_mode),
            },
        )

        return snippets, averages

    def plot_averages(
        self,
        rois=None,
        figsize=(None, None),
        figsize_scale=None,
        axs=None,
        independent_scale=False,
        n_rois_raster=50,
        sort_order=None,
        **kwargs: Any,
    ):
        """
        A function to plot the averages of specified regions of interest (rois) on separate subplots within a figure.

        Parameters
        ----------
        rois : Iterable, optional
            Regions of interest to plot. If not specified, all rois will be plotted.
        figsize : tuple, optional
            Size of the figure to plot the subplots. Default is calculated based on the number of rois.

        Keyword arguments
        ----------
        filter_by : Tuple, optional
            Tuple in format (function, "operator", value) where 'function' is a mathematical function that
            can be applied along axis = 1 for self.vverages, '"operator"' is a mathematical operator (e.g,
            "<", ">=", or "==") in string format, and 'value' is the threshold metric.
        sort_by : String, optional
            String representing attribute of data object, where the metric is ROI-by-ROI, such as a list
            or array where each element represents the metric of each ROI
        label_by : String, optional
            As above, but instead of changing the order of pygor.plotting.plots, changes the label associated with
            each ROI to be the specified metric.
        clim : tuple, optional
            A tuple that determines the lower and upper bounds of the clim for the imshow version of the plot, respectively.

        Returns
        -------
        None
        """
        if self.averages is None:
            warnings.warn("Averages do not exist.")
            return
        return pygor.core.plot.plot_averages(
            self,
            rois,
            figsize,
            figsize_scale,
            axs,
            independent_scale,
            n_rois_raster,
            sort_order,
            **kwargs,
        )

    def plot_filter_preview(self, roi_indices, figsize=(16, 6), title_prefix=""):
        """
        Quick side-by-side visualization: ROI map + filtered averages raster.

        Useful for previewing ROI filtering before committing changes with keep_rois().

        Parameters
        ----------
        roi_indices : array-like
            0-indexed ROI indices to preview
        figsize : tuple, optional
            Figure size (width, height). Default (16, 6).
        title_prefix : str, optional
            Optional prefix for the title (e.g., filter criteria description)

        Returns
        -------
        fig, (ax_map, ax_avg)
            Figure and axes tuple

        Examples
        --------
        >>> spatial_pass = obj.rois_in_range(x_range=(40, 80))
        >>> qc_pass = np.argwhere(obj.quality_indices > 0.25).ravel()
        >>> preview_rois = np.intersect1d(spatial_pass, qc_pass)
        >>> obj.plot_filter_preview(preview_rois, title_prefix="QC>0.25, x:40-80")
        """
        roi_indices = np.asarray(roi_indices).ravel()

        fig, (ax_map, ax_avg) = plt.subplots(1, 2, figsize=figsize)

        # Left: ROI overlay on correlation projection
        if self.correlation_projection is not None:
            temp_mask = np.isin(self.rois_alt, roi_indices)
            rois_map = np.where(temp_mask, self.rois_alt, np.nan)
            ax_map.imshow(self.correlation_projection, cmap="gray", origin="lower")
            im = ax_map.imshow(rois_map, cmap="jet", alpha=0.5, origin="lower")
            plt.colorbar(im, ax=ax_map, label="ROI ID")
        else:
            # Fallback to just ROI mask if no correlation projection
            temp_mask = np.isin(self.rois_alt, roi_indices)
            rois_map = np.where(temp_mask, self.rois_alt, np.nan)
            im = ax_map.imshow(rois_map, cmap="jet", origin="lower")
            plt.colorbar(im, ax=ax_map, label="ROI ID")

        title = f"{len(roi_indices)} ROIs"
        if title_prefix:
            title = f"{title_prefix}: {title}"
        ax_map.set_title(title)
        ax_map.axis("off")

        # Right: averages raster (always use imshow for preview speed)
        if self.averages is not None and len(roi_indices) > 0:
            ax_avg.imshow(
                self.averages[roi_indices],
                aspect="auto",
                cmap="Greys_r",
                interpolation="none",
            )
            ax_avg.set_title("Averages (raster)")
            ax_avg.set_xlabel("Time (samples)")
            ax_avg.set_ylabel("ROI")
        else:
            ax_avg.text(
                0.5,
                0.5,
                "No averages available",
                ha="center",
                va="center",
                transform=ax_avg.transAxes,
            )
            ax_avg.axis("off")

        plt.tight_layout()
        return fig, (ax_map, ax_avg)

    def plot_traces(
        self,
        rois: list[int] | None = None,
        n_rois_imshow: int = 50,
        figsize: tuple[float, float] | None = None,
        cmap: str = "inferno",
        unit: str = "seconds",
        show_baseline: bool = True,
        **kwargs: Any,
    ):
        """
        Plot traces_znorm as stacked line traces or as an imshow heatmap.

        When the number of ROIs is <= n_rois_imshow, each ROI gets its own
        subplot axis with independent y-scaling. Axes are squashed together
        with no gaps so the result looks like one continuous stacked plot.
        When the number exceeds n_rois_imshow, falls back to imshow.

        Parameters
        ----------
        rois : array-like, optional
            Indices of ROIs to plot. If None, plots all.
        n_rois_imshow : int
            Threshold number of ROIs above which imshow is used instead
            of individual line traces. Default 50.
        figsize : tuple, optional
            Figure size (width, height). Auto-calculated if None.
        cmap : str
            Colormap for the imshow fallback. Default "inferno".
        unit : str
            X-axis unit: "seconds" (or "s") to convert frames via
            self.frame_hz, or "frames" (or "f") to keep raw frame indices.
            Default "seconds".
        show_baseline : bool
            If True, overlay a shaded region indicating the baseline window
            used for z-normalization. Requires baseline_info to be available
            (i.e. traces must have been extracted). Default False.
        **kwargs
            Passed to plt.plot (line mode) or ax.imshow (imshow mode).
        """
        if self.traces_znorm is None:
            warnings.warn("traces_znorm is None, nothing to plot.")
            return
        traces = self.traces_znorm  # (n_rois, n_timepoints)
        if isinstance(rois, int):
            rois = [rois]
        if rois is not None:
            traces = traces[np.asarray(rois)]
        n_rois = traces.shape[0]
        n_frames = traces.shape[1]
        # Build x-axis
        if unit in ("seconds", "s"):
            x = np.arange(n_frames) / self.frame_hz
            xlabel = "Time (s)"
        else:
            x = np.arange(n_frames)
            xlabel = "Time (frames)"
        # Resolve baseline span in x-axis units (if requested)
        baseline_span = None
        if show_baseline:
            info = self.baseline_info
            if info is not None:
                if unit in ("seconds", "s"):
                    baseline_span = (
                        info["start_frame"] / self.frame_hz,
                        info["end_frame"] / self.frame_hz,
                    )
                else:
                    baseline_span = (info["start_frame"], info["end_frame"])
            else:
                warnings.warn(
                    "show_baseline=True but no baseline info available "
                    "(traces not yet extracted?)."
                )
        if n_rois > n_rois_imshow:
            # --- imshow mode ---
            if figsize is None:
                figsize = (8, max(3, n_rois / 10))
            fig, ax = plt.subplots(figsize=figsize)
            extent = [x[0], x[-1], n_rois - 0.5, -0.5]
            ax.imshow(
                traces,
                cmap=cmap,
                interpolation="none",
                extent=extent,
                aspect="auto",
                **kwargs,
            )
            if baseline_span is not None:
                ax.axvspan(
                    *baseline_span,
                    facecolor="silver",
                    alpha=0.25,
                    label="Baseline",
                    edgecolor="red",
                    linestyle="--",
                )
            ax.set_xlabel(xlabel)
            ax.set_ylabel("ROI")
            return fig, ax
        # --- line trace mode ---
        if figsize is None:
            figsize = (8, max(3, n_rois * 0.4))
        fig, axs = plt.subplots(
            n_rois,
            1,
            figsize=figsize,
            sharex=True,
            gridspec_kw={"hspace": 0},
        )
        if n_rois == 1:
            axs = [axs]
        colors = plt.cm.jet(np.linspace(0, 1, n_rois))
        for i, ax in enumerate(axs):
            ax.plot(x, traces[i], color=colors[i], linewidth=0.7, **kwargs)
            ax.set_xlim(x[0], x[-1])
            if baseline_span is not None:
                ax.axvspan(
                    *baseline_span, color="silver", alpha=1, zorder=-1, edgecolor=None
                )
            ax.axis("off")
        # Re-enable x-axis on bottom subplot only
        axs[-1].axis("on")
        axs[-1].spines["top"].set_visible(False)
        axs[-1].spines["right"].set_visible(False)
        axs[-1].spines["left"].set_visible(False)
        axs[-1].tick_params(left=False, labelleft=False)
        axs[-1].set_xlabel(xlabel)
        return fig, axs

    def calculate_image_average(self, ignore_skip=False) -> np.ndarray:
        """
        Calculate the average image from a series of trigger frames.

        Parameters:
        ----------
        self : object
            The instance of the class.

        Returns:
        -------
        numpy.ndarray
            The average image calculated from the trigger frames.
        """
        # If no repetitions
        if self.trigger_mode == 1:
            warnings.warn("No repetitions detected, returning original images")
            return
        # Account for trigger skipping logic
        if ignore_skip is True:
            # Ignore skipping parameters
            first_trig_frame = 0
            last_trig_frame = 0
            # triggers_frames = self.triggertimes_frame
        else:
            # Othrwise, account for skipping parameters
            first_trig_frame = self.__skip_first_frames
            last_trig_frame = self.__skip_last_frames
            print(
                f"Skipping first {first_trig_frame} and last {last_trig_frame} frames"
            )
        if last_trig_frame == 0:
            last_trig_frame = None
        triggers_frames = self.triggertimes_frame[first_trig_frame:last_trig_frame]
        # Get the frame interval over which to average the images
        rep_start_frames = triggers_frames[:: self.trigger_mode]
        rep_delta_frames = np.diff(
            triggers_frames[:: self.trigger_mode]
        )  # time between repetitions, by frame number
        rep_delta = int(
            np.floor(np.average(rep_delta_frames))
        )  # take the average of the differentiated values, and round down. This your delta time in frames
        # Calculate number of full repetitions
        percise_reps = (
            len(triggers_frames) / self.trigger_mode
        )  # This may yield a float if partial repetition
        if (
            percise_reps % 1 != 0
        ):  # In that case, we don't have an exact loop so we ignore the residual partial loop
            reps = int(
                triggers_frames[
                    : int(np.floor(percise_reps) % percise_reps * self.trigger_mode)
                ].shape[0]
                / self.trigger_mode
            )  # number of full repetitions, by removing residual non-complete repetitions
            print(
                f"Partial loop detected ({percise_reps}), using only",
                reps,
                "repetitions and ignoring last partial loop",
            )
        else:
            reps = int(percise_reps)
        # Extract frames accordingly (could vectorize this)
        images_to_average = []
        for frame in rep_start_frames[:reps]:
            images_to_average.append(self.images[frame : frame + rep_delta])
        if reps <= 0:
            raise ValueError(
                "No full repetitions available for averaging. Check trigger settings or skipping parameters."
            )
        # Ensure all repetitions have equal length before stacking
        min_len = min(arr.shape[0] for arr in images_to_average)
        if min_len == 0:
            raise ValueError(
                "One or more repetitions have zero frames; cannot compute average."
            )
        if any(arr.shape[0] != min_len for arr in images_to_average):
            warnings.warn(
                "Unequal repetition lengths detected; trimming to shortest repetition for averaging."
            )
        images_to_average = np.stack(
            [arr[:min_len] for arr in images_to_average], axis=0
        )
        # Print results
        print(
            f"{len(triggers_frames)} triggers with a trigger_mode of {self.trigger_mode} gives {reps} full repetitions of {rep_delta} frames each."
        )
        # Average the images
        avg_movie = np.average(images_to_average, axis=0)
        return avg_movie

    def get_average_markers(self):
        return pygor.core.methods.determine_epoch_markers_ms(self)

    def get_epoch_dur(self, rtol=1e-3, atol=1e-3):
        # Differentiate the average markers to get the epoch durations
        diff = np.diff(self.get_average_markers())
        if diff.size == 0:
            warnings.warn("No epochs found, returning 0")
            return 0
        # Calculate the average duration of the epochs
        avg = np.average(diff)
        # Check for unequal epochs
        if np.allclose(diff, avg, rtol=rtol, atol=atol) is False:
            # If unequal, raise error and ask user to manually set epoch durations
            raise ValueError(
                f"Epoch durations are not equal with tolerences {rtol}, {atol}, adjust tolerences or manually set epoch durations."
            )
        return np.floor(np.average(np.diff(self.get_average_markers()))).astype(int)

    def get_correlation_map(self, recompute=False):
        """Compute correlation map and store in self.correlation_projection.

        Uses cached value if available, unless recompute=True.
        """
        if self.correlation_projection is None or recompute:
            self.correlation_projection = pygor.core.methods.correlation_map(
                self.images
            )
        return self.correlation_projection

    def calc_mean_triggertimes(self, unit="index"):
        """
        Calculate the mean trigger times in seconds.

        Returns
        -------
        numpy.ndarray
            Array of mean trigger times in seconds.
        """
        if self.trigger_mode == 1:
            avg_epoch_dur = np.average(np.diff(self.triggertimes))
            markers_arr = self.triggertimes * (1 / self.linedur_s)
            markers_arr -= markers_arr[0]
        else:
            if self.triggertimes.shape[0] % self.trigger_mode != 0:
                print("WARNING: Trigger times are not evenly divisible by trigger mode")
                # Determine amount of triggers to crop out to achieve loop alignment
                num_triggers_to_crop = self.triggertimes.shape[0] % self.trigger_mode
                triggertimes = self.triggertimes[:-num_triggers_to_crop]
                print(
                    f"WARNING: Cropped {num_triggers_to_crop} triggers to achieve loop alignment"
                )
            else:
                triggertimes = self.triggertimes
            # Calculate average trigger times in seconds
            avg_epoch_dur = np.average(
                np.diff(triggertimes.reshape(-1, self.trigger_mode)[:, 0])
            )
            epoch_reshape = triggertimes.reshape(-1, self.trigger_mode)
            temp_arr = np.empty(epoch_reshape.shape)
            for n, i in enumerate(epoch_reshape):
                temp_arr[n] = i - (avg_epoch_dur * n)
            avg_epoch_triggertimes = np.average(temp_arr, axis=0)
            markers_arr_s = avg_epoch_triggertimes  # / self.linedur_s
            markers_arr_s -= markers_arr_s[0]
            if unit == "index" or unit == "indices":
                markers_arr = markers_arr_s * (1 / self.linedur_s)
                markers_arr = np.round(markers_arr, 0).astype(int)
            if unit == "seconds" or unit == "s":
                markers_arr = markers_arr_s
            if unit == "frames" or unit == "frame":
                markers_arr = markers_arr_s * self.frame_hz
                markers_arr = np.round(markers_arr, 0).astype(int)
            if unit == "ms":
                markers_arr = markers_arr_s * 1000
        return markers_arr  # .astype(int)

    @property
    def rois_alt(self):
        temp_rois = self.rois.copy().astype(float)
        temp_rois[temp_rois == 1] = np.nan
        temp_rois *= -1
        temp_rois = temp_rois - 1
        return temp_rois

    @property
    def traces_znorm_ms(self):
        """
        Get interpolated and upscaled traces_znorm in millisecond precision.

        Converts traces from frame precision to line precision (~500 Hz sampling)
        using linear interpolation, matching IGOR Pro's OS_BasicAveraging behavior.

        Returns
        -------
        numpy.ndarray
            Interpolated traces with shape (n_rois, n_timepoints_ms) where
            n_timepoints_ms corresponds to line precision sampling rate.
        """
        if self.traces_znorm is None:
            return None

        # Get dimensions
        n_rois, n_frames = self.traces_znorm.shape

        # Calculate frame duration and line duration
        frame_duration_s = 1.0 / self.frame_hz  # Frame duration in seconds
        line_duration_s = self.linedur_s  # Line duration in seconds (typically ~0.002s)

        # Calculate number of lines per frame (nY equivalent)
        lines_per_frame = int(frame_duration_s / line_duration_s)

        # Total interpolated time points (line precision)
        n_points_ms = n_frames * lines_per_frame

        # Use scipy.ndimage.zoom for fast interpolation
        from scipy.ndimage import zoom

        # Calculate zoom factor for time axis
        zoom_factor = lines_per_frame

        # Interpolate all ROI traces at once using zoom
        # zoom applies along the last axis (time axis)
        traces_ms = zoom(self.traces_znorm, (1, zoom_factor), order=1, mode="nearest")

        return traces_ms

    @property
    def roi_centroids(self, force=True):
        """
        Get the centre of mass for each ROI in the image if not already done.
        """
        if np.all(np.logical_or(np.unique(self.rois) < 0, np.unique(self.rois) == 1)):
            temp_rois = self.rois_alt + 1
            labels = np.unique(temp_rois)
            labels = labels[~np.isnan(labels)]
            centroids = scipy.ndimage.center_of_mass(temp_rois, temp_rois, labels)
        else:
            labels = np.unique(self.rois)
            labels = labels[~np.isnan(labels)]
            centroids = scipy.ndimage.center_of_mass(self.rois, self.rois, labels)
        return np.array(centroids)

    def rois_in_range(self, x_range=None, y_range=None):
        """
        Return ROI indices whose centroids fall within the provided ranges.

        Parameters
        ----------
        x_range : tuple or None
            (xmin, xmax) in pixel coordinates, inclusive. If None, no x filter.
        y_range : tuple or None
            (ymin, ymax) in pixel coordinates, inclusive. If None, no y filter.

        Returns
        -------
        numpy.ndarray
            0-indexed ROI indices (matching averages/traces ordering).
        """
        if self.rois is None:
            raise ValueError("No ROIs defined. Impossible operation.")

        centroids = self.roi_centroids
        roi_indices = np.arange(centroids.shape[0])

        keep = np.ones(centroids.shape[0], dtype=bool)
        if x_range is not None:
            xmin, xmax = x_range
            keep &= (centroids[:, 1] >= xmin) & (centroids[:, 1] <= xmax)
        if y_range is not None:
            ymin, ymax = y_range
            keep &= (centroids[:, 0] >= ymin) & (centroids[:, 0] <= ymax)

        return roi_indices[keep]

    def keep_rois(self, roi_indices, update_dependent=True):
        """
        Keep only the specified ROIs and set the rest to background.

        Parameters
        ----------
        roi_indices : list or array-like
            0-indexed ROI indices to keep (e.g. [0, 1, 2] for the first 3 ROIs).
            These match the indexing used by quality_indices, traces, averages,
            and rois_alt. Internally, ROI mask values are -(index + 1).
        update_dependent : bool, optional
            If True, subset dependent arrays (traces, averages, snippets,
            quality_indices, roi_sizes) to keep indexing consistent.

        Returns
        -------
        numpy.ndarray
            Updated ROI mask with only specified ROIs retained.
        """
        if self.rois is None:
            raise ValueError("No ROIs defined. Impossible operation.")
        roi_indices = np.asarray(roi_indices).astype(int).ravel()

        if roi_indices.size == 0:
            self.rois = np.ones_like(self.rois)
            self.num_rois = 0
            if update_dependent:
                if isinstance(self.traces_raw, np.ndarray):
                    self.traces_raw = self.traces_raw[:0]
                if isinstance(self.traces_znorm, np.ndarray):
                    self.traces_znorm = self.traces_znorm[:0]
                if isinstance(self.averages, np.ndarray):
                    self.averages = self.averages[:0]
                if isinstance(self.snippets, np.ndarray):
                    self.snippets = self.snippets[:0]
                if isinstance(self.quality_indices, np.ndarray):
                    self.quality_indices = self.quality_indices[:0]
                if hasattr(self, "roi_sizes") and isinstance(
                    self.roi_sizes, np.ndarray
                ):
                    self.roi_sizes = self.roi_sizes[:0]
            return self.rois

        roi_indices = np.unique(roi_indices)

        # Convert 0-indexed indices to internal mask values: 0 -> -1, 1 -> -2, etc.
        roi_values = -(roi_indices + 1)

        # Keep only matching ROIs, set everything else to background (1)
        self.rois = np.where(np.isin(self.rois, roi_values), self.rois, 1)

        if update_dependent:

            def _subset_first_dim(array_value):
                if not isinstance(array_value, np.ndarray):
                    return array_value
                if array_value.shape[0] < roi_indices.max() + 1:
                    return array_value
                return array_value[roi_indices]

            self.traces_raw = _subset_first_dim(self.traces_raw)
            self.traces_znorm = _subset_first_dim(self.traces_znorm)
            self.averages = _subset_first_dim(self.averages)
            self.snippets = _subset_first_dim(self.snippets)
            self.quality_indices = _subset_first_dim(self.quality_indices)
            if hasattr(self, "roi_sizes"):
                self.roi_sizes = _subset_first_dim(self.roi_sizes)

        # Reindex remaining ROI labels to keep indices contiguous (-1, -2, ...)
        roi_ids = np.unique(self.rois)
        roi_ids = roi_ids[roi_ids < 0]
        roi_ids = np.sort(roi_ids)[::-1]  # -1, -2, -3, ...
        if roi_ids.size:
            remapped = self.rois.copy()
            for new_idx, old_id in enumerate(roi_ids):
                remapped[self.rois == old_id] = -(new_idx + 1)
            self.rois = remapped

        self.num_rois = (
            np.unique(self.rois).size - 1
        )  # Update num_rois based on unique values (excluding background)
        return self.rois
