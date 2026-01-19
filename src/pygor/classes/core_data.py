from dataclasses import dataclass, field

try:
    from collections import Iterable
except ImportError:
    from collections.abc import Iterable
# Local imports
import pygor.core.methods
import pygor.core.plot
import pygor.data_helpers
import pygor.utils.helpinfo
import pygor.strf.spatial
import pygor.strf.contouring
import pygor.strf.temporal
import pygor.plotting.basic
import pygor.utils
import pygor.core
from pygor.params import AnalysisParams

# Dependencies
import operator
import matplotlib.pyplot as plt
import numpy as np
import pathlib
import h5py
import matplotlib.patheffects as path_effects
import matplotlib
import warnings
import scipy.ndimage
import math
from mpl_toolkits.axes_grid1 import make_axes_locatable
import skimage


def try_fetch(file, key):
    try:
        result = file[key]
        result_shape = result.shape
        if result_shape != (): # if not a scalar
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
            f"'{params_key}' not found in {file.filename}, setting to np.nan", stacklevel=2
        )
        error   
        return np.nan
    
@dataclass
class Core:
    filename: str or pathlib.Path
    config: str | pathlib.Path = None  # Optional path to TOML config file
    do_preprocess: bool | dict = False  # Preprocessing options (for ScanM files)
    metadata: dict = field(init=False)
    rois: dict = field(init=False)
    type: str = field(init=False)
    frame_hz: float = field(init=False)
    averages: np.array = np.nan
    snippets: np.array = np.nan
    ms_dur: int = np.nan
    trigger_mode : int = 1 #defualt value
    num_rois: int = field(init=False)
    params: AnalysisParams = field(init=False)  # Analysis parameters

    def __post_init__(self):
        """initialize Core by auto-detecting file format and loading data."""
        # Ensure path is pathlib compatible
        if isinstance(self.filename, pathlib.Path) is False:
            self.filename = pathlib.Path(self.filename)

        ext = self.filename.suffix.lower()

        # Auto-detect format based on extension
        if ext in ['.smp', '.smh']:
            # ScanM file - delegate to internal loader
            self._load_from_scanm_internal()
        elif ext in ['.h5', '.hdf5']:
            # H5 file - use original loading logic
            if self.do_preprocess:
                warnings.warn("do_preprocess parameter is ignored for H5 files (typically already preprocessed). Call manually if needed.", stacklevel=2)
            self._load_from_h5()
        else:
            raise ValueError(
                f"Unknown file extension '{ext}'. "
                "Supported formats: .h5/.hdf5 (IGOR export), .smp/.smh (ScanM raw)"
            )

        # Initialize analysis parameters from config (both paths)
        self.params = AnalysisParams.from_config(self.config)

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
            self.num_rois = len(np.unique(self.rois)) - 1
            self.roi_sizes = try_fetch(HDF5_file, "RoiSizes")
            if self.roi_sizes is not None:
                self.roi_sizes = self.roi_sizes[:self.num_rois]
            # Timing parameters
            self.triggertimes = try_fetch(HDF5_file, "Triggertimes")
            self.triggertimes = self.triggertimes[~np.isnan(self.triggertimes)].astype(
                float
            )
            self.triggertimes_frame = try_fetch(HDF5_file, "Triggertimes_Frame")
            self.__skip_first_frames = int(try_fetch_table_params(HDF5_file, "Skip_First_Triggers")) # Note name mangling to prevent accidents if 
            self.__skip_last_frames = -int(try_fetch_table_params(HDF5_file, "Skip_Last_Triggers")) # private class attrs share names 
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
                self.stage = try_fetch_table_params(HDF5_file, "stage", 'wExpParams').decode('utf-8')
                self.orientation = try_fetch_table_params(HDF5_file, "orientation", 'wExpParams').decode('utf-8')
                self.depth = try_fetch_table_params(HDF5_file, "depth", 'wExpParams').decode('utf-8')
                self.stimulus = try_fetch_table_params(HDF5_file, "stimulus", 'wExpParams').decode('utf-8')
            if self.images is not None:
                self.frame_hz = float(1/(self.images.shape[1]/self.n_planes*self.linedur_s))
            else:
                self.frame_hz = float(1/(self.average_stack.shape[0]/self.n_planes*self.linedur_s))
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
        header, channel_data = scanm_module.load_scanm(self.filename, channels=channels_to_load)

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
        self._Core__skip_last_frames = -skip_last_triggers if skip_last_triggers > 0 else 0

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
        # We use object.__new__ to bypass dataclass __init__
        instance = object.__new__(cls)
        
        # Set all required attributes manually
        instance.filename = path
        instance.type = cls.__name__
        instance.name = path.stem
        
        # Image data
        instance.images = images
        instance.trigger_images = trigger_stack  # Store trigger channel for visualization
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
        instance._Core__skip_last_frames = -skip_last_triggers if skip_last_triggers > 0 else 0
        
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
        pygor.config : Configuration management for defaults
        """
        import pygor.preproc.scanm as scanm_module

        # Check if already preprocessed
        if self.params.preprocessed and not force:
            warnings.warn(
                "Data has already been preprocessed. Use force=True to re-apply. "
                "Note: re-preprocessing already-preprocessed data may produce artifacts.",
                RuntimeWarning
            )
            return

        # Get defaults from params (loaded from config)
        defaults = self.params.get_defaults("preprocessing")

        # Use current params.artifact_width as the default (can be modified before calling preprocess)
        if artifact_width is None:
            artifact_width = self.params.artifact_width

        # Collect user-provided params (filter out None values)
        user_params = {
            k: v for k, v in {
                'artifact_width': artifact_width,
                'flip_x': flip_x,
                'detrend': detrend,
                'smooth_window_s': smooth_window_s,
                'time_bin': time_bin,
                'fix_first_frame': fix_first_frame,
            }.items() if v is not None
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

        # Record preprocessing in params (sets preprocessed=True and artifact_width)
        self.params.mark_preprocessing(params)

    def register(
        self,
        n_reference_frames: int = None,
        batch_size: int = None,
        artifact_width: int = None,
        upsample_factor: int = None,
        normalization: str = None,
        order: int = None,
        mode: str = None,
        force: bool = False,
        plot: bool = False,
        parallel: bool = True,
        n_jobs: int = -1,
        batch_mode: str = None,
        reference_mode: str = None,
        edge_crop: int = None,
    ) -> dict:
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
                RuntimeWarning
            )
            return self.params.registration or {}

        # Store original for plotting comparison
        original_stack = self.images.copy() if plot else None

        # Get defaults from params (loaded from config)
        defaults = self.params.get_defaults("registration")

        # Collect user-provided params (filter out None values)
        user_params = {
            k: v for k, v in {
                'n_reference_frames': n_reference_frames,
                'batch_size': batch_size,
                'upsample_factor': upsample_factor,
                'normalization': normalization,
                'order': order,
                'mode': mode,
                'parallel': parallel,
                'n_jobs': n_jobs,
                'batch_mode': batch_mode,
                'reference_mode': reference_mode,
                'edge_crop': edge_crop,
            }.items() if v is not None
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
            **params,
        )

        # Update images
        self.images = registered

        # Update average_stack to reflect registered data
        self.average_stack = self.images.mean(axis=0)

        # Compute statistics
        stats = {
            'mean_shift': tuple(shifts.mean(axis=0)),
            'std_shift': tuple(shifts.std(axis=0)),
            'max_shift': tuple(shifts.max(axis=0)),
            'mean_error': float(errors.mean()),
            'shifts': shifts,
            'errors': errors,
        }

        # Record registration in params (sets registered=True)
        self.params.mark_registration(params, stats)

        # Plot if requested
        if plot:
            self._plot_registration_results(
                shifts, errors, original_stack, params.get('reference_mode', 'std')
            )

        # if stats["mean_error"] < 0.05:
        print(f"Registration complete.\n"
            f"  Mean error: {stats['mean_error']:.4f}\n"
            f"  Max shift: (y={stats['max_shift'][0]:.2f}, x={stats['max_shift'][1]:.2f})\n"
            f"  Mean shift: (y={stats['mean_shift'][0]:.2f}, x={stats['mean_shift'][1]:.2f})\n"
            f"  Shift SD: (y={stats['std_shift'][0]:.2f}, x={stats['std_shift'][1]:.2f})")
        # else:
        #     print(f"Warning: Registration error exceeds threshold. Mean error: {stats['mean_error']:.4f}")
        return stats

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

        # Reference image (mean of first N frames)
        ref_image = self.average_stack
        ax_ref.imshow(ref_image, cmap='gray')
        ax_ref.set_title('Reference (mean)')
        ax_ref.axis('off')

        # Original projection
        ax_orig.imshow(proj_original, cmap='gray', vmin=vmin, vmax=vmax)
        ax_orig.set_title(f'Before ({reference_mode})')
        ax_orig.axis('off')

        # Registered projection
        ax_reg.imshow(proj_registered, cmap='gray', vmin=vmin, vmax=vmax)
        ax_reg.set_title(f'After ({reference_mode})')
        ax_reg.axis('off')

        # Shift traces
        ax_shifts.plot(batch_idx, shifts[:, 0], 'b-', label='Y shift', linewidth=1, alpha=0.8)
        ax_shifts.plot(batch_idx, shifts[:, 1], 'r-', label='X shift', linewidth=1, alpha=0.8)
        ax_shifts.axhline(0, color='k', linestyle='--', alpha=0.3)
        ax_shifts.set_xlabel('Batch index')
        ax_shifts.set_ylabel('Shift (pixels)')
        ax_shifts.legend(loc='upper right')
        ax_shifts.grid(True, alpha=0.3)
        ax_shifts.set_title('Registration Shifts Over Time')

        plt.tight_layout()
        plt.show()

    def export_to_h5(
        self,
        output_path=None,
        overwrite: bool = False,
    ):
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
            images_t = self.images.transpose(2, 1, 0)
            # IGOR stores as uint16 (unsigned), matching raw ADC values
            f.create_dataset("wDataCh0_detrended", data=images_t, dtype=np.uint16)
            
            # Trigger channel (if available)
            if hasattr(self, 'trigger_images') and self.trigger_images is not None:
                trigger_t = self.trigger_images.transpose(2, 1, 0)
                f.create_dataset("wDataCh2", data=trigger_t, dtype=np.int16)
            
            # Average stack
            f.create_dataset("Stack_Ave", data=self.average_stack.T, dtype=np.float32)
            
            #  ROIs 
            if self.rois is not None:
                f.create_dataset("ROIs", data=self.rois.T, dtype=np.int16)
                
            if self.roi_sizes is not None:
                f.create_dataset("RoiSizes", data=self.roi_sizes, dtype=np.int32)
            
            #  Traces 
            if self.traces_raw is not None:
                f.create_dataset("Traces0_raw", data=self.traces_raw.T, dtype=np.float32)
                
            if self.traces_znorm is not None:
                f.create_dataset("Traces0_znorm", data=self.traces_znorm.T, dtype=np.float32)
            
            #  Trigger times 
            max_triggers = max(len(self.triggertimes_frame) if self.triggertimes_frame is not None else 0, 1000)
            triggertimes = np.full(max_triggers, np.nan)
            if self.triggertimes is not None and len(self.triggertimes) > 0:
                triggertimes[:len(self.triggertimes)] = self.triggertimes
            f.create_dataset("Triggertimes", data=triggertimes, dtype=np.float64)
            
            triggertimes_frame = np.full(max_triggers, np.nan)
            if self.triggertimes_frame is not None and len(self.triggertimes_frame) > 0:
                triggertimes_frame[:len(self.triggertimes_frame)] = self.triggertimes_frame
            f.create_dataset("Triggertimes_Frame", data=triggertimes_frame, dtype=np.float64)
            
            #  wParamsStr (date/time metadata) 
            exp_date = self.metadata["exp_date"]
            exp_time = self.metadata["exp_time"]
            date_str = f"{exp_date.year}-{exp_date.month:02d}-{exp_date.day:02d}"
            time_str = f"{exp_time.hour:02d}-{exp_time.minute:02d}-{exp_time.second:02d}-00"
            
            params_str = [""] * 10
            params_str[4] = date_str
            params_str[5] = time_str
            params_str[0] = str(self.filename.stem)
            
            dt = h5py.special_dtype(vlen=str)
            params_str_ds = f.create_dataset("wParamsStr", (len(params_str),), dtype=dt)
            for i, s in enumerate(params_str):
                params_str_ds[i] = s.encode("utf-8")
            
            #  wParamsNum (XYZ position) 
            params_num = np.zeros(50, dtype=np.float64)
            xyz = self.metadata.get("objectiveXYZ", (0, 0, 0))
            params_num[26] = xyz[0]
            params_num[27] = xyz[2]
            params_num[28] = xyz[1]
            f.create_dataset("wParamsNum", data=params_num, dtype=np.float64)
            
            #  OS_Parameters 
            os_params_keys = [
                "placeholder",
                "LineDuration",
                "nPlanes", 
                "Trigger_Mode",
                "Skip_First_Triggers",
                "Skip_Last_Triggers",
            ]
            os_params_values = np.array([
                0,
                self.linedur_s,
                self.n_planes,
                self.trigger_mode,
                0,
                0,
            ], dtype=np.float64)
            
            os_params_ds = f.create_dataset("OS_Parameters", data=os_params_values)
            os_params_ds.attrs["OS_Parameters"] = np.array(
                [b"Keys"] + [k.encode() for k in os_params_keys], 
                dtype=object
            )
            
            #  Optional data 
            if self.averages is not None:
                f.create_dataset("Averages0", data=self.averages.T, dtype=np.float32)
                
            if self.snippets is not None:
                f.create_dataset("Snippets0", data=self.snippets.T, dtype=np.float32)
                
            if self.ipl_depths is not None:
                f.create_dataset("Positions", data=self.ipl_depths, dtype=np.float64)
                
            if self.correlation_projection is not None:
                f.create_dataset("correlation_projection", data=self.correlation_projection.T, dtype=np.float32)
                
            if self.quality_indices is not None:
                f.create_dataset("QualityCriterion", data=self.quality_indices, dtype=np.float64)
        
        print(f"Exported to: {output_path}")
        return output_path

    def __repr__(self):
        # For pretty printing
        date = self.metadata["exp_date"].strftime("%d-%m-%Y")
        return f"{date}:{self.__class__.__name__}:{self.filename.stem}"

    def __str__(self):
        # For pretty printing
        return f"{self.__class__}"
    
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
        exclude_patterns = getattr(self, '_help_exclude_patterns', None)
        method_list = pygor.utils.helpinfo.get_methods_list(self, with_returns=types, exclude_patterns=exclude_patterns)
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
        show_axes=False,
        **kwargs,
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
            scanv = ax.imshow(self.average_stack, cmap = "Greys_r", origin = "lower", **kwargs)
        elif func == pygor.core.methods.correlation_map:
            correlation_result = func(self.images[zstart:zstop, ystart:ystop, xstart:xstop])
            scanv = ax.imshow(correlation_result, cmap ="Greys_r", origin = "lower", **kwargs)
        else:
            scanv = ax.imshow(func(self.images[zstart:zstop, ystart:ystop, xstart:xstop:], axis = axis), cmap ="Greys_r", origin = "lower", **kwargs)
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
        **kwargs,
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
            scanv = ax.imshow(self.average_stack, cmap = "Greys_r", origin = "lower", **kwargs)
        elif func == pygor.core.methods.correlation_map:
            correlation_result = func(self.images[zstart:zstop, ystart:ystop, xstart:xstop])
            scanv = ax.imshow(correlation_result, cmap ="Greys_r", origin = "lower", **kwargs)
        else:
            scanv = ax.imshow(func(self.images[zstart:zstop, ystart:ystop, xstart:xstop:], axis = axis), cmap ="Greys_r", origin = "lower", **kwargs)
        if outline:
            # Extract ROI outlines instead of filled regions
            from skimage import measure
            
            # Get unique ROI values from rois_alt (0-based indexing, background is NaN)
            roi_values = np.unique(rois_to_use)
            roi_values = roi_values[~np.isnan(roi_values)].astype(int)  # Remove NaN (background)
            
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

                        ax.plot(contour[:, 1], contour[:, 0], 
                            color=roi_color, linewidth=outline_width, alpha=alpha)
                
                # Create dummy mappable for colorbar compatibility
                import matplotlib.cm as cm
                norm = matplotlib.colors.Normalize(vmin=np.min(roi_values), vmax=np.max(roi_values))
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
            
            rois_masked = np.ma.masked_where(np.isnan(rois_display), rois_display)[ystart:ystop, xstart:xstop]
            rois = ax.imshow(rois_masked, cmap = color, alpha = alpha, origin = "lower", **kwargs)
        ax.grid(False)
        ax.axis("off")
        if cbar == True:
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="5%", pad=0.05)
            plt.colorbar(rois, ax=ax, cax=cax)
        if labels == True:
            # Get ROI values from rois_alt (0-based indexing)
            label_map = np.unique(rois_to_use)
            label_map = label_map[~np.isnan(label_map)].astype(int)  # Remove NaN (background)
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
        ax.imshow(pygor.utilities.min_max_norm(d3, 0, 1))

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
    
    def update_ipl_depths(self, depths=None, overwrite=False):
        """
        Update IPL depths in the H5 file, optionally using interactive depth selection.

        Parameters
        ----------
        depths : array-like, optional
            Pre-calculated depths. If None, launches interactive depth selection.
        overwrite : bool, optional
            Whether to overwrite existing ipl_depths (default: False)

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

        success = self.update_h5_key('Positions', depths, overwrite)
        if success:
            self.ipl_depths = depths  # Update the object attribute
            print(f"Successfully updated ipl_depths for {len(depths)} ROIs")
        return success

    def update_rois(self, roi_mask, overwrite=True):
        """
        Update ROIs in the H5 file with a pre-defined ROI mask. If overwrite=False,
        existing ROIs will not be modified. If the object is associated with an H5 file,
        the 'ROIs' key will be updated accordingly.

        For interactive ROI drawing, use draw_rois(overwrite=True) instead.

        Parameters
        ----------
        roi_mask : array-like
            ROI mask to save (background=1, ROIs=-1,-2,...,-n)
        overwrite : bool, optional
            Whether to overwrite existing ROIs (default: False)

        Returns
        -------
        bool
            True if update was successful, False otherwise
        """
        self.rois = roi_mask  # Update the object attribute
        self.num_rois = len(np.unique(roi_mask))-1
        print(f"Successfully updated object.rois: {self.num_rois} ROIs saved")

        # check if object is associated with an H5 file
        if self.filename.suffix == '.h5' and overwrite:
            print("Associated H5 file detected, updating 'ROIs' key...")
            bool = self.update_h5_key('ROIs', roi_mask.T, overwrite=overwrite) #transpose for H5 format
            if not bool:
                print("H5 file key 'ROIs'not updated, due to overwrite=False.")
        
    def segment_rois(self, mode="cellpose+", overwrite = True, **kwargs):
        """
        Segment ROIs using automated methods.

        Parameters
        ----------
        mode : str
            Segmentation mode:
            - "cellpose+": Cellpose with post-processing heuristics (recommended)
            - "cellpose": Raw Cellpose output only
        model_path : str or Path, optional
            Direct path to a trained Cellpose model file
        model_dir : str or Path, optional
            Directory to search for trained models
        preview : bool
            If True, return masks without updating data.rois
        overwrite : bool
            If True, overwrite existing ROIs in H5 file
        **kwargs
            Additional parameters (see pygor.segmentation.segment_rois for full list)

        Returns
        -------
        masks : ndarray (only if preview=True)
            ROI mask in pygor format

        Examples
        --------
        >>> data.segment_rois(model_dir="./models/synaptic")
        >>> masks = data.segment_rois(model_dir="./models/synaptic", preview=True)
        """
        from pygor.segmentation import segment_rois as _segment_rois
        roi_mask = _segment_rois(self, mode=mode, overwrite=overwrite, **kwargs)
        self.update_rois(roi_mask, overwrite=overwrite)
        return roi_mask

    def view_images_interactive(self, **kwargs):
        """
        View the image stack interactively using Napari.

        Parameters:
        -----------
        **kwargs : dict
            Additional keyword arguments passed to Napari viewer
        """
        session = pygor.core.gui.methods.NapariViewStack(self, **kwargs)
        session.run()

    def draw_rois(self, attribute = "calculate_image_average", style = "stacked", load_existing_rois=True, overwrite=True, show_correlation=False, **kwargs):
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
        def call_method(obj, method_str, *args, **kwargs):
            # Extract method name by stripping trailing parentheses (if present)
            method_name = method_str.split('(')[0].strip()  # Handles "method" or "method()"
            method = getattr(obj, method_name)  # Get the method from the object
            print(method)
            if attribute in obj.__dict__:
                return obj.__getattribute__(attribute)
            else:
                return method(*args, **kwargs)  # Call the method with arguments

        target = call_method(self, attribute)

        # If target is None (e.g., no repetitions), fall back to raw images
        if target is None:
            print("No averaged data available (likely no repetitions). Using raw images instead.")
            target = self.images

        # Check if existing ROIs should be loaded
        existing_roi_mask = None
        if load_existing_rois and hasattr(self, 'rois') and self.rois is not None:
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
            **kwargs
        )
        traces = session.run()

        # Save ROI mask if overwrite is True
        if overwrite:
            # Check if user actually modified ROIs in Napari
            if hasattr(session, 'rois_were_modified') and not session.rois_were_modified:
                # ROIs were not modified - use original mask to prevent growth
                print("No changes detected - ROIs not overwritten")
            else:
                # ROIs were modified - convert and save
                napari_mask = session.mask
                igor_style_mask = session.convert_napari_mask_to_igor_format(napari_mask)
                # Postpone saving to H5 until user executes save operation, but update object attributes

                # success = self.update_h5_key('ROIs', h5_mask, overwrite=True)
                # if success:
                self.rois = igor_style_mask
                self.num_rois = len(np.unique(igor_style_mask)[np.unique(igor_style_mask) < 0])
                print(f"Successfully saved {self.num_rois} ROIs to H5 file")

                # Recompute dependent data since ROIs changed
                print("\nRecomputing traces, snippets, and averages for new ROIs...")

                # Compute both raw and z-normalized traces
                self.compute_traces_from_rois(overwrite=True)

                # Compute snippets and averages
                self.compute_snippets_and_averages(overwrite=True)

                # Verify shapes match
                print("\nVerifying data integrity:")
                print(f"  num_rois: {self.num_rois}")
                print(f"  traces_raw shape: {self.traces_raw.shape if self.traces_raw is not None else 'None'}")
                print(f"  averages shape: {self.averages.shape if self.averages is not None else 'None'}")
                print(f"  snippets shape: {self.snippets.shape if self.snippets is not None else 'None'}")

                print("All dependent data recomputed and saved successfully")
            # else:
            #     print("Failed to save ROIs to H5 file")

        # Plot traces if requested
        if kwargs.get('plot', False):
            print("\nGenerating traces plot...")

            if overwrite and self.traces_znorm is not None:
                # Saved mode: plot from computed traces
                self._plot_traces(style=style, session=session)
            elif hasattr(session, 'mask') and session.mask is not None:
                # Preview mode: plot from temporary mask
                target_images = self.images if target is self.images else target
                self._plot_traces(style=style, session=session,
                                roi_mask=session.mask, images=target_images)
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

            # Extract traces using parallel method
            traces_raw = pygor.core.gui.methods._fetch_traces_parallel(images, roi_mask)
            # Returns (n_rois, n_frames) - IGOR convention

            n_rois, n_frames = traces_raw.shape

            # Compute z-scores using same baseline logic as compute_traces_from_rois()
            traces_znorm = np.zeros_like(traces_raw)

            # Get baseline parameters
            try:
                ignore_first_seconds = self.os_parameters['Ignore1stXseconds']
                baseline_seconds = self.os_parameters['Baseline_nSeconds']
                line_duration = self.os_parameters['LineDuration']
                n_lines = images.shape[1]  # nY from image dimensions
            except (AttributeError, KeyError, TypeError):
                # Fallback if OS_Parameters not available
                print("Warning: OS_Parameters not available, using default baseline settings")
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
                print("No traces available - call compute_traces_from_rois() first or provide roi_mask and images for preview")
                return

            # Check if traces shape matches current ROI count (staleness check)
            if self.rois is not None:
                current_roi_count = len(np.unique(self.rois)[np.unique(self.rois) < 0])
                trace_roi_count = self.traces_znorm.shape[0]  # First dimension is n_rois
                if trace_roi_count != current_roi_count:
                    print(f"WARNING: Trace count ({trace_roi_count}) doesn't match ROI count ({current_roi_count}).")
                    print("Traces may be stale from H5 file. Restart kernel or call compute_traces_from_rois(overwrite=True).")

            traces_plot = self.traces_znorm  # Already (n_rois, n_frames) from IGOR convention
            avg_img = np.mean(self.images, axis=0)

        # Create plot
        fig, ax = plt.subplots(2, 1, figsize=(10, 4))
        colormap = plt.cm.rainbow(np.linspace(0, 1, len(traces_plot)))

        # Top panel: Show average image with ROI overlay
        ax[0].imshow(avg_img, cmap="Greys_r", origin='lower')
        if session is not None and hasattr(session, 'mask'):
            ax[0].imshow(session.mask, cmap="rainbow", alpha=0.25, origin='lower')
        elif roi_mask is not None:
            ax[0].imshow(roi_mask, cmap="rainbow", alpha=0.25, origin='lower')
        ax[0].set_title('ROIs')
        ax[0].axis('off')

        # Bottom panel: Plot z-normalized traces
        if style == "stacked":
            for n, trace in enumerate(traces_plot):
                ax[1].plot(trace, color=colormap[-n], alpha=0.7, linewidth=0.5)
            ax[1].set_ylabel('Z-score', fontsize=10)
            ax[1].set_xlabel('Frame', fontsize=10)
            ax[1].set_title('Z-normalized traces (baseline corrected)')

        elif style == "raster":
            im = ax[1].imshow(traces_plot, aspect="auto", cmap="RdBu_r",
                            interpolation="none", origin='lower')
            ax[1].set_ylabel('ROI #', fontsize=10)
            ax[1].set_xlabel('Frame', fontsize=10)
            ax[1].set_title('Z-normalized traces (baseline corrected)')
            plt.colorbar(im, ax=ax[1], label='Z-score')

        plt.tight_layout()
        plt.show()

    def compute_correlation_projection(self, include_diagonals=True, n_jobs=-1, overwrite=False, force_recompute=False):
        """
        Compute pixel-wise temporal correlation with neighboring pixels.

        This creates a correlation map useful for visualizing functional connectivity
        in 2-photon calcium imaging data. Each pixel's correlation is computed as the
        average correlation coefficient with its immediate neighbors (4 or 8 neighbors).

        Parameters:
        -----------
        include_diagonals : bool, optional
            If True, uses 8-neighbor connectivity (including diagonals).
            If False, uses 4-neighbor connectivity (only cardinal directions).
            Default: True
        n_jobs : int, optional
            Number of parallel jobs to run. -1 uses all available CPUs.
            Default: -1
        overwrite : bool, optional
            If True, saves the result to H5 file, overwriting existing data.
            Default: False
        force_recompute : bool, optional
            If True, recomputes even if correlation_projection already exists.
            Useful for testing or when you want to recompute without saving to H5.
            Default: False

        Returns:
        --------
        np.ndarray
            Correlation projection with shape (height, width).
            Values range from -1 to 1, representing average correlation with neighbors.
        """
        # Check if already computed and not forcing recompute
        if not force_recompute and not overwrite and self.correlation_projection is not None:
            print("Correlation projection already exists. Use force_recompute=True or overwrite=True to recompute.")
            return self.correlation_projection

        if self.images is None:
            raise ValueError("No image data available. Cannot compute correlation projection.")

        # Get image dimensions
        images = self.images
        n_frames, height, width = images.shape

        # Define neighbor offsets
        if include_diagonals:
            # 8-connectivity (all surrounding pixels)
            neighbor_offsets = [(-1, -1), (-1, 0), (-1, 1),
                              (0, -1),           (0, 1),
                              (1, -1),  (1, 0),  (1, 1)]
        else:
            # 4-connectivity (cardinal directions only)
            neighbor_offsets = [(-1, 0), (0, -1), (0, 1), (1, 0)]

        def compute_pixel_correlation(y, x):
            """Compute correlation for a single pixel with its neighbors."""
            pixel_trace = images[:, y, x]
            correlations = []

            for dy, dx in neighbor_offsets:
                ny, nx = y + dy, x + dx
                # Check if neighbor is within bounds
                if 0 <= ny < height and 0 <= nx < width:
                    neighbor_trace = images[:, ny, nx]
                    # Compute Pearson correlation
                    corr = np.corrcoef(pixel_trace, neighbor_trace)[0, 1]
                    if not np.isnan(corr):
                        correlations.append(corr)

            # Return mean correlation with neighbors
            return np.mean(correlations) if correlations else 0.0

        # Create list of all pixel coordinates
        pixel_coords = [(y, x) for y in range(height) for x in range(width)]

        # Compute correlations in parallel
        if n_jobs == 1 or len(pixel_coords) <= 100:
            # Sequential computation for small images or when explicitly requested
            correlation_values = [compute_pixel_correlation(y, x) for y, x in pixel_coords]
        else:
            # Parallel computation using joblib
            from joblib import Parallel, delayed
            print(f"Computing correlation projection using {n_jobs if n_jobs > 0 else 'all'} CPU cores...")
            correlation_values = Parallel(n_jobs=n_jobs, verbose=1)(
                delayed(compute_pixel_correlation)(y, x) for y, x in pixel_coords
            )

        # Reshape to 2D image
        correlation_projection = np.array(correlation_values).reshape(height, width)

        # Save to object attribute
        self.correlation_projection = correlation_projection

        # Optionally save to H5 file if overwrite is True
        if overwrite:
            success = self.update_h5_key('correlation_projection', correlation_projection, overwrite=True)
            if success:
                print("Successfully saved correlation projection to H5 file")
            else:
                print("Failed to save correlation projection to H5 file")

        return correlation_projection

    def compute_traces_from_rois(self, overwrite=False):
        """
        Compute ROI traces from images and ROI mask.

        Extracts the average fluorescence signal for each ROI across all frames.
        Always computes BOTH raw and z-normalized traces to ensure consistency.
        Mimics IGOR's OS_TracesAndTriggers functionality for trace extraction.
        Uses parallel processing for speed and IGOR's baseline z-normalization method.

        Parameters:
        -----------
        overwrite : bool, optional
            If True, saves traces to H5 file (default: False)

        Returns:
        --------
        tuple
            (traces_raw, traces_znorm) both with shape (n_frames, n_rois)
        """
        if self.images is None:
            raise ValueError("No image data available. Cannot compute traces.")
        if self.rois is None:
            raise ValueError("No ROIs defined. Cannot compute traces.")

        # Use pygor.core.gui.methods parallel extraction (fast, handles uint16 overflow)
        import pygor.core.gui.methods

        print(f"Computing traces using parallel extraction...")

        # Extract raw traces using the fast parallel method
        # Returns shape (n_rois, n_frames) - IGOR convention
        traces_raw = pygor.core.gui.methods._fetch_traces_parallel(self.images, self.rois)

        n_rois, n_frames = traces_raw.shape

        # Compute z-normalized traces using IGOR's baseline method
        # IGOR uses first nSeconds_prerun_reference (after ignoring first X seconds) as baseline
        traces_znorm = np.zeros_like(traces_raw)

        # Get baseline parameters from OS_Parameters
        try:
            baseline_seconds = self.__baseline_duration if hasattr(self, '_Core__baseline_duration') else 3.0
            ignore_first_seconds = self.__ignore_first_seconds if hasattr(self, '_Core__ignore_first_seconds') else 0.0
            frame_duration = 1.0 / self.frame_hz

            # Calculate baseline window (in frames)
            n_frames_ignore = int(ignore_first_seconds / frame_duration)
            n_frames_baseline = int(baseline_seconds / frame_duration)

            # Need at least 3 frames for SD calculation
            if n_frames_baseline < 3:
                n_frames_baseline = 3
            # Don't use more than half the recording
            if n_frames_baseline > n_frames // 2:
                n_frames_baseline = n_frames // 2

            baseline_start = n_frames_ignore
            baseline_end = baseline_start + n_frames_baseline

            print(f"Using baseline window: frames {baseline_start}-{baseline_end} ({n_frames_baseline} frames)")

        except:
            # Fallback: use first 10% of recording as baseline
            baseline_start = 0
            baseline_end = max(3, n_frames // 10)
            print(f"Using default baseline: first {baseline_end} frames")

        # Compute z-score for each ROI using its baseline period
        for roi_idx in range(n_rois):
            trace = traces_raw[roi_idx, :]  # Shape (n_rois, n_frames), get one ROI
            baseline = trace[baseline_start:baseline_end]
            baseline_mean = np.mean(baseline)
            baseline_std = np.std(baseline)

            # IGOR formula: (trace - baseline_mean) / baseline_std
            traces_znorm[roi_idx, :] = (trace - baseline_mean) / baseline_std

        # Set both attributes
        self.traces_raw = traces_raw.astype(np.float32)
        self.traces_znorm = traces_znorm.astype(np.float32)

        # Save to H5 if requested
        if overwrite:
            success_raw = self.update_h5_key('Traces0_raw', self.traces_raw, overwrite=True)
            success_znorm = self.update_h5_key('Traces0_znorm', self.traces_znorm, overwrite=True)
            if success_raw and success_znorm:
                print(f"Successfully saved {n_rois} raw and z-normalized traces to H5 file")
            else:
                print("Warning: Some traces failed to save to H5 file")

        return self.traces_raw, self.traces_znorm

    def compute_snippets_and_averages(self, overwrite=False):
        """
        Compute snippets and averages from ROI traces.

        Mimics IGOR's OS_BasicAveraging functionality. Snippets are individual
        stimulus repetitions, and averages are the mean across all repetitions.
        Always uses raw traces (not z-normalized) as IGOR does.

        Parameters:
        -----------
        overwrite : bool, optional
            If True, saves snippets and averages to H5 file (default: False)

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
            self.compute_traces_from_rois(overwrite=overwrite)

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
        snippet_duration_s = triggertimes[trigger_mode + ignore_first_triggers] - triggertimes[ignore_first_triggers]
        snippet_duration_frames = int(snippet_duration_s / frame_duration)

        # Calculate number of complete loops
        valid_triggers = n_triggers - ignore_first_triggers + ignore_last_triggers
        n_loops = valid_triggers // trigger_mode

        print(f"Extracting snippets: {n_triggers} triggers, {n_loops} complete loops, snippet duration: {snippet_duration_frames} frames")

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
                print(f"Warning: Loop {loop_idx} exceeds frame array, truncating at {n_loops}")
                n_loops = loop_idx
                break

            # Extract snippet for all ROIs from frame-level traces
            snippets_frames[:, loop_idx, :] = traces[start_frame:end_frame, :]

        # Trim snippets if we had to stop early
        if n_loops < snippets_frames.shape[1]:
            snippets_frames = snippets_frames[:, :n_loops, :]

        # Compute averages across loops (still at frame level)
        averages_frames = np.mean(snippets_frames, axis=1)  # Shape: (snippet_duration_frames, n_rois)

        print(f"Upsampling averages from {snippet_duration_frames} frames to line-precision")

        # NOW upsample only the averages (much more efficient!)
        # Vectorized upsampling using linear interpolation
        weights_next = np.tile(np.arange(lines_per_frame) / lines_per_frame, snippet_duration_frames - 1)
        weights_curr = 1 - weights_next
        frame_indices = np.repeat(np.arange(snippet_duration_frames - 1), lines_per_frame)

        # Upsample averages
        averages_upsampled_flat = (averages_frames[frame_indices, :] * weights_curr[:, np.newaxis] +
                                   averages_frames[frame_indices + 1, :] * weights_next[:, np.newaxis])

        # Use the actual length from interpolation (not the calculated target)
        snippet_duration_upsampled = len(averages_upsampled_flat)
        averages = averages_upsampled_flat  # Shape: (snippet_upsampled, n_rois)

        # Upsample snippets using the same interpolation approach
        print(f"Upsampling snippets from {snippet_duration_frames} frames to line-precision ({snippet_duration_upsampled} samples)")
        snippets_upsampled = np.zeros((snippet_duration_upsampled, n_loops, n_rois))

        for loop_idx in range(n_loops):
            # Upsample each loop separately
            snippets_upsampled_flat = (snippets_frames[frame_indices, loop_idx, :] * weights_curr[:, np.newaxis] +
                                       snippets_frames[frame_indices + 1, loop_idx, :] * weights_next[:, np.newaxis])
            snippets_upsampled[:, loop_idx, :] = snippets_upsampled_flat

        # For H5 storage, keep as (snippet_length, n_loops, n_rois) and (snippet_length, n_rois)
        snippets_for_h5 = snippets_upsampled
        averages_for_h5 = averages

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

        # Save to H5 file if requested
        if overwrite:
            success_snip = self.update_h5_key('Snippets0', snippets_for_h5, overwrite=True)
            success_avg = self.update_h5_key('Averages0', averages_for_h5, overwrite=True)
            success_qc = self.update_h5_key('QualityCriterion', quality_criterion, overwrite=True)

            if success_snip and success_avg and success_qc:
                print("Successfully saved snippets, averages, and quality criterion to H5 file")
            else:
                print("Warning: Some data failed to save to H5 file")

        return snippets, averages

    def plot_averages(
        self, rois=None, 
        figsize=(None, None), 
        figsize_scale=None, 
        axs=None, 
        independent_scale = False, 
        n_rois_raster = 50,
        sort_order = None,
        **kwargs,
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
        return pygor.core.plot.plot_averages(self, rois, figsize, figsize_scale, axs, independent_scale, n_rois_raster, sort_order, **kwargs)
    
    def calculate_image_average(self, ignore_skip=False):
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
        images_to_average = np.concatenate(images_to_average, axis=0)
        # Print results
        print(
            f"{len(triggers_frames)} triggers with a trigger_mode of {self.trigger_mode} gives {reps} full repetitions of {rep_delta} frames each."
        )
        # Average the images
        split_arr = np.array(np.split(images_to_average, reps))
        avg_movie = np.average(split_arr, axis=0)
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
            self.correlation_projection = pygor.core.methods.correlation_map(self.images)
        return self.correlation_projection

    def calc_mean_triggertimes(self, unit = "index"):
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
                print(
                    "WARNING: Trigger times are not evenly divisible by trigger mode"
                )
                # Determine amount of triggers to crop out to achieve loop alignment
                num_triggers_to_crop = self.triggertimes.shape[0] % self.trigger_mode
                triggertimes = self.triggertimes[:-num_triggers_to_crop]
                print(f"WARNING: Cropped {num_triggers_to_crop} triggers to achieve loop alignment")
            else:
                triggertimes = self.triggertimes
            # Calculate average trigger times in seconds
            avg_epoch_dur = np.average(np.diff(triggertimes.reshape(-1, self.trigger_mode)[:, 0]))
            epoch_reshape = triggertimes.reshape(-1, self.trigger_mode)
            temp_arr = np.empty(epoch_reshape.shape)
            for n, i in enumerate(epoch_reshape):
                temp_arr[n] = i - (avg_epoch_dur * n)
            avg_epoch_triggertimes = np.average(temp_arr, axis=0)
            markers_arr_s = avg_epoch_triggertimes# / self.linedur_s
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
        return markers_arr#.astype(int)

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
        traces_ms = zoom(self.traces_znorm, (1, zoom_factor), order=1, mode='nearest')
        
        return traces_ms
    
    @property
    def roi_centroids(self, force = True):
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