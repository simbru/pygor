from pygor.classes.core_data import Core
from dataclasses import dataclass, field
import numpy as np

from pygor.timeseries.osds.plotting import circular_directional_plots
from pygor.timeseries.osds import tuning_metrics, tuning_computation

import warnings

@dataclass(kw_only=False, repr=False)
class OSDS(Core):
    dir_num: int = field(default=None, metadata={"required": True})
    dir_phase_num: int = field(default=1)
    colour_num: int = field(default=1)
    directions_list: list = field(default=None)

    def __post_init__(self):
        if self.dir_num is None:
            raise ValueError("dir_num must be specified for OSDS data")
        
        # Warn about dir_phase_num if not explicitly set to something other than 1
        if self.dir_phase_num == 1:
            import warnings
            warnings.warn(
                "dir_phase_num is set to 1 (single phase per direction). "
                "If your experiment has multiple phases per direction (e.g., ON→OFF, OFF→ON), "
                "set dir_phase_num=2 when creating the OSDS object for proper phase-aware analysis.",
                UserWarning, stacklevel=2
            )
        
        super().__post_init__()
        # Set default directions if not provided
        if self.directions_list is None:
            if self.dir_num == 8:
                self.directions_list = [0, 180, 45, 225, 90, 270, 135, 315]
            else:
                # Default to evenly spaced
                self.directions_list = list(np.linspace(0, 360, self.dir_num, endpoint=False))
    
    def _check_split_compatibility(self, array: np.ndarray, n_splits: int, axis: int) -> tuple:
        """
        Check if array can be split evenly along specified axis and return adjusted parameters.
        
        Parameters:
        ----------
        array : np.ndarray
            Array to be split
        n_splits : int  
            Number of desired splits
        axis : int
            Axis along which to split
            
        Returns:
        -------
        tuple: (adjusted_array, actual_n_splits, remainder_elements)
        """
        axis_length = array.shape[axis]
        
        if axis_length % n_splits == 0:
            # Perfect division - no adjustment needed
            return array, n_splits, 0
        else:
            # Calculate how many elements to trim to make it divisible
            remainder = axis_length % n_splits
            trim_length = axis_length - remainder
            
            # Create slice objects for trimming
            slices = [slice(None)] * array.ndim
            slices[axis] = slice(0, trim_length)
            
            adjusted_array = array[tuple(slices)]
            
            # print(f"Warning: Trimmed {remainder} elements from axis {axis} "
            #       f"(original length: {axis_length}, new length: {trim_length}) "
            #       f"to allow even splitting into {n_splits} parts.")
            
            return adjusted_array, n_splits, remainder

    def split_snippets_chromatically(self) -> np.ndarray:
        """
        Returns snippets split by chromaticity, expect one more dimension than the averages array (repetitions).
        Handles uneven divisions by trimming excess elements.
        """
        # Remove first column (time axis) and check split compatibility
        data_to_split = self.snippets[:, :, 1:]
        adjusted_data, actual_splits, remainder = self._check_split_compatibility(
            data_to_split, self.colour_num, axis=-1
        )
        
        if remainder > 0:
            print(f"ChromaticSnippets: Lost {remainder} time points to ensure even splitting")
            
        return np.array(np.split(adjusted_data, actual_splits, axis=-1))
    
    def split_averages_chromatically(self) -> np.ndarray:
        """
        Returns averages split by chromaticity.
        Handles uneven divisions by trimming excess elements.
        """
        # Remove first column (time axis) and check split compatibility
        data_to_split = self.averages[:, 1:]
        adjusted_data, actual_splits, remainder = self._check_split_compatibility(
            data_to_split, self.colour_num, axis=-1
        )
        
        if remainder > 0:
            print(f"ChromaticAverages: Lost {remainder} time points to ensure even splitting")
            
        return np.array(np.split(adjusted_data, actual_splits, axis=-1))
    
    def split_snippets_directionally(self) -> np.ndarray:
        """
        Returns snippets split by direction, expect one more dimension than the averages array (repetitions).
        Handles uneven divisions by trimming excess elements.

        Returns:
        --------
        np.ndarray
            Shape: (n_directions, n_loops, n_rois, timepoints_per_direction)
        """
        # Snippets stored as (snippet_length, n_loops, n_rois) in memory
        # Need to split along axis=0 (snippet_length/timepoints) into directions
        adjusted_data, actual_splits, remainder = self._check_split_compatibility(
            self.snippets, self.dir_num, axis=-1
        )

        if remainder > 0:
            print(f"DirectionalSnippets: Lost {remainder} time points to ensure even splitting")

        # Split along timepoints axis (axis=0) and get shape (n_directions, timepoints_per_direction, n_loops, n_rois)
        print(adjusted_data.shape, actual_splits)
        split_data = np.array(np.split(adjusted_data, actual_splits, axis=-1))

        # Transpose to get desired output shape: (n_directions, n_loops, n_rois, timepoints_per_direction)
        return split_data.transpose(0, 2, 1, 3)

    def split_averages_directionally(self) -> np.ndarray:
        """
        Returns averages split by direction.
        Handles uneven divisions by trimming excess elements.

        Returns:
        --------
        np.ndarray
            Shape: (n_directions, n_rois, timepoints_per_direction)
        """
        # Averages stored as (n_rois, timepoints) in memory after try_fetch transpose
        # Need to split along axis=1 (timepoints) into directions
        adjusted_data, actual_splits, remainder = self._check_split_compatibility(
            self.averages, self.dir_num, axis=1
        )

        if remainder > 0:
            print(f"DirectionalAverages: Lost {remainder} time points to ensure even splitting")

        # Split along timepoints axis (axis=1) and get shape (n_directions, n_rois, timepoints_per_direction)
        split_data = np.array(np.split(adjusted_data, actual_splits, axis=1))

        # Result already has the desired shape: (n_directions, n_rois, timepoints_per_direction)
        return split_data

    def compute_tuning_function(self, roi_index=None, window=None, metric='peak', phase_num=None):
        """
        Compute tuning function for each ROI across all directions.
        
        Parameters:
        -----------
        window : int or tuple, optional
            Time window within each direction phase to analyze.
            If int, uses that many frames from start of each direction.
            If tuple (start, end), uses that slice within each direction.
            If None, uses entire duration of each direction.
            Ignored if phase_num is specified.
        metric : str or callable, optional
            Metric to compute for each direction. Built-in options:
            - 'max': maximum value
            - 'absmax': maximum absolute value  
            - 'min': minimum value
            - 'avg' or 'mean': average value
            - 'range': max - min
            - 'auc': area under curve (absolute)
            - 'peak': alias for 'absmax'
            - 'peak_positive': maximum positive value
            - 'peak_negative': minimum negative value
            Or pass a callable function that takes 1D array and returns scalar.
        roi_index : int, optional
            If specified, returns tuning for only this ROI as 1D array.
            If None, returns tuning for all ROIs as 2D array.
        phase_num : int, optional
            If specified, splits each direction duration into n equal phases.
            Returns tuning values for each phase.
            
        Returns:
        --------
        np.ndarray
            If phase_num is None:
                If roi_index is None: tuning values with shape (n_rois, n_directions).
                If roi_index is specified: tuning values with shape (n_directions,).
            If phase_num is specified:
                If roi_index is None: tuning values with shape (n_rois, n_directions, n_phases).
                If roi_index is specified: tuning values with shape (n_directions, n_phases).
            Values are ordered according to self.directions_list.
        """
        # Automatically use dir_phase_num if phase_num not specified AND no window is given.
        # When a window is explicitly provided, skip auto-inference since the caller
        # wants window-based analysis, not automatic phase splitting.
        if phase_num is None and window is None:
            phase_num = self.dir_phase_num if self.dir_phase_num > 1 else None

        return tuning_computation.compute_tuning_function(
            self, roi_index=roi_index, window=window, metric=metric, phase_num=phase_num
        )

    def compute_tuning_metrics(self, roi_indices=None, metric='peak', phase_aware=None):
        """
        Compute all directional tuning metrics for ROIs.

        This is the recommended method for efficient batch computation of
        DSI, OSI, vector magnitude, and other directional tuning metrics.

        Parameters
        ----------
        roi_indices : list, int, or None
            ROI indices to analyze. If None, analyzes all ROIs.
        metric : str or callable
            Metric for computing tuning function ('peak', 'mean', 'auc', etc.)
        phase_aware : bool or None
            Controls phase-aware analysis:
            - None (default): Auto-detect from self.dir_phase_num
              (phase-aware if dir_phase_num > 1, single-phase otherwise)
            - True: Force phase-aware analysis
            - False: Force single-phase analysis (ignore dir_phase_num)

        Returns
        -------
        dict
            Dictionary containing arrays of metrics for each ROI:

            - 'dsi': Directional selectivity index (-1 to 1)
            - 'osi': Orientation selectivity index (0 to 1)
            - 'vector_magnitude': Circular vector magnitude (0 to 1)
            - 'circular_variance': 1 - vector_magnitude (0 to 1)
            - 'preferred_direction': Direction with max response (degrees)
            - 'preferred_orientation': Orientation with max response (degrees)
            - 'mean_direction': Mean direction from circular stats (degrees)
            - 'roi_indices': ROI indices that were analyzed
            - 'n_phases': Number of phases (1 if single-phase)

            Array shapes:
            - Single-phase: (n_rois,)
            - Multi-phase: (n_phases, n_rois)

        Examples
        --------
        >>> metrics = data.compute_tuning_metrics()
        >>> dsi_values = metrics['dsi']
        >>> osi_values = metrics['osi']

        >>> # Force single-phase analysis even if dir_phase_num > 1
        >>> metrics = data.compute_tuning_metrics(phase_aware=False)
        """
        if isinstance(roi_indices, int):
            roi_indices = [roi_indices]
        if self.averages is None:
            raise ValueError("Averages data not found. Cannot compute tuning metrics.")
        return tuning_metrics.compute_all_tuning_metrics(
            self, roi_indices=roi_indices, metric=metric, phase_aware=phase_aware
        )

    # =========================================================================
    # Getter methods - return only the requested metric
    # =========================================================================

    def get_osi(self, roi_indices=None, metric='peak', phase_aware=None):
        """
        Get orientation selectivity index (OSI) for ROIs.

        Convenience method that returns only OSI values. For multiple metrics
        at once, use compute_tuning_metrics() instead.

        Parameters
        ----------
        roi_indices : list, int, or None
            ROI indices to analyze. If None, analyzes all ROIs.
        metric : str or callable
            Metric for computing tuning function ('peak', 'mean', 'auc', etc.)
        phase_aware : bool or None
            Controls phase-aware analysis (None=auto-detect, True=force phases,
            False=force single-phase).

        Returns
        -------
        np.ndarray
            OSI values (0 to 1). Shape: (n_rois,) or (n_phases, n_rois).
        """
        return self.compute_tuning_metrics(roi_indices, metric, phase_aware)['osi']

    def get_dsi(self, roi_indices=None, metric='peak', phase_aware=None):
        """
        Get directional selectivity index (DSI) for ROIs.

        Convenience method that returns only DSI values. For multiple metrics
        at once, use compute_tuning_metrics() instead.

        Parameters
        ----------
        roi_indices : list, int, or None
            ROI indices to analyze. If None, analyzes all ROIs.
        metric : str or callable
            Metric for computing tuning function ('peak', 'mean', 'auc', etc.)
        phase_aware : bool or None
            Controls phase-aware analysis (None=auto-detect, True=force phases,
            False=force single-phase).

        Returns
        -------
        np.ndarray
            DSI values (-1 to 1). Shape: (n_rois,) or (n_phases, n_rois).
        """
        return self.compute_tuning_metrics(roi_indices, metric, phase_aware)['dsi']

    def get_preferred_direction(self, roi_indices=None, metric='peak', phase_aware=None):
        """
        Get preferred direction for ROIs.

        Parameters
        ----------
        roi_indices : list, int, or None
            ROI indices to analyze. If None, analyzes all ROIs.
        metric : str or callable
            Metric for computing tuning function.
        phase_aware : bool or None
            Controls phase-aware analysis.

        Returns
        -------
        np.ndarray
            Preferred direction in degrees (0-360). Shape: (n_rois,) or (n_phases, n_rois).
        """
        return self.compute_tuning_metrics(roi_indices, metric, phase_aware)['preferred_direction']

    def get_preferred_orientation(self, roi_indices=None, metric='peak', phase_aware=None):
        """
        Get preferred orientation for ROIs.

        Parameters
        ----------
        roi_indices : list, int, or None
            ROI indices to analyze. If None, analyzes all ROIs.
        metric : str or callable
            Metric for computing tuning function.
        phase_aware : bool or None
            Controls phase-aware analysis.

        Returns
        -------
        np.ndarray
            Preferred orientation in degrees (0-180). Shape: (n_rois,) or (n_phases, n_rois).
        """
        return self.compute_tuning_metrics(roi_indices, metric, phase_aware)['preferred_orientation']

    def get_vector_magnitude(self, roi_indices=None, metric='peak', phase_aware=None):
        """
        Get vector magnitude (r) for ROIs.

        Vector magnitude measures directional tuning strength (0=no preference, 1=perfect).

        Parameters
        ----------
        roi_indices : list, int, or None
            ROI indices to analyze. If None, analyzes all ROIs.
        metric : str or callable
            Metric for computing tuning function.
        phase_aware : bool or None
            Controls phase-aware analysis.

        Returns
        -------
        np.ndarray
            Vector magnitude (0 to 1). Shape: (n_rois,) or (n_phases, n_rois).
        """
        return self.compute_tuning_metrics(roi_indices, metric, phase_aware)['vector_magnitude']

    def get_circular_variance(self, roi_indices=None, metric='peak', phase_aware=None):
        """
        Get circular variance for ROIs.

        Circular variance = 1 - vector_magnitude. Measures spread of directional response.

        Parameters
        ----------
        roi_indices : list, int, or None
            ROI indices to analyze. If None, analyzes all ROIs.
        metric : str or callable
            Metric for computing tuning function.
        phase_aware : bool or None
            Controls phase-aware analysis.

        Returns
        -------
        np.ndarray
            Circular variance (0 to 1). Shape: (n_rois,) or (n_phases, n_rois).
        """
        return self.compute_tuning_metrics(roi_indices, metric, phase_aware)['circular_variance']

    def get_mean_direction(self, roi_indices=None, metric='peak', phase_aware=None):
        """
        Get mean direction from circular statistics for ROIs.

        Parameters
        ----------
        roi_indices : list, int, or None
            ROI indices to analyze. If None, analyzes all ROIs.
        metric : str or callable
            Metric for computing tuning function.
        phase_aware : bool or None
            Controls phase-aware analysis.

        Returns
        -------
        np.ndarray
            Mean direction in degrees (0-360). Shape: (n_rois,) or (n_phases, n_rois).
        """
        return self.compute_tuning_metrics(roi_indices, metric, phase_aware)['mean_direction']

    # =========================================================================
    # Vector extraction methods (for visualization)
    # =========================================================================

    def _extract_vector_helper(self, roi_index, extract_func, metric='peak', use_phases=None):
        """
        Private helper for vector extraction methods.

        Parameters
        ----------
        roi_index : int
            ROI index to analyze.
        extract_func : callable
            Function from tuning_metrics to apply (e.g., extract_direction_vectors).
        metric : str or callable
            Metric for computing tuning function.
        use_phases : bool or None
            Phase handling (None=auto-detect).

        Returns
        -------
        dict
            Single phase: result from extract_func.
            Multi-phase: dict with 'phase_0', 'phase_1', etc. keys.
        """
        if use_phases is None:
            use_phases = self.dir_phase_num > 1

        if use_phases:
            responses = self.compute_tuning_function(roi_index=roi_index, metric=metric)
            # responses shape: (n_directions, n_phases)
            return {
                f'phase_{i}': extract_func(responses[:, i], self.directions_list)
                for i in range(responses.shape[1])
            }
        else:
            responses = self.compute_tuning_function(
                roi_index=roi_index, metric=metric, phase_num=None
            )
            return extract_func(responses, self.directions_list)

    def extract_direction_vectors(self, roi_index, metric='peak', use_phases=None):
        """
        Extract individual direction vectors for a specific ROI.

        Parameters
        ----------
        roi_index : int
            ROI index to analyze.
        metric : str or callable
            Metric for computing tuning function.
        use_phases : bool or None
            Phase handling (None=auto-detect from dir_phase_num).

        Returns
        -------
        dict
            Contains 'angles', 'magnitudes', 'cartesian_x', 'cartesian_y'.
            Multi-phase: nested dict with 'phase_0', 'phase_1', etc. keys.
        """
        return self._extract_vector_helper(
            roi_index, tuning_metrics.extract_direction_vectors, metric, use_phases
        )

    def extract_mean_vector(self, roi_index, metric='peak', use_phases=None):
        """
        Extract mean vector for a specific ROI.

        Parameters
        ----------
        roi_index : int
            ROI index to analyze.
        metric : str or callable
            Metric for computing tuning function.
        use_phases : bool or None
            Phase handling (None=auto-detect from dir_phase_num).

        Returns
        -------
        dict
            Contains 'angle', 'magnitude', 'cartesian_x', 'cartesian_y'.
            Multi-phase: nested dict with 'phase_0', 'phase_1', etc. keys.
        """
        return self._extract_vector_helper(
            roi_index, tuning_metrics.extract_mean_vector, metric, use_phases
        )

    def extract_orientation_vector(self, roi_index, metric='peak', use_phases=None):
        """
        Extract orientation vector for a specific ROI.

        Parameters
        ----------
        roi_index : int
            ROI index to analyze.
        metric : str or callable
            Metric for computing tuning function.
        use_phases : bool or None
            Phase handling (None=auto-detect from dir_phase_num).

        Returns
        -------
        dict
            Contains 'angle', 'magnitude', 'cartesian_x', 'cartesian_y'.
            Multi-phase: nested dict with 'phase_0', 'phase_1', etc. keys.
        """
        return self._extract_vector_helper(
            roi_index, tuning_metrics.extract_orientation_vector, metric, use_phases
        )
    
    # =========================================================================
    # Deprecated methods (kept for backward compatibility)
    # =========================================================================

    def _compute_single_roi_metric(self, roi_index, compute_func, metric='peak', use_phases=None):
        """
        Private helper for deprecated single-ROI metric computation.

        Parameters
        ----------
        roi_index : int
            ROI index to analyze.
        compute_func : callable
            Function from tuning_metrics to apply.
        metric : str or callable
            Metric for computing tuning function.
        use_phases : bool or None
            Phase handling (None=auto-detect).

        Returns
        -------
        dict
            Single phase: result from compute_func.
            Multi-phase: dict with 'phase_0', 'phase_1', etc. keys.
        """
        if use_phases is None:
            use_phases = self.dir_phase_num > 1

        if use_phases:
            responses = self.compute_tuning_function(roi_index=roi_index, metric=metric)
            return {
                f'phase_{i}': compute_func(responses[:, i], self.directions_list)
                for i in range(responses.shape[1])
            }
        else:
            responses = self.compute_tuning_function(
                roi_index=roi_index, metric=metric, phase_num=None
            )
            return compute_func(responses, self.directions_list)

    def compute_osi(self, roi_index=None, metric='peak', use_phases=None):
        """
        DEPRECATED: Use get_osi() or compute_tuning_metrics()['osi'] instead.
        """
        warnings.warn(
            "compute_osi() is deprecated. Use get_osi() for just OSI values, "
            "or compute_tuning_metrics()['osi'] for all metrics.",
            DeprecationWarning,
            stacklevel=2
        )
        if roi_index is None:
            return self.compute_tuning_metrics(metric=metric, phase_aware=use_phases)
        return self._compute_single_roi_metric(
            roi_index, tuning_metrics.compute_orientation_selectivity_index, metric, use_phases
        )

    def compute_dsi(self, roi_index=None, metric='peak', use_phases=None):
        """
        DEPRECATED: Use get_dsi() or compute_tuning_metrics()['dsi'] instead.
        """
        warnings.warn(
            "compute_dsi() is deprecated. Use get_dsi() for just DSI values, "
            "or compute_tuning_metrics()['dsi'] for all metrics.",
            DeprecationWarning,
            stacklevel=2
        )
        if roi_index is None:
            return self.compute_tuning_metrics(metric=metric, phase_aware=use_phases)
        return self._compute_single_roi_metric(
            roi_index, tuning_metrics.compute_direction_selectivity_index, metric, use_phases
        )

    def compute_orientation_selectivity_index(self, roi_index, metric='peak', use_phases=None):
        """
        DEPRECATED: Use get_osi() instead.
        """
        warnings.warn(
            "compute_orientation_selectivity_index() is deprecated. Use get_osi() instead.",
            DeprecationWarning,
            stacklevel=2
        )
        return self._compute_single_roi_metric(
            roi_index, tuning_metrics.compute_orientation_selectivity_index, metric, use_phases
        )

    def compute_phase_tuning_metrics(self, metric='peak', roi_indices=None,
                                      phase_ranges=None):
        """
        DEPRECATED: Use compute_tuning_metrics(phase_aware=True) instead.

        Compute directional tuning metrics with automatic phase analysis.

        Parameters
        ----------
        metric : str or callable
            Metric to use for computing tuning functions (default 'peak').
        roi_indices : list or None
            ROI indices to analyze. If None, analyzes all ROIs.
        phase_ranges : list of tuples or None
            Ignored in new API. Use compute_tuning_metrics(phase_aware=True).

        Returns
        -------
        dict
            Dictionary containing tuning metrics with (n_phases, n_rois) arrays.
        """
        warnings.warn(
            "compute_phase_tuning_metrics() is deprecated. "
            "Use compute_tuning_metrics(phase_aware=True) instead.",
            DeprecationWarning,
            stacklevel=2
        )
        # Map old API to new API
        # phase_ranges parameter is ignored - always use auto-detection
        return self.compute_tuning_metrics(
            roi_indices=roi_indices, metric=metric, phase_aware=True
        )
    
    def plot_tuning_function_with_traces(self, roi_index, ax=None, show_trials=True, 
                                       metric='peak', trace_scale=0.25, minimal=True, 
                                       polar_color=None, trace_alpha=0.7, use_phases=None, 
                                       phase_colors=None, orbit_distance=0.5, trace_aspect_x=1.0, 
                                       trace_aspect_y=1.0, separate_phase_axes=False, **kwargs):
        """
        Plot tuning function with floating trace snippets in external axes.
        
        Parameters:
        -----------
        roi_index : int
            ROI index to analyze
        ax : matplotlib.axes.Axes, array/list of matplotlib.axes.Axes, or None
            External axes to plot within. If separate_phase_axes=True, should be 
            array/list with one axes per phase. If None, creates axes automatically.
        show_trials : bool
            Whether to show individual trial traces (default True)
        metric : str
            Summary metric ('peak', 'auc', 'mean', etc.)
        trace_scale : float
            Scale factor for trace subplot size (default 0.25)
        minimal : bool
            Use minimal styling (no titles, labels) (default True)
        polar_color : str
            Color for central polar plot (default '#2E8B57')
        trace_alpha : float
            Alpha for trace plots (default 0.7)
        use_phases : bool or None
            If None, uses self.dir_phase_num > 1 to decide
            If True, forces phase analysis  
            If False, forces single-phase analysis
        phase_colors : list or None
            Colors for each phase. If None, uses default colors
        orbit_distance : float
            Distance of trace orbit from polar plot center (default 0.5)
        trace_aspect_x : float
            Horizontal scaling factor for individual trace plots (default 1.0)
        trace_aspect_y : float
            Vertical scaling factor for individual trace plots (default 1.0)
        separate_phase_axes : bool
            If True, create separate polar + orbit plots for each phase.
            If False, overlay all phases on single polar + orbit plots (default False)
        **kwargs
            Additional arguments (data_crop, etc.)
            
        Returns:
        --------
        fig : matplotlib.figure.Figure
            The figure object
        ax_polar : matplotlib.axes.Axes or list of matplotlib.axes.Axes
            If separate_phase_axes=False: single polar plot axes
            If separate_phase_axes=True: list of polar axes (one per phase)
        """
        if use_phases is None:
            if self.dir_phase_num > 1:
                use_phases = True
        return circular_directional_plots.plot_tuning_function_with_traces(
            self, roi_index, ax, show_trials=show_trials, metric=metric, 
            trace_scale=trace_scale, minimal=minimal, polar_color=polar_color, 
            trace_alpha=trace_alpha, use_phases=use_phases, phase_colors=phase_colors,
            orbit_distance=orbit_distance, trace_aspect_x=trace_aspect_x,
            trace_aspect_y=trace_aspect_y, separate_phase_axes=separate_phase_axes, **kwargs
        )


# Backward compatibility alias
MovingBars = OSDS
