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
    tuning_metric : str = field(default='range')
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
#         self.__von_mises_results = None
        self.__tuning_metrics_results = None
        # Set default directions if not provided
        if self.directions_list is None: 
            if self.dir_num == 8: # QDSpy standard
                # self.directions_list = [0, 180, 45, 225, 90, 270, 135, 315] # this was taken from QDSpy but screen is flipped so 
                self.directions_list = [180, 0, 225, 45, 270, 90, 315, 135]  # Adjusted for flipped screen
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

    def compute_tuning_function(self, roi_index=None, window=None, metric=None, phase_num=None):
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
            If None, uses self.tuning_metric (default 'auc').
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
        # Use instance attribute if metric not specified
        if metric is None:
            metric = self.tuning_metric

        # Automatically use dir_phase_num if phase_num not specified AND no window is given.
        # When a window is explicitly provided, skip auto-inference since the caller
        # wants window-based analysis, not automatic phase splitting.
        if phase_num is None and window is None:
            phase_num = self.dir_phase_num if self.dir_phase_num > 1 else None

        return tuning_computation.compute_tuning_function(
            self, roi_index=roi_index, window=window, metric=metric, phase_num=phase_num
        )

    def compute_tuning_metrics(
        self,
        roi_indices=None,
        metric=None,
        phase_aware=None,
#         include_vonmises=False,
#         r_squared_threshold=0.8,
    ):
        """
        Compute all directional tuning metrics for ROIs.

        This is the recommended method for efficient batch computation of
        DSI, OSI, and both circular-statistics and argmax-based metrics.

        Two internally consistent frameworks are returned:
        - Circular statistics: mean_direction + direction_vector_magnitude,
          mean_orientation + orientation_vector_magnitude
        - Argmax/pairwise: preferred_direction + dsi, preferred_orientation + osi

        Parameters
        ----------
        roi_indices : list, int, or None
            ROI indices to analyze. If None, analyzes all ROIs.
        metric : str or callable
            Metric for computing tuning function ('peak', 'mean', 'auc', etc.).
            If None, uses self.tuning_metric.
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

            Circular statistics framework:
            - 'direction_vector_magnitude': Vector magnitude for direction (0 to 1)
            - 'orientation_vector_magnitude': Vector magnitude for orientation (0 to 1)
            - 'mean_direction': Mean direction from circular stats (degrees)
            - 'mean_orientation': Mean orientation from circular stats (degrees)
            - 'preferred_direction_vector_sum': Alias of mean_direction (degrees)
            - 'preferred_orientation_vector_sum': Alias of mean_orientation (degrees)
            - 'circular_variance': 1 - direction_vector_magnitude (0 to 1)

            Argmax/pairwise framework:
            - 'dsi': Directional selectivity index (-1 to 1)
            - 'osi': Orientation selectivity index (0 to 1)
            - 'preferred_direction': Direction with max response (degrees)
            - 'preferred_orientation': Orientation with max response (degrees)

            Backward compatibility (deprecated):
            - 'vector_magnitude': Alias for 'direction_vector_magnitude'
            - 'roi_indices': ROI indices that were analyzed
            - 'n_phases': Number of phases (1 if single-phase)

        Array shapes:
        - Single-phase: (n_rois,)
        - Multi-phase: (n_phases, n_rois)

        Notes
        -----
        Results are cached when parameters match; repeated calls reuse cached outputs.

        Examples
        --------
        >>> metrics = data.compute_tuning_metrics()
        >>> dsi_values = metrics['dsi']
        >>> osi_values = metrics['osi']

        >>> # Force single-phase analysis even if dir_phase_num > 1
        >>> metrics = data.compute_tuning_metrics(phase_aware=False)
        """
        if metric is None:
            metric = self.tuning_metric
        if isinstance(roi_indices, int):
            roi_indices = [roi_indices]
        if self.averages is None:
            raise ValueError("Averages data not found. Cannot compute tuning metrics.")

        if self._can_use_tuning_metrics_cache(
            roi_indices=roi_indices,
            metric=metric,
            phase_aware=phase_aware,
#             include_vonmises=include_vonmises,
#             r_squared_threshold=r_squared_threshold,
        ):
            cached = self._get_cached_tuning_metrics_results()
            meta = cached.get("_meta", {})
            return self._slice_tuning_metrics_results(
                cached,
                roi_indices=roi_indices,
                cached_roi_indices=meta.get("roi_indices"),
            )

        results = tuning_metrics.compute_all_tuning_metrics(
            self,
            roi_indices=roi_indices,
            metric=metric,
            phase_aware=phase_aware,
#             include_vonmises=include_vonmises,
#             r_squared_threshold=r_squared_threshold,
        )
        cached_roi_indices = None
        if roi_indices is not None:
            cached_roi_indices = list(np.atleast_1d(roi_indices))
        self._set_cached_tuning_metrics_results({
            **results,
            "_meta": {
                "roi_indices": cached_roi_indices,
                "metric": metric,
                "phase_aware": phase_aware,
#                 "include_vonmises": include_vonmises,
#                 "r_squared_threshold": r_squared_threshold,
            },
        })

        return results

    # =========================================================================
    # Helper methods
    # =========================================================================

    def _extract_phase(self, result, phase_idx):
        """
        Extract specific phase from multi-phase result.

        Parameters
        ----------
        result : np.ndarray
            Result array, either (n_rois,) or (n_phases, n_rois)
        phase_idx : int or None
            Phase index to extract. If None, returns result unchanged.

        Returns
        -------
        np.ndarray
            If phase_idx is None: original result
            If phase_idx specified: result[phase_idx] with shape (n_rois,)

        Raises
        ------
        IndexError
            If phase_idx is out of range for the number of phases
        """
        if phase_idx is None:
            return result
        if not hasattr(result, 'ndim') or result.ndim != 2:
            # Single-phase result, phase_idx only valid if 0
            if phase_idx != 0:
                raise IndexError(f"phase_idx={phase_idx} invalid for single-phase result")
            return result
        if phase_idx < 0 or phase_idx >= result.shape[0]:
            raise IndexError(f"phase_idx={phase_idx} out of range for {result.shape[0]} phases")
        return result[phase_idx]

    # =========================================================================
    # Getter methods - return only the requested metric
    # =========================================================================

    def get_osi(self, roi_indices=None, metric=None, phase_aware=None, phase_idx=None):
        """
        Get orientation selectivity index (OSI) for ROIs.

        Convenience method that returns only OSI values. For multiple metrics
        at once, use compute_tuning_metrics() instead.

        Parameters
        ----------
        roi_indices : list, int, or None
            ROI indices to analyze. If None, analyzes all ROIs.
        metric : str or callable
            Metric for computing tuning function ('peak', 'mean', 'auc', etc.).
            If None, uses self.tuning_metric.
        phase_aware : bool or None
            Controls phase-aware analysis (None=auto-detect, True=force phases,
            False=force single-phase).
        phase_idx : int or None
            If specified, extract only this phase from multi-phase results.
            Returns (n_rois,) instead of (n_phases, n_rois).

        Returns
        -------
        np.ndarray
            OSI values (0 to 1). Shape: (n_rois,) or (n_phases, n_rois).
            If phase_idx specified: always (n_rois,).
        """
        result = self.compute_tuning_metrics(roi_indices, metric, phase_aware)['osi']
        return self._extract_phase(result, phase_idx)

    def get_dsi(self, roi_indices=None, metric=None, phase_aware=None, phase_idx=None):
        """
        Get directional selectivity index (DSI) for ROIs.

        Convenience method that returns only DSI values. For multiple metrics
        at once, use compute_tuning_metrics() instead.

        Parameters
        ----------
        roi_indices : list, int, or None
            ROI indices to analyze. If None, analyzes all ROIs.
        metric : str or callable
            Metric for computing tuning function ('peak', 'mean', 'auc', etc.).
            If None, uses self.tuning_metric.
        phase_aware : bool or None
            Controls phase-aware analysis (None=auto-detect, True=force phases,
            False=force single-phase).
        phase_idx : int or None
            If specified, extract only this phase from multi-phase results.
            Returns (n_rois,) instead of (n_phases, n_rois).

        Returns
        -------
        np.ndarray
            DSI values (-1 to 1). Shape: (n_rois,) or (n_phases, n_rois).
            If phase_idx specified: always (n_rois,).
        """
        result = self.compute_tuning_metrics(roi_indices, metric, phase_aware)['dsi']
        return self._extract_phase(result, phase_idx)

    def get_preferred_direction(self, roi_indices=None, metric=None, phase_aware=None, phase_idx=None):
        """
        Get preferred direction for ROIs.

        Parameters
        ----------
        roi_indices : list, int, or None
            ROI indices to analyze. If None, analyzes all ROIs.
        metric : str or callable
            Metric for computing tuning function.
            If None, uses self.tuning_metric.
        phase_aware : bool or None
            Controls phase-aware analysis.
        phase_idx : int or None
            If specified, extract only this phase from multi-phase results.

        Returns
        -------
        np.ndarray
            Preferred direction in degrees (0-360). Shape: (n_rois,) or (n_phases, n_rois).
            If phase_idx specified: always (n_rois,).
        """
        result = self.compute_tuning_metrics(roi_indices, metric, phase_aware)['preferred_direction']
        return self._extract_phase(result, phase_idx)

    def get_preferred_direction_vector_sum(self, roi_indices=None, metric=None, phase_aware=None, phase_idx=None):
        """
        Get preferred direction using vector-sum (circular mean).

        This returns the angle of the mean resultant vector (0-360),
        computed from all directions (continuous, not binned).

        Parameters
        ----------
        roi_indices : list, int, or None
            ROI indices to analyze. If None, analyzes all ROIs.
        metric : str or callable
            Metric for computing tuning function.
            If None, uses self.tuning_metric.
        phase_aware : bool or None
            Controls phase-aware analysis.
        phase_idx : int or None
            If specified, extract only this phase from multi-phase results.

        Returns
        -------
        np.ndarray
            Preferred direction in degrees (0-360) from vector sum.
            Shape: (n_rois,) or (n_phases, n_rois).
            If phase_idx specified: always (n_rois,).
        """
        result = self.compute_tuning_metrics(roi_indices, metric, phase_aware)['mean_direction']
        return self._extract_phase(result, phase_idx)

    def get_preferred_orientation(self, roi_indices=None, metric=None, phase_aware=None, phase_idx=None):
        """
        Get preferred orientation for ROIs.

        Parameters
        ----------
        roi_indices : list, int, or None
            ROI indices to analyze. If None, analyzes all ROIs.
        metric : str or callable
            Metric for computing tuning function.
            If None, uses self.tuning_metric.
        phase_aware : bool or None
            Controls phase-aware analysis.
        phase_idx : int or None
            If specified, extract only this phase from multi-phase results.

        Returns
        -------
        np.ndarray
            Preferred orientation in degrees (0-180). Shape: (n_rois,) or (n_phases, n_rois).
            If phase_idx specified: always (n_rois,).
        """
        result = self.compute_tuning_metrics(roi_indices, metric, phase_aware)['preferred_orientation']
        return self._extract_phase(result, phase_idx)

    def get_preferred_orientation_vector_sum(self, roi_indices=None, metric=None, phase_aware=None, phase_idx=None):
        """
        Get preferred orientation using vector-sum (circular mean).

        This returns the orientation of the mean resultant vector (0-180),
        computed from all directions using the doubled-angle method.

        Parameters
        ----------
        roi_indices : list, int, or None
            ROI indices to analyze. If None, analyzes all ROIs.
        metric : str or callable
            Metric for computing tuning function.
            If None, uses self.tuning_metric.
        phase_aware : bool or None
            Controls phase-aware analysis.
        phase_idx : int or None
            If specified, extract only this phase from multi-phase results.

        Returns
        -------
        np.ndarray
            Preferred orientation in degrees (0-180) from vector sum.
            Shape: (n_rois,) or (n_phases, n_rois).
            If phase_idx specified: always (n_rois,).
        """
        result = self.compute_tuning_metrics(roi_indices, metric, phase_aware)['mean_orientation']
        return self._extract_phase(result, phase_idx)

    def get_vector_magnitude(self, roi_indices=None, metric=None, phase_aware=None, phase_idx=None):
        """
        Get vector magnitude (r) for ROIs.

        .. deprecated::
            Use :meth:`get_direction_vector_magnitude` instead for clarity.
            This method will be removed in a future version.

        Vector magnitude measures directional tuning strength (0=no preference, 1=perfect).

        Parameters
        ----------
        roi_indices : list, int, or None
            ROI indices to analyze. If None, analyzes all ROIs.
        metric : str or callable
            Metric for computing tuning function.
            If None, uses self.tuning_metric.
        phase_aware : bool or None
            Controls phase-aware analysis.
        phase_idx : int or None
            If specified, extract only this phase from multi-phase results.

        Returns
        -------
        np.ndarray
            Vector magnitude (0 to 1). Shape: (n_rois,) or (n_phases, n_rois).
            If phase_idx specified: always (n_rois,).

        See Also
        --------
        get_direction_vector_magnitude : Recommended replacement
        get_orientation_vector_magnitude : For orientation selectivity
        """
        warnings.warn(
            "get_vector_magnitude() is deprecated and will be removed in a future version. "
            "Use get_direction_vector_magnitude() for direction selectivity, "
            "or get_orientation_vector_magnitude() for orientation selectivity.",
            DeprecationWarning,
            stacklevel=2
        )
        result = self.compute_tuning_metrics(roi_indices, metric, phase_aware)['direction_vector_magnitude']
        return self._extract_phase(result, phase_idx)

    def get_circular_variance(self, roi_indices=None, metric=None, phase_aware=None, phase_idx=None):
        """
        Get circular variance for ROIs.

        Circular variance = 1 - direction_vector_magnitude. Measures spread of directional response.

        Parameters
        ----------
        roi_indices : list, int, or None
            ROI indices to analyze. If None, analyzes all ROIs.
        metric : str or callable
            Metric for computing tuning function.
            If None, uses self.tuning_metric.
        phase_aware : bool or None
            Controls phase-aware analysis.
        phase_idx : int or None
            If specified, extract only this phase from multi-phase results.

        Returns
        -------
        np.ndarray
            Circular variance (0 to 1). Shape: (n_rois,) or (n_phases, n_rois).
            If phase_idx specified: always (n_rois,).
        """
        result = self.compute_tuning_metrics(roi_indices, metric, phase_aware)['circular_variance']
        return self._extract_phase(result, phase_idx)

    def get_mean_direction(self, roi_indices=None, metric=None, phase_aware=None, phase_idx=None):
        """
        Get mean direction from circular statistics for ROIs.

        Parameters
        ----------
        roi_indices : list, int, or None
            ROI indices to analyze. If None, analyzes all ROIs.
        metric : str or callable
            Metric for computing tuning function.
            If None, uses self.tuning_metric.
        phase_aware : bool or None
            Controls phase-aware analysis.
        phase_idx : int or None
            If specified, extract only this phase from multi-phase results.

        Returns
        -------
        np.ndarray
            Mean direction in degrees (0-360). Shape: (n_rois,) or (n_phases, n_rois).
            If phase_idx specified: always (n_rois,).
        """
        result = self.compute_tuning_metrics(roi_indices, metric, phase_aware)['mean_direction']
        return self._extract_phase(result, phase_idx)

    def get_direction_vector_magnitude(self, roi_indices=None, metric=None, phase_aware=None, phase_idx=None):
        """
        Get direction vector magnitude (r) for ROIs.

        Direction vector magnitude measures directional tuning strength using circular
        statistics in 360-degree space. This is the length of the mean resultant vector
        when treating each direction's response as a vector.

        r = 0 means no directional preference (responses equal in all directions)
        r = 1 means perfect directional tuning (all response in one direction)

        This is part of the **circular statistics framework** along with:
        - get_mean_direction(): angle of the mean vector
        - get_circular_variance(): 1 - direction_vector_magnitude

        Parameters
        ----------
        roi_indices : list, int, or None
            ROI indices to analyze. If None, analyzes all ROIs.
        metric : str or callable
            Metric for computing tuning function ('peak', 'mean', 'auc', etc.).
            If None, uses self.tuning_metric.
        phase_aware : bool or None
            Controls phase-aware analysis.
        phase_idx : int or None
            If specified, extract only this phase from multi-phase results.

        Returns
        -------
        np.ndarray
            Direction vector magnitude (0 to 1). Shape: (n_rois,) or (n_phases, n_rois).
            If phase_idx specified: always (n_rois,).

        See Also
        --------
        get_orientation_vector_magnitude : For orientation selectivity
        get_dsi : For argmax-based directional selectivity index
        """
        result = self.compute_tuning_metrics(roi_indices, metric, phase_aware)['direction_vector_magnitude']
        return self._extract_phase(result, phase_idx)

    def get_orientation_vector_magnitude(self, roi_indices=None, metric=None, phase_aware=None, phase_idx=None):
        """
        Get orientation vector magnitude (r) for ROIs.

        Orientation vector magnitude measures orientation tuning strength using
        circular statistics with the doubled-angle method in 180-degree space.
        Opposite directions (e.g., 0 and 180) are treated as the same orientation.

        r = 0 means no orientation preference
        r = 1 means perfect orientation tuning

        This is part of the **circular statistics framework** along with:
        - get_mean_orientation(): angle of the mean orientation vector
        - get_direction_vector_magnitude(): direction selectivity in 360 space

        Parameters
        ----------
        roi_indices : list, int, or None
            ROI indices to analyze. If None, analyzes all ROIs.
        metric : str or callable
            Metric for computing tuning function ('peak', 'mean', 'auc', etc.).
            If None, uses self.tuning_metric.
        phase_aware : bool or None
            Controls phase-aware analysis.
        phase_idx : int or None
            If specified, extract only this phase from multi-phase results.

        Returns
        -------OSDS\OSDS_population.py
        np.ndarray
            Orientation vector magnitude (0 to 1). Shape: (n_rois,) or (n_phases, n_rois).
            If phase_idx specified: always (n_rois,).

        See Also
        --------
        get_direction_vector_magnitude : For direction selectivity
        get_osi : For argmax-based orientation selectivity index
        """
        result = self.compute_tuning_metrics(roi_indices, metric, phase_aware)['orientation_vector_magnitude']
        return self._extract_phase(result, phase_idx)

    def get_mean_orientation(self, roi_indices=None, metric=None, phase_aware=None, phase_idx=None):
        """
        Get mean orientation from circular statistics for ROIs.

        The mean orientation is the angle of the mean orientation vector,
        computed using the doubled-angle method for proper circular statistics
        in 180-degree space. Opposite directions are treated as the same orientation.

        This may differ from preferred_orientation (argmax) if responses are
        broadly tuned across multiple orientations.

        This is part of the **circular statistics framework** along with:
        - get_orientation_vector_magnitude(): magnitude of the orientation vector

        Parameters
        ----------
        roi_indices : list, int, or None
            ROI indices to analyze. If None, analyzes all ROIs.
        metric : str or callable
            Metric for computing tuning function.
            If None, uses self.tuning_metric.
        phase_aware : bool or None
            Controls phase-aware analysis.
        phase_idx : int or None
            If specified, extract only this phase from multi-phase results.

        Returns
        -------
        np.ndarray
            Mean orientation in degrees (0-180). Shape: (n_rois,) or (n_phases, n_rois).
            If phase_idx specified: always (n_rois,).

        See Also
        --------
        get_mean_direction : For mean direction in 360 space
        get_preferred_orientation : For argmax-based preferred orientation
        """
        result = self.compute_tuning_metrics(roi_indices, metric, phase_aware)['mean_orientation']
        return self._extract_phase(result, phase_idx)

    # # =========================================================================
    # # Von Mises Curve Fitting Methods
    # # =========================================================================

    # def _get_cached_von_mises_results(self):
    #     return getattr(self, "_OSDS__von_mises_results", None)

    # def _set_cached_von_mises_results(self, results):
    #     self.__von_mises_results = results

    def _get_cached_tuning_metrics_results(self):
        return getattr(self, "_OSDS__tuning_metrics_results", None)

    def _set_cached_tuning_metrics_results(self, results):
        self.__tuning_metrics_results = results

    def clear_tuning_metrics_cache(self):
        """Clear cached tuning metrics results."""
        self.__tuning_metrics_results = None

#     def clear_von_mises_cache(self):
#         """Clear cached von Mises results."""
#         self.__von_mises_results = None
# 
    def clear_cached_metrics(self):
        """Clear all cached metrics (tuning metrics)."""
        self.__tuning_metrics_results = None
#         self.__von_mises_results = None

    def _can_use_tuning_metrics_cache(
        self, roi_indices, metric, phase_aware
    ):
        results = self._get_cached_tuning_metrics_results()
        if results is None:
            return False
        meta = results.get("_meta", {})
        if meta.get("metric") != metric:
            return False
        if meta.get("phase_aware") != phase_aware:
            return False
#         if meta.get("include_vonmises") != include_vonmises:
            return False
#         if meta.get("r_squared_threshold") != r_squared_threshold:
            return False
        cached_roi_indices = meta.get("roi_indices")
        if cached_roi_indices is None:
            return True
        if roi_indices is None:
            return False
        roi_indices = np.atleast_1d(roi_indices)
        return all(idx in cached_roi_indices for idx in roi_indices)

    def _slice_tuning_metrics_results(self, results, roi_indices=None, cached_roi_indices=None):
        if roi_indices is None:
            return {k: v for k, v in results.items() if k != "_meta"}

        roi_indices = np.atleast_1d(roi_indices)
        if cached_roi_indices is None:
            indexer = roi_indices
        else:
            index_map = {roi: i for i, roi in enumerate(cached_roi_indices)}
            indexer = [index_map[i] for i in roi_indices]

        sliced = {}
        for key, value in results.items():
            if key == "_meta":
                continue
            if key == "roi_indices":
                sliced[key] = np.array(roi_indices)
                continue
            if isinstance(value, np.ndarray) and value.ndim >= 1:
                if value.shape[-1] == (
                    len(cached_roi_indices) if cached_roi_indices is not None else self.num_rois
                ):
                    sliced[key] = value[..., indexer]
                else:
                    sliced[key] = value
            else:
                sliced[key] = value

        return sliced

#     def _can_use_von_mises_cache(self, roi_indices, metric, phase_aware, r_squared_threshold):
#         results = self._get_cached_von_mises_results()
#         if results is None:
#             return False
#         meta = results.get("_meta", {})
#         if meta.get("metric") != metric:
#             return False
#         if meta.get("phase_aware") != phase_aware:
#             return False
#         if meta.get("r_squared_threshold") != r_squared_threshold:
#             return False
#         cached_roi_indices = meta.get("roi_indices")
#         if cached_roi_indices is None:
#             return True
#         if roi_indices is None:
#             return False
#         roi_indices = np.atleast_1d(roi_indices)
#         return all(idx in cached_roi_indices for idx in roi_indices)
# 
#     def _slice_von_mises_array(self, arr, roi_indices=None, phase_idx=None, cached_roi_indices=None):
#         if phase_idx is not None:
#             if arr.ndim == 2:
#                 arr = arr[phase_idx]
#             elif arr.ndim == 1 and phase_idx != 0:
#                 raise IndexError(f"phase_idx={phase_idx} invalid for single-phase result")
#         if roi_indices is not None:
#             roi_indices = np.atleast_1d(roi_indices)
#             if cached_roi_indices is None:
#                 if arr.ndim == 2:
#                     arr = arr[:, roi_indices]
#                 else:
#                     arr = arr[roi_indices]
#             else:
#                 index_map = {roi: i for i, roi in enumerate(cached_roi_indices)}
#                 mapped = [index_map[i] for i in roi_indices]
#                 if arr.ndim == 2:
#                     arr = arr[:, mapped]
#                 else:
#                     arr = arr[mapped]
#         return arr
# 
#     def calculate_von_mises_results(
#         self,
#         roi_indices=None,
#         metric="auc",
#         phase_aware=None,
#         r_squared_threshold=0.8,
#         overwrite=False,
#     ):
#         """
#         Compute and cache von Mises results for this OSDS object.
# 
#         Results are stored in a hidden attribute (__von_mises_results) and can be
#         accessed via the get_von_mises_* getters.
#         """
#         if not overwrite and self._can_use_von_mises_cache(
#             roi_indices, metric, phase_aware, r_squared_threshold
#         ):
#             return self._get_cached_von_mises_results()
# 
#         metrics = tuning_metrics.compute_all_tuning_metrics(
#             self,
#             roi_indices=roi_indices,
#             metric=metric,
#             phase_aware=phase_aware,
#             include_vonmises=True,
#             r_squared_threshold=r_squared_threshold,
#         )
# 
#         cached_roi_indices = None
#         if roi_indices is not None:
#             cached_roi_indices = list(np.atleast_1d(roi_indices))
# 
#         results = {
#             "vm_preferred_direction": metrics["vm_preferred_direction"],
#             "vm_preferred_orientation": metrics["vm_preferred_orientation"],
#             "vm_dir_r_squared": metrics["vm_dir_r_squared"],
#             "vm_ori_r_squared": metrics["vm_ori_r_squared"],
#             "vm_dir_kappa": metrics["vm_dir_kappa"],
#             "vm_ori_kappa": metrics["vm_ori_kappa"],
#             "vm_dir_fit_valid": metrics["vm_dir_fit_valid"],
#             "vm_ori_fit_valid": metrics["vm_ori_fit_valid"],
#             "_meta": {
#                 "roi_indices": cached_roi_indices,
#                 "metric": metric,
#                 "phase_aware": phase_aware,
#                 "r_squared_threshold": r_squared_threshold,
#             },
#         }
# 
#         self._set_cached_von_mises_results(results)
#         return results
# 
#     def get_von_mises_preferred_direction(
#         self, roi_indices=None, metric="auc", phase_aware=None, phase_idx=None, r_squared_threshold=0.8
#     ):
#         results = self.calculate_von_mises_results(
#             roi_indices=roi_indices,
#             metric=metric,
#             phase_aware=phase_aware,
#             r_squared_threshold=r_squared_threshold,
#         )
#         meta = results.get("_meta", {})
#         return self._slice_von_mises_array(
#             results["vm_preferred_direction"],
#             roi_indices=roi_indices,
#             phase_idx=phase_idx,
#             cached_roi_indices=meta.get("roi_indices"),
#         )
# 
#     def get_von_mises_preferred_orientation(
#         self, roi_indices=None, metric="auc", phase_aware=None, phase_idx=None, r_squared_threshold=0.8
#     ):
#         results = self.calculate_von_mises_results(
#             roi_indices=roi_indices,
#             metric=metric,
#             phase_aware=phase_aware,
#             r_squared_threshold=r_squared_threshold,
#         )
#         meta = results.get("_meta", {})
#         return self._slice_von_mises_array(
#             results["vm_preferred_orientation"],
#             roi_indices=roi_indices,
#             phase_idx=phase_idx,
#             cached_roi_indices=meta.get("roi_indices"),
#         )
# 
#     def get_von_mises_direction_r_squared(
#         self, roi_indices=None, metric="auc", phase_aware=None, phase_idx=None, r_squared_threshold=0.8
#     ):
#         results = self.calculate_von_mises_results(
#             roi_indices=roi_indices,
#             metric=metric,
#             phase_aware=phase_aware,
#             r_squared_threshold=r_squared_threshold,
#         )
#         meta = results.get("_meta", {})
#         return self._slice_von_mises_array(
#             results["vm_dir_r_squared"],
#             roi_indices=roi_indices,
#             phase_idx=phase_idx,
#             cached_roi_indices=meta.get("roi_indices"),
#         )
# 
#     def get_von_mises_orientation_r_squared(
#         self, roi_indices=None, metric="auc", phase_aware=None, phase_idx=None, r_squared_threshold=0.8
#     ):
#         results = self.calculate_von_mises_results(
#             roi_indices=roi_indices,
#             metric=metric,
#             phase_aware=phase_aware,
#             r_squared_threshold=r_squared_threshold,
#         )
#         meta = results.get("_meta", {})
#         return self._slice_von_mises_array(
#             results["vm_ori_r_squared"],
#             roi_indices=roi_indices,
#             phase_idx=phase_idx,
#             cached_roi_indices=meta.get("roi_indices"),
#         )
# 
#     def get_von_mises_direction_kappa(
#         self, roi_indices=None, metric="auc", phase_aware=None, phase_idx=None, r_squared_threshold=0.8
#     ):
#         results = self.calculate_von_mises_results(
#             roi_indices=roi_indices,
#             metric=metric,
#             phase_aware=phase_aware,
#             r_squared_threshold=r_squared_threshold,
#         )
#         meta = results.get("_meta", {})
#         return self._slice_von_mises_array(
#             results["vm_dir_kappa"],
#             roi_indices=roi_indices,
#             phase_idx=phase_idx,
#             cached_roi_indices=meta.get("roi_indices"),
#         )
# 
#     def get_von_mises_orientation_kappa(
#         self, roi_indices=None, metric="auc", phase_aware=None, phase_idx=None, r_squared_threshold=0.8
#     ):
#         results = self.calculate_von_mises_results(
#             roi_indices=roi_indices,
#             metric=metric,
#             phase_aware=phase_aware,
#             r_squared_threshold=r_squared_threshold,
#         )
#         meta = results.get("_meta", {})
#         return self._slice_von_mises_array(
#             results["vm_ori_kappa"],
#             roi_indices=roi_indices,
#             phase_idx=phase_idx,
#             cached_roi_indices=meta.get("roi_indices"),
#         )
# 
#     def get_von_mises_direction_fit_valid(
#         self, roi_indices=None, metric="auc", phase_aware=None, phase_idx=None, r_squared_threshold=0.8
#     ):
#         results = self.calculate_von_mises_results(
#             roi_indices=roi_indices,
#             metric=metric,
#             phase_aware=phase_aware,
#             r_squared_threshold=r_squared_threshold,
#         )
#         meta = results.get("_meta", {})
#         return self._slice_von_mises_array(
#             results["vm_dir_fit_valid"],
#             roi_indices=roi_indices,
#             phase_idx=phase_idx,
#             cached_roi_indices=meta.get("roi_indices"),
#         )
# 
#     def get_von_mises_orientation_fit_valid(
#         self, roi_indices=None, metric="auc", phase_aware=None, phase_idx=None, r_squared_threshold=0.8
#     ):
#         results = self.calculate_von_mises_results(
#             roi_indices=roi_indices,
#             metric=metric,
#             phase_aware=phase_aware,
#             r_squared_threshold=r_squared_threshold,
#         )
#         meta = results.get("_meta", {})
#         return self._slice_von_mises_array(
#             results["vm_ori_fit_valid"],
#             roi_indices=roi_indices,
#             phase_idx=phase_idx,
#             cached_roi_indices=meta.get("roi_indices"),
#         )
# 
#     def get_preferred_direction_vonmises(self, roi_indices=None, metric="auc",
#                                           phase_aware=None, phase_idx=None,
#                                           r_squared_threshold=0.8):
#         """
#         Backward-compatible dict interface for von Mises direction results.
#         Prefer get_von_mises_preferred_direction() and related getters.
#         """
#         results = self.calculate_von_mises_results(
#             roi_indices=roi_indices,
#             metric=metric,
#             phase_aware=phase_aware,
#             r_squared_threshold=r_squared_threshold,
#         )
#         meta = results.get("_meta", {})
#         return {
#             "preferred_direction": self._slice_von_mises_array(
#                 results["vm_preferred_direction"],
#                 roi_indices=roi_indices,
#                 phase_idx=phase_idx,
#                 cached_roi_indices=meta.get("roi_indices"),
#             ),
#             "kappa": self._slice_von_mises_array(
#                 results["vm_dir_kappa"],
#                 roi_indices=roi_indices,
#                 phase_idx=phase_idx,
#                 cached_roi_indices=meta.get("roi_indices"),
#             ),
#             "r_squared": self._slice_von_mises_array(
#                 results["vm_dir_r_squared"],
#                 roi_indices=roi_indices,
#                 phase_idx=phase_idx,
#                 cached_roi_indices=meta.get("roi_indices"),
#             ),
#             "fit_valid": self._slice_von_mises_array(
#                 results["vm_dir_fit_valid"],
#                 roi_indices=roi_indices,
#                 phase_idx=phase_idx,
#                 cached_roi_indices=meta.get("roi_indices"),
#             ),
#         }
# 
#     def get_preferred_orientation_vonmises(self, roi_indices=None, metric="auc",
#                                             phase_aware=None, phase_idx=None,
#                                             r_squared_threshold=0.8):
#         """
#         Backward-compatible dict interface for von Mises orientation results.
#         Prefer get_von_mises_preferred_orientation() and related getters.
#         """
#         results = self.calculate_von_mises_results(
#             roi_indices=roi_indices,
#             metric=metric,
#             phase_aware=phase_aware,
#             r_squared_threshold=r_squared_threshold,
#         )
#         meta = results.get("_meta", {})
#         return {
#             "preferred_orientation": self._slice_von_mises_array(
#                 results["vm_preferred_orientation"],
#                 roi_indices=roi_indices,
#                 phase_idx=phase_idx,
#                 cached_roi_indices=meta.get("roi_indices"),
#             ),
#             "kappa": self._slice_von_mises_array(
#                 results["vm_ori_kappa"],
#                 roi_indices=roi_indices,
#                 phase_idx=phase_idx,
#                 cached_roi_indices=meta.get("roi_indices"),
#             ),
#             "r_squared": self._slice_von_mises_array(
#                 results["vm_ori_r_squared"],
#                 roi_indices=roi_indices,
#                 phase_idx=phase_idx,
#                 cached_roi_indices=meta.get("roi_indices"),
#             ),
#             "fit_valid": self._slice_von_mises_array(
#                 results["vm_ori_fit_valid"],
#                 roi_indices=roi_indices,
#                 phase_idx=phase_idx,
#                 cached_roi_indices=meta.get("roi_indices"),
#             ),
#         }
# 
    # =========================================================================
    # Vector extraction methods (for visualization)
    # =========================================================================
# 
#     def _extract_vector_helper(self, roi_index, extract_func, metric='peak', use_phases=None):
#         """
#         Private helper for vector extraction methods.
# 
#         Parameters
#         ----------
#         roi_index : int
#             ROI index to analyze.
#         extract_func : callable
#             Function from tuning_metrics to apply (e.g., extract_direction_vectors).
#         metric : str or callable
#             Metric for computing tuning function.
#         use_phases : bool or None
#             Phase handling (None=auto-detect).
# 
#         Returns
#         -------
#         dict
#             Single phase: result from extract_func.
#             Multi-phase: dict with 'phase_0', 'phase_1', etc. keys.
#         """
#         if use_phases is None:
#             use_phases = self.dir_phase_num > 1
# 
#         if use_phases:
#             responses = self.compute_tuning_function(roi_index=roi_index, metric=metric)
            # responses shape: (n_directions, n_phases)
#             return {
#                 f'phase_{i}': extract_func(responses[:, i], self.directions_list)
#                 for i in range(responses.shape[1])
#             }
#         else:
#             responses = self.compute_tuning_function(
#                 roi_index=roi_index, metric=metric, phase_num=None
#             )
#             return extract_func(responses, self.directions_list)
# 
#     def extract_direction_vectors(self, roi_index, metric='peak', use_phases=None):
#         """
#         Extract individual direction vectors for a specific ROI.
# 
#         Parameters
#         ----------
#         roi_index : int
#             ROI index to analyze.
#         metric : str or callable
#             Metric for computing tuning function.
#         use_phases : bool or None
#             Phase handling (None=auto-detect from dir_phase_num).
# 
#         Returns
#         -------
#         dict
#             Contains 'angles', 'magnitudes', 'cartesian_x', 'cartesian_y'.
#             Multi-phase: nested dict with 'phase_0', 'phase_1', etc. keys.
#         """
#         return self._extract_vector_helper(
#             roi_index, tuning_metrics.extract_direction_vectors, metric, use_phases
#         )
# 
#     def extract_mean_vector(self, roi_index, metric='peak', use_phases=None):
#         """
#         Extract mean vector for a specific ROI.
# 
#         Parameters
#         ----------
#         roi_index : int
#             ROI index to analyze.
#         metric : str or callable
#             Metric for computing tuning function.
#         use_phases : bool or None
#             Phase handling (None=auto-detect from dir_phase_num).
# 
#         Returns
#         -------
#         dict
#             Contains 'angle', 'magnitude', 'cartesian_x', 'cartesian_y'.
#             Multi-phase: nested dict with 'phase_0', 'phase_1', etc. keys.
#         """
#         return self._extract_vector_helper(
#             roi_index, tuning_metrics.extract_mean_vector, metric, use_phases
#         )
# 
#     def extract_orientation_vector(self, roi_index, metric='peak', use_phases=None):
#         """
#         Extract orientation vector for a specific ROI.
# 
#         Parameters
#         ----------
#         roi_index : int
#             ROI index to analyze.
#         metric : str or callable
#             Metric for computing tuning function.
#         use_phases : bool or None
#             Phase handling (None=auto-detect from dir_phase_num).
# 
#         Returns
#         -------
#         dict
#             Contains 'angle', 'magnitude', 'cartesian_x', 'cartesian_y'.
#             Multi-phase: nested dict with 'phase_0', 'phase_1', etc. keys.
#         """
#         return self._extract_vector_helper(
#             roi_index, tuning_metrics.extract_orientation_vector, metric, use_phases
#         )
#     
    # =========================================================================
    # Deprecated methods (kept for backward compatibility)
    # =========================================================================

    def _compute_single_roi_metric(self, roi_index, compute_func, metric=None, use_phases=None):
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
            If None, uses self.tuning_metric.
        use_phases : bool or None
            Phase handling (None=auto-detect).

        Returns
        -------
        dict
            Single phase: result from compute_func.
            Multi-phase: dict with 'phase_0', 'phase_1', etc. keys.
        """
        if metric is None:
            metric = self.tuning_metric
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

    def compute_osi(self, roi_index=None, metric=None, use_phases=None):
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

    def compute_dsi(self, roi_index=None, metric=None, use_phases=None):
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

    def compute_orientation_selectivity_index(self, roi_index, metric=None, use_phases=None):
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

    def compute_phase_tuning_metrics(self, metric=None, roi_indices=None,
                                      phase_ranges=None):
        """
        DEPRECATED: Use compute_tuning_metrics(phase_aware=True) instead.

        Compute directional tuning metrics with automatic phase analysis.

        Parameters
        ----------
        metric : str or callable
            Metric to use for computing tuning functions.
            If None, uses self.tuning_metric.
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
                                       metric=None, trace_scale=0.25, minimal=True, 
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
            Summary metric ('peak', 'auc', 'mean', etc.).
            If None, uses self.tuning_metric.
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
        if metric is None:
            metric = self.tuning_metric
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
