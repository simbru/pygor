from pygor.classes.core_data import Core
from dataclasses import dataclass, field
import numpy as np

from pygor.timeseries.osds.plotting import circular_directional_plots
from pygor.timeseries.osds import tuning_metrics, tuning_computation

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
    
    # def plot_circular_responses(self, roi_index=-1, figsize=(8, 8)):
    #     """Plot directional responses in circular arrangement"""
    #     arr = np.squeeze(self.split_averages_directionally()[:, [roi_index]])
    #     return circular_directional_plots.plot_directional_responses_circular(arr, self.directions_list, figsize)
    
    # def plot_circular_responses_with_polar(self, roi_index=-1, metric='peak', figsize=(10, 10), 
    #                                     show_trials=True, polar_size=0.3, data_crop=None,
    #                                     use_phases=None, phase_colors=None):
    #     """
    #     Plot directional responses with central polar summary and optional individual trials.
        
    #     Parameters:
    #     -----------
    #     roi_index : int, optional
    #         ROI index to plot (default -1 for last ROI)
    #     metric : str, optional
    #         Summary metric for polar plot ('peak', 'auc', 'mean', etc.)
    #     figsize : tuple, optional
    #         Figure size (width, height)
    #     show_trials : bool, optional
    #         Whether to show individual trial traces as faint lines (default False)
    #     polar_size : float, optional
    #         Size of the central polar plot as fraction of figure (default 0.3)
    #     use_phases : bool or None
    #         If None, uses self.dir_phase_num > 1 to decide
    #         If True, forces phase analysis with overlay in polar plot
    #         If False, forces single-phase analysis
    #     phase_colors : list or None
    #         Colors for each phase. If None, uses default colors
        
    #     Returns:
    #     --------
    #     fig, ax_polar : matplotlib objects
    #         Figure and polar axes objects
    #     """
    #     # Automatically use phases if dir_phase_num > 1 and use_phases not specified
    #     if use_phases is None:
    #         use_phases = self.dir_phase_num > 1
        
    #     # Set default phase colors
    #     if phase_colors is None:
    #         phase_colors = ['#2E8B57', '#B8860B', '#8B4513', '#483D8B']  # Default colors for phases
        
    #     return circular_directional_plots.plot_directional_responses_circular_with_polar(
    #         moving_bars_obj=self,
    #         roi_index=roi_index,
    #         metric=metric,
    #         figsize=figsize,
    #         show_trials=show_trials,
    #         polar_size=polar_size,
    #         data_crop=data_crop,
    #         use_phases=use_phases,
    #         phase_colors=phase_colors
    #     )
    
    # def plot_dual_phase_responses(self, roi_index=-1, phase_split=3200, metric='peak', 
    #                         figsize=(12, 10), show_trials=True, polar_kwargs=None,
    #                         phase_colors=("#2E8B57", "#B8860B")):
    #     """
    #     Plot directional responses for two stimulus phases (OFF->ON and ON->OFF) 
    #     with overlapping polar plots and separate trace arrangements.

    #     Parameters:
    #     -----------
    #     phase_split : int, optional
    #         Frame number where phase 1 ends and phase 2 begins (default 3200)
    #     roi_index : int, optional
    #         ROI index to plot (default -1 for last ROI)
    #     metric : str, optional
    #         Summary metric for polar plots ('peak', 'auc', 'mean', etc.)
    #     figsize : tuple, optional
    #         Figure size (width, height)
    #     show_trials : bool, optional
    #         Whether to show individual trial traces (default True)
    #     polar_kwargs : dict, optional
    #         Additional keyword arguments for polar plot styling
    #     phase_colors : tuple, optional
    #         Colors for (phase1, phase2) plots

    #     Returns:
    #     --------
    #     fig, ax_polar : matplotlib objects
    #         Figure and polar axes objects
    #     """
    #     return circular_directional_plots.plot_directional_responses_dual_phase(
    #         moving_bars_obj=self,
    #         roi_index=roi_index,
    #         metric=metric,
    #         figsize=figsize,
    #         show_trials=show_trials,
    #         polar_kwargs=polar_kwargs,
    #         phase_colors=phase_colors
    #     )

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
        # Automatically use dir_phase_num if phase_num not specified
        if phase_num is None:
            phase_num = self.dir_phase_num if self.dir_phase_num > 1 else None
            
        return tuning_computation.compute_tuning_function(
            self, roi_index=roi_index, window=window, metric=metric, phase_num=phase_num
        )

    # def plot_tuning_function(self, rois=None, figsize=(6, 6), colors=None, ax=None, show_title=True, 
    #                        show_theta_labels=True, show_tuning=True, show_mean_vector=False, 
    #                        mean_vector_color='red', show_orientation_vector=False, 
    #                        orientation_vector_color='orange', use_phases=None, 
    #                        phase_colors=None, overlay_phases=True, legend=True, minimal=False, **kwargs):
    #     """
    #     Plot tuning functions as polar plots.
        
    #     Parameters:
    #     -----------
    #     rois : list of int or None
    #         ROI indices to plot. If None, plots all ROIs
    #     figsize : tuple
    #         Figure size (width, height)
    #     colors : list or None
    #         Colors for each ROI. If None, uses default color cycle
    #     ax : matplotlib.axes.Axes or None
    #         Existing polar axes to plot on. If None, creates new figure and axes
    #     show_title : bool
    #         Whether to show the title on the plot (default True)
    #     show_theta_labels : bool
    #         Whether to show the theta (direction) labels on the plot (default True)
    #     show_tuning : bool
    #         Whether to show the tuning curve itself (default True). When False, only shows vectors.
    #     show_mean_vector : bool
    #         Whether to show mean direction vectors as overlays (default False)
    #     mean_vector_color : str
    #         Color for mean direction vector arrows (default 'red')
    #     show_orientation_vector : bool
    #         Whether to show mean orientation vectors as overlays (default False)
    #     orientation_vector_color : str
    #         Color for mean orientation vector arrows (default 'orange')
    #     use_phases : bool or None
    #         If None, uses self.dir_phase_num > 1 to decide
    #         If True, forces phase analysis
    #         If False, forces single-phase analysis
    #     phase_colors : list or None
    #         Colors for each phase. If None, uses default colors
    #     overlay_phases : bool
    #         Whether to overlay phases on same plot (True) or create separate plots (False)
    #     legend : bool
    #         Whether to show the legend (default True)
    #     minimal : bool
    #         Whether to use minimal plotting (no titles, legends, or labels except axis ticks) (default False)
    #     **kwargs
    #         Additional arguments passed to compute_tuning_function.
        
    #     Returns:
    #     --------
    #     fig : matplotlib.figure.Figure
    #         The figure object
    #     ax : matplotlib.axes.Axes
    #         The polar plot axes object
    #     """
    #     # Automatically use phases if dir_phase_num > 1 and use_phases not specified
    #     if use_phases is None:
    #         use_phases = self.dir_phase_num > 1
        
    #     # Override phase_num in kwargs if use_phases is determined
    #     if use_phases:
    #         kwargs['phase_num'] = self.dir_phase_num
        
    #     # Handle minimal mode by overriding display options
    #     if minimal:
    #         show_title = False
    #         legend = False
        
    #     # Extract parameters from kwargs before passing to compute_tuning_function
    #     # (compute_tuning_function doesn't accept these parameters)
    #     tuning_kwargs = {k: v for k, v in kwargs.items() if k not in ['legend', 'minimal']}
        
    #     # Get tuning functions for all ROIs
    #     tuning_functions = self.compute_tuning_function(**tuning_kwargs)
        
    #     # Handle phase data
    #     if use_phases and len(tuning_functions.shape) == 3:  # (n_rois, n_directions, n_phases)
    #         # Set default phase colors
    #         if phase_colors is None:
    #             phase_colors = ['#2E8B57', '#B8860B', '#8B4513', '#483D8B']  # Default colors for phases
            
    #         if overlay_phases:
    #             # Overlay phases on same plot
    #             return circular_directional_plots.plot_tuning_function_polar_overlay(
    #                 tuning_functions,
    #                 self.directions_list,
    #                 rois=rois,
    #                 figsize=figsize,
    #                 colors=colors,
    #                 phase_colors=phase_colors,
    #                 ax=ax,
    #                 show_title=show_title,
    #                 show_theta_labels=show_theta_labels,
    #                 show_tuning=show_tuning,
    #                 show_mean_vector=show_mean_vector,
    #                 mean_vector_color=mean_vector_color,
    #                 show_orientation_vector=show_orientation_vector,
    #                 orientation_vector_color=orientation_vector_color,
    #                 metric=kwargs.get('metric', 'peak'),
    #                 legend=legend,
    #                 minimal=minimal
    #             )
    #         else:
    #             # Create separate plots for each phase
    #             return circular_directional_plots.plot_tuning_function_multi_phase(
    #                 tuning_functions=tuning_functions,
    #                 directions_list=self.directions_list,
    #                 phase_num=kwargs['phase_num'],
    #                 rois=rois,
    #                 figsize=figsize,
    #                 colors=colors,
    #                 ax=ax,
    #                 show_title=show_title,
    #                 show_theta_labels=show_theta_labels,
    #                 show_tuning=show_tuning,
    #                 show_mean_vector=show_mean_vector,
    #                 mean_vector_color=mean_vector_color,
    #                 show_orientation_vector=show_orientation_vector,
    #                 orientation_vector_color=orientation_vector_color,
    #                 metric=kwargs.get('metric', 'peak'),
    #                 legend=legend,
    #                 minimal=minimal
    #             )
    #     else:
    #         # Regular single-phase plotting
    #         return circular_directional_plots.plot_tuning_function_polar(
    #             tuning_functions.T,  # Transpose to (n_directions, n_rois)
    #             self.directions_list,
    #             rois=rois,
    #             figsize=figsize,
    #             colors=colors,
    #             metric=kwargs.get('metric', 'peak'),
    #             ax=ax,
    #             show_title=show_title,
    #             show_theta_labels=show_theta_labels,
    #             show_tuning=show_tuning,
    #             show_mean_vector=show_mean_vector,
    #             mean_vector_color=mean_vector_color,
    #             show_orientation_vector=show_orientation_vector,
    #             orientation_vector_color=orientation_vector_color,
    #             legend=legend,
    #             minimal=minimal
    #         )

    def compute_tuning_metrics(self, roi_indices=None, metric='peak'):
        """
        Compute directional tuning metrics for ROIs.
        
        Computes vector magnitude (r), circular variance (CV), directional 
        selectivity index (DSI), preferred direction, and mean direction 
        for each ROI using circular statistics.
        
        Parameters:
        -----------
        roi_indices : list or None
            ROI indices to analyze. If None, analyzes all ROIs.
        metric : str or callable
            Metric to use for computing tuning functions (default 'peak')
            
        Returns:
        --------
        dict : Dictionary containing arrays of metrics for each ROI:
            - 'vector_magnitude': How directionally tuned (0-1)
            - 'circular_variance': Spread of directional response (0-1)  
            - 'dsi': Preference for one direction vs opposite (-1 to 1)
            - 'preferred_direction': Direction with maximum response (degrees)
            - 'mean_direction': Mean direction from circular stats (degrees)
            - 'roi_indices': ROI indices that were analyzed
        """
        if isinstance(roi_indices, int):
            roi_indices = [roi_indices]
        return tuning_metrics.compute_all_tuning_metrics(
            self, roi_indices=roi_indices, metric=metric
        )
    
    # def plot_tuning_metrics_histograms(self, roi_indices=None, metric='peak', 
    #                                  figsize=(15, 10), bins=20):
    #     """
    #     Plot histograms of directional tuning metrics.
        
    #     Parameters:
    #     -----------
    #     roi_indices : list or None
    #         ROI indices to analyze. If None, analyzes all ROIs.
    #     metric : str or callable
    #         Metric to use for computing tuning functions (default 'peak')
    #     figsize : tuple
    #         Figure size
    #     bins : int
    #         Number of histogram bins
            
    #     Returns:
    #     --------
    #     fig : matplotlib.figure.Figure
    #         Figure object
    #     metrics_dict : dict
    #         Dictionary of computed metrics
    #     """
    #     metrics_dict = self.compute_tuning_metrics(roi_indices=roi_indices, metric=metric)
    #     fig = tuning_metrics.plot_tuning_metrics_histograms(
    #         metrics_dict, figsize=figsize, bins=bins
    #     )
    #     return fig, metrics_dict
    
    def extract_direction_vectors(self, roi_index, metric='peak', use_phases=None):
        """
        Extract individual direction vectors for a specific ROI.
        
        Parameters:
        -----------
        roi_index : int
            ROI index to analyze
        metric : str or callable
            Metric to use for computing tuning function
        use_phases : bool or None
            If None, uses self.dir_phase_num > 1 to decide
            If True, forces phase analysis
            If False, forces single-phase analysis
            
        Returns:
        --------
        dict : Dictionary containing individual direction vectors
            If single phase: standard direction vectors
            If multi-phase: vectors for each phase with keys 'phase_0', 'phase_1', etc.
        """
        # Automatically use phases if dir_phase_num > 1 and use_phases not specified
        if use_phases is None:
            use_phases = self.dir_phase_num > 1
        
        if use_phases:
            # Get phase-aware tuning function
            responses = self.compute_tuning_function(roi_index=roi_index, metric=metric)
            # responses shape: (n_directions, n_phases)
            phase_vectors = {}
            for phase_idx in range(responses.shape[1]):
                phase_vectors[f'phase_{phase_idx}'] = tuning_metrics.extract_direction_vectors(
                    responses[:, phase_idx], self.directions_list
                )
            return phase_vectors
        else:
            # Single phase analysis
            responses = self.compute_tuning_function(roi_index=roi_index, metric=metric, phase_num=None)
            return tuning_metrics.extract_direction_vectors(responses, self.directions_list)
    
    def extract_mean_vector(self, roi_index, metric='peak', use_phases=None):
        """
        Extract mean vector for a specific ROI.
        
        Parameters:
        -----------
        roi_index : int
            ROI index to analyze
        metric : str or callable
            Metric to use for computing tuning function
        use_phases : bool or None
            If None, uses self.dir_phase_num > 1 to decide
            If True, forces phase analysis
            If False, forces single-phase analysis
            
        Returns:
        --------
        dict : Dictionary containing mean vector information
            If single phase: standard mean vector
            If multi-phase: mean vectors for each phase with keys 'phase_0', 'phase_1', etc.
        """
        # Automatically use phases if dir_phase_num > 1 and use_phases not specified
        if use_phases is None:
            use_phases = self.dir_phase_num > 1
        
        if use_phases:
            # Get phase-aware tuning function
            responses = self.compute_tuning_function(roi_index=roi_index, metric=metric)
            # responses shape: (n_directions, n_phases)
            phase_vectors = {}
            for phase_idx in range(responses.shape[1]):
                phase_vectors[f'phase_{phase_idx}'] = tuning_metrics.extract_mean_vector(
                    responses[:, phase_idx], self.directions_list
                )
            return phase_vectors
        else:
            # Single phase analysis
            responses = self.compute_tuning_function(roi_index=roi_index, metric=metric, phase_num=None)
            return tuning_metrics.extract_mean_vector(responses, self.directions_list)
    
    def extract_orientation_vector(self, roi_index, metric='peak', use_phases=None):
        """
        Extract orientation vector for a specific ROI.
        
        Parameters:
        -----------
        roi_index : int
            ROI index to analyze
        metric : str or callable
            Metric to use for computing tuning function
        use_phases : bool or None
            If None, uses self.dir_phase_num > 1 to decide
            If True, forces phase analysis
            If False, forces single-phase analysis
            
        Returns:
        --------
        dict : Dictionary containing orientation vector information
            If single phase: standard orientation vector
            If multi-phase: orientation vectors for each phase with keys 'phase_0', 'phase_1', etc.
        """
        # Automatically use phases if dir_phase_num > 1 and use_phases not specified
        if use_phases is None:
            use_phases = self.dir_phase_num > 1
        
        if use_phases:
            # Get phase-aware tuning function
            responses = self.compute_tuning_function(roi_index=roi_index, metric=metric)
            # responses shape: (n_directions, n_phases)
            phase_vectors = {}
            for phase_idx in range(responses.shape[1]):
                phase_vectors[f'phase_{phase_idx}'] = tuning_metrics.extract_orientation_vector(
                    responses[:, phase_idx], self.directions_list
                )
            return phase_vectors
        else:
            # Single phase analysis
            responses = self.compute_tuning_function(roi_index=roi_index, metric=metric, phase_num=None)
            return tuning_metrics.extract_orientation_vector(responses, self.directions_list)
    
    def compute_orientation_selectivity_index(self, roi_index, metric='peak', use_phases=None):
        """
        Compute orientation selectivity index (OSI) for a specific ROI.
        
        Parameters:
        -----------
        roi_index : int
            ROI index to analyze
        metric : str or callable
            Metric to use for computing tuning function
        use_phases : bool or None
            If None, uses self.dir_phase_num > 1 to decide
            If True, forces phase analysis
            If False, forces single-phase analysis
            
        Returns:
        --------
        dict : Dictionary containing OSI calculation results
            If single phase: standard OSI results
            If multi-phase: OSI results for each phase with keys 'phase_0', 'phase_1', etc.
        """
        # Automatically use phases if dir_phase_num > 1 and use_phases not specified
        if use_phases is None:
            use_phases = self.dir_phase_num > 1
        
        if use_phases:
            # Get phase-aware tuning function
            responses = self.compute_tuning_function(roi_index=roi_index, metric=metric)
            # responses shape: (n_directions, n_phases)
            phase_osi = {}
            for phase_idx in range(responses.shape[1]):
                phase_osi[f'phase_{phase_idx}'] = tuning_metrics.compute_orientation_selectivity_index(
                    responses[:, phase_idx], self.directions_list
                )
            return phase_osi
        else:
            # Single phase analysis
            responses = self.compute_tuning_function(roi_index=roi_index, metric=metric, phase_num=None)
            return tuning_metrics.compute_orientation_selectivity_index(responses, self.directions_list)
    
    def plot_orientation_tuning_cartesian(self, roi_index, metric='peak', use_phases=None, 
                                         phase_colors=None, **kwargs):
        """
        Plot orientation tuning curve in cartesian coordinates for a specific ROI.
        
        Parameters:
        -----------
        roi_index : int
            ROI index to analyze
        metric : str or callable
            Metric to use for computing tuning function
        use_phases : bool or None
            If None, uses self.dir_phase_num > 1 to decide
            If True, forces phase analysis with overlay
            If False, forces single-phase analysis
        phase_colors : list or None
            Colors for each phase. If None, uses default colors
        **kwargs : additional arguments
            Passed to plot_orientation_tuning_cartesian function
            
        Returns:
        --------
        fig : matplotlib.figure.Figure
            Figure object
        ax : matplotlib.axes.Axes
            Axes object
        osi_info : dict
            Dictionary containing OSI calculation results
        """
        # Automatically use phases if dir_phase_num > 1 and use_phases not specified
        if use_phases is None:
            use_phases = self.dir_phase_num > 1
        
        if use_phases:
            # Get phase-aware tuning function
            responses = self.compute_tuning_function(roi_index=roi_index, metric=metric)
            # responses shape: (n_directions, n_phases)
            
            # Set default phase colors
            if phase_colors is None:
                phase_colors = ['#2E8B57', '#B8860B', '#8B4513', '#483D8B']  # Default colors for phases
            
            return circular_directional_plots.plot_orientation_tuning_cartesian_phases(
                responses, self.directions_list, phase_colors=phase_colors, **kwargs
            )
        else:
            # Single phase analysis
            responses = self.compute_tuning_function(roi_index=roi_index, metric=metric, phase_num=None)
            return circular_directional_plots.plot_orientation_tuning_cartesian(
                responses, self.directions_list, **kwargs
            )
    
    # def plot_orientation_tuning_comparison(self, roi_index, metric='peak', use_phases=None,
    #                                      phase_colors=None, **kwargs):
    #     """
    #     Plot side-by-side comparison of polar and cartesian orientation tuning.
        
    #     Parameters:
    #     -----------
    #     roi_index : int
    #         ROI index to analyze
    #     metric : str or callable
    #         Metric to use for computing tuning function
    #     use_phases : bool or None
    #         If None, uses self.dir_phase_num > 1 to decide
    #         If True, forces phase analysis with overlay
    #         If False, forces single-phase analysis
    #     phase_colors : list or None
    #         Colors for each phase. If None, uses default colors
    #     **kwargs : additional arguments
    #         Passed to plot_orientation_tuning_comparison function
            
    #     Returns:
    #     --------
    #     fig : matplotlib.figure.Figure
    #         Figure object
    #     axes : list
    #         List containing [polar_ax, cartesian_ax]
    #     osi_info : dict
    #         Dictionary containing OSI calculation results
    #     """
    #     # Automatically use phases if dir_phase_num > 1 and use_phases not specified
    #     if use_phases is None:
    #         use_phases = self.dir_phase_num > 1
        
    #     if use_phases:
    #         # Get phase-aware tuning function
    #         responses = self.compute_tuning_function(roi_index=roi_index, metric=metric)
    #         # responses shape: (n_directions, n_phases)
            
    #         # Set default phase colors
    #         if phase_colors is None:
    #             phase_colors = ['#2E8B57', '#B8860B', '#8B4513', '#483D8B']  # Default colors for phases
            
    #         return circular_directional_plots.plot_orientation_tuning_comparison_phases(
    #             responses, self.directions_list, phase_colors=phase_colors, **kwargs
    #         )
    #     else:
    #         # Single phase analysis
    #         responses = self.compute_tuning_function(roi_index=roi_index, metric=metric, phase_num=None)
    #         return circular_directional_plots.plot_orientation_tuning_comparison(
    #             responses, self.directions_list, **kwargs
    #         )
    
    def compute_phase_tuning_metrics(self, metric='peak', roi_indices=None, 
                                    phase_ranges=None):
        """
        Compute directional tuning metrics with automatic phase analysis.
        
        High-performance vectorized computation of all tuning metrics (DSI, OSI, 
        preferred direction, etc.) with automatic phase-based analysis based on 
        the object's dir_phase_num setting.
        
        Parameters:
        -----------
        metric : str or callable
            Metric to use for computing tuning functions (default 'peak')
        roi_indices : list or None
            ROI indices to analyze. If None, analyzes all ROIs.
        phase_ranges : list of tuples or None
            Custom phase ranges to override automatic phase detection.
            If None, uses automatic phase splitting based on dir_phase_num.
            Example: [(0, 60), (60, 120)] for custom ranges
            
        Returns:
        --------
        dict : Dictionary containing tuning metrics
            When dir_phase_num=1: (n_rois,) arrays 
            When dir_phase_num>1: (n_phases, n_rois) arrays
            
            Keys include:
            - 'vector_magnitude': Directional tuning strength (0-1)
            - 'dsi': Directional selectivity index (-1 to 1)
            - 'osi': Orientation selectivity index (0-1)
            - 'preferred_direction': Preferred direction in degrees
            - 'mean_direction': Mean direction from circular statistics
            - 'roi_indices': ROI indices analyzed
            - 'phase_ranges': Phase ranges used (if applicable)
            
        Examples:
        ---------
        # Automatic phase analysis based on dir_phase_num
        metrics = obj.compute_phase_tuning_metrics()
        
        # For dir_phase_num=2 (ON/OFF phases)
        on_dsi = metrics['dsi'][0, :]   # ON phase DSI
        off_dsi = metrics['dsi'][1, :]  # OFF phase DSI
        
        # Custom phase ranges (override automatic)
        metrics = obj.compute_phase_tuning_metrics(phase_ranges=[(0, 60), (60, 120)])
        """
        if isinstance(roi_indices, int):
            roi_indices = [roi_indices]
        
        # Automatically determine phase ranges based on dir_phase_num
        if phase_ranges is None:
            if self.dir_phase_num == 1:
                # Single phase - use entire response period
                phase_ranges = None
            else:
                # Multi-phase - automatically split using get_epoch_dur()
                phase_ranges = "auto"
        
        return tuning_metrics.compute_all_tuning_metrics(
            self, metric=metric, roi_indices=roi_indices, phase_ranges=phase_ranges
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
