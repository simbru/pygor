from pygor.classes.core_data import Core
from dataclasses import dataclass, field
import numpy as np

from pygor.timeseries.moving_bars.plotting import circular_directional_plots
from pygor.timeseries.moving_bars import tuning_metrics, tuning_computation

@dataclass(kw_only=False, repr=False)
class MovingBars(Core):
    dir_num: int = field(default=None, metadata={"required": True})
    colour_num: int = field(default=1)
    directions_list: list = field(default=None)

    def __post_init__(self):
        if self.dir_num is None:
            raise ValueError("dir_num must be specified for MovingBars data")
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
        """
        # Remove first column (time axis) and check split compatibility
        data_to_split = self.snippets[:, :, 1:]
        adjusted_data, actual_splits, remainder = self._check_split_compatibility(
            data_to_split, self.dir_num, axis=-1
        )
        
        if remainder > 0:
            print(f"DirectionalSnippets: Lost {remainder} time points to ensure even splitting")
            
        return np.array(np.split(adjusted_data, actual_splits, axis=-1))
   
    def split_averages_directionally(self) -> np.ndarray:
        """
        Returns averages split by direction.
        Handles uneven divisions by trimming excess elements and column indexing issues.
        """
        # Try both with and without first column removal to handle inconsistent data formats
        try:
            # First try with all columns (newer format)
            adjusted_data, actual_splits, remainder = self._check_split_compatibility(
                self.averages[:, :], self.dir_num, axis=-1
            )
        except (IndexError, ValueError):
            # Fall back to removing first column (older format)
            adjusted_data, actual_splits, remainder = self._check_split_compatibility(
                self.averages[:, 1:], self.dir_num, axis=-1
            )
        
        # if remainder > 0:
        #     print(f"DirectionalAverages: Lost {remainder} time points to ensure even splitting")
            
        return np.array(np.split(adjusted_data, actual_splits, axis=-1))
    
    def plot_circular_responses(self, roi_index=-1, figsize=(8, 8)):
        """Plot directional responses in circular arrangement"""
        arr = np.squeeze(self.split_averages_directionally()[:, [roi_index]])
        return circular_directional_plots.plot_directional_responses_circular(arr, self.directions_list, figsize)
    
    def plot_circular_responses_with_polar(self, roi_index=-1, metric='peak', figsize=(10, 10), 
                                        show_trials=True, polar_size=0.3, data_crop=None):
        """
        Plot directional responses with central polar summary and optional individual trials.
        
        Parameters:
        -----------
        roi_index : int, optional
            ROI index to plot (default -1 for last ROI)
        metric : str, optional
            Summary metric for polar plot ('peak', 'auc', 'mean', etc.)
        figsize : tuple, optional
            Figure size (width, height)
        show_trials : bool, optional
            Whether to show individual trial traces as faint lines (default False)
        polar_size : float, optional
            Size of the central polar plot as fraction of figure (default 0.3)
        
        Returns:
        --------
        fig, ax_polar : matplotlib objects
            Figure and polar axes objects
        """
        return circular_directional_plots.plot_directional_responses_circular_with_polar(
            moving_bars_obj=self,
            roi_index=roi_index,
            metric=metric,
            figsize=figsize,
            show_trials=show_trials,
            polar_size=polar_size,
            data_crop=data_crop
        )
    
    def plot_dual_phase_responses(self, roi_index=-1, phase_split=3200, metric='peak', 
                            figsize=(12, 10), show_trials=True, polar_kwargs=None,
                            phase_colors=("#2E8B57", "#B8860B")):
        """
        Plot directional responses for two stimulus phases (OFF->ON and ON->OFF) 
        with overlapping polar plots and separate trace arrangements.

        Parameters:
        -----------
        phase_split : int, optional
            Frame number where phase 1 ends and phase 2 begins (default 3200)
        roi_index : int, optional
            ROI index to plot (default -1 for last ROI)
        metric : str, optional
            Summary metric for polar plots ('peak', 'auc', 'mean', etc.)
        figsize : tuple, optional
            Figure size (width, height)
        show_trials : bool, optional
            Whether to show individual trial traces (default True)
        polar_kwargs : dict, optional
            Additional keyword arguments for polar plot styling
        phase_colors : tuple, optional
            Colors for (phase1, phase2) plots

        Returns:
        --------
        fig, ax_polar : matplotlib objects
            Figure and polar axes objects
        """
        return circular_directional_plots.plot_directional_responses_dual_phase(
            moving_bars_obj=self,
            roi_index=roi_index,
            metric=metric,
            figsize=figsize,
            show_trials=show_trials,
            polar_kwargs=polar_kwargs,
            phase_colors=phase_colors
        )

    def compute_tuning_function(self, roi_index=None, window=None, metric='max', phase_num=None):
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
        return tuning_computation.compute_tuning_function(
            self, roi_index=roi_index, window=window, metric=metric, phase_num=phase_num
        )

    def plot_tuning_function(self, rois=None, figsize=(6, 6), colors=None, ax=None, show_title=True, show_theta_labels=True, show_tuning=True, show_mean_vector=False, mean_vector_color='red', show_orientation_vector=False, orientation_vector_color='orange', **kwargs):
        """
        Plot tuning functions as polar plots.
        
        Parameters:
        -----------
        rois : list of int or None
            ROI indices to plot. If None, plots all ROIs
        figsize : tuple
            Figure size (width, height)
        colors : list or None
            Colors for each ROI. If None, uses default color cycle
        ax : matplotlib.axes.Axes or None
            Existing polar axes to plot on. If None, creates new figure and axes
        show_title : bool
            Whether to show the title on the plot (default True)
        show_theta_labels : bool
            Whether to show the theta (direction) labels on the plot (default True)
        show_tuning : bool
            Whether to show the tuning curve itself (default True). When False, only shows vectors.
        show_mean_vector : bool
            Whether to show mean direction vectors as overlays (default False)
        mean_vector_color : str
            Color for mean direction vector arrows (default 'red')
        show_orientation_vector : bool
            Whether to show mean orientation vectors as overlays (default False)
        orientation_vector_color : str
            Color for mean orientation vector arrows (default 'orange')
        **kwargs
            Additional arguments passed to compute_tuning_function.
            If phase_num is specified, plots each phase separately with phase-dependent vectors.
        
        Returns:
        --------
        fig : matplotlib.figure.Figure
            The figure object
        ax : matplotlib.axes.Axes
            The polar plot axes object
        """
        # Get tuning functions for all ROIs
        tuning_functions = self.compute_tuning_function(**kwargs)
        
        # Handle phase data by plotting each phase separately
        if 'phase_num' in kwargs and kwargs['phase_num'] is not None:
            return circular_directional_plots.plot_tuning_function_multi_phase(
                tuning_functions=tuning_functions,
                directions_list=self.directions_list,
                phase_num=kwargs['phase_num'],
                rois=rois,
                figsize=figsize,
                colors=colors,
                ax=ax,
                show_title=show_title,
                show_theta_labels=show_theta_labels,
                show_tuning=show_tuning,
                show_mean_vector=show_mean_vector,
                mean_vector_color=mean_vector_color,
                show_orientation_vector=show_orientation_vector,
                orientation_vector_color=orientation_vector_color,
                metric=kwargs.get('metric', 'peak')
            )
        else:
            # Regular single-phase plotting
            return circular_directional_plots.plot_tuning_function_polar(
                tuning_functions.T,  # Transpose to (n_directions, n_rois)
                self.directions_list,
                rois=rois,
                figsize=figsize,
                colors=colors,
                metric=kwargs.get('metric', 'peak'),
                ax=ax,
                show_title=show_title,
                show_theta_labels=show_theta_labels,
                show_tuning=show_tuning,
                show_mean_vector=show_mean_vector,
                mean_vector_color=mean_vector_color,
                show_orientation_vector=show_orientation_vector,
                orientation_vector_color=orientation_vector_color
            )

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
    
    def plot_tuning_metrics_histograms(self, roi_indices=None, metric='peak', 
                                     figsize=(15, 10), bins=20):
        """
        Plot histograms of directional tuning metrics.
        
        Parameters:
        -----------
        roi_indices : list or None
            ROI indices to analyze. If None, analyzes all ROIs.
        metric : str or callable
            Metric to use for computing tuning functions (default 'peak')
        figsize : tuple
            Figure size
        bins : int
            Number of histogram bins
            
        Returns:
        --------
        fig : matplotlib.figure.Figure
            Figure object
        metrics_dict : dict
            Dictionary of computed metrics
        """
        metrics_dict = self.compute_tuning_metrics(roi_indices=roi_indices, metric=metric)
        fig = tuning_metrics.plot_tuning_metrics_histograms(
            metrics_dict, figsize=figsize, bins=bins
        )
        return fig, metrics_dict
    
    def extract_direction_vectors(self, roi_index, metric='peak'):
        """
        Extract individual direction vectors for a specific ROI.
        
        Parameters:
        -----------
        roi_index : int
            ROI index to analyze
        metric : str or callable
            Metric to use for computing tuning function
            
        Returns:
        --------
        dict : Dictionary containing individual direction vectors
        """
        responses = self.compute_tuning_function(roi_index=roi_index, metric=metric)
        return tuning_metrics.extract_direction_vectors(responses, self.directions_list)
    
    def extract_mean_vector(self, roi_index, metric='peak'):
        """
        Extract mean vector for a specific ROI.
        
        Parameters:
        -----------
        roi_index : int
            ROI index to analyze
        metric : str or callable
            Metric to use for computing tuning function
            
        Returns:
        --------
        dict : Dictionary containing mean vector information
        """
        responses = self.compute_tuning_function(roi_index=roi_index, metric=metric)
        return tuning_metrics.extract_mean_vector(responses, self.directions_list)
    
    def extract_orientation_vector(self, roi_index, metric='peak'):
        """
        Extract orientation vector for a specific ROI.
        
        Parameters:
        -----------
        roi_index : int
            ROI index to analyze
        metric : str or callable
            Metric to use for computing tuning function
            
        Returns:
        --------
        dict : Dictionary containing orientation vector information
        """
        responses = self.compute_tuning_function(roi_index=roi_index, metric=metric)
        return tuning_metrics.extract_orientation_vector(responses, self.directions_list)
    
    def compute_orientation_selectivity_index(self, roi_index, metric='peak'):
        """
        Compute orientation selectivity index (OSI) for a specific ROI.
        
        Parameters:
        -----------
        roi_index : int
            ROI index to analyze
        metric : str or callable
            Metric to use for computing tuning function
            
        Returns:
        --------
        dict : Dictionary containing OSI calculation results
        """
        responses = self.compute_tuning_function(roi_index=roi_index, metric=metric)
        return tuning_metrics.compute_orientation_selectivity_index(responses, self.directions_list)
    
    def plot_orientation_tuning_cartesian(self, roi_index, metric='peak', **kwargs):
        """
        Plot orientation tuning curve in cartesian coordinates for a specific ROI.
        
        Parameters:
        -----------
        roi_index : int
            ROI index to analyze
        metric : str or callable
            Metric to use for computing tuning function
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
        responses = self.compute_tuning_function(roi_index=roi_index, metric=metric)
        return circular_directional_plots.plot_orientation_tuning_cartesian(
            responses, self.directions_list, **kwargs
        )
    
    def plot_orientation_tuning_comparison(self, roi_index, metric='peak', **kwargs):
        """
        Plot side-by-side comparison of polar and cartesian orientation tuning.
        
        Parameters:
        -----------
        roi_index : int
            ROI index to analyze
        metric : str or callable
            Metric to use for computing tuning function
        **kwargs : additional arguments
            Passed to plot_orientation_tuning_comparison function
            
        Returns:
        --------
        fig : matplotlib.figure.Figure
            Figure object
        axes : list
            List containing [polar_ax, cartesian_ax]
        osi_info : dict
            Dictionary containing OSI calculation results
        """
        responses = self.compute_tuning_function(roi_index=roi_index, metric=metric)
        return circular_directional_plots.plot_orientation_tuning_comparison(
            responses, self.directions_list, **kwargs
        )
    
