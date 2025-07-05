from pygor.classes.core_data import Core
from dataclasses import dataclass, field
import numpy as np

from pygor.timeseries.moving_bars import circular_directional_plots

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

    def compute_tuning_function(self, roi_index=None, window=None, metric='max'):
        """
        Compute tuning function for each ROI across all directions.
        
        Parameters:
        -----------
        window : int or tuple, optional
            Time window within each direction phase to analyze.
            If int, uses that many frames from start of each direction.
            If tuple (start, end), uses that slice within each direction.
            If None, uses entire duration of each direction.
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
            
        Returns:
        --------
        np.ndarray
            If roi_index is None: tuning values with shape (n_rois, n_directions).
            If roi_index is specified: tuning values with shape (n_directions,).
            Values are ordered according to self.directions_list.
        """
        # Get directionally split averages: (n_directions, n_rois, timepoints_per_direction)
        dir_averages = self.split_averages_directionally()
        
        # Apply window if specified
        if window is not None:
            if isinstance(window, int):
                # Use first 'window' frames
                dir_averages = dir_averages[:, :, :window]
            elif isinstance(window, (tuple, list)) and len(window) == 2:
                # Use slice [start:end]
                start, end = window
                dir_averages = dir_averages[:, :, start:end]
            else:
                raise ValueError("window must be int or tuple of (start, end)")
        
        # Compute metric for each direction and ROI
        if metric == 'max':
            tuning_values = np.max(dir_averages, axis=2)
        elif metric in ['absmax', 'peak']:
            tuning_values = np.max(np.abs(dir_averages), axis=2)
        elif metric == 'min':
            tuning_values = np.min(dir_averages, axis=2)
        elif metric in ['avg', 'mean']:
            tuning_values = np.mean(dir_averages, axis=2)
        elif metric == 'range':
            tuning_values = np.max(dir_averages, axis=2) - np.min(dir_averages, axis=2)
        elif metric == 'auc':
            tuning_values = np.trapz(np.abs(dir_averages), axis=2)
        elif metric == 'peak_positive':
            tuning_values = np.max(dir_averages, axis=2)
        elif metric == 'peak_negative':
            tuning_values = np.min(dir_averages, axis=2)
        elif callable(metric):
            # Apply custom function to each direction/ROI combination
            tuning_values = np.array([[metric(dir_averages[d, r, :]) 
                                     for r in range(dir_averages.shape[1])]
                                    for d in range(dir_averages.shape[0])])
        else:
            raise ValueError(f"Unknown metric: {metric}")
        
        # Transpose to get (n_rois, n_directions) as requested
        tuning_values = tuning_values.T
        
        # Return single ROI if specified
        if roi_index is not None:
            return tuning_values[roi_index, :]
        else:
            return tuning_values

    def plot_tuning_function(self, rois=None, figsize=(6, 6), colors=None, ax=None, show_title=True, show_theta_labels=True, **kwargs):
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
        **kwargs
            Additional arguments passed to compute_tuning_function
        
        Returns:
        --------
        fig : matplotlib.figure.Figure
            The figure object
        ax : matplotlib.axes.Axes
            The polar plot axes object
        """
        # Get tuning functions for all ROIs
        tuning_functions = self.compute_tuning_function(**kwargs)
        
        return circular_directional_plots.plot_tuning_function_polar(
            tuning_functions.T,  # Transpose to (n_directions, n_rois)
            self.directions_list,
            rois=rois,
            figsize=figsize,
            colors=colors,
            metric=kwargs.get('metric', 'peak'),
            ax=ax,
            show_title=show_title,
            show_theta_labels=show_theta_labels
        )