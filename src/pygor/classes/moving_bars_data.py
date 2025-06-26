from pygor.classes.core_data import Core
from dataclasses import dataclass
import numpy as np

from pygor.timeseries.moving_bars import circular_directional_plots

@dataclass(kw_only=True, repr=False)
class MovingBars(Core):
    dir_num: int
    colour_num: int
    
    def __post_init__(self):
        self.directions_list: list = None
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
            
            print(f"Warning: Trimmed {remainder} elements from axis {axis} "
                  f"(original length: {axis_length}, new length: {trim_length}) "
                  f"to allow even splitting into {n_splits} parts.")
            
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
        
        if remainder > 0:
            print(f"DirectionalAverages: Lost {remainder} time points to ensure even splitting")
            
        return np.array(np.split(adjusted_data, actual_splits, axis=-1))
    
    def plot_circular_responses(self, roi_index=-1, figsize=(8, 8)):
        """Plot directional responses in circular arrangement"""
        arr = np.squeeze(self.split_averages_directionally()[:, [roi_index]])
        return circular_directional_plots.plot_directional_responses_circular(arr, self.directions_list, figsize)