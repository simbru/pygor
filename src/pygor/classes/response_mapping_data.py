from dataclasses import dataclass, field
import numpy as np
import pandas as pd
from pygor.classes.core_data import Core
from scipy.signal import savgol_filter


@dataclass
class ResponseMapping(Core):
    """
    A dataclass for mapping responses to brain regions.
    
    Inherits from Core, and adds methods to look at average traces and work based on those. 
    Also features code to pull z-stacks corresponding to recordings if available and find imaging planes in 3d.
    """
    stimuli: np.ndarray = field(default=None)

    def __post_init__(self):
        """Initialize the ResponseMapping object and validate required parameters."""
        super().__post_init__()
        
        # Check that stimuli have been specified
        if self.stimuli is None:
            raise ValueError(
                "Stimuli must be specified when initializing ResponseMapping. "
                "Please provide a stimuli array."
            )

    def calc_roi_sizes(self):
        """calculate sizes of ROIs in pixels based on their masks"""

        num_rois = self.num_rois
        roi_sizes = []
                
        for roi_idx in range(num_rois):
            roi_size = len(np.where(self.rois==-(roi_idx+1))[0])
            roi_sizes.append(roi_size)
        return pd.DataFrame(roi_sizes, columns=['roisize'])


    def calc_response_amplitude(self):
        """
        Calculate response amplitude for each presented stimulus.
        
        Needs stimuli as input to creating the object, and inherits averages and triggertimes from Core. 
        
        Returns
        -------
        pd.DataFrame
            DataFrame with response amplitudes indexed by ROI, with columns named after stimuli strings.
            Shape: (n_rois, n_stimuli)
        """
        # Get triggers from the Core class method
        triggers = self.calc_mean_triggertimes()
        
        # Use averages from Core
        traces = self.averages
        #traces = savgol_filter(traces, 1500, 5, axis=1)  # Smooth traces
        
        # Initialize dictionary to build DataFrame
        response_dict = {}
        
        for trig_idx, trig in enumerate(triggers[:-1]):  # Exclude last trigger since it's just white
            startval = traces[:, trig]
            maxval = np.max(traces[:, trig:triggers[trig_idx + 1]], axis=1)
            response = maxval - startval
            # Add response as a column with stimulus name as key
            stimulus_name = str(self.stimuli[trig_idx])
            response_dict[stimulus_name] = response
        # Create DataFrame from dictionary
        df = pd.DataFrame(response_dict)
        df.index.name = 'ROI'
        
        return df

    def calc_response_sd(self):
        """
        Calculate response standard deviation for each average trace.
        
        Needs stimuli as input to creating the object, and inherits averages and triggertimes from Core. 
        
        Returns
        -------
        pd.DataFrame
            DataFrame with response standard deviations indexed by ROI, with columns named after stimuli strings.
            Shape: (n_rois, n_stimuli)
        """
        
        # Use averages from Core
        average_sd = np.std(self.averages, axis=1)
        return average_sd


    def calc_roi_sizes_microns(self):
        """
        Calculate sizes of ROIs in microns based on 
        FoV and zoom level.
        
        Returns
        -------
        pd.Series
            Series with ROI sizes indexed by ROI number.
        """
        num_rois = self.num_rois
        roi_sizes = []
        fov_sizes = {
            'ntc3': {'0.15':539,
                    '0.21':439,
                    '0.32':308,
                    '0.43':206}, 
            'ntc1': {'0.15': 700,
                    '0.21': 500,
                    '0.32': 400,
                    '0.43': 300}
        }     
        for roi_idx in range(num_rois):
            roi_size = len(np.where(self.rois==-(roi_idx+1))[0])
            roi_sizes.append(roi_size)

        if pd.isna(self.optical_config):
            optical_config = 'ntc3'
            print(f"Optical config not found in metadata, assuming {optical_config}.")
        else:
            optical_config = self.optical_config

        img_size = self.average_stack.shape[1]  # Assuming square images
        zoom_rounded = round(self.zoom, 2)

        fov_size_um = fov_sizes.get(optical_config, {}).get(f"{zoom_rounded:.2f}", None)
        
        if fov_size_um is None:
            raise ValueError(f"FoV size not found for optical config {optical_config} and zoom {self.zoom}")
        pixel_size_um = fov_size_um / img_size
        roi_sizes_microns = np.array(roi_sizes) * pixel_size_um * pixel_size_um  # Area in square microns
        return roi_sizes_microns