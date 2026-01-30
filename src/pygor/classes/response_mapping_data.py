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
    

    def calc_response_amplitude(self, response_window_s=8, baseline_window_s=5):
        """
        Calculate response amplitude for each presented stimulus.
        
        Measures the response (max - start) in a window after stimulus onset,
        and subtracts the baseline response (max - start) from a window before
        stimulus onset.
        
        Needs stimuli as input to creating the object, and inherits averages and triggertimes from Core. 
        
        Parameters
        ----------
        response_window_s : float, optional
            Maximum duration in seconds to measure response after stimulus onset.
            If the next stimulus comes sooner, uses that as the cutoff. Default is 8.
        baseline_window_s : float, optional
            Duration in seconds before stimulus onset to measure baseline response.
            Default is 5.
        
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
        
        # Apply 1 second boxcar filter
        from scipy.ndimage import uniform_filter1d
        sampling_rate = int(1 / self.linedur_s)  # Calculate sampling rate from line duration
        boxcar_window_size = int(1 * sampling_rate)  # 1 second window
        traces_filtered = uniform_filter1d(traces, size=boxcar_window_size, axis=1, mode='nearest')
        
        # Calculate window sizes in samples
        max_response_samples = int(response_window_s * sampling_rate)
        baseline_samples = int(baseline_window_s * sampling_rate)
        
        # Initialize dictionary to build DataFrame
        response_dict = {}
        
        for trig_idx, trig in enumerate(triggers[:-1]):  # Exclude last trigger since it's just white
            # Determine the end of the response window (limited to response_window_s or next trigger)
            next_trig = triggers[trig_idx + 1]
            response_window_end = min(trig + max_response_samples, next_trig)
            
            # Calculate stimulus response (maxval - startval) in the response window
            startval_stim = traces_filtered[:, trig]
            maxval_stim = np.max(traces_filtered[:, trig:response_window_end], axis=1)
            response_stim = maxval_stim - startval_stim
            
            # Calculate baseline response in the pre-stimulus window
            baseline_start = max(0, trig - baseline_samples)
            baseline_end = trig
            
            # Only subtract baseline if there's a valid pre-stimulus period
            if baseline_start < baseline_end:
                startval_baseline = traces_filtered[:, baseline_start]
                maxval_baseline = np.max(traces_filtered[:, baseline_start:baseline_end], axis=1)
                response_baseline = maxval_baseline - startval_baseline
                response = response_stim - response_baseline
            else:
                # No pre-stimulus period available, use stimulus response only
                response = response_stim
            
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