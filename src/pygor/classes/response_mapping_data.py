from dataclasses import dataclass, field
import numpy as np
import pandas as pd
from pygor.classes.core_data import Core


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

    def calc_response_amplitude(self):
        """
        Calculate response amplitude for each presented stimulus.
        
        Needs stimuli as input to creating the object, and inherits averages and triggertimes from Core. 
        
        Returns
        -------
        np.ndarray
            Array of response amplitudes for each trigger, shape (n_triggers, n_rois)
        """
        # Get triggers from the Core class method
        triggers = self.calc_mean_triggertimes()
        
        # Use averages from Core
        traces = self.averages
        responses = []
        for trig_idx, trig in enumerate(triggers[:-1]):  # Exclude last trigger since it's just white
            startval = traces[:, trig]
            maxval = np.max(traces[:, trig:triggers[trig_idx + 1]], axis=1)
            responses.append(maxval - startval)
        return np.array(responses)
