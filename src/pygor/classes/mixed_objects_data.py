from dataclasses import dataclass
from pygor.classes.core_data import Core
import numpy as np
import pygor.timeseries.mixed_stimuli.methods

@dataclass(kw_only=True, repr=False)
class MixedStimuli(Core):
    stimtypes: np.ndarray = np.nan

    def __post_init__(self):

        super().__post_init__()

    def mean_triggertimes_ms(self):
        return pygor.timeseries.mixed_stimuli.methods.mean_triggertimes_ms(self)


