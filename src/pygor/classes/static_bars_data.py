from pygor.classes.core_data import Core
from dataclasses import dataclass
import numpy as np
import pygor.core.methods
import pygor.timeseries.static_bars.methods
import pygor.timeseries.plot
@dataclass(kw_only=True, repr=False)
class StaticBars(Core):

    def __post_init__(self):
        
        # # Post initialise the contents of Data class to be inherited
        super().__post_init__()
        # None
        self.parameters = {
                "trigger_mode" : self.trigger_mode,
                "a_b_ratio" : .5,
                "ignore_percentage" : 0.1
                }

    def get_AB_deltas(self, roi = None):
        return pygor.timeseries.static_bars.methods.calculate_AB_deltas(self, roi = roi, a_b_ratio=self.parameters["a_b_ratio"], ignore_percentage=self.parameters["ignore_percentage"])

    def plot_AB_delta(self, roi = None, phase = 0):
        return pygor.timeseries.plot.plot_AB_delta(self, roi = roi, phase = phase, a_b_ratio=self.parameters["a_b_ratio"], ignore_percentage=self.parameters["ignore_percentage"])