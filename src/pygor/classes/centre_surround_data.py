from dataclasses import dataclass, field
from pygor.classes.core_data import Core

import numpy as np
import matplotlib.pyplot as plt

@dataclass(kw_only=True, repr=False)
class CenterSurround(Core):
    phase_num : int
    def __post_init__(self):
        # Post initialise the contents of Data class to be inherited
        super().__dict__["data_types"].append(self.type)
        super().__post_init__()

    def plot_phasic(self, roi = None, stims = None, bar_interval= 1, plot_avg = False):
        
        """
        TODO 
        - Docstring
        - Add bar_everyother
        """
        
        if stims == None:
            stims : int # type annotation
            stims = self.phase_num
        if roi == None:
            times = self.averages
        else:
            times = self.averages[roi]
        if times.ndim == 1: 
            times = np.array([times])
        for i in times:
            plt.plot(i, label = i)
            try:
                sections = np.split(i, stims * 2)
                #dur = times.shape[1]/2/stims
            except ValueError:
                len_min_remainder = len(i) - len(i) % (stims *2 + 1)
                sections = np.split(i[:len_min_remainder], stims * 2 + 1)
                #dur = len_min_remainder
            if plot_avg == True:
                for i in range(len(sections)):
                    dur = 1
                    raise NotImplementedError("Not implemented yet")
                    point1 = [dur * i, dur * (i+1)]
                    point2 = [np.average(sections[i]), np.average(sections[i])]
                    plt.plot(point1, point2, '-')
        print(stims / bar_interval)
        # for i in range(stims / bar_interval)[::bar_interval]:
        #     span_dur = snippets.shape[1]/stims
        #     plt.axvspan(span_dur * i, span_dur * (i+1) ,alpha = 0.25)

        # for i in range(stims * bar_interval)[::bar_interval]:
        #     print(i * dur)
        #     dur = times.shape[1]/stims/bar_interval
        #     plt.axvspan(dur * i, dur * (i+1) ,alpha = 0.25)
        plt.axhline(0, c = 'grey', ls = '--')
