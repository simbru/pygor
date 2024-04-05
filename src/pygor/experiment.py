from dataclasses import dataclass, field
try:
    from collections import Iterable
except ImportError:
    from collections.abc import Iterable
# Local imports
import pygor.utils.unit_conversion as unit_conversion
from pygor.utils.utilities import multicolour_reshape as reshape
import pygor.steps.signal_analysis as signal_analysis
import pygor.data_helpers
import pygor.utils.helpinfo
import pygor.space
import pygor.steps.contouring
import pygor.temporal
import pygor.plotting.plots
# Get dataobjects
from pygor.data_objs.data_parent import Data
from pygor.data_objs.strf import STRF

# Dependencies
from tqdm.auto import tqdm
import operator
import matplotlib.pyplot as plt
import joblib
import numpy as np
import pickle
import inspect
import datetime
import pathlib
import h5py
import natsort
import textwrap
import pprint
import matplotlib.patheffects as path_effects
import matplotlib
import warnings
import math
import copy
import pandas as pd
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib import cm
import skimage

@dataclass
class Experiment:
    data_types = []
    def __str__(self):
        return "MyClass([])"
    def __repr__(self, string = data_types):
        return f"Experiment({string})"
    def __post_init__(self):
        self.data_types : []    
    @classmethod
    def attach_data(self, obj, assumptions = True):        
        """
        TODO Here there should be several tests that check 
        the plane number, rec number, date, and ROI number (more?) 
        with a reference set 
        """       
        def _insert(obj_instance):
            setattr(self, obj_instance.type, obj)
            if obj_instance.type not in self.data_types:
                self.data_types.append(obj_instance.type)
        # self.__dict__[obj.type] = obj
        if isinstance(obj, Iterable):
            raise AttributeError("'obj' must be single ")
            # for i in obj:
                # _insert(i)
        else:
            _insert(obj)
            # self.__repr__ = "ooglie"
        return None

    def detach_data(self, obj):
        del(self.__dict__[obj.type])
        self.data_types.remove(obj.type)
        # return None

    def set_ref(self, obj):
        return None
    def change_ref(self, obj):
        return None 
    def clear_ref(self, obj):
        return None


@dataclass(kw_only=True)
class CenterSurround(Data):
    phase_num : int
    type : str = "CS"
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

@dataclass(kw_only=True)
class MovingBars(Data):
    dir_num : int 
    col_num : int

    type : str = "FFF"
    def __post_init__(self):
        # Post initialise the contents of Data class to be inherited
        super().__dict__["data_types"].append(self.type)
        super().__post_init__()

    def split_snippets_chromatically(self) -> np.ndarray: 
        "Returns snippets split by chromaticity, expect one more dimension than the averages array (repetitions)"
        return np.array(np.split(self.snippets[:, :, 1:], self.col_num,axis=-1))

    def split_averages_chromatically(self) -> np.ndarray:
        "Returns averages split by chromaticity"
        return np.array(np.split(self.averages[:, 1:], self.col_num,axis=-1))

@dataclass(kw_only=True)
class FullField(Data):
    # key-word only, so phase_num must be specified when initialising Data_FFF
    phase_num : int
    ipl_depths : np.ndarray = np.nan
    type : str = "FFF"
    # Post init attrs
    name : str = field(init=False)
    averages : np.array = field(init=False)
    ms_dur   : int = field(init=False)
    def __post_init__(self):
        # Post initialise the contents of Data class to be inherited
        super().__dict__["data_types"].append(self.type)
        super().__post_init__()
        # with h5py.File(self.filename) as HDF5_file:
        #     # Initilaise object properties 
        #     self.name = self.filename.stem
        #     #self.averages = np.copy(HDF5_file["Averages0"])
        #     # self.raw_traces = np.copy(HDF5_file["Averages0"])
        #     self.ms_dur = self.averages.shape[-1]






    
