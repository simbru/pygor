from dataclasses import dataclass, field
try:
    from collections import Iterable
except ImportError:
    from collections.abc import Iterable
# Local imports
from data_objects import Data, Experiment, Data_STRF
import unit_conversion
import signal_analysis
import utilities
import data_helpers
import helpinfo
import space
import contouring
import temporal
# Dependencies
from tqdm.auto import tqdm
import operator
import matplotlib.pyplot as plt
import joblib
import numpy as np
import pickle
import utilities
import inspect
import datetime
import pathlib
import h5py
import natsort
import textwrap
import pprint
import matplotlib.patheffects as path_effects
from utilities import multicolour_reshape as reshape
import warnings
import math
import copy
import pandas as pd
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib import cm
import skimage
import plotting

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

@dataclass
class Data:
    # def __str__(self):
    #     return "MyClass([])"
    # def __repr__(self):
    #     return f"{self.data_types}"
    filename: str or pathlib.Path
    metadata: dict = field(init=False)
    rois    : dict = field(init=False)
    data_types : list = field(default_factory=list)
    frame_hz : float = field(init=False)
    averages : np.array = np.nan
    snippets : np.array = np.nan
    ms_dur   : int = np.nan
    phase_num : int = 1 # Default to 1, for simplicity in plotting avgs etc...
    num_rois : int = field(init = False)
    def __post_init__(self):
        # Ensure path is pathlib compatible
        if isinstance(self.filename, pathlib.Path) is False:
            self.filename = pathlib.Path(self.filename)
        with h5py.File(self.filename, 'r') as HDF5_file:
            # Basic information
            self.metadata = data_helpers.metadata_dict(HDF5_file)
            self.rois = np.copy(HDF5_file["ROIs"])
            self.num_rois = len(np.unique(self.rois)) - 1
            self.images = np.array(HDF5_file["wDataCh0_detrended"]).T
            #print(HDF5_file["OS_Parameters"])
            #self.os_params = copy.deepcopy(list()
            # Timing parameters
            self.triggertimes = np.array(HDF5_file["Triggertimes"]).T
            self.triggertimes = self.triggertimes[~np.isnan(self.triggertimes)].astype(int)
            self.triggerstimes_frame = np.array(HDF5_file["Triggertimes_Frame"]).T
            self.triggerstimes_frame = self.triggerstimes_frame[~np.isnan(self.triggerstimes_frame)].astype(int)        
            self.__skip_first_frames = int(HDF5_file["OS_Parameters"][22]) # Note name mangling to prevent accidents if 
            self.__skip_last_frames = -int(HDF5_file["OS_Parameters"][23]) # private class attrs share names 
            # if self.__skip_last_frames == 0:
            #     self.__skip_last_frames = None
            # self.triggerstimes_frame = self.triggerstimes_frame[self.__skip_first_frames:self.__skip_last_frames]
            try:
                self.ipl_depths = np.copy(HDF5_file["Positions"])
            except KeyError:
                warnings.warn(f"HDF5 key 'Positions' not found for file {self.filename}", stacklevel=2)
                self.ipl_depths = np.nan
            if "Averages0" in HDF5_file.keys():
                self.averages = np.array(HDF5_file["Averages0"]).T
                self.ms_dur = self.averages.shape[-1]
            if "Snippets0" in HDF5_file.keys():
                self.snippets = np.array(HDF5_file["Snippets0"]).T
            self.frame_hz = float(HDF5_file["OS_Parameters"][58])
            self.trigger_mode = int(HDF5_file["OS_Parameters"][28])
        if self.trigger_mode != self.phase_num:
            warnings.warn(f"{self.filename.stem}: Trigger mode {self.trigger_mode} does not match phase number {self.phase_num}", stacklevel=3)
        # Initialise outside Pathlib context
        self.name = self.filename.stem
        self.__keyword_lables = {
            "ipl_depths" : self.ipl_depths,
        }
        self.__compare_ops_map = {
            '==' : operator.eq,
            '>' : operator.gt,
            '<' : operator.lt,
            '>=' : operator.ge,
            '<=' : operator.le,
        }

    def get_help(self, hints = False, types = False) -> None:
        method_list = helpinfo.get_methods_list(self, with_returns=types)
        attribute_list = helpinfo.get_attribute_list(self, with_types=types)
        welcome=helpinfo.welcome_help(self.data_types, self.metadata, hints = hints)
        attrs = helpinfo.attrs_help(attribute_list, hints = hints)
        meths = helpinfo.meths_help(method_list, hints = hints)
        helpinfo.print_help([welcome, attrs, meths, helpinfo.text_exit()])

    def view_stack_projection(self, func = np.mean, axis = 0, cbar = False, ax = None) -> None:
        """
        Display a projection of the image stack using the specified function.

        Parameters:
        - func: Callable, optional, default: np.mean
            The function used to compute the projection along the specified axis.
        - axis: int, optional, default: 0
            The axis along which the projection is computed.
        - cbar: bool, optional, default: False
            Whether to display a colorbar.
        - ax: matplotlib.axes.Axes, optional, default: None
            The matplotlib axes to use for the display. If None, the current axes will be used.

        Returns:
        None
        """
        if ax is None:
            ax = plt.gca()
        scanv = ax.imshow(func(self.images, axis = axis), cmap = "Greys_r", origin = "lower")
        if cbar == True:
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="5%", pad=0.05)
            plt.colorbar(scanv, ax=ax, cax = cax)
    
    def view_stack_rois(self, labels = True, func = np.mean, axis = 0, cbar = False,
        ax = None, figsize = (None, None), figsize_scale = None, **kwargs) -> None:
        """
        Display a projection of the image stack using the specified function.

        Parameters:
        - func: Callable, optional, default: np.mean
            The function used to compute the projection along the specified axis.
        - axis: int, optional, default: 0
            The axis along which the projection is computed.
        - cbar: bool, optional, default: False
            Whether to display a colorbar.
        - ax: matplotlib.axes.Axes, optional, default: None
            The matplotlib axes to use for the display. If None, the current axes will be used.

        Returns:
        None
        """
        if figsize == (None, None):
            figsize = (5, 5)
        if figsize_scale is not None:
            figsize = np.array(figsize) * np.array(figsize_scale)
        else:
            figsize_scale = 1
        if ax is None:
            fig, ax = plt.subplots(figsize = figsize)
        else:
            fig = plt.gcf()
        if "text_scale" in kwargs:
            txt_scl = kwargs["text_scale"]
        else:
            txt_scl = 1
        num_rois = int(np.abs(np.min(self.rois)))
        color = cm.get_cmap('jet_r', num_rois)
        scanv = ax.imshow(func(self.images, axis = axis), cmap ="Greys_r", origin = "lower")
        rois_masked = np.ma.masked_where(self.rois.T == 1, self.rois.T)
        rois = ax.imshow(rois_masked, cmap = color, alpha = 0.5, origin = "lower")
        ax.grid(False)
        ax.axis('off')
        if cbar == True:
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="5%", pad=0.05)
            plt.colorbar(rois, ax=ax, cax = cax)
        if labels == True:
            label_map = np.unique(self.rois)[:-1].astype(int)[::-1] #-1 to count forwards instead of backwards
            if "label_by" in kwargs:
                labels = self.__keyword_lables[kwargs["label_by"]]
                if np.isnan(labels) is True:
                    raise AttributeError(f"Attribute {kwargs['label_by']} not found in object.")
            else:
                labels= np.abs(label_map) - 1
            for label_loc, label in zip(label_map, labels):
                curr_roi_mask = self.rois.T == label_loc
                curr_roi_centroid = np.mean(np.argwhere(curr_roi_mask == 1), axis = 0)
                ax.text(curr_roi_centroid[1],curr_roi_centroid[0], label,
                    ma='center',va='center',ha = "center", c = "w", size = 12 * np.array(figsize_scale) * txt_scl, 
                    weight = "normal", path_effects=[path_effects.Stroke(linewidth = 2 * np.array(figsize_scale) * txt_scl,
                    foreground='k'), path_effects.Normal()])

    def view_drift(self, frame_num = "auto", butterworth_factor = .5, 
        chan_vese_factor = 0.01, ax = None) -> None:
        """
        View drift of images over time.

        Parameters
        ----------
        frame_num : str, optional
            The number of frames to use for the drift calculation. If "auto", it will use approximately 1/3 of the total frames.
        butterworth_factor : float, optional
            The Butterworth filter factor.
        chan_vese_factor : float, optional
            The Chan-Vese segmentation factor.
        ax : None or axis object, optional
            The axis to plot the result on. If None, it will use the current axis.

        Returns
        -------
        None
        """
        if ax is None:
            ax = plt.gca()
        def _prep_img(image : np.array) -> np.ndarray:
            image = skimage.filters.butterworth(image, .5, high_pass = False)
            image = skimage.morphology.diameter_closing(image, diameter_threshold=image.shape[-1])
            image = skimage.segmentation.chan_vese(image, 0.01).astype(int)
            return image
        # split array at 3 evenly spaced time points
        mid_split = math.floor(self.images.shape[0] / 2)
        # Split in 3 
        if frame_num == "auto":
            frame_num = math.floor(self.images.shape[0]/3)
        base = _prep_img(np.average(self.images[0:frame_num], axis = 0))
        base1 = _prep_img(np.average(self.images[mid_split:mid_split+frame_num], axis = 0))
        base2 = _prep_img(np.average(self.images[-frame_num-1:-1], axis = 0))
        # 
        d3 = plotting.stack_to_rgb(base)
        d3[:, :, 0] = base
        d3[:, :, 1] = base1
        d3[:, :, 2] = base2
        ax.imshow(utilities.min_max_norm(d3, 0, 1))  

    def plot_averages(self, rois = None, figsize = (None, None), figsize_scale = None, axs = None, **kwargs):
        """
        A function to plot the averages of specified regions of interest (rois) on separate subplots within a figure. 

        Parameters
        ----------
        rois : Iterable, optional
            Regions of interest to plot. If not specified, all rois will be plotted.
        figsize : tuple, optional
            Size of the figure to plot the subplots. Default is calculated based on the number of rois.

        Keyword arguments
        ----------
        filter_by : Tuple, optional
            Tuple in format (function, "operator", value) where 'function' is a mathematical function that
            can be applied along axis = 1 for self.vverages, '"operator"' is a mathematical operator (e.g, 
            "<", ">=", or "==") in string format, and 'value' is the threshold metric.
        sort_by : String, optional
            String representing attribute of data object, where the metric is ROI-by-ROI, such as a list
            or array where each element represents the metric of each ROI 
        label_by : String, optional
            As above, but instead of changing the order of plotting, changes the label associated with 
            each ROI to be the specified metric.

        Returns
        -------
        None
        """
        # Handle arguments, keywords, and exceptions
        if np.isnan(self.averages) == True:
            raise AttributeError("self.averages is nan, meaning it has likely not been generated")
        if isinstance(rois, Iterable) is False and not rois: # I dont like this solution, but it gets around user ambigouity error if passing numpy array
            rois = np.arange(0, self.num_rois)
            #fig, axs = plt.subplots(self.num_rois, figsize = figsize, sharey=True, sharex=True)
            # ^ no longer needed, since error handling of 'rois' leads to naturally solving this
        if isinstance(rois, Iterable) is False:
            rois = [rois]
        if isinstance(rois, np.ndarray) is False:
            rois = np.array(rois)
        if rois is not None and isinstance(rois, Iterable) is False:
            rois = np.array([rois])
        if "sort_by" in kwargs:
            rois = rois[np.argsort(self.__keyword_lables[kwargs["sort_by"]][rois].astype(int))]
        if "label_by" in kwargs:
            roi_labels = self.__keyword_lables[kwargs["label_by"]][rois]
        else:  
            roi_labels = rois
        if "filter_by" in kwargs:
              filter_result = self.__compare_ops_map[
                    kwargs["filter_by"][1]](kwargs["filter_by"][0]
                    (self.averages, axis = 1), kwargs["filter_by"][2] )
              rois = rois[filter_result]
        if figsize == (None, None):
            figsize = (10, len(rois))
        if figsize_scale is not None:
            figsize = np.array(figsize) * np.array(figsize_scale)
        # Generate matplotlib plot 
        colormap = plt.cm.jet_r(np.linspace(1,0,self.num_rois))
        if axs is None:
            fig, axs = plt.subplots(len(rois), figsize = figsize, sharey=True, sharex=True)
        else:
            fig = plt.gcf()
        # Loop through and plot wwithin axes
        if len(rois == 1): # This takes care of passing just 1 roi, not breaking axs.flat in the next line
            axs = np.array([axs])
        phase_dur = self.ms_dur / self.phase_num
        for ax, roi, label  in zip(axs.flat, rois, roi_labels):
            ax.plot(self.snippets[roi].T, c = "grey", alpha = .5)
            ax.plot(self.averages[roi], color = colormap[roi])
            ax.set_xlim(0, len(self.averages[0]))
            ax.set_yticklabels([])
            ax.set_ylabel(label, rotation = 0, verticalalignment = 'center')
            ax.spines[['top', "bottom"]].set_visible(False)
            # Now we need to add axvspans (don't think I can avoid for loops inside for loops...)
            if self.phase_num != 1:
                for interval in range(self.phase_num)[::2]:
                    ax.axvspan(interval * phase_dur, (interval + 1) * phase_dur, alpha = 0.2, color = 'gray', lw=0)
            ax.grid(False)
        ax.set_xlabel("Time (ms)")
        fig.subplots_adjust(wspace = 0, hspace = 0)
        #plt.tight_layout()
        return fig, axs
    def calculate_average_images(self, ignore_skip = False):
        """
        Calculate the average image from a series of trigger frames.

        Parameters:
        ----------
        self : object
            The instance of the class.
        
        Returns:
        -------
        numpy.ndarray
            The average image calculated from the trigger frames.
        """
        # Account for trigger skipping logic
        if ignore_skip is True:   
            # Ignore skipping parameters
            first_trig_frame = 0
            last_trig_frame = 0
            #triggers_frames = self.triggerstimes_frame
        else:
            # Othrwise, account for skipping parameters
            first_trig_frame = self.__skip_first_frames
            last_trig_frame = self.__skip_last_frames
            print(f"Skipping first {first_trig_frame} and last {last_trig_frame} frames")
        if last_trig_frame == 0:
            last_trig_frame = None
        triggers_frames = self.triggerstimes_frame[first_trig_frame:last_trig_frame]
        # Get the frame interval over which to average the images
        rep_start_frames = triggers_frames[::self.phase_num]
        rep_delta_frames = np.diff(triggers_frames[::self.phase_num]) # time between repetitions, by frame number
        rep_delta = int(np.floor(np.average(rep_delta_frames)))# take the average of the differentiated values, and round down. This your delta time in frames
        # Calculate number of full repetitions
        percise_reps = len(triggers_frames)/self.phase_num # This may yield a float if partial repetition
        if percise_reps % 1 != 0: # In that case, we don't have an exact loop so we ignore the residual partial loop
            reps = int(triggers_frames[:int(np.floor(percise_reps)%percise_reps*self.phase_num)].shape[0] / self.phase_num) # number of full repetitions, by removing residual non-complete repetitions
            print(f"Partial loop detected ({percise_reps}), using only", reps, "repetitions and ignoring last partial loop")
        else:
            reps = int(percise_reps)
        # Extract frames accordingly (could vectorize this)
        images_to_average = []
        for frame in rep_start_frames[:reps]:
            images_to_average.append(self.images[frame:frame+rep_delta])
        images_to_average = np.concatenate(images_to_average, axis = 0)
        # Print results
        print(f"{len(triggers_frames)} triggers with a phase_num of {self.phase_num} gives {reps} full repetitions of {rep_delta} frames each.")
        # Average the images
        split_arr = np.array(np.split(images_to_average, reps))
        avg_movie = np.average(split_arr, axis = 0)
        return avg_movie


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


@dataclass
class STRF(Data):
    type         : str = "STRF"
    # Params 
    multicolour  : bool = False
    bs_bool      : bool = True
    # Annotations
    strfs        : np.ndarray = field(init=False)
    ipl_depths   : np.ndarray = field(init=False)
    numcolour    : int = field(init=False) # gets interpreted from strf array shape
    strf_keys    : list = field(init=False)

    #bs_settings["do_bootstrap"] = bs_bool
    
    def __post_init__(self):
        # Post initialise the contents of Data class to be inherited
        super().__dict__["data_types"].append(self.type)
        super().__post_init__()
        self.bs_settings = data_helpers.create_bs_dict(do_bootstrap = self.bs_bool)
        with h5py.File(self.filename) as HDF5_file:
            # Get keys for STRF, filter for only STRF + n where n is a number between 0 to 9 
            keys = [i for i in HDF5_file.keys() if "STRF" in i and any(i[4] == np.arange(0, 10).astype("str"))]
            self.strf_keys = natsort.natsorted(keys)
            # Set bool for multi-colour RFs and ensure multicolour attributes set correctly
            bool_partofmulticolour_list = [len(n.removeprefix("STRF").split("_")) > 2 for n in keys]
            if all(bool_partofmulticolour_list) == True:
                multicolour_bool = True
            if all(bool_partofmulticolour_list) == False:
                multicolour_bool = False
            if True in bool_partofmulticolour_list and False in bool_partofmulticolour_list:
                raise AttributeError("There are both single-coloured and multi-coloured STRFs loaded. Manual fix required.")
            if multicolour_bool is True:
                self.numcolour = len(np.unique([int(i.split('_')[-1]) for i in self.strf_keys]))
                self.multicolour = True
            else:
                self.numcolour = 1
            self.strfs = data_helpers.load_strf(HDF5_file)
        self.num_strfs = len(self.strfs)

    ## Attributes

    ## Bootstrapping
    def __calc_pval_time(self) -> np.ndarray:
        """
        Calculate the p-value for each time point in the data.

        Returns:
            np.ndarray: An array of p-values for each time point.
        """
        # Generate bar for beuty
        bar = tqdm(self.strfs, leave = True, position = 0, disable = None, 
            desc = f"Hang on, bootstrapping temporal components {self.bs_settings['time_bs_n']} times")
        self._pval_time = np.array([signal_analysis.bootstrap_time(x, bootstrap_n=self.bs_settings["time_bs_n"]) for x in bar])
        return self._pval_time

    def __calc_pval_space(self) -> np.ndarray:
        """
        Calculate the p-value space for the spatial components.

        This function calculates the p-value space for the spatial components of the data. It performs a bootstrap
        analysis to estimate the spatial component's significance. The p-value space is an array of shape (n_strfs,)
        where n_strfs is the number of spatial frequency components in the data.

        Returns:
            np.ndarray: The p-value space array.

        """
        # Again, bar for niceness
        bar = tqdm(self.strfs, leave = True, position = 0, disable = None,
            desc = f"Hang on, bootstrapping spatial components {self.bs_settings['space_bs_n']} times")
        self._pval_space = np.array([signal_analysis.bootstrap_space(x, bootstrap_n=self.bs_settings["space_bs_n"]) for x in bar])

    def set_default_bootstrap_settings(self) -> None:
        """
        Sets the default bootstrap settings for the object.

        This function sets the default bootstrap settings for the object by calling the `create_bs_dict` method from the `data_helpers` module. The `do_bootstrap` parameter is set to the value of the `bs_bool` attribute of the object.

        Parameters:
            None

        Returns:
            None 
            
        """
        self.bs_settings =  data_helpers.create_bs_dict(do_bootstrap = self.bs_bool)

    def get_bootstrap_settings(self) -> dict:
        """
        Returns the bootstrap settings as a dictionary.

        :return: A dictionary containing the bootstrap settings.
        :rtype: dict
        """
        return self.bs_settings

    def update_bootstrap_settings(self, update_dict, silence_print=False) -> None:
        """
        A function to update the bootstrap settings based on the user input dictionary.

        Parameters
        ----------
        update_dict : dict
            A dictionary containing the settings to update.
        silence_print : bool, optional
            A flag indicating whether to print warnings or not. Defaults to False.

        Returns
        -------
        None
            This function does not return any value.

        Notes
        -----
        The following keys are disallowed and cannot be updated:
        - "bs_already_ran"
        - "bs_datetime"
        - "bs_datetime_str"
        - "bs_dur_timedelta"

        If any of these keys are present in the `update_dict`, they will be ignored and not updated.
        """
        # Generate default dict for default values 
        default_dict = data_helpers.create_bs_dict()
        # Check that keys are in default dict
        allowed_keys = list(default_dict.keys())
        # Specify keys that user is not allowed to change
        disallowed_keys = ["bs_already_ran", "bs_datetime", "bs_datetime_str", "bs_dur_timedelta"]
        # Get keys from user input, as a list
        user_keys = list(update_dict.keys())
        # Remove disallowed keys from user input
        not_allowed_keys = []
        not_found_keys   = []
        none_default_key = []
        for key in user_keys:
            if update_dict[key] == None:
                update_dict[key] = default_dict[key]
                none_default_key.append(key)
            if key not in disallowed_keys and key not in allowed_keys:
                not_found_keys.append(key)
                update_dict.pop(key)
            if key in disallowed_keys:
                not_allowed_keys.append(key)
                update_dict.pop(key)
        # Update self.bs_settings accordingly
        self.bs_settings.update({key : value for key, value in update_dict.items()})
        # Handle prints and warnings
        if silence_print == False:
            if not_allowed_keys or not_found_keys:
                warning_string = f"Input anomalies found"
                warnings.warn(warning_string, stacklevel=0)
                if not_allowed_keys:
                    print(f"Disallowed keys: {not_allowed_keys}")
                if not_found_keys:
                    print(f"Keys not found in self.bs_settings: {not_found_keys}")
                if none_default_key:
                    print(f"Keys set to default values: {[(i, default_dict[i]) for i in none_default_key]}")

    def run_bootstrap(self) -> None:
        """run_bootstrap Runs bootstrapping according to self.bs_settings

        Returns
        -------
        None
            Sets values for self._pval_space and self._pval_time
        """
        if self.bs_settings["do_bootstrap"] == False:
            raise AttributeError("self.bs_settings['do_bootstrap'] is not True")
        if self.bs_settings["do_bootstrap"] == True:
            if self.bs_settings["bs_already_ran"] == True:
                user_verify = input("Do you want re-do bootstrap? Type 'y'/'yes' or 'n'/'no'")
                user_verify = user_verify.lower()
                if user_verify == 'y' or user_verify == "yes":
                    before_time = datetime.datetime.now()
                    self.__calc_pval_time()
                    self.__calc_pval_space()
                    after_time = datetime.datetime.now()
                if user_verify == 'n' or user_verify == "no":
                    print(f"Skipping recomputing bootstrap due to user input:'{user_verify}'")
                    return
                else:
                    print(f"Input '{user_verify}' is invalid, no action done. Please use 'y'/'n'.")
                    return
            else:
                before_time = datetime.datetime.now()
                self.__calc_pval_time()
                self.__calc_pval_space()
                after_time = datetime.datetime.now()
                self.bs_settings["bs_already_ran"] = True
            # Write time metadata
            self.bs_settings["bs_datetime"] = before_time
            self.bs_settings["bs_datetime_str"] = before_time.strftime("%d/%m/%y %H:%M:%S")
            self.bs_settings["bs_dur_timedelta"] = after_time - before_time
            
    def pval_time(self) -> np.ndarray:
        """
        Returns an array of p-values for time calculation.

        Parameters
        ----------
            self (object): The object instance.
        
        Returns
        ---------
            np.ndarray: An array of p-values for time bootsrap.
        """
        if self.bs_settings["do_bootstrap"] == False:
            return [np.nan] * self.num_strfs
        if self.bs_settings["do_bootstrap"] == True:
            try:
                return self._pval_time
            except AttributeError:
                print("p value for time bootstrap did not exist, just making that now")
                self.__calc_pval_time() # Executes calculation and writes to self._pval_time
            return self._pval_time
    
    def pval_space(self) -> np.ndarray:
        """
        Returns an array of p-values for space calculation.

        Parameters
        ----------
            self (object): The object instance.
        
        Returns
        ---------
            np.ndarray: An array of p-values for space bootstrap.
        """
        if self.bs_settings["do_bootstrap"] == False:
            return [np.nan] * self.num_strfs
        if self.bs_settings["do_bootstrap"] == True:
            try:
                return self._pval_space
            except AttributeError:
                print("p value for space bootstrap did not exist, just making that now")
                self.__calc_pval_space() # Executes calculation and writes to self._pval_space
            return self._pval_space

    def pvals_table(self) -> pd.DataFrame:
        dict = {}
        if self.multicolour == True: 
            space_vals = utilities.multicolour_reshape(np.array(self.pval_space()), self.numcolour).T
            time_vals = utilities.multicolour_reshape(np.array(self.pval_time()), self.numcolour).T
            space_sig = space_vals < self.bs_settings["space_sig_thresh"]
            time_sig = time_vals < self.bs_settings["time_sig_thresh"]
            both_sig = time_sig * space_sig
            final_arr = np.hstack((space_vals, time_vals, both_sig), dtype="object")
            column_labels = ["space_R", "space_G", "space_B", "space_UV", "time_R", "time_G", "time_B", "time_UV",
            "sig_R", "sig_G", "sig_B", "sig_UV"]
            return pd.DataFrame(final_arr, columns=column_labels)
        else:
            space_vals = self.pval_space()
            time_vals = self.pval_space()
            space_sig = space_vals <  self.bs_settings["space_sig_thresh"]
            time_sig = time_vals < self.bs_settings["time_sig_thresh"]
            both_sig = time_sig * space_sig
            final_arr = np.stack((space_vals, time_vals, both_sig), dtype = "object").T
            column_labels = ["space", "time", "sig"]
            return pd.DataFrame(final_arr, columns = column_labels)
     
    def contours(self) -> '?':
        """
        Returns the contours of the collapse times.

        This function calculates the contours of the collapse times based on the specified bootstrap settings. If the bootstrap settings indicate that bootstrap should be performed, the function calculates the time and space p-values and uses them to determine whether a contour should be drawn for each collapse time. If bootstrap is not performed, the function simply calculates the contours for all collapse times.

        Returns:
            A numpy array of contours. Each contour is represented as a list of tuples, where each tuple contains the x and y coordinates of a point on the contour. If no contour is drawn for a specific collapse time, an empty list is returned.

        Raises:
            AttributeError: If the contours have not been calculated yet.

        Note:
            The contours are calculated using the `contouring.contour` function.

        Example:
            data = DataObject()
            contours = data.contours()
        """        
        try:
            return self.__contours
        except AttributeError:
            #self.__contours = [space.contour(x) for x in self.collapse_times()]
            if self.bs_settings["do_bootstrap"] == True:
                time_pvals = self.pval_time()
                space_pvals = self.pval_space()
                __contours = [contouring.contour(arr) # ensures no contour is drawn if pval not sig enough
                                if time_pvals[count] < self.bs_settings["time_sig_thresh"] and space_pvals[count] < self.bs_settings["space_sig_thresh"]
                                else  ([], [])
                                for count, arr in enumerate(self.collapse_times())]
            if self.bs_settings["do_bootstrap"] == False:
                __contours = [contouring.contour(arr) for count, arr in enumerate(self.collapse_times())]
            self.__contours = np.array(__contours, dtype = "object")
            return self.__contours    

    def contours_area(self, scaling_factor = 1) -> list:
        """
        Generate the area for each contour in the list of contours using the contours_area_bipolar function with a specified scaling factor.

        Parameters:
            scaling_factor (int): A scaling factor to adjust the area calculation (default is 1).

        Returns:
            list: A list of areas for each contour in the list.
        """
        return [contouring.contours_area_bipolar(__contours, scaling_factor = scaling_factor) for __contours in self.contours()]

    def contours_centroids(self) -> np.ndarray:
        try: 
            return self.__contours_centroids
        except AttributeError:
            #contours_arr = np.array(self.contours(), dtype = "object")
            off_contours = [contouring.contour_centroid(i) for i in self.contours()[:, 0]]
            on_contours = [contouring.contour_centroid(i) for i in self.contours()[:, 1]]
            self.__contours_centroids = np.array([off_contours, on_contours], dtype = "object")
            return self.__contours_centroids

    def contours_centres_by_pol(self) -> np.ndarray:
        try:
            return self.__centres_by_pol
        except AttributeError:
            self.__centres_by_pol = np.array([
                [np.average(i, axis = 0) for i in self.contours_centroids()[0, :]], 
                [np.average(i, axis = 0) for i in self.contours_centroids()[1, :]]])
            return self.__centres_by_pol

    def contours_centres(self, center_on = "biggest") -> np.ndarray:
        if center_on == "pols":
            return np.nanmean(self.contours_centres_by_pol(), axis = 0)
        if center_on == "biggest":
            pos_conts_cents = np.array([i[0] if i.size != 0 else np.array([np.nan, np.nan]) for i in self.contours_centroids()[1, :]])
            neg_conts_cents = np.array([i[0] if i.size != 0 else np.array([np.nan, np.nan]) for i in self.contours_centroids()[0, :]])
            area_cents = self.contours_area()
            neg_pos_largest = np.array([(i[np.argmax(i)], j[np.argmax(j)]) for i,j in area_cents])
            xs = np.where(neg_pos_largest[:, 0] > neg_pos_largest[:, 1], neg_conts_cents[:, 0], pos_conts_cents[:, 0])
            ys = np.where(neg_pos_largest[:, 0] > neg_pos_largest[:, 1], neg_conts_cents[:, 1], pos_conts_cents[:, 1])
            return np.array([xs, ys]).T #centres by biggest area, irrespective of polarity
        else:
            raise ValueError("center_on must be 'pols' or 'biggest'")
     
    def contours_complexities(self) -> np.ndarray:
        return contouring.complexity_weighted(self.contours(), self.contours_area())
     
    def timecourses(self, centre_on_zero = True) -> np.ndarray:
        try:
            return self.__timecourses 
        except AttributeError:
            timecourses = np.average(self.strf_masks(), axis = (3,4))
            first_indexes = np.expand_dims(timecourses[:, :, 0], -1)
            timecourses_centred = timecourses - first_indexes
            self.__timecourses = timecourses_centred
            return self.__timecourses
    
    ## Methods__________________________________________________________________________________________________________

    def plot_chromatic_overview(self):
        with warnings.catch_warnings(record=True) as w:
            # Cause all warnings to always be triggered.
            warnings.simplefilter("always")
            return plotting.chroma_overview(self)

    #def plo

    def dominant_timecourses(self):
        dominant_times = []
        for arr in self.timecourses.data:
            if np.max(np.abs(arr[0]) > np.max(np.abs(arr[1]))):
                dominant_times.append(arr[0])
            else:
                dominant_times.append(arr[1])
        dominant_times = np.array(dominant_times)
        return dominant_times

    def strf_masks(self, level = None):
        if self.strfs is np.nan:
            return np.nan
        else:
            # Apply space.rf_mask3d to all arrays and return as a new masksed array
            all_strf_masks = np.ma.array([space.rf_mask3d(x, level = None) for x in self.strfs])
            # # Get masks that fail criteria
            pval_fail_time = np.argwhere(np.array(self.pval_time()) > self.bs_settings["time_sig_thresh"]) # nan > thresh always yields false, so thats really convenient 
            pval_fail_space = np.argwhere(np.array(self.pval_space()) > self.bs_settings["space_sig_thresh"]) # because if pval is nan, it returns everything
            pval_fail = np.unique(np.concatenate((pval_fail_time, pval_fail_space)))
            # Set entire mask to True conditionally 
            all_strf_masks.mask[pval_fail] = True
        return all_strf_masks

    ## Methods 
    def calc_LED_offset(self, reference_LED_index = [0,1,2], compare_LED_index = [3]) -> np.ndarray:
        """
        Calculate the offset between average positions of reference and comparison LEDs.

        This method computes the offset between the average positions of reference and comparison
        light-emitting diodes (LEDs) based on their indices in a multicolored setup.

        Parameters:
        ----------
        num_colours : int, optional
            The number of colors or LED groups in the multicolored setup. Default is 4.
        reference_LED_index : list of int, optional
            List of indices corresponding to the reference LEDs for which the average position
            will be calculated. Must be list, even if only one LED. Default is [0, 1, 2].
        compare_LED_index : list of int, optional
            List of indices corresponding to the comparison LEDs for which the average position
            will be calculated. Must be list, even if only one LED. Default is [3].

        Returns:
        -------
        numpy.ndarray
            A 1-dimensional array representing the calculated offset between the average positions
            of comparison and reference LEDs.

        Raises:
        ------
        AttributeError
            If the object is not a multicolored spatial-temporal receptive field (STRF),
            indicated by `self.multicolour` not being True.

        Notes:
        ------
        This method assumes that the class instance has attributes:
        - multicolour : bool
            A flag indicating whether the STRF is multicolored.
        - contours_centres : list of numpy.ndarray
            List containing the center positions of contours for each color in the multicolored setup.

        Example:
        --------
        >>> strf_instance = STRF()
        >>> offset = strf_instance.calc_LED_offset(num_colours=4,
        ...                                       reference_LED_index=[0, 1, 2],
        ...                                       compare_LED_index=[3])
        >>> print(offset)
        array([x_offset, y_offset])
        """
        if self.multicolour == True:
            # Get the average centre position for each colour
            avg_colour_centre = np.array([np.nanmean(yx, axis = 0) for yx in utilities.multicolour_reshape(self.contours_centres(), self.numcolour)])
            # Get the average position for the reference LEDs and the comparison LEDs
            avg_reference_pos = np.nanmean(np.take(avg_colour_centre, reference_LED_index, axis = 0), axis = 0)
            avg_compare_pos = np.nanmean(np.take(avg_colour_centre, compare_LED_index, axis = 0), axis = 0)
            # Compute the difference 
            difference = np.diff((avg_compare_pos, avg_reference_pos), axis = 0)[0]
            return difference
        else:
            raise AttributeError("Not a multicoloured STRF, self.multicolour != True.")

    def timecourses_noncentred(self) -> np.ndarray:
        # Just return raw timecourses 
        self.__timecourses = np.average(self.strf_masks(), axis = (3,4))
    
    # This one should probably also just be a property to make syntax easier
    def rf_masks(self) -> (np.ndarray, np.ndarray):
        neg_mask2d, pos_mask2d = self.strf_masks().mask[:, :, 0][:, 0], self.strf_masks().mask[:, :, 0][:, 1]
        #return np.array([neg_mask2d, pos_mask2d])
        return (neg_mask2d, pos_mask2d)

    def rf_masks_combined(self) -> np.ndarray:
        mask_2d = self.rf_masks()
        neg_mask2d, pos_mask2d = mask_2d[0], mask_2d[1]
        mask2d_combined  = np.invert(neg_mask2d * -1) + np.invert(pos_mask2d)
        return mask2d_combined

    def contours_count(self) -> list:
        count_list = []
        for i in self.contours():
            neg_contours, pos_contours = i
            count_tup = (len(neg_contours), len(pos_contours))
            count_list.append(count_tup)
        return count_list

    # def contours_centered
    #     # Estimate centre pixel
    #     pixel_centres = self.contours_centres()
    #     avg_centre = np.round(np.nanmean(pixel_centres, axis = 0))
    #     # Find offset between centre coordinate andthe mean coordinate 
    #     differences = np.nan_to_num(avg_centre - pixel_centres).astype("int")
    #     # Loop through and correct
    #     overlap = np.ma.array([np.roll(arr, (x,y), axis = (1,0)) for arr, (y, x) in zip(collapsed_strf_arr, differences)])
    #     collapsed_strf_arr = overlap
    #     return 

    def collapse_times(self, zscore = False, mode = "var", spatial_centre = False) -> np.ma.masked_array:
        target_shape = (self.strfs.shape[0], 
                        self.strfs.shape[2], 
                        self.strfs.shape[3])    
        collapsed_strf_arr = np.ma.empty(target_shape)
        for n, strf in enumerate(self.strfs):
            collapsed_strf_arr[n] = space.collapse_3d(self.strfs[n], zscore = zscore, mode = mode)
        if spatial_centre == True:
            # Calculate shifts required for each image (vectorised)
            arr3d = collapsed_strf_arr
            # Ideally (but does not seem to work correctly, skewed by spurrious contours)
            contours_centers = np.where(self.contours_centres() > 0, np.floor(self.contours_centres()), np.ceil(self.contours_centres()))
            target_pos = np.array(arr3d.shape[1:]) / 2
            shift_by = target_pos - contours_centers
            #print("Shift by", shift_by)
            shift_by = np.nan_to_num(shift_by).astype(int) 
            shift_by = shift_by
            # np.roll does not support rolling 3D along depth, so 
            shifted = np.ma.array([np.roll(arr, shift_by[i], axis = (0,1)) for i, arr in enumerate(arr3d)])
            return shifted
            #collapsed_strf_arr = shifted
        return collapsed_strf_arr
        # space.collapse_3d(recording.strfs[strf_num])


    def polarities(self, exclude_FirstLast=(1,1)) -> np.ndarray:
        if self.strfs is np.nan:
            return np.array([np.nan])
        # Get polarities for time courses (which are 2D arrays containing a 
        # timecourse for negative and positive)
        polarities = temporal.polarity(self.timecourses, exclude_FirstLast)
        # Feed that to helper function to break it down into 1D/category
        return utilities.polarity_neat(polarities)

    def opponency_bool(self) -> [bool]:
        if self.multicolour == True:
            arr = utilities.multicolour_reshape(self.polarities(), self.numcolour).T
            # This line looks through rearranged chromatic arr roi by roi 
            # and checks the poliarty by getting the unique values and checking 
            # if the length is more than 0 or 1, excluding NaNs. 
            # I.e., if there is only 1 unique value, there is no opponency
            opponent_bool = [False if len(np.unique(i[~np.isnan(i)])) == 1|0 else True for i in arr]
            return opponent_bool
        else:
            raise AttributeError("Operation cannot be done since object property '.multicolour' is False")

    def polarity_category(self) -> [str]:
        result = []
        arr = utilities.multicolour_reshape(self.polarities(), self.numcolour).T
        for i in arr:
            inner_no_nan = np.unique(i)[~np.isnan(np.unique(i))]
            inner_no_nan = inner_no_nan[inner_no_nan != 0]
            if not any(inner_no_nan):
                result.append('empty')
            elif np.all(inner_no_nan == -1):
                result.append("off")
            elif np.all(inner_no_nan == 1):
                result.append("on")
            elif -1 in inner_no_nan and 1 in inner_no_nan and 2 not in inner_no_nan:
                result.append("opp")
            elif 2 in inner_no_nan:
                result.append("mix")
            else:
                result.append("other")
        return result

    #def amplitude_tuning_functions(self):
    def tunings_amplitude(self) -> np.ndarray:
        if self.multicolour == True:
            # maxes = np.max(self.collapse_times().data, axis = (1, 2))
            # mins = np.min(self.collapse_times().data, axis = (1, 2))
            maxes = np.max(self.dominant_timecourses().data, axis = (1))
            mins = np.min(self.dominant_timecourses().data, axis = (1))
            largest_mag = np.where(maxes > np.abs(mins), maxes, mins) # search and insert values to retain sign
            largest_by_colour = utilities.multicolour_reshape(largest_mag, self.numcolour)
            # signs = np.sign(largest_by_colour)
            # min_max_scaled = np.apply_along_axis(utilities.min_max_norm, 1, np.abs(largest_by_colour), 0, 1)
            tuning_functions = largest_by_colour
            return tuning_functions.T #transpose for simplicity, invert for UV - R by wavelength (increasing)
        else:
            raise AttributeError("Operation cannot be done since object property '.multicolour.' is False")

    def tunings_area(self, size = None, upscale_factor = 4) -> np.ndarray:
        if self.multicolour == True:
            # Step 1: Pull contour areas (note, the size is in A.U. for now)
            # Step 2: Split by positive and negative areas
            if size == None:
                warnings.warn("size stimates in arbitrary cartesian units")
                neg_contour_areas = [i[0] for i in self.contours_area()]
                pos_contour_areas = [i[1] for i in self.contours_area()]
            else:
                neg_contour_areas = [i[0] for i in self.contours_area(unit_conversion.au_to_visang(size)/upscale_factor)]
                pos_contour_areas = [i[1] for i in self.contours_area(unit_conversion.au_to_visang(size)/upscale_factor)]
            # Step 3: Sum these by polarity
            tot_neg_areas, tot_pos_areas = [np.sum(i) for i in neg_contour_areas], [np.sum(i) for i in pos_contour_areas]
            # Step 4: Sum across polarities 
            total_areas = np.sum((tot_neg_areas, tot_pos_areas), axis = 0)
            # Step 5: Reshape to multichromatic format
            area_by_colour = utilities.multicolour_reshape(total_areas, self.numcolour).T
            return area_by_colour #transpose for simplicity, invert for UV - R by wavelength (increasing)
        else:
            raise AttributeError("Operation cannot be done since object contains no property '.multicolour.")
    
    def tunings_centroids(self) -> np.ndarray:
        if self.multicolour == True:
            # Step 1: Pull centroids
            # Step 2: Not sure how to treat ons and offs yet, just do ONS for now 
            neg_centroids = self.spectral_centroids()[0]
            pos_centroids = self.spectral_centroids()[1]
            # Step 3: Reshaoe ti multichromatic 
            speed_by_colour_neg = utilities.multicolour_reshape(neg_centroids, self.numcolour).T
            speed_by_colour_pos = utilities.multicolour_reshape(pos_centroids, self.numcolour).T 
            # NaNs are 0 
            # roi = 4
            # speed_by_colour_neg = np.nan_to_num(speed_by_colour_neg)
            # speed_by_colour_pos = np.nan_to_num(speed_by_colour_pos)
            # plt.plot(speed_by_colour_neg[roi])
            # plt.plot(speed_by_colour_pos[roi])
            # speed_by_colour[1]
            return np.array([speed_by_colour_neg, speed_by_colour_pos])
        else:
            raise AttributeError("Operation cannot be done since object contains no property '.multicolour.")

    def tunings_peaktime(self, dur_s = 1.3) -> np.ndarray:
        if self.multicolour == True:
            # First get timecourses
            # Split by polarity 
            neg_times, pos_times = self.timecourses[:, 0], self.timecourses[:, 1]
            # Find max position in pos times and neg position in neg times 
            argmins = np.ma.argmin(neg_times, axis = 1)
            argmaxs = np.ma.argmax(pos_times, axis = 1)
            # Reshape result to multichroma 
            argmins  = utilities.multicolour_reshape(argmins, self.numcolour).T
            argmaxs  = utilities.multicolour_reshape(argmaxs, self.numcolour).T
            if dur_s != None:
                return  (dur_s / neg_times.shape[1]) * np.array([argmins, argmaxs])
            else:
                warnings.warn("Time values are in arbitary numbers (frames)")
                return np.array([argmins, argmaxs])
        else:
            raise AttributeError("Operation cannot be done since object contains no property '.multicolour.")

    def spectral_centroids(self) -> (np.ndarray, np.ndarray):
        spectroids_neg = np.apply_along_axis(temporal.only_centroid, 1, self.timecourses[:, 0])
        spectroids_pos = np.apply_along_axis(temporal.only_centroid, 1, self.timecourses[:, 1])
        return spectroids_neg, spectroids_pos

    def spectrums(self) -> (np.ndarray, np.ndarray):
        spectrum_neg = np.apply_along_axis(temporal.only_spectrum, 1, self.timecourses[:, 0])
        spectrum_pos = np.apply_along_axis(temporal.only_spectrum, 1, self.timecourses[:, 1])
        return spectrum_neg, spectrum_pos

    # def check_ipl_orientation(self):
    #     raise NotImplementedError("Current implementation does not yield sensible result")
    #     maxes = np.max(self.dominant_timecourses(), axis = 1)
    #     mins = np.min(self.dominant_timecourses(), axis = 1)
    #     # Combine the arrays into a 2D array
    #     combined_array = np.vstack((mins, maxes))
    #     # Find the index of the maximum absolute value along axis 0
    #     max_abs_index = np.argmax(np.abs(combined_array), axis=0)
    #     # Use the index to get the values with their signs intact
    #     result_values = combined_array[max_abs_index, range(combined_array.shape[1])]
    #     # Average every 4th value to get a weighted polarity for each ROI
    #     mean_pols_by_roi = np.nanmean(result_values.reshape(self.numcolour, -1), axis=0)
    #     return (np.sum(mean_pols_by_roi[:int(len(mean_pols_by_roi)/2)]), np.sum(mean_pols_by_roi[int(len(mean_pols_by_roi)/2):]))

    # def save_pkl(self, save_path, filename):
    #     fileehandling.save_pkl(self, save_path, filename)
    def save_pkl(self, save_path, filename) -> None:
        final_path = pathlib.Path(save_path, filename).with_suffix(".pkl")
        print("Storing as:", final_path, end = "\r")
        with open(final_path, 'wb') as outp:
            joblib.dump(self, outp, compress='zlib')




    
