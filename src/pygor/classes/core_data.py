from dataclasses import dataclass, field

try:
    from collections import Iterable
except ImportError:
    from collections.abc import Iterable
# Local imports
import pygor.core.methods
import pygor.core.plot
import pygor.data_helpers
import pygor.utils.helpinfo
import pygor.strf.spatial
import pygor.strf.contouring
import pygor.strf.temporal
import pygor.plotting.basic
import pygor.utils
import pygor.core

# Dependencies
import operator
import matplotlib.pyplot as plt
import numpy as np
import pathlib
import h5py
import matplotlib.patheffects as path_effects
import matplotlib
import warnings
import scipy.ndimage
import math
from mpl_toolkits.axes_grid1 import make_axes_locatable
import skimage


def try_fetch(file, key):
    try:
        result = file[key]
        result_shape = result.shape
        if result_shape != ():
            result = np.array(result).T
    except KeyError as error:
        result = None
        # # raise KeyError(f"'{key}' not found in {file.filename}, setting to np.nan") from error
        # warnings.warn(
        #     f"'{key}' not found in {file.filename}, setting to np.nan", stacklevel=2
        # )
        error
    return result
    
def try_fetch_os_params(file, params_key):
    """
    Will always default to fetching given key from file["OS_Parameters"]
    """
    keys = np.squeeze(list(file["OS_Parameters"].attrs.items())[0][1])[1:] # this worked :')
    key_indices = np.arange(len(keys))
    key_dict = {keys[i]: i for i in key_indices}
    return file["OS_Parameters"][key_dict[params_key]]
    
@dataclass
class Core:
    filename: str or pathlib.Path
    metadata: dict = field(init=False)
    rois: dict = field(init=False)
    type: str = field(init=False)
    frame_hz: float = field(init=False)
    averages: np.array = np.nan
    snippets: np.array = np.nan
    ms_dur: int = np.nan
    trigger_mode : int = 1 #defualt value
    num_rois: int = field(init=False)

    def __post_init__(self):
        # Ensure path is pathlib compatible
        if isinstance(self.filename, pathlib.Path) is False:
            self.filename = pathlib.Path(self.filename)
        # Set type attribute
        self.type = self.__class__.__name__
        # Fetch all relevant data from the HDF5 file (if not in file, gets set to None)
        with h5py.File(self.filename, "r") as HDF5_file:
            # Data
            self.traces_raw = try_fetch(HDF5_file, "Traces0_raw")
            self.traces_znorm = try_fetch(HDF5_file, "Traces0_znorm")
            self.images = try_fetch(HDF5_file, "wDataCh0_detrended")
            # Basic information
            self.metadata = pygor.data_helpers.metadata_dict(HDF5_file)
            self.rois = try_fetch(HDF5_file, "ROIs")
            self.num_rois = len(np.unique(self.rois)) - 1
            # Timing parameters
            self.triggertimes = try_fetch(HDF5_file, "Triggertimes")
            self.triggertimes = self.triggertimes[~np.isnan(self.triggertimes)].astype(
                float
            )
            self.triggertimes_frame = try_fetch(HDF5_file, "Triggertimes_Frame")
            self.__skip_first_frames = int(try_fetch_os_params(HDF5_file, "Skip_First_Triggers")) # Note name mangling to prevent accidents if 
            self.__skip_last_frames = -int(try_fetch_os_params(HDF5_file, "Skip_Last_Triggers")) # private class attrs share names 
            self.ipl_depths = try_fetch(HDF5_file, "Positions")
            self.averages = try_fetch(HDF5_file, "Averages0")
            self.snippets = try_fetch(HDF5_file, "Snippets0")
            self.trigger_mode = int(try_fetch_os_params(HDF5_file, "Trigger_Mode"))
            self.n_planes = int(try_fetch_os_params(HDF5_file, "nPlanes"))
            self.linedur_s = float(try_fetch_os_params(HDF5_file, "LineDuration"))
            self.average_stack = try_fetch(HDF5_file, "Stack_Ave")
            # TODO: change the logic to conditionally pull info from OS_Params instead,
            # because images or averege_stack can sometimes be cropped, leading to 
            # inaccurate frame_hz. Better to fetch from metadata.
            if self.images is not None:
                self.frame_hz = float(1/(self.images.shape[1]/self.n_planes*self.linedur_s))
            else:
                self.frame_hz = float(1/(self.average_stack.shape[0]/self.n_planes*self.linedur_s))
        # Check that trigger mode matches phase number
        # if self.trigger_mode != self.phase_num:
        #     warnings.warn(
        #         f"{self.filename.stem}: Trigger mode {self.trigger_mode} does not match phase number {self.phase_num}",
        #         stacklevel=3,
        #     )
        # Imply from averages the ms_duration of one repeat
        if self.averages is not None:
            self.ms_dur = self.averages.shape[-1]
        else:
            self.ms_dur = None
        # Ensure triggertimes_frame does not include uneccessary nans
        if self.triggertimes_frame is not None:
            self.triggertimes_frame = self.triggertimes_frame[
                ~np.isnan(self.triggertimes_frame)
            ].astype(int)
        # Set name
        self.name = self.filename.stem
        # Set keyword lables
        self.__keyword_lables = {
            "ipl_depths": self.ipl_depths,
        }
        self.__compare_ops_map = {
            "==": operator.eq,
            ">": operator.gt,
            "<": operator.lt,
            ">=": operator.ge,
            "<=": operator.le,
        }

    def __repr__(self):
        # For pretty printing
        date = self.metadata["exp_date"].strftime("%d-%m-%Y")
        return f"{date}:{self.__class__.__name__}:{self.filename.stem}"

    def __str__(self):
        # For pretty printing
        return f"{self.__class__}"

    def get_help(self, hints=False, types=False) -> None:
        """
        Get help information for the object, including methods and attributes.

        Parameters
        ----------
        hints : bool, optional
            Whether to include hints in the help information (default is False)
        types : bool, optional
            Whether to include types in the help information (default is False)

        Returns
        -------
        None
        """
        method_list = pygor.utils.helpinfo.get_methods_list(self, with_returns=types)
        attribute_list = pygor.utils.helpinfo.get_attribute_list(self, with_types=types)
        welcome = pygor.utils.helpinfo.welcome_help(
            self.type, self.metadata, hints=hints
        )
        attrs = pygor.utils.helpinfo.attrs_help(attribute_list, hints=hints)
        meths = pygor.utils.helpinfo.meths_help(method_list, hints=hints)
        pygor.utils.helpinfo.print_help(
            [welcome, attrs, meths, pygor.utils.helpinfo.text_exit()]
        )

    def view_stack_projection(
        self,
        func=np.mean,
        axis=0,
        cbar=False,
        ax=None,
        figsize=(None, None),
        figsize_scale=None,
        zcrop: tuple = None,
        xcrop: tuple = None,
        ycrop: tuple = None,
        alpha=0.5,
        show_axes=False,
        **kwargs,
    ):
        """
        Display a projection of the image stack using the specified function.

        Parameters:
        - func: Callable or str, optional, default: np.mean
            The function used to compute the projection along the specified axis. 
            If "average_stack", uses self.average_stack directly.
            If pygor.core.methods.correlation_map, applies correlation mapping.
        - axis: int, optional, default: 0
            The axis along which the projection is computed.
        - cbar: bool, optional, default: False
            Whether to display a colorbar.
        - ax: matplotlib.axes.Axes, optional, default: None
            The matplotlib axes to use for the display. If None, the current axes will be used.
        - figsize: tuple, optional, default: (None, None)
            Figure size. If (None, None), defaults to (10, 10).
        - figsize_scale: float, optional, default: None
            Scale factor for figure size.
        - alpha: float, optional, default: 0.5
            Alpha transparency value (for consistency with view_stack_rois).

        Returns:
        tuple
            (fig, ax) - matplotlib figure and axes objects
        """
        if figsize == (None, None):
            figsize = (10, 10)
        if figsize_scale is not None:
            figsize = np.array(figsize) * np.array(figsize_scale)
        else:
            figsize_scale = 1
        if ax is None:
            fig, ax = plt.subplots(figsize=figsize)
        else:
            fig = plt.gcf()
        if zcrop is None:
            zstart = None
            zstop = None
        else:
            zstart = zcrop[0]
            zstop = zcrop[1]
        if xcrop is None:
            xstart = None
            xstop = None
        else:
            xstart = xcrop[0]
            xstop = xcrop[1]
        if ycrop is None:
            ystart = None
            ystop = None
        else:
            ystart = ycrop[0]
            ystop = ycrop[1]
        if func == "average_stack":
            scanv = ax.imshow(self.average_stack, cmap = "Greys_r", origin = "lower", **kwargs)
        elif func == pygor.core.methods.correlation_map:
            correlation_result = func(self.images[zstart:zstop, ystart:ystop, xstart:xstop])
            scanv = ax.imshow(correlation_result, cmap ="Greys_r", origin = "lower", **kwargs)
        else:
            scanv = ax.imshow(func(self.images[zstart:zstop, ystart:ystop, xstart:xstop:], axis = axis), cmap ="Greys_r", origin = "lower", **kwargs)
        if cbar == True:
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="5%", pad=0.05)
            plt.colorbar(scanv, ax=ax, cax=cax)
        if show_axes == False:
            ax.axis("off")
        return fig, ax

    def view_stack_rois(
        self,
        labels=True,
        func=np.mean,
        axis=0,
        cbar=False,
        ax=None,
        figsize=(None, None),
        figsize_scale=None,
        zcrop: tuple = None,
        xcrop: tuple = None,
        ycrop: tuple = None,
        alpha=0.5,
        outline=True,
        outline_smooth=True,
        outline_width=2,
        roi_indices=None,
        **kwargs,
    ):
        """
        Display a projection of the image stack using the specified function.

        Parameters:
        - func: Callable or str, optional, default: np.mean
            The function used to compute the projection along the specified axis.
            If "average_stack", uses self.average_stack directly.
            If pygor.core.methods.correlation_map, applies correlation mapping.
        - axis: int, optional, default: 0
            The axis along which the projection is computed.
        - cbar: bool, optional, default: False
            Whether to display a colorbar.
        - ax: matplotlib.axes.Axes, optional, default: None
            The matplotlib axes to use for the display. If None, the current axes will be used.
        - outline: bool, optional, default: False
            Whether to show ROIs as outlines instead of filled regions.
        - outline_smooth: bool, optional, default: True
            Whether to smooth the ROI outlines when outline=True.
        - outline_width: int, optional, default: 2
            Line width for ROI outlines when outline=True.
        - roi_indices: list or array-like, optional, default: None
            Indices of specific ROIs to display using 0-based indexing (0, 1, 2, etc.).
            If None, all ROIs are shown.

        Returns:
        tuple
            (fig, ax) - matplotlib figure and axes objects
        """
        if figsize == (None, None):
            figsize = (10, 10)
        if figsize_scale is not None:
            figsize = np.array(figsize) * np.array(figsize_scale)
        else:
            figsize_scale = 1
        if ax is None:
            fig, ax = plt.subplots(figsize=figsize)
        else:
            fig = plt.gcf()
        if "text_scale" in kwargs:
            txt_scl = kwargs["text_scale"]
        else:
            txt_scl = 1
        if zcrop is None:
            zstart = None
            zstop = None
        else:
            zstart = zcrop[0]
            zstop = zcrop[1]
        if xcrop is None:
            xstart = None
            xstop = None
        else:
            xstart = xcrop[0]
            xstop = xcrop[1]
        if ycrop is None:
            ystart = None
            ystop = None
        else:
            ystart = ycrop[0]
            ystop = ycrop[1]

        # Use rois_alt for 0-based indexing
        rois_to_use = self.rois_alt
        
        num_rois = int(np.nanmax(rois_to_use)) + 1  # +1 because 0-based indexing
        color = matplotlib.colormaps["jet_r"]
        
        if func == "average_stack":
            scanv = ax.imshow(self.average_stack, cmap = "Greys_r", origin = "lower", **kwargs)
        elif func == pygor.core.methods.correlation_map:
            correlation_result = func(self.images[zstart:zstop, ystart:ystop, xstart:xstop])
            scanv = ax.imshow(correlation_result, cmap ="Greys_r", origin = "lower", **kwargs)
        else:
            scanv = ax.imshow(func(self.images[zstart:zstop, ystart:ystop, xstart:xstop:], axis = axis), cmap ="Greys_r", origin = "lower", **kwargs)
        if outline:
            # Extract ROI outlines instead of filled regions
            from skimage import measure
            
            # Get unique ROI values from rois_alt (0-based indexing, background is NaN)
            roi_values = np.unique(rois_to_use)
            roi_values = roi_values[~np.isnan(roi_values)].astype(int)  # Remove NaN (background)
            
            # Filter by specific ROI indices if provided
            if roi_indices is not None:
                roi_indices = np.array(roi_indices)
                roi_values = roi_values[np.isin(roi_values, roi_indices)]
            
            # Check if any ROIs remain after filtering
            if len(roi_values) == 0:
                print(f"Warning: No ROIs found with indices {roi_indices}")
                # Create empty ScalarMappable for consistency
                import matplotlib.cm as cm
                norm = matplotlib.colors.Normalize(vmin=0, vmax=1)
                rois = cm.ScalarMappable(norm=norm, cmap=color)
            else:
                # Apply cropping to ROIs
                rois_cropped = rois_to_use[ystart:ystop, xstart:xstop]
                
                # Plot each ROI outline individually
                for i, roi_val in enumerate(roi_values):
                    # Create binary mask for current ROI
                    roi_mask = (rois_cropped == roi_val).astype(int)
                    
                    if np.sum(roi_mask) == 0:  # Skip if ROI not in cropped region
                        continue
                    
                    # Find contours
                    contours = measure.find_contours(roi_mask, 0.5)
                    
                    # Get color for this ROI
                    roi_color = color(i / len(roi_values))
                    
                    # Plot each contour
                    for contour in contours:
                        if outline_smooth:
                            # Apply Gaussian smoothing to contour coordinates
                            from scipy.ndimage import gaussian_filter1d
                            contour[:, 0] = gaussian_filter1d(contour[:, 0], sigma=1.5)
                            contour[:, 1] = gaussian_filter1d(contour[:, 1], sigma=1.5)
                            
                            # Close the contour by adding the first point to the end
                            contour = np.vstack([contour, contour[0]])

                        ax.plot(contour[:, 1], contour[:, 0], 
                            color=roi_color, linewidth=outline_width, alpha=alpha)
                
                # Create dummy mappable for colorbar compatibility
                import matplotlib.cm as cm
                norm = matplotlib.colors.Normalize(vmin=np.min(roi_values), vmax=np.max(roi_values))
                rois = cm.ScalarMappable(norm=norm, cmap=color)
        else:
            # Original filled region display using rois_alt
            rois_display = rois_to_use.copy()
            
            # Filter by specific ROI indices if provided
            if roi_indices is not None:
                roi_indices = np.array(roi_indices)
                # Create mask for ROIs not in roi_indices (set them to NaN)
                mask = ~np.isin(rois_display, roi_indices) & ~np.isnan(rois_display)
                rois_display[mask] = np.nan
            
            rois_masked = np.ma.masked_where(np.isnan(rois_display), rois_display)[ystart:ystop, xstart:xstop]
            rois = ax.imshow(rois_masked, cmap = color, alpha = alpha, origin = "lower", **kwargs)
        ax.grid(False)
        ax.axis("off")
        if cbar == True:
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="5%", pad=0.05)
            plt.colorbar(rois, ax=ax, cax=cax)
        if labels == True:
            # Get ROI values from rois_alt (0-based indexing)
            label_map = np.unique(rois_to_use)
            label_map = label_map[~np.isnan(label_map)].astype(int)  # Remove NaN (background)
            label_map = np.sort(label_map)  # Sort in ascending order
            
            # Filter label_map by roi_indices if provided
            if roi_indices is not None:
                roi_indices = np.array(roi_indices)
                label_map = label_map[np.isin(label_map, roi_indices)]
            if "label_by" in kwargs:
                labels = self.__keyword_lables[kwargs["label_by"]]
                if np.isnan(labels) is True:
                    raise AttributeError(
                        f"Attribute {kwargs['label_by']} not found in object."
                    )
            else:
                labels = label_map  # Use 0-based indexing directly
            for label_loc, label in zip(label_map, labels):
                curr_roi_mask = rois_to_use == label_loc
                curr_roi_centroid = np.mean(np.argwhere(curr_roi_mask == 1), axis=0)
                ax.text(
                    curr_roi_centroid[1],
                    curr_roi_centroid[0],
                    label,
                    ma="center",
                    va="center",
                    ha="center",
                    c="w",
                    size=12 * np.array(figsize_scale) * txt_scl,
                    weight="normal",
                    path_effects=[
                        path_effects.Stroke(
                            linewidth=2 * np.array(figsize_scale) * txt_scl,
                            foreground="k",
                        ),
                        path_effects.Normal(),
                    ],
                )
        
        return fig, ax

    def view_drift(
        self, frame_num="auto", butterworth_factor=0.5, chan_vese_factor=0.01, ax=None
    ) -> None:
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

        def _prep_img(image: np.array) -> np.ndarray:
            image = skimage.filters.butterworth(image, 0.5, high_pass=False)
            image = skimage.morphology.diameter_closing(
                image, diameter_threshold=image.shape[-1]
            )
            image = skimage.segmentation.chan_vese(image, 0.01).astype(int)
            return image

        # split array at 3 evenly spaced time points
        mid_split = math.floor(self.images.shape[0] / 2)
        # Split in 3
        if frame_num == "auto":
            frame_num = math.floor(self.images.shape[0] / 3)
        base = _prep_img(np.average(self.images[0:frame_num], axis=0))
        base1 = _prep_img(
            np.average(self.images[mid_split : mid_split + frame_num], axis=0)
        )
        base2 = _prep_img(np.average(self.images[-frame_num - 1 : -1], axis=0))
        #
        d3 = pygor.plotting.basic.stack_to_rgb(base)
        d3[:, :, 0] = base
        d3[:, :, 1] = base1
        d3[:, :, 2] = base2
        ax.imshow(pygor.utilities.min_max_norm(d3, 0, 1))

    def get_depth(self):
        """
        Get the depth of the images in the stack.

        Returns
        -------
        int
            The depth of the images in the stack.
        """
        session = pygor.core.NapariDepthPrompt(self)
        return session.run()

    def draw_rois(self, attribute = "calculate_image_average", style = "stacked",**kwargs):
        """
        Draw ROIs on the image stack.
        """
        def call_method(obj, method_str, *args, **kwargs):
            # Extract method name by stripping trailing parentheses (if present)
            method_name = method_str.split('(')[0].strip()  # Handles "method" or "method()"
            method = getattr(obj, method_name)  # Get the method from the object
            print(method)
            if attribute in obj.__dict__:
                return obj.__getattribute__(attribute)
            else:
                return method(*args, **kwargs)  # Call the method with arguments
        target = call_method(self, attribute)
        session = pygor.core.gui.methods.NapariRoiPrompt(target, traces_plot_style = style,**kwargs)
        return session.run()

    def plot_averages(
        self, rois=None, 
        figsize=(None, None), 
        figsize_scale=None, 
        axs=None, 
        independent_scale = False, 
        n_rois_raster = 50,
        sort_order = None,
        **kwargs,
    ):
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
            As above, but instead of changing the order of pygor.plotting.plots, changes the label associated with
            each ROI to be the specified metric.
        clim : tuple, optional
            A tuple that determines the lower and upper bounds of the clim for the imshow version of the plot, respectively.

        Returns
        -------
        None
        """
        if self.averages is None:
            warnings.warn("Averages do not exist.")
            return
        return pygor.core.plot.plot_averages(self, rois, figsize, figsize_scale, axs, independent_scale, n_rois_raster, sort_order, **kwargs)

    def calculate_image_average(self, ignore_skip=False):
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
        # If no repetitions
        if self.trigger_mode == 1:
            warnings.warn("No repetitions detected, returning original images")
            return
        # Account for trigger skipping logic
        if ignore_skip is True:
            # Ignore skipping parameters
            first_trig_frame = 0
            last_trig_frame = 0
            # triggers_frames = self.triggertimes_frame
        else:
            # Othrwise, account for skipping parameters
            first_trig_frame = self.__skip_first_frames
            last_trig_frame = self.__skip_last_frames
            print(
                f"Skipping first {first_trig_frame} and last {last_trig_frame} frames"
            )
        if last_trig_frame == 0:
            last_trig_frame = None
        triggers_frames = self.triggertimes_frame[first_trig_frame:last_trig_frame]
        # Get the frame interval over which to average the images
        rep_start_frames = triggers_frames[:: self.trigger_mode]
        rep_delta_frames = np.diff(
            triggers_frames[:: self.trigger_mode]
        )  # time between repetitions, by frame number
        rep_delta = int(
            np.floor(np.average(rep_delta_frames))
        )  # take the average of the differentiated values, and round down. This your delta time in frames
        # Calculate number of full repetitions
        percise_reps = (
            len(triggers_frames) / self.trigger_mode
        )  # This may yield a float if partial repetition
        if (
            percise_reps % 1 != 0
        ):  # In that case, we don't have an exact loop so we ignore the residual partial loop
            reps = int(
                triggers_frames[
                    : int(np.floor(percise_reps) % percise_reps * self.trigger_mode)
                ].shape[0]
                / self.trigger_mode
            )  # number of full repetitions, by removing residual non-complete repetitions
            print(
                f"Partial loop detected ({percise_reps}), using only",
                reps,
                "repetitions and ignoring last partial loop",
            )
        else:
            reps = int(percise_reps)
        # Extract frames accordingly (could vectorize this)
        images_to_average = []
        for frame in rep_start_frames[:reps]:
            images_to_average.append(self.images[frame : frame + rep_delta])
        images_to_average = np.concatenate(images_to_average, axis=0)
        # Print results
        print(
            f"{len(triggers_frames)} triggers with a trigger_mode of {self.trigger_mode} gives {reps} full repetitions of {rep_delta} frames each."
        )
        # Average the images
        split_arr = np.array(np.split(images_to_average, reps))
        avg_movie = np.average(split_arr, axis=0)
        return avg_movie

    def get_average_markers(self):
        return pygor.core.methods.determine_epoch_markers_ms(self)
    
    def get_epoch_dur(self, rtol=1e-3, atol=1e-3):
        # Differentiate the average markers to get the epoch durations
        diff = np.diff(self.get_average_markers())
        if diff.size == 0:
            warnings.warn("No epochs found, returning 0")
            return 0
        # Calculate the average duration of the epochs
        avg = np.average(diff)
        # Check for unequal epochs
        if np.allclose(diff, avg, rtol=rtol, atol=atol) is False:
            # If unequal, raise error and ask user to manually set epoch durations
            raise ValueError(
                f"Epoch durations are not equal with tolerences {rtol}, {atol}, adjust tolerences or manually set epoch durations."
            )
        return np.floor(np.average(np.diff(self.get_average_markers()))).astype(int)

    def get_correlation_map(self):
        return pygor.core.methods.correlation_map(self.images)

    @property
    def rois_alt(self):
        temp_rois = self.rois.copy()
        temp_rois[temp_rois == 1] = np.nan
        temp_rois *= -1
        temp_rois = temp_rois - 1
        return temp_rois
    
    @property
    def roi_centroids(self, force = True):
        """
        Get the centre of mass for each ROI in the image if not already done.
        """
        if np.all(np.logical_or(np.unique(self.rois) < 0, np.unique(self.rois) == 1)):
            temp_rois = self.rois_alt + 1
            labels = np.unique(temp_rois)
            labels = labels[~np.isnan(labels)]
            centroids = scipy.ndimage.center_of_mass(temp_rois, temp_rois, labels)
        else:
            labels = np.unique(self.rois)
            labels = labels[~np.isnan(labels)]
            centroids = scipy.ndimage.center_of_mass(self.rois, self.rois, labels)
        return np.array(centroids)