import threading
import time
import logging
import sys
import numpy as np
from scipy.signal import savgol_filter
from qtpy.QtCore import QEventLoop
import matplotlib
from skimage.draw import polygon
from IPython import get_ipython
import matplotlib.pyplot as plt
import napari
import napari
from qtpy.QtWidgets import QApplication
from qtpy.QtCore import QEventLoop
import pygor.strf.spatial
class NapariViewRois:
    def __init__(self, pygor_object):
        self.viewer = napari.Viewer()
        self.pygor_object = pygor_object
        # Create a Qt event loop
        self.event_loop = QEventLoop()

    def run(self):
        """Launch Napari and block execution properly."""
        avg_movie = np.average(self.pygor_object.images, axis=0)
        std_movie = np.std(self.pygor_object.images, axis=0)
        corr_movie = pygor.strfs.spatial._legacy_corr_spacetime(self.pygor_object.images)
        # Make layers
        self.viewer.add_image(avg_movie, name = "Average", colormap = "Greys_r")
        self.viewer.add_image(std_movie, name = "SD", colormap = "Greys_r")
        self.viewer.add_points(self.pygor_object.roi_centroids, name = "ROI centroids", opacity = 1, face_color="orange", size = 1.5)
        self.viewer.add_image(self.pygor_object.rois_alt, name = "ROIs", colormap = "rainbow", opacity = 0.25)

        napari.run()
        
class NapariDepthPrompt:
    def __init__(self, pygor_object):
        self.result = None  # Store the computed value
        self.viewer = napari.Viewer()
        self.pygor_object = pygor_object
        self.last_active_layer = None # set this later when Viewer is populated

        # Create a Qt event loop
        self.event_loop = QEventLoop()

        # Override the close event
        original_close_event = self.viewer.window._qt_window.closeEvent

        def custom_close_event(event):
            self.on_close()  # Run computations on close
            original_close_event(event)  # Ensure proper closing
            self.event_loop.quit()  # Exit the event loop

        self.viewer.window._qt_window.closeEvent = custom_close_event

    def interp_coords(self, coords, n_points = 1000, smooth = True, smooth_window = None, poly_order_smooth = 3):
        if isinstance(coords, list):
            coords = np.array(coords)
        # Compute the cumulative distance along the path
        distances = np.cumsum(np.linalg.norm(np.diff(coords, axis=0), axis=1))
        distances = np.insert(distances, 0, 0)  # Insert 0 at the beginning
        # Generate 128 evenlybb spaced points along the distance
        interp_distances = np.linspace(0, distances[-1], n_points)
        # Interpolate x and y separately
        x_interp = np.interp(interp_distances, distances, coords[:, 0])
        y_interp = np.interp(interp_distances, distances, coords[:, 1])
        # Combine interpolated points
        interpolated_coords = np.column_stack((x_interp, y_interp))
        if smooth:
            # Apply Savitzky-Golay filter
            if smooth_window is None:
                smooth_window = int(len(x_interp)/2)
            if smooth_window % 2 == 0:
                smooth_window -= 1
            window_length = smooth_window# Must be an odd number
            x_smooth = savgol_filter(x_interp, window_length, poly_order_smooth)
            y_smooth = savgol_filter(y_interp, window_length, poly_order_smooth)
            # Stack smoothed coordinates
            smoothed_coords = np.column_stack((x_smooth, y_smooth))
            return smoothed_coords
        else:
            return interpolated_coords

    def determine_orientation(self, x_func, y_func):
        """
        Determines if the variations in the X direction are greater than or equal to the variations in the Y direction.
        
        Parameters:
            x_func (numpy.ndarray): Array representing the X component.
            y_func (numpy.ndarray): Array representing the Y component.
        
        Returns:
            int: 1 if the function is more "horizontal" (X variations dominate), 0 otherwise.
        """
        # Ensure input is a NumPy array
        x_func = np.asarray(x_func)
        y_func = np.asarray(y_func)

        # Normalize the waves
        x_max = np.max(np.abs(x_func)) if np.any(x_func) else 1
        y_max = np.max(np.abs(y_func)) if np.any(y_func) else 1
        x_calc = x_func / x_max
        y_calc = y_func / y_max

        # Compute numerical derivative
        calc_dif_x = np.diff(x_calc)
        calc_dif_y = np.diff(y_calc)

        # Compute range as product of max and min values
        x_range = np.max(calc_dif_x) * np.min(calc_dif_x) if len(calc_dif_x) > 0 else 0
        y_range = np.max(calc_dif_y) * np.min(calc_dif_y) if len(calc_dif_y) > 0 else 0
        # Determine dominant direction
        return int(x_range >= y_range)

    def reorder(self, xwave, ywave):
        """
        Sorts xwave in ascending order and reorders ywave accordingly.

        Parameters:
            xwave (numpy.ndarray): X values.
            ywave (numpy.ndarray): Y values corresponding to xwave.

        Returns:
            tuple: (sorted_xwave, sorted_ywave), or (-1, -1) if input sizes do not match.
        """
        # Ensure input is a NumPy array
        xwave = np.asarray(xwave)
        ywave = np.asarray(ywave)

        # Check if both waves have the same number of points
        if len(xwave) != len(ywave):
            print("X and Y wave must have the same number of points!")
            return -1, -1  # Error case

        # Sort xwave and reorder ywave accordingly
        sorted_indices = np.argsort(xwave)  # Get indices for sorting xwave
        sorted_xwave = xwave[sorted_indices]  # Sort xwave
        sorted_ywave = ywave[sorted_indices]  # Reorder ywave accordingly

        return np.array([sorted_xwave, sorted_ywave]).T

    def calculate_depths(self, lower, upper, roi_centroids):
        upper_x_coords = upper[:, 1]
        lower_x_coords = lower[:, 1]
        # find closest corresponding x point on upper and lower contours
        roi_x_coords = roi_centroids[:, 1] # Shape (y, x)
        roi_y_coords = roi_centroids[:, 0]
        # compute absolute differences using broadcasting
        diff_upper = np.abs(upper_x_coords[:, None] - roi_x_coords)  # Shape (len(arr), len(targets))
        diff_lower = np.abs(lower_x_coords[:, None] - roi_x_coords)
        # find index of the minimum difference for each target on X axis
        closest_indices_upper = np.argmin(diff_upper, axis=0)
        closest_indices_lower = np.argmin(diff_lower, axis=0)
        # # get the closest values along X axis 
        # closest_values_upper = upper_x_coords[closest_indices_upper]
        # closest_values_lower = lower_x_coords[closest_indices_lower]
        # get the corresponding y values for the closest values along X axis
        closest_y_upper = upper[closest_indices_upper, 0]  # Y-coords of upper contour
        closest_y_lower = lower[closest_indices_lower, 0]  # Y-coords of lower contour
        percent_position = (roi_y_coords - closest_y_lower) / (closest_y_upper - closest_y_lower) * 100
        return percent_position

    def calculate_depths_vertical_fixed(self, lower, upper, roi_centroids):
        """
        Calculate depths for vertical scans - match ROIs by Y coordinates, measure depth along X axis.
        This is the corrected version that properly handles vertical scan geometry.
        """
        # For vertical scans: match by Y coordinates, measure depth along X axis
        upper_y_coords = upper[:, 0]  # Y coordinates for matching
        lower_y_coords = lower[:, 0]  # Y coordinates for matching
        
        # ROI coordinates (roi_centroids is in (y, x) format)
        roi_y_coords = roi_centroids[:, 0]  # Y coordinates for matching
        roi_x_coords = roi_centroids[:, 1]  # X coordinates for depth measurement
        
        # Find closest points on boundaries using Y coordinate matching
        diff_upper = np.abs(upper_y_coords[:, None] - roi_y_coords)
        diff_lower = np.abs(lower_y_coords[:, None] - roi_y_coords)
        
        closest_indices_upper = np.argmin(diff_upper, axis=0)
        closest_indices_lower = np.argmin(diff_lower, axis=0)
        
        # Get the corresponding X coordinates for depth calculation
        closest_x_upper = upper[closest_indices_upper, 1]  # X coords of upper boundary
        closest_x_lower = lower[closest_indices_lower, 1]  # X coords of lower boundary
        
        # Calculate depth percentage along X axis
        # Check if boundaries are reversed (0% on right, 100% on left)
        if np.mean(closest_x_lower) > np.mean(closest_x_upper):
            # Reversed case: 0% is on right (higher X), 100% is on left (lower X)
            percent_position = (closest_x_lower - roi_x_coords) / (closest_x_lower - closest_x_upper) * 100
        else:
            # Normal case: 0% is on left (lower X), 100% is on right (higher X) 
            percent_position = (roi_x_coords - closest_x_lower) / (closest_x_upper - closest_x_lower) * 100
        return percent_position


    def process_data(self):
        """Calculates the depth of each ROI between the 0% and 100% boundaries."""
        print("Processing user selection...")
        # Finally, do the calcluation
        upper, lower = np.squeeze(self.viewer.layers["0% boundary"].data[0]), np.squeeze(self.viewer.layers["100% boundary"].data[0])
        orientation = self.determine_orientation(lower[:, 1], lower[:, 0])
        or_str = "Vertical" if orientation == 0 else "Horizontal"
        logging.info(f"Orientation is: {or_str}")
        if orientation == 1:
            depths = self.calculate_depths(lower, upper, self.pygor_object.roi_centroids)
        else:
            logging.info("Vertical orientation detected")
            # For vertical scans, use original coordinates directly (no reordering)
            depths = self.calculate_depths_vertical_fixed(lower, upper, self.pygor_object.roi_centroids)
        self.result = depths

    def on_close(self):
        """Function triggered when the viewer closes."""
        print("Viewer closed. Running final computation...")
        if self.viewer.layers["0% boundary"].data != [] and self.viewer.layers["100% boundary"].data != []:
            self.process_data()  # Compute the final result
        else:
            print("No data to process. Missing boundaries.")
            self.result = None

    def on_layer_switch(self, event):
        """Triggered when the active layer changes."""
        current_layer = self.viewer.layers.selection.active        
        if current_layer is None:
            return  # No active layer selected
        print(f"Switched from '{self.last_active_layer.name}' to '{current_layer.name}'")
        if self.last_active_layer.name == "100% boundary" or self.last_active_layer.name == "0% boundary":
            if self.last_active_layer.data != []:
                print(f"Interpolating coordinates for '{self.last_active_layer.name}'")
                self.last_active_layer.data = self.interp_coords(self.last_active_layer.data[-1])
        # Update the last active layer
        self.last_active_layer = current_layer

    def run(self):
        """Launch Napari and block execution properly."""
        # Average movie to give a 2D projection
        avg_movie = np.average(self.pygor_object.images, axis=0)
        # Make layers
        self.viewer.add_image(avg_movie, name = "Average stack", colormap = "Greys_r")
        self.viewer.add_image(self.pygor_object.rois_alt, name = "ROIs", colormap = "rainbow", opacity = 0.25)
        self.viewer.add_points(self.pygor_object.roi_centroids, name = "ROI centroids", opacity = 1, face_color="orange", size = 1.5)
        upper_layer = self.viewer.add_shapes(name = "100% boundary", edge_color = "red")
        lower_layer = self.viewer.add_shapes(name = "0% boundary", edge_color = "blue")
        # Set tool so its ready-to-go for clicking
        upper_layer.mode = 'add_polyline'
        lower_layer.mode = 'add_polyline'
        self.viewer.layers.selection.events.active.connect(self.on_layer_switch)
        self.last_active_layer = self.viewer.layers[-1] #last added layer is always active first, and therefore the last_active_layer on firs layer switch
        napari.run()
        # Set the exception hook globally
        # sys.excepthook = self.handle_exception
        self.event_loop.exec_()  # Block until close event triggers
        return self.result  # Now `self.result` is updated before returning

class NapariRoiPrompt():
    def __init__(self, array_input, traces_plot_style = "individual", plot = False):
        if get_ipython() is not None:
            get_ipython().run_line_magic('matplotlib', 'Qt5Agg')
        import napari
        self.traces_plot_style = traces_plot_style
        self.roi_coordinates = None  # Store the computed value
        self.mask = np.zeros(array_input[0].shape)
        self.viewer = napari.Viewer()
        self.arr = array_input
        self.plot = plot
        # Create a Qt event loop
        self.event_loop = QEventLoop()
        self.make_fig = True
        self.already_plotted = []

        @self.viewer.bind_key("c")
        def plot_on_demand(viewer):
            self.grab_coordinates()
            self.generate_plot()

        @self.viewer.bind_key("ctrl+k")
        def clear_all_rois(viewer):
            self.update_self()
            if len(self.viewer.layers["place ROIs"].data) > 0:
                self.viewer.layers["place ROIs"].data = []

        @self.viewer.bind_key("k")
        def clear_prev_roi(viewer):
            self.update_self()
            if len(self.viewer.layers["place ROIs"].data) > 0:
                if self.traces_plot_style == "stacked":
                    if self.traces is not None:
                        plt.gcf().axes[1].lines[-1].remove()
                        plt.gcf().axes[0].collections[-1].remove()

                self.viewer.layers["place ROIs"].data = self.viewer.layers["place ROIs"].data[:-1]

        # Override the close event
        original_close_event = self.viewer.window._qt_window.closeEvent

        def custom_close_event(event):
            self.on_close()  # Run computations on close
            original_close_event(event)  # Ensure proper closing
            self.event_loop.quit()  # Exit the event loop

        self.viewer.window._qt_window.closeEvent = custom_close_event

    def grab_coordinates(self):
        """Retrieve ROI coordinates, ensuring the latest data is captured."""
        print("Processing user selection...")
        # roi_layer.refresh()  # Force update before reading data
        if self.viewer.layers["place ROIs"].data is not None:
            # coords = self.viewer.layers["place ROIs"].data
            # coords = [[np.clip(x, (0, self.arr[0].shape[1])), np.clip(y, 0, (self.arr[0].shape[0]))] for x, y in zip(coords[:, 0], coords[:, 1])]
            # self.roi_coordinates = coords
            """
            TODO: Fix coordinates landing outside of the image. Its just annoying for keeping ROI count. 
            Just delete them.
            """
            coordinates = self.viewer.layers["place ROIs"].data
            shape = self.arr[0].shape
            coordinates = [np.dstack([np.clip(curr_coord[:, 0], 0 , shape[0]), np.clip(curr_coord[:, 1], 0 , shape[1])]) for curr_coord in coordinates]

            self.roi_coordinates = coordinates
            print(coordinates)
            # self.viewer.layers["place ROIs"].data = coordinates
            
            # self.roi_coordinates = self.viewer.layers["place ROIs"].data
            

    def on_close(self):
        """Function triggered when the viewer closes."""
        print("Napari GUI closed. Generating final plot...")
        self.grab_coordinates()  # Compute the final result
        if self.viewer.layers["place ROIs"].data is not None:
            self.generate_plot()

    # Methods to handle ROI output
    def mask_from_coords(self, coords_list, mask_shape):
        mask = np.ones(mask_shape) * np.nan  # Boolean mask
        for n, coords in enumerate(coords_list):
            coords = np.ceil(coords)
            # Fill the mask
            x, y = polygon(coords[:, 0], coords[:, 1], shape=mask_shape)
            mask[x, y] = n  # Fill the mask
        mask = mask[:self.arr[0].shape[0], :self.arr[0].shape[1]]
        return mask
        
    def fetch_traces(self, img_stack, roi_mask):
        unique_vals = np.unique(roi_mask)
        unique_vals = unique_vals[~np.isnan(unique_vals)]
        traces = []
        for n in unique_vals:
            mask = (roi_mask == n)
            # Efficiently compute the sum and count using einsum
            sum_per_frame = np.einsum('ijk,jk->i', img_stack, mask, optimize=True)
            count = mask.sum()
            traces.append(sum_per_frame / count)
        return np.array(traces)

    def get_or_create_figure(self):
        """Return the specific labeled figure, or create it if missing."""
        FIGURE_LABEL = "Plot selected ROIs"
        # Check all existing figures by iterating over them
        if self.make_fig == False:
            for fig_num in plt.get_fignums():
                fig = plt.figure(fig_num)  # Get figure by number
                if fig.get_label() == FIGURE_LABEL:  # Check label
                    print("Reusing figure")
                    # plt.show(block = False)
                    plt.pause(.1)
                    return fig, fig.axes
        # Create a new figure if not found
        print("Generating new plot")
        fig, ax = plt.subplots(2, 1, figsize=(10,4), label = FIGURE_LABEL)
        self.make_fig = False
        plt.show(block=True)
        return fig, ax

    def update_self(self):
        self.roi_coordinates = self.viewer.layers["place ROIs"].data
        self.mask = self.mask_from_coords(self.roi_coordinates, self.arr[0].shape)
        self.traces = self.fetch_traces(self.arr, self.mask)

    def generate_plot(self):
        self.update_self()
        fig, ax = self.get_or_create_figure()
        colormap = plt.cm.rainbow(np.linspace(0, 1, len(self.traces)))
        ax[0].imshow(np.average(self.arr, axis = 0), cmap = "Greys_r")
        ax[0].imshow(self.mask, cmap = "rainbow", alpha = 0.25)
        for n, coords in enumerate(self.roi_coordinates):
                ax[0].scatter(coords[:, 1], coords[:, 0], s = 1)
                ax[0].text(np.average(coords[:, 1]), np.average(coords[:, 0]), str(n), fontsize=16, ha="center", va="center", color="black")
                ax[0].text(np.average(coords[:, 1]), np.average(coords[:, 0]), str(n), fontsize=14, ha="center", va="center", color="white")
        if self.traces_plot_style == "stacked":
            for n, i in enumerate(self.traces):
                ax[1].plot(i, color = colormap[-n], label = f"ROI {n}")
                ax[1].legend()
        if self.traces_plot_style == "individual":
            # Get the figure and number of traces
            fig = ax[0].figure
            num_traces = len(self.traces)
            # Clear existing axes except ax[0]
            if len(self.traces) < 5:
                for a in fig.axes[1:]:  # Keep ax[0], remove others
                    a.remove()
            else:
                for a in fig.axes:
                    a.remove()
            # Create new gridspec: 1 main plot (ax[0]) + n trace plots
            gs = fig.add_gridspec(num_traces + 1, 1)
            # Reassign ax[0] to the top of the new grid
            ax[0].set_subplotspec(gs[0, 0])
            # Create new axes for traces below ax[0]
            trace_axes = []
            for n in range(num_traces):
                new_ax = fig.add_subplot(gs[n+1, 0])
                trace_axes.append(new_ax)
            # Plot traces in new axes
            for n, (trace, ax_trace) in enumerate(zip(self.traces, trace_axes)):
                ax_trace.plot(trace, color=colormap[-n], label=f"ROI {n}")
                ax_trace.spines[["top", "bottom", "right"]].set_visible(False)
                ax_trace.set_yticks([])
                ax_trace.set_ylabel(f"ROI {n}", rotation=90)
            fig.subplots_adjust(hspace=0, wspace=0)
        if self.traces_plot_style == "raster":
            if len(self.traces) > 0:
                ax[1].imshow(self.traces, aspect = "auto", cmap = "Greys_r", interpolation="none")
        plt.draw()
        
    def run(self):
        import napari
        """Launch Napari and block execution properly."""
        vmin, vmax = np.percentile(self.arr, (1, 99))
        self.viewer.add_image(np.std(self.arr, axis = 0), name="SD")
        self.viewer.add_image(np.mean(self.arr, axis = 0), name="Average")
#        try:
#            self.viewer.add_image(self.calculate_image_average(), name="Average movie", opacity=.6, contrast_limits=(vmin, vmax))
#        except ValueError:
#            pass
        self.viewer.add_image(self.arr, name="Image", opacity=.6, contrast_limits=(vmin, vmax))
        roi_layer = self.viewer.add_shapes(name = "place ROIs", shape_type = 'polygon', opacity=.75, edge_width=.25, edge_color='yellow', face_color='transparent')
        roi_layer.mode = 'add_ellipse'
        self.viewer.layers[0]._keep_auto_contrast = True
        napari.run()
        self.event_loop.exec_()  # Block until close event triggers
        if get_ipython() is not None:
            get_ipython().run_line_magic('matplotlib', 'inline')
        if self.viewer.layers["place ROIs"].data != []:
            plt.show(block = False)
        else:
            plt.close()
        return self.fetch_traces(self.arr, self.mask)  # Now `self.result` is updated before returning