# %%
from pygor.load import OSDS, STRF
import numpy as np
import napari 
import h5py
import matplotlib.pyplot as plt
# %%
# Load up data
load_path_osds = r"D:\Igor analyses\OSDS\251104 OSDS\0_1_gradient_contrast_400_white.smh"
load_path_strf = r"D:\Igor analyses\OSDS\251104 OSDS\0_1_SWN_200_White.smp"
# load_path = r"D:\Igor analyses\SWN BC main\231018\1_0_SWN_200_Colour_0.smp"
data_ref = STRF(load_path_strf) 
data_dir = OSDS(load_path_osds, dir_num = 8, dir_phase_num = 2, trigger_mode = 8) 

# %%
# Preprocessing on reference data
data_ref.preprocess(detrend = False)
data_ref.register(plot = True)
data_ref.segment_rois(plot=True, input_mode = "average")

# Preprocess and register direcitonal data
#%%
data_dir = OSDS(load_path_osds, dir_num = 8, dir_phase_num = 2) 
data_dir.preprocess(detrend = False)
data_dir.register(ref_plane=data_ref.average_stack, plot=True)


# Transfer ROIs from reference to directional data
data_dir.transfer_rois_from(data_ref, plot=True)

#%%
# Estimate mean image correspondance 
img1 = data_ref.images.mean(axis=0)
img2 = data_dir.images.mean(axis=0)
correlation_pearsons = np.corrcoef(img1.flatten(), img2.flatten())[0, 1]
print(f"Pearson correlation between mean images: {correlation_pearsons:.4f}")
#%%
# Extract traces in both datasets
data_ref.extract_traces_from_rois()
data_dir.extract_traces_from_rois()
#%%
# Compute STRFs
NOISE_PATH = r"C:\Users\SimenLab\OneDrive - University of Sussex\Desktop\jitternoise_SINGLEcolour_100000x6x10_0.25_1.hdf5"
# load up noise array
with h5py.File(NOISE_PATH, "r") as f:
    print(list(f.keys()))
    noise_array = np.array(f["noise_jitter"][:30000])  # Use subset for speed
data_ref.calculate_strf(noise_array=noise_array, n_jobs = -1)
#%%
# Compute direction tuning
data_dir.trigger_mode = 8
data_dir.compute_snippets_and_averages()
data_dir.plot_averages()
#%%
# for i in range(data_ref.num_rois):
#     fig, ax = plt.subplots(1, 2, figsize = (15, 5))
#     # plot direction tuning for ROI in directional data
#     data_dir.plot_tuning_function_with_traces(i, ax = ax[0])
#     # plot STRF for ROI in reference data
#     rfim = ax[1].imshow(data_ref.collapse_times(i)[0], cmap = "bwr", vmin = -np.max(np.abs(data_ref.collapse_times(i)[0])), vmax = np.max(np.abs(data_ref.collapse_times(i)[0])))
#     plt.colorbar(rfim, ax=ax[1])
#     plt.show()
#%%
roi = 3 #nr 2 is sick
fig, ax = plt.subplots(1, 2, figsize = (15, 5))
# plot direction tuning for ROI in directional data
data_dir.plot_tuning_function_with_traces(roi, ax = ax[0])
# plot STRF for ROI in reference data
rfim = ax[1].imshow(data_ref.collapse_times(roi)[0], cmap = "bwr", vmin = -np.max(np.abs(data_ref.collapse_times(roi)[0])), vmax = np.max(np.abs(data_ref.collapse_times(roi)[0])))
plt.colorbar(rfim, ax=ax[1])
data_ref.play_strf(roi, dur_s = 4)
#%%
# viewer = napari.Viewer()
# viewer.add_image(data_dir.images, name = "Reference mean")