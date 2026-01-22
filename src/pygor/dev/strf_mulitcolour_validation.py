#%%

import pathlib
import pygor.load
import matplotlib.pyplot as plt
import h5py
import numpy as np
import napari
# from scipy.stats import pearsonr
"""
Test:
Place ROIs and preprocess in pygor, export to H5,
re-run inside IGOR, import back into pygor,
and compare pygor calculations to IGOR calculations.

Result:
obj.strfs shape: (252, 20, 24, 40)
obj_igor.strfs shape: (252, 20, 24, 40)

obj.strfs[3] shape: (20, 24, 40)
obj_igor.strfs[3] shape: (20, 24, 40)
ROI 3 STRF correlation: 1.0000

All ROIs correlation: 0.9993 ± 0.0015
Min: 0.9926, Max: 1.0000
"""


#%%
# Path to example data - update this to your file
EXAMPLE_PATH = pathlib.Path(r"D:\Igor analyses\SWN BC main\240517 ctrl data\0_0_SWN_200_5hz_RGBUV.smp")
# Path to a custom config (create one or use the example)
CUSTOM_CONFIG = r"configs\example.toml"
#%%

obj = pygor.load.STRF(EXAMPLE_PATH, config=CUSTOM_CONFIG)


obj.preprocess(detrend=False, artifact_width  = 4)
obj.register(batch_size=10, upsample_factor=5, n_reference_frames=10000, force = True)


#%%
obj.segment_rois(model_path = r"C:\Users\SimenLab\Git_repos\pygor\models\synaptic\cellpose_rois")
obj.extract_traces_from_rois

#%%

noise_arr_path = r"C:\Users\SimenLab\OneDrive - University of Sussex\Desktop\jitternoise_SINGLEcolour_100000x6x10_0.25_1.hdf5"
with h5py.File(noise_arr_path, "r") as f:
    print(list(f.keys()))
    noise_array = np.array(f["noise_jitter"])[:, :, :10000]
print(f"Loaded noise array shape: {noise_array.shape}")
#%%

out = obj.calculate_strf(noise_array=noise_array, n_colours = 4,
        sta_past_window = 1.3, sta_future_window = 0, n_triggers_per_colour = 100,
        verbose=True, normalize_strfs=True, n_jobs=-1)
out_path_validation = pathlib.Path(r"D:\Igor analyses\SWN BC main\240517 ctrl data")
#%% Export for IGOR validation
print("Export to H5 and run inside IGOR, then re-import to compare.")
# Uncomment below to export, then process in IGOR and re-import to compare
# obj.export_to_h5(out_path_validation / "pygor_for_igor_validation.h5", overwrite=True)
# print("Exported! Now run STRF calculation in IGOR with sta_past_window=1.3, then run cells below.")

#%%
# After running in IGOR, import the IGOR-processed HDF5
path = r"D:\Igor analyses\SWN BC main\240517 ctrl data\2024-05-17_send_out_hdf5data.h5"
obj_igor = pygor.load.STRF(path)


#%%
roi = -7
print("Plotting pygor result")
obj.plot_chromatic_overview(roi=roi)
print("Plotting IGOR result")
obj_igor.plot_chromatic_overview(roi=roi)
# %%

# # viewer.add_labels(obj_igor.rois, name="igor_rois")
# %%
roi = 3

# Check the shapes
print(f"obj.strfs shape: {obj.strfs.shape}")
print(f"obj_igor.strfs shape: {obj_igor.strfs.shape}")

# Break down the dimensions
# STRF is typically (n_rois, n_colours, time, height, width) or similar
print(f"\nobj.strfs[{roi}] shape: {obj.strfs[roi].shape}")
print(f"obj_igor.strfs[{roi}] shape: {obj_igor.strfs[roi].shape}")

corr = np.corrcoef(obj.strfs[roi].flatten(), obj_igor.strfs[roi].flatten())[0, 1]
print(f"ROI {roi} STRF correlation: {corr:.4f}")

# Check across all ROIs
all_corrs = []
for r in range(obj.strfs.shape[0]):
    c = np.corrcoef(obj.strfs[r].flatten(), obj_igor.strfs[r].flatten())[0, 1]
    all_corrs.append(c)
all_corrs = np.array(all_corrs)

print(f"\nAll ROIs correlation: {np.nanmean(all_corrs):.4f} ± {np.nanstd(all_corrs):.4f}")
print(f"Min: {np.nanmin(all_corrs):.4f}, Max: {np.nanmax(all_corrs):.4f}")
