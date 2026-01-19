#%%
"""
STRF Validation Script: Pygor vs IGOR Comparison (Single Colour)
================================================================

Workflow for 1:1 ROI comparison:
1. Start with an SMP file in pygor
2. Preprocess, register, segment ROIs
3. Export to H5: obj.export_to_h5("path/to/output.h5")
4. Import this H5 into IGOR Pro
5. Run STRF calculation in IGOR with same parameters (sta_past_window=2)
6. Export from IGOR back to H5
7. Load that H5 here and compare - both use SAME ROIs

Result (expected):
obj.strfs shape: (n_rois, time, y, x)
obj_igor.strfs shape: (n_rois, time, y, x)
All ROIs correlation: ~0.99+ (accounting for float precision)
"""

import pathlib
import pygor.load
import matplotlib.pyplot as plt
import h5py
import numpy as np

#%%
# Path to example data - update this to your file
EXAMPLE_PATH = pathlib.Path(r"D:\Igor analyses\OSDS\251103 OSDS\2_0_SWN_200_White.smp")
CUSTOM_CONFIG = r"configs\example.toml"

#%%
obj = pygor.load.STRF(EXAMPLE_PATH, config=CUSTOM_CONFIG)
obj.preprocess(detrend=False)
obj.register()

#%%
obj.segment_rois(model_path=r"C:\Users\SimenLab\Git_repos\pygor\models\synaptic\cellpose_rois",
    diameter=None, split_large=True, cellprob_threshold=2)
obj.compute_traces_from_rois()

#%%
noise_arr_path = r"C:\Users\SimenLab\OneDrive - University of Sussex\Desktop\jitternoise_SINGLEcolour_100000x6x10_0.25_1.hdf5"
with h5py.File(noise_arr_path, "r") as f:
    print(list(f.keys()))
    noise_array = np.array(f["noise_jitter"])
print(f"Loaded noise array shape: {noise_array.shape}")

#%%
out = obj.calculate_strf(noise_array=noise_array,
        sta_past_window=2, sta_future_window=0,
        verbose=True, normalize_strfs=True, n_jobs=-1)

#%% Export for IGOR validation
out_path_validation = pathlib.Path(r"D:\Igor analyses\OSDS\251103 OSDS")
print("Export to H5 and run inside IGOR, then re-import to compare.")
# Uncomment below to export, then process in IGOR and re-import to compare
obj.export_to_h5(out_path_validation / "pygor_for_igor_validation.h5", overwrite=True)
# print("Exported! Now run STRF calculation in IGOR with sta_past_window=2, then run cells below.")

#%%
# After running in IGOR, import the IGOR-processed HDF5
igor_path = r"D:\Igor analyses\OSDS\251103 OSDS\process_and_back_into_pygor.h5"
obj_igor = pygor.load.STRF(igor_path)

# #%%
# # Visual comparison for a single ROI

# print("Plotting pygor result")
# obj.play_strf(roi, dur_s=2)
# print("Plotting IGOR result")
# obj_igor.play_strf(roi, dur_s=2)

#%%
roi = 0
# Check the shapes
print(f"obj.strfs shape: {obj.strfs.shape}")
print(f"obj_igor.strfs shape: {obj_igor.strfs.shape}")

# Break down the dimensions
print(f"\nobj.strfs[{roi}] shape: {obj.strfs[roi].shape}")
print(f"obj_igor.strfs[{roi}] shape: {obj_igor.strfs[roi].shape}")

corr = np.corrcoef(obj.strfs[roi].flatten(), obj_igor.strfs[roi].flatten())[0, 1]
print(f"ROI {roi} STRF correlation: {corr:.4f}")

#%%
# Check across all comparable ROIs
n_rois_pygor = obj.strfs.shape[0]
n_rois_igor = obj_igor.strfs.shape[0]
n_compare = min(n_rois_pygor, n_rois_igor)

if n_rois_pygor != n_rois_igor:
    print(f"\nWARNING: ROI count mismatch!")
    print(f"  Pygor: {n_rois_pygor} ROIs")
    print(f"  IGOR: {n_rois_igor} ROIs")
    print(f"  Comparing first {n_compare} ROIs only")
    # Dont run, prompt user to recalculate on IGOR side
    raise ValueError("ROI count mismatch between pygor and IGOR results.")
else:
    print(f"\nROI count match: {n_rois_pygor} ROIs")
#%%

all_corrs = []
for r in range(n_compare):
    # Check shape compatibility
    if obj.strfs[r].shape != obj_igor.strfs[r].shape:
        print(f"ROI {r}: Shape mismatch - pygor {obj.strfs[r].shape} vs igor {obj_igor.strfs[r].shape}")
        all_corrs.append(np.nan)
        continue
    c = np.corrcoef(obj.strfs[r].flatten(), obj_igor.strfs[r].flatten())[0, 1]
    all_corrs.append(c)
all_corrs = np.array(all_corrs)

print(f"\nAll ROIs correlation: {np.nanmean(all_corrs):.4f} ± {np.nanstd(all_corrs):.4f}")
print(f"Min: {np.nanmin(all_corrs):.4f}, Max: {np.nanmax(all_corrs):.4f}")

#%%
# Detailed visual comparison for a single ROI
roi = 10

pygor_data = obj.strfs[roi]
igor_data = obj_igor.strfs[roi]

# Find max frame for each
pygor_max_frame = np.unravel_index(np.abs(pygor_data).argmax(), pygor_data.shape)[0]
igor_max_frame = np.unravel_index(np.abs(igor_data).argmax(), igor_data.shape)[0]

fig, ax = plt.subplots(2, 2, figsize=(10, 5))

# Receptive field comparison
im0 = ax[0, 0].imshow(pygor_data[pygor_max_frame], cmap='bwr', clim=(-10, 10))
fig.colorbar(im0, ax=ax[0, 0])
ax[0, 0].set_title(f'Pygor STRF ROI {roi} (frame {pygor_max_frame})')

im1 = ax[1, 0].imshow(igor_data[igor_max_frame], cmap='bwr', clim=(-10, 10))
fig.colorbar(im1, ax=ax[1, 0])
ax[1, 0].set_title(f'IGOR STRF ROI {roi} (frame {igor_max_frame})')
    
# Histograms
pygor_nonzero = pygor_data.flatten()[pygor_data.flatten() != 0]
igor_nonzero = igor_data.flatten()[igor_data.flatten() != 0]

ax[0, 1].hist(pygor_nonzero, bins=50)
ax[0, 1].set_yscale('log')
ax[0, 1].set_title('Pygor histogram')

ax[1, 1].hist(igor_nonzero, bins=50)
ax[1, 1].set_yscale('log')
ax[1, 1].set_title('IGOR histogram')

# Match histogram ranges
# if len(pygor_nonzero) > 0 and len(igor_nonzero) > 0:
#     combined_std = np.std(np.concatenate([pygor_nonzero, igor_nonzero]))
#     for a in ax[:, 1]:
#         a.set_xlim(-10 * combined_std, 10 * combined_std)

plt.tight_layout()
plt.show()

# Print statistics
print(f"\nStatistics comparison (ROI {roi}):")
print(f"Pygor - min: {pygor_nonzero.min():.3f}, max: {pygor_nonzero.max():.3f}, std: {pygor_nonzero.std():.3f}")
print(f"IGOR  - min: {igor_nonzero.min():.3f}, max: {igor_nonzero.max():.3f}, std: {igor_nonzero.std():.3f}")
print(f"Scale ratio (IGOR/pygor std): {igor_nonzero.std() / pygor_nonzero.std():.3f}")

# %%
