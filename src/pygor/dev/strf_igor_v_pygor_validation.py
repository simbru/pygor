#%%
import pathlib
import pygor.load
import matplotlib.pyplot as plt
import h5py
import numpy as np
from scipy.stats import pearsonr


# Path to example data - update this to your file
# EXAMPLE_PATH = pathlib.Path(r"D:\Igor analyses\OSDS\251105 OSDS\1_0_SWN_200_White.smp")
EXAMPLE_PATH = pathlib.Path(r"D:\Igor analyses\OSDS\251103 OSDS\2_0_SWN_200_White.smp")
# Path to a custom config (create one or use the example)
CUSTOM_CONFIG = r"configs\example.toml"
obj = pygor.load.STRF(EXAMPLE_PATH, config=CUSTOM_CONFIG)

obj = pygor.load.STRF(EXAMPLE_PATH, config=CUSTOM_CONFIG) # old
obj.preprocess(detrend=False)
obj.register()
#%%
# load_path = r"D:\Igor analyses\OSDS\251103 OSDS\2025-11-3_SMP_2_0_SWN_200_White.h5"
load_path = r"D:\Igor analyses\OSDS\251103 OSDS\validation_2025-11-3_SMP_2_0_SWN_200_White.h5"

obj2 = pygor.load.STRF(load_path)
obj.rois = obj2.rois

obj.compute_traces_from_rois()

noise_arr_path = r"C:\Users\SimenLab\OneDrive - University of Sussex\Desktop\jitternoise_SINGLEcolour_100000x6x10_0.25_1.hdf5"
with h5py.File(noise_arr_path, "r") as f:
    print(list(f.keys()))
    noise_array = np.array(f["noise_jitter"])
# print(f"Loaded noise array shape: {noise_array.shape}")
# exit()

out = obj.calculate_strf(noise_array=noise_array, 
        sta_past_window = 2, sta_future_window = 0, 
        verbose=True, normalize_strfs=True, n_jobs=-1)

# # Check if masking is working correctly now
# print("\n--- Result Check ---")
# print(f"STRF shape: {out['strfs'].shape}")
# print(f"STRF is masked array: {isinstance(out['strfs'], np.ma.MaskedArray)}")
# if isinstance(out['strfs'], np.ma.MaskedArray):
#     print(f"Mask unique values: {np.unique(out['strfs'].mask)}")
#     print(f"Mask all True: {np.all(out['strfs'].mask)}")
#     print(f"Non-masked data count: {np.sum(~out['strfs'].mask)}")
# else:
#     print("Not a masked array - checking raw values")
#     print(f"Has NaN: {np.any(np.isnan(out['strfs']))}")
#     print(f"Min: {np.nanmin(out['strfs'])}")
#     print(f"Max: {np.nanmax(out['strfs'])}")
#%%
roi = 12

# Need to extract correctly for comparison

# Get pygor calculated STRF for this ROI (from calculation result, not obj.strfs!)
pygor_strf = out['strfs'][0, roi]

# Get IGOR reference STRF for this ROI
igor_strf = obj2.strfs[roi] 

print(f"\n--- Shape Comparison ---")
print(f"Pygor STRF shape: {pygor_strf.shape}")
print(f"IGOR STRF shape: {igor_strf.shape}")

# Handle masked arrays for finding max frame
pygor_data = pygor_strf.data if isinstance(pygor_strf, np.ma.MaskedArray) else pygor_strf
igor_data = igor_strf.data if isinstance(igor_strf, np.ma.MaskedArray) else igor_strf

# find most extreme frame (not index)
pygor_max_frame = np.unravel_index(np.abs(pygor_data).argmax(), pygor_data.shape)[0]
igor_max_frame = np.unravel_index(np.abs(igor_data).argmax(), igor_data.shape)[0]

print(f"Pygor max frame: {pygor_max_frame}")
print(f"IGOR max frame: {igor_max_frame}")

fig, ax = plt.subplots(2, 2, figsize=(10, 10))

# receptive field - pygor (from out['strfs'])
ax[0, 0].imshow(pygor_strf[pygor_max_frame], cmap='bwr',
                vmin=-np.max(np.abs(pygor_data)), vmax=np.max(np.abs(pygor_data)))
ax[0, 0].set_title(f'pygor STRF ROI {roi} max frame {pygor_max_frame}')

# receptive field - IGOR (from obj2.strfs)
ax[1, 0].imshow(igor_strf[igor_max_frame], cmap='bwr',
                vmin=-np.max(np.abs(igor_data)), vmax=np.max(np.abs(igor_data)))
ax[1, 0].set_title(f'IGOR STRF ROI {roi} max frame {igor_max_frame}')

# histograms - ignore masked/zero values
pygor_flat = pygor_data.flatten()
pygor_nonzero = pygor_flat[pygor_flat != 0]
ax[0, 1].hist(pygor_nonzero, bins=50)
ax[0, 1].set_title(f'pygor STRF ROI {roi} histogram')

igor_flat = igor_data.flatten()
igor_nonzero = igor_flat[igor_flat != 0]
ax[1, 1].hist(igor_nonzero, bins=50)
ax[1, 1].set_title(f'IGOR STRF ROI {roi} histogram')

# adjust range based on combined data
combined_std = np.std(np.concatenate([pygor_nonzero, igor_nonzero]))
for a in ax[:, 1]:
    a.set_xlim(-10 * combined_std, 10 * combined_std)

# Print statistics for comparison
print(f"\n--- Statistics Comparison ---")
print(f"Pygor - min: {pygor_nonzero.min():.3f}, max: {pygor_nonzero.max():.3f}, std: {pygor_nonzero.std():.3f}")
print(f"IGOR  - min: {igor_nonzero.min():.3f}, max: {igor_nonzero.max():.3f}, std: {igor_nonzero.std():.3f}")
print(f"Scale ratio (IGOR/pygor std): {igor_nonzero.std() / pygor_nonzero.std():.3f}")

# Add to validation script
corr, _ = pearsonr(pygor_data.flatten(), igor_data.flatten())
print(f"Pixel-wise correlation: {corr:.6f}")


plt.tight_layout()
plt.show()

#%% Compare all ROIs
n_rois = out['strfs'].shape[1]
print(f"\n{'='*60}")
print(f"COMPARING ALL {n_rois} ROIs")
print(f"{'='*60}\n")

# Storage for statistics
correlations = []
scale_ratios = []
max_frame_matches = []

for roi in range(n_rois):
    # Get pygor calculated STRF for this ROI
    pygor_strf = out['strfs'][0, roi]

    # Get IGOR reference STRF for this ROI
    igor_strf = obj2.strfs[roi]

    # Handle masked arrays
    pygor_data = pygor_strf.data if isinstance(pygor_strf, np.ma.MaskedArray) else pygor_strf
    igor_data = igor_strf.data if isinstance(igor_strf, np.ma.MaskedArray) else igor_strf

    # Check shapes match
    if pygor_data.shape != igor_data.shape:
        print(f"ROI {roi}: SHAPE MISMATCH - pygor {pygor_data.shape} vs igor {igor_data.shape}")
        correlations.append(np.nan)
        scale_ratios.append(np.nan)
        max_frame_matches.append(False)
        continue

    # Find max frames
    pygor_max_frame = np.unravel_index(np.abs(pygor_data).argmax(), pygor_data.shape)[0]
    igor_max_frame = np.unravel_index(np.abs(igor_data).argmax(), igor_data.shape)[0]
    max_frame_match = pygor_max_frame == igor_max_frame
    max_frame_matches.append(max_frame_match)

    # Get non-zero values for std comparison
    pygor_flat = pygor_data.flatten()
    igor_flat = igor_data.flatten()
    pygor_nonzero = pygor_flat[pygor_flat != 0]
    igor_nonzero = igor_flat[igor_flat != 0]

    # Calculate scale ratio
    if len(pygor_nonzero) > 0 and len(igor_nonzero) > 0 and pygor_nonzero.std() > 0:
        scale_ratio = igor_nonzero.std() / pygor_nonzero.std()
    else:
        scale_ratio = np.nan
    scale_ratios.append(scale_ratio)

    # Calculate correlation
    corr, _ = pearsonr(pygor_flat, igor_flat)
    correlations.append(corr)

# Convert to arrays
correlations = np.array(correlations)
scale_ratios = np.array(scale_ratios)
max_frame_matches = np.array(max_frame_matches)

# Print summary statistics
print(f"{'ROI':<6} {'Correlation':>12} {'Scale Ratio':>12} {'Max Frame Match':>16}")
print("-" * 50)
for roi in range(n_rois):
    match_str = "✓" if max_frame_matches[roi] else "✗"
    print(f"{roi:<6} {correlations[roi]:>12.6f} {scale_ratios[roi]:>12.4f} {match_str:>16}")

print(f"\n{'='*60}")
print("SUMMARY STATISTICS")
print(f"{'='*60}")
print(f"Correlation:    mean={np.nanmean(correlations):.6f}, min={np.nanmin(correlations):.6f}, max={np.nanmax(correlations):.6f}")
print(f"Scale ratio:    mean={np.nanmean(scale_ratios):.4f}, min={np.nanmin(scale_ratios):.4f}, max={np.nanmax(scale_ratios):.4f}")
print(f"Max frame match: {np.sum(max_frame_matches)}/{n_rois} ({100*np.mean(max_frame_matches):.1f}%)")

# Plot summary
fig, axes = plt.subplots(1, 3, figsize=(15, 4))

# Correlation distribution
axes[0].hist(correlations[~np.isnan(correlations)], bins=20, edgecolor='black')
axes[0].axvline(np.nanmean(correlations), color='red', linestyle='--', label=f'mean={np.nanmean(correlations):.4f}')
axes[0].set_xlabel('Pixel-wise Correlation')
axes[0].set_ylabel('Count')
axes[0].set_title('Correlation Distribution (pygor vs IGOR)')
axes[0].legend()

# Scale ratio distribution
axes[1].hist(scale_ratios[~np.isnan(scale_ratios)], bins=20, edgecolor='black')
axes[1].axvline(1.0, color='green', linestyle='--', label='ideal=1.0')
axes[1].axvline(np.nanmean(scale_ratios), color='red', linestyle='--', label=f'mean={np.nanmean(scale_ratios):.4f}')
axes[1].set_xlabel('Scale Ratio (IGOR/pygor std)')
axes[1].set_ylabel('Count')
axes[1].set_title('Scale Ratio Distribution')
axes[1].legend()

# Correlation vs ROI index
axes[2].scatter(range(n_rois), correlations, c=scale_ratios, cmap='RdYlGn', vmin=0.9, vmax=1.1)
axes[2].axhline(1.0, color='green', linestyle='--', alpha=0.5)
axes[2].set_xlabel('ROI Index')
axes[2].set_ylabel('Correlation')
axes[2].set_title('Correlation by ROI (color=scale ratio)')
plt.colorbar(axes[2].collections[0], ax=axes[2], label='Scale Ratio')

plt.tight_layout()
plt.show()

# %%
