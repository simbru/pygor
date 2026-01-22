#%%
import pathlib
import pygor.load
import h5py
import numpy as np
from scipy.stats import pearsonr
import time

# Import both correlation functions directly
from pygor.strf.calculate_strf import (
    means_subtracted_correlation,
    means_subtracted_correlation_fft
)

#%% Load and preprocess data
EXAMPLE_PATH = pathlib.Path(r"D:\Igor analyses\OSDS\251103 OSDS\2_0_SWN_200_White.smp")
CUSTOM_CONFIG = r"configs\example.toml"
load_h5 = r"D:\Igor analyses\OSDS\251103 OSDS\validation_2025-11-3_SMP_2_0_SWN_200_White.h5"
obj = pygor.load.STRF(EXAMPLE_PATH, config=CUSTOM_CONFIG)
obj.preprocess(detrend=False)
obj.register()
obj_h5 = pygor.load.STRF(load_h5)
obj.rois = obj_h5.rois  # Ensure same ROIs for fair comparison

obj.extract_traces_from_rois

# Load noise array
noise_arr_path = r"C:\Users\SimenLab\OneDrive - University of Sussex\Desktop\jitternoise_SINGLEcolour_100000x6x10_0.25_1.hdf5"
with h5py.File(noise_arr_path, "r") as f:
    noise_array = np.array(f["noise_jitter"])

print(f"Raw noise_array shape: {noise_array.shape}")

#%% Prepare test data - extract a single trace and noise stimulus chunk
# Use first ROI's trace as test signal
f_signal = obj.traces_raw[0].astype(np.float64)

# NOTE: In actual STRF calculation, noise_stimulus has shape (n_x_cropped, n_y_cropped, n_f_relevant)
# where n_f_relevant is determined by trigger timings (typically a few thousand frames).
# This test uses the full noise array which may not reflect typical STRF calculation sizes.

# Flatten noise spatially: assume shape is (n_y, n_x, n_frames) or (n_frames, n_y, n_x)
# The filename suggests 100000x6x10 = (frames, y, x), so transpose if needed
if noise_array.shape[0] > noise_array.shape[2]:
    # Shape is (frames, y, x) - transpose to (y, x, frames)
    noise_array = np.transpose(noise_array, (1, 2, 0))
    print(f"Transposed to: {noise_array.shape}")

n_y, n_x, n_z = noise_array.shape
noise_2d = noise_array.reshape(n_y * n_x, n_z)

print(f"f_signal shape: {f_signal.shape}")
print(f"noise_2d shape: {noise_2d.shape}")

#%% Time and compare both methods
n_runs = 3

# Loop-based method
times_loop = []
for _ in range(n_runs):
    t0 = time.perf_counter()
    result_loop = means_subtracted_correlation(f_signal, noise_2d)
    times_loop.append(time.perf_counter() - t0)

# FFT-based method
times_fft = []
for _ in range(n_runs):
    t0 = time.perf_counter()
    result_fft = means_subtracted_correlation_fft(f_signal, noise_2d)
    times_fft.append(time.perf_counter() - t0)

print(f"\n--- Timing (mean of {n_runs} runs) ---")
print(f"Loop method: {np.mean(times_loop):.3f}s")
print(f"FFT method:  {np.mean(times_fft):.3f}s")
print(f"Speedup:     {np.mean(times_loop) / np.mean(times_fft):.1f}x")

#%% Compare outputs
print("\n--- Output comparison ---")
print(f"Loop result shape: {result_loop.shape}")
print(f"FFT result shape:  {result_fft.shape}")

# Trim to same length for comparison
min_len = min(result_loop.shape[1], result_fft.shape[1])
loop_trimmed = result_loop[:, :min_len]
fft_trimmed = result_fft[:, :min_len]

# Stats
corr, _ = pearsonr(loop_trimmed.flatten(), fft_trimmed.flatten())
max_diff = np.max(np.abs(loop_trimmed - fft_trimmed))
mean_diff = np.mean(np.abs(loop_trimmed - fft_trimmed))

print(f"Correlation: {corr:.10f}")
print(f"Max abs diff: {max_diff:.2e}")
print(f"Mean abs diff: {mean_diff:.2e}")

#%% Plot max frame of STRFs side by side
import matplotlib.pyplot as plt

# Reshape correlation results back to spatial dimensions for visualization
# result shape is (n_pixels, correlation_length) where n_pixels = n_y * n_x
loop_spatial = loop_trimmed.reshape(n_y, n_x, -1)
fft_spatial = fft_trimmed.reshape(n_y, n_x, -1)

# Find max frame (frame with highest absolute value across all pixels)
loop_abs_sum = np.sum(np.abs(loop_spatial), axis=(0, 1))
fft_abs_sum = np.sum(np.abs(fft_spatial), axis=(0, 1))
max_frame_loop = np.argmax(loop_abs_sum)
max_frame_fft = np.argmax(fft_abs_sum)

fig, axes = plt.subplots(1, 3, figsize=(12, 4))

# Loop method max frame
im0 = axes[0].imshow(loop_spatial[:, :, max_frame_loop], cmap='RdBu_r', aspect='auto')
axes[0].set_title(f'Loop method (frame {max_frame_loop})')
plt.colorbar(im0, ax=axes[0])

# FFT method max frame
im1 = axes[1].imshow(fft_spatial[:, :, max_frame_fft], cmap='RdBu_r', aspect='auto')
axes[1].set_title(f'FFT method (frame {max_frame_fft})')
plt.colorbar(im1, ax=axes[1])

# Difference (using same frame for fair comparison)
diff = loop_spatial[:, :, max_frame_loop] - fft_spatial[:, :, max_frame_loop]
im2 = axes[2].imshow(diff, cmap='RdBu_r', aspect='auto')
axes[2].set_title(f'Difference (max={np.max(np.abs(diff)):.2e})')
plt.colorbar(im2, ax=axes[2])

plt.tight_layout()
plt.show()

# %%
