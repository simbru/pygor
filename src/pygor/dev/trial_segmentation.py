#%%
import pathlib
import os
import h5py
from skimage.filters import unsharp_mask
from pygor.load import Core, STRF
import pygor.segmentation
import pygor.segmentation.plotting
os.chdir(pathlib.Path(__file__).parent.parent.parent.parent)
import numpy as np 
import importlib
import pygor
importlib.reload(pygor.segmentation.plotting)

#%%
PATH = r"D:\Igor analyses\SWN achrom"
# list all h5 and smp files

h5_files = sorted([str(p) for p in pathlib.Path(PATH).rglob("*.h5")])
smp_files = sorted([str(p) for p in pathlib.Path(PATH).rglob("*.smp")])

# Pair up h5 files with matching smp files based on core name (without date_SMP_ prefix)
paired_files = []
unmatched_h5 = []

def get_core_name(filename):
    """Remove date_SMP_ prefix to get core name (without extension)"""
    # Remove extension first
    name_no_ext = filename.replace(".h5", "").replace(".smp", "").replace(".smh", "")

    parts = name_no_ext.split("_")

    # If there's only one part (no underscores), return the whole name
    if len(parts) == 1:
        return name_no_ext

    # Check if first part should be removed:
    # - Looks like a date (contains dash or is all digits like YYYYMMDD)
    # - OR is a non-numeric prefix (like "CTinj", "inj", "ctrl") but NOT a single digit or short number
    first_is_date = "-" in parts[0] or (parts[0].isdigit() and len(parts[0]) >= 6)
    first_is_text_prefix = not parts[0].isdigit() and len(parts[0]) > 1

    start_idx = 0
    if first_is_date or first_is_text_prefix:
        start_idx = 1
        # If next part is "SMP", skip that too
        if len(parts) > 1 and parts[1].upper() == "SMP":
            start_idx = 2

    # Return everything from start_idx onwards
    core = "_".join(parts[start_idx:])

    # If core is empty, return the whole name
    return core if core else name_no_ext

for h5_file in h5_files:
    base_name = os.path.basename(h5_file)
    core_name = get_core_name(base_name)

    # Find matching smp files
    matching_smp = [s for s in smp_files if get_core_name(os.path.basename(s)) == core_name]

    if matching_smp:
        # Take the first match if multiple exist
        paired_files.append((matching_smp[0], h5_file))
    else:
        unmatched_h5.append(h5_file)

print(f"Total H5 files: {len(h5_files)}")
print(f"Total SMP files: {len(smp_files)}")
print(f"Successfully paired: {len(paired_files)}")
print(f"Unmatched H5 files: {len(unmatched_h5)}")

# Debug: check a few unmatched files
if unmatched_h5:
    print("\nDiagnostics for first 3 unmatched H5 files:")
    for f in unmatched_h5[:3]:
        base = os.path.basename(f)
        core = get_core_name(base)
        # Find similar SMP files (same directory)
        h5_dir = os.path.dirname(f)
        smp_in_dir = [s for s in smp_files if os.path.dirname(s) == h5_dir]
        print(f"\n  H5: {base}")
        print(f"  Core name: {core}")
        print(f"  Directory: {h5_dir}")
        print(f"  SMP files in same dir: {len(smp_in_dir)}")
        if smp_in_dir:
            for smp in smp_in_dir[:2]:
                smp_core = get_core_name(os.path.basename(smp))
                print(f"    - {os.path.basename(smp)} (core: {smp_core})")

print("\nFirst 5 paired files:")
for smp, h5 in paired_files[:5]:
    print(f"  SMP: {os.path.basename(smp)}")
    print(f"  H5:  {os.path.basename(h5)}")
    print()

#%%
index = np.random.randint(0, len(paired_files))
smp_file, h5_file = paired_files[index]
print(f"Selected file pair:\n  SMP: {(smp_file)}\n  H5:  {(h5_file)}")
# #%%
# """
# Nice demo data:
# Selected file pair:
# SMP: D:\IGOR analyses\SWN single and double colour\230202\0_1_G_100_5Hz_0.smp
# H5:  D:\IGOR analyses\SWN single and double colour\230202\2023-2-2_SMP_0_1_G_100_5Hz_0.h5
# """
# smp_file = r"D:\IGOR analyses\SWN single and double colour\230202\0_1_G_100_5Hz_0.smp"
# h5_file = r"D:\IGOR analyses\SWN single and double colour\230202\2023-2-2_SMP_0_1_G_100_5Hz_0.h5"
#%%
# smp_file = r"D:\Igor analyses\SWN achrom\230331\0_1_G_100_5Hz_0.smp"
# h5_file = r"D:\Igor analyses\SWN achrom\2025-2-26_1_2_SWN_200_Achrom.h5"
# h5_file = r"D:\Igor analyses\SWN achrom\2025-2-26_1_0_SWN_200_Achrom.h5"
# h5_file = r"D:\Igor analyses\OSDS\251103 OSDS\2_0_SWN_200_White.smp"
# smp_file = r"D:\Igor analyses\OSDS\251104 OSDS\0_0_SWN_200_White.smp"
smp_file = r"D:\Igor analyses\OSDS\251103 OSDS\2_0_SWN_200_White.smp"
#%%
# preprocess, segment, and plot for each file
# for smp_file, h5_file in paired_files:
print(f"Processing file: {smp_file}")
obj = STRF(smp_file)
# objh5 = STRF(h5_file)

obj.preprocess(detrend = False)
obj.register(plot=True)
#%%
obj.compute_correlation_projection(timecompress = 5, binpix = 1)
masks = obj.segment_rois(input_mode = "combined",
                        mode = "blob", roi_order = "LR", plot = True)
                        # unsharp_radius = .5, unsharp_amount = 5)
obj.view_stack_rois()
# objh5.view_stack_rois()
#%% load up noise and calculate RFs

obj.extract_traces_from_rois()

NOISE_PATH = r"C:\Users\SimenLab\OneDrive - University of Sussex\Desktop\jitternoise_SINGLEcolour_100000x6x10_0.25_1.hdf5"

# load up noise array
with h5py.File(NOISE_PATH, "r") as f:
    print(list(f.keys()))
    noise_array = np.array(f["noise_jitter"][:30000])  # Use subset for speed

#%%
obj.calculate_strf(noise_array, n_jobs = -1, sta_future_window = 0)
#%%
obj.plot_strfs_space()



# %%
import matplotlib.pyplot as plt
plt.imshow(obj.traces_raw)
# %%
