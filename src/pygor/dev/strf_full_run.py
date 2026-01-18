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

obj.segment_rois(mode="cellpose+", overwrite=True, verbose=True, model_path = r"models\synaptic\cellpose_rois")

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

obj.plot_strfs_space()
plt.show()