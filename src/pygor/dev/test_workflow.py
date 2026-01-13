import pathlib
import numpy as np
import napari
import matplotlib.pyplot as plt
from skimage.registration import phase_cross_correlation
from scipy.ndimage import shift
from pygor.classes.core_data import Core
import timeit

"""
Simple batch-averaged registration for calcium imaging.
"""

def main():
    
    # Load data
    example_path = pathlib.Path(r"D:\Igor analyses\OSDS\251112 OSDS\1_0_SWN_200_White.smp")
    print("Loading data...")
    data = Core.from_scanm(example_path)
    data.preprocess(detrend=False)
    image_stack = data.images#[:, :, 2:]  # (time, height, width)
    
    # Parameters
    n_reference_frames = 1000  # More frames for stable reference
    batch_size = 10     # ~64 stimulus cycles at 5Hz with 15.625Hz imaging
    upsample_factor = 10      # Lower for speed
    # Run registration (preprocessing already handled artifact removal)
    reference = np.mean(image_stack[:n_reference_frames], axis=0)
    print("Running registration...")
    time_start = timeit.default_timer()
    stats = data.register(
        n_reference_frames=n_reference_frames,
        batch_size=batch_size,
        upsample_factor=upsample_factor,
        order = 2, 
        plot = True,
        parallel=True,
        n_jobs = -1,
    )

    data = Core.from_scanm(example_path)
    data.preprocess(detrend=False)
    image_stack = data.images#[:, :, 2:]  # (time, height, width)
    stats = data.register(
        n_reference_frames=n_reference_frames,
        batch_size=batch_size,
        upsample_factor=upsample_factor,
        order = 2, 
        force=True,
        plot = True,
        parallel=False,
    )



if __name__ == "__main__":
    main()
