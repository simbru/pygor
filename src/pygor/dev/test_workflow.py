import pathlib
import numpy as np
# import napari
from cellpose import models
import matplotlib.pyplot as plt
from skimage.registration import phase_cross_correlation
from scipy.ndimage import shift
from pygor.classes.core_data import Core
import timeit

"""
Simple batch-averaged registration for calcium imaging.
"""

def main():
    time_start = timeit.default_timer()

    # Load data
    example_path = pathlib.Path(r"D:\Igor analyses\OSDS\251112 OSDS\1_0_SWN_200_White.smp")
    print("Loading data...")
    data = Core.from_scanm(example_path)
    data.preprocess(detrend=False)
    image_stack = data.images#[:, :, 2:]  # (time, height, width)
    pre_register_mean = np.mean(image_stack, axis=0)
    # Parameters
    n_reference_frames = 1000  # More frames for stable reference (1000)
    batch_size = 10     # ~64 stimulus cycles at 5Hz with 15.625Hz imaging (10)
    upsample_factor = 2      # Lower for speed (10)
    # Run registration (preprocessing already handled artifact removal)
    reference = np.mean(image_stack[:n_reference_frames], axis=0)
    print("Running registration...")
    stats = data.register(
        n_reference_frames=n_reference_frames,
        batch_size=batch_size,
        upsample_factor=upsample_factor,
        order = 2,
        plot = False,
    )
    
    if stats["mean_error"] < 0.05:
        print("Registration successful.")
    else:
        print("Warning: Registration error exceeds threshold.")
        print("Exiting without saving. Adjust parameters and try again.")
        exit()

    post_register_mean = np.mean(data.images, axis=0)

    # Segment ROIs using trained Cellpose model
    print("Segmenting ROIs...")
    masks = data.segment_rois(model_dir="./models/synaptic", preview=True)

    # Visualize results with matplotlib
    fig, ax = plt.subplots(2, 2, figsize=(12, 6))
    ax[0, 0].imshow(pre_register_mean, cmap='gray')
    ax[0, 0].set_title("Pre-registration average stack")
    ax[0, 1].imshow(post_register_mean, cmap='gray')
    ax[0, 1].set_title("Post-registration average stack")
    ax[1, 0].imshow(reference, cmap='gray')
    ax[1, 0].set_title("Reference Image")
    ax[1, 1].imshow(post_register_mean, cmap='gray', alpha=1) 
    ax[1, 1].imshow(np.ma.masked_equal(masks, 1), cmap='nipy_spectral')
    ax[1, 1].set_title("Segmented ROIs on averaged stack")
    
    plt.show()

if __name__ == "__main__":
    main()
