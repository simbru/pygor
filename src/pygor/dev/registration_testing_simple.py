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
    time_start = timeit.default_timer()
    
    # Load data
    # example_path = pathlib.Path(r"D:\Igor analyses\OSDS\251112 OSDS\1_0_SWN_200_White.smp")
    example_path = pathlib.Path(r"D:\Igor analyses\SWN BC main\240517 ctrl data\0_0_SWN_200_5hz_RGBUV.smp")
    print("Loading data...")
    data = Core.from_scanm(example_path)
    data.preprocess(detrend=False)
    image_stack = data.images#[:, :, 2:]  # (time, height, width)
    
    # Parameters
    n_reference_frames = 100  # More frames for stable reference
    batch_size = 10 # ~64 stimulus cycles at 5Hz with 15.625Hz imaging
    upsample_factor = 10      # Lower for speed
    # Run registration (preprocessing already handled artifact removal)
    reference = np.mean(image_stack[:n_reference_frames], axis=0)
    reference_alt = np.std(image_stack[:n_reference_frames], axis=0)
    print("Running registration...")
    stats = data.register(
        n_reference_frames=n_reference_frames,
        batch_size=batch_size,
        upsample_factor=upsample_factor,
        order = 1,
        plot = True,
        edge_crop = 1,
        mode = "reflect",
        batch_mode = "std",
        reference_mode = "mean",
    )
    
    print(f"Registration completed in {timeit.default_timer() - time_start:.2f} seconds.")
    print("Registration stats:")
    for key, value in stats.items():
        print(f"  {key}: {value}")
    registered = data.images#[:, :, 2:]
    # Visualize
    viewer = napari.Viewer()
    viewer.add_image(reference, name='Reference')
    viewer.add_image(reference_alt, name='Reference (std)')
    viewer.add_image(image_stack[:batch_size].mean(axis=0) - registered[:batch_size].mean(axis=0), 
                     name='Difference', colormap='bwr')
    viewer.add_image(image_stack, name='Original')
    viewer.add_image(registered, name='Registered')
    viewer.add_image(np.std(image_stack, axis=0), name='Original std')
    viewer.add_image(np.std(registered, axis=0), name='Registered std')
    # average over every n frames in original image stack
    n = 10
    avg_original = np.std(image_stack[:(image_stack.shape[0] // n) * n].reshape(-1, n, image_stack.shape[1], image_stack.shape[2]), axis=1)
    avg_registered = np.std(registered[:(registered.shape[0] // n) * n].reshape(-1, n, registered.shape[1], registered.shape[2]), axis=1)
    viewer.add_image(avg_original, name=f'Original avg over {n} frames')
    viewer.add_image(avg_registered, name=f'Registered avg over {n} frames')
    napari.run()


if __name__ == "__main__":
    main()
