import pathlib
import numpy as np
import napari
from cellpose import models
import matplotlib.pyplot as plt
from skimage.registration import phase_cross_correlation
from scipy.ndimage import shift
from pygor.classes.core_data import Core
import timeit
import h5py
"""
Simple batch-averaged registration for calcium imaging.
"""

def main():
    time_start = timeit.default_timer()

    # Load data
    # example_path = pathlib.Path(r"D:\Igor analyses\OSDS\251112 OSDS\1_0_SWN_200_White.smp")
    example_path = pathlib.Path(r"D:\Igor analyses\OSDS\251105 OSDS\1_0_SWN_200_White.smp")
    # print("Loading data...")
    data = Core.from_scanm(example_path)
    data.preprocess(detrend=False)
    # compare to h5 load
    # data_h5 = Core(r"D:\Igor analyses\OSDS\251020 different OSDS\clone.h5")
    # data_h5 = r"D:\Igor analyses\OSDS\251020 different OSDS\clone.h5"
    # with h5py.File(data_h5, 'r') as f:
    #     os_params = f["OS_Parameters"]
    #     print(list(os_params))
    # plt.imshow(data.images[0, :, 2:], cmap='gray')
    # plt.show()
    # exit() 
    image_stack = data.images#[:, :, 2:]  # (time, height, width)
    pre_register_mean = np.mean(image_stack, axis=0)
    # Parameters
    n_reference_frames = 1000  # More frames for stable reference (1000)
    batch_size = 100     # ~64 stimulus cycles at 5Hz with 15.625Hz imaging (10)
    upsample_factor = 2      # Lower for speed (10)
    # Run registration (preprocessing already handled artifact removal)
    reference = np.mean(image_stack[:n_reference_frames], axis=0)

    # fig, ax = plt.subplots(1, 2, figsize=(5, 5))
    # ax[0].imshow(image_stack[0], cmap='gray')
    # ax[1].plot(image_stack[0, 30], color='red')
    # plt.show()
    # exit()    


    # plt.imshow(data.rois, cmap='prism')
    # plt.show()
    
    viewer = napari.Viewer()
    viewer.add_image(data.images, name="Pre-registration")
    
    print("Running registration...")
    time_start = timeit.default_timer()
    stats = data.register(
        n_reference_frames=n_reference_frames,
        batch_size=batch_size,
        upsample_factor=upsample_factor,
        order = 2,
        mode = 'nearest',
        plot = False,
        artefact_crop = 3,  # Crop edges to remove light artefacts correction
    )
    
    """
    TODO:
    - Check registration after artefact crop
    - Add artefact crop factor to some variable 
    - Check that registration works as intended
    """

    if stats["mean_error"] < 0.05:
        print("Registration successful.")
    else:
        print("Warning: Registration error exceeds threshold.")
        # print("Exiting without saving. Adjust parameters and try again.")
        # exit()

    post_register_mean = np.mean(data.images, axis=0)

    # # Segment ROIs using trained Cellpose model
    print("Segmenting ROIs...")
    masks = data.segment_rois(model_path=r"models\synaptic\cellpose_rois")


    viewer.add_image(data.images, name="Post-registration")
    viewer.add_image(pre_register_mean, name="Pre-registration Mean", colormap='gray', opacity=0.5)
    viewer.add_image(post_register_mean, name="Post-registration Mean", colormap='gray', opacity=0.5)
    viewer.add_image(data.rois_alt, name="Segmented ROIs", colormap='prism', opacity=0.5)
    napari.run()

# # Reload h5 file to verify ROIs saved correctly
# data = Core(r"D:\Igor analyses\OSDS\251020 different OSDS\clone.h5")
# fig, ax = plt.subplots(1, 2, figsize=(8, 4))
# ax[0].imshow(data.rois, cmap='prism')
# ax[1].imshow(masks, cmap='prism')
# plt.show()

    # # # Visualize results with matplotlib
    # fig, ax = plt.subplots(2, 2, figsize=(12, 6))
    # ax[0, 0].imshow(pre_register_mean, cmap='gray')
    # ax[0, 0].set_title("Pre-registration average stack")
    # ax[0, 1].imshow(post_register_mean, cmap='gray')
    # ax[0, 1].set_title("Post-registration average stack")
    # ax[1, 0].imshow(reference, cmap='gray')
    # ax[1, 0].set_title("Reference Image")
    # ax[1, 1].imshow(post_register_mean, cmap='gray', alpha=1) 
    # # ax[1, 1].imshow(np.ma.masked_equal(masks, 1), cmap='prism')
    # ax[1, 1].imshow(data.rois, cmap='prism')
    # ax[1, 1].set_title("Segmented ROIs on averaged stack")    
    # plt.show()



    # print(data.num_rois)
    # print(np.unique(data.rois))
    # print(np.unique(data.rois_alt))

    print("Elapsed time: {:.1f} seconds".format(timeit.default_timer() - time_start))
if __name__ == "__main__":
    main()