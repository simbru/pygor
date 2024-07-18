import numpy as np
import scipy.ndimage
import warnings
import matplotlib.pyplot as plt

# def estimate_rf_mask_metrics(strf_object, sanity_plot = False):
#     """Generate cartesian centroid and size estimate for each mask
#     in the STRF object."""
#     num_colours = 4
#     fetch_masks = strf_object.get_spatial_masks()
#     combine_masks = fetch_masks[0] * fetch_masks[1]
#     num_masks = combine_masks.shape[0]/num_colours
#     coms = []
#     sizes = []
#     # Note that here the estimated mask will come out
#     # systematically larger than the actual mask
#     # because we combine the masks, which means including 
#     # the surround component
#     for arr in combine_masks:
#         # Flip masks to make life easier
#         arr = np.invert(arr)
#         #arr = scipy.ndimage.binary_fill_holes(arr) # need this if not combining masks
#         size = np.sum(arr)
#         sizes.append(size)
#         com = scipy.ndimage.center_of_mass(arr)
#         coms.append(com)
#     coms = np.array(coms)
#     sizes = np.array(sizes)
#     nanmean = np.nanmean(coms, axis = 0)
#     # Logic: Pool all data from all masks irrespective of polarity,
#     # then split it into 8 in order to get negative, positive, RGBUV estimated
#     # centre of masses for all STRFs
#     smart_sizes = np.nanmean(np.split(sizes, 4, axis = 0), axis = 0)
#     smart_coms = np.nanmean(coms.reshape(int(coms.shape[0]/num_masks), -1, 2, order = 'F'), axis = 0)
#     smart_radius = np.sqrt(smart_sizes / np.pi)
#     if sanity_plot == True:
#         plt.imshow(combine_masks[0])
#         plt.scatter(coms[:, 1], coms[:, 0], c = "blue")
#         plt.scatter(nanmean[1], nanmean[0], c = "red")
#         plt.scatter(smart_coms[:, 1], smart_coms[:, 0], c = "green")
#     return smart_coms, smart_radius

# def gen_mask(shape, radius, centreXY = None, plot_res = False):
#     # Gen "canvas" array
#     arr = np.zeros(shape)
#     # Put centre in middle if not specified
#     if centreXY is None:
#         cy, cx = np.array(arr.shape)/2
#     else:
#         cx = centreXY[1]
#         cy = centreXY[0]
#     # Generate mask
#     y = np.arange(0, shape[0])
#     x = np.arange(0, shape[1])
#     # Logic from https://stackoverflow.com/questions/49330080/numpy-2d-array-selecting-indices-in-a-circle
#     mask = (x[np.newaxis,:]-cx)**2 + (y[:,np.newaxis]-cy)**2 < radius**2
#     arr[mask] = 1
#     return arr

# def gen_spoof_masks(strf_object):
#     output = []
#     smart_coms, smart_radius = estimate_rf_mask_metrics(strf_object)
#     for i, j in zip(smart_radius, smart_coms):
#         output.append(gen_mask(shape = strf_object.strfs[0, 0].shape, radius = i, centreXY = j))
#     output = np.array(output)
#     return output

def estimate_rf_mask_metrics(strf_object, sanity_plot = False):
    """Generate cartesian centroid and size estimate for each mask
    in the STRF object."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        num_colours = strf_object.numcolour
        fetch_masks = strf_object.get_spatial_masks()
        combine_masks = np.invert(np.logical_and(fetch_masks[0], fetch_masks[1]))
        # Note that here the estimated mask will come out
        # systematically larger than the actual mask
        # because we combine the masks, which means including 
        # the surround component
        num_masks = combine_masks.shape[0]/num_colours
        coms = np.array([scipy.ndimage.center_of_mass(mask) for mask in combine_masks])
        sizes = np.sum(combine_masks, axis=(1, 2))
        # sizes = np.nansum(combine_masks, axis=(1, 2))
        nanmean = np.nanmean(coms, axis = 0)
        # Logic: Pool all data from all masks irrespective of polarity,
        # then split it into 8 in order to get negative, positive, RGBUV estimated
        # centre of masses for all STRFs
        smart_sizes = np.nanmean(sizes.reshape(int(coms.shape[0]/num_masks), -1, order = 'F'), axis = 0)
        smart_coms = np.nanmean(coms.reshape(int(coms.shape[0]/num_masks), -1, 2, order = 'F'), axis = 0)
        smart_radius = np.sqrt(smart_sizes / np.pi)

    if sanity_plot == True:
        plt.imshow(combine_masks[0])
        plt.colorbar()
        plt.scatter(coms[:, 1], coms[:, 0], c = "blue")
        plt.scatter(nanmean[1], nanmean[0], c = "red")
        plt.scatter(smart_coms[:, 1], smart_coms[:, 0], c = "green")
    return smart_coms, smart_radius

def gen_mask(shape, radius, centreXY):
    # Gen "canvas" array
    arr = np.ones(shape)
    cx = centreXY[1]
    cy = centreXY[0]
    # Generate mask
    y = np.arange(0, shape[0])
    x = np.arange(0, shape[1])
    # Logic from https://stackoverflow.com/questions/49330080/numpy-2d-array-selecting-indices-in-a-circle
    mask = (x[np.newaxis,:]-cx)**2 + (y[:,np.newaxis]-cy)**2 < radius**2
    arr[mask] = 0
    return arr

def gen_spoof_masks(strf_object, output_shape = "masks"):
    shape = strf_object.strfs[0, 0].shape
    smart_coms, smart_radius = estimate_rf_mask_metrics(strf_object)
    output = np.zeros((smart_radius.size,) + shape, dtype=bool)
    for i, (r, c) in enumerate(zip(smart_radius, smart_coms)):
        output[i] = gen_mask(shape, r, c)
    # output = np.invert(output.astype(bool)).astype(bool) # flip back to correct polarity
    output = output.astype(int)
    spoofed_masks = output
    # fig, ax = plt.subplots(1, 1)
    # plt.imshow(spoofed_masks[1])
    # plt.colorbar()
    if output_shape == "rois":
        return spoofed_masks
    if output_shape == "masks":
        # Scale that to match the number of masks (ROIs x polarities x colours)
        spoofed_masks = np.expand_dims(spoofed_masks, axis = (1,2)) #set up axes
        spoofed_masks = np.repeat(spoofed_masks, strf_object.numcolour, axis = 0)  # set up ROIs
        spoofed_masks = np.repeat(spoofed_masks, 2, axis = 1) # set up polarities
        spoofed_masks = np.repeat(spoofed_masks, 20, axis = 2) # set up time
        return spoofed_masks
    
