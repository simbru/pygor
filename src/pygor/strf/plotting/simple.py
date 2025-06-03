import pygor

import numpy as np
import matplotlib.pyplot as plt

def plot_collapsed_strfs(self, cval=None, channel = None, cmap = "bwr", origin = "upper"):
    if channel is not None:
        array = pygor.utilities.multicolour_reshape(self.collapse_times(), channel)[channel-1]
    else:
        array = self.collapse_times()
    # Grid layout
    max_x = 10
    num_slices = array.shape[0]
    num_rows = int(np.ceil(num_slices / max_x))  # Compute number of rows

    # Pad with empty slices if necessary
    empty_slices = num_rows * max_x - num_slices
    pad_array = np.zeros((empty_slices, array.shape[1], array.shape[2]))  # Padding with zeros
    array_padded = np.vstack([array, pad_array])  # Ensure full rows

    # Reshape into (num_rows, max_x, height, width)
    array_grid = array_padded.reshape(num_rows, max_x, array.shape[1], array.shape[2])

    # Concatenate along y and x to form a 2D image
    image = np.block([[array_grid[i, j] for j in range(max_x)] for i in range(num_rows)])

    if cval is None:
        percentile = np.percentile(image, [1, 99])
        cval = max(abs(percentile[0]), abs(percentile[1]))  # Symmetric color limits

    # Display
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.imshow(image, cmap=cmap, interpolation="none", clim=(-cval, cval), origin = origin)
    ax.axis("off")
    print(max_x, num_slices)
    # Overlay slice numbers
    h, w = array.shape[1], array.shape[2]  # Individual block height and width
    for i in range(num_rows):
        for j in range(max_x):
            if channel is not None:
                slice_idx = i * max_x + j * channel
            else:
                slice_idx = i * max_x + j
            # if slice_idx < num_slices:  # Avoid labeling padded regions
            x_pos = j * w + 1  # Adjust text position slightly
            y_pos = i * h + 1  # Adjust for visibility
            ax.text(x_pos, y_pos, str(slice_idx), fontsize=8, color="black", 
                    bbox=dict(facecolor='white', alpha=0.5, edgecolor='none'))
    plt.show()