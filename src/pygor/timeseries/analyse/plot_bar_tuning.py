
import pygor.load
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

sns.set_style("white")
file = r"C:\Users\Simen\Downloads\2025-2-12_0_3_15deg_2s_lines_RGB.h5"
def main(file):
    load = pygor.load.StaticBars(file)
    
    # plt.figure()  # Create a new figure
    load.plot_AB_delta(roi = 1, phase = 1)
    plt.show(block=False)


    plt.figure()  # Create a new figure for the second plot
    deltas = load.get_AB_deltas()
    sorted_idx = np.argsort(load.ipl_depths)
    absmax_delta = np.abs(np.max(deltas))
    # Take out FFF response
    deltas_a = deltas[:, :8-1]
    deltas_b = deltas[:, 8:-1]
    deltas = np.concatenate((deltas_a, deltas_b), axis=1)
    plt.axvline(7.5, c = "k", lw = 2)
    plt.imshow(deltas, cmap="coolwarm", aspect=.1, interpolation="none", clim = [-absmax_delta, absmax_delta])
    plt.gca().set_xlim(.5, 13.5)
    plt.grid(False)
    plt.text(3.5, 0, "ON bars", ha = "center", va = "bottom")
    plt.text(10.5, 0, "OFF bars", ha = "center", va = "bottom")
    plt.colorbar(label = r"$\Delta$ (A-B)")
    plt.ylabel("ROI (L to R)")
    # plt.ylabel("ROI (depth-sorted)")
    plt.xlabel("Bar position (15 deg.)")
    plt.show()
    # plt.pause(.1)
    return

if __name__ == "__main__":
    main(file)