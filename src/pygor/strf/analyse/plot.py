from typing import Callable
try:
    from collections import Iterable
except:
    from collections.abc import Iterable
import numpy as np
import matplotlib.pyplot as plt

def plot_areas_vs(df, row1 : str, row2 : str, colour = None, valmod : Callable = None, ax : plt.axes = None, legend : bool = True, 
                  labels : tuple = None) -> (plt.figure, plt.axis):
    # Get max value for axis limit
    max_val = np.max([df[row1], df[row2]]) 
    max_val_leeway = max_val + max_val * 0.025
    # Handle colour input
    if isinstance(colour, tuple):
        c1, c2 = colour
    if isinstance(colour, str):
        c1 = colour
        c2 = colour
    elif colour is None:
        c1 = "lightgrey"
        c2 = "darkgrey"
    # Replace 0 values with NaN
    df = df.replace(0, np.nan)
    # Generate figure
    fig = plt.figure(figsize = (5, 5))
    # Generate gridspec
    gs = fig.add_gridspec(3, 3,  width_ratios=(4, 1, 1), height_ratios=(1,1,4),
                        left=0.1, right=0.9, bottom=0.1, top=0.9,
                        wspace=.01, hspace=0.01)
    # Start building main subplot
    ax = fig.add_subplot(gs[2, 0])
    # Label handling 
    if labels is None:
        labels = (row1, row2)
    if isinstance(labels, Iterable) is False: 
        raise ValueError("Labels must be a tuple or list-like of strings")
    ax.set_xlabel(row1)
    ax.set_ylabel(row2)
    ax.scatter(df[row1], df[row2], s = 20, c = "k", label = "Both")
    ax.plot(np.arange(0, max_val, 1), np.arange(0, max_val, 1), color = "grey", ls = "--", alpha = .5)
    ax_histx = fig.add_subplot(gs[1, 0], sharex=ax)
    sns.kdeplot(df[row1], color =  "k", ax = ax_histx)
    sns.kdeplot(df[row1], color = pygor.plotting.fish_palette[-1], ax = ax_histx)
    ax_stackx = fig.add_subplot(gs[0, 0], sharex=ax)
    sns.stripplot(x = df[row1], s = 5, c = pygor.plotting.fish_palette[-1], label = f"{row1} only", ax = ax_stackx)
    ax_stackx.legend_.remove()
    ax_histy = fig.add_subplot(gs[2, 1], sharey=ax)
    sns.kdeplot(y = df[row2], color =  "k", ax = ax_histy)
    sns.kdeplot(y = df[row2], color = pygor.plotting.fish_palette[0], ax = ax_histy)
    ax_stacky = fig.add_subplot(gs[2, 2], sharey=ax)
    sns.stripplot(y = df[row2], s = 5, c = pygor.plotting.fish_palette[0], label = f"{row2} only", ax = ax_stacky)
    ax_stacky.legend_.remove()
    ax_histx.axis("off")
    ax_histy.axis("off")
    ax_stackx.axis("off")
    ax_stacky.axis("off")
    if legend is True:
        dots_labels = [ax.get_legend_handles_labels() for ax in fig.axes] # combine handles and labels
        dots, labels = [sum(lol, []) for lol in zip(*dots_labels)]
        fig.legend(reversed(dots), reversed(labels), bbox_to_anchor=(.9, .8))
    ax.set_xlim(0, max_val_leeway)
    ax.set_ylim(0, max_val_leeway)
    ax.set_title("Diameter (° vis. ang.)", y = -.3)
    ax.set_ylabel("LWS")
    ax.set_xlabel("SWS1")
    
    return fig, ax