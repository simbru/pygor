"""
Functions that extend Numpy functionality
"""
import numpy as np

def maxabs(x, axis):
    # Get the indices of the maximum absolute values along the specified axis
    idx = np.ma.argmax(np.abs(x), axis=axis, keepdims=True)
    # Use np.take_along_axis to get the original values at those indices
    return np.take_along_axis(x, idx, axis=axis).squeeze(axis)

