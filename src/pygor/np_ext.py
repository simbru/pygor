"""
Functions that extend Numpy functionality
"""
import numpy as np

def maxabs(x, axis = None):
    if axis is None:
        idx = np.ma.argmax(np.abs(x))
        return x.flat[idx]
    else: 
        # Get the indices of the maximum absolute values along the specified axis
        idx = np.ma.argmax(np.abs(x), axis=axis, keepdims=True)
        # Use np.take_along_axis to get the original values at those indices
        return np.take_along_axis(x, idx, axis=axis).squeeze(axis)
    
def absmax(x, axis = None):
    return np.max(np.abs(x), axis = None)