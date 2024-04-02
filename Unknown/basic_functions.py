"""
The most basic functions for the pipeline. These may be re-organsised at any point. 
"""
raise DeprecationWarning("DEPRICATED DO NOT USE")
import numpy as np

# Initialise functions
def determine_centre_polarity(pols):
    centre_pol = pols[0]
    return centre_pol

def how_many_STRFs(list_of_keys):
    if any("STRF0" in s for s in list_of_keys):
        count = 0
        for i in list_of_keys: 
            if i[:5] == "STRF0":
                count += 1
    return count

