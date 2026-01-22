#%%
import pathlib
import os
if os.getcwd() != str(pathlib.Path(__file__).parent):
    os.chdir(pathlib.Path(__file__).parent)

# Activate your environment with pygor installed, and import pygor.load
# pygor.load will output a message about which classes are available to you.
# This takes a moment to fetch (expect waiting for a few seconds), and will
# only output the first time it is run.
import pygor.load

 #%%`
# Path to example data - update this to your file
EXAMPLE_PATH = r"FullFieldFlash_5_colour_demo.smp"
CUSTOM_CONFIG = r"configs\example.toml" # Specify custom config if needed


#%%
obj = pygor.load.Core(EXAMPLE_PATH, config=CUSTOM_CONFIG)

#%%
obj.preprocess(detrend=True)


#%% 
obj.register(plot=True)

#%%
# obj.segment_rois()

# obj.extract_traces_from_rois



# %%
