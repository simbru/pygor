# %%
from pygor.load import Core
# Load up data
load_path = r"FullFieldFlash_5_colour_demo.smp"
# load_path = r"D:\Igor analyses\SWN BC main\231018\1_0_SWN_200_Colour_0.smp"
data = Core(load_path, trigger_mode = 5)
# %%
# Preprocessing
data.preprocess(detrend = False)
data.register(plot = True)
# %%
# Get ROIs
data.segment_rois(plot=True, input_mode = "average")
data.view_stack_rois()
# %%
# Get traces and averages
data.extract_traces_from_rois()
data.compute_snippets_and_averages()
# Plot population and individual rois
data.plot_averages()
data.plot_averages([60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70],
    figsize = (10,8), independent_scale = True)

# %%
