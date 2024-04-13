import matplotlib

uv_cmap = matplotlib.colors.LinearSegmentedColormap.from_list("custom_uv", [(0.0, "white"), (1, "violet")])
uv_r_cmap = matplotlib.colors.LinearSegmentedColormap.from_list("custom_uv_r", [(0.0, "violet"), (1, "white")])

chromatic_palette = ['red', 'green', 'blue', 'violet'] # R G B UV
rgbuv_palette = ['red', 'green', 'violet']
chromatic_hues = ["R", "G", "B", "UV"]

achromatic_palette = ['black', 'brown'] # BW BWnoUV 
achromatic_hues = ["BW", "BWnoUV"]
all_palette = chromatic_palette + achromatic_palette
all_hues = ["R", "G", "B", "UV", "BW", "BWnoUV"]
rguv_hues = ["R", "G", "UV"]
nanometers = ["588", "478", "422", "375"]
fish_palette = ["orange", "royalblue","blueviolet", "fuchsia"]
