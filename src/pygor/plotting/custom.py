import matplotlib

uv_cmap = matplotlib.colors.LinearSegmentedColormap.from_list("custom_uv", [(0.0, "white"), (1, "violet")])
uv_r_cmap = matplotlib.colors.LinearSegmentedColormap.from_list("custom_uv_r", [(0.0, "violet"), (1, "white")])
chromatic_palette = ['red', 'green', 'blue', 'violet'] # R G B UV
rgbuv_palette = ["r", "g", "b", "violet"]
chromatic_hues = ["R", "G", "B", "UV"]
achromatic_palette = ['black', 'brown'] # BW BWnoUV 
achromatic_hues = ["BW", "BWnoUV"]
all_palette = chromatic_palette + achromatic_palette
all_hues = ["R", "G", "B", "UV", "BW", "BWnoUV"]
rguv_hues = ["R", "G", "UV"]
nanometers = ["588", "478", "422", "375"]
fish_palette = ["#ef8e00", "teal","#5600fe", "fuchsia"]
polarity_palette = ["grey", "black", "gainsboro"]
red_map = matplotlib.colors.LinearSegmentedColormap.from_list("", ["dimgrey", "grey", "white","#ef8e00","darkred"])
green_map = matplotlib.colors.LinearSegmentedColormap.from_list("", ["dimgrey", "grey","white","mediumaquamarine","teal"])
blue_map = matplotlib.colors.LinearSegmentedColormap.from_list("", ["dimgrey", "grey","white","#5600fe","#4400cb"])
violet_map = matplotlib.colors.LinearSegmentedColormap.from_list("", ["dimgrey", "grey","white","fuchsia","#b22cb2"])
