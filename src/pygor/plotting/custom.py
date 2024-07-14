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
polarity_palette = ["black", "gainsboro", "grey"]
compare_conditions = {2 : ["gainsboro", "tomato"],
                    3 : ["gainsboro", "tomato", "darkblue"],
                    4 : ["gainsboro", "tomato", "darkblue", "purple"],
                    5 : ["gainsboro", "tomato", "darkblue", "purple", "darkgreen"]}

# red_map = matplotlib.colors.LinearSegmentedColormap.from_list("", ["dimgrey", "grey", "white","#ef8e00","darkred"])
# green_map = matplotlib.colors.LinearSegmentedColormap.from_list("", ["dimgrey", "grey","white","mediumaquamarine","teal"])
# blue_map = matplotlib.colors.LinearSegmentedColormap.from_list("", ["dimgrey", "grey","white","#5600fe","#4400cb"])
# violet_map = matplotlib.colors.LinearSegmentedColormap.from_list("", ["dimgrey", "grey","white","fuchsia","#b22cb2"])
# red_map    = matplotlib.colors.LinearSegmentedColormap.from_list("", ["black", "lightgrey", "#ef8e00"])
# green_map  = matplotlib.colors.LinearSegmentedColormap.from_list("", ["black", "lightgrey", "teal"])
# blue_map   = matplotlib.colors.LinearSegmentedColormap.from_list("", ["black", "lightgrey", "#5600fe"])
# violet_map = matplotlib.colors.LinearSegmentedColormap.from_list("", ["black", "lightgrey", "#ff00ff"])
# red_map    = matplotlib.colors.LinearSegmentedColormap.from_list("", ["black", "white", "#ef8e00"])
# green_map  = matplotlib.colors.LinearSegmentedColormap.from_list("", ["black", "white", "teal"])
# blue_map   = matplotlib.colors.LinearSegmentedColormap.from_list("", ["black", "white", "#5600fe"])
# violet_map = matplotlib.colors.LinearSegmentedColormap.from_list("", ["black", "white", "#ff00ff"])
red_map    = matplotlib.colors.LinearSegmentedColormap.from_list("", ["black", "silver", "#ef8e00"])
green_map  = matplotlib.colors.LinearSegmentedColormap.from_list("", ["black", "silver", "teal"])
blue_map   = matplotlib.colors.LinearSegmentedColormap.from_list("", ["black", "silver", "#5600fe"])
violet_map = matplotlib.colors.LinearSegmentedColormap.from_list("", ["black", "silver", "#ff00ff"])


maps_concat = [red_map, green_map, blue_map, violet_map]