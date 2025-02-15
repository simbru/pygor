import numpy as np
import utilities


def stack_to_rgb(stack, eight_bit=True):
    stack = np.repeat(np.expand_dims(stack, -1), 3, -1)
    if eight_bit == True:
        stack = utilities.min_max_norm(stack, 0, 255).astype("int")
    else:
        stack = utilities.min_max_norm(stack, 0, 1).astype("float")
    return stack


# rgb_image = stack_to_rgb(load_image)


def basic_stim_overlay(
    rgb_array,
    frame_width=125,
    xy_loc=(3, 3),
    frame_duration=32,
    size=10,
    repeat_interval=4,
    colour_list=[(255, 128, 0), (0, 128, 128), (0, 0, 180), (128, 0, 128)],
):
    # Prep colour list
    if len(colour_list) == 1:
        np.tile(colour_list, (repeat_interval, 1))
    for i in range(repeat_interval):
        # Frame, y, x
        rgb_array[
            0:frame_duration,
            xy_loc[1] : xy_loc[1] + size,
            xy_loc[0] + frame_width * i : xy_loc[0] + size + frame_width * i,
        ] = colour_list[i]
    return rgb_array
