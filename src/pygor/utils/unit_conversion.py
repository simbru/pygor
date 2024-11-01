import numpy as np

screen_height_width_visang = (54.059, 86.305)


def calculate_boxes_on_screen(
    box_width_pix, screen_width_pix_au=1820, screen_height_pix_au=1140
):
    # Work out how many blocks fit on on screen
    blocks_on_screen_horz = screen_width_pix_au / box_width_pix
    blocks_on_screen_vert = screen_height_pix_au / box_width_pix
    return np.array([blocks_on_screen_vert, blocks_on_screen_horz])


def au_to_visang(box_width_pix, screen_width_pix=1820, screen_width_visang=86.306):
    """
    The coordinates in STRFs are arbitrarily determined by the input array during STA.
    This function returns a scaler which can be used to make these arbitrary values tied to
    real measurements of visual angle for the stimulator screen.
    """
    # Calculte the visang per pix
    single_pix_visang = screen_width_visang / screen_width_pix
    # Calculate block vis ang
    block_visang = single_pix_visang * float(box_width_pix)
    return block_visang


def area_conversion_old(area_float, boxsize_um, sta_boxes_tuple=(15, 20)):
    """
    Leave in for backwards compatability with experiments done before 04/04/2023
    """
    # Get the amount of boxes that fit on the real screen
    screen_boxes_tuple = calculate_boxes_on_screen(boxsize_um)
    ## Calculate that area
    screen_area_sq = np.multiply(screen_boxes_tuple[0], screen_boxes_tuple[1])
    # Calculate the area of the STA
    sta_area_sq = np.multiply(sta_boxes_tuple[0], sta_boxes_tuple[1])
    ## Determine the ratio between the boxes on screen and the STA boxes (without upscale factor)
    screen_areas_ratio = sta_area_sq / screen_area_sq
    ## Multiply that ratio by the area_float to scale the area proportionally
    return area_float * (au_to_visang(boxsize_um) ** 2) * screen_areas_ratio


def area_conversion(area_float, boxsize_um, upscale_factor=4):
    """Streamlined, simple area conversion that assumes no whacky
    weirdness with STA containing boxes that aren't actually displayed on screen.
    In short, make sure displayed STA == analysis STA (then should work fine)
    """
    return (area_float * au_to_visang(boxsize_um) ** 2) / upscale_factor
    # return ((au_to_visang(boxsize_um))**2 * area_float) / upscale_factor


# def area_conversion_new()

"""
TODO
- Correct area conversion (both new and old)
- Validate area conversion 
"""

# def area_conversion(area_float, boxsize_um):
#     # Get the amount of boxes that fit on the real screen
#     screen_boxes_tuple = calculate_boxes_on_screen(boxsize_um)
#     ## Calculate that area
#     screen_area_sq = np.multiply(screen_boxes_tuple[0], screen_boxes_tuple[1])
#     # Calculate the area of the STA
#     sta_area_sq = np.multiply(screen_boxes_tuple[0], screen_boxes_tuple[1])
#     ## Determine the ratio between the boxes on screen and the STA boxes (without upscale factor)
#     screen_areas_ratio = sta_area_sq / screen_area_sq
#     ## Multiply that ratio by the area_float to scale the area proportionally
#     return np.sqrt(screen_areas_ratio * area_float)
