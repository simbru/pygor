import re

import numpy as np

screen_height_width_visang = (54.059, 86.305)

# Plausible noise box widths in screen au. Wide enough for any box that has been
# run, narrow enough to exclude a YYMMDD date or a YYYY year.
BOX_SIZE_AU_RANGE = (10, 1000)

# The box width is written next to the stimulus label: "SWN_200", "SWN200",
# "ColourSWN_200", "200_SWN".
_SWN_BOX_PATTERN = re.compile(r"SWN[_\-. ]?(\d+)|(\d+)[_\-. ]?SWN", re.IGNORECASE)


def parse_box_size_au(name, valid_range=BOX_SIZE_AU_RANGE):
    """Infer the noise box width in screen au from a recording name.

    Recording names are the only place the box width is recorded, so it has to
    be read back out of them. The number sitting next to the stimulus label is
    the box width, and that pairing is tried first. Failing that, a single
    plausible number anywhere in the name is accepted.

    Returns None when the name gives no unambiguous answer -- notably for the
    YYMMDD-prefixed names, where taking the largest number would return the
    date. Callers must handle None rather than fall back to a guess: a wrong
    box width silently rescales every metric reported in visual angle.
    """
    lo, hi = valid_range
    for match in _SWN_BOX_PATTERN.finditer(str(name)):
        found = int(match.group(1) or match.group(2))
        if lo <= found <= hi:
            return found
    numbers = {int(tok) for tok in re.split(r"\D+", str(name)) if tok}
    candidates = sorted(num for num in numbers if lo <= num <= hi)
    if len(candidates) == 1:
        return candidates[0]
    return None


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
