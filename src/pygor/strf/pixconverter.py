import numpy as np

### Global values (relatively constant, as long as experimental config does not change)
# Solve for angle between fish and screen
fish_screen_dist_mm = 160  # measured from fish to center
screen_width_mm = 300  # total width
screen_half_width_mm = screen_width_mm / 2  # right-angle
# According to pythagorean theorem
hypoth_mm = np.sqrt(fish_screen_dist_mm**2 + screen_half_width_mm**2)
# Compute angle between fish and one side of screen (such that we get a right angle)
sin = screen_half_width_mm / hypoth_mm  # oppose/hypth
cos = fish_screen_dist_mm / hypoth_mm  # adjecent/hypth
tan = screen_half_width_mm / fish_screen_dist_mm  # opposite/adjecent
# Sanity check
arcsin = np.round(np.arcsin(sin), 8)
arccos = np.round(np.arccos(cos), 8)
arctan = np.round(np.arctan(tan), 8)
assert np.isclose(arcsin, arccos, arctan)
# Get screen width in degrees vis ang
half_screen_vis_ang = np.rad2deg(arcsin)
screen_vis_ang = (
    half_screen_vis_ang * 2
)  # because two right angle triangles added, e.g. 2x the angle
# Initilaise software parameters
screen_width_pix_au = 1820  # QDSpy display area in QDSpy units (confusingly named 'um' but is arbitrary unit (so 'au'))
screen_height_pix_au = 1140
# Based on this ratio, screen_width_height is:
screen_width_height_visang = (
    screen_vis_ang,
    screen_vis_ang * (screen_height_pix_au / screen_width_pix_au),
)


# Calculate pixel size in vis ang given the above
def calculate_boxes_on_screen(
    box_width_pix, screen_width_pix_au=1820, screen_height_pix_au=1140
):
    """This function calculates the number of blocks that can fit on a screen based on the box width
    and screen dimensions.

    Parameters
    ----------
    box_width_pix
        The `box_width_pix` parameter represents the width of each box in pixels that you want to display
    on the screen.
    screen_width_pix_au, optional
        The `screen_width_pix_au` parameter represents the width of the screen in pixels. In the provided
    function `calculate_boxes_on_screen`, this parameter is used to calculate the number of blocks that
    can fit horizontally on the screen based on the given box width in pixels.
    screen_height_pix_au, optional
        The `screen_height_pix_au` parameter represents the height of the screen in pixels. In this
    function, it is used to calculate the number of blocks that can fit vertically on the screen based
    on the given box width in pixels.

    Returns
    -------
        The function `calculate_boxes_on_screen` returns a NumPy array containing the number of blocks that
    can fit vertically and horizontally on the screen based on the provided box width in pixels and the
    default screen width and height in pixels.

    """
    # Work out how many blocks fit on on screen
    blocks_on_screen_horz = screen_width_pix_au / box_width_pix
    blocks_on_screen_vert = screen_height_pix_au / box_width_pix
    return np.array([blocks_on_screen_vert, blocks_on_screen_horz])


def pix_to_visang(
    *pix_nums,
    block_size,
    jitter_upsale=4,
    screen_width_pix=1820,
    screen_width_visang=86.306,
):
    """This Python function converts pixel values to visual angles using specified parameters.

    Parameters
    ----------
    block_size
        The `block_size` parameter represents the size of a block in pixels.
    jitter_upsale, optional
        The `jitter_upsale` parameter in the `pix_to_visang` function is used to specify the amount by
    which the block size is increased for jittering. It is a factor that determines how much the block
    size is scaled up before converting it to visual angle units.
    screen_width_pix, optional
        The `screen_width_pix` parameter represents the width of the screen in pixels. In the context of
    the `pix_to_visang` function, this parameter is used to calculate the visual angle based on the
    pixel input.
    screen_width_visang
        The `screen_width_visang` parameter represents the visual angle subtended by the width of the
    screen in degrees.

    Returns
    -------
        The function `pix_to_visang` returns either a single value (if only one `pix_nums` argument is
    provided) or an array of values (if multiple `pix_nums` arguments are provided) after performing
    some calculations.

    """
    output = [au_to_visang(block_size / jitter_upsale) * num for num in pix_nums]
    if len(pix_nums) == 1:
        output = output[0]
    else:
        output = np.array(output)
    return output


def visang_to_pix(
    *pix,
    pixwidth,
    block_size=None,
    jitter_upsale=None,
    screen_width_pix=None,
    screen_width_visang=86.306,
):
    """The function `visang_to_pix` converts visual angle measurements to pixel measurements based on
    specified parameters.

    Parameters
    ----------
    pixwidth
        The `pixwidth` parameter in the `visang_to_pix` function represents the width of a single pixel in
    visual angle units. This parameter is used in the calculation to convert visual angle units to
    pixels.
    block_size
        The `block_size` parameter in the `visang_to_pix` function represents the size of a block in the
    visualization in some unit of measurement (e.g., inches, centimeters, etc.). This parameter is used
    in the calculation to convert visual angles (visang) to pixels.
    jitter_upsale, optional
        The `jitter_upsale` parameter in the `visang_to_pix` function is set to a default value of 4. This
    parameter is used in the calculation within the function to determine the output based on the input
    parameters provided. If a specific value for `jitter_upsale` is
    screen_width_pix, optional
        The `screen_width_pix` parameter represents the width of the screen in pixels. In the provided
    function `visang_to_pix`, this parameter is used to calculate the output based on the input pixel
    values and other parameters such as `pixwidth`, `block_size`, `jitter_upsale`, and
    screen_width_visang
        The `screen_width_visang` parameter represents the width of the screen in visual angle units. This
    value is set to 86.306 in the function `visang_to_pix`.

    Returns
    -------
        The function `visang_to_pix` returns the conversion of visual angles to pixels based on the input
    parameters provided. The output will be a list of pixel values corresponding to the visual angles
    passed as input arguments. If only one visual angle is provided, a single pixel value will be
    returned.

    """
    if block_size is not None:
        raise DeprecationWarning("block_size has been deprecated")
    if jitter_upsale is not None:
        raise DeprecationWarning("jitter_upsale has been deprecated")
    if screen_width_pix is not None:
        raise DeprecationWarning("screen_width_pix has been deprecated")
    # output = [pix_to_visang(1, block_size = block_size) * num for num in visang]
    output = [num / (screen_width_visang / pixwidth) for num in pix]
    if len(pix) == 1:
        output = output[0]
    else:
        output = np.array(output)
    return output


def au_to_visang(box_width_pix, screen_width_pix=1820, screen_width_visang=86.306):
    """The function `au_to_visang` converts pixel measurements to visual angle measurements on a stimulator
    screen. The coordinates in STRFs are arbitrarily determined by the input array during STA. This function
    returns a scaler which can be used to make these arbitrary values tied to real measurements of visual
    angle for the stimulator screen.

    Parameters
    ----------
    box_width_pix
        The `box_width_pix` parameter represents the width of a box in pixels that you want to convert to
    visual angle. This function `au_to_visang` calculates the visual angle of the box based on the
    screen width in pixels and the corresponding visual angle of the screen.
    screen_width_pix, optional
        The `screen_width_pix` parameter represents the width of the screen in pixels. In the provided
    function `au_to_visang`, this parameter is used to calculate the visual angle per pixel on the
    screen.
    screen_width_visang
        The `screen_width_visang` parameter represents the visual angle of the screen width in degrees. In
    the provided function `au_to_visang`, this parameter is used to calculate the visual angle per pixel
    on the screen.

    Returns
    -------
        The function `au_to_visang` returns the visual angle in degrees corresponding to the input
    `box_width_pix` on the screen.

    """
    # Calculte the visang per pix
    single_pix_visang = screen_width_visang / screen_width_pix
    # Calculate block vis ang
    block_visang = single_pix_visang * box_width_pix
    return block_visang


def visang_to_au(width_visang, screen_width_pix=1820, screen_width_visang=86.306):
    """The function `visang_to_au` converts a given width in visual angles (visang) to pixels based on the
    screen width in pixels and visual angles.

    Parameters
    ----------
    width_visang
        The `width_visang` parameter represents the width in visangs that you want to convert to pixels on
    the screen. You can provide a value for `width_visang` to calculate the equivalent width in pixels.
    screen_width_pix, optional
        The `screen_width_pix` parameter represents the width of the screen in pixels. In the provided
    function `visang_to_au`, this parameter is used to calculate the pixels per visual angle ratio.
    screen_width_visang
        The `screen_width_visang` parameter represents the width of the screen in visual angle units
    (visang). This function `visang_to_au` takes a width in visual angle units and converts it to pixels
    based on the screen width in pixels and the screen width in visual angle units.

    Returns
    -------
        the width in pixels after converting the input width from visangs to pixels based on the screen
    width in pixels and the screen width in visangs.

    """
    # Calculate pix per visang ratio
    single_visang_pix = screen_width_pix / screen_width_visang
    # Convert width
    width_pix = single_visang_pix * width_visang
    return width_pix


def area_conversion(area_float, boxsize_um, sta_boxes_tuple=(15, 20)):
    """The function `area_conversion` calculates and scales an area based on the ratio of boxes on the
    screen to STA boxes.

    Parameters
    ----------
    area_float
        The `area_float` parameter is a floating-point number representing the area that you want to
    convert. It is the original area that you want to scale proportionally based on the screen and STA
    box sizes.
    boxsize_um
        The `boxsize_um` parameter represents the size of a box in micrometers. It is used in the
    `area_conversion` function to calculate the number of boxes that fit on the real screen.
    sta_boxes_tuple
        The `sta_boxes_tuple` parameter represents the dimensions of the STA (spatial temporal activity)
    boxes in a tuple format. The default value is set to (15, 20), which means the STA boxes have a
    width of 15 units and a height of 20 units. These dimensions are

    Returns
    -------
        the scaled area based on the input area, box size, and standard boxes tuple.

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
    return np.sqrt(screen_areas_ratio * area_float)


def area_to_diameter(float):
    """The function `area_to_diameter` calculates the diameter of a circle given its area.

    Parameters
    ----------
    float
        The `float` parameter in the `area_to_diameter` function likely represents the area of a circle.
    The function calculates the diameter of a circle based on the given area.

    Returns
    -------
        The function `area_to_diameter` is returning the diameter of a circle given the area as input. The
    formula used to calculate the diameter is 2 times the square root of half of the area.

    """
    return 2 * np.sqrt(float / 2)

def visang_deg_to_um(
    visual_angle_deg, 
    lens_to_retina_distance_um=50, 
    method="precise"
):
    """
    Convert visual angle in degrees to retinal projection size in micrometers
    for zebrafish larvae.
    
    Parameters:
    -----------
    visual_angle_deg : float or array
        Visual angle in degrees
    lens_to_retina_distance_um : float, optional
        Distance from lens to retina in micrometers (default: 50 µm for zebrafish larvae)
    method : str, optional
        Calculation method: "simple" (small angle approximation) or "precise" (exact trigonometry)
        
    Returns:
    --------
    retinal_projection_um : float or array
        Size of projection on retina in micrometers
    """
    
    # Convert degrees to radians
    visual_angle_rad = np.deg2rad(visual_angle_deg)
    
    if method == "simple":
        # Small angle approximation: retinal_size ≈ visual_angle_rad * lens_to_retina_distance
        retinal_projection_um = visual_angle_rad * lens_to_retina_distance_um
        
    elif method == "precise":
        # Exact trigonometry: retinal_size = 2 * lens_distance * tan(visual_angle/2)
        retinal_projection_um = 2 * lens_to_retina_distance_um * np.tan(visual_angle_rad / 2)
    
    else:
        raise ValueError("method must be 'simple' or 'precise'")
    
    return retinal_projection_um

def retinal_projection_um_to_visang_deg(
    retinal_projection_um, 
    lens_to_retina_distance_um=50, 
    method="precise"
):
    """
    Convert retinal projection size in micrometers to visual angle in degrees
    for zebrafish larvae (inverse function).
    
    Parameters:
    -----------
    retinal_projection_um : float or array
        Size of projection on retina in micrometers
    lens_to_retina_distance_um : float, optional
        Distance from lens to retina in micrometers (default: 50 µm for zebrafish larvae)
    method : str, optional
        Calculation method: "simple" or "precise"
        
    Returns:
    --------
    visual_angle_deg : float or array
        Visual angle in degrees
    """
    
    if method == "simple":
        # Inverse of small angle approximation
        visual_angle_rad = retinal_projection_um / lens_to_retina_distance_um
        
    elif method == "precise":
        # Inverse of exact trigonometry
        visual_angle_rad = 2 * np.arctan(retinal_projection_um / (2 * lens_to_retina_distance_um))
    
    else:
        raise ValueError("method must be 'simple' or 'precise'")
    
    # Convert radians to degrees
    visual_angle_deg = np.rad2deg(visual_angle_rad)
    
    return visual_angle_deg
