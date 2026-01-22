"""
Lightweight segmentation methods for pygor.

These methods use classical image processing (watershed, flood fill) and don't
require deep learning frameworks like Cellpose.
"""

import numpy as np
import warnings
from collections import deque

from scipy.ndimage import binary_erosion, binary_dilation, generate_binary_structure
from skimage.segmentation import watershed
from skimage.feature import peak_local_max


def _shrink_rois(masks, iterations=1, min_size_to_shrink=10):
    """
    Shrink ROIs by erosion to create gaps between adjacent ROIs.

    Parameters
    ----------
    masks : ndarray
        ROI mask (background=0, ROIs=1,2,3...)
    iterations : int
        Number of erosion iterations
    min_size_to_shrink : int
        ROIs smaller than this won't be eroded (to avoid eliminating small ROIs)

    Returns
    -------
    result : ndarray
        Mask with shrunk ROIs
    """
    result = np.zeros_like(masks)
    for roi_id in np.unique(masks):
        if roi_id == 0:
            continue
        roi_mask = masks == roi_id
        roi_size = roi_mask.sum()
        if roi_size >= min_size_to_shrink:
            shrunk = binary_erosion(roi_mask, iterations=iterations)
            if shrunk.any():
                result[shrunk] = roi_id
            else:
                # If erosion removes everything, keep original
                result[roi_mask] = roi_id
        else:
            # Small ROIs: keep as-is
            result[roi_mask] = roi_id
    return result


def _filter_small_rois(masks, min_size=3):
    """
    Remove ROIs smaller than min_size pixels.

    Parameters
    ----------
    masks : ndarray
        ROI mask (background=0, ROIs=1,2,3...)
    min_size : int
        Minimum ROI size in pixels

    Returns
    -------
    result : ndarray
        Mask with small ROIs removed
    """
    result = masks.copy()
    for roi_id in np.unique(masks):
        if roi_id == 0:
            continue
        if (masks == roi_id).sum() < min_size:
            result[masks == roi_id] = 0
    return result


def prepare_image(data, input_mode="combined", artifact_width=None):
    """
    Prepare image for segmentation from a Core/STRF object.

    Extracts the appropriate image based on input_mode, normalizes to 0-1,
    and masks out the light artifact region.

    Parameters
    ----------
    data : Core or STRF
        Pygor data object with images and/or correlation_projection
    input_mode : str
        Which image to use:
        - "correlation": Use correlation_projection
        - "average": Use average of images stack
        - "combined": Correlation weighted by average (recommended)
    artifact_width : int, optional
        Override artifact width. If None, uses data.params.artifact_width

    Returns
    -------
    img : ndarray
        Normalized image (0-1) with artifact region masked to 0
    artifact_fill_width : int
        Number of columns masked (for reference)
    """
    # Validate input_mode
    valid_modes = ("correlation", "average", "std", "combined")
    if input_mode not in valid_modes:
        raise ValueError(f"Unknown input_mode: '{input_mode}'. Use one of {valid_modes}")
    
    # Check data availability upfront
    has_images = data.images is not None
    has_average_stack = data.average_stack is not None
    has_correlation = data.correlation_projection is not None
    
    needs_correlation = input_mode in ("correlation", "combined")
    needs_images_stack = input_mode == "std"
    needs_average = input_mode in ("average", "combined")
    
    if needs_correlation and not has_correlation:
        raise ValueError("correlation_projection not available. Call compute_correlation_projection() first.")
    if needs_images_stack and not has_images:
        raise ValueError("Images stack required for 'std' mode but not available.")
    if needs_average and not (has_images or has_average_stack):
        raise ValueError("No image data available for computing average.")

    # Extract image based on mode
    if input_mode == "correlation":
        img = data.correlation_projection.copy()
    elif input_mode == "average":
        img = data.images.mean(axis=0)
    elif input_mode == "std":
        img = data.images.std(axis=0)
    elif input_mode == "combined":
        # Correlation weighted by average (structural + functional)
        corr = data.correlation_projection.copy()
        avg = data.images.mean(axis=0)
        # Normalize each and multiply
        corr_n = (corr - corr.min()) / (corr.max() - corr.min() + 1e-10)
        avg_n = (avg - avg.min()) / (avg.max() - avg.min() + 1e-10)
        img = corr_n * avg_n

    # Normalize to 0-1 BEFORE masking artifact (so artifact doesn't affect scaling)
    img = (img - img.min()) / (img.max() - img.min() + 1e-10)

    # Mask out light artifact region
    if artifact_width is None:
        artifact_width = data.params.artifact_width
    # IGOR uses inclusive indexing, so artifact_width=2 means pixels 0,1,2 are affected
    artifact_fill_width = artifact_width + 1

    # Set artifact region to 0 so it won't be segmented
    img[:, :artifact_fill_width] = 0

    return img.astype(np.float32), artifact_fill_width


def segment_watershed(
    img,
    threshold=0.1,
    min_distance=1,
    gap_pixels=2,
    min_size_to_shrink=10,
    min_roi_size=3,
):
    """
    Watershed segmentation seeded from local maxima.

    Parameters
    ----------
    img : ndarray
        2D image, normalized to 0-1
    threshold : float
        Intensity threshold for foreground (pixels below this are background)
    min_distance : int
        Minimum distance between seed peaks (controls ROI density)
    gap_pixels : int
        Erosion iterations to create gaps between ROIs (0 to disable)
    min_size_to_shrink : int
        ROIs smaller than this won't be eroded
    min_roi_size : int
        Remove ROIs smaller than this after segmentation

    Returns
    -------
    masks : ndarray
        ROI mask in Cellpose format (background=0, ROIs=1,2,3...)
    """
    # Threshold to get foreground
    binary = img > threshold

    # Find local maxima as seeds
    coords = peak_local_max(img, min_distance=min_distance, labels=binary.astype(int))

    # Create marker image
    markers = np.zeros_like(img, dtype=int)
    for i, (y, x) in enumerate(coords, start=1):
        markers[y, x] = i

    # Watershed (use negative image so watershed flows toward peaks)
    masks = watershed(-img, markers, mask=binary)

    # Erode to create gaps between ROIs
    if gap_pixels > 0:
        masks = _shrink_rois(masks, iterations=gap_pixels, min_size_to_shrink=min_size_to_shrink)

    # Remove tiny ROIs
    if min_roi_size > 0:
        masks = _filter_small_rois(masks, min_size=min_roi_size)

    return masks


def segment_flood_fill(
    img,
    threshold=0.15,
    min_distance=1,
    max_size=20,
    drop_fraction=0.2,
    min_gap=0,
    min_roi_size=3,
):
    """
    IGOR-style flood fill segmentation (region growing from peaks).

    Starts at intensity peaks and grows outward, stopping when:
    - Pixel intensity drops below peak * drop_fraction
    - Region reaches max_size pixels
    - Pixel is already claimed or in buffer zone

    Parameters
    ----------
    img : ndarray
        2D image, normalized to 0-1
    threshold : float
        Intensity threshold for foreground
    min_distance : int
        Minimum distance between seed peaks
    max_size : int
        Maximum pixels per ROI
    drop_fraction : float
        Stop growing when intensity drops below peak * drop_fraction
    min_gap : int
        Minimum gap (in pixels) to maintain from other ROIs (0 to disable)
    min_roi_size : int
        Remove ROIs smaller than this after segmentation

    Returns
    -------
    masks : ndarray
        ROI mask in Cellpose format (background=0, ROIs=1,2,3...)
    """
    # Threshold to get foreground
    binary = img > threshold

    # Find local maxima as seeds
    coords = peak_local_max(img, min_distance=min_distance, labels=binary.astype(int))

    # Sort peaks by intensity (brightest first get priority)
    peak_intensities = img[coords[:, 0], coords[:, 1]]
    coords = coords[np.argsort(peak_intensities)[::-1]]

    masks = np.zeros_like(img, dtype=int)
    buffer_zone = np.zeros_like(img, dtype=bool)

    # Structuring element for buffer dilation (8-connectivity)
    struct = generate_binary_structure(2, 2) if min_gap > 0 else None

    # 4-connectivity neighbors for flood fill
    neighbors = [(-1, 0), (1, 0), (0, -1), (0, 1)]

    roi_id = 0
    for py, px in coords:
        # Skip if seed is in buffer zone or already claimed
        if buffer_zone[py, px] or masks[py, px] != 0:
            continue

        roi_id += 1
        peak_val = img[py, px]
        stop_val = peak_val * drop_fraction

        # BFS flood fill using deque for O(1) popleft
        visited = set()
        queue = deque([(py, px)])
        region = []

        while queue and len(region) < max_size:
            y, x = queue.popleft()

            if (y, x) in visited:
                continue
            if y < 0 or y >= img.shape[0] or x < 0 or x >= img.shape[1]:
                continue
            if masks[y, x] != 0:  # already claimed
                continue
            if buffer_zone[y, x]:  # in buffer zone
                continue
            if img[y, x] < stop_val:
                continue
            if not binary[y, x]:
                continue

            visited.add((y, x))
            region.append((y, x))

            # Add neighbors to queue
            for dy, dx in neighbors:
                queue.append((y + dy, x + dx))

        # Assign region to mask
        for y, x in region:
            masks[y, x] = roi_id

        # Update buffer zone around this ROI
        if min_gap > 0 and len(region) > 0:
            roi_mask = masks == roi_id
            dilated = binary_dilation(roi_mask, structure=struct, iterations=min_gap)
            buffer_zone = buffer_zone | (dilated & ~roi_mask)

    # Remove tiny ROIs
    if min_roi_size > 0:
        masks = _filter_small_rois(masks, min_size=min_roi_size)

    return masks


def segment(
    image_or_data,
    *,
    method="watershed",
    input_mode="combined",
    postprocess=False,
    verbose=True,
    **kwargs,
):
    """
    Segment an image using lightweight methods (no deep learning required).

    This is the main entry point for lightweight segmentation. It can accept either
    a raw image array or a pygor Core/STRF object.

    Parameters
    ----------
    image_or_data : ndarray or Core/STRF
        Either a 2D image array or a pygor data object.
        If a data object, the image is extracted based on input_mode and
        artifact masking is applied automatically.
    method : str
        Segmentation method: "watershed" (default) or "flood_fill"
    input_mode : str
        For data objects: which image to use ("correlation", "average", "combined")
    postprocess : bool
        Not used for lightweight methods (kept for API consistency)
    verbose : bool
        Print progress messages
    **kwargs
        Method-specific parameters:

        Common parameters:
        - threshold : float (watershed: 0.1, flood_fill: 0.15)
        - min_distance : int (default: 1)
        - min_roi_size : int (default: 3)

        Watershed parameters:
        - gap_pixels : int (default: 2)
        - min_size_to_shrink : int (default: 10)

        Flood fill parameters:
        - max_size : int (default: 20)
        - drop_fraction : float (default: 0.2)
        - min_gap : int (default: 0)

        For raw images only:
        - artifact_width : int (columns to mask, default: 0)

    Returns
    -------
    masks : ndarray
        ROI mask in Cellpose format (background=0, ROIs=1,2,3...)

    Examples
    --------
    >>> # With pygor data object (recommended)
    >>> obj = pygor.load.Core("recording.h5")
    >>> obj.segment_rois(mode="watershed", input_mode="combined")

    >>> # With raw image
    >>> from pygor.segmentation.lightweight import segment
    >>> masks = segment(my_image, method="flood_fill", threshold=0.2)
    """
    # Check if we have a data object or raw image
    is_data_object = hasattr(image_or_data, 'params') and hasattr(image_or_data, 'images')

    if is_data_object:
        # Extract and prepare image from data object
        if verbose:
            print(f"Preparing image (mode={input_mode})...")
        img, artifact_fill_width = prepare_image(image_or_data, input_mode=input_mode)
        if verbose:
            print(f"  Masked artifact region: columns 0-{artifact_fill_width - 1}")
    else:
        # Raw image - apply minimal preprocessing
        img = np.asarray(image_or_data, dtype=np.float32)
        # Normalize if not already
        if img.max() > 1.0 or img.min() < 0.0:
            img = (img - img.min()) / (img.max() - img.min() + 1e-10)
        # Handle artifact masking for raw images
        artifact_width = kwargs.pop("artifact_width", 0)
        if artifact_width > 0:
            artifact_fill_width = artifact_width + 1
            img[:, :artifact_fill_width] = 0
            if verbose:
                print(f"  Masked artifact region: columns 0-{artifact_fill_width - 1}")

    # Extract common parameters
    min_roi_size = kwargs.pop("min_roi_size", 3)

    # Run segmentation
    if method == "watershed":
        if verbose:
            print("Running watershed segmentation...")
        masks = segment_watershed(
            img,
            threshold=kwargs.pop("threshold", 0.1),
            min_distance=kwargs.pop("min_distance", 1),
            gap_pixels=kwargs.pop("gap_pixels", 2),
            min_size_to_shrink=kwargs.pop("min_size_to_shrink", 10),
            min_roi_size=min_roi_size,
        )
    elif method == "flood_fill":
        if verbose:
            print("Running flood fill segmentation...")
        masks = segment_flood_fill(
            img,
            threshold=kwargs.pop("threshold", 0.15),
            min_distance=kwargs.pop("min_distance", 1),
            max_size=kwargs.pop("max_size", 20),
            drop_fraction=kwargs.pop("drop_fraction", 0.2),
            min_gap=kwargs.pop("min_gap", 0),
            min_roi_size=min_roi_size,
        )
    else:
        raise ValueError(f"Unknown method: '{method}'. Use 'watershed' or 'flood_fill'")

    # Warn about unused kwargs
    if kwargs:
        warnings.warn(f"Unused parameters: {list(kwargs.keys())}")

    n_rois = len(np.unique(masks)) - 1  # -1 for background
    if verbose:
        print(f"  Found {n_rois} ROIs")

    return masks
