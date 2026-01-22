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

from pygor.segmentation.preprocessing import (
    normalize,
    enhance_unsharp,
    create_anatomy_mask,
    filter_points_by_mask,
    prepare_image,
)


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


def segment_watershed(
    img,
    threshold=0.1,
    min_distance=1,
    gap_pixels=2,
    min_size_to_shrink=10,
    min_roi_size=3,
    anatomy_mask=None,
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
    anatomy_mask : ndarray (bool), optional
        Binary mask where True = valid region for segmentation.
        If provided, only seeds within this mask are used.

    Returns
    -------
    masks : ndarray
        ROI mask in Cellpose format (background=0, ROIs=1,2,3...)
    """
    # Threshold to get foreground
    binary = img > threshold

    # Combine with anatomy mask if provided
    if anatomy_mask is not None:
        binary = binary & anatomy_mask

    # Find local maxima as seeds
    coords = peak_local_max(img, min_distance=min_distance, labels=binary.astype(int))

    # Filter seeds by anatomy mask if provided (extra safety)
    if anatomy_mask is not None and len(coords) > 0:
        coords = filter_points_by_mask(coords, anatomy_mask)

    # Create marker image
    markers = np.zeros_like(img, dtype=int)
    for i, (y, x) in enumerate(coords, start=1):
        markers[int(y), int(x)] = i

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
    anatomy_mask=None,
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
    anatomy_mask : ndarray (bool), optional
        Binary mask where True = valid region for segmentation.
        If provided, only seeds within this mask are used and
        flood fill is constrained to the mask region.

    Returns
    -------
    masks : ndarray
        ROI mask in Cellpose format (background=0, ROIs=1,2,3...)
    """
    # Threshold to get foreground
    binary = img > threshold

    # Combine with anatomy mask if provided
    if anatomy_mask is not None:
        binary = binary & anatomy_mask

    # Find local maxima as seeds
    coords = peak_local_max(img, min_distance=min_distance, labels=binary.astype(int))

    # Filter seeds by anatomy mask if provided
    if anatomy_mask is not None and len(coords) > 0:
        coords = filter_points_by_mask(coords, anatomy_mask)

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
    plot=False,
    # Image enhancement (optional)
    unsharp_radius=None,
    unsharp_amount=None,
    # Anatomy masking (optional)
    anatomy_threshold=None,
    anatomy_thresh_mult=0.2,
    erode_iterations=1,
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
    plot : bool
        If True, display a figure showing the input image, ROI masks, and overlay

    Image Enhancement Parameters (optional)
    ---------------------------------------
    unsharp_radius : float, optional
        Radius for unsharp masking (0.5-2.0 typical). If None, no enhancement.
    unsharp_amount : float, optional
        Strength of sharpening (1.0-5.0 typical). If None, no enhancement.
        Both radius and amount must be set to enable enhancement.

    Anatomy Masking Parameters (optional)
    -------------------------------------
    anatomy_threshold : str or float, optional
        If set, creates an anatomy mask to exclude background regions.
        - 'otsu': Automatic threshold using Otsu's method
        - float: Manual threshold value (0-1 range)
        If None (default), no anatomy masking is applied.
    anatomy_thresh_mult : float
        Multiply threshold by this (lower = more permissive). Default: 0.2
    erode_iterations : int
        Erode mask to exclude edge regions. Default: 1

    Method-Specific Parameters
    --------------------------
    **kwargs
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

    >>> # With anatomy masking to exclude background
    >>> obj.segment_rois(mode="watershed", anatomy_threshold='otsu')

    >>> # With image enhancement for weak signals
    >>> obj.segment_rois(mode="flood_fill", unsharp_radius=1.0, unsharp_amount=2.5)

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
            img = normalize(img)
        # Handle artifact masking for raw images
        artifact_width = kwargs.pop("artifact_width", 0)
        if artifact_width > 0:
            artifact_fill_width = artifact_width + 1
            img[:, :artifact_fill_width] = 0
            if verbose:
                print(f"  Masked artifact region: columns 0-{artifact_fill_width - 1}")

    # Apply optional image enhancement
    if unsharp_radius is not None and unsharp_amount is not None:
        if verbose:
            print(f"Enhancing image (unsharp: r={unsharp_radius}, a={unsharp_amount})...")
        img = enhance_unsharp(img, radius=unsharp_radius, amount=unsharp_amount)

    # Create optional anatomy mask
    anatomy_mask = None
    if anatomy_threshold is not None:
        if verbose:
            print(f"Creating anatomy mask (threshold={anatomy_threshold}, mult={anatomy_thresh_mult})...")
        anatomy_mask = create_anatomy_mask(
            img,
            method=anatomy_threshold,
            thresh_mult=anatomy_thresh_mult,
            erode_iterations=erode_iterations
        )
        mask_coverage = anatomy_mask.sum() / anatomy_mask.size * 100
        if verbose:
            print(f"  Anatomy mask covers {mask_coverage:.1f}% of image")

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
            anatomy_mask=anatomy_mask,
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
            anatomy_mask=anatomy_mask,
        )
    else:
        raise ValueError(f"Unknown method: '{method}'. Use 'watershed' or 'flood_fill'")

    # Warn about unused kwargs
    if kwargs:
        warnings.warn(f"Unused parameters: {list(kwargs.keys())}")

    n_rois = len(np.unique(masks)) - 1  # -1 for background
    if verbose:
        print(f"  Found {n_rois} ROIs")

    # Plot if requested
    if plot:
        from pygor.segmentation.plotting import plot_segmentation
        was_enhanced = unsharp_radius is not None and unsharp_amount is not None
        plot_segmentation(
            img,
            masks,
            input_mode=input_mode,
            method=method,
            anatomy_mask=anatomy_mask,
            enhanced=was_enhanced,
        )

    return masks
