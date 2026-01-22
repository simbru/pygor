"""
Blob detection segmentation using Difference of Gaussian (DoG).

Designed for detecting synaptic terminals and other blob-like structures.
"""

import numpy as np
import warnings

from skimage.feature import blob_dog

from pygor.segmentation.preprocessing import (
    normalize,
    enhance_unsharp,
    create_anatomy_mask,
    filter_points_by_mask,
    prepare_image,
)


def _detect_blobs(img, min_sigma, max_sigma, threshold, overlap, artifact_width=0):
    """
    Detect blobs using DoG and convert sigma to radius.

    Returns array of (y, x, radius) for each blob.
    """
    blobs = blob_dog(img, min_sigma=min_sigma, max_sigma=max_sigma,
                     threshold=threshold, overlap=overlap)

    if len(blobs) == 0:
        return blobs

    # DoG returns sigma, convert to radius: radius = sqrt(2) * sigma
    blobs = blobs.copy()
    blobs[:, 2] *= np.sqrt(2)

    # Filter out blobs in artifact region
    if artifact_width > 0:
        blobs = blobs[blobs[:, 1] >= artifact_width]

    return blobs


def _blobs_to_masks(blobs, img_shape, radius_multiplier=1.0, min_radius=2, merge_overlap=0.3):
    """
    Convert blob detections to labeled mask array, merging overlapping blobs.

    Uses union-find for connected component grouping after drawing
    all circles, then merges components based on overlap threshold.
    """
    if len(blobs) == 0:
        return np.zeros(img_shape, dtype=np.int32)

    centers = blobs[:, :2]
    radii = np.maximum(min_radius, radius_multiplier * blobs[:, 2])

    # Create distance grids once
    yy, xx = np.ogrid[:img_shape[0], :img_shape[1]]

    # Draw each blob as a separate temporary mask, then merge based on overlap
    n_blobs = len(blobs)
    blob_masks = np.zeros((n_blobs, *img_shape), dtype=bool)

    for i, ((cy, cx), r) in enumerate(zip(centers, radii)):
        blob_masks[i] = (yy - cy)**2 + (xx - cx)**2 <= r**2

    # Find which blobs should merge (union-find via adjacency)
    parent = np.arange(n_blobs)

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]  # Path compression
            i = parent[i]
        return i

    # Check overlap between blob pairs
    for i in range(n_blobs):
        for j in range(i + 1, n_blobs):
            intersection = blob_masks[i] & blob_masks[j]
            if not intersection.any():
                continue

            overlap_pixels = intersection.sum()
            smaller_area = min(blob_masks[i].sum(), blob_masks[j].sum())

            if overlap_pixels / smaller_area >= merge_overlap:
                parent[find(i)] = find(j)

    # Build final masks by group
    masks = np.zeros(img_shape, dtype=np.int32)
    group_labels = {}
    current_label = 0

    for i in range(n_blobs):
        root = find(i)
        if root not in group_labels:
            current_label += 1
            group_labels[root] = current_label
        masks[blob_masks[i]] = group_labels[root]

    return masks


def segment(
    image_or_data,
    *,
    method="blob",
    input_mode="combined",
    postprocess=False,
    verbose=True,
    plot=False,
    # Image enhancement
    unsharp_radius=1.0,
    unsharp_amount=2.5,
    # Blob detection
    min_sigma=1,
    max_sigma=2,
    threshold=0.01,
    eliminate_overlap=1.0,
    merge_overlap=0.6,
    # Anatomy mask (exclude border regions)
    anatomy_threshold='otsu',
    anatomy_thresh_mult=0.2,
    erode_iterations=1,
    # Mask creation
    radius_multiplier=1.5,
    min_radius=1,
    **kwargs,
):
    """
    Segment an image using Difference of Gaussian (DoG) blob detection.

    Designed for detecting synaptic terminals and other blob-like structures.

    Parameters
    ----------
    image_or_data : ndarray or Core/STRF
        Either a 2D image array or a pygor data object.
        If a data object, the image is extracted based on input_mode and
        artifact masking is applied automatically.
    method : str
        Kept for API consistency (always "blob")
    input_mode : str
        For data objects: which image to use ("correlation", "average", "combined")
    postprocess : bool
        Not used for blob methods (kept for API consistency)
    verbose : bool
        Print progress messages
    plot : bool
        If True, display a figure showing the input image, ROI masks, and
        overlay. Shows the actual image representation used (input_mode).

    Image Enhancement Parameters
    ----------------------------
    unsharp_radius : float
        Radius for unsharp masking (0.5-2.0 typical)
    unsharp_amount : float
        Strength of sharpening (1.0-5.0 typical)

    Blob Detection Parameters
    -------------------------
    min_sigma : float
        Minimum sigma for DoG (controls minimum blob size)
    max_sigma : float
        Maximum sigma for DoG (controls maximum blob size)
    threshold : float
        Detection threshold (lower = catch weaker blobs)
    eliminate_overlap : float
        Fraction overlap before DoG merges blobs (1.0 = no merging in DoG)
    merge_overlap : float
        Merge blobs if overlap > this fraction of smaller blob (1.0 to disable)

    Anatomy Mask Parameters
    -----------------------
    anatomy_threshold : str or float
        'otsu' for automatic, or float for manual threshold
    anatomy_thresh_mult : float
        Multiply threshold by this (lower = more permissive)
    erode_iterations : int
        Push boundary inward to exclude edge blobs (0 to disable)

    Mask Creation Parameters
    ------------------------
    radius_multiplier : float
        Scale factor for blob radius when creating masks
    min_radius : float
        Minimum radius in pixels (prevents single-pixel ROIs)

    **kwargs
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
    >>> obj.segment_rois(mode="blob", input_mode="combined")

    >>> # With raw image
    >>> from pygor.segmentation.blob import segment
    >>> masks = segment(my_image, threshold=0.02, min_sigma=1.5)
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
        artifact_fill_width = artifact_width + 1 if artifact_width > 0 else 0
        if artifact_width > 0:
            img[:, :artifact_fill_width] = 0
            if verbose:
                print(f"  Masked artifact region: columns 0-{artifact_fill_width - 1}")

    # Apply unsharp masking for enhancement
    if verbose:
        print(f"Enhancing image (unsharp: r={unsharp_radius}, a={unsharp_amount})...")
    img = enhance_unsharp(img, radius=unsharp_radius, amount=unsharp_amount)

    # Create anatomy mask to exclude border regions
    if verbose:
        print(f"Creating anatomy mask (threshold={anatomy_threshold}, mult={anatomy_thresh_mult})...")
    anatomy_mask = create_anatomy_mask(
        img,
        method=anatomy_threshold,
        thresh_mult=anatomy_thresh_mult,
        erode_iterations=erode_iterations
    )

    # Detect blobs
    if verbose:
        print(f"Detecting blobs (sigma={min_sigma}-{max_sigma}, thresh={threshold})...")
    blobs_raw = _detect_blobs(
        img,
        min_sigma=min_sigma,
        max_sigma=max_sigma,
        threshold=threshold,
        overlap=eliminate_overlap,
        artifact_width=artifact_fill_width
    )

    # Filter by anatomy mask
    blobs = filter_points_by_mask(blobs_raw, anatomy_mask)

    if verbose:
        n_filtered = len(blobs_raw) - len(blobs)
        print(f"  Detected {len(blobs_raw)} blobs, filtered {n_filtered} border blobs -> {len(blobs)} remaining")

    # Convert to masks
    masks = _blobs_to_masks(
        blobs,
        img.shape,
        radius_multiplier=radius_multiplier,
        min_radius=min_radius,
        merge_overlap=merge_overlap
    )

    # Warn about unused kwargs
    if kwargs:
        warnings.warn(f"Unused parameters: {list(kwargs.keys())}")

    n_rois = masks.max()
    if verbose:
        print(f"  Found {n_rois} ROIs")

    # Plot if requested
    if plot:
        from pygor.segmentation.plotting import plot_segmentation
        plot_segmentation(img, masks, input_mode=input_mode, method="blob")

    return masks
