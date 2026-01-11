"""
Test script for registration functionality.

This script demonstrates the complete preprocessing + registration workflow
with napari visualization for before/after comparison and tuning.
"""

import numpy as np
import napari
from pygor.classes.core_data import Core


def test_registration(
    example_path: str = r"D:\Igor analyses\OSDS\251112 OSDS\1_0_SWN_200_White.smp",
    preprocess: bool = True,
    n_reference_frames: int = 100,
    batch_size: int = 100,
    upsample_factor: int = 10,
):
    """
    Test registration with napari visualization.

    Parameters
    ----------
    example_path : str
        Path to SMP file
    preprocess : bool
        Whether to preprocess before registration
    n_reference_frames : int
        Frames to average for reference
    batch_size : int
        Frames per batch for shift computation
    upsample_factor : int
        Subpixel precision factor

    Returns
    -------
    data : Core
        Core object with registered images
    stats : dict
        Registration statistics
    """
    print("=" * 70)
    print("REGISTRATION TEST WITH NAPARI VISUALIZATION")
    print("=" * 70)

    # Load and optionally preprocess
    print("\n1. Loading data...")
    data = Core.from_scanm(example_path, preprocess=False)
    print(f"   Loaded {data.images.shape[0]} frames")
    print(f"   Frame rate: {data.frame_hz:.2f} Hz")
    print(f"   Image shape: {data.images.shape[1:]} (height, width)")

    # Store original for comparison
    original_images = data.images.copy()

    if preprocess:
        print("\n2. Preprocessing...")
        data.preprocess(detrend=False)  # Skip detrend for speed
        print("   Preprocessing complete")
    else:
        print("\n2. Skipping preprocessing")

    # Store preprocessed (but not registered) for comparison
    preprocessed_images = data.images.copy()

    # Create reference image for visualization
    reference = preprocessed_images[:n_reference_frames].mean(axis=0)

    # Register with plot
    print("\n3. Running registration...")
    print(f"   n_reference_frames: {n_reference_frames}")
    print(f"   batch_size: {batch_size}")
    print(f"   upsample_factor: {upsample_factor}")

    stats = data.register(
        n_reference_frames=n_reference_frames,
        batch_size=batch_size,
        upsample_factor=upsample_factor,
        plot=True,  # Show matplotlib plot of shifts
    )

    # Print results
    print("\n" + "=" * 70)
    print("REGISTRATION RESULTS")
    print("=" * 70)
    print(f"Mean shift:     Y={stats['mean_shift'][0]:6.3f}, X={stats['mean_shift'][1]:6.3f} pixels")
    print(f"Std shift:      Y={stats['std_shift'][0]:6.3f}, X={stats['std_shift'][1]:6.3f} pixels")
    print(f"Max shift:      Y={stats['max_shift'][0]:6.3f}, X={stats['max_shift'][1]:6.3f} pixels")
    print(f"Mean error:     {stats['mean_error']:.6f} (lower is better)")
    print(f"Batches:        {len(stats['shifts'])}")

    # Compute improvement metrics
    print("\n4. Computing improvement metrics...")
    original_std = np.std(original_images, axis=0)
    preprocessed_std = np.std(preprocessed_images, axis=0)
    registered_std = np.std(data.images, axis=0)

    print(f"   Mean std (original):      {original_std.mean():.2f}")
    print(f"   Mean std (preprocessed):  {preprocessed_std.mean():.2f}")
    print(f"   Mean std (registered):    {registered_std.mean():.2f}")
    print(f"   Improvement vs preprocessed: {(preprocessed_std.mean() - registered_std.mean()):.2f}")

    # Napari visualization
    print("\n5. Launching napari viewer...")
    print("   Showing:")
    print("   - Original stack (if preprocessed)")
    print("   - Preprocessed stack")
    print("   - Registered stack")
    print("   - Reference image")
    print("   - Std projections (compare sharpness)")

    viewer = napari.Viewer(title="Registration Comparison")

    # Add stacks
    if preprocess:
        viewer.add_image(
            original_images,
            name='1. Original',
            colormap='gray',
            visible=False,
        )

    viewer.add_image(
        preprocessed_images,
        name='2. Preprocessed (before registration)',
        colormap='gray',
        contrast_limits=[preprocessed_images.min(), preprocessed_images.max()],
    )

    viewer.add_image(
        data.images,
        name='3. Registered',
        colormap='green',
        blending='additive',
        contrast_limits=[data.images.min(), data.images.max()],
    )

    # Add reference
    viewer.add_image(
        reference,
        name='Reference (target)',
        colormap='magma',
    )

    # Add std projections for quality assessment
    viewer.add_image(
        preprocessed_std,
        name='Std: Preprocessed',
        colormap='turbo',
    )

    viewer.add_image(
        registered_std,
        name='Std: Registered',
        colormap='turbo',
    )

    # Add difference to show improvement
    diff = preprocessed_std - registered_std
    viewer.add_image(
        diff,
        name='Std improvement (positive = better)',
        colormap='bwr',
        contrast_limits=[-diff.std(), diff.std()],
    )

    print("\n" + "=" * 70)
    print("NAPARI VISUALIZATION TIPS")
    print("=" * 70)
    print("- Toggle layers on/off to compare before/after")
    print("- Use 'Preprocessed' and 'Registered' together (additive blend)")
    print("  to see misalignment as colored fringes")
    print("- Compare 'Std: Preprocessed' vs 'Std: Registered'")
    print("  - Lower std in registered = sharper, better aligned")
    print("- 'Std improvement' shows where registration helped most")
    print("  - Red areas = registration made sharper")
    print("  - Blue areas = registration made worse (shouldn't happen much)")
    print("\nClose napari window to continue...")

    napari.run()

    return data, stats


def test_roi_transfer(
    path_a: str = r"D:\Igor analyses\OSDS\251112 OSDS\1_0_SWN_200_White.smp",
    path_b: str = r"D:\Igor analyses\OSDS\251112 OSDS\1_0_SWN_200_White.smp",
):
    """
    Test ROI transfer between recordings with napari visualization.

    Parameters
    ----------
    path_a : str
        Path to first recording (source)
    path_b : str
        Path to second recording (target)

    Returns
    -------
    data_a : Core
        First recording
    data_b : Core
        Second recording with transferred ROIs
    transform : dict
        Transform information
    """
    from pygor.preproc.registration import transfer_rois

    print("\n" + "=" * 70)
    print("ROI TRANSFER TEST WITH NAPARI VISUALIZATION")
    print("=" * 70)

    # Load recordings
    print("\n1. Loading recordings...")
    print(f"   Recording A: {path_a}")
    data_a = Core.from_scanm(path_a, preprocess=True)
    data_a.register()

    print(f"   Recording B: {path_b}")
    data_b = Core.from_scanm(path_b, preprocess=True)
    data_b.register()

    # Create test ROI mask
    print("\n2. Creating test ROI mask...")
    roi_mask = np.ones(data_a.average_stack.shape, dtype=np.int32)

    # Add multiple test ROIs (circles at different positions)
    y, x = np.ogrid[:roi_mask.shape[0], :roi_mask.shape[1]]
    cy, cx = roi_mask.shape[0] // 2, roi_mask.shape[1] // 2

    # ROI 1: Center
    r = 10
    mask = (y - cy)**2 + (x - cx)**2 <= r**2
    roi_mask[mask] = -1

    # ROI 2: Upper left
    cy2, cx2 = cy - 30, cx - 30
    mask2 = (y - cy2)**2 + (x - cx2)**2 <= r**2
    roi_mask[mask2] = -2

    # ROI 3: Lower right
    cy3, cx3 = cy + 30, cx + 30
    mask3 = (y - cy3)**2 + (x - cx3)**2 <= r**2
    roi_mask[mask3] = -3

    print(f"   Created {-roi_mask.min()} test ROIs")

    # Transfer ROIs
    print("\n3. Transferring ROIs from A to B...")
    shifted_rois, transform = transfer_rois(
        roi_mask=roi_mask,
        ref_projection=data_a.average_stack,
        target_projection=data_b.average_stack,
    )

    print(f"   Detected offset: Y={transform['shift'][0]:.3f}, X={transform['shift'][1]:.3f} pixels")
    print(f"   Registration error: {transform['error']:.6f}")

    # Napari visualization
    print("\n4. Launching napari viewer...")
    print("   Showing:")
    print("   - Recording A mean projection")
    print("   - Recording B mean projection")
    print("   - Original ROIs on A")
    print("   - Transferred ROIs on B")
    print("   - ROI overlay comparison")

    viewer = napari.Viewer(title="ROI Transfer Comparison")

    # Add mean projections
    viewer.add_image(
        data_a.average_stack,
        name='Recording A (source)',
        colormap='gray',
    )

    viewer.add_image(
        data_b.average_stack,
        name='Recording B (target)',
        colormap='gray',
    )

    # Add original ROIs
    viewer.add_labels(
        roi_mask,
        name='ROIs: Original (on A)',
        opacity=0.5,
    )

    # Add transferred ROIs
    viewer.add_labels(
        shifted_rois,
        name='ROIs: Transferred (on B)',
        opacity=0.5,
    )

    # Create overlay to show alignment
    # Make an RGB image showing A in red, B in green
    overlay = np.zeros((*data_a.average_stack.shape, 3), dtype=np.float32)
    # Normalize to 0-1
    a_norm = (data_a.average_stack - data_a.average_stack.min()) / (data_a.average_stack.max() - data_a.average_stack.min())
    b_norm = (data_b.average_stack - data_b.average_stack.min()) / (data_b.average_stack.max() - data_b.average_stack.min())
    overlay[..., 0] = a_norm  # Red channel
    overlay[..., 1] = b_norm  # Green channel
    # Where they align perfectly = yellow, misalignment = red or green fringes

    viewer.add_image(
        overlay,
        name='Overlay (A=red, B=green, aligned=yellow)',
        rgb=True,
        visible=False,
    )

    print("\n" + "=" * 70)
    print("NAPARI VISUALIZATION TIPS")
    print("=" * 70)
    print("- Toggle 'Recording A' and 'Recording B' to see differences")
    print("- Compare 'ROIs: Original' on Recording A")
    print("  with 'ROIs: Transferred' on Recording B")
    print("- Enable 'Overlay' layer to check alignment quality")
    print("  - Yellow = perfect alignment")
    print("  - Red/green fringes = misalignment")
    print("- If ROIs don't align well, the recordings may have shifted")
    print("\nClose napari window to continue...")

    napari.run()

    return data_a, data_b, transform


def main():
    """Run registration tests."""
    import argparse

    parser = argparse.ArgumentParser(description="Test registration functionality")
    parser.add_argument(
        "--path",
        type=str,
        default=r"D:\Igor analyses\OSDS\251112 OSDS\1_0_SWN_200_White.smp",
        help="Path to SMP file",
    )
    parser.add_argument(
        "--no-preprocess",
        action="store_true",
        help="Skip preprocessing step",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=10,
        help="Frames per batch (default: 10)",
    )
    parser.add_argument(
        "--upsample",
        type=int,
        default=10,
        help="Upsample factor (default: 10)",
    )
    parser.add_argument(
        "--test-roi-transfer",
        action="store_true",
        help="Run ROI transfer test instead",
    )

    args = parser.parse_args()

    try:
        if args.test_roi_transfer:
            # Test ROI transfer
            test_roi_transfer(path_a=args.path, path_b=args.path)
        else:
            # Test registration
            data, stats = test_registration(
                example_path=args.path,
                preprocess=not args.no_preprocess,
                batch_size=args.batch_size,
                upsample_factor=args.upsample,
            )

            print("\n" + "=" * 70)
            print("TEST COMPLETE")
            print("=" * 70)
            print("\nData is available as 'data' variable if running interactively")
            print("Registration stats available as 'stats' variable")

    except FileNotFoundError:
        print("\nERROR: Test data file not found.")
        print(f"Please check the path: {args.path}")
        print("\nUsage: python test_registration.py --path <path_to_smp_file>")
    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
