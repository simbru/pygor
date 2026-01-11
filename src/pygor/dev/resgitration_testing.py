import pygor
import pygor.preproc as preproc
import timeit
import napari
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import shift, gaussian_filter

"""
Template matching registration (IGOR approach).
Tests all shifts in a window, picks the one that minimizes difference.
"""

def find_best_shift_bruteforce(reference, target, max_shift=2, exclude_artifact_width=0):
    """
    Brute-force search for best shift by minimizing std of difference.
    This is the IGOR approach - simple but robust.
    
    Args:
        reference: Reference image
        target: Image to register
        max_shift: Maximum shift to test in pixels (tests ±max_shift)
        exclude_artifact_width: Pixels to exclude from left edge
    
    Returns:
        (shift_y, shift_x, min_std): Best shift and resulting std
    """
    ny, nx = reference.shape
    best_shift = (0, 0)
    best_std = float('inf')
    
    # Create valid region mask (exclude edges and artifact)
    valid_y_start = max_shift
    valid_y_end = ny - max_shift
    valid_x_start = max_shift + exclude_artifact_width
    valid_x_end = nx - max_shift
    
    # Test all shifts in the window
    for dy in range(-max_shift, max_shift + 1):
        for dx in range(-max_shift, max_shift + 1):
            # Extract overlapping regions
            ref_region = reference[valid_y_start:valid_y_end, valid_x_start:valid_x_end]
            target_region = target[valid_y_start + dy:valid_y_end + dy, 
                                  valid_x_start + dx:valid_x_end + dx]
            
            # Compute difference
            diff = ref_region - target_region
            std = np.nanstd(diff)  # Use nanstd to ignore NaNs
            
            if std < best_std:
                best_std = std
                best_shift = (dy, dx)
    
    return best_shift[0], best_shift[1], best_std

def apply_shift_to_frame(frame, shift_yx, original_min, original_max, artifact_width=0):
    """Apply shift to frame and handle artifact region."""
    shifted_frame = shift(frame, shift=shift_yx, order=1, mode='reflect')
    shifted_frame = np.clip(shifted_frame, original_min, original_max)
    
    # Fill artifact region with mean of rest of image
    if artifact_width > 0:
        shifted_frame[:, :artifact_width] = np.nan
        mean_val = np.nanmean(shifted_frame)
        shifted_frame[:, :artifact_width] = mean_val
    
    return shifted_frame

def main():
    # start timeit
    time_start = timeit.default_timer()
    example_path = r"D:\Igor analyses\OSDS\251112 OSDS\1_0_SWN_200_White.smp"
    print("Loading ScanM file...")
    from pygor.classes.core_data import Core
    
    data = Core.from_scanm(example_path)
    data.preprocess(detrend=False)
    image_stack = data.images[:, :, 2:]  # shape (time, height, width)
    
    # ===== REGISTRATION PARAMETERS (IGOR-style) =====
    n_reference_frames = 500   # Average many frames for stable reference
    skip_frames = 200          # Register every N frames (skipN in IGOR)
    average_frames = 100       # Average N frames for each registration point
    max_drift = 5              # Maximum shift to search (pixels)
    gaussian_sigma = 1.5       # Smoothing (GaussFilter in IGOR)
    artifact_width = 2         # Light artifact pixels to exclude from left edge
    # ================================================
    
    original_min, original_max = image_stack.min(), image_stack.max()
    n_frames = len(image_stack)
    
    # Apply Gaussian smoothing to entire stack
    print(f"Applying Gaussian filter (sigma={gaussian_sigma})...")
    smoothed_stack = np.zeros_like(image_stack)
    for i in range(n_frames):
        smoothed_stack[i] = gaussian_filter(image_stack[i], sigma=gaussian_sigma)
    
    # Create reference from initial frames
    print(f"Creating reference from first {n_reference_frames} frames...")
    reference = smoothed_stack[:n_reference_frames].mean(axis=0)
    
    # Mask artifact region
    reference[:, :artifact_width] = np.nan
    
    print(f"\nRegistration settings:")
    print(f"  Stack: {image_stack.shape}")
    print(f"  Skip: {skip_frames} frames")
    print(f"  Average: {average_frames} frames per registration")
    print(f"  Max drift: ±{max_drift} pixels")
    print(f"  Artifact width: {artifact_width} pixels")
    
    # Compute shifts using IGOR approach
    n_registration_points = int(np.ceil(n_frames / skip_frames))
    shifts = np.zeros((n_registration_points, 2))
    stds = []
    
    print(f"\nComputing shifts for {n_registration_points} registration points...")
    for i in range(n_registration_points):
        start_idx = i * skip_frames
        end_idx = min(start_idx + average_frames, n_frames)
        
        if end_idx >= n_frames:
            end_idx = n_frames
            start_idx = max(0, n_frames - average_frames)
        
        # Average frames for this registration point
        target = smoothed_stack[start_idx:end_idx].mean(axis=0)
        target[:, :artifact_width] = np.nan
        
        # Find best shift RELATIVE TO REFERENCE (not cumulative)
        dy, dx, min_std = find_best_shift_bruteforce(
            reference, target, max_shift=max_drift, exclude_artifact_width=artifact_width
        )
        
        shifts[i] = [dy, dx]
        stds.append(min_std)
        
        if i % 10 == 0 or i < 3:
            print(f"  Point {i+1}/{n_registration_points} (frames {start_idx:5d}-{end_idx:5d}): "
                  f"shift=[{dy}, {dx}], std={min_std:.2f}")
    
    print(f"\nApplying shifts to full stack...")
    registered_stack = np.zeros_like(image_stack)
    
    # Interpolate shifts for all frames
    all_shifts = np.zeros((n_frames, 2))
    for i in range(n_frames):
        reg_idx = min(i // skip_frames, len(shifts) - 1)
        all_shifts[i] = shifts[reg_idx]
    
    # Apply shifts
    for i in range(n_frames):
        registered_stack[i] = apply_shift_to_frame(
            image_stack[i], all_shifts[i], original_min, original_max, artifact_width
        )
        if i % 5000 == 0:
            print(f"  Applied shift to frame {i}/{n_frames}")
    
    stds = np.array(stds)
    print(f"\nRegistration complete!")
    print(f"  Mean shift: Y={shifts[:, 0].mean():.2f}, X={shifts[:, 1].mean():.2f} pixels")
    print(f"  Shift range: Y=[{shifts[:, 0].min():.1f}, {shifts[:, 0].max():.1f}], X=[{shifts[:, 1].min():.1f}, {shifts[:, 1].max():.1f}]")
    print(f"  Mean std: {stds.mean():.2f}")
    print(f"  Time: {timeit.default_timer() - time_start:.1f}s")
    
    # Plot
    fig, axes = plt.subplots(2, 1, figsize=(10, 6))
    reg_indices = np.arange(len(shifts))
    
    axes[0].plot(reg_indices, shifts[:, 0], 'b.-', label='Y shift')
    axes[0].plot(reg_indices, shifts[:, 1], 'r.-', label='X shift')
    axes[0].set_ylabel('Shift (pixels)')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    axes[0].set_title('IGOR-style Template Matching Registration')
    
    axes[1].plot(reg_indices, stds, 'k.-')
    axes[1].set_xlabel('Registration point')
    axes[1].set_ylabel('Std of difference')
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # Visualize
    viewer = napari.Viewer()
    viewer.add_image(image_stack, name='Original')
    viewer.add_image(registered_stack, name='Registered')
    viewer.add_image(reference, name='Reference')
    napari.run()


if __name__ == "__main__":
    main()
