"""
Diagnostic script to compare IGOR vs Python detrending parameters.

This script helps identify why the detrending outputs differ.
"""

import numpy as np
import tifffile
from pygor.classes.core_data import Core

def compare_detrend_params():
    """Compare detrending parameters between IGOR and Python."""

    example_path = r"D:\Igor analyses\OSDS\251112 OSDS\0_1_gradient_contrast_400_white.smh"

    # Load without preprocessing
    data = Core.from_scanm(example_path)

    print("=" * 60)
    print("PARAMETER COMPARISON: IGOR vs Python")
    print("=" * 60)

    # Get dimensions
    n_frames, n_lines, n_width = data.images.shape
    print(f"\nData dimensions:")
    print(f"  Frames (nF): {n_frames}")
    print(f"  Lines (nY):  {n_lines}")
    print(f"  Width (nX):  {n_width}")

    # IGOR's assumed parameters (hardcoded in OS_DetrendStack.ipf)
    igor_line_duration = 0.002  # 2ms - HARDCODED in IGOR
    igor_frame_rate = 1 / (n_lines * igor_line_duration)

    # Python's actual parameters (from header)
    python_line_duration = data.linedur_s
    python_frame_rate = data.frame_hz

    print(f"\n--- Line Duration ---")
    print(f"  IGOR (hardcoded):   {igor_line_duration * 1000:.3f} ms")
    print(f"  Python (from header): {python_line_duration * 1000:.3f} ms")
    print(f"  Ratio (Python/IGOR):  {python_line_duration / igor_line_duration:.4f}")

    print(f"\n--- Frame Rate ---")
    print(f"  IGOR (calculated):  {igor_frame_rate:.4f} Hz")
    print(f"  Python (from header): {python_frame_rate:.4f} Hz")
    print(f"  Ratio (Python/IGOR):  {python_frame_rate / igor_frame_rate:.4f}")

    # Detrending parameters
    smooth_window_s = 1000.0
    time_bin = 10

    # IGOR's smoothing factor
    igor_smooth_factor = (igor_frame_rate * smooth_window_s) / time_bin
    igor_smooth_factor = min(igor_smooth_factor, 2**15 - 1)  # IGOR limit

    # Python's smoothing factor
    python_smooth_factor = int(python_frame_rate * smooth_window_s / time_bin)
    python_smooth_factor = min(python_smooth_factor, 2**15 - 1)
    python_smooth_factor = max(python_smooth_factor, 1)

    print(f"\n--- Smoothing Factor (smooth_window_s={smooth_window_s}, time_bin={time_bin}) ---")
    print(f"  IGOR:   {igor_smooth_factor:.1f} frames (in binned space)")
    print(f"  Python: {python_smooth_factor} frames (in binned space)")
    print(f"  Difference: {abs(igor_smooth_factor - python_smooth_factor):.1f} frames")

    # Binned frames
    n_binned_frames = int(np.ceil(n_frames / time_bin))
    print(f"\n--- Binned Stack ---")
    print(f"  Original frames: {n_frames}")
    print(f"  Binned frames:   {n_binned_frames}")
    print(f"  Smooth window / binned frames: {igor_smooth_factor / n_binned_frames:.2f}")

    print("\n" + "=" * 60)
    print("DIAGNOSIS")
    print("=" * 60)

    if abs(python_line_duration - igor_line_duration) > 0.0001:
        print("\n[!] Line duration MISMATCH detected!")
        print("    IGOR uses hardcoded 2ms, Python uses actual value from header.")
        print("    This causes different smoothing factor calculations.")
        print()
        print("    FIX: Use IGOR's hardcoded 2ms line duration for frame rate")
        print("         calculation in the detrending smoothing factor.")
    else:
        print("\n[OK] Line duration matches.")

    # Additional check: what is the actual pixel duration?
    header = data.metadata.get('scanm_header', {})
    pixel_duration_us = header.get('RealPixelDuration_µs',
                                   header.get('TargetedPixelDuration_µs', 'unknown'))
    frame_width_raw = header.get('FrameWidth', 'unknown')

    print(f"\n--- Raw Header Values ---")
    print(f"  RealPixelDuration_µs: {pixel_duration_us}")
    print(f"  FrameWidth (raw): {frame_width_raw}")
    if isinstance(pixel_duration_us, (int, float)) and isinstance(frame_width_raw, int):
        actual_line_dur = frame_width_raw * pixel_duration_us * 1e-6
        print(f"  Calculated line duration: {actual_line_dur * 1000:.3f} ms")
        print(f"  IGOR's assumed line duration: 2.000 ms")

    return {
        'igor_frame_rate': igor_frame_rate,
        'python_frame_rate': python_frame_rate,
        'igor_smooth_factor': igor_smooth_factor,
        'python_smooth_factor': python_smooth_factor,
    }


def test_with_igor_framerate():
    """Test detrending using IGOR's hardcoded frame rate calculation."""

    example_path = r"D:\Igor analyses\OSDS\251112 OSDS\0_1_gradient_contrast_400_white.smh"
    igor_detrended_path = r"D:\Igor analyses\OSDS\251112 OSDS\wDataCh0_detrended.tif"

    # Load raw data
    data = Core.from_scanm(example_path)

    # Get dimensions
    n_frames, n_lines, n_width = data.images.shape

    # Calculate IGOR's assumed frame rate
    igor_line_duration = 0.002  # 2ms hardcoded
    igor_frame_rate = 1 / (n_lines * igor_line_duration)

    print(f"\nTesting with IGOR's assumed frame rate: {igor_frame_rate:.4f} Hz")
    print(f"  (vs Python's actual frame rate: {data.frame_hz:.4f} Hz)")

    # Import preprocessing functions
    from pygor.preproc.scanm import preprocess_stack

    # Preprocess using IGOR's frame rate
    preprocessed = preprocess_stack(
        data.images,
        frame_rate=igor_frame_rate,  # Use IGOR's rate
        artifact_width=2,
        flip_x=True,
        detrend=True,
        smooth_window_s=1000.0,
        time_bin=10,
        fix_first_frame=True,
    )

    # Load IGOR's result
    igor_detrended = tifffile.imread(igor_detrended_path)
    print(f"\nIGOR detrended shape: {igor_detrended.shape}")
    print(f"Python detrended shape: {preprocessed.shape}")

    # Handle frame count mismatch - truncate to smaller
    min_frames = min(preprocessed.shape[0], igor_detrended.shape[0])
    print(f"\nUsing first {min_frames} frames for comparison")
    preprocessed_cmp = preprocessed[:min_frames]
    igor_cmp = igor_detrended[:min_frames]

    # Compare
    diff = preprocessed_cmp - igor_cmp
    print(f"\n--- Comparison (using IGOR frame rate = {igor_frame_rate:.4f} Hz) ---")
    print(f"  Max absolute difference: {np.abs(diff).max():.2f}")
    print(f"  Mean absolute difference: {np.abs(diff).mean():.4f}")
    print(f"  Mean signed difference: {diff.mean():.4f}")
    print(f"  Std of difference: {diff.std():.4f}")

    # Percentage match
    for threshold in [1.0, 0.5, 0.1]:
        match_pct = (np.abs(diff) < threshold).mean() * 100
        print(f"  Pixels within ±{threshold}: {match_pct:.2f}%")

    return preprocessed_cmp, igor_cmp, diff


def check_raw_data_match():
    """Check if the raw (pre-detrend) data matches what IGOR started with."""

    example_path = r"D:\Igor analyses\OSDS\251112 OSDS\0_1_gradient_contrast_400_white.smh"
    igor_detrended_path = r"D:\Igor analyses\OSDS\251112 OSDS\wDataCh0_detrended.tif"

    # First, check what files exist in that directory
    import os
    dir_path = r"D:\Igor analyses\OSDS\251112 OSDS"
    print(f"\n--- Files in directory ---")
    tif_files = [f for f in os.listdir(dir_path) if f.endswith('.tif')]
    for f in tif_files:
        full_path = os.path.join(dir_path, f)
        try:
            img = tifffile.imread(full_path)
            print(f"  {f}: shape={img.shape}")
        except Exception as e:
            print(f"  {f}: error reading - {e}")

    # Load raw data
    data = Core.from_scanm(example_path)

    # Apply only x-flip and artifact fix (no detrend) to compare intermediate state
    from pygor.preproc.scanm import fix_light_artifact, fill_light_artifact

    flipped, stack_ave = fix_light_artifact(data.images, artifact_width=2, flip_x=True)

    # Load IGOR result
    igor_detrended = tifffile.imread(igor_detrended_path)

    print(f"\n--- Frame Count Analysis ---")
    print(f"  Python raw frames: {data.images.shape[0]}")
    print(f"  IGOR detrended frames: {igor_detrended.shape[0]}")
    print(f"  Difference: {data.images.shape[0] - igor_detrended.shape[0]} frames")

    # Check if IGOR file might be from a different recording
    ratio = igor_detrended.shape[0] / data.images.shape[0]
    print(f"  Ratio: {ratio:.4f}")

    # Compare statistics of the data
    min_frames = min(flipped.shape[0], igor_detrended.shape[0])

    print(f"\n--- Data Statistics (first {min_frames} frames) ---")
    print(f"  Python x-flipped mean: {flipped[:min_frames].mean():.2f}")
    print(f"  Python x-flipped std:  {flipped[:min_frames].std():.2f}")
    print(f"  Python x-flipped min:  {flipped[:min_frames].min():.2f}")
    print(f"  Python x-flipped max:  {flipped[:min_frames].max():.2f}")
    print()
    print(f"  IGOR detrended mean: {igor_detrended[:min_frames].mean():.2f}")
    print(f"  IGOR detrended std:  {igor_detrended[:min_frames].std():.2f}")
    print(f"  IGOR detrended min:  {igor_detrended[:min_frames].min():.2f}")
    print(f"  IGOR detrended max:  {igor_detrended[:min_frames].max():.2f}")

    # Check a specific frame
    frame_idx = 100
    print(f"\n--- Frame {frame_idx} comparison ---")
    print(f"  Python frame mean: {flipped[frame_idx].mean():.2f}")
    print(f"  IGOR frame mean:   {igor_detrended[frame_idx].mean():.2f}")

    # Check if values are in similar ranges
    print(f"\n--- Value Range Analysis ---")
    print(f"  Are both uint16 range? Python: {flipped.min():.0f}-{flipped.max():.0f}, IGOR: {igor_detrended.min()}-{igor_detrended.max()}")


def test_with_binomial_smoothing():
    """Test with IGOR's binomial (Gaussian) smoothing instead of boxcar."""

    example_path = r"D:\Igor analyses\OSDS\251112 OSDS\0_1_gradient_contrast_400_white.smh"
    igor_detrended_path = r"D:\Igor analyses\OSDS\251112 OSDS\wDataCh0_detrended.tif"

    from pygor.preproc.scanm import fix_light_artifact, fill_light_artifact
    from scipy.ndimage import gaussian_filter1d

    data = Core.from_scanm(example_path)
    igor_detrended = tifffile.imread(igor_detrended_path)

    n_frames, n_lines, n_width = data.images.shape

    # IGOR parameters
    time_bin = 10
    igor_smooth_iterations = 781  # Number of binomial smoothing iterations

    # For binomial smoothing, the relationship to Gaussian sigma is:
    # sigma = sqrt(num / 2) where num is the number of iterations
    # Reference: IGOR docs and Marchand & Marmet 1983
    sigma = np.sqrt(igor_smooth_iterations / 2)

    print(f"\n--- Testing with BINOMIAL (Gaussian) smoothing ---")
    print(f"  IGOR smoothing iterations: {igor_smooth_iterations}")
    print(f"  Equivalent Gaussian sigma: {sigma:.2f}")

    # Step 1: X-flip
    result, stack_ave = fix_light_artifact(data.images, artifact_width=2, flip_x=True)

    # Step 2: Detrend with binomial smoothing
    pixel_mean = stack_ave

    # Temporal binning (IGOR subsamples)
    binned_stack = result[::time_bin, :, :]
    print(f"  Binned stack shape: {binned_stack.shape}")

    # Use Gaussian smoothing with 'reflect' mode (IGOR's bounce/default)
    smoothed = gaussian_filter1d(binned_stack.astype(np.float32),
                                  sigma=sigma, axis=0, mode='reflect')

    # Upsample
    baseline = np.repeat(smoothed, time_bin, axis=0)[:n_frames]

    # Apply detrending formula
    result = result.astype(np.float32) - baseline + pixel_mean

    # Step 3: Fix first frame
    result[0] = result[1]

    # Step 4: Fill artifact
    result = fill_light_artifact(result, stack_ave, artifact_width=2)

    # Compare
    diff = result - igor_detrended
    print(f"\n--- Comparison with BINOMIAL smoothing ---")
    print(f"  Max absolute difference: {np.abs(diff).max():.2f}")
    print(f"  Mean absolute difference: {np.abs(diff).mean():.4f}")
    print(f"  Mean signed difference: {diff.mean():.4f}")

    # Non-artifact region
    non_artifact_diff = diff[:, :, 3:]
    print(f"\n--- Non-Artifact Region ---")
    print(f"  Max abs diff: {np.abs(non_artifact_diff).max():.2f}")
    print(f"  Mean diff: {non_artifact_diff.mean():.4f}")
    for threshold in [100.0, 50.0, 10.0, 5.0, 1.0, 0.5]:
        match_pct = (np.abs(non_artifact_diff) < threshold).mean() * 100
        print(f"  Pixels within ±{threshold}: {match_pct:.2f}%")

    # Check negative values
    negative_count = (result < 0).sum()
    print(f"\n--- Negative value check ---")
    print(f"  Negative values in Python result: {negative_count}")
    print(f"  Min value in Python result: {result.min():.2f}")
    print(f"  Min value in IGOR result: {igor_detrended.min()}")

    # Find location of max difference
    max_diff_idx = np.unravel_index(np.abs(diff).argmax(), diff.shape)
    print(f"\n--- Largest Difference Location ---")
    print(f"  Location (frame, line, pixel): {max_diff_idx}")
    print(f"  Python value: {result[max_diff_idx]:.2f}")
    print(f"  IGOR value: {igor_detrended[max_diff_idx]}")

    return result, igor_detrended, diff


def detailed_pixel_comparison():
    """Compare specific pixels and frames to understand the difference pattern."""

    example_path = r"D:\Igor analyses\OSDS\251112 OSDS\0_1_gradient_contrast_400_white.smh"
    igor_detrended_path = r"D:\Igor analyses\OSDS\251112 OSDS\wDataCh0_detrended.tif"

    from pygor.preproc.scanm import preprocess_stack

    data = Core.from_scanm(example_path)
    igor_detrended = tifffile.imread(igor_detrended_path)

    # Get IGOR's assumed frame rate
    n_lines = data.images.shape[1]
    igor_frame_rate = 1 / (n_lines * 0.002)

    # Preprocess with IGOR frame rate
    preprocessed = preprocess_stack(
        data.images,
        frame_rate=igor_frame_rate,
        artifact_width=2,
        flip_x=True,
        detrend=True,
        smooth_window_s=1000.0,
        time_bin=10,
        fix_first_frame=True,
    )

    diff = preprocessed - igor_detrended

    # Find where the largest differences are
    max_diff_idx = np.unravel_index(np.abs(diff).argmax(), diff.shape)
    print(f"\n--- Largest Difference Location ---")
    print(f"  Location (frame, line, pixel): {max_diff_idx}")
    print(f"  Python value: {preprocessed[max_diff_idx]:.2f}")
    print(f"  IGOR value: {igor_detrended[max_diff_idx]:.2f}")
    print(f"  Difference: {diff[max_diff_idx]:.2f}")

    # Check the artifact region specifically
    print(f"\n--- Artifact Region (first 3 pixels) ---")
    artifact_diff = diff[:, :, :3]
    print(f"  Mean diff in artifact region: {artifact_diff.mean():.2f}")
    print(f"  Max diff in artifact region: {np.abs(artifact_diff).max():.2f}")

    # Check non-artifact region
    non_artifact_diff = diff[:, :, 3:]
    print(f"\n--- Non-Artifact Region (pixels 3+) ---")
    print(f"  Mean diff: {non_artifact_diff.mean():.4f}")
    print(f"  Std diff: {non_artifact_diff.std():.4f}")
    print(f"  Max abs diff: {np.abs(non_artifact_diff).max():.2f}")

    for threshold in [100.0, 50.0, 10.0, 5.0, 1.0, 0.5]:
        match_pct = (np.abs(non_artifact_diff) < threshold).mean() * 100
        print(f"  Pixels within ±{threshold}: {match_pct:.2f}%")

    # Check frame 0 (first frame fix)
    print(f"\n--- Frame 0 vs Frame 1 ---")
    print(f"  Python frame 0 mean: {preprocessed[0].mean():.2f}")
    print(f"  Python frame 1 mean: {preprocessed[1].mean():.2f}")
    print(f"  IGOR frame 0 mean: {igor_detrended[0].mean():.2f}")
    print(f"  IGOR frame 1 mean: {igor_detrended[1].mean():.2f}")
    print(f"  Python f0==f1? {np.allclose(preprocessed[0], preprocessed[1])}")
    print(f"  IGOR f0==f1? {np.allclose(igor_detrended[0], igor_detrended[1])}")

    # Sample a middle frame
    mid_frame = 1000
    print(f"\n--- Frame {mid_frame} Sample ---")
    print(f"  Python: min={preprocessed[mid_frame].min():.0f}, max={preprocessed[mid_frame].max():.0f}, mean={preprocessed[mid_frame].mean():.2f}")
    print(f"  IGOR:   min={igor_detrended[mid_frame].min():.0f}, max={igor_detrended[mid_frame].max():.0f}, mean={igor_detrended[mid_frame].mean():.2f}")

    # Check if difference is constant offset or varies
    frame_mean_diffs = []
    for i in range(0, min(1000, preprocessed.shape[0]), 100):
        frame_diff = (preprocessed[i, :, 3:] - igor_detrended[i, :, 3:]).mean()
        frame_mean_diffs.append(frame_diff)
    print(f"\n--- Mean Difference Per Frame (non-artifact, every 100 frames) ---")
    print(f"  Frame diffs: {[f'{d:.2f}' for d in frame_mean_diffs]}")
    print(f"  Std of frame diffs: {np.std(frame_mean_diffs):.4f}")


if __name__ == "__main__":
    params = compare_detrend_params()

    print("\n" + "=" * 60)
    print("CHECKING RAW DATA")
    print("=" * 60)
    check_raw_data_match()

    print("\n" + "=" * 60)
    print("TESTING WITH IGOR FRAME RATE")
    print("=" * 60)
    preprocessed, igor, diff = test_with_igor_framerate()

    print("\n" + "=" * 60)
    print("TESTING WITH BINOMIAL (GAUSSIAN) SMOOTHING")
    print("=" * 60)
    test_with_binomial_smoothing()

    print("\n" + "=" * 60)
    print("DETAILED PIXEL COMPARISON")
    print("=" * 60)
    detailed_pixel_comparison()
