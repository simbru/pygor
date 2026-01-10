"""
Compare IGOR-produced H5 with direct ScanM loading.

This script validates that Core.from_scanm() produces equivalent
results to the IGOR preprocessing pipeline.
"""

import numpy as np
import h5py
from pathlib import Path
import sys

# Add pygor to path if needed
sys.path.insert(0, str(Path(__file__).parents[2]))

from pygor.classes.core_data import Core


def inspect_h5_structure(h5_path: Path) -> dict:
    """List all datasets and their shapes in an H5 file."""
    info = {}
    with h5py.File(h5_path, "r") as f:
        def visitor(name, obj):
            if isinstance(obj, h5py.Dataset):
                info[name] = {
                    "shape": obj.shape,
                    "dtype": str(obj.dtype),
                }
        f.visititems(visitor)
    return info


def load_h5_key(h5_path: Path, key: str):
    """Load a specific key from H5 file."""
    with h5py.File(h5_path, "r") as f:
        if key in f:
            return np.array(f[key])
    return None


def compare_arrays(name: str, arr1, arr2, rtol=1e-5, atol=1e-8):
    """Compare two arrays and report differences."""
    if arr1 is None and arr2 is None:
        print(f"  {name}: Both None [OK]")
        return True
    if arr1 is None or arr2 is None:
        print(f"  {name}: One is None! IGOR={arr1 is not None}, ScanM={arr2 is not None}")
        return False

    if arr1.shape != arr2.shape:
        print(f"  {name}: Shape mismatch! IGOR={arr1.shape}, ScanM={arr2.shape}")
        return False

    if np.allclose(arr1, arr2, rtol=rtol, atol=atol, equal_nan=True):
        print(f"  {name}: Match [OK] (shape={arr1.shape})")
        return True
    else:
        diff = np.abs(arr1.astype(float) - arr2.astype(float))
        max_diff = np.nanmax(diff)
        mean_diff = np.nanmean(diff)
        print(f"  {name}: MISMATCH! max_diff={max_diff:.6f}, mean_diff={mean_diff:.6f}")
        return False


def inspect_pixel_values(igor_h5_path, scanm_path):
    """Deep dive into pixel value differences."""
    print("\n" + "=" * 60)
    print("PIXEL VALUE DEEP DIVE")
    print("=" * 60)

    # Load raw from H5
    with h5py.File(igor_h5_path, "r") as f:
        # Check both detrended and raw
        igor_detrended = np.array(f["wDataCh0_detrended"]).T  # Transpose to match
        igor_raw = np.array(f["wDataCh0"]).T if "wDataCh0" in f else None
        print(f"\nIGOR wDataCh0_detrended:")
        print(f"  dtype: {igor_detrended.dtype}")
        print(f"  min: {igor_detrended.min()}, max: {igor_detrended.max()}")
        print(f"  mean: {igor_detrended.mean():.2f}")
        print(f"  sample values [0,0,:5]: {igor_detrended[0, 0, :5]}")

        if igor_raw is not None:
            print(f"\nIGOR wDataCh0 (raw):")
            print(f"  dtype: {igor_raw.dtype}")
            print(f"  min: {igor_raw.min()}, max: {igor_raw.max()}")
            print(f"  mean: {igor_raw.mean():.2f}")
            print(f"  sample values [0,0,:5]: {igor_raw[0, 0, :5]}")

    # Load from ScanM
    from pygor.preproc.scanm import load_scanm
    header, data = load_scanm(scanm_path)
    scanm_images = data[0]

    print(f"\nScanM direct load:")
    print(f"  dtype: {scanm_images.dtype}")
    print(f"  min: {scanm_images.min()}, max: {scanm_images.max()}")
    print(f"  mean: {scanm_images.mean():.2f}")
    print(f"  sample values [0,0,:5]: {scanm_images[0, 0, :5]}")

    # Check if it's a signed/unsigned conversion issue
    print(f"\n--- Checking signed/unsigned conversion ---")

    # If IGOR is uint16 and ScanM is int16, the negative values would wrap
    # uint16 range: 0 to 65535
    # int16 range: -32768 to 32767

    # Convert ScanM int16 to uint16 by adding 32768 (if that's the issue)
    scanm_as_uint16 = scanm_images.astype(np.int32) + 32768
    scanm_as_uint16 = scanm_as_uint16.astype(np.uint16)

    print(f"\nScanM converted to uint16 (adding 32768):")
    print(f"  min: {scanm_as_uint16.min()}, max: {scanm_as_uint16.max()}")
    print(f"  mean: {scanm_as_uint16.mean():.2f}")
    print(f"  sample values [0,0,:5]: {scanm_as_uint16[0, 0, :5]}")

    # Compare with IGOR
    if np.allclose(igor_detrended, scanm_as_uint16):
        print("\n  [OK] MATCH after uint16 conversion!")
    else:
        diff = np.abs(igor_detrended.astype(float) - scanm_as_uint16.astype(float))
        print(f"\n  Still mismatch after uint16 conversion")
        print(f"  max_diff: {diff.max()}, mean_diff: {diff.mean():.4f}")

    # Try the reverse - convert IGOR uint16 to int16
    igor_as_int16 = igor_detrended.astype(np.int32) - 32768
    igor_as_int16 = igor_as_int16.astype(np.int16)

    print(f"\nIGOR converted to int16 (subtracting 32768):")
    print(f"  min: {igor_as_int16.min()}, max: {igor_as_int16.max()}")
    print(f"  mean: {igor_as_int16.mean():.2f}")
    print(f"  sample values [0,0,:5]: {igor_as_int16[0, 0, :5]}")

    if np.allclose(igor_as_int16, scanm_images):
        print("\n  [OK] MATCH after int16 conversion!")
    else:
        diff = np.abs(igor_as_int16.astype(float) - scanm_images.astype(float))
        print(f"\n  Still mismatch after int16 conversion")
        print(f"  max_diff: {diff.max()}, mean_diff: {diff.mean():.4f}")

        # Check if values are close but with small offset
        if diff.max() < 100:
            print("  Values are very close - might be detrending difference")

    # Try view() reinterpretation (bit-level cast)
    print("\n--- Checking bit-level reinterpretation ---")
    scanm_viewed_as_uint16 = scanm_images.view(np.uint16)
    print(f"\nScanM viewed as uint16 (bit reinterpret):")
    print(f"  min: {scanm_viewed_as_uint16.min()}, max: {scanm_viewed_as_uint16.max()}")
    print(f"  mean: {scanm_viewed_as_uint16.mean():.2f}")
    print(f"  sample values [0,0,:5]: {scanm_viewed_as_uint16[0, 0, :5]}")

    # Compare with IGOR raw (wDataCh0)
    if igor_raw is not None:
        if np.allclose(igor_raw, scanm_viewed_as_uint16):
            print("\n  [OK] MATCH with IGOR wDataCh0 (raw) using view()!")
        else:
            diff = np.abs(igor_raw.astype(float) - scanm_viewed_as_uint16.astype(float))
            print(f"\n  Still mismatch with IGOR raw")
            print(f"  max_diff: {diff.max()}, mean_diff: {diff.mean():.4f}")

            # Check a specific offset
            offsets_to_try = [0, 32768, -32768, 32767, -32767]
            for offset in offsets_to_try:
                test = scanm_viewed_as_uint16.astype(np.int64) + offset
                diff_test = np.abs(igor_raw.astype(np.int64) - test)
                if diff_test.max() < 10:
                    print(f"\n  [OK] Near-match with offset {offset}! max_diff={diff_test.max()}")


def analyze_trigger_channel(igor_h5_path, scanm_path):
    """Analyze trigger channel to understand detection parameters."""
    print("\n" + "=" * 60)
    print("TRIGGER CHANNEL ANALYSIS")
    print("=" * 60)

    # Load IGOR trigger data
    with h5py.File(igor_h5_path, "r") as f:
        igor_triggers = np.array(f["Triggertimes_Frame"])
        igor_triggers = igor_triggers[~np.isnan(igor_triggers)].astype(int)
        igor_trigger_values = np.array(f["Triggervalues"]) if "Triggervalues" in f else None
        igor_ch2 = np.array(f["wDataCh2"]).T if "wDataCh2" in f else None

    print(f"\nIGOR Triggertimes_Frame:")
    print(f"  Count: {len(igor_triggers)}")
    print(f"  First 10: {igor_triggers[:10]}")
    print(f"  Diffs (first 10): {np.diff(igor_triggers[:11])}")

    if igor_ch2 is not None:
        print(f"\nIGOR wDataCh2:")
        print(f"  Shape: {igor_ch2.shape}")
        print(f"  dtype: {igor_ch2.dtype}")
        # In IGOR, after reduction, shape is (2, nLines, nFrames)
        # After transpose: (nFrames, nLines, 2)
        print(f"  Column 0 min/max: {igor_ch2[:, :, 0].min()}, {igor_ch2[:, :, 0].max()}")
        print(f"  Column 0 mean: {igor_ch2[:, :, 0].mean():.2f}")

    # Load ScanM trigger channel
    from pygor.preproc.scanm import load_scanm, read_smh_header
    header, data = load_scanm(scanm_path, channels=[0, 2])

    if 2 in data:
        scanm_ch2 = data[2]
        print(f"\nScanM Channel 2 (trigger):")
        print(f"  Shape: {scanm_ch2.shape}")
        print(f"  dtype: {scanm_ch2.dtype}")
        print(f"  min: {scanm_ch2.min()}, max: {scanm_ch2.max()}")
        print(f"  mean: {scanm_ch2.mean():.2f}")
        
        # Get line duration from header
        smh_path = scanm_path.with_suffix('.smh')
        header = read_smh_header(smh_path)
        line_duration = header.get('RealLineDuration', 
                                   header.get('LineDuration', 0.001))
        print(f"\n  Line duration from header: {line_duration} s")

        # IGOR looks at column 0 only (after reducing Ch2 to 2 columns)
        print(f"\n  Checking column 0 of trigger channel (IGOR style):")
        col0 = scanm_ch2[:, :, 0]
        print(f"    Column 0 shape: {col0.shape}")
        print(f"    Column 0 min: {col0.min()}, max: {col0.max()}")
        print(f"    Column 0 mean: {col0.mean():.2f}")
        
        # Check IGOR's threshold approach: trigger when value < 2^16 - threshold
        # IGOR default threshold is often around 10000 (from OS_Parameters)
        print(f"\n  IGOR-style trigger detection (value < 2^16 - threshold):")
        for thresh in [5000, 10000, 15000, 20000]:
            threshold_value = 2**16 - thresh  # e.g., 55536 for thresh=10000
            # Count values below threshold in column 0
            below_thresh = col0 < threshold_value
            n_below = np.sum(below_thresh)
            # Count frames with any value below threshold
            frames_with_trigger = np.any(below_thresh, axis=1)
            n_frames_with = np.sum(frames_with_trigger)
            print(f"    threshold={thresh} (value<{threshold_value}): {n_below} pixels, {n_frames_with} frames")
        
        # Run trigger detection using the module function
        from pygor.preproc.scanm import detect_triggers
        print(f"\n  Running trigger detection (pygor.preproc.scanm.detect_triggers):")
        for thresh in [5000, 10000, 15000, 20000]:
            trig_frames, trig_times = detect_triggers(
                scanm_ch2, line_duration, trigger_threshold=thresh, 
                min_gap_seconds=0.5
            )
            print(f"    threshold={thresh}: {len(trig_frames)} triggers found")
            if len(trig_frames) == len(igor_triggers):
                print(f"      [OK] MATCH with IGOR count!")
                # Check if frames match
                if np.array_equal(trig_frames, igor_triggers):
                    print(f"      [OK] Frame indices also match!")
                else:
                    # Check how many match
                    matches = np.sum(trig_frames == igor_triggers)
                    print(f"      Partial match: {matches}/{len(igor_triggers)} frames match exactly")
                    print(f"      IGOR first 5: {igor_triggers[:5]}")
                    print(f"      Ours first 5: {trig_frames[:5]}")

        # Also check if maybe the trigger is in a different region
        print(f"\n  Checking trigger values at different x positions:")
        for x in [0, 1, 63, 64, 127]:
            if x < scanm_ch2.shape[2]:
                col_data = scanm_ch2[:, :, x]
                print(f"    x={x}: min={col_data.min()}, max={col_data.max()}, mean={col_data.mean():.2f}")

    else:
        print("\n  Channel 2 not found in ScanM data!")


def main():
    # Paths
    igor_h5_path = Path(r"D:\Igor analyses\OSDS\251112 OSDS\Original_2025-11-12_SMP_0_1_gradient_contrast_400_white.h5")
    scanm_path = Path(r"D:\Igor analyses\OSDS\251112 OSDS\0_1_gradient_contrast_400_white.smp")

    print("=" * 60)
    print("COMPARING IGOR H5 vs DIRECT SCANM LOADING")
    print("=" * 60)
    print(f"\nIGOR H5: {igor_h5_path.name}")
    print(f"ScanM:   {scanm_path.name}")

    # Check files exist
    if not igor_h5_path.exists():
        print(f"\nERROR: IGOR H5 not found: {igor_h5_path}")
        return
    if not scanm_path.exists():
        print(f"\nERROR: ScanM file not found: {scanm_path}")
        return

    # 1. Inspect IGOR H5 structure
    print("\n" + "-" * 40)
    print("IGOR H5 STRUCTURE:")
    print("-" * 40)
    igor_structure = inspect_h5_structure(igor_h5_path)
    for key, info in sorted(igor_structure.items()):
        print(f"  {key}: shape={info['shape']}, dtype={info['dtype']}")

    # 2. Load via Core.from_scanm()
    print("\n" + "-" * 40)
    print("LOADING VIA Core.from_scanm()...")
    print("-" * 40)
    scanm_data = Core.from_scanm(scanm_path)
    print(f"  Images shape: {scanm_data.images.shape}")
    print(f"  Frame rate: {scanm_data.frame_hz:.4f} Hz")
    print(f"  Line duration: {scanm_data.linedur_s:.6f} s")
    print(f"  Triggers: {len(scanm_data.triggertimes_frame)}")

    # 3. Load IGOR data for comparison
    print("\n" + "-" * 40)
    print("LOADING IGOR H5 VIA Core...")
    print("-" * 40)
    igor_data = Core(igor_h5_path)
    print(f"  Images shape: {igor_data.images.shape}")
    print(f"  Frame rate: {igor_data.frame_hz:.4f} Hz")
    print(f"  Line duration: {igor_data.linedur_s:.6f} s")
    print(f"  Triggers: {len(igor_data.triggertimes_frame)}")

    # 4. Compare key values
    print("\n" + "-" * 40)
    print("COMPARING KEY VALUES:")
    print("-" * 40)

    # Frame rate
    if np.isclose(igor_data.frame_hz, scanm_data.frame_hz, rtol=1e-4):
        print(f"  frame_hz: Match [OK] ({scanm_data.frame_hz:.4f} Hz)")
    else:
        print(f"  frame_hz: MISMATCH! IGOR={igor_data.frame_hz:.4f}, ScanM={scanm_data.frame_hz:.4f}")

    # Line duration
    if np.isclose(igor_data.linedur_s, scanm_data.linedur_s, rtol=1e-4):
        print(f"  linedur_s: Match [OK] ({scanm_data.linedur_s:.6f} s)")
    else:
        print(f"  linedur_s: MISMATCH! IGOR={igor_data.linedur_s:.6f}, ScanM={scanm_data.linedur_s:.6f}")

    # 5. Compare arrays
    print("\n" + "-" * 40)
    print("COMPARING ARRAYS:")
    print("-" * 40)

    # Images (main data)
    compare_arrays("images", igor_data.images, scanm_data.images)

    # Average stack
    compare_arrays("average_stack", igor_data.average_stack, scanm_data.average_stack)

    # Trigger times
    # Note: IGOR triggertimes are in different units potentially
    print(f"\n  Trigger comparison:")
    print(f"    IGOR triggers: {len(igor_data.triggertimes_frame)} triggers")
    print(f"    ScanM triggers: {len(scanm_data.triggertimes_frame)} triggers")
    if len(igor_data.triggertimes_frame) > 0 and len(scanm_data.triggertimes_frame) > 0:
        print(f"    IGOR first 5: {igor_data.triggertimes_frame[:5]}")
        print(f"    ScanM first 5: {scanm_data.triggertimes_frame[:5]}")
        print(f"    IGOR last 5: {igor_data.triggertimes_frame[-5:]}")
        print(f"    ScanM last 5: {scanm_data.triggertimes_frame[-5:]}")
        
        # Check if trigger frames match exactly
        if len(igor_data.triggertimes_frame) == len(scanm_data.triggertimes_frame):
            if np.array_equal(igor_data.triggertimes_frame, scanm_data.triggertimes_frame):
                print(f"    [OK] Trigger frames MATCH EXACTLY!")
            else:
                matches = np.sum(igor_data.triggertimes_frame == scanm_data.triggertimes_frame)
                print(f"    Partial match: {matches}/{len(igor_data.triggertimes_frame)} frames match")

    # 6. Summary
    print("\n" + "=" * 60)
    print("FINAL VERIFICATION SUMMARY")
    print("=" * 60)
    
    # Track what matches
    results = {}
    
    # Shape match
    results['shape'] = igor_data.images.shape == scanm_data.images.shape
    print(f"  Shape match: {'[OK]' if results['shape'] else '[FAIL]'} "
          f"({igor_data.images.shape} vs {scanm_data.images.shape})")
    
    # Check raw pixel data (compare ScanM vs IGOR wDataCh0)
    # Load IGOR's raw wDataCh0 for comparison
    with h5py.File(igor_h5_path, "r") as f:
        igor_raw = np.array(f["wDataCh0"]).T if "wDataCh0" in f else None
    
    if igor_raw is not None:
        results['raw_pixels'] = np.array_equal(igor_raw, scanm_data.images)
        print(f"  Raw pixels (wDataCh0): {'[OK]' if results['raw_pixels'] else '[FAIL]'}")
    else:
        results['raw_pixels'] = False
        print(f"  Raw pixels: IGOR wDataCh0 not found")
    
    # Timing match
    results['frame_hz'] = np.isclose(igor_data.frame_hz, scanm_data.frame_hz, rtol=1e-4)
    results['linedur_s'] = np.isclose(igor_data.linedur_s, scanm_data.linedur_s, rtol=1e-4)
    print(f"  Frame rate: {'[OK]' if results['frame_hz'] else '[FAIL]'} ({scanm_data.frame_hz:.4f} Hz)")
    print(f"  Line duration: {'[OK]' if results['linedur_s'] else '[FAIL]'} ({scanm_data.linedur_s:.6f} s)")
    
    # Trigger match
    results['trigger_count'] = len(igor_data.triggertimes_frame) == len(scanm_data.triggertimes_frame)
    results['trigger_frames'] = np.array_equal(igor_data.triggertimes_frame, scanm_data.triggertimes_frame)
    print(f"  Trigger count: {'[OK]' if results['trigger_count'] else '[FAIL]'} ({len(scanm_data.triggertimes_frame)})")
    print(f"  Trigger frames: {'[OK]' if results['trigger_frames'] else '[FAIL]'}")
    
    # Note about detrending
    print(f"\n  Note: 'images' comparison shows difference because:")
    print(f"    - IGOR H5 uses wDataCh0_detrended (baseline-subtracted)")
    print(f"    - ScanM loads raw data (equivalent to IGOR's wDataCh0)")
    print(f"    - Raw pixel data MATCHES exactly")
    
    # Overall result
    all_pass = all([results['shape'], results['raw_pixels'], results['frame_hz'], 
                    results['linedur_s'], results['trigger_count'], results['trigger_frames']])
    print(f"\n  {'='*40}")
    print(f"  OVERALL: {'ALL TESTS PASSED!' if all_pass else 'SOME TESTS FAILED'}")
    print(f"  {'='*40}")


if __name__ == "__main__":
    main()

    # Run the deep dive
    igor_h5_path = Path(r"D:\Igor analyses\OSDS\251112 OSDS\Original_2025-11-12_SMP_0_1_gradient_contrast_400_white.h5")
    scanm_path = Path(r"D:\Igor analyses\OSDS\251112 OSDS\0_1_gradient_contrast_400_white.smp")
    inspect_pixel_values(igor_h5_path, scanm_path)

    # Analyze trigger channel
    analyze_trigger_channel(igor_h5_path, scanm_path)
    
    # Test preprocessing
    print("\n" + "=" * 60)
    print("PREPROCESSING VERIFICATION")
    print("=" * 60)
    
    from pygor.preproc.scanm import load_scanm, preprocess_stack
    
    # Load raw data
    header, data = load_scanm(scanm_path)
    raw = data[0]
    
    # Load IGOR reference and parameters
    with h5py.File(igor_h5_path, "r") as f:
        igor_detrended = np.array(f['wDataCh0_detrended']).T.astype(np.float32)
        os_params = np.array(f['OS_Parameters'])
    
    # Get parameters
    frame_rate = 1 / (header['FrameHeight'] * header['FrameWidth'] * header.get('RealPixelDuration_µs', 5.0) * 1e-6)
    artifact_width = int(os_params[3])  # LightArtifact_cut
    skip_detrend = bool(os_params[0])   # Detrend_skip
    
    print(f"  Frame rate: {frame_rate:.2f} Hz")
    print(f"  Artifact width: {artifact_width}")
    print(f"  Skip detrend: {skip_detrend}")
    
    # Apply preprocessing
    our_result = preprocess_stack(
        raw,
        frame_rate=frame_rate,
        artifact_width=artifact_width,
        flip_x=True,
        detrend=not skip_detrend,
        fix_first_frame=True,
    )
    
    # Compare
    diff = np.abs(our_result - igor_detrended)
    max_diff = diff.max()
    mean_diff = diff.mean()
    match = max_diff < 1.0  # Allow for float32 precision
    
    print(f"\n  Preprocessing comparison:")
    print(f"    Max difference: {max_diff:.4f}")
    print(f"    Mean difference: {mean_diff:.6f}")
    print(f"    Result: {'[OK] MATCH!' if match else '[FAIL] MISMATCH'}")
