import numpy as np
import napari
import matplotlib.pyplot as plt
from skimage.registration import phase_cross_correlation
from scipy.ndimage import shift
from pygor.classes.core_data import Core
import timeit

"""
Simple batch-averaged registration for calcium imaging.
"""

def main():
    time_start = timeit.default_timer()
    
    # Load data
    example_path = r"D:\Igor analyses\OSDS\251112 OSDS\1_0_SWN_200_White.smp"
    print("Loading data...")
    data = Core.from_scanm(example_path)
    data.preprocess(detrend=False)
    image_stack = data.images[:, :, 2:]  # (time, height, width)
    
    # Parameters
    n_reference_frames = 1000  # More frames for stable reference
    batch_size = 10     # ~64 stimulus cycles at 5Hz with 15.625Hz imaging
    upsample_factor = 10      # Lower for speed
    
    print(f"\nRegistration settings:")
    print(f"  Stack: {image_stack.shape} frames")
    print(f"  Reference: {n_reference_frames} frames")
    print(f"  Batch size: {batch_size} frames")
    print(f"  Upsample: {upsample_factor}")
    
    # Create reference from initial frames
    reference = image_stack[:n_reference_frames].mean(axis=0)
    
    # Compute shifts per batch
    n_frames = len(image_stack)
    n_batches = int(np.ceil(n_frames / batch_size))
    shifts = []
    errors = []
    
    print(f"\nComputing shifts for {n_batches} batches...")
    for i in range(n_batches):
        start = i * batch_size
        end = min((i + 1) * batch_size, n_frames)
        
        # Average batch
        batch_avg = image_stack[start:end].mean(axis=0)
        
        # Compute shift
        shift_yx, error, _ = phase_cross_correlation(
            reference, batch_avg, upsample_factor=upsample_factor, normalization=None, #<-- Crucial! Probably because low SNR data, you must disable normalization. That yields sensible error values and far better restults than if in 'phase' mode.
        )
        
        shifts.append(shift_yx)
        errors.append(error)
        
        if i % 20 == 0:
            print(f"  Batch {i+1}/{n_batches}: shift={shift_yx}, error={error:.4f}")
    
    shifts = np.array(shifts)
    errors = np.array(errors)
    
    # Apply shifts
    print("\nApplying shifts...")
    original_min, original_max = image_stack.min(), image_stack.max()
    registered = np.zeros_like(image_stack)
    
    for i in range(n_batches):
        start = i * batch_size
        end = min((i + 1) * batch_size, n_frames)
        
        for j in range(start, end):
            shifted = shift(image_stack[j], shift=shifts[i], order=1, mode='reflect')
            registered[j] = np.clip(shifted, original_min, original_max)
    
    # Print results
    print(f"\nResults:")
    print(f"  Mean shift: {shifts.mean(axis=0)}")
    print(f"  Std shift: {shifts.std(axis=0)}")
    print(f"  Max shift: {shifts.max(axis=0)}")
    print(f"  Mean error: {errors.mean():.4f} (0=perfect, 1=worst)")
    print(f"  Time: {timeit.default_timer() - time_start:.1f}s")
    
    # Plot shifts
    fig, axes = plt.subplots(2, 1, figsize=(10, 6))
    batch_idx = np.arange(len(shifts))
    
    axes[0].plot(batch_idx, shifts[:, 0], 'b.-', label='Y shift')
    axes[0].plot(batch_idx, shifts[:, 1], 'r.-', label='X shift')
    axes[0].set_ylabel('Shift (pixels)')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    axes[0].set_title('Registration Shifts Over Time')
    
    axes[1].plot(batch_idx, errors, 'k.-')
    axes[1].set_xlabel('Batch index')
    axes[1].set_ylabel('Error')
    axes[1].grid(True, alpha=0.3)
    axes[1].set_title('Registration Error (lower is better)')
    
    plt.tight_layout()
    plt.show()
    
    # Visualize
    viewer = napari.Viewer()
    viewer.add_image(reference, name='Reference')
    viewer.add_image(image_stack[:n_batches].mean(axis=0) - registered[:n_batches].mean(axis=0), 
                     name='Difference', colormap='bwr')
    viewer.add_image(image_stack, name='Original')
    viewer.add_image(registered, name='Registered')
    viewer.add_image(np.std(image_stack, axis=0), name='Original std')
    viewer.add_image(np.std(registered, axis=0), name='Registered std')

    napari.run()


if __name__ == "__main__":
    main()
