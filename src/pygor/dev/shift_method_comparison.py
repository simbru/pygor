"""
Compare FFT-based vs scipy.ndimage.shift for frame shifting.

Tests speed, memory usage, and edge artifact behavior for both methods.
Since real-world calcium imaging shifts are typically small (<5 pixels),
the interpolation-based approach (scipy.ndimage.shift) might be competitive
while avoiding FFT edge wrapping artifacts.
"""

import pathlib
import numpy as np
import timeit
import tracemalloc
from concurrent.futures import ThreadPoolExecutor
import os

from scipy.ndimage import shift as scipy_shift, fourier_shift
from scipy.fft import fft2, ifft2

try:
    from joblib import Parallel, delayed
    _HAS_JOBLIB = True
except ImportError:
    _HAS_JOBLIB = False
    print("joblib not available, using ThreadPoolExecutor")

from pygor.classes.core_data import Core


# === Shift implementations ===

def shift_fft(frame: np.ndarray, shift_yx: np.ndarray) -> np.ndarray:
    """FFT-based shift (wraps at edges)."""
    freq = fft2(frame)
    shifted_freq = fourier_shift(freq, shift_yx)
    return np.real(ifft2(shifted_freq))


def shift_scipy(frame: np.ndarray, shift_yx: np.ndarray, order: int = 1, mode: str = "reflect") -> np.ndarray:
    """Scipy ndimage shift (configurable edge handling)."""
    return scipy_shift(frame, shift=shift_yx, order=order, mode=mode)


def shift_fft_padded(frame: np.ndarray, shift_yx: np.ndarray, pad: int = 10) -> np.ndarray:
    """FFT-based shift with padding to avoid wrap artifacts."""
    padded = np.pad(frame, pad, mode='reflect')
    freq = fft2(padded)
    shifted_freq = fourier_shift(freq, shift_yx)
    shifted = np.real(ifft2(shifted_freq))
    return shifted[pad:-pad, pad:-pad]


# === Apply shift to stack functions ===

def apply_shifts_fft(stack: np.ndarray, shifts: np.ndarray, n_jobs: int = -1) -> np.ndarray:
    """Apply shifts using FFT method with parallelization."""
    n_workers = os.cpu_count() if n_jobs == -1 else n_jobs

    if _HAS_JOBLIB:
        results = Parallel(n_jobs=n_workers, prefer="threads")(
            delayed(shift_fft)(stack[i], shifts[i])
            for i in range(len(stack))
        )
    else:
        with ThreadPoolExecutor(max_workers=n_workers) as executor:
            results = list(executor.map(
                lambda i: shift_fft(stack[i], shifts[i]),
                range(len(stack))
            ))
    return np.array(results)


def apply_shifts_fft_padded(stack: np.ndarray, shifts: np.ndarray, pad: int = 10, n_jobs: int = -1) -> np.ndarray:
    """Apply shifts using padded FFT method with parallelization."""
    n_workers = os.cpu_count() if n_jobs == -1 else n_jobs

    if _HAS_JOBLIB:
        results = Parallel(n_jobs=n_workers, prefer="threads")(
            delayed(shift_fft_padded)(stack[i], shifts[i], pad)
            for i in range(len(stack))
        )
    else:
        with ThreadPoolExecutor(max_workers=n_workers) as executor:
            results = list(executor.map(
                lambda i: shift_fft_padded(stack[i], shifts[i], pad),
                range(len(stack))
            ))
    return np.array(results)


def apply_shifts_scipy_parallel(stack: np.ndarray, shifts: np.ndarray, order: int = 1, mode: str = "reflect", n_jobs: int = -1) -> np.ndarray:
    """Apply shifts using scipy ndimage with parallelization."""
    n_workers = os.cpu_count() if n_jobs == -1 else n_jobs

    if _HAS_JOBLIB:
        results = Parallel(n_jobs=n_workers, prefer="threads")(
            delayed(shift_scipy)(stack[i], shifts[i], order, mode)
            for i in range(len(stack))
        )
    else:
        with ThreadPoolExecutor(max_workers=n_workers) as executor:
            results = list(executor.map(
                lambda i: shift_scipy(stack[i], shifts[i], order, mode),
                range(len(stack))
            ))
    return np.array(results)


def apply_shifts_scipy_sequential(stack: np.ndarray, shifts: np.ndarray, order: int = 1, mode: str = "reflect") -> np.ndarray:
    """Apply shifts using scipy ndimage sequentially."""
    result = np.zeros_like(stack)
    for i in range(len(stack)):
        result[i] = shift_scipy(stack[i], shifts[i], order, mode)
    return result


# === Benchmark helpers ===

def measure_time(func, *args, n_repeats: int = 3, **kwargs) -> tuple[float, float]:
    """Measure execution time, return (mean, std) in seconds."""
    times = []
    for _ in range(n_repeats):
        start = timeit.default_timer()
        func(*args, **kwargs)
        times.append(timeit.default_timer() - start)
    return np.mean(times), np.std(times)


def measure_memory(func, *args, **kwargs) -> float:
    """Measure peak memory usage in MB."""
    tracemalloc.start()
    func(*args, **kwargs)
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return peak / 1024 / 1024


def compute_edge_difference(original: np.ndarray, shifted: np.ndarray, edge_width: int = 5) -> float:
    """Compute mean absolute difference at edges (higher = more artifacts)."""
    # Get edge regions
    top = shifted[:, :edge_width, :]
    bottom = shifted[:, -edge_width:, :]
    left = shifted[:, :, :edge_width]
    right = shifted[:, :, -edge_width:]

    # Compare to original edges
    top_orig = original[:, :edge_width, :]
    bottom_orig = original[:, -edge_width:, :]
    left_orig = original[:, :, :edge_width]
    right_orig = original[:, :, -edge_width:]

    diff = (
        np.abs(top - top_orig).mean() +
        np.abs(bottom - bottom_orig).mean() +
        np.abs(left - left_orig).mean() +
        np.abs(right - right_orig).mean()
    ) / 4
    return diff


def main():
    # Load real data
    example_path = pathlib.Path(r"D:\Igor analyses\SWN BC main\240517 ctrl data\0_0_SWN_200_5hz_RGBUV.smp")

    print("=" * 60)
    print("SHIFT METHOD COMPARISON")
    print("=" * 60)

    print("\nLoading data...")
    data = Core.from_scanm(example_path)
    data.preprocess(detrend=False)

    # Use a subset for testing
    n_test_frames = 1000
    stack = data.images[:n_test_frames].copy()
    print(f"Stack shape: {stack.shape}")
    print(f"Stack dtype: {stack.dtype}")

    # Generate realistic shifts (small, typical for calcium imaging)
    np.random.seed(42)
    shifts = np.random.uniform(-2, 2, size=(n_test_frames, 2))  # ±2 pixel shifts

    print(f"\nShift statistics:")
    print(f"  Range: [{shifts.min():.2f}, {shifts.max():.2f}] pixels")
    print(f"  Mean absolute: {np.abs(shifts).mean():.2f} pixels")

    # === Speed tests ===
    print("\n" + "-" * 60)
    print("SPEED COMPARISON (n_repeats=3)")
    print("-" * 60)

    methods = {
        "FFT (parallel)": lambda: apply_shifts_fft(stack, shifts),
        "FFT padded (parallel)": lambda: apply_shifts_fft_padded(stack, shifts, pad=10),
        "scipy (parallel, order=1)": lambda: apply_shifts_scipy_parallel(stack, shifts, order=1),
        "scipy (parallel, order=0)": lambda: apply_shifts_scipy_parallel(stack, shifts, order=0),
        "scipy (sequential, order=1)": lambda: apply_shifts_scipy_sequential(stack, shifts, order=1),
    }

    results = {}
    for name, func in methods.items():
        print(f"\n  {name}...")
        mean_time, std_time = measure_time(func, n_repeats=3)
        results[name] = {"time_mean": mean_time, "time_std": std_time}
        print(f"    Time: {mean_time:.2f} ± {std_time:.2f} s")

    # === Memory tests ===
    print("\n" + "-" * 60)
    print("MEMORY COMPARISON (peak usage)")
    print("-" * 60)

    for name, func in methods.items():
        print(f"\n  {name}...")
        peak_mb = measure_memory(func)
        results[name]["memory_mb"] = peak_mb
        print(f"    Peak memory: {peak_mb:.1f} MB")

    # === Edge artifact comparison ===
    print("\n" + "-" * 60)
    print("EDGE ARTIFACT COMPARISON")
    print("-" * 60)

    # Apply each method and compute edge differences
    shifted_fft = apply_shifts_fft(stack, shifts)
    shifted_fft_padded = apply_shifts_fft_padded(stack, shifts, pad=10)
    shifted_scipy = apply_shifts_scipy_parallel(stack, shifts, order=1, mode="reflect")

    edge_methods = {
        "FFT (no padding)": shifted_fft,
        "FFT (padded)": shifted_fft_padded,
        "scipy (reflect)": shifted_scipy,
    }

    for name, shifted in edge_methods.items():
        edge_diff = compute_edge_difference(stack, shifted)
        print(f"  {name}: {edge_diff:.4f} mean abs edge difference")

    # === Visual comparison ===
    print("\n" + "-" * 60)
    print("VISUAL COMPARISON")
    print("-" * 60)

    try:
        import napari

        # Show std projections of each method
        viewer = napari.Viewer()
        viewer.add_image(np.std(stack, axis=0), name="Original (std)")
        viewer.add_image(np.std(shifted_fft, axis=0), name="FFT (std)")
        viewer.add_image(np.std(shifted_fft_padded, axis=0), name="FFT padded (std)")
        viewer.add_image(np.std(shifted_scipy, axis=0), name="scipy reflect (std)")

        # Difference images at edges
        diff_fft = shifted_fft - stack
        diff_scipy = shifted_scipy - stack
        viewer.add_image(np.mean(diff_fft, axis=0), name="FFT - Original", colormap="bwr")
        viewer.add_image(np.mean(diff_scipy, axis=0), name="scipy - Original", colormap="bwr")

        print("Napari viewer opened with comparison images.")
        napari.run()

    except ImportError:
        print("napari not available, skipping visual comparison")

    # === Summary ===
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    # Find fastest method
    fastest = min(results.items(), key=lambda x: x[1]["time_mean"])
    print(f"\nFastest method: {fastest[0]} ({fastest[1]['time_mean']:.2f}s)")

    # Speed comparison table
    print("\nRelative speeds (normalized to fastest):")
    for name, res in sorted(results.items(), key=lambda x: x[1]["time_mean"]):
        relative = res["time_mean"] / fastest[1]["time_mean"]
        print(f"  {name}: {relative:.2f}x")

    print("\nRecommendation:")
    print("  - If edge wrapping artifacts are unacceptable: use scipy parallel")
    print("  - If speed is critical and edges don't matter: use FFT parallel")
    print("  - Middle ground: use FFT padded (slight overhead, no wrapping)")


if __name__ == "__main__":
    main()
