import pygor
import pygor.preproc as preproc
import timeit
import napari
import matplotlib.pyplot as plt
import tifffile

"""
Imaginative workflow for matching files in a directory based on specific naming patterns, 
loading them as Pygor objects, making sure their ROIs line up and are independently registered
to their imaging data, and then performing a tandem analysis on the matched pairs. 
"""

# def match_files(stim_type_A="type_A", stim_type_B="type_B"):
#     # Match files in directory according to pattern (0_1_stim_type_A with 0_1_stim_type_B)
#     dir_path = pathlib.Path(r"D:\\Igor analyses\\OSDS\\251112 OSDS")
#     print("Matching files in directory:", dir_path)
#     ## list all files 
#     listed_files = os.listdir(dir_path)
#     print("Listed files:", listed_files)
#     pattern = rf"(\d+_\d+{stim_type_A})\.smh$"
#     matched_pairs = []
#     for file_name in listed_files:
#         match = re.match(pattern, file_name)
#         if match:
#             base_name = match.group(1)
#             file_A = dir_path / f"{base_name}.smh"
#             file_B = dir_path / f"{base_name.replace(stim_type_A, stim_type_B)}.smh"
#             if os.path.exists(file_B):
#                 matched_pairs.append((file_A, file_B))
#     print("Matched file pairs:", matched_pairs)
#     return matched_pairs



# def tandem_analysis(analysis_A, analysis_B, pair_tuple):
#     # Perform tandem analysis on matched file pairs
#     file_A, file_B = pair_tuple
#     print(f"Performing tandem analysis on:\n  File A: {file_A}\n  File B: {file_B}")
#     # Load analyses
#     ana_A = pygor.load_analysis(analysis_A, file_A)
#     ana_B = pygor.load_analysis(analysis_B, file_B)
#     # Perform tandem analysis (placeholder for actual analysis logic)
#     result = pygor.tandem_analysis(ana_A, ana_B)
#     print("Tandem analysis result:", result)
#     return result

def main():
    # start timeit
    time_start = timeit.default_timer()
    # example_path = r"D:\Igor analyses\OSDS\251112 OSDS\0_1_SWN_200_White.smh"
    example_path = r"D:\Igor analyses\OSDS\251112 OSDS\0_1_gradient_contrast_400_white.smh"
    # Test loading directly as Core from ScanM files
    print("Loading ScanM file directly as Core...")
    from pygor.classes.core_data import Core
    
    data = Core.from_scanm(example_path)
    data_detrend = Core.from_scanm(example_path)
    print(f"  Type: {data.type}")
    print(f"  Images shape: {data.images.shape}")
    print(f"  Frame rate: {data.frame_hz:.2f} Hz")
    print(f"  Line duration: {data.linedur_s*1000:.3f} ms")
    print(f"  Triggers detected: {len(data.triggertimes_frame)}")
    print(f"  Date: {data.metadata['exp_date']}")
    print(f"  Position (XYZ): {data.metadata['objectiveXYZ']}")
    print(f"\nObject repr: {repr(data)}")
    
    # Test that Core methods work
    print("\n--- Testing Core methods ---")
    print(f"  average_stack shape: {data.average_stack.shape}")
    print(f"  n_planes: {data.n_planes}")
    print(f"  trigger_mode: {data.trigger_mode}")
    

    ## Optionally:
    # Test export to H5
    # print("\n--- Testing H5 Export ---")
    # h5_path = data.export_to_h5(overwrite=True)
    
    # # Reload as Core from H5 to verify roundtrip
    # print("\nReloading exported H5...")
    # reloaded = pygor.filehandling.load(h5_path, as_class=Core)
    # print(f"  Reloaded images shape: {reloaded.images.shape}")
    # print(f"  Reloaded frame_hz: {reloaded.frame_hz:.2f} Hz")
    # print(f"  Reloaded triggers: {len(reloaded.triggertimes_frame)}")
    
    # Apply preprocessing (light artifact fix, x-flip, optional detrend)
    print("\n--- Applying Preprocessing ---")
    data.preprocess(detrend=False)  # Skip detrend for faster testing
    data_detrend.preprocess(detrend=True)

    non_detrended_images = data.images
    detrended_images = data_detrend.images
    fig, ax = plt.subplots(2, 2, figsize=(10, 10))

    # Calculate data ranges and bins
    avg_no_detrend = non_detrended_images.mean(axis=0)
    avg_detrend = detrended_images.mean(axis=0)
    frame_no_detrend = non_detrended_images[100]
    frame_detrend = detrended_images[100]

    # Create histograms with fine bins and data-limited ranges
    ax[0, 0].hist(avg_no_detrend.flatten(), bins=100,
                  range=(avg_no_detrend.min(), avg_no_detrend.max()),
                  edgecolor='none')
    ax[0, 0].set_title("Average Image (no detrend)")
    ax[0, 0].set_xlabel("Pixel Intensity")
    ax[0, 0].set_ylabel("Count")

    ax[1, 0].hist(avg_detrend.flatten(), bins=100,
                  range=(avg_detrend.min(), avg_detrend.max()),
                  edgecolor='none')
    ax[1, 0].set_title("Average Image (with detrend)")
    ax[1, 0].set_xlabel("Pixel Intensity")
    ax[1, 0].set_ylabel("Count")

    ax[0, 1].hist(frame_no_detrend.flatten(), bins=100,
                  range=(frame_no_detrend.min(), frame_no_detrend.max()),
                  edgecolor='none')
    ax[0, 1].set_title("Frame 100 (no detrend)")
    ax[0, 1].set_xlabel("Pixel Intensity")
    ax[0, 1].set_ylabel("Count")

    ax[1, 1].hist(frame_detrend.flatten(), bins=100,
                  range=(frame_detrend.min(), frame_detrend.max()),
                  edgecolor='none')
    ax[1, 1].set_title("Frame 100 (with detrend)")
    ax[1, 1].set_xlabel("Pixel Intensity")
    ax[1, 1].set_ylabel("Count")

    plt.show()

    # Load Igor detrended version for comparison
    print("\n--- Comparing with Igor detrended version ---")
    igor_detrended_path = r"D:\Igor analyses\OSDS\251112 OSDS\wDataCh0_detrended.tif"
    igor_detrended = tifffile.imread(igor_detrended_path)
    print(f"  Igor detrended shape: {igor_detrended.shape}")

    # Create comparison figure with histograms
    fig2, ax2 = plt.subplots(2, 2, figsize=(12, 10))

    # Pygor detrended
    avg_pygor = detrended_images.mean(axis=0)
    frame_pygor = detrended_images[100]

    # Igor detrended
    avg_igor = igor_detrended.mean(axis=0)
    frame_igor = igor_detrended[100]

    # Determine common range for each comparison
    avg_min = min(avg_pygor.min(), avg_igor.min())
    avg_max = max(avg_pygor.max(), avg_igor.max())
    frame_min = min(frame_pygor.min(), frame_igor.min())
    frame_max = max(frame_pygor.max(), frame_igor.max())

    # Average image histograms (overlaid)
    ax2[0, 0].hist(avg_pygor.flatten(), bins=100, range=(avg_min, avg_max),
                   alpha=0.6, label='Pygor', edgecolor='none')
    ax2[0, 0].hist(avg_igor.flatten(), bins=100, range=(avg_min, avg_max),
                   alpha=0.6, label='Igor', edgecolor='none')
    ax2[0, 0].set_title("Average Image Distribution")
    ax2[0, 0].set_xlabel("Pixel Intensity")
    ax2[0, 0].set_ylabel("Count")
    ax2[0, 0].legend()

    # Frame 100 histograms (overlaid)
    ax2[0, 1].hist(frame_pygor.flatten(), bins=100, range=(frame_min, frame_max),
                   alpha=0.6, label='Pygor', edgecolor='none')
    ax2[0, 1].hist(frame_igor.flatten(), bins=100, range=(frame_min, frame_max),
                   alpha=0.6, label='Igor', edgecolor='none')
    ax2[0, 1].set_title("Frame 100 Distribution")
    ax2[0, 1].set_xlabel("Pixel Intensity")
    ax2[0, 1].set_ylabel("Count")
    ax2[0, 1].legend()

    # Difference histograms
    avg_diff = avg_pygor - avg_igor
    frame_diff = frame_pygor - frame_igor

    ax2[1, 0].hist(avg_diff.flatten(), bins=100,
                   range=(avg_diff.min(), avg_diff.max()),
                   edgecolor='none')
    ax2[1, 0].set_title("Average Image Difference (Pygor - Igor)")
    ax2[1, 0].set_xlabel("Difference")
    ax2[1, 0].set_ylabel("Count")
    ax2[1, 0].axvline(0, color='red', linestyle='--', alpha=0.7)

    ax2[1, 1].hist(frame_diff.flatten(), bins=100,
                   range=(frame_diff.min(), frame_diff.max()),
                   edgecolor='none')
    ax2[1, 1].set_title("Frame 100 Difference (Pygor - Igor)")
    ax2[1, 1].set_xlabel("Difference")
    ax2[1, 1].set_ylabel("Count")
    ax2[1, 1].axvline(0, color='red', linestyle='--', alpha=0.7)

    plt.tight_layout()
    plt.show()


    # viewer = napari.Viewer()
    # viewer.add_image(non_detrended_images, name="Preprocessed (no detrend)")
    # viewer.add_image(detrended_images, name="Preprocessed (with detrend)")
    
    # napari.run()

    # print(f"  Preprocessing applied: {data._preprocessed}")
    # print(f"  Preprocessing params: {data.metadata.get('preprocessing', {})}")
    

    # # End timeit
    # time_end = timeit.default_timer()
    # print(f"\nTotal execution time: {time_end - time_start:.2f} seconds")
if __name__ == "__main__":
    main()