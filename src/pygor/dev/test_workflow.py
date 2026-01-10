import time
import pygor
import pygor.preproc as preproc
import pygor.filehandling
import pygor.classes.core_data
import shutil
import os 
import pathlib
import re
import timeit

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
    example_path = r"D:\Igor analyses\OSDS\251112 OSDS\0_1_SWN_200_White.smh"

    # Test loading directly as Core from ScanM files
    print("Loading ScanM file directly as Core...")
    from pygor.classes.core_data import Core
    
    data = Core.from_scanm(example_path)
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
    print(f"  Preprocessing applied: {data._preprocessed}")
    print(f"  Preprocessing params: {data.metadata.get('preprocessing', {})}")
    
    data.view_images_interactive()

    # End timeit
    time_end = timeit.default_timer()
    print(f"\nTotal execution time: {time_end - time_start:.2f} seconds")
if __name__ == "__main__":
    main()