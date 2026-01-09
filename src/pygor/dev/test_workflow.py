import pygor
import pygor.preproc as preproc
import shutil
import os 
import pathlib
import re

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
    example_path = r"D:\Igor analyses\OSDS\251112 OSDS\0_1_SWN_200_White.smh"
    
    # Test header loading
    header = preproc.read_smh_header(example_path)
    print("Header loaded successfully!")
    print(f"  FrameWidth: {header['FrameWidth']}, FrameHeight: {header['FrameHeight']}")
    print(f"  NumberOfFrames: {header['NumberOfFrames']}, FrameCounter: {header['FrameCounter']}")
    
    # Test full data loading
    header, data = preproc.load_scanm(example_path)
    print(f"\nData loaded: {list(data.keys())} channels")
    for ch, stack in data.items():
        print(f"  Channel {ch}: {stack.shape}")
    
    # Test pygor-compatible object creation
    print("\nCreating pygor-compatible object...")
    scanm_data = preproc.to_pygor_data(example_path)
    print(f"  Type: {scanm_data.type}")
    print(f"  Images shape: {scanm_data.images.shape}")
    print(f"  Frame rate: {scanm_data.frame_hz:.2f} Hz")
    print(f"  Line duration: {scanm_data.linedur_s*1000:.3f} ms")
    print(f"  Triggers detected: {len(scanm_data.triggertimes_frame)}")
    print(f"  Date: {scanm_data.metadata['exp_date']}")
    print(f"  Position (XYZ): {scanm_data.metadata['objectiveXYZ']}")
    print(f"\nObject repr: {repr(scanm_data)}")

if __name__ == "__main__":
    main()