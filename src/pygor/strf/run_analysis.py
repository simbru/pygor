
import pathlib
import os
import pandas as pd

# Pygor imports
from pygor.classes.experiment import Experiment
import pygor.strf.analyse
import pygor.filehandling

"""
The programatic way of running your STRF analysis, 
aka the .py version of the analysis notebooks.
"""

user = pathlib.Path(os.getcwd()).parents[1].stem

## Specify files to load:
# inj_files = pygor.filehandling.find_files_in(".h5", r"D:\Igor analyses", match_all = ["inj", "SWN"], recursive=True) #"ColoursSWN"
# ctrl_files =pygor.filehandling.find_files_in(".h5", r"D:\Igor analyses\SWN BC main", match = "SWN", recursive=True) #"ColoursSWN"

def check_user() -> None:
    print(f"Found user:", user)
    inp = input("Done? Press enter to continue, or type 'q' to quit.")
    if inp == 'q':
        exit()
    else:
        pass
    return None

def check_output_loc(output_path) -> None:
    if output_path.exists() is False:
        try:
            output_path.mkdir()
        except FileExistsError:
            print("Output directory already exists.")
            exit()
        except Exception as e:
            print("Could not create output directory with error", e)
            exit()
    return None

## Load pre-processed files from .pklexp:

def global_load() -> None:
    global exp_inj, exp_ctrl
    exp_inj = pygor.filehandling.load_pkl(rf"C:\Users\{user}\OneDrive\pickles\inj.pklexp")
    exp_ctrl = pygor.filehandling.load_pkl(rf"C:\Users\{user}\OneDrive\pickles\ctrls.pklexp")
    exp_ctrl.detach_data([10]) # <-- look into why this guy is missing its ROIs 

def chromaticity(output_path) -> None: 
    # import pygor.load

    
    ## Run chromaticity analysis:
    chroma_df_inj = pygor.strf.analyse.chromatic_stats(exp_inj, parallel=True   , store_arrs=True)
    chroma_df_ctrl = pygor.strf.analyse.chromatic_stats(exp_ctrl,  parallel=True, store_arrs=True)
    print("- Proportion responding ROIs in control:", chroma_df_ctrl.query("bool_pass == True").shape[0]/chroma_df_ctrl.shape[0], chroma_df_ctrl.query("bool_pass == True").shape[0], "ROIs after filtering")
    print("- Proportion responding ROIs in injection:", chroma_df_inj.query("bool_pass == True").shape[0]/chroma_df_inj.shape[0], chroma_df_inj.query("bool_pass == True").shape[0], "ROIs after filtering")

    ## Merge dataframes:
    chroma_df_inj["Group"] = ["AC block"] * chroma_df_inj.shape[0]
    chroma_df_ctrl["Group"] = ["Control"] * chroma_df_ctrl.shape[0]
    chroma_df = pd.concat([chroma_df_ctrl, chroma_df_inj], ignore_index=True)

    ## Save results:
    print("Storing results")
    # pd.to_pickle(chroma_df_inj, output_path.joinpath("chroma_df_inj.pkl"))
    # pd.to_pickle(chroma_df_ctrl, output_path.joinpath("chroma_df_ctrl.pkl"))
    pd.to_pickle(chroma_df, output_path.joinpath("chroma_df.pkl"))

    return None

def population(output_path) -> None:

    ## Run population analysis:
    roi_df_ctrl = pygor.strf.analyse.roi_stats(exp_ctrl)
    roi_df_inj = pygor.strf.analyse.roi_stats(exp_inj)

    ## Merge dataframes:
    roi_df_ctrl["Group"] = ["Control"] * roi_df_ctrl.shape[0]
    roi_df_inj["Group"] = ["AC block"] * roi_df_inj.shape[0]
    roi_df = pd.concat([roi_df_ctrl, roi_df_inj], ignore_index=True)
    roi_df["colour"] = roi_df["colour"].astype('category')

    ## Save results:
    print("Storing results")
    pd.to_pickle(roi_df, output_path.joinpath("roi_df.pkl"))

if __name__ == "__main__":
    print("Where do you want to output to?")
    output_path = pathlib.Path(input("Output path: "))
    print("Output set to:", output_path)
    confirm = input("Is this correct? Press enter to continue, or type 'q' to quit.")
    check_output_loc(output_path)
    if confirm == 'q':
        exit()
    else:
        print("Loading up data...")
        global_load()
        print("Running, be patient...")
        # check_user()
        chromaticity(output_path)
        population(output_path)
        print("Done!")
        print("Found in output folder:")
        for i in list(output_path.glob("*")):
            print("-", i)
        print("Load up the resulting dataframes in a notebook")