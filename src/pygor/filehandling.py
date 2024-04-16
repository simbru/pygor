
import numpy as np 
import pathlib 
import h5py
import warnings
import pathlib
import math 
from collections.abc import Iterable
import joblib
# from tqdm.autonotebook import tqdm
from tqdm.auto import tqdm
from ipywidgets import Output
import dacite
import natsort # a godsend 

import pygor.classes.strf

# ROI frame needs to contain all stuff from each roi 
# REC frame just keeps tally of recording, and will essentially be only 1 row for each recording
from dataclasses import dataclass
def find_files_in(filetype_ext_str, dir_path, recursive = False, **kwargs) -> list:
    """
    Searches the specified directory for files with the specified file extension.
    The function takes in three parameters:
    - filetype_ext_str (str): The file extension to search for, including the '.', e.g. '.txt'
    - dir_path (str or pathlib.PurePath): The directory path to search in. If a string is provided, it will be converted to a pathlib.PurePath object
    - recursive (bool): If set to True, the function will search recursively through all subdirectories. Default is False.
    
    Returns a list of pathlib.Path objects representing the paths of the files found.
    """
    #  Handle paths using pathlib for maximum enjoyment and minimal life hatered
    if isinstance(dir_path, pathlib.PurePath) is False:
        dir_path = pathlib.Path(dir_path)
    if recursive is False:
        paths = [path for path in dir_path.glob('*' + filetype_ext_str)]
    if recursive is True:
        paths = [path for path in dir_path.rglob('*' + filetype_ext_str)]
    # If search terms are given
    if "match" in kwargs:
        if isinstance(kwargs["match"], str):
            paths = [file for file in paths if kwargs["match"] in file.name]
        else:
            raise AttributeError("kwargs 'match' expected a single str. Consider kwargs 'match_all' or 'match_any' if you want to use a list of strings as search terms.")
    if "match_all" in kwargs:
        if isinstance(kwargs["match_all"], list):
            paths = [file for file in paths if all(map(file.name.__contains__, kwargs["match_all"]))]
        else:
            raise AttributeError("kwargs 'match_all' expected list of strings. Consider kwargs 'match' if you want to specify a single str as search term.")
    if "match_any" in kwargs:
        if isinstance(kwargs["match_any"], list):
            paths = [file for file in paths if any(map(file.name.__contains__, kwargs["match_any"]))]
        else:
            raise AttributeError("kwargs 'match_any' expected list of strings. Consider kwargs 'match' if you want to specify a single str as search term.")
    return paths

def save_pkl(object, save_path, filename):
    final_path = pathlib.Path(save_path, filename).with_suffix(".pkl")
    print("Storing as:", final_path, end = "\r")
    with open(final_path, 'wb') as outp:
        joblib.dump(object, outp, compress='zlib')
        
def load_pkl(full_path):
    with open(full_path, 'rb') as inp:
        object = joblib.load(inp)
        object.metadata["curr_path"] = full_path
        return object

def _load_parser(file_path, as_class = None, **kwargs):
    #print("Current file:", i)
    file_type = pathlib.Path(file_path).suffix
    if file_type == ".pkl":
        loaded = load_pkl(file_path)
    if file_type == ".h5":
        if as_class is None:
            raise AttributeError("Must specify 'as_class = pygor.load.class' class to load .h5 files")
        if as_class is not None:
            loaded = as_class(file_path,  **kwargs)            
    # # # if isinstance(loaded.strfs, np.ndarray) is False and math.isnan(loaded.strfs) is True:
    # # #         print("No STRFs found for", file_path, ", skipping...")
    # #         return None
    # if loaded.multicolour is False:
    #         print("Listed file not multichromatic for file", file_path, ", skipping...")
    #         return None
    return loaded

def load(file_path, as_class = None, **kwargs):
    return _load_parser(file_path, as_class = as_class, **kwargs)

def _load_and_save(file_path, output_folder, as_class, **kwargs):
    loaded = _load_parser(file_path, as_class=as_class, **kwargs)
    name = pathlib.Path(loaded.metadata["filename"]).stem
    loaded.save_pkl(output_folder, name)
    # out.clear_output()

def picklestore_objects(file_paths, output_folder, **kwargs):
    if isinstance(file_paths, Iterable) is False:
        file_paths = [file_paths]
    output_folder = pathlib.Path(output_folder)
    # progress_bar = alive_it(input_objects, spinner = "fishes", bar = 'blocks', calibrate = 50, force_tty=True)
    progress_bar = tqdm(file_paths, desc = "Iterating through, loading, and storing listed files as objects")
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=RuntimeWarning)
        warnings.filterwarnings("ignore", category=UserWarning)
        out = Output()
        display(out)  # noqa: F821
        with out:
            for i in progress_bar:
                _load_and_save(i, output_folder, **kwargs)
                out.clear_output()


# Instantiates the very basic Data object
# def load_data(filename, img_stack = True):
#     with h5py.File(filename) as HDF5_file:
#         rois = np.array(HDF5_file["ROIs"])
#         if img_stack == True:
#             images = data_helpers.load_wDataCh0(HDF5_file)
#         else:
#             images = np.nan
#         meta_data = metadata_dict(HDF5_file)
#     Data_obj = Data(images = images, rois = rois, metadata = meta_data)
#     return Data_obj