
import pathlib 
import warnings

try:
    from collections.abc import Iterable
except ImportError:
    from collections import Iterable
from IPython.core import display
import joblib
# from tqdm.autonotebook import tqdm
from tqdm.auto import tqdm
from ipywidgets import Output
import shutil
import contextlib
import joblib

def find_files_in(filetype_ext_str, dir_path, recursive=False, **kwargs) -> list:
    """
    Searches the specified directory for files with the specified file extension.

    Parameters
    ----------
    filetype_ext_str : str
        The file extension to search for, including the '.', e.g. '.txt'.
    dir_path : str or pathlib.PurePath
        The directory path to search in. If a string is provided, it will be converted to a pathlib.PurePath object.
    recursive : bool, optional
        If set to True, the function will search recursively through all subdirectories. Default is False.
    **kwargs
        - match = str: If provided, the function will filter files based on this single search term.
        - match_all = [str]: If provided as a list of strings, the function will filter files that contain 
            all of the specified search terms.
        - match_any = [str]: If provided as a list of strings, the function will filter files that contain 
            any of the specified search terms.

    Returns
    -------
    list of pathlib.Path
        A list of pathlib.Path objects representing the paths of the files found.

    Raises
    ------
    AttributeError
        If the 'match' or 'match_all' or 'match_any' kwargs are not used correctly.

    Notes
    -----
    The function uses pathlib for handling paths.
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

def _load_parser(file_path, as_class = None, **kwargs):
    """
    Parse and load data from a file based on its file type.

    Parameters
    ----------
    file_path : str
        Path to the file to be loaded.
    as_class : class, optional
        Class to be used for loading .h5 files.
        Defaults to None. If not provided when loading .h5 files,
        an AttributeError will be raised.
    **kwargs : dict
        Additional keyword arguments passed to the class initializer
        when loading .h5 files.

    Returns
    -------
    loaded : object
        The loaded data from the file. Type of the object depends
        on the file type and `as_class` parameter.
    """
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
    """
    Loads data from a file specified by `file_path` using a given class or default parser.
    
    Parameters
    ----------
    file_path : str
        The path to the file to be loaded.
    as_class : class, optional
        The class to be used for loading the data. If None, an error will be raised
        when loading .h5 files.
    **kwargs
        Arbitrary keyword arguments passed to the loading function.
    
    Returns
    -------
    object
        The loaded data from the file. Type of the object depends on the file type and `as_class` parameter.
    """
    return _load_parser(file_path, as_class = as_class, **kwargs)

@contextlib.contextmanager
def tqdm_joblib(tqdm_object):
    """Context manager to patch joblib to report into tqdm progress bar given as argument
    from: https://stackoverflow.com/questions/24983493/tracking-progress-of-joblib-parallel-execution/58936697#58936697
    """
    class TqdmBatchCompletionCallback(joblib.parallel.BatchCompletionCallBack):
        def __call__(self, *args, **kwargs):
            tqdm_object.update(n=self.batch_size)
            return super().__call__(*args, **kwargs)
    old_batch_callback = joblib.parallel.BatchCompletionCallBack
    joblib.parallel.BatchCompletionCallBack = TqdmBatchCompletionCallback
    try:
        yield tqdm_object
    finally:
        joblib.parallel.BatchCompletionCallBack = old_batch_callback
        tqdm_object.close()

def load_list(paths_list, as_class = None, parallel = True, **kwargs) -> list[pathlib.WindowsPath]:
    """
    Converts a list of paths to a list of objects, optionally using a specified class for instantiation
    of .h5 files (otherwise will throw an error).

    Parameters
    ----------
    paths_list : list
        A list containing path-like elements.
    as_class : class, optional
        A class to be used for creating objects from paths (default is None).

    Returns
    -------
    list of pathlib.WindowsPath
        A list of objects created from the paths, as instances of pathlib.WindowsPath or `as_class` if provided.

    Errors
    ------
    AttributeError
        If `as_class` is not specified when loading .h5 files, an AttributeError is raised 
        reminding you to specify `as_class` for accurate initialisation.
    """


    if parallel is True:
        print("Launching parallel loading...")
        with tqdm_joblib(tqdm(paths_list, desc = "Loading and instantiating listed files", position = 0, leave = True, total = len(paths_list))) as progress_bar:
            objects_list = joblib.Parallel(n_jobs = -1)(joblib.delayed(_load_parser)(i, as_class=as_class, **kwargs) for i in paths_list)
    else:
        print("Iterating through and loading listed files...")
        progress_bar = tqdm(paths_list, desc = "Iterating through and loading listed files", position = 0,
        leave = True)

        objects_list = []
        out = Output()
        display(out)  # noqa: F821
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=RuntimeWarning)
            warnings.filterwarnings("ignore", category=UserWarning)
            with out:
                for i in progress_bar:
                    objects_list.append(_load_parser(i, as_class=as_class, **kwargs))
                    out.clear_output()
    return objects_list

def _load_and_save(file_path, output_folder, as_class, **kwargs):
    """
    Load data from the specified file and save it to the given output folder using the provided class.

    Parameters
    ----------
    file_path : str
        The path to the file to be loaded.
    output_folder : str
        The folder where the output file will be saved.
    as_class : class
        The class to be used for loading the data.
    **kwargs
        Arbitrary keyword arguments passed to the loading function.

    Returns
    -------
    None
    """
    loaded = _load_parser(file_path, as_class=as_class, **kwargs)
    name = pathlib.Path(loaded.metadata["filename"]).stem
    loaded.save_pkl(output_folder, name)
    # out.clear_output()


def save_pkl(object, save_path, filename):
    """
    Save an object to a pickle file using joblib with zlib compression.

    Parameters
    ----------
    object : any type
        The object to save.
    save_path : str or pathlib.Path
        The directory path where the pickle file will be saved.
    filename : str
        The name of the file without the extension.

    Returns
    -------
    None
    """
    final_path = pathlib.Path(save_path, filename).with_suffix(".pkl")
    print("Storing as:", final_path, end = "\r")
    with open(final_path, 'wb') as outp:
        joblib.dump(object, outp, compress='zlib')
        
def load_pkl(full_path):
    """
    Load a pickled object from the given full path and update its metadata.

    Parameters
    ----------
    full_path : str
        The full file path to the pickled object file.

    Returns
    -------
    object
        The loaded object with updated metadata.
    """
    with open(full_path, 'rb') as inp:
        object = joblib.load(inp)
        try:
            object.metadata["curr_path"] = full_path
        except AttributeError:
            warnings.warn(f"Object {full_path} has no metadata attribute")
        return object


def picklestore_objects(file_paths, output_folder, **kwargs):
    """
    Pickle store objects from given file paths to the specified output folder.

    Parameters
    ----------
    file_paths : str or Iterable
        A single file path or an iterable of file paths to be processed.
    output_folder : str
        The folder where the pickled objects will be stored.
    **kwargs : dict
        Arbitrary keyword arguments passed on to the loading function.

    Returns
    -------
    None
    """
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

def pickleload_objects(file_paths, **kwargs):
    """
    Load pickle objects from given file paths to the specified output folder.

    Parameters
    ----------
    file_paths : str or Iterable
        A single file path or an iterable of file paths to be processed.
    output_folder : str
        The folder where the pickled objects will be stored.
    **kwargs : dict
        Arbitrary keyword arguments passed on to the loading function.

    Returns
    -------
    None
    """
    if isinstance(file_paths, Iterable) is False:
        file_paths = [file_paths]
    output_list = []
    # progress_bar = alive_it(input_objects, spinner = "fishes", bar = 'blocks', calibrate = 50, force_tty=True)
    progress_bar = tqdm(file_paths, desc = "Iterating through and loading listed .pkl files as objects")
    with warnings.catch_warnings():
        out = Output()
        display(out)  # noqa: F821
        with out:
            for i in progress_bar:
                output_list.append(load_pkl(i))
                out.clear_output()
    return output_list

def copy_files(sources_list, target_path):
    """
    Move a file from one location to another.

    Parameters
    ----------
    source_path : str
        The source path of the file to be moved.
    target_path : str
        The target path where the file will be moved.

    Returns
    -------
    None
    """
    for i in sources_list:
        source_path = pathlib.Path(i)
        shutil.copy2(source_path, target_path)