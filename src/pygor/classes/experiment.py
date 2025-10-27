# Dependencies
from dataclasses import dataclass
from dataclasses import field

try:
    from collections import Iterable
except ImportError:
    from collections.abc import Iterable
from collections import defaultdict
import pandas as pd
import pathlib
import numpy as np
import joblib
# Local imports
import pygor.load


@dataclass
class Experiment:
    # Initialise object properties
    recording: list = field(default_factory=list)
    id_dict: dict = field(default_factory=lambda: defaultdict(list))
    # if isinstance(recording, dataclasses.Field) is False:
    #     raise AttributeError("Class input or 'self.recording' must be iterable")

    def __repr__(self):
        return f"Experiment(recording={self.recording}, id_dict={self.id_dict})"

    def __str__(self):
        return f"Experiment with {len(self.recording)} recordings and {len(self.id_dict)} entries in id_dict"

    def __post_init__(self):
        # Update recording and id_dict if data is passed directly during object creation
        if isinstance(self.recording, Iterable) is False:
            self.recording = [self.recording]
        self.__update_data__()

    @classmethod
    def from_files(cls, file_paths, pygor_class_name, n_jobs=-1, **class_kwargs):
        """
        Initialize an Experiment from a list of file paths.

        Parameters
        ----------
        file_paths : list or str
            List of file paths to load, or single file path
        pygor_class_name : str
            Name of the pygor class to use for loading (e.g., 'STRF', 'MovingBars', 'FullField')
        n_jobs : int, optional
            Number of parallel jobs for loading files. -1 uses all cores, 1 disables parallelization (default: -1)
        **class_kwargs : dict, optional
            Additional keyword arguments to pass to the pygor class constructor
            (e.g., dir_num=8 for OSDS data)

        Returns
        -------
        Experiment
            New Experiment object with loaded recordings

        Examples
        --------
        >>> # Load multiple STRF files in parallel
        >>> exp = Experiment.from_files(['file1.h5', 'file2.h5'], 'STRF')

        >>> # Load OSDS with dir_num
        >>> exp = Experiment.from_files('file.h5', 'OSDS', dir_num=8)

        >>> # Load with 4 parallel workers and additional parameters
        >>> exp = Experiment.from_files(file_list, 'OSDS', n_jobs=4, dir_num=8, dir_phase_num=2)
        """
        # Handle single file input
        if isinstance(file_paths, (str, pathlib.Path)):
            file_paths = [file_paths]
        
        # Get the pygor class
        try:
            pygor_class = getattr(pygor.load, pygor_class_name)
        except AttributeError:
            # Only list actual class names (types that start with uppercase)
            available_classes = [
                name for name in dir(pygor.load)
                if not name.startswith('_')
                and hasattr(pygor.load, name)
                and isinstance(getattr(pygor.load, name), type)
            ]
            raise ValueError(f"Unknown pygor class '{pygor_class_name}'. Available classes: {available_classes}")
        
        def load_single_file(file_path):
            """Helper function to load a single file"""
            try:
                recording = pygor_class(file_path, **class_kwargs)
                return ('success', file_path, recording)
            except Exception as e:
                return ('failed', file_path, str(e))
        
        # Load files in parallel or sequential
        if n_jobs == 1 or len(file_paths) == 1:
            # Sequential loading for single file or when explicitly requested
            results = [load_single_file(fp) for fp in file_paths]
        else:
            # Parallel loading
            from joblib import Parallel, delayed
            results = Parallel(n_jobs=n_jobs, verbose=1)(
                delayed(load_single_file)(file_path) for file_path in file_paths
            )
        
        # Process results
        recordings = []
        failed_files = []
        
        for status, file_path, result in results:
            if status == 'success':
                recordings.append(result)
                print(f"Loaded: {pathlib.Path(file_path).name}")
            else:
                failed_files.append((file_path, result))
                print(f"Failed to load {pathlib.Path(file_path).name}: {result}")
        
        if failed_files:
            print(f"\nWarning: {len(failed_files)} files failed to load:")
            for file_path, error in failed_files:
                print(f"  {pathlib.Path(file_path).name}: {error}")
        
        if not recordings:
            raise ValueError("No files could be loaded successfully")
        
        print(f"\nSuccessfully loaded {len(recordings)} recordings")
        
        # Create experiment with loaded recordings
        return cls(recording=recordings)

    def __update_data__(self):
        # Clear id_dict before updating to avoid duplicate entries
        self.id_dict.clear()
        # Update id_dict based on recording data
        for n, data in enumerate(self.recording):
            # Update id_dict accordingly
            self.__exp_dict_setter__(data)
            pass

    def __exp_dict_setter__(self, object):
        self.id_dict["id"].append(len(self.id_dict["name"]))
        self.id_dict["date"].append(object.metadata["exp_date"].strftime("%d-%m-%Y"))
        self.id_dict["name"].append(pathlib.Path(object.metadata["filename"]).stem)
        self.id_dict["num_rois"].append(object.num_rois)
        self.id_dict["type"].append(object.type)
        self.id_dict["path"].append(object.metadata["filename"])

    def __exp_list_setter__(self, object):
        self.recording.append(object)

    def __exp_setter__(self, object):
        # if object.metadata["filename"] in self.id_dict["path"]:
        #     raise ValueError("Object already in experiment")
        # else:
        self.__exp_dict_setter__(object)
        self.__exp_list_setter__(object)

    def __exp_forgetter__(self, indices: int or list[int]):
        if isinstance(indices, Iterable) is False:
            indices = [indices]
        # Deal with recording list
        for index in sorted(
            indices, reverse=True
        ):  # reverse because we want to remove from the end and back
            del self.recording[index]
            # Deal with id_dict
            for key in self.id_dict.keys():
                del self.id_dict[key][index]
        
        # Reset recording_id indices to match new logical order
        self.__update_data__()

    @property
    def recording_id(self):
        return pd.DataFrame(self.id_dict)

    def attach_data(self, objects: object or list[object]):
        if isinstance(objects, Iterable) is False:
            self.__exp_setter__(objects)
        else:
            for i in objects:
                self.__exp_setter__(i)
        print(f"Attached data: {objects}")

    def detach_data(self, indices):
        """
        Detach recordings by index or name.
        
        Parameters
        ----------
        indices : int, list of int, str, or list of str
            Recording indices or names to detach. If name not found, it's ignored.
        """
        # Handle string input (name-based detaching)
        if isinstance(indices, str):
            # Find recording index by name
            recording_names = [pathlib.Path(rec.metadata["filename"]).stem for rec in self.recording]
            try:
                index = recording_names.index(indices)
                indices = [index]
                print(f"Detaching data by name: {indices}")
            except ValueError:
                print(f"Recording '{indices}' not found - skipping")
                return
        elif isinstance(indices, list) and len(indices) > 0 and isinstance(indices[0], str):
            # Handle list of names
            recording_names = [pathlib.Path(rec.metadata["filename"]).stem for rec in self.recording]
            found_indices = []
            not_found = []
            for name in indices:
                try:
                    index = recording_names.index(name)
                    found_indices.append(index)
                except ValueError:
                    not_found.append(name)
            
            if not_found:
                print(f"Recordings not found (skipping): {not_found}")
            if not found_indices:
                print("No recordings found to detach")
                return
            
            indices = found_indices
            names_to_detach = [recording_names[i] for i in indices]
            print(f"Detaching data by names: {names_to_detach}")
        else:
            # Handle index-based detaching (original behavior)
            to_print = self.recording_id.iloc[indices]["name"]
            if isinstance(to_print, str):
                to_print = to_print
            if isinstance(to_print, pd.Series) or isinstance(to_print, np.ndarray):
                to_print = to_print.to_list()
            print(f"Detaching data: {to_print}")
        
        self.__exp_forgetter__(indices)

    def fetch_all(self, key: str, **kwargs):
        all_collated = []
        for i in self.recording:
            requested_attr = getattr(i, key)
            if hasattr(requested_attr, "__call__"):
                all_collated.append(requested_attr(**kwargs))
            else:
                all_collated.append(requested_attr)
        try:
            all_collated = np.ma.array(all_collated)
        except Exception as e:
            print(e)
            print("Returning as Numpy array failed, returning as list instead.")
        return all_collated

    def fetch(self, methods, as_dataframe=False, level='recording', include_metadata=True, **global_kwargs):
        """
        Flexibly fetch any methods/attributes from all recordings.
        
        Parameters
        ----------
        methods : list, dict, or mixed
            Method specifications in flexible formats:
            - List: ['method1', 'method2'] - uses global_kwargs for all
            - Dict: {'result_name': 'method_name'} or {'result_name': ('method_name', {'param': value})}
            - Mixed: [{'name': ('method', {'param': val})}, 'simple_method']
        as_dataframe : bool, optional
            If True, returns pandas DataFrame. If False, returns dict of lists (default: False)
        level : str, optional
            'recording' for one row per recording (default), 'roi' for one row per ROI
        include_metadata : bool, optional
            Whether to include recording metadata (default: True)
        **global_kwargs
            Default keyword arguments for methods that don't specify their own
            
        Returns
        -------
        dict or pandas.DataFrame
            If as_dataframe=False: dict with keys = result names, values = list of results
            If as_dataframe=True and level='recording': DataFrame with one row per recording
            If as_dataframe=True and level='roi': DataFrame with one row per ROI
            
        Examples
        --------
        >>> # Simple list usage
        >>> exp.fetch(['get_polarity_category_cell', 'num_rois'])
        
        >>> # Dictionary format - cleanest for complex parameters
        >>> exp.fetch({
        ...     'polarities': 'get_polarity_category_cell',
        ...     'masked_polarities': ('get_polarity_category_cell', {'mask_by_channel': True}),
        ...     'bool_channels': ('bool_by_channel', {'threshold': 2.0}),
        ...     'depths': 'ipl_depths'
        ... }, as_dataframe=True)
        
        >>> # ROI-level analysis for population studies
        >>> roi_data = exp.fetch({
        ...     'depths': 'ipl_depths',
        ...     'spatial_corr': 'spatial_correlation_index',
        w...     'category': 'get_polarity_category_cell'
        ... }, as_dataframe=True, level='roi')
        """
        
        # Handle different input formats
        if isinstance(methods, dict):
            method_specs = methods
            result_keys = list(methods.keys())
        elif isinstance(methods, list):
            # Convert list to dict format
            method_specs = {}
            result_keys = []
            for method in methods:
                if isinstance(method, dict):
                    # Handle mixed format: [{'name': 'method'}, 'simple']
                    method_specs.update(method)
                    result_keys.extend(method.keys())
                else:
                    # Simple string method
                    method_specs[method] = method
                    result_keys.append(method)
        else:
            raise ValueError("Methods must be a list or dictionary")
        
        results = {key: [] for key in result_keys}
        
        # Add metadata containers if requested
        if include_metadata and as_dataframe:
            metadata_keys = ['recording_id', 'name', 'date', 'type', 'num_rois']
            for key in metadata_keys:
                if key not in results:
                    results[key] = []
        
        # Fetch from each recording
        for rec_idx, recording in enumerate(self.recording):
            # Add metadata if requested
            if include_metadata and as_dataframe:
                results['recording_id'].append(rec_idx)
                results['name'].append(self.id_dict['name'][rec_idx])
                results['date'].append(self.id_dict['date'][rec_idx])
                results['type'].append(self.id_dict['type'][rec_idx])
                results['num_rois'].append(self.id_dict['num_rois'][rec_idx])
            
            # Fetch each requested method/attribute
            for result_name, method_spec in method_specs.items():
                try:
                    # Parse method specification
                    if isinstance(method_spec, tuple):
                        method_name, method_kwargs = method_spec
                    elif isinstance(method_spec, str):
                        method_name = method_spec
                        method_kwargs = global_kwargs
                    else:
                        raise ValueError(f"Invalid method spec for {result_name}: {method_spec}")
                    
                    # Get the attribute/method
                    attr = getattr(recording, method_name)
                    
                    # Call if it's a method, otherwise just get the attribute
                    if hasattr(attr, '__call__'):
                        try:
                            result = attr(**method_kwargs)
                        except TypeError:
                            # Method doesn't accept these kwargs, try without
                            result = attr()
                    else:
                        result = attr
                    
                    results[result_name].append(result)
                    
                except AttributeError:
                    print(f"Warning: {recording.type} object has no method/attribute '{method_name}' (recording_id: {rec_idx})")
                    results[result_name].append(None)
                except Exception as e:
                    print(f"Warning: Error calling {method_name} on {recording.name} (recording_id: {rec_idx}): {e}")
                    results[result_name].append(None)
        
        # Return as requested format
        if as_dataframe:
            df = pd.DataFrame(results)
            
            # Handle ROI-level analysis
            if level == 'roi':
                # Explode recording-level data into ROI-level data
                roi_rows = []
                
                for rec_idx, recording in enumerate(self.recording):
                    num_rois = recording.num_rois
                    
                    for roi_idx in range(num_rois):
                        roi_row = {}
                        
                        # Add ROI-specific metadata
                        if include_metadata:
                            roi_row['recording_id'] = rec_idx
                            roi_row['roi_id'] = roi_idx
                            roi_row['name'] = self.id_dict['name'][rec_idx]
                            roi_row['date'] = self.id_dict['date'][rec_idx]
                            roi_row['type'] = self.id_dict['type'][rec_idx]
                        
                        #TODO: add warning if longer than 1000 datapoints: likely not a tuning function

                        # Extract ROI-specific data from each method result
                        for result_name in result_keys:
                            result_data = df.loc[rec_idx, result_name]
                            
                            # Handle different result types
                            if result_data is None:
                                roi_row[result_name] = None
                            elif isinstance(result_data, pd.DataFrame) and len(result_data) == num_rois:
                                # DataFrame with ROI index - auto-unpack columns
                                for col_name in result_data.columns:
                                    column_name = f"{result_name}_{col_name}"
                                    roi_row[column_name] = result_data.iloc[roi_idx][col_name]
                            elif isinstance(result_data, (list, np.ndarray)):
                                handled = False
                                if hasattr(result_data, 'ndim'):
                                    # NumPy array - check dimensions
                                    if result_data.ndim == 1 and len(result_data) == num_rois:
                                        # Simple 1D case: (num_rois,)
                                        value = result_data[roi_idx]
                                        # Convert numpy scalars to Python types
                                        if isinstance(value, np.ndarray) and value.ndim == 0:
                                            value = value.item()
                                        elif isinstance(value, (np.float32, np.float64)):
                                            value = float(value)
                                        elif isinstance(value, (np.int32, np.int64)):
                                            value = int(value)
                                        roi_row[result_name] = value
                                        handled = True
                                    elif result_data.ndim == 2 and result_data.shape[1] == num_rois:
                                        # 2D case: (n_features, num_rois) - auto-unpack features
                                        for feature_idx in range(result_data.shape[0]):
                                            column_name = f"{result_name}_{feature_idx}"
                                            value = result_data[feature_idx, roi_idx]
                                            # Convert numpy scalars to Python types
                                            if isinstance(value, np.ndarray) and value.ndim == 0:
                                                value = value.item()
                                            elif isinstance(value, (np.float32, np.float64)):
                                                value = float(value)
                                            elif isinstance(value, (np.int32, np.int64)):
                                                value = int(value)
                                            roi_row[column_name] = value
                                        handled = True
                                    elif result_data.ndim == 2 and result_data.shape[0] == num_rois:
                                        # 2D case: (num_rois, n_features) - auto-unpack features
                                        for feature_idx in range(result_data.shape[1]):
                                            column_name = f"{result_name}_{feature_idx}"
                                            value = result_data[roi_idx, feature_idx]
                                            # Convert numpy scalars to Python types
                                            if isinstance(value, np.ndarray) and value.ndim == 0:
                                                value = value.item()
                                            elif isinstance(value, (np.float32, np.float64)):
                                                value = float(value)
                                            elif isinstance(value, (np.int32, np.int64)):
                                                value = int(value)
                                            roi_row[column_name] = value
                                        handled = True
                                else:
                                    # List - check if it's per-ROI
                                    if len(result_data) == num_rois:
                                        roi_row[result_name] = result_data[roi_idx]
                                        handled = True
                                
                                # Only fall back to complex data if not handled
                                if not handled:
                                    roi_row[result_name] = result_data
                            elif np.isscalar(result_data) or (isinstance(result_data, np.ndarray) and result_data.ndim == 0):
                                # Scalar data - duplicate for all ROIs
                                value = result_data
                                # Convert numpy scalars to Python types for better pandas compatibility
                                if isinstance(value, np.ndarray) and value.ndim == 0:
                                    value = value.item()
                                elif isinstance(value, (np.float32, np.float64)):
                                    value = float(value)
                                elif isinstance(value, (np.int32, np.int64)):
                                    value = int(value)
                                roi_row[result_name] = value
                            else:
                                # Complex data - keep as is (might be useful for some methods)
                                roi_row[result_name] = result_data
                        
                        roi_rows.append(roi_row)
                
                # Create the final DataFrame
                roi_df = pd.DataFrame(roi_rows)
                
                # Remove any original columns that were successfully unpacked into multiple columns
                columns_to_remove = []
                for result_name in result_keys:
                    # Check if we have unpacked columns for this result
                    unpacked_columns = [col for col in roi_df.columns if col.startswith(f"{result_name}_")]
                    if unpacked_columns and result_name in roi_df.columns:
                        # We successfully unpacked this column, so remove the original
                        columns_to_remove.append(result_name)
                
                # Drop the original columns that were unpacked
                if columns_to_remove:
                    roi_df = roi_df.drop(columns=columns_to_remove)
                
                return roi_df
            else:
                return df
        else:
            return results

    def fetch_averages(self):
        '''
        gets a numpy array with the average traces of all ROIs from multiple recordings
        contained in an pygor experiment object.
        
        Parameters
        ----------
        recordings: experiment object, containing data from one or more recordings 
        exported in .h5 format. 
        '''
        # compute lengths of recordings
        rec_lengths = [int(rec.averages.shape[1]) for rec in self.recording]
        min_trace_len = np.min(rec_lengths)
        print('trace lengths differ up to', np.max(rec_lengths)-np.min(rec_lengths), 'datapoints')
        print('truncating traces to', min_trace_len, 'datapoints')

        # Stack ROI traces across recordings into one 2D array (rows=ROIs, cols=time)
        rows = []

        for rec in self.recording:
            # truncate to min_trace_len along time axis
            truncated = rec.averages[:, :min_trace_len]
            # Append truncated ROI rows and mapping
            rows.append(truncated)
            
        return(np.vstack(rows))

    def pickle_store(self, save_path, filename, compress=False, protocol=None):
        """
        Store experiment as compressed pickle file.
        
        Parameters
        ----------
        save_path : str or Path
            Directory to save the file
        filename : str
            Base filename (extension will be added automatically)
        compress : tuple or False, optional
            Compression method. Options:
            - 0 or False: No compression (fastest, largest files)
            - ("lz4", 1): Fast compression
            - ("zlib", 3): Good balance of speed/size
            - ("zlib", 9): Maximum compression (slowest but smallest)
        protocol : int, optional
            Pickle protocol version. Higher versions are faster for numpy arrays.
            None uses joblib default (usually optimal).
        """
        final_path = pathlib.Path(save_path, filename).with_suffix(".pklexp")
        print("Storing as:", final_path, end="\r")
        
        # Use highest pickle protocol for speed with numpy arrays
        if protocol is None:
            import pickle
            protocol = pickle.HIGHEST_PROTOCOL
        
        with open(final_path, "wb") as outp:
            joblib.dump(self, outp, compress=compress, protocol=protocol)
