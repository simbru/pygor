# Dependencies
from dataclasses import dataclass
from dataclasses import field

try:
    from collections import Iterable
except ImportError:
    from collections.abc import Iterable
from collections import defaultdict
import os
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
        Detach recordings by index, name, or full path.
        
        Parameters
        ----------
        indices : int, list of int, str, or list of str
            Recording indices, names, or full paths to detach. If not found, it's ignored.
        """
        recording_paths = [str(pathlib.Path(rec.metadata["filename"])) for rec in self.recording]
        recording_paths_norm = [os.path.normcase(os.path.normpath(p)) for p in recording_paths]
        recording_names = [pathlib.Path(rec.metadata["filename"]).stem for rec in self.recording]

        def _is_pathlike(value):
            return isinstance(value, (str, pathlib.Path, os.PathLike, np.str_))

        def _to_str(value):
            if isinstance(value, (pathlib.Path, os.PathLike)):
                return str(value)
            return str(value)

        # Handle string input (name-based detaching)
        if _is_pathlike(indices):
            target_norm = os.path.normcase(os.path.normpath(indices))
            if target_norm in recording_paths_norm:
                index = recording_paths_norm.index(target_norm)
                indices = [index]
                print(f"Detaching data by path: {recording_paths[index]}")
            else:
                # Find recording index by name
                try:
                    name = pathlib.Path(_to_str(indices)).stem
                    index = recording_names.index(name)
                    if recording_names.count(name) > 1:
                        print(f"Multiple recordings match name '{name}'. Use full path to disambiguate.")
                    indices = [index]
                    print(f"Detaching data by name: {indices}")
                except ValueError:
                    print(f"Recording '{indices}' not found - skipping")
                    return
        elif isinstance(indices, list) and len(indices) > 0 and _is_pathlike(indices[0]):
            # Handle list of names
            found_indices = []
            not_found = []
            for name_or_path in indices:
                name_or_path_str = _to_str(name_or_path)
                target_norm = os.path.normcase(os.path.normpath(name_or_path_str))
                if target_norm in recording_paths_norm:
                    index = recording_paths_norm.index(target_norm)
                    found_indices.append(index)
                else:
                    try:
                        name = pathlib.Path(name_or_path_str).stem
                        index = recording_names.index(name)
                        if recording_names.count(name) > 1:
                            print(f"Multiple recordings match name '{name}'. Use full path to disambiguate.")
                        found_indices.append(index)
                    except ValueError:
                        not_found.append(name_or_path_str)
            
            if not_found:
                print(f"Recordings not found (skipping): {not_found}")
            if not found_indices:
                print("No recordings found to detach")
                return
            
            indices = found_indices
            names_to_detach = [recording_names[i] for i in indices]
            print(f"Detaching data by names/paths: {names_to_detach}")
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

    def fetch(
        self,
        methods,
        as_dataframe=False,
        level="recording",
        include_metadata=True,
        unpack_ndarrays=False,
        unpack_axis=0,
        concat_unpacked=False,
        **global_kwargs,
    ):
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
        unpack_ndarrays : bool, optional
            If True and as_dataframe=False, unpack 2D ndarray outputs into separate keys
            (default: False)
        unpack_axis : int, optional
            Feature axis for unpacking 2D arrays. 0 expects (n_features, n_rois),
            1 expects (n_rois, n_features) (default: 0)
        concat_unpacked : bool, optional
            If True, concatenate per-recording arrays into a single 1D array per feature.
            If False, keep per-recording lists (default: False)
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
        method_specs, result_keys, return_single = self._parse_method_specs(methods)

        if unpack_axis not in (0, 1):
            raise ValueError("unpack_axis must be 0 or 1")

        results = {key: [] for key in result_keys}

        # Add metadata containers if requested
        if include_metadata and as_dataframe:
            self._init_metadata_results(results)

        # Fetch from each recording
        for rec_idx, recording in enumerate(self.recording):
            # Add metadata if requested
            if include_metadata and as_dataframe:
                self._append_recording_metadata(results, rec_idx)

            # Fetch each requested method/attribute
            for result_name, method_spec in method_specs.items():
                result = self._call_recording_method(
                    recording,
                    result_name,
                    method_spec,
                    rec_idx,
                    global_kwargs,
                )
                results[result_name].append(result)
        
        # Return as requested format
        if as_dataframe:
            df = pd.DataFrame(results)
            
            # Handle ROI-level analysis
            if level == 'roi':
                # Explode recording-level data into ROI-level data
                roi_rows = self._build_roi_rows(
                    df,
                    result_keys,
                    include_metadata=include_metadata,
                )
                roi_df = pd.DataFrame(roi_rows)
                roi_df = self._drop_unpacked_columns(roi_df, result_keys)
                return roi_df
            else:
                return df
        else:
            if unpack_ndarrays:
                results = self._unpack_ndarray_results(
                    results,
                    unpack_axis=unpack_axis,
                    concat_unpacked=concat_unpacked,
                )
            if return_single and not self._has_unpacked_keys(results, result_keys[0]):
                return results[result_keys[0]]
            return results

    def _parse_method_specs(self, methods):
        """
        Normalize method specs and determine if single-result shortcut applies.
        """
        return_single = False
        if isinstance(methods, str):
            method_specs = {methods: methods}
            result_keys = [methods]
            return_single = True
        elif isinstance(methods, dict):
            method_specs = methods
            result_keys = list(methods.keys())
        elif isinstance(methods, list):
            method_specs = {}
            result_keys = []
            for method in methods:
                if isinstance(method, dict):
                    method_specs.update(method)
                    result_keys.extend(method.keys())
                else:
                    method_specs[method] = method
                    result_keys.append(method)
        else:
            raise ValueError("Methods must be a list or dictionary")
        return method_specs, result_keys, return_single

    def _init_metadata_results(self, results):
        metadata_keys = ["recording_id", "name", "date", "type", "num_rois"]
        for key in metadata_keys:
            if key not in results:
                results[key] = []

    def _append_recording_metadata(self, results, rec_idx):
        results["recording_id"].append(rec_idx)
        results["name"].append(self.id_dict["name"][rec_idx])
        results["date"].append(self.id_dict["date"][rec_idx])
        results["type"].append(self.id_dict["type"][rec_idx])
        results["num_rois"].append(self.id_dict["num_rois"][rec_idx])

    def _call_recording_method(
        self,
        recording,
        result_name,
        method_spec,
        rec_idx,
        global_kwargs,
    ):
        try:
            method_name, method_kwargs = self._parse_method_spec(method_spec, result_name, global_kwargs)
            attr = getattr(recording, method_name)
            if hasattr(attr, "__call__"):
                try:
                    return attr(**method_kwargs)
                except TypeError:
                    return attr()
            return attr
        except AttributeError:
            print(
                f"Warning: {recording.type} object has no method/attribute "
                f"'{method_name}' (recording_id: {rec_idx})"
            )
            return None
        except Exception as e:
            print(
                f"Warning: Error calling {method_name} on {recording.name} "
                f"(recording_id: {rec_idx}): {e}"
            )
            return None

    def _parse_method_spec(self, method_spec, result_name, global_kwargs):
        if isinstance(method_spec, tuple):
            return method_spec[0], method_spec[1]
        if isinstance(method_spec, str):
            return method_spec, global_kwargs
        raise ValueError(f"Invalid method spec for {result_name}: {method_spec}")

    def _build_roi_rows(self, df, result_keys, include_metadata=True):
        roi_rows = []
        for rec_idx, recording in enumerate(self.recording):
            num_rois = recording.num_rois
            for roi_idx in range(num_rois):
                roi_row = {}
                if include_metadata:
                    self._append_roi_metadata(roi_row, rec_idx, roi_idx)
                # TODO: add warning if longer than 1000 datapoints: likely not a tuning function
                for result_name in result_keys:
                    result_data = df.loc[rec_idx, result_name]
                    self._add_roi_result(roi_row, result_name, result_data, num_rois, roi_idx)
                roi_rows.append(roi_row)
        return roi_rows

    def _append_roi_metadata(self, roi_row, rec_idx, roi_idx):
        roi_row["recording_id"] = rec_idx
        roi_row["roi_id"] = roi_idx
        roi_row["name"] = self.id_dict["name"][rec_idx]
        roi_row["date"] = self.id_dict["date"][rec_idx]
        roi_row["type"] = self.id_dict["type"][rec_idx]

    def _add_roi_result(self, roi_row, result_name, result_data, num_rois, roi_idx):
        if result_data is None:
            roi_row[result_name] = None
            return
        if isinstance(result_data, pd.DataFrame) and len(result_data) == num_rois:
            for col_name in result_data.columns:
                column_name = f"{result_name}_{col_name}"
                roi_row[column_name] = result_data.iloc[roi_idx][col_name]
            return
        if isinstance(result_data, (list, np.ndarray)):
            handled = self._try_unpack_array(roi_row, result_name, result_data, num_rois, roi_idx)
            if not handled:
                roi_row[result_name] = result_data
            return
        if np.isscalar(result_data) or (isinstance(result_data, np.ndarray) and result_data.ndim == 0):
            roi_row[result_name] = self._coerce_numpy_scalar(result_data)
            return
        roi_row[result_name] = result_data

    def _try_unpack_array(self, roi_row, result_name, result_data, num_rois, roi_idx):
        if hasattr(result_data, "ndim"):
            if result_data.ndim == 1 and len(result_data) == num_rois:
                roi_row[result_name] = self._coerce_numpy_scalar(result_data[roi_idx])
                return True
            if result_data.ndim == 2 and result_data.shape[1] == num_rois:
                for feature_idx in range(result_data.shape[0]):
                    column_name = f"{result_name}_{feature_idx}"
                    roi_row[column_name] = self._coerce_numpy_scalar(
                        result_data[feature_idx, roi_idx]
                    )
                return True
            if result_data.ndim == 2 and result_data.shape[0] == num_rois:
                for feature_idx in range(result_data.shape[1]):
                    column_name = f"{result_name}_{feature_idx}"
                    roi_row[column_name] = self._coerce_numpy_scalar(
                        result_data[roi_idx, feature_idx]
                    )
                return True
            return False
        if len(result_data) == num_rois:
            roi_row[result_name] = result_data[roi_idx]
            return True
        return False

    def _coerce_numpy_scalar(self, value):
        if isinstance(value, np.ndarray) and value.ndim == 0:
            return value.item()
        if isinstance(value, (np.float32, np.float64)):
            return float(value)
        if isinstance(value, (np.int32, np.int64)):
            return int(value)
        return value

    def _drop_unpacked_columns(self, roi_df, result_keys):
        columns_to_remove = []
        for result_name in result_keys:
            unpacked_columns = [col for col in roi_df.columns if col.startswith(f"{result_name}_")]
            if unpacked_columns and result_name in roi_df.columns:
                columns_to_remove.append(result_name)
        if columns_to_remove:
            roi_df = roi_df.drop(columns=columns_to_remove)
        return roi_df

    def _unpack_ndarray_results(self, results, unpack_axis=0, concat_unpacked=False):
        """
        Unpack 2D ndarray results into separate feature keys.
        """
        new_results = {}
        for key, values in results.items():
            if not values:
                new_results[key] = values
                continue

            non_null = [v for v in values if v is not None]
            if not non_null:
                new_results[key] = values
                continue

            arrays = [
                v for v in non_null if isinstance(v, np.ndarray) and v.ndim == 2
            ]
            if len(arrays) != len(non_null):
                new_results[key] = values
                continue

            # Determine n_features from the first array
            n_features = arrays[0].shape[0 if unpack_axis == 0 else 1]
            if any(
                a.shape[0 if unpack_axis == 0 else 1] != n_features for a in arrays
            ):
                new_results[key] = values
                continue

            for feat_idx in range(n_features):
                feature_key = f"{key}_{feat_idx}"
                series = []
                for value in values:
                    if value is None:
                        series.append(None)
                    elif unpack_axis == 0:
                        series.append(value[feat_idx])
                    else:
                        series.append(value[:, feat_idx])
                if concat_unpacked:
                    concatenable = [v for v in series if v is not None]
                    new_results[feature_key] = (
                        np.concatenate(concatenable) if concatenable else np.array([])
                    )
                else:
                    new_results[feature_key] = series
        return new_results

    def _has_unpacked_keys(self, results, base_key):
        prefix = f"{base_key}_"
        return any(key.startswith(prefix) for key in results.keys())

    def fetch_raw(self, method, **kwargs):
        """
        Fetch raw results from all recordings without any transformations.

        This is the simplest fetch method - returns exactly what each recording
        returns, as a list. No shape detection, no DataFrame conversion, no
        unpacking.

        Parameters
        ----------
        method : str
            Method or attribute name to fetch from each recording
        **kwargs
            Keyword arguments to pass to the method

        Returns
        -------
        list
            List of raw results from each recording (one per recording).
            Results may have heterogeneous shapes/types. Recordings where
            the method fails or doesn't exist will have None in their position.

        Examples
        --------
        >>> # Get raw DSI arrays (may have different shapes if phase_aware differs)
        >>> dsi_list = exp.fetch_raw('get_dsi', metric='peak')

        >>> # Get raw tuning metrics dicts
        >>> metrics_list = exp.fetch_raw('compute_tuning_metrics')

        >>> # Use phase_idx for consistent shapes
        >>> dsi_phase0 = exp.fetch_raw('get_dsi', phase_idx=0)
        """
        results = []
        for rec_idx, recording in enumerate(self.recording):
            try:
                attr = getattr(recording, method)
                if callable(attr):
                    result = attr(**kwargs)
                else:
                    result = attr
                results.append(result)
            except AttributeError:
                print(
                    f"Warning: {recording.type} has no attribute '{method}' "
                    f"(recording_id: {rec_idx})"
                )
                results.append(None)
            except Exception as e:
                print(
                    f"Warning: Error calling {method} on {recording.name} "
                    f"(recording_id: {rec_idx}): {e}"
                )
                results.append(None)
        return results

    def fetch_concat(self, method, type_filter=None, axis=0, **kwargs):
        """
        Fetch and concatenate results from recordings into a single array.

        This method fetches results from all (or filtered) recordings and
        concatenates them along the specified axis. Useful for population
        analyses where you want all ROIs pooled together.

        Parameters
        ----------
        method : str
            Method or attribute name to fetch from each recording
        type_filter : str or list of str, optional
            Only include recordings of specified type(s). E.g., 'OSDS' or
            ['OSDS', 'MovingBars']. If None, includes all recordings.
        axis : int, optional
            Axis along which to concatenate results (default: 0, typically ROI axis)
        **kwargs
            Keyword arguments to pass to the method

        Returns
        -------
        np.ndarray or None
            Concatenated array of results, or None if all results are None/empty

        Raises
        ------
        ValueError
            If results have incompatible shapes for concatenation

        Examples
        --------
        >>> # Get all DSI values pooled across OSDS recordings
        >>> all_dsi = exp.fetch_concat('get_dsi', type_filter='OSDS',
        ...                             metric='peak', phase_idx=0)

        >>> # Get all ROI depths pooled
        >>> all_depths = exp.fetch_concat('ipl_depths')

        >>> # Filter by multiple types
        >>> data = exp.fetch_concat('averages', type_filter=['OSDS', 'FullField'])

        Notes
        -----
        - Recordings returning None are skipped
        - For consistent results with phase-aware OSDS data, use phase_idx
          parameter in the method kwargs
        """
        # Normalize type_filter to list
        if type_filter is not None:
            if isinstance(type_filter, str):
                type_filter = [type_filter]

        results = []
        for rec_idx, recording in enumerate(self.recording):
            # Apply type filter
            if type_filter is not None and recording.type not in type_filter:
                continue

            try:
                attr = getattr(recording, method)
                if callable(attr):
                    result = attr(**kwargs)
                else:
                    result = attr

                if result is not None:
                    # Ensure result is array-like for concatenation
                    results.append(np.asarray(result))

            except AttributeError:
                print(
                    f"Warning: {recording.type} has no attribute '{method}' "
                    f"(recording_id: {rec_idx})"
                )
            except Exception as e:
                print(
                    f"Warning: Error calling {method} on {recording.name} "
                    f"(recording_id: {rec_idx}): {e}"
                )

        if not results:
            return None

        try:
            return np.concatenate(results, axis=axis)
        except ValueError as e:
            shapes = [r.shape for r in results]
            raise ValueError(
                f"Cannot concatenate results with shapes {shapes} along axis {axis}. "
                f"Use phase_idx parameter for consistent OSDS output shapes."
            ) from e

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
