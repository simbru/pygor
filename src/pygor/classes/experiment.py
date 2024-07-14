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
import warnings
import numpy as np
import joblib
# Local imports
import pygor.filehandling

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
        self.id_dict["date"].append(object.metadata["exp_date"].strftime('%d-%m-%Y'))        
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
        for index in sorted(indices, reverse=True): # reverse because we want to remove from the end and back
            del self.recording[index]
            # Deal with id_dict
            for key in self.id_dict.keys():
                del self.id_dict[key][index]

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

    def detach_data(self, indices:int or list(int) or str):
        to_print = self.recording_id.iloc[indices]["name"]
        if isinstance(to_print, str):
            to_print = to_print
        if isinstance(to_print, pd.Series) or isinstance(to_print, np.ndarray):
            to_print = to_print.to_list()
        print(f"Detaching data: {to_print}")
        self.__exp_forgetter__(indices)

    def fetch_all(self, key:str, **kwargs):
        all_collated = []
        for i in self.recording:
            requested_attr = getattr(i, key)
            if hasattr(requested_attr, '__call__'):
                all_collated.append(requested_attr(**kwargs))
            else:
                all_collated.append(requested_attr)
        try:
            all_collated = np.ma.array(all_collated)
        except Exception as e:
            print(e)
            print("Returning as Numpy array failed, returning as list instead.")
        return all_collated

    def pickle_store(self, save_path, filename):
        final_path = pathlib.Path(save_path, filename).with_suffix(".pklexp")
        print("Storing as:", final_path, end = "\r")
        with open(final_path, 'wb') as outp:
            joblib.dump(self, outp, compress=('zlib', 1))