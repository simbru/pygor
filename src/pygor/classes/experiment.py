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
        raise NotImplementedError("Bug in the code, not implemented yet")
        # Deal with recording list
        for index in sorted(indices, reverse=True): # reverse because we want to remove from the end and back
            del self.recording[index]
            # Deal with id_dict
            for key, value in self.id_dict.items():
                print(self.id_dict[key])
                del self.id_dict[key][index]
        
        # if isinstance(indices, Iterable) is False:
        #     _input = [indices]
        # else:
        #     _input = indices
        # for k, v in self.id_dict.items():
        #     # Use list comprehension to remove elements at specified indices
        #     v[:] = [v[i] for i in range(len(v)) if i not in _input and i - len(v) not in _input]
            #print(k, v)

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

    def detach_data(self, indices:int or list(int)):
        """
        TODO Account for moving indices when detaching
        """
        # if isinstance(indices, Iterable) is False:
        #     self.__exp_forgetter__(indices)
        # else:
        #     for i in indices:
        #         self.__exp_forgetter__(i)
        self.__exp_forgetter__(indices)
        to_print = self.recording_id.iloc[indices]["name"]
        if isinstance(to_print, Iterable):
            to_print = to_print.to_list()
        print(f"Detached data: {to_print}")
    
    def df_strf_chromatic(self):
        roi_list = []
        rec_list = []
        chroma_list = []
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=RuntimeWarning)
            warnings.filterwarnings("ignore", category=UserWarning)
            for obj in self.recording:
                print("Current object:", obj.name)
                

        # Create dataframe to store results
        # pygor.filehandling.compile_chroma_strf_df(self.)