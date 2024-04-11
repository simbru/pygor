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

# Local imports


@dataclass
class Experiment:
    # Initialise object properties
    recording: list = field(default_factory=list)
    id_dict: dict = field(default_factory=lambda: defaultdict(list))

    def __exp_setter__(self, object):
        # if object.metadata["filename"] in self.id_dict["path"]:
        #     raise ValueError("Object already in experiment")
        # else:
            self.recording.append(object)
            self.id_dict["id"].append(len(self.id_dict["name"]))
            self.id_dict["date"].append(object.metadata["exp_date"].strftime('%d-%m-%Y'))        
            self.id_dict["name"].append(pathlib.Path(object.metadata["filename"]).stem)
            self.id_dict["num_rois"].append(object.num_rois)
            self.id_dict["type"].append(object.type)
            self.id_dict["path"].append(object.metadata["filename"])   

    def __exp_forgetter__(self, indices: int or list[int]):
        # Deal with recording list
        self.recording.pop(indices)
        # Deal with id_dict
        if isinstance(indices, Iterable) is False:
            _input = [indices]
        for k, v in self.id_dict.items():
            # Use list comprehension to remove elements at specified indices
            v[:] = [v[i] for i in range(len(v)) if i not in _input and i - len(v) not in _input]
    
    def attach_data(self, objects: object or list[object]):
        if isinstance(objects, Iterable) is False:
            self.__exp_setter__(objects)
        else:
            for i in objects:
                self.__exp_setter__(i)

    def detach_data(self, indices:int or list(int)):
        if isinstance(indices, Iterable) is False:
            self.__exp_forgetter__(indices)
        else:
            for i in indices:
                self.__exp_forgetter__(i)

    @property
    def recording_id(self):
        return pd.DataFrame(self.id_dict)
