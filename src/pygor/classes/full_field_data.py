from dataclasses import dataclass, field
from pygor.classes.core_data import Core
import numpy as np


@dataclass(kw_only=False, repr=False) #Decide if kw_only is necessary
class FullField(Core):
    ipl_depths: np.ndarray = np.nan
    # Post init attrs
    name: str = field(init=False)
    averages: np.array = field(init=False)
    ms_dur: int = field(init=False)

    def __post_init__(self):
        # Post initialise the contents of Data class to be inherited
        super().__dict__["data_types"].append(self.type)
        super().__post_init__()
        # with h5py.File(self.filename) as HDF5_file:
        #     # Initilaise object properties
        #     self.name = self.filename.stem
        #     #self.averages = np.copy(HDF5_file["Averages0"])
        #     # self.raw_traces = np.copy(HDF5_file["Averages0"])
        #     self.ms_dur = self.averages.shape[-1]
