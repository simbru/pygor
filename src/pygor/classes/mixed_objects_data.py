from dataclasses import dataclass
from pygor.classes.core_data import Core
import numpy as np

@dataclass(kw_only=True, repr=False)
class MixedObjects(Core):
    stimtypes: np.ndarray = np.nan

