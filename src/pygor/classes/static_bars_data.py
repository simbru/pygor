from pygor.classes.core_data import Core
from dataclasses import dataclass
import numpy as np


@dataclass(kw_only=True, repr=False)
class StaticBars(Core):

    def __post_init__(self):
        
        # # Post initialise the contents of Data class to be inherited
        # super().__dict__["data_types"].append(self.type)
        super().__post_init__()
        # None
