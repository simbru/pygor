from pygor.classes.core_data import Core
from dataclasses import dataclass
import numpy as np


@dataclass(kw_only=True, repr=False)
class MovingBars(Core):
    dir_num: int
    col_num: int

    def __post_init__(self):
        # Post initialise the contents of Data class to be inherited
        super().__dict__["data_types"].append(self.type)
        super().__post_init__()

    def split_snippets_chromatically(self) -> np.ndarray:
        "Returns snippets split by chromaticity, expect one more dimension than the averages array (repetitions)"
        return np.array(np.split(self.snippets[:, :, 1:], self.col_num, axis=-1))

    def split_averages_chromatically(self) -> np.ndarray:
        "Returns averages split by chromaticity"
        return np.array(np.split(self.averages[:, 1:], self.col_num, axis=-1))
