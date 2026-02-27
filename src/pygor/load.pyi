# Stub file for pygor.load
# This module uses dynamic imports at runtime, so we declare exports here for type checkers

from pygor.classes.experiment import Experiment as Experiment
from pygor.classes.osds_data import OSDS as OSDS
from pygor.classes.strf_data import STRF as STRF
from pygor.classes.core_data import Core as Core
from pygor.classes.centre_surround_data import CenterSurround as CenterSurround
from pygor.classes.full_field_data import FullField as FullField
from pygor.classes.response_mapping_data import ResponseMapping as ResponseMapping
from pygor.classes.static_bars_data import StaticBars as StaticBars

# Backward compatibility alias
MovingBars = OSDS
