from . import calculations
from .calculations import compute_correlation_projection
from .trace_extraction import extract_traces, znorm_traces

# GUI methods are imported lazily to avoid requiring napari/pyqt5
# Access them via: import pygor.core.gui.methods
