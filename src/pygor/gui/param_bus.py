"""One source of truth for parameters, shared by every panel.

Each tab used to hold its own copy of the values it cared about, read
from the config once at construction. Editing a tab did not reach the
parameter table and editing the table did not reach the tabs, so the two
drifted apart with no indication which one the next run would use.

Everything now reads and writes ``recording.params`` through this bus,
which reports what changed so the other panels can follow.
"""

from qtpy.QtCore import QObject, Signal


class ParamBus(QObject):
    """Reads and writes ``AnalysisParams``, announcing every change."""

    #: Emitted with the dotted path of a parameter whose value changed
    changed = Signal(str)

    def __init__(self, params):
        super().__init__()
        self.params = params

    def get(self, path, default=None):
        """Value at a dotted path, or ``default`` if it is not there."""
        try:
            return self.params[path]
        except (KeyError, TypeError):
            return default

    def set(self, path, value):
        """Write a value, announcing it only if it actually changed.

        Returns True when something was written. Unchanged writes are
        dropped so that panels echoing each other cannot loop.
        """
        try:
            current = self.params[path]
        except (KeyError, TypeError):
            return False

        if type(current) is type(value) and current == value:
            return False

        try:
            self.params[path] = value
        except (KeyError, ValueError, TypeError):
            return False

        self.changed.emit(path)
        return True

    def section(self, path):
        """The dict at a dotted path, or an empty dict."""
        value = self.get(path)
        return value if isinstance(value, dict) else {}
