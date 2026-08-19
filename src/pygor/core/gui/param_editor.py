"""Interactive parameter editor for pygor AnalysisParams.

Provides an IGOR-style editable table for viewing and modifying
analysis parameters per-recording. Requires Qt (available via the
``[gui]`` extra which pulls in napari/qtpy).

Usage
-----
>>> rec.params.edit()                    # edit all params
>>> rec.params.edit("segmentation")      # edit just segmentation
"""

from qtpy.QtWidgets import (
    QApplication,
    QHeaderView,
    QVBoxLayout,
    QWidget,
    QLabel,
    QLineEdit,
    QAbstractItemView,
    QTreeWidget,
    QTreeWidgetItem,
)
from qtpy.QtCore import Qt, Signal, QEventLoop
from qtpy.QtGui import QColor


# Type display names and colours for the type column
# bool must precede int: bool is a subclass of int, so checking int first
# labels every True/False as an int
_TYPE_INFO = {
    bool: ("bool", QColor(80, 170, 80)),        # green
    int: ("int", QColor(100, 149, 237)),       # cornflower blue
    float: ("float", QColor(100, 149, 237)),
    str: ("str", QColor(160, 160, 160)),         # grey
    list: ("list", QColor(180, 140, 80)),        # amber
}


# Neutral grey at low alpha: lightens a dark theme, darkens a light one
_STRIPE_COLOUR = QColor(128, 128, 128, 38)


def _type_label(value):
    """Return (type_name, colour) for a value."""
    for typ, info in _TYPE_INFO.items():
        if isinstance(value, typ):
            return info
    return (type(value).__name__, QColor(160, 160, 160))


def _parse_value(text, original_value):
    """Parse edited text back to the original type.

    Parameters
    ----------
    text : str
        The edited string from the table cell.
    original_value
        The original value, used to infer the target type.

    Returns
    -------
    parsed_value
        The text parsed to the original type.

    Raises
    ------
    ValueError
        If the text cannot be parsed to the expected type.
    """
    if isinstance(original_value, bool):
        lower = text.strip().lower()
        if lower in ("true", "1", "yes"):
            return True
        elif lower in ("false", "0", "no"):
            return False
        raise ValueError(f"Cannot parse '{text}' as bool")

    if isinstance(original_value, int):
        # Allow float-like strings that are whole numbers
        f = float(text)
        if f == int(f):
            return int(f)
        return int(f)

    if isinstance(original_value, float):
        return float(text)

    if isinstance(original_value, list):
        import ast
        parsed = ast.literal_eval(text)
        if not isinstance(parsed, list):
            raise ValueError(f"Expected a list, got {type(parsed).__name__}")
        return parsed

    # str or unknown — return as-is
    return text


class ParamEditorWidget(QWidget):
    """Editable parameter tree widget.

    Shows three columns (Parameter, Value, Type) populated from an
    ``AnalysisParams`` instance, grouped into collapsible sections by the
    first part of each dotted path. Edits to the Value column are written
    back to ``params._defaults`` immediately when the user clicks away
    from the cell (matching IGOR's behaviour).

    Parameters
    ----------
    params : AnalysisParams
        The params object to edit. Changes are written in-place.
    section : str or None
        If given, only show parameters under this top-level section.
    title : str
        Window title.
    """

    closed = Signal()

    def __init__(self, params, section=None, title="Parameter Editor"):
        super().__init__()
        self.params = params
        self.section = section
        self.setWindowTitle(title)
        self.setMinimumSize(550, 400)
        self.resize(600, 700)

        # Build flat list of (dotted_path, value)
        if section is not None:
            if section not in params._defaults:
                raise KeyError(
                    f"Unknown section '{section}'. "
                    f"Available: {list(params._defaults.keys())}"
                )
            self._items = params._flatten_defaults(section, params._defaults[section])
        else:
            self._items = params._flatten_defaults()

        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)

        # Filter bar
        self._filter = QLineEdit()
        self._filter.setPlaceholderText("Filter parameters...")
        self._filter.textChanged.connect(self._apply_filter)
        layout.addWidget(self._filter)

        self._tree = QTreeWidget()
        self._tree.setColumnCount(3)
        self._tree.setHeaderLabels(["Parameter", "Value", "Type"])
        self._tree.header().setSectionResizeMode(0, QHeaderView.Stretch)
        self._tree.header().setSectionResizeMode(1, QHeaderView.ResizeToContents)
        self._tree.header().setSectionResizeMode(2, QHeaderView.ResizeToContents)
        self._tree.setSelectionBehavior(QAbstractItemView.SelectRows)
        self._tree.setRootIsDecorated(True)
        self._tree.setUniformRowHeights(True)

        # path -> leaf item, so edits can be traced back to a parameter
        self._items_by_path = {}
        self._groups = {}

        for path, value in self._items:
            group = self._group_for(path)
            leaf = QTreeWidgetItem(group)
            leaf.setText(0, path.split(".", 1)[-1] if "." in path else path)
            leaf.setText(1, repr(value))
            type_name, type_colour = _type_label(value)
            leaf.setText(2, type_name)
            leaf.setForeground(2, type_colour)
            # Only the value is editable; the path and type are labels
            leaf.setFlags(leaf.flags() | Qt.ItemIsEditable)
            leaf.setData(0, Qt.UserRole, path)
            self._items_by_path[path] = leaf

        self._tree.expandAll()
        self._tree.itemChanged.connect(self._on_item_changed)
        layout.addWidget(self._tree)

        # Status bar
        self._status = QLabel(
            f"{len(self._items)} parameters in {len(self._groups)} sections"
        )
        self._status.setStyleSheet("color: grey; font-size: 11px;")
        layout.addWidget(self._status)

    def _group_for(self, path):
        """Return the collapsible section a dotted path belongs to."""
        name = path.split(".", 1)[0] if "." in path else "general"
        if name not in self._groups:
            group = QTreeWidgetItem(self._tree)
            group.setText(0, name)
            group.setFirstColumnSpanned(True)
            # Not editable, and not mistakable for a parameter row
            group.setFlags(Qt.ItemIsEnabled)
            for column in range(3):
                group.setBackground(column, _STRIPE_COLOUR)
            self._groups[name] = group
        return self._groups[name]

    def _index_of(self, path):
        for index, (item_path, _) in enumerate(self._items):
            if item_path == path:
                return index
        return -1

    def _on_item_changed(self, item, column):
        """Write an edited value back to params when the user clicks away."""
        if column != 1:
            return
        path = item.data(0, Qt.UserRole)
        if path is None:
            return

        index = self._index_of(path)
        original_value = self._items[index][1]

        self._tree.blockSignals(True)
        try:
            parsed = _parse_value(item.text(1), original_value)
            self.params[path] = parsed
            self._items[index] = (path, parsed)
            # Show the canonical repr rather than whatever was typed
            item.setText(1, repr(parsed))
            self._status.setText(f"Set {path} = {parsed!r}")
            self._status.setStyleSheet("color: green; font-size: 11px;")
        except (ValueError, KeyError) as e:
            item.setText(1, repr(original_value))
            self._status.setText(f"Error: {e}")
            self._status.setStyleSheet("color: red; font-size: 11px;")
        finally:
            self._tree.blockSignals(False)

    def _apply_filter(self, text):
        """Hide parameters that do not match, and sections left empty."""
        text = text.lower()
        for path, leaf in self._items_by_path.items():
            leaf.setHidden(text not in path.lower())

        for name, group in self._groups.items():
            visible = any(
                not group.child(i).isHidden() for i in range(group.childCount())
            )
            group.setHidden(not visible)
            # Matching a section name should reveal what is inside it
            if visible and text:
                group.setExpanded(True)

    def closeEvent(self, event):
        """Emit closed signal for optional blocking callers."""
        super().closeEvent(event)
        self.closed.emit()


def _ensure_qt_event_loop():
    """Enable Qt event loop integration in IPython if available.

    Equivalent to running ``%gui qt`` — ensures Qt windows appear and
    stay responsive without the user needing to know about the magic.
    """
    try:
        ipy = get_ipython()
        if ipy is not None:
            ipy.run_line_magic("gui", "qt")
    except (NameError, ImportError):
        pass


def launch_editor(params, section=None, title="Parameter Editor", blocking=True):
    """Launch the parameter editor window.

    Parameters
    ----------
    params : AnalysisParams
        The params object to edit in-place.
    section : str or None
        Filter to a top-level section (e.g. "segmentation").
    title : str
        Window title.
    blocking : bool, default True
        If True, the call blocks until the editor window is closed.
        If False, the window opens and control returns immediately.

    Returns
    -------
    ParamEditorWidget
        The widget instance (kept alive so the window persists).
    """
    _ensure_qt_event_loop()

    app = QApplication.instance()
    if app is None:
        app = QApplication([])

    widget = ParamEditorWidget(params, section=section, title=title)
    widget.show()
    widget.raise_()

    if blocking:
        loop = QEventLoop()
        widget.closed.connect(loop.quit)
        loop.exec_()

    return widget
