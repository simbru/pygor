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
    QTableWidget,
    QTableWidgetItem,
    QHeaderView,
    QVBoxLayout,
    QWidget,
    QLabel,
    QLineEdit,
    QAbstractItemView,
)
from qtpy.QtCore import Qt, Signal, QEventLoop
from qtpy.QtGui import QColor


# Type display names and colours for the type column
_TYPE_INFO = {
    int: ("int", QColor(100, 149, 237)),       # cornflower blue
    float: ("float", QColor(100, 149, 237)),
    bool: ("bool", QColor(80, 170, 80)),        # green
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
    """Editable parameter table widget.

    Shows a three-column table (Parameter, Value, Type) populated from
    an ``AnalysisParams`` instance. Edits to the Value column are written
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

        # Table
        self._table = QTableWidget(len(self._items), 3)
        self._table.setHorizontalHeaderLabels(["Parameter", "Value", "Type"])
        self._table.horizontalHeader().setSectionResizeMode(
            0, QHeaderView.Stretch
        )
        self._table.horizontalHeader().setSectionResizeMode(
            1, QHeaderView.ResizeToContents
        )
        self._table.horizontalHeader().setSectionResizeMode(
            2, QHeaderView.ResizeToContents
        )
        self._table.verticalHeader().setVisible(False)
        self._table.setSelectionBehavior(QAbstractItemView.SelectRows)
        # Qt's own alternating colours come from the widget palette, which
        # stays light even when the surrounding theme is dark - inside
        # napari that gave light text on a light band. A translucent grey
        # set per row composites over whatever base colour the theme uses,
        # so it works in either.
        self._table.setAlternatingRowColors(False)

        # Populate rows
        for row, (path, value) in enumerate(self._items):
            # Parameter path (read-only)
            path_item = QTableWidgetItem(path)
            path_item.setFlags(path_item.flags() & ~Qt.ItemIsEditable)
            self._table.setItem(row, 0, path_item)

            # Value (editable)
            val_item = QTableWidgetItem(repr(value))
            self._table.setItem(row, 1, val_item)

            # Type (read-only)
            type_name, type_colour = _type_label(value)
            type_item = QTableWidgetItem(type_name)
            type_item.setFlags(type_item.flags() & ~Qt.ItemIsEditable)
            type_item.setForeground(type_colour)
            self._table.setItem(row, 2, type_item)

            if row % 2:
                for column in range(3):
                    self._table.item(row, column).setBackground(_STRIPE_COLOUR)

        # Connect edit signal — fires when user finishes editing a cell
        self._table.cellChanged.connect(self._on_cell_changed)

        layout.addWidget(self._table)

        # Status bar
        self._status = QLabel(f"{len(self._items)} parameters")
        self._status.setStyleSheet("color: grey; font-size: 11px;")
        layout.addWidget(self._status)

    def _on_cell_changed(self, row, col):
        """Write edited value back to params when user clicks away."""
        if col != 1:
            return

        path = self._items[row][0]
        original_value = self._items[row][1]
        new_text = self._table.item(row, 1).text()

        try:
            parsed = _parse_value(new_text, original_value)
            self.params[path] = parsed
            # Update our local cache
            self._items[row] = (path, parsed)
            # Update display to show canonical repr
            self._table.blockSignals(True)
            self._table.item(row, 1).setText(repr(parsed))
            self._table.blockSignals(False)
            self._status.setText(f"Set {path} = {parsed!r}")
            self._status.setStyleSheet("color: green; font-size: 11px;")
        except (ValueError, KeyError) as e:
            # Revert to original value
            self._table.blockSignals(True)
            self._table.item(row, 1).setText(repr(original_value))
            self._table.blockSignals(False)
            self._status.setText(f"Error: {e}")
            self._status.setStyleSheet("color: red; font-size: 11px;")

    def _apply_filter(self, text):
        """Show/hide rows based on filter text."""
        text = text.lower()
        for row in range(self._table.rowCount()):
            path = self._items[row][0].lower()
            self._table.setRowHidden(row, text not in path)

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
