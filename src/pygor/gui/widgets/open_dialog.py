"""Ask which recording to open, as what, with which config.

The class decides which analyses a recording offers, and the config which
parameters its steps start from, so both belong at the moment of opening
rather than in code the user has to write first. A saved pygor object
already knows its class and its parameters, so for one of those the two
fields are disabled rather than silently ignored.
"""

import pathlib

from qtpy.QtWidgets import (
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QFormLayout,
    QHBoxLayout,
    QLineEdit,
    QPushButton,
    QSpinBox,
    QWidget,
)

from pygor.tui import reader

RECORDING_FILTER = "Recordings (*.h5 *.hdf5 *.smp);;All files (*)"
CONFIG_FILTER = "TOML config (*.toml);;All files (*)"


class OpenDialog(QDialog):
    """File, analysis class, config and colour count for one recording."""

    def __init__(self, parent=None, path=None, config=None, class_name="Core"):
        super().__init__(parent)
        self.setWindowTitle("Open recording")
        self.classes = reader.analysis_classes()

        self.path_edit = QLineEdit(str(path or ""))
        self.class_box = QComboBox()
        self.class_box.addItems(list(self.classes))
        if class_name in self.classes:
            self.class_box.setCurrentText(class_name)
        self.class_box.setToolTip(
            "Which pygor class to open a raw file as. It decides which "
            "analyses the recording offers."
        )
        self.config_edit = QLineEdit(str(config or ""))
        self.config_edit.setPlaceholderText("package defaults")
        self.colours_box = QSpinBox()
        self.colours_box.setRange(1, 8)
        self.colours_box.setToolTip("Number of interleaved colour channels")

        form = QFormLayout()
        form.addRow("File", _with_browse(self.path_edit, self._browse_path))
        form.addRow("Open as", self.class_box)
        form.addRow("Config", _with_browse(self.config_edit, self._browse_config))
        form.addRow("Colours", self.colours_box)
        self._form = form

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Open
            | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        self._open_button = buttons.button(QDialogButtonBox.StandardButton.Open)
        form.addRow(buttons)
        self.setLayout(form)

        self.path_edit.textChanged.connect(self._update)
        self.class_box.currentTextChanged.connect(self._update)
        self._update()

    @property
    def path(self):
        text = self.path_edit.text().strip()
        return pathlib.Path(text) if text else None

    @property
    def cls(self):
        return self.classes[self.class_box.currentText()]

    @property
    def config(self):
        return self.config_edit.text().strip() or None

    @property
    def n_colours(self):
        if not self._wants_colours:
            return None
        return self.colours_box.value()

    def _update(self, *_):
        """Enable only the fields that apply to this file and class."""
        path = self.path
        kind = reader.kind(path) if path else None
        saved = kind == "saved"
        self.class_box.setEnabled(not saved)
        self.config_edit.setEnabled(not saved)
        # Qt 5's QFormLayout has no setRowVisible, so the field and its
        # label are hidden separately
        self._wants_colours = not saved and reader.takes_n_colours(self.cls)
        self.colours_box.setVisible(self._wants_colours)
        self._form.labelForField(self.colours_box).setVisible(self._wants_colours)
        self._open_button.setEnabled(kind is not None and path.exists())

    def _browse_path(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Open recording", self.path_edit.text(), RECORDING_FILTER
        )
        if path:
            self.path_edit.setText(path)

    def _browse_config(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Config", self.config_edit.text(), CONFIG_FILTER
        )
        if path:
            self.config_edit.setText(path)


def _with_browse(edit, on_browse):
    """A line edit with a Browse button beside it."""
    button = QPushButton("Browse...")
    button.clicked.connect(on_browse)
    row = QWidget()
    layout = QHBoxLayout(row)
    layout.setContentsMargins(0, 0, 0, 0)
    layout.addWidget(edit)
    layout.addWidget(button)
    return row
