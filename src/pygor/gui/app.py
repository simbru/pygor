"""``pygor-gui`` -- open a window first, then a recording in it.

:func:`~pygor.gui.launch.launch` wants a recording already built in Python,
which is the step a lab member without a notebook open cannot take. This
starts from an empty viewer with a File menu, the way IGOR's OS_GUI starts
from its Load buttons, and swaps one recording for the next in the same
window rather than opening a viewer per file.

    pygor-gui
    pygor-gui recording.h5 --as OSDS --config my_settings.toml
"""

import argparse
import pathlib

from qtpy.QtCore import QSettings
from qtpy.QtWidgets import QMessageBox

from pygor.gui.launch import _import_napari, attach, detach, recording_title

# napari keeps its own File menu, so pygor's is named for what it opens
RECORDING_MENU_TITLE = "&Recording"
_SETTINGS = ("pygor", "gui")
_RECENT_KEY = "recent_files"
_MAX_RECENT = 10


class PygorApp:
    """One viewer, and whichever recording is attached to it."""

    def __init__(self, viewer):
        self.viewer = viewer
        self.session = None
        self._worker = None
        # The saved recording that was opened, offered again on save
        self._opened_path = None
        self._settings = QSettings(*_SETTINGS)
        self._build_recording_menu()

    @property
    def recording(self):
        return None if self.session is None else self.session.recording

    @property
    def _window(self):
        return self.viewer.window._qt_window

    def _build_recording_menu(self):
        main_menu = self.viewer.window.main_menu
        menu = main_menu.addMenu(RECORDING_MENU_TITLE)
        # First on the bar, since nothing else does anything until it is used
        main_menu.insertMenu(main_menu.actions()[0], menu)
        # Ctrl+O is napari's Open Files; a shared shortcut runs neither
        menu.addAction("Open recording...", self.open_dialog).setShortcut("Ctrl+R")
        self.recent_menu = menu.addMenu("Open recent")
        self.recent_menu.aboutToShow.connect(self._fill_recent)
        menu.addSeparator()
        self.save_action = menu.addAction("Save recording as...", self.save)
        # Not Ctrl+S: napari's "Save Selected Layers" has it, and Qt runs
        # neither action when two share a shortcut
        self.save_action.setShortcut("Ctrl+Shift+S")
        self.close_action = menu.addAction("Close recording", self.close_recording)
        self.menu = menu
        self._update_actions()

    def _update_actions(self):
        has = self.session is not None
        self.save_action.setEnabled(has)
        self.close_action.setEnabled(has)

    def open_dialog(self, path=None):
        """Ask for a file, class and config, then open it."""
        from pygor.gui.widgets.open_dialog import OpenDialog

        last = self._settings.value("last_class", "Core")
        dialog = OpenDialog(
            self._window,
            path=path,
            config=self._settings.value("last_config", ""),
            class_name=last,
        )
        if not dialog.exec_():
            return None
        self._settings.setValue("last_class", dialog.class_box.currentText())
        self._settings.setValue("last_config", dialog.config or "")
        return self.open(
            dialog.path, cls=dialog.cls, config=dialog.config,
            n_colours=dialog.n_colours,
        )

    def open(self, path, cls=None, config=None, n_colours=None, background=True):
        """Load a recording and attach it in place of the current one.

        Loading a large ScanM file takes seconds, so it runs off the GUI
        thread; the current recording stays attached until the new one is
        ready, and is only let go of if the user agrees to.
        """
        from pygor.tui.reader import SAVED_SUFFIX, load_recording

        if not self._confirm_close():
            return None
        path = pathlib.Path(path)

        def load():
            return load_recording(path, n_colours=n_colours, cls=cls, config=config)

        def loaded(recording):
            self._worker = None
            # A saved object's filename is the raw file it came from, which
            # is not the file that was just opened
            self.attach_recording(recording, title=path.name)
            if path.name.endswith(SAVED_SUFFIX):
                self._opened_path = path
            self._remember(path)

        def failed(exc):
            self._worker = None
            self.viewer.status = "Ready"
            QMessageBox.critical(
                self._window, "Open failed", f"Could not open {path.name}:\n{exc}"
            )

        self.viewer.status = f"Opening {path.name}..."
        if not background:
            try:
                recording = load()
            except Exception as exc:
                failed(exc)
                return None
            loaded(recording)
            return recording

        from napari.qt.threading import thread_worker

        worker = thread_worker(load)()
        worker.returned.connect(loaded)
        worker.errored.connect(failed)
        self._worker = worker
        worker.start()
        return worker

    def attach_recording(self, recording, title=None):
        """Put an already loaded recording in the window."""
        if self.session is not None:
            detach(self.viewer, self.session)
        self.session = attach(self.viewer, recording)
        self._opened_path = None
        self.viewer.title = title or recording_title(recording)
        self.viewer.status = f"Opened {self.viewer.title}"
        self._update_actions()

    def save(self):
        from pygor.gui.menus import save_recording_as

        if self.session is None:
            return False
        return save_recording_as(self.viewer, self.recording, default=self._opened_path)

    def close_recording(self, confirm=True):
        """Take the current recording out of the window."""
        if self.session is None:
            return True
        if confirm and not self._confirm_close():
            return False
        detach(self.viewer, self.session)
        self.session = None
        self.viewer.title = "pygor"
        self._update_actions()
        return True

    def _confirm_close(self):
        """Offer to save the current recording. False means cancel.

        Nothing tracks whether a recording has changed since it was
        opened, so the question is asked whenever there is one.
        """
        if self.session is None:
            return True
        # Unpushed ROI edits should be in what gets saved
        self.session.actions.sync_rois_from_layer()
        answer = QMessageBox.question(
            self._window,
            "Save recording?",
            f"Save {self.viewer.title} before closing it?",
            QMessageBox.Save | QMessageBox.Discard | QMessageBox.Cancel,
            QMessageBox.Save,
        )
        if answer == QMessageBox.Cancel:
            return False
        if answer == QMessageBox.Save:
            return self.save()
        return True

    def recent_files(self):
        recent = self._settings.value(_RECENT_KEY, []) or []
        # QSettings hands back a bare string when only one was stored
        return [recent] if isinstance(recent, str) else list(recent)

    def _remember(self, path):
        path = str(pathlib.Path(path).resolve())
        recent = [p for p in self.recent_files() if p != path]
        self._settings.setValue(_RECENT_KEY, [path, *recent][:_MAX_RECENT])

    def _fill_recent(self):
        self.recent_menu.clear()
        recent = [p for p in self.recent_files() if pathlib.Path(p).exists()]
        if not recent:
            self.recent_menu.addAction("(none)").setEnabled(False)
            return
        for path in recent:
            self.recent_menu.addAction(
                path, lambda p=path: self.open_dialog(path=p)
            )


def open_app(path=None, cls=None, config=None, n_colours=None, show=True,
             block=False):
    """Open an empty pygor window, and optionally a recording in it.

    With a path, the recording opens straight away as ``cls`` (default
    Core) rather than through the dialog. Returns the :class:`PygorApp`.
    """
    napari = _import_napari()
    viewer = napari.Viewer(title="pygor", show=show)
    app = PygorApp(viewer)
    # The window owns the app; without a reference the menu callbacks
    # would outlive the object they belong to
    viewer.window._pygor_app = app
    if path is not None:
        if cls is None:
            from pygor.classes.core_data import Core

            cls = Core
        app.open(path, cls=cls, config=config, n_colours=n_colours)
    if block:
        napari.run()
    return app


def main(argv=None):
    from pygor.tui.reader import analysis_classes

    classes = analysis_classes()
    parser = argparse.ArgumentParser(
        prog="pygor-gui", description="Open pygor recordings in napari."
    )
    parser.add_argument("path", nargs="?", help="recording to open")
    parser.add_argument(
        "--as", dest="cls", choices=list(classes), default="Core",
        help="class to open a raw recording as (default: Core)",
    )
    parser.add_argument("--config", help="TOML config to merge with defaults")
    parser.add_argument("--n-colours", type=int, help="interleaved colours (STRF)")
    args = parser.parse_args(argv)
    open_app(
        args.path, cls=classes[args.cls], config=args.config,
        n_colours=args.n_colours, block=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
