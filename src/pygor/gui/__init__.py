"""napari-based GUI for pygor recordings.

Requires the ``[gui]`` extra. Imports are deferred so that importing
pygor without napari installed stays cheap and error-free.
"""

__all__ = ["launch", "open_app"]


def launch(*args, **kwargs):
    """Open a napari viewer for a recording. See :func:`pygor.gui.launch.launch`."""
    from pygor.gui.launch import launch as _launch

    return _launch(*args, **kwargs)


def open_app(*args, **kwargs):
    """Open an empty pygor window. See :func:`pygor.gui.app.open_app`."""
    from pygor.gui.app import open_app as _open_app

    return _open_app(*args, **kwargs)
