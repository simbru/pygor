"""napari-based GUI for pygor recordings.

Requires the ``[gui]`` extra. Imports are deferred so that importing
pygor without napari installed stays cheap and error-free.
"""

__all__ = ["launch"]


def launch(*args, **kwargs):
    """Open a napari viewer for a recording. See :func:`pygor.gui.launch.launch`."""
    from pygor.gui.launch import launch as _launch

    return _launch(*args, **kwargs)
