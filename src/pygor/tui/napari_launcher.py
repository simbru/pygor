"""Open a recording in napari, in a process of its own.

Never imported by the cockpit. Qt and Textual cannot share a process: a Qt event
loop either blocks Textual's asyncio loop or fights it, and Qt writes warnings
straight to the terminal, which corrupts a full-screen interface. The repo
already records the related hazard for the in-process case -- ``napari.run()``
outside ``%gui qt`` hangs after the window closes -- in ``core/gui/methods.py``
and ``dev/ipl_method_shootout.py``.

So the cockpit spawns this with stdin closed and output redirected, and forgets
about it.
"""

from __future__ import annotations

import argparse
import sys


def spawn(args, *, label):
    """Start this launcher detached, with its output in a log file.

    Returns ``(process, message)``, or ``(None, why)`` when napari cannot run
    here -- checked before spawning, so the caller has a sentence to show
    instead of a window that never appears.
    """
    import pathlib
    import subprocess
    import tempfile

    from pygor.tui.capabilities import napari_availability

    ok, why = napari_availability()
    if not ok:
        return None, why
    log = pathlib.Path(tempfile.gettempdir()) / f"pygor-napari-{label}.log"
    handle = log.open("w")
    process = subprocess.Popen(
        [sys.executable, "-m", "pygor.tui.napari_launcher", *args],
        stdin=subprocess.DEVNULL, stdout=handle, stderr=handle,
        start_new_session=True,
    )
    return process, f"napari starting (log: {log})"


def _add_rois(viewer, recording, name, **kwargs):
    """Add a recording's ROIs as a labels layer, if it has any.

    Labels rather than an image: napari gives per-ROI identity and picking for
    free, which is the whole reason to leave the terminal for this. An
    unsegmented recording simply contributes no layer -- it is a normal thing
    to open one, and it was a crash before.
    """
    import numpy as np

    if getattr(recording, "rois", None) is None:
        return
    viewer.add_labels(
        np.where(recording.rois < 0, -recording.rois, 0).astype(int), name=name, **kwargs
    )


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--path", required=True, help="recording to open")
    parser.add_argument("--master", help="the field of view's master, for comparison")
    parser.add_argument("--roi", type=int)
    parser.add_argument("--n-colours", type=int,
                        help="for a raw ScanM file whose name does not say")
    args = parser.parse_args(argv)

    import numpy as np

    from pygor.core.gui.methods import _import_napari
    from pygor.tui.reader import load_recording

    napari, _, _ = _import_napari()

    recording = load_recording(args.path, args.n_colours)
    viewer = napari.Viewer(title=f"pygor review — {recording.name}")
    viewer.add_image(recording.images, name=f"{recording.name} stack", colormap="gray")
    viewer.add_image(
        np.mean(recording.images, axis=0), name="mean projection", colormap="gray",
    )
    _add_rois(viewer, recording, "ROIs")

    if args.master and args.master != args.path:
        master = load_recording(args.master)
        viewer.add_image(
            np.mean(master.images, axis=0), name=f"{master.name} mean",
            colormap="magenta", blending="additive", opacity=0.6,
        )
        _add_rois(viewer, master, f"{master.name} ROIs", visible=False)

    napari.run()
    return 0


if __name__ == "__main__":
    sys.exit(main())
