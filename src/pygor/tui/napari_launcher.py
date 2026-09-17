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


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--path", required=True, help="recording to open")
    parser.add_argument("--master", help="the field of view's master, for comparison")
    parser.add_argument("--roi", type=int)
    args = parser.parse_args(argv)

    import numpy as np

    import pygor.load
    from pygor.core.gui.methods import _import_napari

    napari, _, _ = _import_napari()

    recording = pygor.load.Core.load_object(args.path)
    viewer = napari.Viewer(title=f"pygor review — {recording.name}")
    viewer.add_image(recording.images, name=f"{recording.name} stack", colormap="gray")
    viewer.add_image(
        np.mean(recording.images, axis=0), name="mean projection", colormap="gray",
    )
    # Labels rather than an image: napari gives per-ROI identity and picking for
    # free, which is the whole reason to leave the terminal for this.
    viewer.add_labels(
        np.where(recording.rois < 0, -recording.rois, 0).astype(int), name="ROIs",
    )

    if args.master and args.master != args.path:
        master = pygor.load.Core.load_object(args.master)
        viewer.add_image(
            np.mean(master.images, axis=0), name=f"{master.name} mean",
            colormap="magenta", blending="additive", opacity=0.6,
        )
        viewer.add_labels(
            np.where(master.rois < 0, -master.rois, 0).astype(int),
            name=f"{master.name} ROIs", visible=False,
        )

    napari.run()
    return 0


if __name__ == "__main__":
    sys.exit(main())
