"""``pygor-tui`` -- open recordings, look at them, re-segment one if it needs it.

A pygor terminal environment that does not need a dataset binding. Given a
recording it opens it; given a directory, or nothing at all, it opens a browser
rooted there and lets you pick, which is the blank slate: the tool starts
without having decided which file you meant.

    pygor-tui                          # browse the working directory
    pygor-tui ~/data/2025-06-12        # browse there
    pygor-tui recording.recording.h5   # straight to that one
    pygor-tui recording.smp --n-colours 4

Either way you land on the inspector, which is read-only: what is in the file,
a picture of it, and the signal under the cursor. ``p`` cycles the views --
ROIs, correlation, numbered, and the raw stack, which ``,`` and ``.`` scrub
through in time. A click, or the arrow keys, moves a probe: over a
segmented ROI the pane plots that ROI's extracted trace, anywhere else the raw
neighbourhood, and ``t`` swaps the plot between braille and a rendered figure.
``+`` and ``-``, or the scroll wheel over the trace, zoom it in time (the wheel
about the point under the pointer), ``[`` and ``]`` pan, ``0`` shows the whole
thing again; the pointer over the trace reads out time and value in the status
line. The window is kept when the probe moves, so the same stretch can be
compared across cells. Triggers are drawn on the same x axis as the trace,
individually where they can be told apart and as a density bar where they
cannot, with the count, period and any gaps stated underneath.

``r`` opens the same reprocess screen the proofreading cockpit uses, bound to
this recording's own ``segment_rois``. ``d`` opens a screen for drawing ROIs by
hand -- a disk or a lasso, added where no existing ROI already is, deleted with
``x``. Both work in memory; nothing is written until save is pressed and
confirmed, and the previous file is kept as ``.presegment``.

Steps beyond segmentation (trace extraction, STA) are deliberately not run
here yet: which of them apply, and with what trigger mode, is a property of the
stimulus, which a dataset binding knows and a lone recording does not.
"""

from __future__ import annotations

import argparse
import os
import pathlib
import shutil
import sys

from pygor.tui.reader import load_recording  # noqa: F401 - re-exported

# Which of a recording's parameter sections the table offers.
SECTIONS = ("segmentation", "preprocessing.artifact_width")


def segmentation_kwargs(recording, overrides):
    """Turn recipe-shaped overrides into ``segment_rois`` arguments.

    The recipe nests blob parameters under ``segmentation.blob`` and the mode
    under ``segmentation.general``; ``segment_rois`` takes them flat.
    """
    seg = overrides.get("segmentation", {}) if overrides else {}
    general = seg.get("general", {})
    mode = general.get("mode") or recording.params.get_defaults("segmentation").get(
        "general", {}).get("mode", "blob")
    kwargs = dict(seg.get(mode, {}))
    if "roi_order" in general:
        kwargs["roi_order"] = general["roi_order"]
    artifact = (overrides or {}).get("preprocessing", {}).get("artifact_width")
    if artifact is not None:
        kwargs["artifact_width"] = artifact
    return mode, kwargs


def effective_values(recording) -> dict:
    """The recording's saved parameters over today's package defaults.

    A saved object carries the defaults of the pygor that processed it. A
    parameter added since is in effect at its default when the object is
    re-segmented now, so it belongs in the table; the saved values win
    wherever both exist.
    """
    from pygor.params import AnalysisParams

    current = AnalysisParams.from_config(None).to_dict().get("_defaults", {})
    saved = recording.params.to_dict().get("_defaults", {})

    def merge(base, over):
        out = dict(base)
        for key, value in over.items():
            if isinstance(value, dict) and isinstance(out.get(key), dict):
                out[key] = merge(out[key], value)
            else:
                out[key] = value
        return out

    return merge(current, saved)


def saved_path(source) -> pathlib.Path:
    """Where a recording opened from ``source`` is saved: itself if it is a
    saved object already, a ``.recording.h5`` beside it if it is raw."""
    source = pathlib.Path(source)
    if source.name.endswith(".recording.h5"):
        return source
    return source.with_name(source.stem + ".recording.h5")


def save_recording(recording, source) -> None:
    """Write ``recording`` to :func:`saved_path`, keeping what was there as
    ``.presegment``."""
    target = saved_path(source)
    if target.exists():
        shutil.copy2(target, target.with_suffix(target.suffix + ".presegment"))
    recording.save_object(target.with_name(target.name.removesuffix(".recording.h5")),
                          overwrite=True)


def _present(value) -> bool:
    import numpy as np

    return isinstance(value, np.ndarray) and value.size > 0


def apply_mask(recording, mask) -> None:
    """Put a hand-edited mask on ``recording`` and bring its traces along.

    ``segment_rois`` swaps the mask and leaves traces for the reader to
    re-extract; a hand edit cannot, because the inspector reads per-ROI rows
    by position and a deleted ROI would shift every later cell onto its
    neighbour's trace. So traces are re-extracted here, and snippets and
    averages recomputed if the recording had them. STRFs are not: they need
    the noise stimulus and minutes of compute, and the save confirmation says
    they are stale.
    """
    import numpy as np

    from pygor.tui import roi_edit

    previous = getattr(recording, "roi_origin", None)
    dtype = recording.rois.dtype if getattr(recording, "rois", None) is not None else np.int16
    mask = roi_edit.compact(mask).astype(dtype)
    had_averages = _present(getattr(recording, "averages", None))
    recording.update_rois(mask)
    recording.num_rois = roi_edit.count(mask)
    recording.roi_origin = {
        "method": "manual", "source": recording.name,
        "edited_from": previous.get("method") if isinstance(previous, dict) else None,
    }
    if recording.num_rois == 0:
        for attribute in ("traces_raw", "traces_znorm", "snippets", "averages"):
            setattr(recording, attribute, np.nan)
        return
    recording.extract_traces_from_rois()
    if had_averages:
        recording.compute_snippets_and_averages()


def make_draw_screen(recording, source, caps):
    """The ROI drawing screen for one loaded recording, saving where reprocess does."""
    from pygor.tui import roi_edit
    from pygor.tui.draw_screen import DrawScreen

    target = saved_path(source)

    def save(mask):
        apply_mask(recording, mask)
        save_recording(recording, source)

    def describe_save(mask):
        text = (f"Save {roi_edit.count(mask)} ROIs to {target.name}? Ids are renumbered "
                f"without gaps and traces re-extracted. The old file is kept as "
                f".presegment.")
        strfs = getattr(recording, "strfs", None)
        if _present(strfs):
            text += (f"\n\n[b]The {len(strfs)} STRFs on this recording were computed "
                     f"from the old ROIs and are not recomputed here.[/b]")
        return text

    return DrawScreen(recording, caps=caps, save=save, describe_save=describe_save)


def make_reprocess_screen(recording, source, caps):
    """The reprocess screen for one already-loaded recording.

    A factory rather than inline setup because the inspector reaches it too,
    with a recording it loaded itself, and the two must agree on where a save
    goes.
    """
    from pygor.tui.imaging import PREVIEWS, recording_preview
    from pygor.tui.reprocess_screen import ReprocessScreen, segmentation_gating

    source = pathlib.Path(source)
    values = effective_values(recording)
    choices, gates = segmentation_gating(values)

    def run(overrides):
        mode, kwargs = segmentation_kwargs(recording, overrides)
        recording.segment_rois(mode=mode, plot=False, **kwargs)
        return recording

    def preview(result, which, width, height):
        return recording_preview(result or recording, which, width, height)

    def save(result, overrides):
        save_recording(result, source)

    return ReprocessScreen(
        title=f"{recording.name}  ({recording.num_rois} ROIs on disk)",
        values=values, sections=SECTIONS, run=run, preview=preview,
        save=save, caps=caps, previews=PREVIEWS, choices=choices, gates=gates,
        save_text=f"Overwrite {source.name}? The old file is kept as .presegment.",
    )


APP_CSS = """
Screen { background: $surface; }
#params { width: 2fr; height: 1fr; }
#reprocess-preview { width: 3fr; padding: 0 1; height: 1fr; overflow: hidden; align: center middle; }
#tree { width: 2fr; height: 1fr; }
#browse-side { width: 1fr; padding: 1 2; height: 1fr; background: $panel; }
#meta { width: 32; padding: 1 2; height: 1fr; overflow-y: auto; }
#inspect-right { width: 1fr; height: 1fr; }
#inspect-preview { width: 1fr; padding: 0 1; height: 2fr; overflow: hidden; align: center middle; }
#trace { width: 1fr; height: 12; padding: 0 1; background: $panel; overflow: hidden; }
#trace .trace-text { width: 1fr; height: 1fr; }
#status { height: 1; padding: 0 1; background: $panel; }
#draw-help { width: 32; padding: 1 2; height: 1fr; overflow-y: auto; }
#draw-preview { width: 1fr; padding: 0 1; height: 1fr; overflow: hidden; align: center middle; }
#value-box, #confirm-box { padding: 1 2; width: 70%; height: auto; background: $panel; }
.error { color: $error; }
.dim { color: $text-muted; }
"""


def build_app(target, caps, n_colours=None):
    """The app, landing on the browser or on one recording's inspector."""
    from textual.app import App

    from pygor.tui.browse_screen import BrowseScreen, InspectScreen
    from pygor.tui.imaging import PREVIEWS
    from pygor.tui.napari_launcher import spawn

    target = pathlib.Path(target).expanduser()

    def reprocess(recording, path):
        return make_reprocess_screen(recording, path, caps)

    def draw(recording, path):
        return make_draw_screen(recording, path, caps)

    class StandaloneApp(App):
        CSS = APP_CSS
        BINDINGS = [("q", "quit", "quit")]

        def __init__(self):
            super().__init__()
            # Held so the processes are not garbage-collected mid-launch, and
            # so a later version can tell you which viewers are still open.
            self._napari_procs = []

        def on_mount(self):
            if target.is_dir():
                self.push_screen(BrowseScreen(
                    target, caps=caps, previews=PREVIEWS, n_colours=n_colours,
                    reprocess=reprocess, draw=draw,
                ))
            else:
                # Escape from the only screen there is means leaving, not
                # falling through to a browser the user never asked for.
                self.push_screen(
                    InspectScreen(target, caps=caps, previews=PREVIEWS,
                                  n_colours=n_colours, reprocess=reprocess, draw=draw),
                    lambda outcome: self.exit(),
                )

        def spawn_napari(self, args, label):
            process, message = spawn(args, label=label)
            if process is None:
                self.notify(message, severity="error", timeout=10)
                return
            self._napari_procs.append(process)
            self.notify(message, timeout=6)

    return StandaloneApp()


def main(argv=None) -> int:
    os.environ["MPLBACKEND"] = "Agg"
    parser = argparse.ArgumentParser(prog="pygor-tui", description=__doc__.splitlines()[0])
    parser.add_argument("target", nargs="?", default=".",
                        help="a .recording.h5, an IGOR .h5, a ScanM .smp, or a "
                             "directory to browse (default: the working directory)")
    parser.add_argument("--n-colours", type=int)
    parser.add_argument("--graphics", default="auto",
                        choices=("auto", "tgp", "sixel", "halfcell", "unicode", "none"))
    args = parser.parse_args(argv)

    target = pathlib.Path(args.target).expanduser()
    if not target.exists():
        raise SystemExit(f"no such file or directory: {target}")

    from pygor.tui.capabilities import probe

    # Must happen before Textual starts: its input thread eats the reply.
    caps = probe(args.graphics)

    try:
        app = build_app(target, caps, args.n_colours)
    except ImportError as error:
        raise SystemExit(
            "pygor-tui needs the [tui] extra:  uv pip install 'pygor[tui]'\n" f"({error})"
        ) from error

    # Restore the terminal on every exit path, not only the clean one, and
    # again at interpreter exit in case something after us leaves it dirty.
    import atexit

    from pygor.tui.imaging import clear_terminal_images, restore_terminal

    atexit.register(restore_terminal)
    try:
        app.run()
    finally:
        clear_terminal_images()
        restore_terminal()
    return 0


if __name__ == "__main__":
    sys.exit(main())
