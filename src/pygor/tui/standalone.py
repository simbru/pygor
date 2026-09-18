"""``pygor-tui`` -- open one recording, edit its parameters, re-segment, save.

The skeleton of a pygor terminal environment that does not need a dataset
binding: the same reprocess screen the proofreading cockpit uses, bound to a
single recording's own methods. Re-segmentation happens in memory; nothing is
written until save is pressed and confirmed, and the previous file is kept as
``.presegment``.

    pygor-tui recording.recording.h5
    pygor-tui recording.smp --n-colours 4

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

# Which of a recording's parameter sections the table offers.
SECTIONS = ("segmentation", "preprocessing.artifact_width")


def load_recording(path, n_colours=None):
    import pygor.load

    path = pathlib.Path(path)
    if path.name.endswith(".recording.h5"):
        return pygor.load.Core.load_object(path)
    kwargs = {"n_colours": n_colours} if n_colours else {}
    return pygor.load.STRF(str(path), **kwargs)


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


def main(argv=None) -> int:
    os.environ["MPLBACKEND"] = "Agg"
    parser = argparse.ArgumentParser(prog="pygor-tui", description=__doc__.splitlines()[0])
    parser.add_argument("recording", help=".recording.h5 or a ScanM .smp/.smh")
    parser.add_argument("--n-colours", type=int)
    parser.add_argument("--graphics", default="auto",
                        choices=("auto", "tgp", "sixel", "halfcell", "unicode", "none"))
    args = parser.parse_args(argv)

    from pygor.tui.capabilities import probe

    caps = probe(args.graphics)

    try:
        from textual.app import App

        from pygor.tui.imaging import PREVIEWS, recording_preview
        from pygor.tui.reprocess_screen import ReprocessScreen
    except ImportError as error:
        raise SystemExit(
            "pygor-tui needs the [tui] extra:  uv pip install 'pygor[tui]'\n" f"({error})"
        ) from error

    recording = load_recording(args.recording, args.n_colours)
    source = pathlib.Path(args.recording)
    values = effective_values(recording)

    def run(overrides):
        mode, kwargs = segmentation_kwargs(recording, overrides)
        recording.segment_rois(mode=mode, plot=False, **kwargs)
        return recording

    def preview(result, which, width, height):
        return recording_preview(result or recording, which, width, height)

    def save(result, overrides):
        if source.name.endswith(".recording.h5"):
            target = source
        else:
            target = source.with_name(source.stem + ".recording.h5")
        if target.exists():
            shutil.copy2(target, target.with_suffix(target.suffix + ".presegment"))
        result.save_object(target.with_name(target.name.removesuffix(".recording.h5")),
                           overwrite=True)

    class StandaloneApp(App):
        CSS = """
        Screen { background: $surface; }
        #params { width: 2fr; height: 1fr; }
        #reprocess-preview { width: 3fr; padding: 0 1; height: 1fr; overflow: hidden; align: center middle; }
        #status { height: 1; padding: 0 1; background: $panel; }
        #value-box, #confirm-box { padding: 1 2; width: 70%; height: auto; background: $panel; }
        .error { color: $error; }
        .dim { color: $text-muted; }
        """
        BINDINGS = [("q", "quit", "quit")]

        def on_mount(self):
            self.push_screen(
                ReprocessScreen(
                    title=f"{recording.name}  ({recording.num_rois} ROIs on disk)",
                    values=values, sections=SECTIONS, run=run, preview=preview,
                    save=save, caps=caps, previews=PREVIEWS,
                    save_text=f"Overwrite {source.name}? The old file is kept as .presegment.",
                ),
                lambda outcome: self.exit(),
            )

    StandaloneApp().run()
    return 0


if __name__ == "__main__":
    sys.exit(main())
