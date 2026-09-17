"""What this terminal can draw, decided once before the interface starts.

The probe has to run before Textual does. Textual starts a thread that reads
stdin, and it will swallow the terminal's reply to a capability query, so asking
afterwards gets nothing back.

The renderer is chosen explicitly rather than taken from ``textual_image``'s own
auto-detection, which prefers Sixel over the kitty protocol. Sixel is the wrong
default here: the library's own documentation says images drawn that way flicker
when scrolled or restyled, because they are injected into the output rather than
composited as cells. The kitty protocol's Unicode placeholders occupy real cells,
so Textual clips and scrolls them like any other content.
"""

from __future__ import annotations

import dataclasses
import os
import sys

MODES = ("auto", "tgp", "sixel", "halfcell", "unicode", "none")

# Fallback when the terminal will not say. VT340 dimensions, as textual-image uses.
DEFAULT_CELL = (10, 20)


@dataclasses.dataclass(frozen=True)
class Capabilities:
    mode: str
    cell_width: int
    cell_height: int
    is_tty: bool
    tgp: bool = False
    sixel: bool = False
    tmux: bool = False
    term: str = ""
    note: str = ""

    @property
    def graphical(self) -> bool:
        return self.mode in ("tgp", "sixel", "halfcell")

    def describe(self) -> str:
        bits = [f"mode={self.mode}", f"cell={self.cell_width}x{self.cell_height}px",
                f"TERM={self.term or '?'}"]
        if self.tmux:
            bits.append("tmux")
        if self.note:
            bits.append(self.note)
        return "  ".join(bits)


def probe(requested="auto") -> Capabilities:
    """Ask the terminal what it supports. Must run before the app starts."""
    if requested not in MODES:
        raise ValueError(f"graphics mode {requested!r} not in {MODES}")

    term = os.environ.get("TERM", "")
    in_tmux = bool(os.environ.get("TMUX"))
    is_tty = bool(sys.__stdout__ and sys.__stdout__.isatty())

    if requested == "none" or not is_tty:
        return Capabilities(
            mode="none", cell_width=DEFAULT_CELL[0], cell_height=DEFAULT_CELL[1],
            is_tty=is_tty, tmux=in_tmux, term=term,
            note="" if requested == "none" else "not a terminal",
        )

    tgp = sixel = False
    note = ""
    try:
        from textual_image.renderable import sixel as sixel_mod
        from textual_image.renderable import tgp as tgp_mod

        tgp = bool(tgp_mod.query_terminal_support())
        sixel = bool(sixel_mod.query_terminal_support())
    except Exception as error:  # a terminal that mangles the probe must not be fatal
        note = f"probe failed: {error}"

    cell_width, cell_height = _cell_size()

    if requested != "auto":
        mode = requested
    elif tgp:
        mode = "tgp"
    elif sixel:
        mode = "sixel"
    else:
        mode = "halfcell"

    if in_tmux and not (tgp or sixel) and not note:
        note = "in tmux with no graphics; needs 'set -g allow-passthrough on'"

    return Capabilities(mode=mode, cell_width=cell_width, cell_height=cell_height,
                        is_tty=is_tty, tgp=tgp, sixel=sixel, tmux=in_tmux,
                        term=term, note=note)


def _cell_size():
    try:
        from textual_image._terminal import get_cell_size

        size = get_cell_size()
        if size.width > 0 and size.height > 0:
            return int(size.width), int(size.height)
    except Exception:
        pass
    return DEFAULT_CELL


def image_widget_class(mode):
    """The textual-image widget class for a mode, or None when not drawing."""
    if mode == "none":
        return None
    from textual_image.widget import HalfcellImage, SixelImage, TGPImage, UnicodeImage

    return {
        "tgp": TGPImage,
        "sixel": SixelImage,
        "halfcell": HalfcellImage,
        "unicode": UnicodeImage,
    }.get(mode)


def napari_availability() -> tuple[bool, str]:
    """Whether the napari hand-off can work here, and why not when it cannot.

    Checked before spawning rather than after, so the answer is a sentence the
    reviewer can act on instead of a viewer that never appears.
    """
    import importlib.util

    if importlib.util.find_spec("napari") is None:
        return False, "napari is not installed - uv pip install 'pygor[gui]'"
    if os.environ.get("QT_QPA_PLATFORM") in ("offscreen", "minimal"):
        return False, "QT_QPA_PLATFORM is forced headless"
    if not (os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY")):
        return False, "no display on this session - run the cockpit at the workstation"
    return True, ""


def _main(argv=None) -> int:
    """``python -m pygor.tui.capabilities`` - report and draw a test image."""
    import argparse

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--graphics", default="auto", choices=MODES)
    args = parser.parse_args(argv)

    caps = probe(args.graphics)
    print(caps.describe())
    print(f"  tty={caps.is_tty}  tgp={caps.tgp}  sixel={caps.sixel}")
    ok, why = napari_availability()
    print(f"  napari hand-off: {'available' if ok else why}")

    if not caps.graphical:
        print("\nno graphics; the cockpit would run in text mode")
        return 0

    import io

    import numpy as np
    from matplotlib.figure import Figure

    from pygor.review.rasterise import figure_to_png

    figure = Figure(figsize=(4, 2), dpi=100)
    axis = figure.add_subplot(111)
    axis.imshow(np.add.outer(np.linspace(0, 1, 40), np.linspace(0, 1, 80)), cmap="magma")
    axis.set_title("pygor tui test pattern")
    png = figure_to_png(figure, width=400, height=200)

    from textual_image.renderable import HalfcellImage, SixelImage, TGPImage

    renderable = {"tgp": TGPImage, "sixel": SixelImage, "halfcell": HalfcellImage}[caps.mode]
    from rich.console import Console

    Console().print(renderable(io.BytesIO(png)))
    print(f"\nrendered {len(png) / 1024:.1f} KiB via {caps.mode}")
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
