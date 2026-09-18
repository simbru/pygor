"""The one place that knows about pixels, cells and the image library.

Kept to a single seam deliberately. ``textual-image`` is a small project with one
maintainer, and its kitty renderable is about two hundred lines; if it stops
being maintained, vendoring it means rewriting this module and nothing else.

Panels are asked for a size in pixels, but a terminal lays out in character
cells, so the conversion happens here and only here.
"""

from __future__ import annotations

import io

from pygor.tui.capabilities import image_widget_class

# Leave the panel a little smaller than its pane so a rounding error cannot
# push it into scrolling the container.
CELL_MARGIN = 1

# Ceiling on a rendered panel. A terminal cannot show more than this usefully,
# and every pixel beyond it is render time and, over SSH, bytes on the wire.
MAX_PANEL_PX = (1400, 900)


def pixels_for_cells(cols, rows, caps, margin=CELL_MARGIN):
    """Pixel size for a pane that many character cells across and down.

    Capped: on a full-screen pane the raw figure would be a couple of thousand
    pixels across, which costs render time and, over SSH, wire time, for detail
    no one can see in a terminal cell grid.
    """
    cols = max(int(cols) - margin, 1)
    rows = max(int(rows) - margin, 1)
    width = min(cols * caps.cell_width, MAX_PANEL_PX[0])
    height = min(rows * caps.cell_height, MAX_PANEL_PX[1])
    return width, height


def fit_cells(image_width, image_height, cols, rows, caps):
    """Cell size that fits the image in the pane without changing its shape.

    Letting the widget fill both dimensions is what stretches a panel: a wide
    receptive-field mosaic forced into a tall pane comes out smeared vertically.
    Scaling by the smaller of the two ratios keeps the pixels square.
    """
    cols = max(int(cols), 1)
    rows = max(int(rows), 1)
    if image_width <= 0 or image_height <= 0:
        return cols, rows
    available_w = cols * caps.cell_width
    available_h = rows * caps.cell_height
    scale = min(available_w / image_width, available_h / image_height)
    out_w = max(int(image_width * scale), 1)
    out_h = max(int(image_height * scale), 1)
    return (
        max(min(out_w // caps.cell_width, cols), 1),
        max(min(out_h // caps.cell_height, rows), 1),
    )


def make_widget(panel_image, caps, **kwargs):
    """A Textual widget showing a rendered panel, or None in text mode.

    The widget is given the PNG bytes directly rather than a PIL image: the
    panel was already rasterised at the requested size, and handing over decoded
    pixels would invite a resample on the way in.
    """
    widget_class = image_widget_class(caps.mode)
    if widget_class is None:
        return None
    widget = widget_class(io.BytesIO(panel_image.png), **kwargs)
    # No focus or hover styling anywhere on an image widget: a repaint
    # re-transmits the whole PNG to the terminal, so a focus ring costs tens of
    # kilobytes every time the cursor moves.
    widget.can_focus = False
    return widget


def describe_panel(panel_image) -> str:
    """One line about a panel, for the text-mode fallback."""
    meta = panel_image.meta or {}
    return (
        f"[{meta.get('panel', '?')}] {panel_image.width}x{panel_image.height}px "
        f"{panel_image.nbytes / 1024:.0f} KiB — no graphics in this terminal"
    )


def show(obj, caps, *, width=900, dpi=100, cmap="Greys_r") -> None:
    """Print a figure, an image array or a PNG inline, in the terminal.

    This is what makes pygor's matplotlib API usable over SSH. Every
    ``view_*`` and ``plot_*`` method returns a figure that would normally need
    a window; here it is rasterised and drawn as terminal cells instead, so
    the interactive loop -- segment, look, adjust, look again -- works from
    the REPL without a display.

    Accepts a Figure, a (fig, ax) tuple as pygor returns, a 2-D array, or PNG
    bytes.
    """
    import io

    import numpy as np
    from matplotlib.figure import Figure

    from pygor.review.rasterise import figure_to_png

    if isinstance(obj, tuple) and obj and isinstance(obj[0], Figure):
        obj = obj[0]
    if isinstance(obj, Figure):
        height = int(width * obj.get_figheight() / obj.get_figwidth())
        png = figure_to_png(obj, width=width, height=height, dpi=dpi)
    elif isinstance(obj, (bytes, bytearray)):
        png = bytes(obj)
    else:
        array = np.asarray(obj)
        if array.ndim not in (2, 3):
            raise TypeError(f"cannot show an array of shape {array.shape}")
        figure = Figure(figsize=(width / dpi, width * array.shape[0] / array.shape[1] / dpi),
                        dpi=dpi)
        axis = figure.add_axes([0, 0, 1, 1])
        axis.imshow(array, cmap=cmap if array.ndim == 2 else None, origin="lower")
        axis.set_axis_off()
        png = figure_to_png(figure, dpi=dpi)

    renderable_class = _renderable_class(caps.mode)
    if renderable_class is None:
        import sys

        sys.stdout.write(f"[image {len(png) // 1024} KiB; no terminal graphics]\n")
        return
    from rich.console import Console

    Console().print(renderable_class(io.BytesIO(png)))


def show_rois(recording, caps, *, width=1000, labels=False, low=1, high=99,
              image=None) -> None:
    """Draw a recording's ROI outlines over its projection, inline.

    The stand-in for ``view_stack_rois`` on a remote connection. That method
    hands the raw average to imshow with min-max scaling, so the bright
    blanking stripe compresses the tissue into a band of grey; this one
    stretches between the 1st and 99th percentiles first, and draws outlines
    rather than a filled overlay so the cells underneath stay visible.

    Pass ``image=`` to draw the mask over something else -- a correlation
    projection, a std projection -- when the mean does not show what the
    segmenter saw.
    """
    import numpy as np
    from matplotlib.figure import Figure

    from pygor.review.panels import _stretch, draw_outlines

    projection = image if image is not None else np.mean(recording.images, axis=0)
    mask = recording.rois
    rows, cols = projection.shape
    figure = Figure(figsize=(width / 100, width * rows / cols / 100), dpi=100)
    figure.set_facecolor("black")
    axis = figure.add_axes([0, 0, 1, 1])
    axis.set_axis_off()
    axis.imshow(_stretch(projection, low, high), cmap="Greys_r", origin="lower")
    draw_outlines(axis, mask, colour="yellow", linewidth=0.8)
    if labels:
        for roi_id in np.unique(mask):
            if roi_id >= 0:
                continue
            ys, xs = np.nonzero(mask == roi_id)
            axis.text(xs.mean(), ys.mean(), str(abs(int(roi_id)) - 1), color="white",
                      fontsize=5, ha="center", va="center")
    n_rois = int((np.unique(mask) < 0).sum())
    axis.text(0.01, 0.98, f"{getattr(recording, 'name', '')}  ·  {n_rois} ROIs",
              transform=axis.transAxes, color="white", fontsize=8, va="top")
    show(figure, caps, width=width)


def _renderable_class(mode):
    if mode == "none":
        return None
    from textual_image.renderable import HalfcellImage, SixelImage, TGPImage, UnicodeImage

    return {"tgp": TGPImage, "sixel": SixelImage, "halfcell": HalfcellImage,
            "unicode": UnicodeImage}.get(mode)


def clear_terminal_images() -> None:
    """Delete every image this process put on the screen.

    Needed on exit, on a crash, and before suspending to a shell: the kitty
    protocol stores images in the terminal, and placements left behind end up
    drawn over the shell prompt.
    """
    import sys

    try:
        stream = sys.__stdout__
        if stream is not None and stream.isatty():
            stream.write("\x1b_Ga=d,d=A\x1b\\")
            stream.flush()
    except Exception:
        pass
