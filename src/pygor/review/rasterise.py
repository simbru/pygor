"""Turn a panel's figure into PNG bytes, at an exact pixel size.

A terminal front-end has a fixed grid of character cells, so a panel is asked for
a specific number of pixels and must come back at that size: anything else gets
resampled on the way to the screen, and a 24x40 receptive field cannot afford to
be softened. So figures are built at ``px / dpi`` inches rather than scaled
afterwards.

Two disciplines live here because getting them wrong is expensive and silent.
Figures are closed unconditionally, including when the renderer raises -- several
pygor plotting functions build theirs through pyplot, which keeps a global
reference, so a review session that renders a few hundred panels would otherwise
grow until it died. And renderers that are not thread-safe are serialised, since
pyplot's global state cannot survive two panels being drawn at once.
"""

from __future__ import annotations

import dataclasses
import hashlib
import io
import threading
from collections import OrderedDict

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import image
from matplotlib.backends.backend_agg import FigureCanvasAgg

DEFAULT_DPI = 100

# Serialises renderers that touch pyplot. An RLock because a panel may call
# another panel's renderer.
PYPLOT_LOCK = threading.RLock()


@dataclasses.dataclass(frozen=True)
class PanelKey:
    """What makes two rendered panels the same picture.

    ``sources`` carries the (mtime, size) of every recording the panel drew
    from, so reprocessing a recording invalidates its panels rather than
    leaving a stale picture next to a fresh verdict.
    """

    panel: str
    fov_uid: str
    condition: str
    role: str
    roi: int | None
    channel: int
    width: int
    height: int
    dpi: int
    params_hash: str
    sources: tuple

    def digest(self) -> str:
        return hashlib.sha1(repr(dataclasses.astuple(self)).encode()).hexdigest()


@dataclasses.dataclass(frozen=True)
class PanelImage:
    png: bytes
    width: int
    height: int
    key: PanelKey
    meta: dict = dataclasses.field(default_factory=dict)

    @property
    def nbytes(self) -> int:
        return len(self.png)


def png_size(data: bytes) -> tuple[int, int]:
    """(width, height) from a PNG's IHDR, without decoding the image.

    The rendered size is not always the requested one -- trimming changes it --
    and a caller that scales the panel needs its real aspect or it stretches it.
    """
    if len(data) < 24 or data[:8] != b"\x89PNG\r\n\x1a\n":
        return (0, 0)
    return (
        int.from_bytes(data[16:20], "big"),
        int.from_bytes(data[20:24], "big"),
    )


def hash_params(params: dict) -> str:
    if not params:
        return ""
    items = sorted((str(k), repr(v)) for k, v in params.items())
    return hashlib.sha1(repr(items).encode()).hexdigest()[:12]


def size_figure(figure, width, height, dpi=DEFAULT_DPI):
    """Force a figure to an exact pixel size.

    Used for figures a pygor plotting function built itself, where the size was
    chosen for a paper rather than for a terminal pane.
    """
    figure.set_dpi(dpi)
    figure.set_size_inches(width / dpi, height / dpi)
    return figure


def trim_uniform_border(rgba):
    """Crop away rows and columns that are a single flat colour.

    Some pygor plotting functions lay their content out for a page and leave a
    wide empty band when squeezed into a pane. In a terminal that band is the
    scarcest resource there is, so it goes. Only entirely uniform edges are
    removed, so nothing that carries information can be cropped.
    """
    if rgba.ndim != 3 or rgba.shape[0] < 2 or rgba.shape[1] < 2:
        return rgba
    rows = ~np.all(rgba == rgba[:, :1, :], axis=(1, 2))
    cols = ~np.all(rgba == rgba[:1, :, :], axis=(0, 2))
    if not rows.any() or not cols.any():
        return rgba
    top, bottom = np.flatnonzero(rows)[[0, -1]]
    left, right = np.flatnonzero(cols)[[0, -1]]
    return rgba[top : bottom + 1, left : right + 1]


def figure_to_png(figure, *, width=None, height=None, dpi=DEFAULT_DPI,
                  close=True, transparent=False, trim=False) -> bytes:
    """Rasterise a figure, closing it afterwards whether or not this succeeds."""
    try:
        if width and height:
            size_figure(figure, width, height, dpi)
        # Agg explicitly: the renderer may have been built under whatever
        # backend the session happens to have.
        if not isinstance(figure.canvas, FigureCanvasAgg):
            FigureCanvasAgg(figure)
        if trim:
            figure.canvas.draw()
            rgba = np.asarray(figure.canvas.buffer_rgba())
            cropped = trim_uniform_border(rgba)
            buffer = io.BytesIO()
            image.imsave(buffer, cropped, format="png")
            return buffer.getvalue()
        buffer = io.BytesIO()
        figure.savefig(
            buffer,
            format="png",
            dpi=figure.get_dpi(),
            transparent=transparent,
            bbox_inches=None,  # honour the exact size rather than trimming to content
        )
        return buffer.getvalue()
    finally:
        if close:
            close_figure(figure)


def close_figure(figure) -> None:
    """Drop a figure from pyplot's registry as well as releasing it.

    ``plt.close`` is the only thing that removes a pyplot-created figure from
    the global manager; ``del`` leaves it there forever.
    """
    try:
        plt.close(figure)
    except Exception:
        pass


class PanelCache:
    """LRU of rendered PNGs, so going back to a panel is free."""

    def __init__(self, max_items=512):
        self.max_items = max_items
        self._items: OrderedDict[str, PanelImage] = OrderedDict()
        self.hits = 0
        self.misses = 0

    def get(self, key: PanelKey):
        digest = key.digest()
        image = self._items.get(digest)
        if image is None:
            self.misses += 1
            return None
        self.hits += 1
        self._items.move_to_end(digest)
        return image

    def put(self, image: PanelImage) -> PanelImage:
        digest = image.key.digest()
        self._items[digest] = image
        self._items.move_to_end(digest)
        while len(self._items) > self.max_items:
            self._items.popitem(last=False)
        return image

    def clear(self) -> None:
        self._items.clear()

    def stats(self) -> dict:
        return {
            "items": len(self._items),
            "bytes": sum(i.nbytes for i in self._items.values()),
            "hits": self.hits,
            "misses": self.misses,
        }
