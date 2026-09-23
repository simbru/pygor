"""A line plot made of braille dots, for panes that must redraw at mouse speed.

Every picture elsewhere in the interface is a PNG the terminal has to be sent
in full. That is fine for a panel that changes when you press a key and wrong
for one that follows the pointer: the trace under the cursor would arrive a
frame or two behind the cursor itself, and over SSH considerably worse. Braille
is text, so it costs a line repaint and nothing else, and it is the only thing
that works at all in ``--graphics none``.

Two dots across and four down per character, which gives a 60x6 pane an
effective 120x24 plotting grid.
"""

from __future__ import annotations

# Braille dot numbering is historical rather than sequential: the fourth row
# was added to the standard later and its bits sit above the other six.
#   1 4     0x01 0x08
#   2 5     0x02 0x10
#   3 6     0x04 0x20
#   7 8     0x40 0x80
DOT_BITS = ((0x01, 0x08), (0x02, 0x10), (0x04, 0x20), (0x40, 0x80))
BRAILLE_BASE = 0x2800
BLANK = chr(BRAILLE_BASE)


class Canvas:
    """A grid of braille cells addressed in dots, with (0, 0) top left."""

    def __init__(self, cols: int, rows: int):
        self.cols = max(1, cols)
        self.rows = max(1, rows)
        self.cells = [[0] * self.cols for _ in range(self.rows)]

    @property
    def width(self) -> int:
        return self.cols * 2

    @property
    def height(self) -> int:
        return self.rows * 4

    def set(self, x: int, y: int) -> None:
        if 0 <= x < self.width and 0 <= y < self.height:
            self.cells[y // 4][x // 2] |= DOT_BITS[y % 4][x % 2]

    def vline(self, x: int, y0: int, y1: int) -> None:
        for y in range(min(y0, y1), max(y0, y1) + 1):
            self.set(x, y)

    def lines(self) -> list[str]:
        return ["".join(chr(BRAILLE_BASE + bits) for bits in row) for row in self.cells]


def _tick(value: float) -> str:
    """A number narrow enough for the gutter and still distinguishable.

    Raw two-photon counts sit around 50000-60000, where an exponent format
    renders the top, middle and bottom of the axis as ``6e+04`` three times
    over. Thousands get a ``k`` instead, which keeps them apart at the same
    width.
    """
    if value == 0:
        return "0"
    magnitude = abs(value)
    if magnitude >= 1e6 or magnitude < 0.01:
        return f"{value:.0e}"
    if magnitude >= 1000:
        return f"{value / 1000:.1f}k"
    if magnitude >= 100:
        return f"{value:.0f}"
    if magnitude >= 10:
        return f"{value:.1f}"
    return f"{value:.2f}"


def plot(values, *, cols: int = 60, rows: int = 6, low=None, high=None) -> list[str]:
    """``values`` as braille lines, top row first.

    Each dot column averages the samples that fall in it, and consecutive
    columns are joined vertically, so a trace with more samples than dots reads
    as a line rather than as a dotted cloud.
    """
    import numpy as np

    canvas = Canvas(cols, rows)
    values = np.asarray(values, dtype=float)
    if values.size == 0:
        return canvas.lines()

    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return canvas.lines()
    low = float(finite.min()) if low is None else float(low)
    high = float(finite.max()) if high is None else float(high)
    if high <= low:
        high = low + 1.0

    # Fewer samples than dot columns happens as soon as the trace is zoomed
    # in. Split as-is, most buckets would be empty and the line would be a
    # staircase of isolated dots, so stretch the samples across the width.
    # A NaN spreads to its neighbouring interval, which keeps the gap a gap.
    if 1 < values.size < canvas.width:
        values = np.interp(np.linspace(0, values.size - 1, canvas.width),
                           np.arange(values.size), values)

    # One bucket per dot column, averaged.
    buckets = np.array_split(values, canvas.width)
    span = canvas.height - 1
    previous = None
    for x, bucket in enumerate(buckets):
        # A bucket that is entirely NaN is a gap in the trace, not a zero, and
        # nanmean warns rather than answering. Leaving the column blank draws
        # the gap, which is the honest picture.
        if bucket.size == 0 or not np.any(np.isfinite(bucket)):
            continue
        sample = np.nanmean(bucket)
        scaled = (sample - low) / (high - low)
        y = int(round((1.0 - min(max(scaled, 0.0), 1.0)) * span))
        if previous is None:
            canvas.set(x, y)
        else:
            canvas.vline(x, previous, y)
        previous = y
    return canvas.lines()


# Eighth-blocks, for a density bar. Built by code point rather than written
# out, so nothing in the pipeline between here and the terminal has to survive
# the literals.
BLOCKS = (" ", *(chr(0x2580 + n) for n in range(1, 9)))


def event_row(positions, *, cols: int, x_max: float,
              x_min: float = 0.0) -> tuple[str, bool]:
    """A row marking where events fall along the same x axis as a plot.

    Returns the row and whether the events were drawn individually. Triggers
    come at 5 Hz over a recording an hour long, which is a hundred and thirty
    of them per dot column: drawn as ticks that is a solid rule, and a solid
    rule hides the one thing worth seeing, which is where the rate changed. So
    past the point where ticks can be told apart this switches to a density
    bar, and the caller says the rest in numbers.
    """
    import numpy as np

    times = np.asarray(positions, dtype=float)
    times = times[np.isfinite(times) & (times >= x_min) & (times <= x_max)]
    span = x_max - x_min
    if times.size == 0 or span <= 0:
        return " " * cols, True

    # Resolvable when ticks can be spaced at least two dot columns apart;
    # closer than that and adjacent ticks merge into a line.
    if times.size <= cols:
        canvas = Canvas(cols, 1)
        for time in times:
            x = int((time - x_min) / span * (canvas.width - 1))
            canvas.vline(x, 0, canvas.height - 1)
        return canvas.lines()[0], True

    counts, _ = np.histogram(times, bins=cols, range=(x_min, x_max))
    ceiling = counts.max() or 1
    scaled = np.ceil(counts / ceiling * 8).astype(int)
    return "".join(BLOCKS[min(n, 8)] for n in scaled), False


def framed(values, *, cols: int = 60, rows: int = 6, low=None, high=None,
           x_max=None, x_min=0.0, x_unit="s", events=None, event_label="trig",
           event_note="") -> str:
    """:func:`plot` with a labelled y axis down the left and an x axis under it.

    ``values`` spans ``x_min`` to ``x_max``; a zoomed-in caller passes the
    slice and where it sits, and the events are placed on the same stretch.
    """
    import numpy as np

    array = np.asarray(values, dtype=float)
    finite = array[np.isfinite(array)] if array.size else array
    if finite.size:
        lo = float(finite.min()) if low is None else float(low)
        hi = float(finite.max()) if high is None else float(high)
    else:
        lo, hi = 0.0, 1.0
    if hi <= lo:
        hi = lo + 1.0

    body = plot(array, cols=cols, rows=rows, low=lo, high=hi)

    def label_for(value):
        # A midpoint that is zero to within rounding is zero: "-1e-05" on the
        # axis of a trace centred on zero is noise reported as a measurement.
        return _tick(0.0 if abs(value) < (hi - lo) * 1e-4 else value)

    labels = [""] * len(body)
    if labels:
        labels[0] = label_for(hi)
        labels[-1] = label_for(lo)
        if len(labels) > 2:
            labels[len(labels) // 2] = label_for((hi + lo) / 2)
    gutter = max(len(label) for label in labels)

    lines = [f"{label.rjust(gutter)} │{row}" for label, row in zip(labels, body)]

    # The event row shares the gutter and the bar, so a tick is directly under
    # the sample it happened at. Built here rather than by the caller for
    # exactly that reason.
    if events is not None and x_max:
        row, _ = event_row(events, cols=cols, x_max=x_max, x_min=x_min)
        lines.append(f"{event_label.rjust(gutter)} │{row}")

    lines.append(f"{' ' * gutter} └{'─' * cols}")
    if x_max:
        # Whole seconds for a whole recording, which is 327s or 3174s, never
        # 3.2ks; a zoomed window narrower than that needs the decimals, or
        # both ends of a 0.4s stretch read the same.
        decimals = 0 if x_max - x_min >= 10 else 1 if x_max - x_min >= 1 else 2

        def at(value):
            return f"{value:.{decimals}f}"

        left = f"{' ' * gutter}  {at(x_min) if x_min else '0'}"
        right = f"{at(x_max)}{x_unit}"
        pad = max(1, cols - len(left) - len(right) + gutter + 2)
        lines.append(f"{left}{' ' * pad}{right}")
    if event_note:
        lines.append(f"{' ' * (gutter + 2)}{event_note}")
    return "\n".join(lines)
