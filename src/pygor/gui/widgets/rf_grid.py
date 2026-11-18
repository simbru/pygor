"""Every ROI's receptive field at once, as one mosaic.

One tile per ROI, in ROI order and numbered with its label, either as the
time-collapsed map or played through time. Each tile is scaled to its own
largest value, so weak RFs stay visible next to strong ones: the panel shows
shape, not relative strength. Clicking a tile selects that ROI everywhere
else, so the plot dock's RF view can take over for a closer look.

The mosaic is a single image rebuilt per frame rather than an axis per ROI,
which keeps playback smooth with a hundred or more ROIs. Tiles keep a usable
size and the mosaic scrolls, rather than shrinking every tile to fit.
"""

import numpy as np
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from matplotlib.patches import Rectangle
from qtpy.QtCore import Qt, QTimer
from qtpy.QtWidgets import (
    QCheckBox,
    QComboBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSlider,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from pygor.gui.rf import GAP as _GAP
from pygor.gui.rf import RFData, draw_fit, mosaic, normalised  # noqa: F401
from pygor.gui.roi_bridge import label_to_trace_index, trace_index_to_label

# Screen width a tile aims for when the column count is left on auto
_TILE_PX = 110


class RFGridDock(QWidget):
    """Mosaic of every ROI's RF, static or as a movie, in one colour channel."""

    def __init__(self, recording, viewer, labels_layer=None):
        super().__init__()
        self.recording = recording
        self.viewer = viewer
        self._labels_layer = None
        self._rf = RFData(recording)
        self._image = None
        self._highlight = None
        self._rejected_marks = []
        self._n_cols = 1
        # Per-ROI normalised maps and movies for the shown channel
        self._maps = self._movies = None
        # napari pads a dock with stretch unless it asks to expand
        self.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Expanding)

        self.channel_box = QComboBox()
        self.channel_box.setToolTip("Colour channel shown")
        self.channel_box.currentIndexChanged.connect(lambda _: self.redraw())

        self.movie_box = QCheckBox("Movie")
        self.movie_box.setToolTip(
            "Show the STRFs frame by frame instead of the time-collapsed maps"
        )
        self.movie_box.toggled.connect(self._on_movie_toggled)

        self.fits_box = QCheckBox("Fits")
        self.fits_box.setToolTip(
            "Draw each RF's Gaussian fit (solid) and the region its fit index "
            "is scored over (dashed)"
        )
        self.fits_box.toggled.connect(lambda _: self.redraw())

        self.columns_spin = QSpinBox()
        self.columns_spin.setPrefix("cols ")
        self.columns_spin.setRange(0, 50)
        # 0 follows the panel's width
        self.columns_spin.setSpecialValueText("cols auto")
        self.columns_spin.setToolTip("Tiles per row; auto fits the panel width")
        self.columns_spin.valueChanged.connect(lambda _: self.redraw())

        self.play_button = QPushButton("Play")
        self.play_button.setCheckable(True)
        self.play_button.toggled.connect(self._on_play_toggled)
        self.frame_slider = QSlider(Qt.Horizontal)
        self.frame_slider.valueChanged.connect(self._show_frame)
        self.frame_label = QLabel()
        self.timer = QTimer(self)
        self.timer.timeout.connect(self._advance)

        self.status = QLabel()
        self.status.setWordWrap(True)
        self.fits_note = QLabel(
            "Solid: Gaussian fitted to |RF|. Dashed: region the fit index is "
            "scored over. Both at the configured n_sigma."
        )
        self.fits_note.setWordWrap(True)
        self.fits_note.setVisible(False)

        # The mosaic fills the figure; the canvas is as tall as the mosaic
        # needs at the panel's width, and scrolls
        self.figure = Figure(figsize=(4, 4))
        self.canvas = FigureCanvas(self.figure)
        self.canvas.mpl_connect("button_press_event", self._on_click)
        self.scroll = QScrollArea()
        self.scroll.setWidgetResizable(True)
        self.scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.scroll.setWidget(self.canvas)

        options = QHBoxLayout()
        options.addWidget(self.channel_box)
        options.addWidget(self.movie_box)
        options.addWidget(self.fits_box)
        options.addStretch(1)
        options.addWidget(self.columns_spin)

        playback = QHBoxLayout()
        playback.addWidget(self.play_button)
        playback.addWidget(self.frame_slider, stretch=1)
        playback.addWidget(self.frame_label)

        layout = QVBoxLayout()
        layout.addLayout(options)
        layout.addLayout(playback)
        layout.addWidget(self.status)
        layout.addWidget(self.fits_note)
        layout.addWidget(self.scroll, stretch=1)
        self.setLayout(layout)

        self._first_draw = True
        self._frame_set = False
        self._set_playback_enabled(False)
        self.bind_layer(labels_layer)

    # ------------------------------------------------------------------
    # Layer binding, as the other docks do it
    # ------------------------------------------------------------------

    @property
    def labels_layer(self):
        layer = self._labels_layer
        if layer is None or layer not in self.viewer.layers:
            return None
        return layer

    def bind_layer(self, layer):
        self._labels_layer = layer
        if layer is not None:
            layer.events.selected_label.connect(self._on_label_changed)

    def rebind(self, layer):
        self.bind_layer(layer)
        self._on_label_changed()

    def disconnect_events(self):
        """Stop playback before the recording is detached."""
        self.play_button.setChecked(False)

    def showEvent(self, event):
        """Fit and draw on first sight: fitting a recording takes seconds."""
        super().showEvent(event)
        if self._first_draw:
            self._first_draw = False
            self.redraw()

    def resizeEvent(self, event):
        """Keep tiles at their size: reflow on auto columns, else rescale."""
        super().resizeEvent(event)
        if self._image is None:
            return
        if self._columns() != self._n_cols:
            self.redraw()
        else:
            self._fit_canvas()

    def _columns(self):
        """Tiles per row: the spin box's, or as many as fit the panel width."""
        chosen = self.columns_spin.value()
        if chosen:
            return chosen
        return max(1, self.scroll.viewport().width() // _TILE_PX)

    def _fit_canvas(self):
        """Make the canvas as tall as the mosaic is at the panel's width."""
        if self._maps is None:
            return
        n, h, w = self._maps.shape
        n_rows = int(np.ceil(n / self._n_cols))
        aspect = (n_rows * (h + _GAP) - _GAP) / (self._n_cols * (w + _GAP) - _GAP)
        width = self.scroll.viewport().width()
        self.canvas.setMinimumHeight(int(np.ceil(width * aspect)))

    # ------------------------------------------------------------------
    # Drawing
    # ------------------------------------------------------------------

    def _load(self):
        data = self._rf.refresh()
        if self.channel_box.count() != data.n_colours:
            self.channel_box.blockSignals(True)
            self.channel_box.clear()
            self.channel_box.addItems([f"ch{c}" for c in range(data.n_colours)])
            self.channel_box.blockSignals(False)
        self.channel_box.setVisible(data.n_colours > 1)
        return data

    def _rows(self, data):
        channel = max(0, self.channel_box.currentIndex())
        return [data.flat(roi, channel) for roi in range(data.n_rois)]

    def redraw(self):
        data = self._load()
        rows = self._rows(data)
        self._maps = normalised(data.maps[rows])
        self._movies = normalised(data.movies[rows])
        n_cols = self._n_cols = self._columns()
        h, w = self._maps.shape[-2:]

        self.figure.clear()
        ax = self.ax = self.figure.add_axes((0, 0, 1, 1))
        cmap = self._cmap()
        self._image = ax.imshow(mosaic(self._maps, n_cols), cmap=cmap, vmin=-1,
                                vmax=1, interpolation="nearest")
        ax.set_axis_off()
        ax.set_anchor("N")
        for roi in range(data.n_rois):
            row, col = divmod(roi, n_cols)
            ax.text(col * (w + _GAP), row * (h + _GAP),
                    str(trace_index_to_label(roi, self.recording.rois)),
                    fontsize=7, va="top", ha="left", color="0.2")
            if self.fits_box.isChecked():
                x, y = col * (w + _GAP), row * (h + _GAP)
                # A failed fit can be far larger than its map; keep it on its tile
                tile = Rectangle((x - 0.5, y - 0.5), w, h, transform=ax.transData)
                for patch in draw_fit(ax, data, rows[roi], offset=(x, y)):
                    patch.set_clip_path(tile)
        self.fits_note.setVisible(self.fits_box.isChecked())
        self._highlight = None
        self._rejected_marks = []
        self.mark_rejected(draw=False)
        self._fit_canvas()

        self.frame_slider.blockSignals(True)
        self.frame_slider.setRange(0, self._movies.shape[1] - 1)
        if not self._frame_set:
            # Open on the frame where the population responds most strongly
            self._frame_set = True
            strength = np.abs(np.nan_to_num(self._movies)).mean(axis=(0, 2, 3))
            self.frame_slider.setValue(int(np.argmax(strength)))
        self.frame_slider.blockSignals(False)
        self.status.setText(
            f"{data.n_rois} ROIs, each scaled to its own peak. Click one to select "
            "it; crossed out = RF rejected."
        )
        if self.movie_box.isChecked():
            self._show_frame(self.frame_slider.value(), draw=False)
        self._on_label_changed(draw=False)
        self.canvas.draw_idle()

    @staticmethod
    def _cmap():
        import matplotlib

        cmap = matplotlib.colormaps["RdBu_r"].copy()
        cmap.set_bad("white")
        return cmap

    def _tile_origin(self, roi):
        h, w = self._maps.shape[-2:]
        row, col = divmod(roi, self._n_cols)
        return col * (w + _GAP), row * (h + _GAP)

    def mark_rejected(self, draw=True):
        """Cross out the tiles whose RF is rejected on the recording."""
        from pygor.strf.rejection import rejected

        if self._image is None:
            return
        for artist in self._rejected_marks:
            artist.remove()
        self._rejected_marks = []
        h, w = self._maps.shape[-2:]
        for roi in rejected(self.recording):
            if not 0 <= roi < len(self._maps):
                continue
            x, y = self._tile_origin(roi)
            xs, ys = [x - 0.5, x + w - 0.5], [y - 0.5, y + h - 0.5]
            self._rejected_marks += self.ax.plot(xs, ys, color="k", lw=1.5)
            self._rejected_marks += self.ax.plot(xs, ys[::-1], color="k", lw=1.5)
        if draw:
            self.canvas.draw_idle()

    def _on_label_changed(self, event=None, draw=True):
        """Outline the selected ROI's tile."""
        if self._image is None:
            return
        if self._highlight is not None:
            self._highlight.remove()
            self._highlight = None
        layer = self.labels_layer
        if layer is not None:
            roi = label_to_trace_index(int(layer.selected_label), self.recording.rois)
            if 0 <= roi < len(self._maps):
                h, w = self._maps.shape[-2:]
                x, y = self._tile_origin(roi)
                self._highlight = self.ax.add_patch(Rectangle(
                    (x - 0.5, y - 0.5), w, h, fill=False, lw=1.5, color="k",
                ))
        if draw:
            self.canvas.draw_idle()

    def _on_click(self, event):
        if event.inaxes is not getattr(self, "ax", None) or event.xdata is None:
            return
        h, w = self._maps.shape[-2:]
        col = int((event.xdata + 0.5) // (w + _GAP))
        row = int((event.ydata + 0.5) // (h + _GAP))
        n_cols = self._n_cols
        roi = row * n_cols + col
        if col >= n_cols or not 0 <= roi < len(self._maps):
            return
        layer = self.labels_layer
        if layer is not None:
            layer.selected_label = trace_index_to_label(roi, self.recording.rois)

    # ------------------------------------------------------------------
    # Movie
    # ------------------------------------------------------------------

    def _set_playback_enabled(self, enabled):
        for widget in (self.play_button, self.frame_slider, self.frame_label):
            widget.setEnabled(enabled)

    def _on_movie_toggled(self, movie):
        self._set_playback_enabled(movie)
        if not movie:
            self.play_button.setChecked(False)
            if self._image is not None:
                self._image.set_data(mosaic(self._maps, self._n_cols))
                self.canvas.draw_idle()
            self.frame_label.clear()
            return
        self._show_frame(self.frame_slider.value())

    def _show_frame(self, frame, draw=True):
        if self._image is None or not self.movie_box.isChecked():
            return
        self._image.set_data(mosaic(self._movies[:, frame], self._n_cols))
        self.frame_label.setText(self._rf.frame_label(frame))
        if draw:
            self.canvas.draw_idle()

    def _advance(self):
        self.frame_slider.setValue(
            (self.frame_slider.value() + 1) % (self.frame_slider.maximum() + 1)
        )

    def _on_play_toggled(self, playing):
        self.play_button.setText("Pause" if playing else "Play")
        if playing:
            self.timer.start(self._rf.refresh().frame_interval_ms())
        else:
            self.timer.stop()
