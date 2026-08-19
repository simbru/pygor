"""Preprocessing and registration, with the image layers kept in step.

Both steps replace ``recording.images`` rather than editing it in place,
so the layers holding the old array have to be pointed at the new one
afterwards. Both are also destructive and slow, so they run off the GUI
thread and the panel says plainly what has already been applied.
"""

import numpy as np
from magicgui import magicgui
from napari.qt.threading import thread_worker
from qtpy.QtWidgets import QLabel, QPushButton, QVBoxLayout, QWidget

# Edge handling when shifting frames, from register()'s docstring
SHIFT_MODES = ("reflect", "constant", "nearest", "mirror", "wrap")

# Projection modes for batch and reference images
PROJECTIONS = ("std", "mean", "var", "median", "max")

NORMALIZATIONS = ("None", "phase")


def _default(recording, section, key, fallback):
    """Read a configured default, falling back if the config lacks it."""
    try:
        return recording.params._defaults[section][key]
    except (AttributeError, KeyError, TypeError):
        return fallback


class PreprocessingDock(QWidget):
    """Dock running the steps that alter the image stack itself."""

    def __init__(self, recording, viewer, bus=None, on_images_changed=None):
        super().__init__()
        self.recording = recording
        self.viewer = viewer
        self.bus = bus
        self.on_images_changed = on_images_changed

        self.status = QLabel()
        self.status.setWordWrap(True)

        self.reset_button = QPushButton("Reset images")
        self.reset_button.setToolTip(
            "Restore the stack from the backup taken before the first "
            "destructive step, and clear the applied flags."
        )
        self.reset_button.clicked.connect(self.reset_images)

        preprocess_form = self._preprocess_widget()
        register_form = self._register_widget()
        self._bind_to_bus(preprocess_form, "preprocessing")
        self._bind_to_bus(register_form, "registration")

        layout = QVBoxLayout()
        layout.addWidget(self.status)
        layout.addWidget(preprocess_form.native)
        layout.addWidget(register_form.native)
        layout.addWidget(self.reset_button)
        layout.addStretch(1)
        self.setLayout(layout)

        self.refresh_status()

    def _bind_to_bus(self, form, prefix):
        """Make a form a view of its config section rather than a copy.

        Only fields the config actually holds are bound; the rest, such
        as ``force``, are per-run choices with nothing to write back to.
        """
        if self.bus is None:
            return
        section = self.bus.section(prefix)
        for name in section:
            field = getattr(form, name, None)
            if field is None:
                continue
            field.changed.connect(
                lambda value, path=f"{prefix}.{name}": self.bus.set(path, value)
            )

        def _follow(path, form=form, prefix=prefix):
            if not path.startswith(f"{prefix}."):
                return
            name = path[len(prefix) + 1 :]
            field = getattr(form, name, None)
            if field is not None and field.value != self.bus.get(path):
                field.value = self.bus.get(path)

        self.bus.changed.connect(_follow)

    # ------------------------------------------------------------------
    # State
    # ------------------------------------------------------------------

    def refresh_status(self, message=None):
        """Show which steps have been applied, plus an optional message."""
        params = getattr(self.recording, "params", None)
        preprocessed = bool(getattr(params, "preprocessed", False))
        registered = bool(getattr(params, "registered", False))
        state = (
            f"Preprocessed: {'yes' if preprocessed else 'no'}    "
            f"Registered: {'yes' if registered else 'no'}"
        )
        self.status.setText(f"{state}\n{message}" if message else state)

    def _images_changed(self, message):
        self.refresh_status(message)
        if self.on_images_changed is not None:
            self.on_images_changed()

    def _run(self, worker, done_message):
        worker.returned.connect(lambda _=None: self._images_changed(done_message))
        worker.errored.connect(
            lambda exc: self.refresh_status(f"Failed: {exc}")
        )
        worker.start()
        return worker

    # ------------------------------------------------------------------
    # Steps
    # ------------------------------------------------------------------

    def run_preprocess(self, **kwargs):
        """Apply light-artifact correction, flip and detrending."""
        self.refresh_status("Preprocessing...")

        @thread_worker
        def job():
            self.recording.preprocess(**kwargs)
            return True

        return self._run(job(), "Preprocessing applied")

    def run_registration(self, **kwargs):
        """Motion-correct the stack."""
        if kwargs.get("normalization") == "None":
            kwargs["normalization"] = None
        self.refresh_status("Registering — this takes a while...")

        @thread_worker
        def job():
            return self.recording.register(**kwargs)

        return self._run(job(), "Registration applied")

    def reset_images(self):
        """Undo preprocessing and registration together."""
        try:
            self.recording.reset_images()
        except RuntimeError as exc:
            self.refresh_status(str(exc))
            return False
        self._images_changed("Images reset to the original stack")
        return True

    # ------------------------------------------------------------------
    # Widgets
    # ------------------------------------------------------------------

    def _preprocess_widget(self):
        get = lambda key, fallback: _default(
            self.recording, "preprocessing", key, fallback
        )

        @magicgui(call_button="Preprocess", layout="vertical")
        def preprocess(
            artifact_width: int = get("artifact_width", 3),
            flip_x: bool = get("flip_x", True),
            detrend: bool = get("detrend", True),
            smooth_window_s: float = get("smooth_window_s", 1000.0),
            time_bin: int = get("time_bin", 10),
            fix_first_frame: bool = get("fix_first_frame", True),
            force: bool = False,
        ):
            self.run_preprocess(
                artifact_width=artifact_width,
                flip_x=flip_x,
                detrend=detrend,
                smooth_window_s=smooth_window_s,
                time_bin=time_bin,
                fix_first_frame=fix_first_frame,
                force=force,
            )

        return preprocess

    def _register_widget(self):
        get = lambda key, fallback: _default(
            self.recording, "registration", key, fallback
        )

        @magicgui(
            call_button="Register",
            layout="vertical",
            mode={"choices": SHIFT_MODES},
            batch_mode={"choices": PROJECTIONS},
            reference_mode={"choices": PROJECTIONS},
            normalization={"choices": NORMALIZATIONS},
        )
        def register(
            n_reference_frames: int = get("n_reference_frames", 100),
            batch_size: int = get("batch_size", 50),
            upsample_factor: int = get("upsample_factor", 10),
            edge_crop: int = get("edge_crop", 1),
            order: int = get("order", 1),
            mode: str = get("mode", "reflect"),
            batch_mode: str = get("batch_mode", "std"),
            reference_mode: str = get("reference_mode", "mean"),
            normalization: str = str(get("normalization", "None")),
            parallel: bool = get("parallel", True),
            force: bool = False,
        ):
            self.run_registration(
                n_reference_frames=n_reference_frames,
                batch_size=batch_size,
                upsample_factor=upsample_factor,
                edge_crop=edge_crop,
                order=order,
                mode=mode,
                batch_mode=batch_mode,
                reference_mode=reference_mode,
                normalization=normalization,
                parallel=parallel,
                force=force,
            )

        return register
