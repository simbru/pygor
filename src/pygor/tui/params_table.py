"""An editable parameter table for the terminal.

The counterpart of the Qt ``ParamEditorWidget``: three columns, parameter /
value / type, over any nested dict. Both share one parser, so an edit means
the same thing in either. Edits are held as overrides rather than written into
the source dict, so a caller can tell what changed from what was there.
"""

from __future__ import annotations

from textual import on
from textual.binding import Binding
from textual.containers import Vertical
from textual.coordinate import Coordinate
from textual.screen import Screen
from textual.widgets import DataTable, Input, OptionList, Static
from textual.widgets.option_list import Option

from pygor.core.gui.param_values import flatten, nest, parse_value, type_name


class ValuePrompt(Screen):
    """One value, typed; parsed to the original's type before it is accepted."""

    BINDINGS = [Binding("escape", "cancel", "cancel")]

    def __init__(self, path, current):
        super().__init__()
        self.path = path
        self.current = current

    def compose(self):
        yield Vertical(
            Static(f"[b]{self.path}[/b]  ({type_name(self.current)}, currently {self.current!r})"),
            Input(value=repr(self.current) if isinstance(self.current, str) else str(self.current),
                  id="value"),
            Static("", id="error", classes="error"),
            id="value-box",
        )

    def on_mount(self):
        self.query_one("#value", Input).focus()

    @on(Input.Submitted)
    def submit(self, event):
        text = event.value
        if isinstance(self.current, str) and len(text) >= 2 and text[0] == text[-1] == "'":
            text = text[1:-1]
        try:
            value = parse_value(text, self.current)
        except (ValueError, SyntaxError) as error:
            self.query_one("#error", Static).update(f"{error}")
            return
        self.dismiss((self.path, value))

    def action_cancel(self):
        self.dismiss(None)


class ChoicePrompt(Screen):
    """Pick one of a fixed set: the terminal's drop-down."""

    BINDINGS = [Binding("escape", "cancel", "cancel")]

    def __init__(self, path, current, choices):
        super().__init__()
        self.path = path
        self.current = current
        self.choices = list(choices)

    def compose(self):
        options = OptionList(*[Option(str(c), id=str(c)) for c in self.choices], id="choices")
        yield Vertical(Static(f"[b]{self.path}[/b]  (currently {self.current!r})"), options,
                       id="value-box")

    def on_mount(self):
        options = self.query_one("#choices", OptionList)
        options.focus()
        if str(self.current) in [str(c) for c in self.choices]:
            options.highlighted = [str(c) for c in self.choices].index(str(self.current))

    def on_option_list_option_selected(self, event):
        self.dismiss((self.path, event.option.id))

    def action_cancel(self):
        self.dismiss(None)


class ParamTable(DataTable):
    """(parameter, value, type) rows over a nested dict, with edits as overrides.

    ``sections`` narrows the rows to dotted prefixes such as
    ``("segmentation.blob",)``; everything else in the dict is left out rather
    than shown greyed, because a table of two hundred parameters is not
    something anyone edits in a terminal.
    """

    BINDINGS = [
        Binding("enter", "edit", "edit value"),
        Binding("backspace", "revert", "revert", show=False),
        Binding("u", "revert_all", "revert all"),
    ]

    def __init__(self, values: dict, *, sections=None, choices=None, gates=None, **kwargs):
        super().__init__(**kwargs)
        self.source = values
        self.sections = tuple(sections or ())
        # path -> allowed values; such a row edits through a picker, not free text.
        self.choices: dict[str, tuple] = dict(choices or {})
        # selector path -> section it gates. Rows under <section>.<X>.* show only
        # when X is the selector's current value (or X is "general"), so choosing
        # a segmentation mode hides every other mode's parameters.
        self.gates: dict[str, str] = dict(gates or {})
        self.overrides: dict[str, object] = {}
        self._rows: list[tuple[str, object]] = []
        self.cursor_type = "row"
        self.zebra_stripes = True

    def on_mount(self):
        self.add_column("parameter", key="parameter")
        self.add_column("value", key="value")
        self.add_column("type", key="type")
        self.reload()

    def current_value(self, path):
        return self.overrides.get(path, dict(flatten(self.source)).get(path))

    def _visible(self, path) -> bool:
        if self.sections and not any(path.startswith(s) for s in self.sections):
            return False
        for selector, section in self.gates.items():
            prefix = section + "."
            if not path.startswith(prefix) or path == selector:
                continue
            sub = path[len(prefix):].split(".", 1)[0]
            chosen = str(self.current_value(selector))
            # startswith, so a mode's helper section (cellpose_postprocess)
            # travels with it.
            if sub != "general" and not sub.startswith(chosen):
                return False
        return True

    def reload(self):
        self.clear()
        self._rows = [(path, value) for path, value in flatten(self.source) if self._visible(path)]
        for path, value in self._rows:
            self.add_row(path, self._shown(path, value), type_name(value), key=path)

    def _shown(self, path, original):
        if path in self.overrides:
            return f"[b yellow]{self.overrides[path]!r}[/]  (was {original!r})"
        return repr(original)

    @property
    def current_path(self):
        if not self.row_count:
            return None
        return self._rows[self.cursor_row][0]

    def nested_overrides(self) -> dict:
        """Edits as a nested dict, the shape a recipe or config wants."""
        return nest(self.overrides)

    def action_edit(self):
        path = self.current_path
        if path is None:
            return
        original = dict(self._rows)[path]
        current = self.overrides.get(path, original)
        if path in self.choices:
            self.app.push_screen(ChoicePrompt(path, current, self.choices[path]), self._apply)
        else:
            self.app.push_screen(ValuePrompt(path, current), self._apply)

    def _apply(self, result):
        if result is None:
            return
        path, value = result
        original = dict(self._rows)[path]
        if value == original:
            self.overrides.pop(path, None)
        else:
            self.overrides[path] = value
        self._after_change(path)

    def _repaint(self, path):
        index = next(i for i, (p, _) in enumerate(self._rows) if p == path)
        self.update_cell_at(Coordinate(index, 1), self._shown(path, self._rows[index][1]))

    def action_revert(self):
        path = self.current_path
        if path in self.overrides:
            del self.overrides[path]
            self._after_change(path)

    def action_revert_all(self):
        paths = list(self.overrides)
        self.overrides.clear()
        if any(p in self.gates for p in paths):
            self.reload()
        else:
            for path in paths:
                self._repaint(path)

    def _after_change(self, path):
        if path in self.gates:
            self.reload()
            self.move_cursor(row=next(i for i, (p, _) in enumerate(self._rows) if p == path))
        else:
            self._repaint(path)
