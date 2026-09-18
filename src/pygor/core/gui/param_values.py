"""Turning edited text back into typed parameter values.

Shared by the Qt parameter editor and the terminal one, so the two cannot
disagree about what "0.02" or "true" means. Kept free of any GUI import so the
terminal side can use it where Qt is not installed.
"""

from __future__ import annotations

import ast

# bool before int: a bool is an int to isinstance, and the other order would
# label True as "int" and then parse an edit of it as a number.
TYPE_NAMES = {bool: "bool", int: "int", float: "float", str: "str", list: "list"}


def type_name(value) -> str:
    for typ, name in TYPE_NAMES.items():
        if isinstance(value, typ):
            return name
    return type(value).__name__


def parse_value(text, original_value):
    """Parse edited text back to the original value's type.

    Raises ValueError when the text cannot be read as that type, so an editor
    can refuse the edit and keep the old value rather than store a string
    where a number was.
    """
    if isinstance(original_value, bool):
        lower = text.strip().lower()
        if lower in ("true", "1", "yes"):
            return True
        if lower in ("false", "0", "no"):
            return False
        raise ValueError(f"Cannot parse '{text}' as bool")

    if isinstance(original_value, int):
        # An int in a TOML file is usually a float quantity that happened to be
        # written without a decimal point (max_sigma = 2), so a fractional edit
        # promotes to float rather than being truncated to the nearest int --
        # which is what this used to do, silently, turning 2.5 into 2. A
        # parameter that is genuinely integer fails loudly downstream instead.
        number = float(text)
        return int(number) if number == int(number) else number

    if isinstance(original_value, float):
        return float(text)

    if isinstance(original_value, list):
        parsed = ast.literal_eval(text)
        if not isinstance(parsed, list):
            raise ValueError(f"Expected a list, got {type(parsed).__name__}")
        return parsed

    if original_value is None:
        # No type to infer from: take a literal if the text is one, else a string.
        try:
            return ast.literal_eval(text)
        except (ValueError, SyntaxError):
            return text

    # str or unknown -- return as-is
    return text


def flatten(section: dict, prefix="") -> list[tuple[str, object]]:
    """A nested dict as (dotted.path, leaf) pairs, in definition order."""
    rows = []
    for key, value in section.items():
        path = f"{prefix}.{key}" if prefix else key
        if isinstance(value, dict):
            rows.extend(flatten(value, path))
        else:
            rows.append((path, value))
    return rows


def nest(overrides: dict[str, object]) -> dict:
    """The inverse of flatten: {"a.b": 1} -> {"a": {"b": 1}}."""
    tree: dict = {}
    for path, value in overrides.items():
        node = tree
        parts = path.split(".")
        for part in parts[:-1]:
            node = node.setdefault(part, {})
        node[parts[-1]] = value
    return tree
