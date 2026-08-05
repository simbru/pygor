"""Shared paths, plus the introspection used to build the method smoke tests."""

import inspect
import pathlib

import pygor.core.gui
import pygor.utils.helpinfo

# Downloaded separately (see the README), so tests that need it carry the
# `demo_data` marker and conftest skips them when it is missing.
PYGOR_ROOT = pathlib.Path(__file__).parents[3]
DEMO_DATA = PYGOR_ROOT / "examples" / "strf_demo_data.h5"

_REQUIRED_KINDS = (
    inspect.Parameter.POSITIONAL_ONLY,
    inspect.Parameter.POSITIONAL_OR_KEYWORD,
    inspect.Parameter.KEYWORD_ONLY,
)


def required_argument_names(func) -> list[str] | None:
    """Arguments `func` needs, ignoring self and anything with a default.

    None means the signature could not be read, which for our purposes means
    the method cannot be called blind.
    """
    try:
        params = list(inspect.signature(func).parameters.values())
    except (TypeError, ValueError):  # builtins and C functions have no signature
        return None
    return [
        param.name
        for param in params[1:]  # drop self
        if param.default is inspect.Parameter.empty and param.kind in _REQUIRED_KINDS
    ]


def _candidates(cls, exclude):
    """Public, non-interactive methods of `cls`, with their required arguments.

    Interactive methods are recognised by the marker that `pygor.core.gui` puts
    on them, not by name, so adding a napari method cannot silently give the
    suite something that blocks on a window. The `_by_channel` wrappers are left
    out because they are generated from the very methods already listed.
    """
    names = pygor.utils.helpinfo.get_methods_list(
        cls, with_returns=False, exclude_patterns=["_by_channel"]
    )
    for name in names:
        if name in exclude:
            continue
        func = inspect.getattr_static(cls, name)
        if pygor.core.gui.is_interactive(func):
            continue
        required = required_argument_names(func)
        if required is not None:
            yield name, required


def callable_without_arguments(cls, exclude=()) -> list[str]:
    """Public methods of `cls` a smoke test can call with no arguments."""
    return sorted(name for name, required in _candidates(cls, exclude) if not required)


def callable_with_roi_only(cls, exclude=()) -> list[str]:
    """Public methods of `cls` whose single required argument is an ROI index.

    Worth calling too: the no-argument smoke test never reaches them, which is
    how three delegating methods came to reference submodule functions that had
    been renamed.
    """
    return sorted(
        name for name, required in _candidates(cls, exclude) if required == ["roi"]
    )
