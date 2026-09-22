"""``pygor-proofread`` — start the terminal cockpit.

Everything heavy is imported inside :func:`main`, so a missing ``[tui]`` extra
produces a sentence telling you how to fix it rather than an import traceback
from somewhere in the middle of Textual.
"""

from __future__ import annotations

import argparse
import importlib
import os
import sys

DATASETS = {
    "Chromatic SWN": "analyses.review_datasets.chromatic_swn",
    "Achromatic SWN unified": "analyses.review_datasets.achromatic_swn",
    # short forms, for typing on a command line
    "chromatic": "analyses.review_datasets.chromatic_swn",
    "achromatic": "analyses.review_datasets.achromatic_swn",
}

INSTALL_HINT = (
    "The proofreading cockpit needs the [tui] extra.\n\n"
    "    uv pip install 'pygor[tui]'\n\n"
    "Or directly:\n"
    "    uv pip install 'textual>=8.0,<9' 'textual-image[textual]>=0.14'\n"
)


def build_parser():
    parser = argparse.ArgumentParser(prog="pygor-proofread", description=__doc__.splitlines()[0])
    parser.add_argument("dataset", nargs="?", default="Chromatic SWN")
    parser.add_argument("--binding", help="import path of a review binding module, "
                                          "instead of a name from the built-in list")
    parser.add_argument("--reviewer")
    parser.add_argument("--graphics", default="auto",
                        choices=("auto", "tgp", "sixel", "halfcell", "unicode", "none"))
    parser.add_argument("--goto", metavar="FOV_UID", help="open this field of view")
    parser.add_argument("--probe", action="store_true",
                        help="report terminal capabilities and exit")
    return parser


def main(argv=None) -> int:
    # Forced, not defaulted, and before anything imports matplotlib. Panels are
    # rasterised to PNG and drawn as terminal cells; any interactive backend
    # here either opens a window over the interface or fails to import, which
    # is what a matplotlibrc naming a custom backend would otherwise cause.
    os.environ["MPLBACKEND"] = "Agg"
    args = build_parser().parse_args(argv)

    from pygor.tui.capabilities import probe

    # Must happen before Textual starts: its input thread eats the reply.
    caps = probe(args.graphics)
    if args.probe:
        print(caps.describe())
        return 0

    try:
        from pygor.tui.app import ProofreadApp
    except ImportError as error:
        raise SystemExit(f"{INSTALL_HINT}\n({error})") from error

    module = args.binding or DATASETS.get(args.dataset)
    if module is None:
        raise SystemExit(f"unknown dataset {args.dataset!r}; have {sorted(DATASETS)}")
    binding = importlib.import_module(module)

    from pygor.review.session import ReviewSession

    session = ReviewSession(binding, reviewer=args.reviewer)
    app = ProofreadApp(session, binding, caps, goto=args.goto)
    # Restore the terminal on every exit path, not only the clean one, and
    # again at interpreter exit in case something after us leaves it dirty.
    import atexit

    from pygor.tui.imaging import clear_terminal_images, restore_terminal

    atexit.register(restore_terminal)
    try:
        app.run()
    finally:
        clear_terminal_images()
        restore_terminal()
    return 0


if __name__ == "__main__":
    sys.exit(main())
