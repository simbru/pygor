"""``python -m pygor.review <root>`` -- scan a dataset and report what is there.

A separate module so the CLI does not run as a re-import of ``index``, which
runpy warns about when a package's ``__init__`` has already imported it.
"""

from pygor.review.index import _main

if __name__ == "__main__":
    raise SystemExit(_main())
