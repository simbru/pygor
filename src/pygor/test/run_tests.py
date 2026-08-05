#!/usr/bin/env python3
"""Run the pygor test suite.

Kept as an entry point because the README and run_test.bat point at it. The
configuration (test paths, per-test timeout, markers) lives in pyproject.toml,
so this only forwards arguments to pytest:

    python src/pygor/test/run_tests.py                    # everything
    python src/pygor/test/run_tests.py -k bootstrap       # one topic
    python src/pygor/test/run_tests.py -m "not slow"      # skip the slow ones
"""

import pathlib
import sys

PYGOR_ROOT = pathlib.Path(__file__).parents[3]


def main(argv) -> int:
    try:
        import pytest
    except ImportError:
        print(
            "pytest is not installed. Install the dev dependencies with:\n"
            "  uv sync\n"
            "or, without uv:\n"
            "  pip install pytest pytest-timeout",
            file=sys.stderr,
        )
        return 1
    return pytest.main([*argv, "--rootdir", str(PYGOR_ROOT)])


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
