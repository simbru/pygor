"""Fixtures, plus the headless setup the suite needs to run unattended.

The backend and Qt settings are applied at import time, before any test module
pulls in pygor, because both matplotlib and napari decide what to do at import.
"""

import os

os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pathlib

import matplotlib

matplotlib.use("Agg", force=True)

import matplotlib.pyplot as plt
import pytest

import pygor.load
from pygor.test.helpers import DEMO_DATA


def pytest_collection_modifyitems(config, items):
    """Skip the demo-data tests rather than erroring when the file is absent.

    The recording is downloaded separately (see the README), so a fresh clone
    has everything but that, and the unit tests should still run there.
    """
    if DEMO_DATA.exists():
        return
    skip = pytest.mark.skip(reason=f"demo data not found at {DEMO_DATA}")
    for item in items:
        if "demo_data" in item.keywords:
            item.add_marker(skip)


@pytest.fixture(autouse=True)
def close_figures():
    """Discard figures between tests so the plotting tests cannot exhaust memory."""
    yield
    plt.close("all")


@pytest.fixture(scope="session")
def demo_path() -> pathlib.Path:
    return DEMO_DATA


@pytest.fixture(scope="session")
def core(demo_path):
    """Core object loaded once. Use `fresh_core` if the test mutates it."""
    return pygor.load.Core(demo_path)


@pytest.fixture(scope="session")
def strf(demo_path):
    """STRF object loaded once. Use `fresh_strf` if the test mutates it."""
    return pygor.load.STRF(demo_path)


@pytest.fixture(scope="module")
def scratch_core(demo_path):
    """Core object the method smoke test may mutate freely.

    Calling every method leaves the object in an arbitrary state, so it must
    not be the one the property tests read.
    """
    return pygor.load.Core(demo_path)


@pytest.fixture(scope="module")
def scratch_strf(demo_path):
    """STRF object the method smoke test may mutate freely."""
    return pygor.load.STRF(demo_path)


@pytest.fixture
def fresh_core(demo_path):
    return pygor.load.Core(demo_path)


@pytest.fixture
def fresh_strf(demo_path):
    return pygor.load.STRF(demo_path)
