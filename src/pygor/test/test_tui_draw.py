"""Tests for hand-drawn ROIs: the mask rules, the picture, and the screen.

The mask rules are pure array functions and tested directly. The screen is
driven through Textual's pilot with the keyboard, which is exact to the pixel;
the mouse path shares ``pixel_under`` with the inspector, tested there.
"""

from __future__ import annotations

import asyncio

import numpy as np
import pytest

from pygor.test.test_review_index import write_recording


def empty(shape=(10, 10)):
    return np.ones(shape, dtype=np.int16)


class TestShapes:
    def test_a_disk_of_radius_zero_is_one_pixel(self):
        from pygor.tui.roi_edit import disk

        region = disk((9, 9), (4, 4), 0)
        assert region.sum() == 1 and region[4, 4]

    def test_a_disk_is_round_and_centred(self):
        from pygor.tui.roi_edit import disk

        region = disk((9, 9), (4, 4), 2)
        assert region.sum() == 13  # the 5x5 square minus its 12 corner-ish pixels
        assert region[4, 2] and region[4, 6] and region[2, 4] and region[6, 4]
        assert not region[2, 2]

    def test_a_disk_at_the_edge_is_clipped_not_wrapped(self):
        from pygor.tui.roi_edit import disk

        region = disk((9, 9), (0, 0), 2)
        assert not region[8, 8] and not region[0, 8]
        assert region[0, 0]

    def test_a_polygon_keeps_its_outline(self):
        """A tight lasso must not come back smaller than the line drawn."""
        from pygor.tui.roi_edit import polygon

        region = polygon((10, 10), [(2, 2), (2, 6), (6, 6), (6, 2)])
        assert region[2, 2] and region[6, 6] and region[2, 6] and region[6, 2]
        assert region[4, 4]
        assert region.sum() == 25

    def test_two_vertices_are_a_line_not_nothing(self):
        from pygor.tui.roi_edit import polygon

        assert polygon((10, 10), [(1, 1), (1, 5)]).sum() == 5


class TestMaskRules:
    def test_add_takes_the_next_id_down(self):
        from pygor.tui.roi_edit import add, disk

        rois = empty()
        rois[0, 0] = -1
        rois[0, 1] = -3
        added = add(rois, disk(rois.shape, (5, 5), 1))
        assert added.roi_id == -4
        assert (added.rois == -4).sum() == 5

    def test_existing_rois_keep_their_pixels(self):
        from pygor.tui.roi_edit import add, disk

        rois = empty()
        rois[5, 5] = -1
        added = add(rois, disk(rois.shape, (5, 5), 1))
        assert added.rois[5, 5] == -1
        assert added.taken == 4
        assert added.blocked == 1

    def test_a_shape_entirely_on_other_rois_adds_nothing(self):
        from pygor.tui.roi_edit import add, disk

        rois = empty()
        rois[4:7, 4:7] = -1
        added = add(rois, disk(rois.shape, (5, 5), 1))
        assert added.roi_id is None
        assert added.rois is rois

    def test_add_does_not_modify_its_input(self):
        """Undo keeps the previous mask by reference."""
        from pygor.tui.roi_edit import add, disk

        rois = empty()
        add(rois, disk(rois.shape, (5, 5), 1))
        assert (rois == 1).all()

    def test_remove_restores_the_masks_own_background(self):
        from pygor.tui.roi_edit import remove

        rois = np.zeros((4, 4), dtype=np.int16)
        rois[1, 1] = -2
        out = remove(rois, -2)
        assert (out == 0).all()

    def test_compact_closes_gaps_and_keeps_order(self):
        from pygor.tui.roi_edit import compact

        rois = empty((1, 4))
        rois[0] = [-1, -3, 1, -7]
        assert compact(rois).tolist() == [[-1, -2, 1, -3]]


class TestEditPreview:
    def picture(self, **kwargs):
        import io

        from PIL import Image

        from pygor.tui.imaging import edit_preview

        rois = kwargs.pop("rois", empty((8, 8)))
        panel = edit_preview(np.zeros((8, 8)), (0.0, 1.0), rois, 80, 80, **kwargs)
        return np.asarray(Image.open(io.BytesIO(panel.png)).convert("RGB"))

    def inks(self, picture):
        return {tuple(px) for px in picture.reshape(-1, 3)}

    def test_one_block_per_pixel(self):
        assert self.picture().shape[:2] == (80, 80)

    def test_rois_are_outlined(self):
        from pygor.tui.imaging import OUTLINE_INK

        rois = empty((8, 8))
        rois[2:5, 2:5] = -1
        assert OUTLINE_INK in self.inks(self.picture(rois=rois))

    def test_blocked_pixels_are_shown_apart_from_free_ones(self):
        rois = empty((8, 8))
        rois[4, 4] = -1
        pending = np.zeros((8, 8), dtype=bool)
        pending[4, 3:6] = True
        picture = self.picture(rois=rois, pending=pending)
        # Row 4 of the array is row 3 of the picture, counting from the top.
        free = picture[3 * 10 + 5, 3 * 10 + 5]
        blocked = picture[3 * 10 + 5, 4 * 10 + 5]
        assert free[1] > free[0]
        assert blocked[0] > blocked[1]


def run_app(app, steps):
    async def body():
        async with app.run_test(size=(160, 45)) as pilot:
            await pilot.pause()
            await steps(pilot)

    asyncio.run(body())


async def settle(pilot, predicate, limit=200):
    for _ in range(limit):
        await pilot.pause()
        if predicate():
            return
        await asyncio.sleep(0.05)


@pytest.fixture
def recording_path(tmp_path):
    """A saved recording with 60 frames of noise and no ROIs.

    Trigger frames too, which trace extraction reads to place its baseline
    window and the shared fixture does not carry.
    """
    pytest.importorskip("textual")
    import h5py

    path = write_recording(tmp_path, stem="0_0_SWN_200")
    with h5py.File(path, "r+") as handle:
        group = handle["recording_000"]
        del group["images"]
        group.create_dataset(
            "images", data=np.random.default_rng(0).random((60, 8, 8)).astype(np.float32))
        if "triggertimes_frame" not in group:
            group.create_dataset("triggertimes_frame", data=np.arange(10, 60, 10))
    return path


def draw_app(path):
    from pygor.tui.capabilities import Capabilities
    from pygor.tui.standalone import build_app

    caps = Capabilities(mode="halfcell", cell_width=10, cell_height=20,
                        is_tty=True, tmux=False, term="xterm-256color")
    return build_app(path, caps)


async def live(pilot, screen):
    """The image widget as mounted now; see ``live_widget`` in test_tui_browse."""
    for _ in range(5):
        await pilot.pause()
    await settle(pilot, lambda: screen.image_widget() is not None)
    return screen.image_widget()


async def open_draw(pilot, app):
    from pygor.tui.browse_screen import InspectScreen
    from pygor.tui.draw_screen import DrawScreen

    await settle(pilot, lambda: isinstance(app.screen, InspectScreen)
                 and app.screen.recording is not None)
    await pilot.press("d")
    await settle(pilot, lambda: isinstance(app.screen, DrawScreen))
    return app.screen


class TestDrawScreen:
    def test_d_opens_it_at_the_inspectors_probe(self, recording_path):
        app = draw_app(recording_path)
        seen = {}

        async def steps(pilot):
            screen = await open_draw(pilot, app)
            seen["cursor"] = screen.cursor

        run_app(app, steps)
        assert seen["cursor"] == (4, 4)

    def test_enter_adds_a_disk_and_undo_takes_it_back(self, recording_path):
        from pygor.tui.roi_edit import count

        app = draw_app(recording_path)
        seen = {}

        async def steps(pilot):
            screen = await open_draw(pilot, app)
            await pilot.press("minus")  # radius 1
            await pilot.press("enter")
            seen["after_add"] = count(screen.rois)
            seen["px"] = int((screen.rois < 0).sum())
            seen["on_disk_untouched"] = count(app.screen.recording.rois)
            await pilot.press("u")
            seen["after_undo"] = count(screen.rois)

        run_app(app, steps)
        assert seen["after_add"] == 1
        assert seen["px"] == 5
        assert seen["on_disk_untouched"] == 0
        assert seen["after_undo"] == 0

    def test_a_keyboard_lasso(self, recording_path):
        app = draw_app(recording_path)
        seen = {}

        async def steps(pilot):
            screen = await open_draw(pilot, app)
            await pilot.press("l")
            for key in ("space", "right", "right", "space", "up", "up", "space"):
                await pilot.press(key)
            seen["path"] = list(screen.path)
            await pilot.press("enter")
            seen["px"] = int((screen.rois < 0).sum())
            seen["path_after"] = list(screen.path)

        run_app(app, steps)
        assert seen["path"] == [(4, 4), (4, 6), (6, 6)]
        assert seen["px"] == 6  # the triangle and its outline
        assert seen["path_after"] == []

    def test_a_drag_is_a_lasso_stroke(self, recording_path):
        pytest.importorskip("textual_image")
        app = draw_app(recording_path)
        seen = {}

        async def steps(pilot):
            screen = await open_draw(pilot, app)
            await pilot.press("l")
            widget = await live(pilot, screen)
            w, h = widget.size.width, widget.size.height
            # Every redraw re-mounts the image widget, so it is looked up
            # again for each event rather than held: a stale one reports a
            # zero region and the pilot's event lands on the header.
            await pilot.mouse_down(await live(pilot, screen), offset=(1, 1))
            await pilot.hover(await live(pilot, screen), offset=(w // 2, 1))
            await pilot.hover(await live(pilot, screen), offset=(w // 2, h // 2))
            await pilot.mouse_up(await live(pilot, screen), offset=(w // 2, h // 2))
            await settle(pilot, lambda: screen._stroke is None)
            seen["path"] = list(screen.path)

        run_app(app, steps)
        assert len(seen["path"]) == 3
        # Top-left of the picture is the last row: origin="lower".
        assert seen["path"][0] == (7, 0)

    def test_clicks_add_vertices_one_at_a_time(self, recording_path):
        pytest.importorskip("textual_image")
        app = draw_app(recording_path)
        seen = {}

        async def steps(pilot):
            screen = await open_draw(pilot, app)
            await pilot.press("l")
            widget = await live(pilot, screen)
            w, h = widget.size.width, widget.size.height
            for offset in ((1, 1), (w - 2, 1), (w - 2, h - 2)):
                await pilot.click(await live(pilot, screen), offset=offset)
            seen["path"] = list(screen.path)

        run_app(app, steps)
        assert len(seen["path"]) == 3

    def test_x_deletes_the_roi_under_the_cursor(self, recording_path):
        from pygor.tui.roi_edit import count

        app = draw_app(recording_path)
        seen = {}

        async def steps(pilot):
            screen = await open_draw(pilot, app)
            await pilot.press("enter")
            await pilot.press("x")
            seen["count"] = count(screen.rois)

        run_app(app, steps)
        assert seen["count"] == 0

    def test_escape_with_edits_asks_before_discarding(self, recording_path):
        from pygor.tui.draw_screen import DrawScreen
        from pygor.tui.reprocess_screen import ConfirmSave

        app = draw_app(recording_path)
        seen = {}

        async def steps(pilot):
            await open_draw(pilot, app)
            await pilot.press("enter")
            await pilot.press("escape")
            await settle(pilot, lambda: isinstance(app.screen, ConfirmSave))
            seen["asked"] = isinstance(app.screen, ConfirmSave)
            await pilot.press("n")
            await settle(pilot, lambda: isinstance(app.screen, DrawScreen))
            seen["stayed"] = isinstance(app.screen, DrawScreen)

        run_app(app, steps)
        assert seen["asked"] and seen["stayed"]

    def test_save_writes_the_mask_and_traces(self, recording_path):
        import h5py

        from pygor.tui.browse_screen import InspectScreen

        app = draw_app(recording_path)
        seen = {}

        async def steps(pilot):
            screen = await open_draw(pilot, app)
            await pilot.press("enter")
            await pilot.press("left", "left", "left", "left")
            await pilot.press("minus", "minus", "enter")  # a single pixel at (4, 0)
            await pilot.press("S")
            await pilot.press("y")
            await settle(pilot, lambda: screen.saved)
            await pilot.press("escape")
            await settle(pilot, lambda: isinstance(app.screen, InspectScreen))
            seen["traces"] = app.screen.recording.traces_raw.shape

        run_app(app, steps)
        assert seen["traces"] == (2, 60)
        with h5py.File(recording_path, "r") as handle:
            rois = handle["recording_000"]["rois"][()]
        assert sorted(np.unique(rois).tolist()) == [-2, -1, 1]
        assert rois[4, 0] == -2
        assert recording_path.with_suffix(".h5.presegment").exists()
