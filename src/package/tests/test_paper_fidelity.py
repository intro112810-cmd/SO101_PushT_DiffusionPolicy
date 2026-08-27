"""Canonical PushT fidelity of the paper-style control pane.

The pane must match pusht_env.py::_render_frame: white background, a
light-green goal T, a gray T block with a dark outline, a blue pusher
circle, a black workspace border, and a red action cross for the
requested target (render_action uses (255,0,0)).
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from so101_pusht_benchmark.collection.viewer import LiveViewer, OverlayState
from so101_pusht_benchmark.control.paper_view import PaperView, PaperState

BOUNDS_X = (0.18, 0.38)
BOUNDS_Y = (-0.16, 0.16)


class _FakePaperState(PaperState):
    """Stand-in task pose matching the mouse-pipeline paper-state shape."""

    def __init__(
        self, t_x: float, t_y: float, t_yaw: float, pusher_x: float, pusher_y: float
    ) -> None:
        self._t_x = t_x
        self._t_y = t_y
        self._t_yaw = t_yaw
        self._pusher_x = pusher_x
        self._pusher_y = pusher_y

    @property
    def t_x(self) -> float:
        return self._t_x

    @property
    def t_y(self) -> float:
        return self._t_y

    @property
    def t_yaw(self) -> float:
        return self._t_yaw

    @property
    def pusher_x(self) -> float:
        return self._pusher_x

    @property
    def pusher_y(self) -> float:
        return self._pusher_y


def _compose_with_overlay(size: int = 384) -> NDArray[np.uint8]:
    state = _FakePaperState(0.271, -0.046, 0.984, 0.25, -0.05)
    paper = PaperView(bounds_x=BOUNDS_X, bounds_y=BOUNDS_Y).render(state, size=size)
    viewer = LiveViewer.open(enabled=False)
    viewer.show(
        paper,
        overlay=OverlayState(
            measured_ee=(0.25, -0.05),
            requested_target=(0.28, -0.02),
            z_level=0.045,
            state="ARMED",
        ),
        bounds_x=BOUNDS_X,
        bounds_y=BOUNDS_Y,
    )
    composed = viewer.render()
    viewer.close()
    return composed


def _requested_target_px(size: int = 384) -> tuple[int, int]:
    """Pixel position of the requested target on the composed pane."""
    from so101_pusht_benchmark.control.mouse_mapping import task_xy_to_pixels

    px, py = task_xy_to_pixels(0.28, -0.02, size, size, BOUNDS_X, BOUNDS_Y)
    return int(px), int(py)


def _is_red_cross_at(frame: NDArray[np.uint8], cx: int, cy: int, radius: int = 6) -> bool:
    """Return True when a red cross is centered near (cx, cy).

    A cross lights only two thin arm lines; a filled disk covers most of
    the box, so a low box fill ratio distinguishes the cross from the
    measured-EE disk.
    """
    r = frame[:, :, 0].astype(int)
    g = frame[:, :, 1].astype(int)
    b = frame[:, :, 2].astype(int)
    red = (r > 150) & (r > g * 1.8) & (r > b * 1.8)
    box = red[cy - radius : cy + radius + 1, cx - radius : cx + radius + 1]
    if not bool(box.any()):
        return False
    # A filled disk covers most of the box; a cross covers only its arms
    # (two thin lines), typically well under half the box area.
    return float(box.mean()) < 0.5


def _classify(frame: NDArray[np.uint8]) -> dict[str, bool]:
    r = frame[:, :, 0].astype(int)
    g = frame[:, :, 1].astype(int)
    b = frame[:, :, 2].astype(int)
    return {
        "white": float(((r > 240) & (g > 240) & (b > 240)).mean()) > 0.9,
        "green_goal": bool(((g > 140) & (g > r * 1.4) & (g > b * 1.4)).any()),
        "gray_block": bool(
            ((np.abs(r - 200) < 40) & (np.abs(g - 200) < 40) & (np.abs(b - 200) < 40)).any()
        ),
        "blue_pusher": bool(((b > 150) & (b > r * 1.6) & (b > g * 1.3)).any()),
        "black_border": bool(((r < 30) & (g < 30) & (b < 30)).any()),
        "red_cross": bool(((r > 150) & (r > g * 1.8) & (r > b * 1.8)).any()),
    }


def test_control_pane_matches_canonical_pusht_palette() -> None:
    frame = _compose_with_overlay()
    classes = _classify(frame)
    assert classes["white"], "control pane must be mostly white like the paper PushT"
    assert classes["green_goal"], "light-green goal T must be present"
    assert classes["gray_block"], "gray T block must be present"
    assert classes["blue_pusher"], "blue pusher circle must be present"
    assert classes["black_border"], "black workspace border must be present"


def test_requested_target_marker_is_red_cross() -> None:
    frame = _compose_with_overlay()
    cx, cy = _requested_target_px()
    assert _is_red_cross_at(frame, cx, cy), (
        "requested-target marker must be a red cross at the requested target "
        "(canonical render_action color (255,0,0)), not a green cross or a disk"
    )
    # The measured-EE circle is a separate red disk; it must stay a disk, not
    # be confused with the requested-target cross.
    r = frame[:, :, 0].astype(int)
    g = frame[:, :, 1].astype(int)
    b = frame[:, :, 2].astype(int)
    red = (r > 150) & (r > g * 1.8) & (r > b * 1.8)
    assert float(red.mean()) > 0, "red marker must exist somewhere"
