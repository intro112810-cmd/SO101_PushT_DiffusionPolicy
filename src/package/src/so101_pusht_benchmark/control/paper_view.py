"""Canonical PushT-style 2D paper view of the SO-101 push task.

The canonical Stanford PushT renders a flat white canvas with a light-green
goal T, a gray T block, and a blue agent circle (see
``pusht_env.py::_render_frame`` and pymunk's debug draw).  This module draws
the same flat style from the MuJoCo state so the topdown control pane and
the raw policy image match the paper look: white background, light-green
goal outline, gray T block with outline, blue pusher circle, and black
workspace border.  All drawing is vectorized with numpy (no per-pixel
Python loops) so a 384px pane renders in a few milliseconds.
"""

from __future__ import annotations

from typing import Protocol

import numpy as np
from numpy.typing import NDArray

U8 = NDArray[np.uint8]
RGB = tuple[float, float, float]

CANVAS = 512
WHITE: RGB = (255, 255, 255)
GOAL_GREEN: RGB = (144, 238, 144)  # pygame 'LightGreen'
BLOCK_FILL: RGB = (200, 200, 200)  # pymunk light gray fill
BLOCK_EDGE: RGB = (60, 60, 60)
AGENT_BLUE: RGB = (70, 130, 200)
AGENT_CORE: RGB = (150, 190, 240)
BORDER: RGB = (0, 0, 0)

# T block geometry (pixels on the 512 canvas), sized to the canonical PushT
# proportions (bar ~60px, stem ~20px, agent radius ~15px on a 512 window).
BAR_HALF = 30.0
STEM_HALF = 10.0
BAR_THICK_HALF = 8.0
STEM_BOTTOM = 26.0  # stem bottom edge below the bar center, in pixels
AGENT_RADIUS = 15.0


class PaperState(Protocol):
    """Pose sources the paper view needs: T pose and pusher XY."""

    @property
    def t_x(self) -> float: ...
    @property
    def t_y(self) -> float: ...
    @property
    def t_yaw(self) -> float: ...
    @property
    def pusher_x(self) -> float: ...
    @property
    def pusher_y(self) -> float: ...


class PaperView:
    """Draw one canonical PushT-style frame from task-space poses.

    The canvas is 512x512 px mapping ``bounds`` to pixel space with the
    canonical PushT y-down convention (task +y maps to screen -y).
    ``render(size=96)`` returns the downscaled uint8 frame.
    """

    def __init__(
        self,
        bounds_x: tuple[float, float] = (0.18, 0.38),
        bounds_y: tuple[float, float] = (-0.16, 0.16),
    ) -> None:
        self.bounds_x = bounds_x
        self.bounds_y = bounds_y
        self.scale = CANVAS / max(bounds_x[1] - bounds_x[0], bounds_y[1] - bounds_y[0])

    def _to_px(self, x: float, y: float) -> tuple[float, float]:
        px = (x - self.bounds_x[0]) * self.scale
        py = (y - self.bounds_y[0]) * self.scale
        return px, CANVAS - py

    def _tee_vertices(self, cx: float, cy: float, yaw: float) -> NDArray[np.float64]:
        """T corners (N,2) in canvas pixels, rotated by yaw around its center."""
        c = np.cos(yaw)
        s = np.sin(yaw)
        corners = np.array(
            [
                (-BAR_HALF, -BAR_THICK_HALF),
                (BAR_HALF, -BAR_THICK_HALF),
                (BAR_HALF, BAR_THICK_HALF),
                (STEM_HALF, BAR_THICK_HALF),
                (STEM_HALF, STEM_BOTTOM),
                (-STEM_HALF, STEM_BOTTOM),
                (-STEM_HALF, BAR_THICK_HALF),
                (-BAR_HALF, BAR_THICK_HALF),
            ],
            dtype=np.float64,
        )
        rot = np.array([[c, -s], [s, c]], dtype=np.float64)
        px0, py0 = self._to_px(cx, cy)
        return corners @ rot.T + np.array([px0, py0], dtype=np.float64)  # type: ignore[reportUnknownMemberType, reportUnknownVariableType]

    @staticmethod
    def _fill_polygon(
        canvas: U8, vertices: NDArray[np.float64], color: RGB, outline: RGB | None = None
    ) -> None:
        """Fill a simple polygon with the given color using a scanline test."""
        ys, xs = vertices[:, 1], vertices[:, 0]
        min_x, max_x = max(0, int(np.floor(xs.min()))), min(CANVAS - 1, int(np.ceil(xs.max())))
        min_y, max_y = max(0, int(np.floor(ys.min()))), min(CANVAS - 1, int(np.ceil(ys.max())))
        if min_x > max_x or min_y > max_y:
            return
        # Grid of pixel centers inside the polygon bounding box.
        yy, xx = np.mgrid[min_y : max_y + 1, min_x : max_x + 1]
        px = xx.astype(np.float64) + 0.5
        py = yy.astype(np.float64) + 0.5
        inside = np.zeros(xx.shape, dtype=bool)
        n = len(vertices)
        with np.errstate(divide="ignore", invalid="ignore"):
            for i in range(n):
                x1, y1 = vertices[i]
                x2, y2 = vertices[(i + 1) % n]
                cond = (y1 > py) != (y2 > py)
                dy = y2 - y1
                x_cross = np.where(dy != 0, (x2 - x1) * (py - y1) / dy + x1, np.inf)  # type: ignore[reportUnknownMemberType]
                inside ^= cond & (px < x_cross)
        canvas[min_y : max_y + 1, min_x : max_x + 1][inside] = np.asarray(color, dtype=np.uint8)
        if outline is not None:
            closed = np.vstack([vertices, vertices[0]])  # type: ignore[reportUnknownMemberType]
            for i in range(len(closed) - 1):
                PaperView._line(canvas, closed[i], closed[i + 1], outline)

    @staticmethod
    def _line(canvas: U8, a: NDArray[np.float64], b: NDArray[np.float64], color: RGB) -> None:
        x0, y0 = a
        x1, y1 = b
        steps = max(int(abs(x1 - x0)), int(abs(y1 - y0)), 1)
        ts = np.linspace(0, 1, steps + 1)
        xs = np.round(x0 + (x1 - x0) * ts).astype(int)  # type: ignore[reportUnknownMemberType]
        ys = np.round(y0 + (y1 - y0) * ts).astype(int)  # type: ignore[reportUnknownMemberType]
        valid = (xs >= 0) & (xs < CANVAS) & (ys >= 0) & (ys < CANVAS)  # type: ignore[reportUnknownVariableType]
        canvas[ys[valid], xs[valid]] = np.asarray(color, dtype=np.uint8)

    @staticmethod
    def _disc(canvas: U8, cx: int, cy: int, radius: int, color: RGB) -> None:
        yy, xx = np.ogrid[0:CANVAS, 0:CANVAS]
        mask = (xx - cx) ** 2 + (yy - cy) ** 2 <= radius * radius
        canvas[mask] = np.asarray(color, dtype=np.uint8)

    def render(self, state: PaperState, size: int = 96) -> U8:
        canvas = np.full((CANVAS, CANVAS, 3), WHITE, dtype=np.uint8)

        # Light-green goal T at the fixed target pose (target_t body in the
        # overlay: x=0.34, y=0, no yaw).
        goal = self._tee_vertices(0.34, 0.0, 0.0)
        self._fill_polygon(canvas, goal, GOAL_GREEN)

        # Gray T block with dark outline at the live block pose.
        block = self._tee_vertices(state.t_x, state.t_y, state.t_yaw)
        self._fill_polygon(canvas, block, BLOCK_FILL, BLOCK_EDGE)

        # Blue agent circle (pusher) with a lighter core, like pymunk's draw.
        px, py = self._to_px(state.pusher_x, state.pusher_y)
        self._disc(canvas, int(px), int(py), int(AGENT_RADIUS), AGENT_BLUE)
        self._disc(canvas, int(px), int(py), max(1, int(AGENT_RADIUS) - 4), AGENT_CORE)

        if size == CANVAS:
            self._draw_border(canvas, BORDER)
            return canvas
        # Downscale the full canvas; cv2 keeps it vectorized (a per-pixel
        # Python loop here measured ~340ms, far over the 10Hz tick budget).
        import cv2

        interp = cv2.INTER_AREA if size < 128 else cv2.INTER_LINEAR
        out = cv2.resize(canvas, (size, size), interpolation=interp)
        out = np.asarray(out, dtype=np.uint8)
        # Display panes get the border redrawn at target resolution: a 2px
        # line downscaled to 384 dilutes to ~192 gray, losing the canonical
        # black workspace outline.
        if size >= 128:
            self._draw_border(out, BORDER)
        return out

    @staticmethod
    def _draw_border(canvas: U8, color: RGB) -> None:
        n = canvas.shape[0]
        edge = np.array([[1, 1], [n - 2, 1], [n - 2, n - 2], [1, n - 2]])
        closed = np.vstack([edge, edge[0]])  # type: ignore[reportUnknownMemberType]
        for i in range(len(closed) - 1):
            PaperView._line(canvas, closed[i], closed[i + 1], color)

    @staticmethod
    def _downscale(canvas: U8, size: int) -> U8:
        """Area-average downscale of the full canvas to ``size`` x ``size``."""
        import cv2

        interp = cv2.INTER_AREA if size < 128 else cv2.INTER_LINEAR
        return np.asarray(cv2.resize(canvas, (size, size), interpolation=interp), dtype=np.uint8)


__all__ = ["PaperView"]
