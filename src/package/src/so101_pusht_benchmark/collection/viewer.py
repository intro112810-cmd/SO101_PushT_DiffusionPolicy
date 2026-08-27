"""Headless-safe display of the RGB frames already produced for collection."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
import os
import time
import tkinter as tk
from typing import Protocol

import numpy as np
from numpy.typing import NDArray


class _MonotonicClock(Protocol):
    def monotonic(self) -> float: ...


class RealtimePacer:
    """Schedule collection ticks at a fixed rate without changing frame timestamps."""

    def __init__(
        self,
        clock: _MonotonicClock,
        fps: int = 10,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        if fps <= 0:
            raise ValueError("fps must be positive")
        self._clock = clock
        self._period = 1.0 / fps
        self._sleep = sleep
        self._next_tick: float | None = None

    def wait(self) -> None:
        """Wait until the next tick, dropping missed deadlines instead of catching up."""
        now = self._clock.monotonic()
        if self._next_tick is None:
            self._next_tick = now
        delay = self._next_tick - now
        if delay > 0:
            self._sleep(delay)
            now = self._clock.monotonic()
        self._next_tick = max(self._next_tick + self._period, now + self._period)


@dataclass
class OverlayState:
    """Operator-visible status drawn onto the topdown control pane."""

    measured_ee: tuple[float, float] | None = None
    requested_target: tuple[float, float] | None = None
    z_level: float | None = None
    state: str | None = None


class LiveViewer:
    """Show a supplied RGB frame without performing a second MuJoCo render.

    All windows share one Tk root (created by the first ``open`` call); each
    viewer window is a ``tk.Toplevel`` on that root.  A single root keeps the
    default-root binding valid for every ``tk.PhotoImage`` and avoids the
    EGL/Tk conflict that a second ``tk.Tk()`` triggered.  Any display setup
    failure leaves the viewer offscreen-only; collection continues with the
    original renderer and recorder unchanged.
    """

    _shared_root: tk.Tk | None = None

    def __init__(
        self,
        root: tk.Tk | None = None,
        window: tk.Toplevel | tk.Tk | None = None,
        label: tk.Label | None = None,
        on_close: Callable[[], None] | None = None,
    ) -> None:
        self._root = root
        self._window = window
        self._label = label
        self._on_close = on_close
        self._image: tk.PhotoImage | None = None
        self.last_frame: NDArray[np.uint8] | None = None

    @property
    def enabled(self) -> bool:
        return self._root is not None

    @property
    def root(self) -> tk.Tk | None:
        return self._root

    @classmethod
    def open(
        cls,
        enabled: bool | None = None,
        on_close: Callable[[], None] | None = None,
        title: str = "SO-101 PushT collection (recorded RGB)",
    ) -> LiveViewer:
        if enabled is None:
            enabled = bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))
        if not enabled:
            return cls(on_close=on_close)
        try:
            if cls._shared_root is None:
                # First window: use the Tk root itself so the control window
                # is never hidden by a withdrawn default root.
                root = tk.Tk()
                cls._shared_root = root
                root.title(title)
                label = tk.Label(root)
                label.pack()
                window: tk.Toplevel | tk.Tk = root
            else:
                window = tk.Toplevel(cls._shared_root)
                window.title(title)
                label = tk.Label(window)
                label.pack()
            viewer = cls(cls._shared_root, window, label, on_close)
            window.protocol("WM_DELETE_WINDOW", viewer.close)
            window.bind("<FocusOut>", viewer.on_focus_out)
            window.update_idletasks()
            window.update()
            # Center the window on the primary screen; without this the WM may
            # place it off-screen on multi-monitor setups.
            sw = cls._shared_root.winfo_screenwidth()
            sh = cls._shared_root.winfo_screenheight()
            w = window.winfo_width()
            h = window.winfo_height()
            x = max(0, (sw - w) // 2)
            y = max(0, (sh - h) // 2)
            window.geometry(f"{w}x{h}+{x}+{y}")
            window.update_idletasks()
            window.update()
        except (RuntimeError, tk.TclError):
            return cls(on_close=on_close)
        else:
            return viewer

    def show(
        self,
        frame: NDArray[np.uint8],
        overlay: OverlayState | None = None,
        bounds_x: tuple[float, float] = (0.15, 0.45),
        bounds_y: tuple[float, float] = (-0.15, 0.15),
    ) -> None:
        """Display a single supplied frame without modifying the source array.

        Schema-1 (v1) callers pass one frame, retained by identity with no
        copy, exactly as before. Schema-3 uses two viewers on the shared root:
        the control window receives the paper-style topdown plus its overlay,
        the observer window receives the 3D front frame. Zoom scales with the
        frame height so the window fits the screen at any input resolution.
        """
        composed: NDArray[np.uint8] = frame
        if overlay is not None:
            # Overlays touch only this display copy, never the source.
            display_copy = np.array(frame, dtype=np.uint8, copy=True)
            self._draw_overlay(display_copy, overlay, bounds_x, bounds_y)
            composed = display_copy

        self.last_frame = composed
        if self._root is None or self._label is None or self._window is None:
            return
        try:
            height, width, channels = composed.shape
            if composed.dtype != np.uint8 or channels != 3:
                raise ValueError("live viewport requires uint8 RGB frames")
            # Scale so the window is ~768px tall; zoom must stay >= 1.
            zoom = max(1, min(8, 768 // max(1, height)))
            ppm = f"P6 {width} {height} 255\n".encode() + composed.tobytes()
            image = tk.PhotoImage(data=ppm, format="PPM")
            display_image = image.zoom(zoom, zoom)
            self._label.configure(image=display_image)
            self._image = display_image
            # Release the initial fixed geometry so the packer can grow the
            # window to fit the larger label image.
            self._window.geometry("")
            self._root.update_idletasks()
            self._root.update()
        except tk.TclError:
            self.close()

    def _draw_overlay(
        self,
        composed: NDArray[np.uint8],
        overlay: OverlayState,
        bounds_x: tuple[float, float],
        bounds_y: tuple[float, float],
    ) -> None:
        """Draw control status onto the composed buffer (display-only copy)."""
        import cv2
        from so101_pusht_benchmark.control.mouse_mapping import task_xy_to_pixels

        h, w = composed.shape[:2]
        scale = max(1.0, h / 96.0)

        if overlay.measured_ee is not None:
            try:
                px, py = task_xy_to_pixels(
                    overlay.measured_ee[0], overlay.measured_ee[1], w, h, bounds_x, bounds_y
                )
                cv2.circle(composed, (int(px), int(py)), max(1, int(3 * scale)), (255, 0, 0), -1)
            except ValueError:
                pass

        if overlay.requested_target is not None:
            try:
                px, py = task_xy_to_pixels(
                    overlay.requested_target[0],
                    overlay.requested_target[1],
                    w,
                    h,
                    bounds_x,
                    bounds_y,
                )
                cv2.drawMarker(
                    composed,
                    (int(px), int(py)),
                    (255, 0, 0),  # canonical render_action cross color
                    cv2.MARKER_CROSS,
                    max(2, int(6 * scale)),
                    1,
                )
            except ValueError:
                pass

        if overlay.z_level is not None:
            cv2.putText(
                composed,
                f"Z: {overlay.z_level:.3f}",
                (int(5 * scale), h - int(5 * scale)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.3 * scale,
                (255, 255, 255),
                max(1, int(scale)),
            )

        if overlay.state is not None:
            cv2.putText(
                composed,
                overlay.state,
                (int(5 * scale), int(15 * scale)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.3 * scale,
                (255, 255, 0),
                max(1, int(scale)),
            )

    def render(self) -> NDArray[np.uint8]:
        """Return the composed frame for offscreen rendering."""
        if self.last_frame is None:
            return np.zeros((96, 192, 3), dtype=np.uint8)
        return self.last_frame.astype(np.uint8, copy=True)

    def on_focus_out(self, _event: object = None) -> None:
        """Notify the owner of focus loss without destroying the window.

        The control window is the Tk root; destroying it on a transient focus
        change would kill the whole application (and any observer window), so
        focus loss only reports the event upward.
        """
        if self._on_close is not None:
            self._on_close()
            self._on_close = None

    def close(self, _event: object = None) -> None:
        window, self._window = self._window, None
        self._label = None
        self._image = None
        if self._on_close is not None:
            self._on_close()
            self._on_close = None
        if window is not None:
            try:
                window.destroy()
            except tk.TclError:
                return


__all__ = ["LiveViewer", "OverlayState", "RealtimePacer"]
