from dataclasses import dataclass
import tkinter as tk
from typing import Any, Protocol

from ..control.mouse_mapping import pixels_to_task_xy


class TkLike(Protocol):
    """Structural surface of the Tk root that MouseKeyboardSource depends on.

    bind/protocol stay Any-typed because tkinter's stub declares them with
    overloads that no structural Protocol declaration can match; the read
    surface (winfo/update/destroy) remains fully typed.
    """

    def bind(self, sequence: str, func: Any, add: Any = None) -> Any: ...
    def protocol(self, name: str, func: Any) -> Any: ...
    def update(self) -> None: ...
    def update_idletasks(self) -> None: ...
    def winfo_width(self) -> int: ...
    def winfo_height(self) -> int: ...
    def destroy(self) -> None: ...


@dataclass(frozen=True, slots=True)
class InputSample:
    target: tuple[float, float, float] | None
    deadman: bool
    success: bool = False
    stop: bool = False
    rerecord: bool = False
    connected: bool = True
    fresh: bool = True


class MouseKeyboardSource:
    def __init__(
        self,
        root: TkLike,
        bounds_x: tuple[float, float],
        bounds_y: tuple[float, float],
    ) -> None:
        self.root = root
        self.bounds_x = bounds_x
        self.bounds_y = bounds_y

        self.contact_z = 0.045
        self.clearance_z = 0.065
        self.current_z = self.clearance_z

        self.deadman = False
        self.connected = True
        self.has_focus = True

        self.success_cmd = False
        self.stop_cmd = False
        self.rerecord_cmd = False
        self._pressed_keys: set[str] = set()

        self.mouse_x = 0.0
        self.mouse_y = 0.0

        self.root.bind("<ButtonPress-1>", self._on_press)
        self.root.bind("<ButtonRelease-1>", self._on_release)
        self.root.bind("<Motion>", self._on_motion)
        self.root.bind("<KeyPress>", self._on_key_press)
        self.root.bind("<KeyRelease>", self._on_key_release)
        self.root.bind("<FocusOut>", self._on_focus_out)
        self.root.bind("<FocusIn>", self._on_focus_in)
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

    def _on_press(self, event: tk.Event) -> None:
        self.deadman = True
        self._update_mouse(event)

    def _on_release(self, event: tk.Event) -> None:
        self.deadman = False
        self._update_mouse(event)

    def _on_motion(self, event: tk.Event) -> None:
        self._update_mouse(event)

    def _update_mouse(self, event: tk.Event) -> None:
        self.mouse_x = float(event.x)
        self.mouse_y = float(event.y)

    def _on_key_press(self, event: tk.Event) -> None:
        keysym = str(event.keysym).lower()
        if keysym in self._pressed_keys:
            return
        self._pressed_keys.add(keysym)

        if keysym == "c":
            self.current_z = self.contact_z
        elif keysym == "v":
            self.current_z = self.clearance_z
        elif keysym == "enter":
            self.success_cmd = True
        elif keysym == "escape":
            self.stop_cmd = True
        elif keysym == "backspace":
            self.rerecord_cmd = True

    def _on_key_release(self, _event: tk.Event) -> None:
        keysym = str(_event.keysym).lower()
        if keysym in self._pressed_keys:
            self._pressed_keys.remove(keysym)

    def _on_focus_out(self, _event: tk.Event) -> None:
        self.has_focus = False
        self.deadman = False

    def _on_focus_in(self, _event: tk.Event) -> None:
        self.has_focus = True

    def _on_close(self) -> None:
        self.connected = False
        try:
            self.root.destroy()
        except Exception:  # noqa: S110  teardown is best-effort
            pass

    def close(self) -> None:
        self._on_close()

    def poll(self) -> InputSample:
        try:
            self.root.update_idletasks()
            self.root.update()
        except Exception:
            self.connected = False

        success = self.success_cmd
        stop = self.stop_cmd
        rerecord = self.rerecord_cmd
        self.success_cmd = False
        self.stop_cmd = False
        self.rerecord_cmd = False

        if not self.connected or not self.has_focus:
            return InputSample(None, False, success, stop, rerecord, self.connected)

        try:
            pane_width = self.root.winfo_width()
            pane_height = self.root.winfo_height()
        except Exception:
            self.connected = False
            return InputSample(None, False, success, stop, rerecord, False)

        target = None
        if self.deadman:
            try:
                req = pixels_to_task_xy(
                    self.mouse_x,
                    self.mouse_y,
                    float(pane_width),
                    float(pane_height),
                    self.bounds_x,
                    self.bounds_y,
                )
                if not req.outside:
                    target = (req.x, req.y, self.current_z)
            except Exception:  # noqa: S110  outside-bounds pixels yield no target
                pass

        return InputSample(target, self.deadman, success, stop, rerecord, self.connected)
