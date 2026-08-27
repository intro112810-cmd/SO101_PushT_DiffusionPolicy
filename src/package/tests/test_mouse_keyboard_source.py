from __future__ import annotations

from collections.abc import Callable

from so101_pusht_benchmark.input.mouse_keyboard import MouseKeyboardSource


class MockEvent:
    def __init__(self, x: int = 0, y: int = 0, keysym: str = "") -> None:
        self.x = x
        self.y = y
        self.keysym = keysym


class MockTk:
    def __init__(self) -> None:
        self.events: list[object] = []
        self.bindings: dict[str, Callable[..., object]] = {}
        self.focus = True
        self.w = 800
        self.h = 600

    def bind(
        self,
        sequence: str,
        func: Callable[..., object],
        add: object = None,  # noqa: ARG002
    ) -> None:
        self.bindings[sequence] = func

    def protocol(self, name: str, func: Callable[..., object]) -> None:
        self.bindings[name] = func

    def fire(self, sequence: str, *args: object) -> None:
        handler = self.bindings[sequence]
        assert handler is not None, f"no handler bound for {sequence}"
        handler(*args)

    def update(self) -> None:
        pass

    def update_idletasks(self) -> None:
        pass

    def winfo_width(self) -> int:
        return self.w

    def winfo_height(self) -> int:
        return self.h

    def destroy(self) -> None:
        pass


def test_init_and_neutral_to_ready() -> None:
    root = MockTk()
    source = MouseKeyboardSource(root, (0.18, 0.38), (-0.16, 0.16))

    # Neutral/ready state without deadman
    sample = source.poll()
    assert sample.deadman is False
    assert sample.connected is True
    assert sample.target is None


def test_mouse_hold_arms_deadman() -> None:
    root = MockTk()
    source = MouseKeyboardSource(root, (0.18, 0.38), (-0.16, 0.16))

    # Simulate press
    root.fire("<ButtonPress-1>", MockEvent(x=400, y=300))
    sample = source.poll()
    assert sample.deadman is True
    assert sample.target is not None


def test_deadman_release() -> None:
    root = MockTk()
    source = MouseKeyboardSource(root, (0.18, 0.38), (-0.16, 0.16))

    root.fire("<ButtonPress-1>", MockEvent(x=400, y=300))
    sample = source.poll()
    assert sample.deadman is True

    root.fire("<ButtonRelease-1>", MockEvent(x=400, y=300))
    sample = source.poll()
    assert sample.deadman is False
    assert sample.target is None


def test_z_contact_clearance_transitions() -> None:
    root = MockTk()
    source = MouseKeyboardSource(root, (0.18, 0.38), (-0.16, 0.16))
    root.fire("<ButtonPress-1>", MockEvent(x=400, y=300))

    sample = source.poll()
    assert sample.target is not None
    assert sample.target[2] == 0.065

    root.fire("<KeyPress>", MockEvent(keysym="c"))
    sample = source.poll()
    assert sample.target is not None
    assert sample.target[2] == 0.045

    root.fire("<KeyPress>", MockEvent(keysym="v"))
    sample = source.poll()
    assert sample.target is not None
    assert sample.target[2] == 0.065


def test_repeated_key_debounce() -> None:
    root = MockTk()
    source = MouseKeyboardSource(root, (0.18, 0.38), (-0.16, 0.16))

    root.fire("<KeyPress>", MockEvent(keysym="enter"))
    sample = source.poll()
    assert sample.success is True

    # Should not fire again on hold (bounce)
    root.fire("<KeyPress>", MockEvent(keysym="enter"))
    sample = source.poll()
    assert sample.success is False

    root.fire("<KeyRelease>", MockEvent(keysym="enter"))
    root.fire("<KeyPress>", MockEvent(keysym="enter"))
    sample = source.poll()
    assert sample.success is True


def test_focus_loss_fails_closed() -> None:
    root = MockTk()
    source = MouseKeyboardSource(root, (0.18, 0.38), (-0.16, 0.16))

    root.fire("<ButtonPress-1>", MockEvent(x=400, y=300))
    sample = source.poll()
    assert sample.deadman is True

    root.fire("<FocusOut>", MockEvent())
    sample = source.poll()
    assert sample.deadman is False
    assert sample.target is None


def test_disconnect_fails_closed() -> None:
    root = MockTk()
    source = MouseKeyboardSource(root, (0.18, 0.38), (-0.16, 0.16))

    root.fire("<ButtonPress-1>", MockEvent(x=400, y=300))
    root.bindings["WM_DELETE_WINDOW"]()

    sample = source.poll()
    assert sample.connected is False
    assert sample.target is None
