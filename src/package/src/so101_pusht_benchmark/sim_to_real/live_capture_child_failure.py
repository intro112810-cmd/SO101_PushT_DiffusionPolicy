"""Private file-backed child crash journal independent of command IPC."""

from __future__ import annotations

import json
from pathlib import Path
from typing import TypeAlias, cast

from .live_capture_protocol import ProviderFailed, ProviderRole

__all__ = ("read_child_failure", "record_child_failure")
JsonValue: TypeAlias = bool | int | float | str | list["JsonValue"] | dict[str, "JsonValue"] | None


def record_child_failure(path: Path, event: ProviderFailed) -> None:
    """Atomically preserve the first child error before attempting live IPC."""
    if path.exists():
        return
    payload = {
        "role": event.role.value,
        "phase": event.phase,
        "sample_index": event.sample_index,
        "observed_at": event.observed_at,
        "error_type": event.error_type,
        "error_message": event.error_message,
        "traceback_text": event.traceback_text,
    }
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def _journal_error(role: ProviderRole, detail: str) -> ProviderFailed:
    return ProviderFailed(
        role,
        "child_process",
        None,
        0.0,
        "ChildFailureJournalError",
        detail,
        "",
    )


def read_child_failure(path: Path, role: ProviderRole) -> ProviderFailed | None:
    """Parse a private journal without allowing malformed bytes to block reap evidence."""
    try:
        raw: JsonValue = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return None
    except (OSError, json.JSONDecodeError) as exc:
        return _journal_error(role, str(exc))
    if not isinstance(raw, dict):
        return _journal_error(role, "child failure journal is not a mapping")
    document = cast("dict[str, JsonValue]", raw)
    phase = document.get("phase")
    observed_at = document.get("observed_at")
    error_type = document.get("error_type")
    error_message = document.get("error_message")
    traceback_text = document.get("traceback_text")
    sample_index = document.get("sample_index")
    if (
        document.get("role") != role.value
        or not isinstance(phase, str)
        or not isinstance(observed_at, int | float)
        or not isinstance(error_type, str)
        or not isinstance(error_message, str)
        or not isinstance(traceback_text, str)
        or (sample_index is not None and not isinstance(sample_index, int))
    ):
        return _journal_error(role, "child failure journal fields are invalid")
    return ProviderFailed(
        role,
        phase,
        sample_index,
        float(observed_at),
        error_type,
        error_message,
        traceback_text,
    )
