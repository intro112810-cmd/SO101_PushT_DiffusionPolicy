"""Build the compact lineage document consumed by physical shadow inference."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import cast

from .read_only_authority_types import canonical_authority_bytes

PRODUCTION_ARTIFACT_ID = "local-dp_cnn-recovered-v4-seed0"


def build_compact_lineage(source: Path, output: Path) -> dict[str, object]:
    """Reduce a verified full lineage receipt without changing its authority identity."""
    raw: object = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise TypeError("source lineage receipt must be a mapping")
    typed = cast("dict[str, object]", raw)
    artifact_id = typed.get("artifact_id")
    authority_digest = typed.get("authority_digest")
    if artifact_id != PRODUCTION_ARTIFACT_ID or typed.get("valid") is not True:
        raise ValueError("source lineage receipt is not the frozen production artifact")
    if (
        not isinstance(authority_digest, str)
        or len(authority_digest) != 64
        or any(character not in "0123456789abcdef" for character in authority_digest)
    ):
        raise ValueError("source lineage authority digest is invalid")
    compact: dict[str, object] = {
        "artifact_id": artifact_id,
        "authority_digest": authority_digest,
        "valid": True,
    }
    compact["lineage_digest"] = hashlib.sha256(canonical_authority_bytes(compact)).hexdigest()
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        existing: object = json.loads(output.read_text(encoding="utf-8"))
        if existing != compact:
            raise ValueError("compact lineage output identity drift")
    else:
        output.write_bytes(canonical_authority_bytes(compact))
    return compact
