#!/usr/bin/env python3
"""Verify the terminal 2026-08-27 handoff contract without hardware access."""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = ROOT / "integrity" / "FINAL_HANDOFF_MANIFEST.tsv"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def fail(message: str) -> int:
    print(f"FINAL_HANDOFF_ERROR: {message}", file=sys.stderr)
    return 1


def verify_manifest(path: Path) -> str | None:
    try:
        with path.open(newline="", encoding="utf-8") as stream:
            rows = list(csv.DictReader(stream, delimiter="	"))
    except (OSError, csv.Error) as exc:
        return f"cannot read manifest: {exc}"
    if not rows:
        return "manifest is empty"
    for row in rows:
        relative = row.get("path", "")
        expected_hash = row.get("sha256", "")
        expected_size = row.get("size_bytes", "")
        candidate = Path(relative)
        if not relative or candidate.is_absolute() or ".." in candidate.parts:
            return f"unsafe manifest path: {relative!r}"
        target = ROOT / candidate
        if not target.is_file():
            return f"missing file: {relative}"
        if str(target.stat().st_size) != expected_size:
            return f"size mismatch: {relative}"
        if sha256(target) != expected_hash:
            return f"sha256 mismatch: {relative}"
    return None


def main(argv: list[str]) -> int:
    manifest = Path(argv[1]) if len(argv) == 2 else DEFAULT_MANIFEST
    if len(argv) > 2:
        return fail("usage: verify_final_handoff.py [manifest.tsv]")
    error = verify_manifest(manifest)
    if error is not None:
        return fail(error)
    status = json.loads((ROOT / "sim_to_real/final_results/FINAL_STATUS.json").read_text())
    physical = status["physical"]
    if physical["production_shadow"] != "hold" or physical["cycles_completed"] != 0:
        return fail("physical terminal state is not HOLD/0-cycle")
    if physical["policy_motor_writes"] != 0 or physical["actuation_performed"] is not False:
        return fail("physical write/actuation invariant failed")
    if any(value != 0 for value in physical["torque_enable_final"].values()):
        return fail("final torque state is not all zero")
    receipt = json.loads((ROOT / "sim_to_real/evidence/shadow/terminal_receipt.json").read_text())
    if receipt["terminal_state"] != "HOLD" or receipt["cycles_completed"] != 0:
        return fail("terminal receipt drift")
    if receipt["motor_writes_performed"] is not False or receipt["actuation_performed"] is not False:
        return fail("terminal receipt write invariant failed")
    evidence = json.loads((ROOT / "sim_to_real/evidence/EVIDENCE_INDEX.json").read_text())
    if evidence["terminal_state"] != "HOLD" or evidence["physical_motor_writes"] != 0:
        return fail("evidence index drift")
    private_keys = list(ROOT.rglob("owner-signing-private-key.pem"))
    if private_keys:
        return fail("private signing key is present")
    print("FINAL_HANDOFF_OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
