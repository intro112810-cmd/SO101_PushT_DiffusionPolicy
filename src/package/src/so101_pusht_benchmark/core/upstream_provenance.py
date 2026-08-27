"""Fail-closed verification for the runtime-consumed pushT-so100 checkout."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import TypedDict, cast


class UpstreamProvenanceError(RuntimeError):
    """Raised when the pushT-so100 checkout does not match its approved manifest."""


class UpstreamProvenanceReport(TypedDict):
    head: str
    remote: str
    environment_sha256: str
    runtime_manifest_sha256: str
    runtime_member_count: int
    approved_patches: list[str]
    excluded_untracked: list[str]


_RECEIPT_FIELDS: set[str] = {
    "pre-patch RED",
    "wrapper cannot faithfully solve",
    "smallest runtime-consumed source surface",
    "native schema/F710 unchanged",
    "updated manifest",
    "focused regression GREEN",
    "no unrelated upstream diff",
}
_MODEL_ROOT = "chernyadev mujoco_menagerie add-so-arm100 trs_so_arm100"
_REQUIRED_RUNTIME_MEMBERS = {
    "src/env_gym_ee.py": "python",
    "src/env_human_ee.py": "python",
    "src/helper.py": "python",
    f"{_MODEL_ROOT}/human_env.xml": "xml",
    f"{_MODEL_ROOT}/so_arm100_leader.xml": "xml",
    **{
        f"{_MODEL_ROOT}/assets/{name}.stl": "asset"
        for name in (
            "Base",
            "Base_Motor",
            "Fixed_Jaw",
            "Fixed_Jaw_Collision_1",
            "Fixed_Jaw_Collision_2",
            "Fixed_Jaw_Motor",
            "Lower_Arm",
            "Lower_Arm_Motor",
            "Moving_Jaw",
            "Moving_Jaw_Collision_1",
            "Moving_Jaw_Collision_2",
            "Moving_Jaw_Collision_3",
            "Rotation_Pitch",
            "Rotation_Pitch_Motor",
            "Upper_Arm",
            "Upper_Arm_Motor",
            "Wrist_Pitch_Roll",
            "Wrist_Pitch_Roll_Motor",
        )
    },
}
_REQUIRED_EXCLUSION = "MUJOCO_LOG.TXT"
_TRUSTED_GIT = "/usr/bin/git"
_EXECUTABLE_CACHE_SUFFIXES = {".pyc", ".pyo"}
_INTERPRETER_CACHE_NAME = re.compile(
    r"(?P<stem>.+)\.cpython-\d+(?:\.opt-\d+)?\.pyc"
)


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
    except OSError as exc:
        raise UpstreamProvenanceError(f"undeclared upstream drift: unreadable {path}") from exc
    return digest.hexdigest()


def _mapping(value: object, label: str) -> dict[str, object]:
    if not isinstance(value, dict):
        raise UpstreamProvenanceError(f"malformed upstream provenance manifest: {label}")
    return cast("dict[str, object]", value)


def _list(value: object, label: str) -> list[object]:
    if not isinstance(value, list):
        raise UpstreamProvenanceError(f"malformed upstream provenance manifest: {label}")
    return cast("list[object]", value)


def _string(value: object, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise UpstreamProvenanceError(f"malformed upstream provenance manifest: {label}")
    return value


def _digest(value: object, label: str) -> str:
    digest = _string(value, label)
    if len(digest) != 64 or any(char not in "0123456789abcdef" for char in digest):
        raise UpstreamProvenanceError(f"malformed upstream provenance manifest: {label}")
    return digest


def _safe_member(root: Path, value: object, label: str) -> tuple[str, Path]:
    name = _string(value, label)
    relative = Path(name)
    if relative.is_absolute() or ".." in relative.parts:
        raise UpstreamProvenanceError(f"malformed upstream provenance manifest: unsafe {label}")
    lexical = root / relative
    if lexical.is_symlink():
        raise UpstreamProvenanceError(f"undeclared upstream drift: symlink member {name}")
    try:
        resolved = lexical.resolve(strict=True)
        resolved.relative_to(root.resolve(strict=True))
    except (OSError, ValueError) as exc:
        raise UpstreamProvenanceError(f"undeclared upstream drift: missing member {name}") from exc
    if not resolved.is_file():
        raise UpstreamProvenanceError(f"undeclared upstream drift: non-file member {name}")
    return name, resolved


def _run_git(root: Path, *args: str) -> bytes:
    try:
        process = subprocess.run(
            [_TRUSTED_GIT, "-C", str(root), *args],
            check=False,
            capture_output=True,
            env={**os.environ, "GIT_OPTIONAL_LOCKS": "0", "GIT_LFS_SKIP_SMUDGE": "1"},
        )
    except OSError as exc:
        raise UpstreamProvenanceError(f"git verification failed: {exc}") from exc
    if process.returncode != 0:
        detail = process.stderr.decode(errors="replace").strip()
        raise UpstreamProvenanceError(f"git verification failed ({process.returncode}): {detail}")
    return process.stdout


def ignored_executable_artifacts(root: Path) -> tuple[str, ...]:
    """Return unsafe ignored bytecode artifacts under the verified checkout."""
    cache_prefix = sys.pycache_prefix
    external_cache_isolation = False
    if cache_prefix is not None:
        candidate = Path(cache_prefix)
        if candidate.is_absolute():
            try:
                candidate.resolve().relative_to(root.resolve())
            except ValueError:
                external_cache_isolation = True
    artifacts: list[str] = []
    for path in root.rglob("*"):
        relative = path.relative_to(root)
        is_executable_cache = (
            path.suffix in _EXECUTABLE_CACHE_SUFFIXES
            or "__pycache__" in relative.parts
        )
        match = _INTERPRETER_CACHE_NAME.fullmatch(path.name)
        sibling_source = (
            path.parent.parent / f"{match.group('stem')}.py"
            if match is not None and path.parent.name == "__pycache__"
            else None
        )
        is_safe_regular_cache = (
            external_cache_isolation
            and path.is_file()
            and not path.is_symlink()
            and sibling_source is not None
            and sibling_source.is_file()
            and not sibling_source.is_symlink()
        )
        is_unsafe_cache = not is_safe_regular_cache
        if is_executable_cache and is_unsafe_cache and (path.is_symlink() or path.is_file()):
            artifacts.append(relative.as_posix())
    return tuple(sorted(artifacts))


def _load_manifest(path: Path) -> tuple[dict[str, object], bytes]:
    try:
        raw = path.read_bytes()
        parsed: object = json.loads(raw)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise UpstreamProvenanceError(f"malformed upstream provenance manifest: {path}") from exc
    manifest = _mapping(parsed, "root")
    required = {
        "schema",
        "source",
        "environment",
        "runtime_members",
        "approved_patches",
        "excluded_untracked",
    }
    if manifest.get("schema") != 1 or set(manifest) != required:
        raise UpstreamProvenanceError("malformed upstream provenance manifest: schema or keys")
    return manifest, raw


def _nul_paths(value: bytes) -> set[str]:
    return {item.decode("utf-8") for item in value.split(b"\0") if item}


def verify_pusht_so100(
    manifest_path: str | Path,
    upstream_root: str | Path,
) -> UpstreamProvenanceReport:
    """Verify HEAD, origin, runtime bytes, approved patches, and all checkout drift."""
    manifest, raw_manifest = _load_manifest(Path(manifest_path))
    root = Path(upstream_root)
    if root.is_symlink() or not root.is_dir():
        raise UpstreamProvenanceError(f"undeclared upstream drift: unsafe root {root}")
    ignored_executables = ignored_executable_artifacts(root)
    if ignored_executables:
        raise UpstreamProvenanceError(
            f"ignored executable artifact: {', '.join(ignored_executables)}"
        )

    source = _mapping(manifest["source"], "source")
    if set(source) != {"name", "head", "remote"} or source.get("name") != "pushT-so100":
        raise UpstreamProvenanceError("malformed upstream provenance manifest: source")
    expected_head = _string(source["head"], "source.head")
    expected_remote = _string(source["remote"], "source.remote")
    head = _run_git(root, "rev-parse", "HEAD").decode().strip()
    remote = _run_git(root, "remote", "get-url", "origin").decode().strip()
    if head != expected_head:
        raise UpstreamProvenanceError(f"undeclared upstream drift: HEAD {head}")
    if remote != expected_remote:
        raise UpstreamProvenanceError(f"undeclared upstream drift: remote {remote}")

    runtime_members: dict[str, tuple[Path, str]] = {}
    raw_members = _list(manifest["runtime_members"], "runtime_members")
    for index, raw_member in enumerate(raw_members):
        member = _mapping(raw_member, f"runtime_members[{index}]")
        if set(member) != {"path", "kind", "sha256"}:
            raise UpstreamProvenanceError("malformed upstream provenance manifest: runtime member")
        name, path = _safe_member(root, member["path"], "runtime member path")
        kind = _string(member["kind"], f"runtime member kind {name}")
        digest = _digest(member["sha256"], f"runtime member digest {name}")
        if name in runtime_members:
            raise UpstreamProvenanceError(
                "malformed upstream provenance manifest: duplicate member"
            )
        runtime_members[name] = (path, digest)
        if _REQUIRED_RUNTIME_MEMBERS.get(name) != kind:
            raise UpstreamProvenanceError(
                f"malformed upstream provenance manifest: runtime kind {name}"
            )
    if list(runtime_members) != sorted(_REQUIRED_RUNTIME_MEMBERS) or set(runtime_members) != set(
        _REQUIRED_RUNTIME_MEMBERS
    ):
        raise UpstreamProvenanceError(
            "malformed upstream provenance manifest: incomplete runtime member closure"
        )

    environment = _mapping(manifest["environment"], "environment")
    if set(environment) != {"path", "sha256"}:
        raise UpstreamProvenanceError("malformed upstream provenance manifest: environment")
    environment_name, environment_path = _safe_member(root, environment["path"], "environment path")
    if environment_name != "environment.yml":
        raise UpstreamProvenanceError("malformed upstream provenance manifest: environment path")
    environment_digest = _digest(environment["sha256"], "environment digest")

    approved: dict[str, tuple[str, str]] = {}
    for index, raw_patch in enumerate(_list(manifest["approved_patches"], "approved_patches")):
        patch = _mapping(raw_patch, f"approved_patches[{index}]")
        if set(patch) != {"path", "base_sha256", "sha256", "rationale", "receipt"}:
            raise UpstreamProvenanceError("malformed upstream provenance manifest: patch keys")
        name = _string(patch["path"], "patch path")
        _string(patch["rationale"], f"patch rationale {name}")
        base_digest = _digest(patch["base_sha256"], f"patch base digest {name}")
        patched_digest = _digest(patch["sha256"], f"patch digest {name}")
        receipt = _mapping(patch["receipt"], f"patch receipt {name}")
        missing = _RECEIPT_FIELDS - set(receipt)
        if set(receipt) != _RECEIPT_FIELDS or any(
            not isinstance(value, str) or not value.strip() for value in receipt.values()
        ):
            raise UpstreamProvenanceError(
                f"required patch receipt fields missing for {name}: {sorted(missing)}"
            )
        if name in approved or name not in runtime_members:
            raise UpstreamProvenanceError("malformed upstream provenance manifest: patch path")
        approved[name] = (base_digest, patched_digest)

    raw_exclusions = _list(manifest["excluded_untracked"], "excluded_untracked")
    exclusions: set[str] = set()
    for index, raw_exclusion in enumerate(raw_exclusions):
        exclusion = _mapping(raw_exclusion, f"excluded_untracked[{index}]")
        if set(exclusion) != {"path", "reason"}:
            raise UpstreamProvenanceError("malformed upstream provenance manifest: exclusion")
        name = _string(exclusion["path"], "exclusion path")
        _string(exclusion["reason"], f"exclusion reason {name}")
        exclusions.add(name)
    if exclusions != {_REQUIRED_EXCLUSION}:
        raise UpstreamProvenanceError("malformed upstream provenance manifest: exclusions")

    tracked_drift = _nul_paths(_run_git(root, "diff", "HEAD", "--name-only", "-z", "--"))
    if tracked_drift != set(approved):
        raise UpstreamProvenanceError(
            f"undeclared upstream drift: tracked paths {sorted(tracked_drift - set(approved))}"
        )
    untracked = _nul_paths(_run_git(root, "ls-files", "--others", "--exclude-standard", "-z"))
    unexpected_untracked = untracked - exclusions
    if unexpected_untracked:
        raise UpstreamProvenanceError(
            f"undeclared upstream drift: untracked paths {sorted(unexpected_untracked)}"
        )

    for name, (path, expected_digest) in runtime_members.items():
        actual_digest = _sha256(path)
        if actual_digest != expected_digest:
            raise UpstreamProvenanceError(f"undeclared upstream drift: digest mismatch for {name}")
        head_digest = _sha256_bytes(_run_git(root, "show", f"HEAD:{name}"))
        if name in approved:
            base_digest, patched_digest = approved[name]
            if head_digest != base_digest or actual_digest != patched_digest:
                raise UpstreamProvenanceError(
                    f"undeclared upstream drift: approved patch bytes for {name}"
                )
        elif head_digest != expected_digest:
            raise UpstreamProvenanceError(f"undeclared upstream drift: HEAD bytes for {name}")

    if _sha256(environment_path) != environment_digest:
        raise UpstreamProvenanceError("undeclared upstream drift: environment digest mismatch")
    if _sha256_bytes(_run_git(root, "show", f"HEAD:{environment_name}")) != environment_digest:
        raise UpstreamProvenanceError("undeclared upstream drift: environment HEAD bytes")

    return {
        "head": head,
        "remote": remote,
        "environment_sha256": environment_digest,
        "runtime_manifest_sha256": _sha256_bytes(raw_manifest),
        "runtime_member_count": len(runtime_members),
        "approved_patches": sorted(approved),
        "excluded_untracked": sorted(untracked & exclusions),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="verify-pusht-so100")
    package_root = Path(__file__).resolve().parents[3]
    project_root = package_root.parents[1]
    parser.add_argument(
        "--manifest",
        type=Path,
        default=package_root / "configs/provenance/pusht_so100_upstream.json",
    )
    parser.add_argument(
        "--upstream-root",
        type=Path,
        default=project_root / "05_references/external_repos/pushT-so100",
    )
    args = parser.parse_args(argv)
    try:
        report = verify_pusht_so100(args.manifest, args.upstream_root)
    except UpstreamProvenanceError as exc:
        print(f"FAIL CLOSED: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(report, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
