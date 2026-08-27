"""Fail-closed verification of benchmark inputs and runtime boundaries."""

from __future__ import annotations

import os
import posixpath
import subprocess
import tarfile
from pathlib import Path
from typing import TypedDict, cast

import yaml


from .provenance_helpers import (
    ProvenanceError,
    TrustedPathAliases,
    is_inside,
    manifest,
    mapping,
    path as _path,
    sha256 as _sha256,
    source_path,
    strings,
    validate_archive_members,
)


class _Robomimic(TypedDict):
    archive_sha256: str


class ProvenanceReport(TypedDict):
    revisions: dict[str, str]
    trees: dict[str, str]
    robomimic: _Robomimic
    locks: dict[str, object]


def _git(root: Path, *args: str) -> str:
    try:
        return subprocess.check_output(
            ["git", "-C", str(root), *args],
            text=True,
            stderr=subprocess.STDOUT,
            env={**os.environ, "GIT_OPTIONAL_LOCKS": "0", "GIT_LFS_SKIP_SMUDGE": "1"},
        ).strip()
    except (OSError, subprocess.CalledProcessError) as exc:
        raise ProvenanceError(f"git verification failed for {root}: {exc}") from exc


def verify_clean_detached(root: Path) -> None:
    branch = subprocess.run(
        ["git", "-C", str(root), "symbolic-ref", "-q", "--short", "HEAD"],
        text=True,
        capture_output=True,
        check=False,
    )
    if branch.returncode != 1 or branch.stdout.strip():
        raise ProvenanceError(f"checkout is not detached: {root}")
    status = subprocess.run(
        ["git", "-C", str(root), "diff", "--quiet", "--", ".", ":(exclude)tests/artifacts/**"],
        text=True,
        capture_output=True,
        check=False,
        env={**os.environ, "GIT_OPTIONAL_LOCKS": "0", "GIT_LFS_SKIP_SMUDGE": "1"},
    )
    if status.returncode != 0:
        raise ProvenanceError(f"dirty checkout: {root}")
    if _git(root, "ls-files", "--others", "--exclude-standard"):
        raise ProvenanceError(f"untracked checkout content: {root}")


def _verify_source(
    name: str,
    spec: dict[str, object],
    data: dict[str, object],
    root: Path,
    trusted_aliases: TrustedPathAliases,
) -> None:
    if spec.get("mode") != "detached_git":
        raise ProvenanceError(f"{name}: source requires detached_git mode")
    source = source_path(str(spec["path"]), root, trusted_aliases=trusted_aliases)
    if not source.is_dir() or source.is_symlink():
        raise ProvenanceError(f"missing or unsafe source: {source}")
    if any(item.stat().st_mode & 0o222 for item in source.rglob("*") if item.exists()):
        raise ProvenanceError(f"writable source: {source}")
    verify_clean_detached(source)
    revisions = cast("dict[str, str]", data["revisions"])
    trees = cast("dict[str, str]", data["trees"])
    revision = revisions[name]
    tree = trees[name]
    if (
        _git(source, "rev-parse", "HEAD") != revision
        or _git(source, "rev-parse", "HEAD^{tree}") != tree
    ):
        raise ProvenanceError(f"{name}: revision/tree mismatch")
    origin = posixpath.normpath(_git(source, "remote", "get-url", "origin"))
    allowed = posixpath.normpath(str(spec["allowed_origin"]))
    if origin != allowed:
        raise ProvenanceError(f"{name}: unapproved origin {origin}")
    try:
        evidence_commit = (
            _path(str(spec["commit_evidence"]), root, trusted_aliases=trusted_aliases)
            .read_text()
            .strip()
        )
        evidence_tree = (
            _path(str(spec["tree_evidence"]), root, trusted_aliases=trusted_aliases)
            .read_text()
            .strip()
        )
        expected = _path(str(spec["manifest"]), root, trusted_aliases=trusted_aliases).read_text()
    except OSError as exc:
        raise ProvenanceError(f"{name}: missing provenance evidence") from exc
    if evidence_commit != revision or evidence_tree != tree:
        raise ProvenanceError(f"{name}: stale commit/tree evidence")
    if manifest(source) != expected:
        raise ProvenanceError(f"{name}: file manifest mismatch")


def _verify_archive(
    spec: dict[str, object], root: Path, trusted_aliases: TrustedPathAliases
) -> None:
    if spec.get("mode") != "archive_and_materialized_manifest":
        raise ProvenanceError("robomimic requires archive_and_materialized_manifest mode")
    archive = _path(str(spec["archive"]), root, trusted_aliases=trusted_aliases)
    extracted = _path(str(spec["path"]), root, trusted_aliases=trusted_aliases)
    if _sha256(archive) != str(spec["archive_sha256"]):
        raise ProvenanceError("robomimic archive hash mismatch")
    if not extracted.is_dir() or (extracted / ".git").exists():
        raise ProvenanceError("unsafe robomimic extraction")
    expected = _path(str(spec["manifest"]), root, trusted_aliases=trusted_aliases).read_text()
    archive_manifest = _path(
        str(spec["archive_manifest"]), root, trusted_aliases=trusted_aliases
    ).read_text()
    if manifest(extracted) != expected or expected != archive_manifest:
        raise ProvenanceError("robomimic archive/extracted manifest mismatch")
    with tarfile.open(archive, "r:gz") as stream:
        prefix = f"robomimic-{spec['commit']}/"
        members = [
            {
                "name": member.name,
                "type": "dir" if member.isdir() else "file",
                "linkname": member.linkname,
            }
            for member in stream.getmembers()
        ]
        validate_archive_members(members, prefix)


def _load(config: Path) -> dict[str, object]:
    try:
        value: object = cast("object", yaml.safe_load(config.read_text(encoding="utf-8")))
    except (OSError, yaml.YAMLError) as exc:
        raise ProvenanceError(f"invalid provenance config: {config}") from exc
    if not isinstance(value, dict):
        raise ProvenanceError("provenance config must be a mapping")
    parsed = cast("dict[str, object]", value)
    required = {"project_root", "sources", "robomimic", "locks", "revisions", "trees"}
    if parsed.get("schema") != 1 or not required.issubset(parsed):
        raise ProvenanceError("unsupported or incomplete provenance schema")
    mapping(parsed["sources"], "sources")
    mapping(parsed["locks"], "locks")
    return parsed


def _trusted_path_aliases(data: dict[str, object], root: Path) -> TrustedPathAliases:
    aliases = mapping(data.get("trusted_path_aliases", {}), "trusted_path_aliases")
    result: list[tuple[Path, Path]] = []
    for relative, raw_target in aliases.items():
        relative_path = Path(relative)
        if (
            not isinstance(raw_target, str)
            or not Path(raw_target).is_absolute()
            or relative_path.is_absolute()
            or len(relative_path.parts) != 1
            or relative_path.parts[0] in {"", ".", ".."}
        ):
            raise ProvenanceError("invalid trusted path alias declaration")
        lexical = root / relative_path
        target = Path(raw_target).absolute()
        if not lexical.is_symlink() or lexical.resolve() != target or not target.is_dir():
            raise ProvenanceError(f"trusted path alias topology mismatch: {relative}")
        result.append((lexical, target))
    return tuple(result)


def load_and_verify(config: str | Path, project_root: str | Path) -> ProvenanceReport:
    """Load and verify all pinned sources, evidence, archives, and locks."""
    try:
        config_path = Path(config)
        if not config_path.is_file():
            raise ProvenanceError(f"missing config: {config_path}")
        root = Path(project_root).resolve()
        data = _load(config_path)
        trusted_aliases = _trusted_path_aliases(data, root)
        if _path(str(data["project_root"]), root) != root:
            raise ProvenanceError("declared project root mismatch")
        for name, raw_spec in mapping(data["sources"], "sources").items():
            spec = mapping(raw_spec, f"source:{name}")
            _verify_source(name, spec, data, root, trusted_aliases)
        _verify_archive(mapping(data["robomimic"], "robomimic"), root, trusted_aliases)
        for raw_lock in mapping(data["locks"], "locks").values():
            lock = mapping(raw_lock, "lock")
            path_value = lock.get("path")
            hash_value = lock.get("sha256")
            if not isinstance(path_value, str) or not isinstance(hash_value, str):
                raise ProvenanceError("invalid runtime lock spec")
            path = _path(path_value, root, trusted_aliases=trusted_aliases)
            if not path.is_file() or _sha256(path) != hash_value:
                raise ProvenanceError(f"runtime lock mismatch: {path}")
        return cast(
            "ProvenanceReport",
            cast(
                "object",
                {
                    "revisions": data["revisions"],
                    "trees": data["trees"],
                    "robomimic": cast("_Robomimic", data["robomimic"]),
                    "locks": data["locks"],
                },
            ),
        )
    except ProvenanceError:
        raise
    except (KeyError, TypeError, ValueError, OSError, tarfile.TarError) as exc:
        raise ProvenanceError(f"invalid provenance data: {exc}") from exc


def verify_import_origins(
    runtime: str, imported_origins: dict[str, str | Path], config: dict[str, object]
) -> None:
    """Reject forbidden dependencies and imports outside approved roots."""
    try:
        imports = mapping(config["imports"], "imports")
        origins = mapping(imports["allowed_origins"], "allowed_origins")
        forbidden = strings(imports.get(f"{runtime}_forbidden", []), "forbidden")
        root = Path(str(config["project_root"])).resolve()
        trusted_aliases = _trusted_path_aliases(config, root)
        allowed = [
            _path(value, root, trusted_aliases=trusted_aliases)
            for value in strings(origins["project"], "project")
        ]
        allowed += [
            _path(value, root, trusted_aliases=trusted_aliases)
            for value in strings(origins["upstream"], "upstream")
        ]
        for module, origin in imported_origins.items():
            if any(module == name or module.startswith(name + ".") for name in forbidden):
                raise ProvenanceError(f"{runtime}: forbidden import {module}")
            if not any(is_inside(Path(origin).resolve(), allowed_root) for allowed_root in allowed):
                raise ProvenanceError(f"{runtime}: unapproved import origin {module}: {origin}")
    except ProvenanceError:
        raise
    except (KeyError, TypeError, ValueError) as exc:
        raise ProvenanceError(f"invalid import policy: {exc}") from exc


def main(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(prog="show-provenance")
    parser.add_argument("--config")
    parser.add_argument("--project-root")
    args = parser.parse_args(argv)
    package_root = Path(__file__).resolve().parents[3]
    default_root = package_root.parent.parent
    config = (
        Path(args.config)
        if args.config
        else package_root / "configs/provenance/external_inputs.yaml"
    )
    project_root = Path(args.project_root) if args.project_root else default_root
    try:
        report = load_and_verify(config, project_root)
    except ProvenanceError as exc:
        print(f"FAIL CLOSED: {exc}")
        return 1
    for group, values in report.items():
        print(f"{group}:")
        for key, value in cast("dict[str, object]", values).items():
            print(f"  {key}: {value}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
