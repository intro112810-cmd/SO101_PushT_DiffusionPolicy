from __future__ import annotations

from pathlib import Path
import shutil
import subprocess
import sys
from typing import cast

import pytest
import yaml

from so101_pusht_benchmark.core.provenance import (
    ProvenanceError,
    manifest,
    source_path,
    validate_archive_members,
    verify_clean_detached,
    load_and_verify,
    verify_import_origins,
)
from so101_pusht_benchmark.core.upstream_provenance import (
    UpstreamProvenanceError,
    ignored_executable_artifacts,
    verify_pusht_so100,
)


PACKAGE = Path(__file__).parents[1]
ROOT = PACKAGE.parents[1]
CONFIG = PACKAGE / "configs/provenance/external_inputs.yaml"
PUSHT_SO100_CONFIG = PACKAGE / "configs/provenance/pusht_so100_upstream.json"
PUSHT_SO100_ROOT = ROOT / "05_references/external_repos/pushT-so100"


def test_pusht_so100_current_patch_manifest_is_accepted() -> None:
    report = verify_pusht_so100(PUSHT_SO100_CONFIG, PUSHT_SO100_ROOT)
    assert report["head"] == "f4d6d1311bc0b43ce65458a9edd856f3c7e0a520"
    assert report["remote"] == "https://github.com/boaoqian/pushT-so100.git"
    assert report["environment_sha256"] == (
        "a7aab5a14bb18b6bb94cd1ecf13616384c6af87ba131ae5dc86fec7e94920f70"
    )
    assert report["runtime_member_count"] == 23
    assert report["approved_patches"] == ["src/env_human_ee.py", "src/helper.py"]
    assert report["excluded_untracked"] == ["MUJOCO_LOG.TXT"]


def test_upstream_provenance_ignores_operator_path_git_hijack(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    fake_git = fake_bin / "git"
    fake_git.write_text("#!/bin/sh\nprintf 'forged-by-path\\n'\n", encoding="utf-8")
    fake_git.chmod(0o755)
    monkeypatch.setenv("PATH", str(fake_bin))

    report = verify_pusht_so100(PUSHT_SO100_CONFIG, PUSHT_SO100_ROOT)
    assert report["head"] == "f4d6d1311bc0b43ce65458a9edd856f3c7e0a520"


def test_upstream_provenance_rejects_ignored_executable_bytecode(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    copied_checkout = tmp_path / "pushT-so100"
    cache_dir = copied_checkout / "src/__pycache__"
    cache_dir.mkdir(parents=True)
    (copied_checkout / "src/helper.py").write_text("value = 1\n", encoding="utf-8")
    (cache_dir / "helper.cpython-310.pyc").write_bytes(b"unverified executable")
    monkeypatch.setattr(sys, "pycache_prefix", None)

    assert ignored_executable_artifacts(copied_checkout) == (
        "src/__pycache__/helper.cpython-310.pyc",
    )


def test_upstream_provenance_accepts_regular_bytecode_with_external_cache_prefix(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    copied_checkout = tmp_path / "pushT-so100"
    cache_dir = copied_checkout / "src/__pycache__"
    cache_dir.mkdir(parents=True)
    (copied_checkout / "src/helper.py").write_text("value = 1\n", encoding="utf-8")
    (cache_dir / "helper.cpython-310.pyc").write_bytes(b"stale interpreter cache")
    external_cache = tmp_path / "external-python-cache"
    monkeypatch.setattr(sys, "pycache_prefix", str(external_cache))

    assert ignored_executable_artifacts(copied_checkout) == ()


def test_upstream_provenance_rejects_orphan_bytecode_with_external_cache_prefix(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    copied_checkout = tmp_path / "pushT-so100"
    (copied_checkout / "src").mkdir(parents=True)
    (copied_checkout / "src/orphan.pyc").write_bytes(b"unverified executable")
    monkeypatch.setattr(sys, "pycache_prefix", str(tmp_path / "external-python-cache"))

    assert ignored_executable_artifacts(copied_checkout) == ("src/orphan.pyc",)


def test_upstream_provenance_rejects_symlinked_ignored_bytecode_cache(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    copied_checkout = tmp_path / "pushT-so100"
    (copied_checkout / "src").mkdir(parents=True)
    outside = tmp_path / "outside-cache"
    outside.mkdir()
    (outside / "helper.cpython-310.pyc").write_bytes(b"unverified executable")
    (copied_checkout / "src/__pycache__").symlink_to(
        outside,
        target_is_directory=True,
    )
    monkeypatch.setattr(sys, "pycache_prefix", str(tmp_path / "external-python-cache"))

    assert ignored_executable_artifacts(copied_checkout) == ("src/__pycache__",)


def test_upstream_patch_manifest_mutated_member_fails_as_undeclared_drift(
    tmp_path: Path,
) -> None:
    copied_manifest = tmp_path / "pusht_so100_upstream.json"
    copied_manifest.write_text(
        PUSHT_SO100_CONFIG.read_text(encoding="utf-8").replace(
            "4b3dc360edfa1bfdc69eb25bb859541ce3f95638d316eb4792ef78a90c267cb3",
            "0" * 64,
            1,
        ),
        encoding="utf-8",
    )
    with pytest.raises(UpstreamProvenanceError, match="undeclared upstream drift"):
        verify_pusht_so100(copied_manifest, PUSHT_SO100_ROOT)


def test_upstream_patch_manifest_requires_complete_patch_receipt(tmp_path: Path) -> None:
    copied_manifest = tmp_path / "pusht_so100_upstream.json"
    copied_manifest.write_text(
        PUSHT_SO100_CONFIG.read_text(encoding="utf-8").replace(
            '"focused regression GREEN":', '"missing focused regression GREEN":', 1
        ),
        encoding="utf-8",
    )
    with pytest.raises(UpstreamProvenanceError, match="required patch receipt fields"):
        verify_pusht_so100(copied_manifest, PUSHT_SO100_ROOT)


def test_upstream_patch_manifest_rejects_unrelated_checkout_drift(tmp_path: Path) -> None:
    copied_checkout = tmp_path / "pushT-so100"
    shutil.copytree(PUSHT_SO100_ROOT, copied_checkout)
    verify_pusht_so100(PUSHT_SO100_CONFIG, copied_checkout)

    unrelated = copied_checkout / "README.md"
    unrelated.write_bytes(unrelated.read_bytes() + b"\nundeclared tracked drift\n")
    with pytest.raises(UpstreamProvenanceError, match="undeclared upstream drift"):
        verify_pusht_so100(PUSHT_SO100_CONFIG, copied_checkout)
    unrelated.write_bytes(
        subprocess.run(
            ["git", "-C", str(copied_checkout), "show", "HEAD:README.md"],
            check=True,
            capture_output=True,
        ).stdout
    )

    generated = copied_checkout / "unexpected-generated.bin"
    generated.write_bytes(b"undeclared generated artifact")
    with pytest.raises(UpstreamProvenanceError, match="undeclared upstream drift"):
        verify_pusht_so100(PUSHT_SO100_CONFIG, copied_checkout)


def test_upstream_patch_manifest_malformed_input_is_classified(tmp_path: Path) -> None:
    malformed = tmp_path / "pusht_so100_upstream.json"
    malformed.write_text('{"schema":', encoding="utf-8")
    with pytest.raises(UpstreamProvenanceError, match="malformed upstream provenance manifest"):
        verify_pusht_so100(malformed, PUSHT_SO100_ROOT)


def test_current_manifest_is_accepted() -> None:
    report = load_and_verify(CONFIG, ROOT)
    assert report["revisions"]["stanford"] == "5ba07ac6661db573af695b419a7947ecb704690f"
    assert report["revisions"]["lerobot"] == "e40b58a8dfa9e7b86918c374791599d070518d11"
    assert report["revisions"]["so101"] == "fda892cba81032c46c40976a48c9ceadbf40a9ca"
    assert report["robomimic"]["archive_sha256"].startswith("e8d0c9e9")


@pytest.mark.parametrize(
    ("needle", "replacement"),
    [
        ("5ba07ac6661db573af695b419a7947ecb704690f", "0" * 40),
        ("5a8bd7058d1dce44312e2e1a78949237884dab32", "1" * 64),
        ("e8d0c9e9edfa9564e6313898508c7aef878f98aeb909529b49e13639adaa8ee3", "2" * 64),
    ],
)
def test_revision_tree_and_archive_drift_fail_closed(
    tmp_path: Path, needle: str, replacement: str
) -> None:
    mutated = tmp_path / "provenance.yaml"
    mutated.write_text(CONFIG.read_text().replace(needle, replacement), encoding="utf-8")
    with pytest.raises(ProvenanceError):
        load_and_verify(mutated, ROOT)


def test_supported_modes_and_mode_specific_invariants_fail_closed(tmp_path: Path) -> None:
    wrong_source_mode = tmp_path / "wrong-source-mode.yaml"
    wrong_source_mode.write_text(
        CONFIG.read_text().replace(
            "mode: detached_git", "mode: archive_and_materialized_manifest", 1
        ),
        encoding="utf-8",
    )
    with pytest.raises(ProvenanceError):
        load_and_verify(wrong_source_mode, ROOT)
    wrong_archive_mode = tmp_path / "wrong-archive-mode.yaml"
    wrong_archive_mode.write_text(
        CONFIG.read_text().replace("mode: archive_and_materialized_manifest", "mode: detached_git"),
        encoding="utf-8",
    )
    with pytest.raises(ProvenanceError):
        load_and_verify(wrong_archive_mode, ROOT)
    unsupported = tmp_path / "unsupported-mode.yaml"
    unsupported.write_text(
        CONFIG.read_text().replace("mode: detached_git", "mode: unknown"), encoding="utf-8"
    )
    with pytest.raises(ProvenanceError):
        load_and_verify(unsupported, ROOT)


def test_allowed_origin_uses_exact_normalized_equality(tmp_path: Path) -> None:
    mutated = tmp_path / "near-prefix.yaml"
    mutated.write_text(
        CONFIG.read_text().replace(
            "file:///home/intro/InternLab/02_InTro_Project/04_experiments/so101_pusht_benchmark/cache/upstream/",
            "file:///home/intro/InternLab/02_InTro_Project/04_experiments/so101_pusht_benchmark/cache/upstream/stan",
            1,
        ),
        encoding="utf-8",
    )
    with pytest.raises(ProvenanceError):
        load_and_verify(mutated, ROOT)


def test_malformed_typed_config_is_always_provenance_error(tmp_path: Path) -> None:
    for text in ("schema: 1\n", "schema: wrong\n", "sources: []\n"):
        config = tmp_path / "invalid.yaml"
        config.write_text(text, encoding="utf-8")
        with pytest.raises(ProvenanceError):
            load_and_verify(config, ROOT)


def test_configured_source_root_symlink_fails_before_resolution(tmp_path: Path) -> None:
    real = tmp_path / "real"
    real.mkdir()
    link = tmp_path / "configured-source"
    link.symlink_to(real, target_is_directory=True)
    with pytest.raises(ProvenanceError):
        source_path("configured-source", tmp_path)


def test_declared_top_level_alias_allows_regular_read_only_source(tmp_path: Path) -> None:
    backing = tmp_path / "backing"
    source = backing / "cache/upstream/source"
    source.mkdir(parents=True)
    alias = tmp_path / "artifacts"
    alias.symlink_to(backing, target_is_directory=True)

    assert (
        source_path(
            "artifacts/cache/upstream/source",
            tmp_path,
            trusted_aliases=((alias, backing),),
        )
        == source
    )


def test_declared_alias_does_not_bless_nested_or_leaf_symlinks(tmp_path: Path) -> None:
    backing = tmp_path / "backing"
    upstream = backing / "cache/upstream"
    upstream.mkdir(parents=True)
    outside = tmp_path / "outside"
    outside.mkdir()
    (upstream / "source").symlink_to(outside, target_is_directory=True)
    alias = tmp_path / "artifacts"
    alias.symlink_to(backing, target_is_directory=True)

    with pytest.raises(ProvenanceError, match="source root is a symlink"):
        source_path(
            "artifacts/cache/upstream/source",
            tmp_path,
            trusted_aliases=((alias, backing),),
        )


def test_canonical_alias_declaration_target_drift_rejects(tmp_path: Path) -> None:
    mutated = tmp_path / "alias-drift.yaml"
    untrusted_target = tmp_path / "attacker-controlled-artifacts"
    mutated.write_text(
        CONFIG.read_text(encoding="utf-8").replace(
            "/data/df/02_InTro_Project/04_experiments",
            str(untrusted_target),
            1,
        ),
        encoding="utf-8",
    )
    with pytest.raises(ProvenanceError, match="trusted path alias"):
        load_and_verify(mutated, ROOT)


def test_symlink_and_archive_special_entries_fail_closed(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    outside = tmp_path / "outside.txt"
    outside.write_text("escape", encoding="utf-8")
    (source / "escape").symlink_to(outside)
    with pytest.raises(ProvenanceError):
        manifest(source)
    with pytest.raises(ProvenanceError):
        validate_archive_members(
            [{"name": "root/link", "type": "symlink", "linkname": "/etc"}], "root/"
        )


def test_dirty_copied_detached_repo_is_rejected(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    subprocess.run(["git", "init", "--quiet", str(repo)], check=True)
    (repo / "tracked.txt").write_text("clean", encoding="utf-8")
    subprocess.run(["git", "-C", str(repo), "add", "tracked.txt"], check=True)
    subprocess.run(
        [
            "git",
            "-C",
            str(repo),
            "-c",
            "user.email=test@example.com",
            "-c",
            "user.name=test",
            "commit",
            "--quiet",
            "-m",
            "init",
        ],
        check=True,
    )
    subprocess.run(["git", "-C", str(repo), "switch", "--detach", "--quiet", "HEAD"], check=True)
    (repo / "tracked.txt").write_text("dirty", encoding="utf-8")
    with pytest.raises(ProvenanceError):
        verify_clean_detached(repo)


def test_stale_evidence_and_lock_drift_fail_closed(tmp_path: Path) -> None:
    stale = tmp_path / "stale.yaml"
    stale.write_text(CONFIG.read_text().replace("stanford.tree", "missing.tree"), encoding="utf-8")
    with pytest.raises(ProvenanceError):
        load_and_verify(stale, ROOT)
    lock = tmp_path / "lock.yaml"
    config_text = CONFIG.read_text(encoding="utf-8")
    current = yaml.safe_load(config_text)["locks"]["collection"]["sha256"]
    lock.write_text(config_text.replace(current, "f" * 64, 1), encoding="utf-8")
    with pytest.raises(ProvenanceError):
        load_and_verify(lock, ROOT)


def test_unsafe_path_and_wrong_origin_fail_closed(tmp_path: Path) -> None:
    unsafe = tmp_path / "unsafe.yaml"
    unsafe.write_text(
        CONFIG.read_text().replace("cache/upstream/stanford", "../outside"), encoding="utf-8"
    )
    with pytest.raises(ProvenanceError):
        load_and_verify(unsafe, ROOT)
    wrong = tmp_path / "origin.yaml"
    wrong.write_text(CONFIG.read_text().replace("file://", "https://"), encoding="utf-8")
    with pytest.raises(ProvenanceError):
        load_and_verify(wrong, ROOT)


def test_runtime_partition_and_import_origin_fail_closed() -> None:
    config = cast("dict[str, object]", yaml.safe_load(CONFIG.read_text()))
    config["project_root"] = str(ROOT)
    verify_import_origins("paper", {"project.module": PACKAGE / "src/module.py"}, config)
    with pytest.raises(ProvenanceError):
        verify_import_origins("paper", {"lerobot": PACKAGE / "src/module.py"}, config)
    with pytest.raises(ProvenanceError):
        verify_import_origins("collection", {"diffusion_policy": PACKAGE / "src/module.py"}, config)
    with pytest.raises(ProvenanceError):
        verify_import_origins("paper", {"third_party": "/untrusted.py"}, config)


def test_missing_or_invalid_cli_config_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(ProvenanceError):
        load_and_verify(tmp_path / "missing.yaml", ROOT)
    invalid = tmp_path / "invalid.yaml"
    invalid.write_text("not: [valid", encoding="utf-8")
    with pytest.raises(ProvenanceError):
        load_and_verify(invalid, ROOT)


def test_package_directory_is_not_an_approved_project_root() -> None:
    with pytest.raises(ProvenanceError):
        load_and_verify(CONFIG, PACKAGE)
