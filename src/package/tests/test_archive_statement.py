from __future__ import annotations

from so101_pusht_benchmark.workspace import load_workspace_policy


def test_active_workspace_and_frozen_prototype_boundary_are_machine_readable() -> None:
    policy = load_workspace_policy()
    workspace = policy["workspace"]
    assert workspace["status"] == "active"
    assert workspace["mode"] == "native_pusht_so100_four_model_benchmark"
    assert workspace["path"] == "03_code/so101_pusht_benchmark"
    assert set(workspace["legacy_modes_superseded"]) == {
        "multi_task",
        "scripted_expert",
        "custom_sim",
        "mouse_keyboard",
        "selected_view",
        "schema_3",
    }

    prototypes = policy["prototypes"]
    assert {entry["path"] for entry in prototypes} == {
        "03_code/so101_diffusion_policy",
        "03_code/so101",
    }
    assert all(entry["status"] == "frozen" for entry in prototypes)
    assert all(entry["runtime_import"] == "forbidden" for entry in prototypes)
    assert policy["archive"]["physical_archive"] == "after_replacement_vertical_slice"


def test_package_source_has_no_frozen_prototype_imports() -> None:
    package = __import__("so101_pusht_benchmark").__path__[0]
    source = __import__("pathlib").Path(package)
    forbidden = {"so101_dp", "so101"}
    for path in source.rglob("*.py"):
        text = path.read_text(encoding="utf-8")
        assert "sys.path" not in text
        for line in text.splitlines():
            if line.startswith(("import ", "from ")):
                assert not any(name in line.split() for name in forbidden)
