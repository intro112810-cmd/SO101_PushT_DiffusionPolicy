"""Planner ledger extraction into execution packages."""

from so101_pusht_benchmark.sim_to_real.execution_package_builder import build_execution_package


def test_build_single_package_binds_same_cycle_samples_proposal_and_token() -> None:
    records: list[dict[str, object]] = [
        {
            "kind": "samples",
            "cycle": 2,
            "sample_records": [
                {"created_at": 1.0, "digest": "a" * 64},
                {"created_at": 2.0, "digest": "b" * 64},
            ],
        },
        {"kind": "ik_proposal", "cycle": 2, "ik_proposal": {"proposal_hash": "c" * 64}},
        {
            "kind": "supervisor_decision",
            "cycle": 2,
            "decision": "ACCEPT",
            "authorization_token": {"token_id": "d" * 64},
        },
    ]
    package = build_execution_package(records, cycle=2, previous_evidence_digest=None)
    assert package["schema"] == "production-single-step-package-v1"
    assert package["pre_sample_digests"] == ["a" * 64, "b" * 64]
    assert package["newer_than"] == 2.0
