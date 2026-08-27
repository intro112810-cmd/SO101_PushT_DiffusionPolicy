"""Run one planner-complete continuous sim-to-real shadow campaign.

Fixture campaigns prove the full C1-C4 chain (samples, registration, inference,
transform, physical IK, supervisor, ledger) without importing or invoking any
motor-write symbol. Stale or invalid campaigns latch HOLD with zero writes.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time
from typing import cast

from so101_pusht_benchmark.hardware_profile import load_hardware_profile
from so101_pusht_benchmark.sim_to_real.policy_approval import (
    ProductionTrustStore,
    RsaPkcs1v15Sha256Anchor,
)
from so101_pusht_benchmark.sim_to_real.policy_parser import (
    load_fixture_safety_policy,
    load_production_safety_policy,
)
from so101_pusht_benchmark.sim_to_real.receipt_routing import ReceiptRoutingError
from so101_pusht_benchmark.sim_to_real.replay_history import validate_lineage_receipt
from so101_pusht_benchmark.sim_to_real.rollout_codes import RolloutCode, RolloutViolation
from so101_pusht_benchmark.sim_to_real.shadow_campaign import (
    ShadowCampaignInput,
    run_shadow_campaign,
)
from so101_pusht_benchmark.sim_to_real.shadow_types import FixtureClock

REPLAY_LINEAGE_AUTHORITY_DIGEST = "192d568795b756ac1edcde78a4a24ed8d37f1fef3bde14cd32a6d441c221a5e4"
REPLAY_SOURCE_FRAME = (
    Path(__file__).resolve().parents[1] / "tests/fixtures/sim_to_real/physical_frame.png"
)
FIXTURE_ROOT = Path(__file__).resolve().parents[1] / "tests/fixtures/sim_to_real"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    evidence = parser.add_mutually_exclusive_group(required=True)
    evidence.add_argument("--fixture", type=Path)
    evidence.add_argument("--production-evidence-dir", type=Path)
    parser.add_argument("--policy", required=True, type=Path)
    parser.add_argument("--profile", type=Path)
    parser.add_argument("--trust-anchor", type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--cycle-limit", type=int, default=1)
    parser.add_argument("--policy-seed", type=int, default=8)
    return parser.parse_args()


def _json(path: Path) -> dict[str, object]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise RolloutViolation(RolloutCode.R_MISSING, f"{path} must be a JSON mapping")
    return cast("dict[str, object]", raw)


def _production_inputs(path: Path) -> tuple[str, ...]:
    required = (
        "samples.json",
        "lineage.json",
        "joint-equivalence.json",
        "camera-registration.json",
        "camera-corpus.json",
        "source-frame.png",
        "lineage-authority-digest.txt",
    )
    missing = tuple(name for name in required if not (path / name).is_file())
    if missing:
        raise RolloutViolation(
            RolloutCode.R_MISSING, f"production evidence missing: {','.join(missing)}"
        )
    return required


def main() -> int:
    args = parse_args()
    try:
        production = args.production_evidence_dir is not None
        if production:
            if not args.production_evidence_dir.is_dir():
                raise RolloutViolation(
                    RolloutCode.R_MISSING,
                    "production frozen-policy evidence directory is unavailable",
                )
            if args.profile is None or args.trust_anchor is None:
                raise RolloutViolation(
                    RolloutCode.R_POLICY_UNAUTHORIZED,
                    "production profile and trust anchor required",
                )
            evidence = args.production_evidence_dir
            _production_inputs(evidence)
            profile = load_hardware_profile(args.profile)
            trust = ProductionTrustStore.from_owner_anchors(
                (RsaPkcs1v15Sha256Anchor.from_pem_file(args.trust_anchor),)
            )
            policy = load_production_safety_policy(args.policy, trust_store=trust)
            lineage = _json(evidence / "lineage.json")
            joint = _json(evidence / "joint-equivalence.json")
            camera = _json(evidence / "camera-registration.json")
            corpus = _json(evidence / "camera-corpus.json")
            lineage_authority = (
                (evidence / "lineage-authority-digest.txt").read_text(encoding="utf-8").strip()
            )
            fixture_dir = evidence
            source_frame = evidence / "source-frame.png"
            clock = time.monotonic
            production_digests = (
                profile.joint_equivalence_digest,
                profile.camera_registration_digest,
            )
        else:
            if args.fixture is None:
                raise RolloutViolation(RolloutCode.R_MISSING, "fixture evidence is missing")
            policy = load_fixture_safety_policy(args.policy)
            lineage = _json(FIXTURE_ROOT / "lineage.json")
            joint = _json(FIXTURE_ROOT / "joint-equivalence.json")
            camera = _json(FIXTURE_ROOT / "camera-registration.json")
            corpus = _json(FIXTURE_ROOT / "camera_registration_valid" / "corpus.json")
            lineage_authority = REPLAY_LINEAGE_AUTHORITY_DIGEST
            fixture_dir = args.fixture
            source_frame = REPLAY_SOURCE_FRAME
            clock = FixtureClock(start=1_000_000.0, step=0.01)
            production_digests = None
        validate_lineage_receipt(lineage, expected_digest=lineage_authority)
        inputs = ShadowCampaignInput(
            fixture_dir=fixture_dir,
            policy=policy,
            lineage_document=lineage,
            lineage_authority_digest=lineage_authority,
            joint_document=joint,
            camera_document=camera,
            camera_corpus=corpus,
            source_frame_path=source_frame,
            output_dir=args.output_dir,
            clock=clock,
            cycle_limit=args.cycle_limit,
            policy_seed=args.policy_seed,
            production_receipt_digests=production_digests,
        )
        result = run_shadow_campaign(inputs)
    except (ReceiptRoutingError, RolloutViolation) as exc:
        print(str(exc), file=sys.stderr)
        return 2
    if result.terminal_state == "SHADOW_COMPLETE":
        print(json.dumps(result.to_document(), indent=2, sort_keys=True))
        return 0
    print(json.dumps(result.to_document(), indent=2, sort_keys=True), file=sys.stderr)
    return 3


if __name__ == "__main__":
    raise SystemExit(main())
