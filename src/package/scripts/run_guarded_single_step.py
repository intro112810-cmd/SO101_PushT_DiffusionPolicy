"""Todo 18 CLI: run one guarded fixture single-step and print the receipt JSON."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from datetime import datetime, timezone

from so101_pusht_benchmark.hardware_profile import load_hardware_profile
from so101_pusht_benchmark.sim_to_real.arming import ArmingCheckInput, check_production_arming
from so101_pusht_benchmark.sim_to_real.bounded_execution import DurableIntentLedger
from so101_pusht_benchmark.sim_to_real.direct_bus_adapter import (
    build_real_direct_bus_adapter,
    lerobot_so101_factory,
)
from so101_pusht_benchmark.sim_to_real.live_capture_provider import LIVE_READ_PROVIDER_DIGEST
from so101_pusht_benchmark.sim_to_real.policy_approval import (
    ProductionTrustStore,
    RsaPkcs1v15Sha256Anchor,
)
from so101_pusht_benchmark.sim_to_real.production_evidence import (
    DirectBusEvidenceConfig,
    DirectBusEvidenceProvider,
    ReadbackRobot,
)
from so101_pusht_benchmark.sim_to_real.production_frame import OpenCvPostFrameSource
from so101_pusht_benchmark.sim_to_real.production_package import load_production_package
from so101_pusht_benchmark.sim_to_real.production_single_step import (
    ProductionSingleStepRuntime,
    execute_production_single_step,
)
from so101_pusht_benchmark.sim_to_real.receipt_routing import (
    ReceiptRoutingError,
    prepare_receipt_directory,
)
from so101_pusht_benchmark.sim_to_real.single_step_authorization import (
    load_single_step_authorization,
)
from typing import cast
from so101_pusht_benchmark.sim_to_real.rollout_codes import RolloutViolation
from so101_pusht_benchmark.sim_to_real.single_step import (
    SingleStepRunInput,
    run_fixture_single_step,
)


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run one guarded fixture single-step write and print the receipt.",
    )
    parser.add_argument("--fixture", type=Path, required=False, help="fixture directory (optional)")
    parser.add_argument("--authorization", type=Path, required=True, help="authorization file")
    parser.add_argument("--output-dir", type=Path, required=True, help="receipt output directory")
    parser.add_argument("--profile", type=Path, required=False, help="profile file (optional)")
    parser.add_argument("--policy", type=Path)
    parser.add_argument("--trust-anchor", type=Path)
    parser.add_argument("--shadow-ledger", type=Path)
    parser.add_argument("--operational-evidence", type=Path)
    parser.add_argument("--package", type=Path)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--confirm-command-id")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    if args.profile is not None:
        if args.fixture is not None:
            print("--profile cannot be combined with fixture execution", file=sys.stderr)
            return 2
        required = (
            args.policy,
            args.trust_anchor,
            args.shadow_ledger,
            args.operational_evidence,
            args.package,
        )
        if (
            any(path is None for path in required)
            or not args.execute
            or not args.confirm_command_id
        ):
            print(
                "R_POLICY_UNAUTHORIZED: production execution requires all signed inputs, --execute, and --confirm-command-id",
                file=sys.stderr,
            )
            return 2
        try:
            profile = load_hardware_profile(args.profile)
            anchor = RsaPkcs1v15Sha256Anchor.from_pem_file(args.trust_anchor)
            trust = ProductionTrustStore.from_owner_anchors((anchor,))
            now = datetime.now(timezone.utc)
            authorization = load_single_step_authorization(
                args.authorization, now=now, production_verifier=trust
            )
            if args.confirm_command_id != authorization.command_id:
                print("R_HASH_MISMATCH: command confirmation mismatch", file=sys.stderr)
                return 2
            armed = check_production_arming(
                ArmingCheckInput(
                    args.profile,
                    args.policy,
                    args.shadow_ledger,
                    args.authorization,
                    args.operational_evidence,
                    now,
                ),
                trust,
            )
            prepared = load_production_package(args.package, authorization, armed).prepared
            robot = build_real_direct_bus_adapter(
                profile, authorization, armed, lerobot_so101_factory
            )
            output_dir = prepare_receipt_directory(args.output_dir, production=True)
            intent = DurableIntentLedger(output_dir / "intents.jsonl")
            evidence = DirectBusEvidenceProvider(
                DirectBusEvidenceConfig(
                    cast("ReadbackRobot", robot),
                    LIVE_READ_PROVIDER_DIGEST,
                    authorization.command_id,
                    prepared.proposal.proposal_hash,
                    prepared.proposal.body_degrees,
                    __import__("time").monotonic,
                    OpenCvPostFrameSource(profile.camera.device, output_dir / "post-frame.png"),
                )
            )
            outcome = execute_production_single_step(
                prepared, ProductionSingleStepRuntime(robot, evidence, evidence, intent.append)
            )
            (output_dir / "receipt.json").write_text(
                json.dumps(outcome, indent=2, sort_keys=True) + "\n", encoding="utf-8"
            )
        except (OSError, ReceiptRoutingError, RolloutViolation, TypeError, ValueError) as exc:
            print(exc, file=sys.stderr)
            return 2
        print(json.dumps(outcome, sort_keys=True))
        return 0
    if args.fixture is None:
        print("fixture or production --profile is required", file=sys.stderr)
        return 2
    if not args.authorization.is_file():
        print("authorization missing", file=sys.stderr)
        return 2
    try:
        outcome = run_fixture_single_step(
            SingleStepRunInput(args.fixture, args.authorization, args.output_dir)
        )
    except (ReceiptRoutingError, RolloutViolation) as exc:
        print(exc, file=sys.stderr)
        return 2
    print(json.dumps(outcome, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
