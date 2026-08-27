"""Todo 19 CLI: run one guarded bounded rollout and print the terminal receipt."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
import sys
import time

from so101_pusht_benchmark.hardware_profile import compatibility_blockers, load_hardware_profile
from so101_pusht_benchmark.sim_to_real.bounded_authorization import (
    load_bounded_authorization,
    verify_single_step_receipt,
)
from so101_pusht_benchmark.sim_to_real.bounded_input import check_bounded_budgets
from so101_pusht_benchmark.sim_to_real.bounded_rollout import (
    BoundedRolloutInput,
    run_fixture_bounded_rollout,
)
from so101_pusht_benchmark.sim_to_real.direct_bus_adapter import lerobot_so101_factory
from so101_pusht_benchmark.sim_to_real.ledger_chain import canonical_hash
from so101_pusht_benchmark.sim_to_real.live_capture_provider import LIVE_READ_PROVIDER_DIGEST
from so101_pusht_benchmark.sim_to_real.policy_approval import (
    ProductionTrustStore,
    RsaPkcs1v15Sha256Anchor,
)
from so101_pusht_benchmark.sim_to_real.policy_parser import load_production_safety_policy
from so101_pusht_benchmark.sim_to_real.production_bounded import (
    ProductionBoundedBudget,
    execute_production_bounded,
)
from so101_pusht_benchmark.sim_to_real.production_cycle_provider import (
    FILE_CYCLE_PROVIDER_DIGEST,
    CommandCyclePackageBuilder,
    FileCycleProviderConfig,
    FileProductionCycleProvider,
)
from so101_pusht_benchmark.sim_to_real.production_evidence import ReadbackRobot
from so101_pusht_benchmark.sim_to_real.receipt_routing import (
    ReceiptRoutingError,
    prepare_receipt_directory,
)
from typing import cast
from so101_pusht_benchmark.sim_to_real.rollout_codes import RolloutCode, RolloutViolation
from so101_pusht_benchmark.sim_to_real.shadow_types import FixtureClock


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a guarded bounded one-action rollout and print the receipt.",
    )
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--fixture", type=Path, help="bounded fixture directory")
    mode.add_argument("--profile", type=Path, help="real hardware profile")
    parser.add_argument(
        "--authorization", type=Path, required=True, help="bounded authorization file"
    )
    parser.add_argument(
        "--single-step-receipt",
        type=Path,
        required=True,
        help="verified single-step promotion receipt",
    )
    parser.add_argument("--output-dir", type=Path, required=True, help="receipt output directory")
    parser.add_argument("--policy", type=Path, default=None, help="safety policy (fixture default)")
    parser.add_argument("--trust-anchor", type=Path)
    parser.add_argument("--cycle-package-dir", type=Path)
    parser.add_argument("--cycle-builder", type=Path)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--confirm-approval-id")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    policy_path = (
        args.policy
        if args.policy is not None
        else Path(__file__).resolve().parents[1]
        / "tests/fixtures/sim_to_real/collision_approved_policy.yaml"
    )
    if args.profile is not None:
        if not args.authorization.is_file():
            print("R_MISSING: bounded authorization missing", file=sys.stderr)
            return 2
        if (
            args.policy is None
            or args.trust_anchor is None
            or args.cycle_package_dir is None
            or not args.execute
            or not args.confirm_approval_id
        ):
            print(
                "R_POLICY_UNAUTHORIZED: production bounded execution requires signed inputs, cycle packages, --execute, and confirmation",
                file=sys.stderr,
            )
            return 2
        try:
            profile = load_hardware_profile(args.profile)
            blockers = compatibility_blockers(profile)
            if blockers:
                raise RolloutViolation(RolloutCode.R_POLICY_UNAUTHORIZED, "; ".join(blockers))
            trust = ProductionTrustStore.from_owner_anchors(
                (RsaPkcs1v15Sha256Anchor.from_pem_file(args.trust_anchor),)
            )
            now = datetime.now(timezone.utc)
            single_digest = verify_single_step_receipt(args.single_step_receipt)
            authorization = load_bounded_authorization(
                args.authorization,
                now=now,
                single_step_receipt_digest=single_digest,
                production_verifier=trust,
            )
            if authorization.signed_document.get("approval_id") != args.confirm_approval_id:
                raise RolloutViolation(RolloutCode.R_HASH_MISMATCH, "bounded approval confirmation")
            policy = load_production_safety_policy(args.policy, trust_store=trust, now=now)
            builder = (
                CommandCyclePackageBuilder(args.cycle_builder, authorization.cycle_provider_digest)
                if args.cycle_builder is not None
                else None
            )
            expected_provider_digest = (
                builder.digest if builder is not None else FILE_CYCLE_PROVIDER_DIGEST
            )
            if authorization.cycle_provider_digest != expected_provider_digest:
                raise RolloutViolation(
                    RolloutCode.R_HASH_MISMATCH, "bounded cycle provider binding"
                )
            if (
                authorization.policy_digest != policy.canonical_digest
                or profile.policy_digest != policy.canonical_digest
            ):
                raise RolloutViolation(RolloutCode.R_HASH_MISMATCH, "bounded policy binding")
            check_bounded_budgets(authorization, policy)
            output_dir = prepare_receipt_directory(args.output_dir, production=True)
            robot = lerobot_so101_factory(profile)
            provider = FileProductionCycleProvider(
                FileCycleProviderConfig(
                    args.cycle_package_dir,
                    output_dir,
                    robot,
                    cast("ReadbackRobot", robot),
                    LIVE_READ_PROVIDER_DIGEST,
                    policy.canonical_digest,
                    profile.camera.device,
                    time.monotonic,
                    builder,
                )
            )
            result = execute_production_bounded(
                provider,
                ProductionBoundedBudget(
                    authorization.max_commands,
                    authorization.max_duration_seconds,
                    authorization.max_path_length_m,
                    authorization.max_error_count,
                ),
                clock=time.monotonic,
                initial_evidence_digest=single_digest,
            )
            document = {
                "schema": 3,
                "mode": "production_bounded_rollout",
                "state": result.state,
                "write_count": result.write_count,
                "command_ids": list(result.command_ids),
                "fault_code": result.fault_code,
                "error_count": result.error_count,
                "authorization_digest": authorization.digest,
                "single_step_receipt_digest": single_digest,
                "evidence_scope": "authorized_physical_diagnostic",
            }
            document["digest"] = canonical_hash(document)
            (output_dir / "receipt.json").write_text(
                json.dumps(document, indent=2, sort_keys=True) + "\n", encoding="utf-8"
            )
        except (OSError, ReceiptRoutingError, RolloutViolation, TypeError, ValueError) as exc:
            print(exc, file=sys.stderr)
            return 2
        encoded = json.dumps(document, sort_keys=True)
        if result.state == "FAULT":
            print(encoded, file=sys.stderr)
            return 3
        print(encoded)
        return 0
    if args.fixture is None:
        print("R_MISSING: bounded fixture missing", file=sys.stderr)
        return 2
    inputs = BoundedRolloutInput(
        fixture_dir=args.fixture,
        authorization_path=args.authorization,
        policy_path=policy_path,
        single_step_receipt_path=args.single_step_receipt,
        output_dir=args.output_dir,
        now=datetime(2026, 8, 23, 12, tzinfo=timezone.utc),
        clock=FixtureClock(start=1000.0, step=0.01),
    )
    try:
        result = run_fixture_bounded_rollout(inputs)
    except (ReceiptRoutingError, RolloutViolation) as exc:
        print(exc, file=sys.stderr)
        return 2
    encoded = json.dumps(result.to_document(), sort_keys=True)
    if result.state == "FAULT":
        print(encoded, file=sys.stderr)
        return 3
    print(encoded)
    return 0


if __name__ == "__main__":
    sys.exit(main())
