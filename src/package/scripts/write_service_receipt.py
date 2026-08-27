from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--unit", required=True)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--model-output", required=True, type=Path)
    parser.add_argument("--resource-log", required=True, type=Path)
    args = parser.parse_args()
    properties = subprocess.run(
        [
            "systemctl",
            "--user",
            "show",
            args.unit,
            (
                "--property=Result,ExecMainCode,ExecMainStatus,ActiveEnterTimestamp,"
                "InactiveEnterTimestamp,InvocationID"
            ),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    fields = dict(
        line.split("=", 1) for line in properties.stdout.splitlines() if "=" in line
    )
    value = {
        "schema": "pusht-training-service-receipt-v1",
        "unit": args.unit,
        "captured_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "systemd": fields,
        "final_output_exists": args.model_output.is_dir(),
        "training_receipt_exists": (args.model_output / "training_receipt.json").is_file(),
        "resource_log_exists": args.resource_log.is_file(),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    staging = args.output.with_name(f".{args.output.name}.tmp")
    staging.write_text(json.dumps(value, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    staging.replace(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
