from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import time

import psutil


def gpu_sample() -> dict[str, int | float]:
    result = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=utilization.gpu,memory.used,power.draw,temperature.gpu",
            "--format=csv,noheader,nounits",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    utilization, memory, power, temperature = (
        value.strip() for value in result.stdout.splitlines()[0].split(",")
    )
    return {
        "gpu_utilization_percent": int(utilization),
        "gpu_memory_used_mib": int(memory),
        "gpu_power_watts": float(power),
        "gpu_temperature_celsius": int(temperature),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pid", required=True, type=int)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--interval", default=30, type=int)
    args = parser.parse_args()
    process = psutil.Process(args.pid)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    while process.is_running():
        memory = psutil.virtual_memory()
        value: dict[str, object] = {
            "schema": "pusht-training-resource-sample-v1",
            "elapsed_seconds": time.monotonic() - started,
            "pid": args.pid,
            "cpu_percent": process.cpu_percent(),
            "rss_bytes": process.memory_info().rss,
            "system_memory_used_bytes": memory.used,
            **gpu_sample(),
        }
        with args.output.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(value, sort_keys=True) + "\n")
        time.sleep(args.interval)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
