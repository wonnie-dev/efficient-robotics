#!/usr/bin/env python3
"""Run one command and terminate it if its process group touches a GPU.

This is a fail-closed safety wrapper for shared servers. It monitors both
compute and graphics process types through ``nvidia-smi pmon``.
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import time
from pathlib import Path
from typing import Any


def parse_pmon(output: str) -> list[dict[str, Any]]:
    rows = []
    for line in output.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        fields = stripped.split()
        if len(fields) < 3 or fields[1] == "-":
            continue
        rows.append(
            {
                "gpu": int(fields[0]),
                "pid": int(fields[1]),
                "type": fields[2],
                "memory_mib": (
                    int(fields[3]) if fields[3].isdigit() else None
                ),
            }
        )
    return rows


def process_group(pid: int) -> int | None:
    try:
        return os.getpgid(pid)
    except (ProcessLookupError, PermissionError):
        return None


def terminate_group(group_id: int, grace_seconds: float = 5.0) -> None:
    try:
        os.killpg(group_id, signal.SIGTERM)
    except ProcessLookupError:
        return
    deadline = time.monotonic() + grace_seconds
    while time.monotonic() < deadline:
        try:
            os.killpg(group_id, 0)
        except ProcessLookupError:
            return
        time.sleep(0.05)
    try:
        os.killpg(group_id, signal.SIGKILL)
    except ProcessLookupError:
        pass


def write_status(path: Path | None, value: dict[str, Any]) -> None:
    if path is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def single_gpu_environment(
    base_environment: dict[str, str], physical_gpu: int | None
) -> dict[str, str]:
    """Return a child environment with one explicitly visible physical GPU."""
    environment = dict(base_environment)
    if physical_gpu is None:
        return environment
    if physical_gpu < 0:
        raise ValueError("physical_gpu must be non-negative")
    value = str(physical_gpu)
    environment.update(
        {
            "PHYSICAL_GPU": value,
            "CUDA_VISIBLE_DEVICES": value,
            "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
            "NVIDIA_VISIBLE_DEVICES": value,
        }
    )
    for name in ("RANK", "LOCAL_RANK", "WORLD_SIZE", "MASTER_ADDR", "MASTER_PORT"):
        environment.pop(name, None)
    return environment


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--forbidden-gpu",
        type=int,
        action="append",
        required=True,
        help="Forbidden physical GPU index; repeat to forbid multiple GPUs.",
    )
    parser.add_argument("--poll-seconds", type=float, default=0.1)
    parser.add_argument("--status-json", type=Path)
    parser.add_argument(
        "--physical-gpu",
        type=int,
        help="Expose exactly this physical GPU to the child command.",
    )
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = list(args.command)
    if command and command[0] == "--":
        command = command[1:]
    if not command:
        parser.error("A command is required after --")
    if any(index < 0 for index in args.forbidden_gpu):
        parser.error("--forbidden-gpu must be non-negative")
    forbidden_gpus = sorted(set(args.forbidden_gpu))
    if args.poll_seconds < 0.0:
        parser.error("--poll-seconds must be non-negative")
    if args.physical_gpu is not None:
        if args.physical_gpu < 0:
            parser.error("--physical-gpu must be non-negative")
        if args.physical_gpu in forbidden_gpus:
            parser.error("--physical-gpu cannot also be forbidden")

    started = time.time()
    child_environment = single_gpu_environment(
        os.environ, args.physical_gpu
    )
    child = subprocess.Popen(
        command,
        start_new_session=True,
        env=child_environment,
    )
    status: dict[str, Any] = {
        "schema_version": "forbidden-gpu-watchdog-v1",
        "forbidden_gpus": forbidden_gpus,
        "command": command,
        "process_group_id": child.pid,
        "started_unix_seconds": started,
        "status": "running",
        "violation": None,
        "physical_gpu": args.physical_gpu,
        "child_cuda_visible_devices": child_environment.get(
            "CUDA_VISIBLE_DEVICES"
        ),
    }
    write_status(args.status_json, status)
    try:
        while child.poll() is None:
            probe = subprocess.run(
                ["nvidia-smi", "pmon", "-c", "1", "-s", "m"],
                check=False,
                capture_output=True,
                text=True,
            )
            if child.poll() is not None:
                break
            if probe.returncode != 0:
                status["status"] = "terminated_monitor_failure"
                status["monitor_error"] = probe.stderr.strip()
                terminate_group(child.pid)
                child.wait()
                break
            for row in parse_pmon(probe.stdout):
                if (
                    row["gpu"] in forbidden_gpus
                    and process_group(row["pid"]) == child.pid
                ):
                    status["status"] = "terminated_forbidden_gpu"
                    status["violation"] = row
                    terminate_group(child.pid)
                    child.wait()
                    break
            if status["status"] != "running":
                break
            time.sleep(args.poll_seconds)
    except BaseException:
        terminate_group(child.pid)
        raise

    if status["status"] == "running":
        status["status"] = (
            "completed" if child.returncode == 0 else "command_failed"
        )
    status["returncode"] = child.returncode
    status["finished_unix_seconds"] = time.time()
    status["runtime_seconds"] = (
        status["finished_unix_seconds"] - started
    )
    write_status(args.status_json, status)
    print(json.dumps(status, indent=2, sort_keys=True))
    if status["status"] == "terminated_forbidden_gpu":
        raise SystemExit(97)
    if status["status"] == "terminated_monitor_failure":
        raise SystemExit(98)
    raise SystemExit(child.returncode)


if __name__ == "__main__":
    main()
