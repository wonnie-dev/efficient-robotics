"""Smoke-test the persistent same-stage composite grasp without loading Qwen."""

from __future__ import annotations

import json
import argparse
import os
import subprocess
import time
from pathlib import Path

from run_live_single_gpu_pipeline import (
    ISAAC_PYTHON,
    ROOT,
    wait_for_path,
    write_json_atomic,
)
from run_single_gpu_pilot import require_single_gpu_policy


OUTPUT_ROOT = (
    ROOT / "outputs" / "live_pipeline" / "persistent_composite_grasp_smoke"
)


def next_output_dir() -> Path:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    indices = []
    for path in OUTPUT_ROOT.glob("run*"):
        try:
            indices.append(int(path.name.removeprefix("run")))
        except ValueError:
            continue
    return OUTPUT_ROOT / f"run{max(indices, default=0) + 1:03d}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--physical-gpu", type=int, default=5)
    args = parser.parse_args()
    if args.physical_gpu < 0:
        raise ValueError("physical GPU index must be non-negative")
    require_single_gpu_policy()
    output_dir = next_output_dir()
    output_dir.mkdir(parents=True, exist_ok=False)
    stdout_path = output_dir / "isaac_stdout.log"
    stderr_path = output_dir / "isaac_stderr.log"
    command = [
        str(ISAAC_PYTHON),
        str(ROOT / "scripts" / "open_minimal_scene.py"),
        "--scene-profile",
        "benchmark",
        "--headless",
        "--renderer-gpu",
        str(args.physical_gpu),
        "--physics-gpu",
        "0",
        "--live-pipeline-server",
        "--live-session-dir",
        str(output_dir),
        "--method-config",
        str(
            ROOT
            / "configs"
            / "research"
            / "first_belief_mpc_integration.json"
        ),
        "--seed",
        "0",
        "--execute-persistent-composite-grasp",
        "--rg6-coupling-mode",
        "passive_mimic",
    ]
    started = time.perf_counter()
    with stdout_path.open("w", encoding="utf-8") as stdout_stream, (
        stderr_path.open("w", encoding="utf-8")
    ) as stderr_stream:
        server = subprocess.Popen(
            command,
            cwd=ROOT,
            stdout=stdout_stream,
            stderr=stderr_stream,
            text=True,
            env={
                **os.environ,
                "CUDA_VISIBLE_DEVICES": str(args.physical_gpu),
                "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
                "NVIDIA_VISIBLE_DEVICES": str(args.physical_gpu),
            },
        )
        try:
            wait_for_path(
                output_dir / "observation_ready_000.json",
                server,
                timeout=180.0,
            )
            write_json_atomic(
                output_dir / "action_request_000.json",
                {
                    "schema_version": "persistent-grasp-smoke-request-v1",
                    "index": 0,
                    "type": "grasp",
                    "selected_candidate": "target_red",
                    "source_decision": "debug_smoke_without_vlm",
                },
            )
            wait_for_path(
                output_dir / "server_result.json",
                server,
                timeout=600.0,
            )
            server.wait(timeout=60.0)
        finally:
            if server.poll() is None:
                server.terminate()
                server.wait(timeout=30.0)

    result = json.loads(
        (output_dir / "server_result.json").read_text(encoding="utf-8")
    )
    if (
        server.returncode != 0
        or not result.get("grasp_executed")
        or not result.get("grasp_execution", {}).get("lift_verified")
    ):
        raise RuntimeError(
            "Persistent composite smoke failed: "
            f"returncode={server.returncode}, result={result}"
        )
    summary = {
        "schema_version": "persistent-composite-grasp-smoke-v1",
        "status": "completed",
        "purpose": "same_stage_physics_debug_without_vlm",
        "runtime_seconds": time.perf_counter() - started,
        "gpu_policy": {
            "physical_gpu": args.physical_gpu,
            "renderer_active_gpu": args.physical_gpu,
            "physics_cuda_device": 0,
            "multi_gpu": False,
        },
        "rg6_coupling_mode": "passive_mimic",
        "target_force_controller_max_torque_nm": 1.2,
        "server_result": result,
        "training_performed": False,
        "valid_for_final_evaluation": False,
    }
    write_json_atomic(output_dir / "smoke_result.json", summary)
    print(f"PERSISTENT_COMPOSITE_SMOKE={output_dir / 'smoke_result.json'}")


if __name__ == "__main__":
    main()
