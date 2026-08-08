"""Run one physics-only UR10e+RG6 grasp against the scanned basket colliders."""

from __future__ import annotations

import json
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
    ROOT
    / "outputs"
    / "live_pipeline"
    / "scanned_basket_collision_grasp_smoke"
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
        "5",
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
        "--household-perception-pilot",
        "--scanned-basket-perception-pilot",
        "--basket-collision-physics-pilot",
        "--execute-persistent-composite-grasp",
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
                "CUDA_VISIBLE_DEVICES": "5",
                "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
                "NVIDIA_VISIBLE_DEVICES": "5",
            },
        )
        try:
            wait_for_path(
                output_dir / "observation_ready_000.json",
                server,
                timeout=180.0,
            )
            household_scene = json.loads(
                (output_dir / "household_scene.json").read_text(
                    encoding="utf-8"
                )
            )
            collision = household_scene["reference"]["collision_geometry"]
            if (
                collision.get("type")
                != "five_box_static_approximation"
                or len(collision.get("boxes", [])) != 5
            ):
                raise RuntimeError(
                    f"Scanned basket collision was not authored: {collision}"
                )
            write_json_atomic(
                output_dir / "action_request_000.json",
                {
                    "schema_version": (
                        "scanned-basket-collision-grasp-request-v1"
                    ),
                    "index": 0,
                    "type": "grasp",
                    "selected_candidate": "target_red",
                    "source_decision": (
                        "physics_only_debug_without_vlm_or_planner"
                    ),
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

    server_result = json.loads(
        (output_dir / "server_result.json").read_text(encoding="utf-8")
    )
    grasp = server_result.get("grasp_execution") or {}
    success = (
        server.returncode == 0
        and server_result.get("grasp_executed")
        and grasp.get("bilateral_contact_before_lift")
        and grasp.get("lift_verified")
        and not grasp.get("unexpected_environment_pairs")
    )
    summary = {
        "schema_version": "scanned-basket-collision-grasp-smoke-v1",
        "status": "completed" if success else "failed",
        "purpose": (
            "physics_only_scanned_basket_collision_and_contact_validation"
        ),
        "runtime_seconds": time.perf_counter() - started,
        "basket_collision": collision,
        "gpu_policy": {
            "physical_gpu": 5,
            "renderer_active_gpu": 5,
            "physics_cuda_device": 0,
            "multi_gpu": False,
        },
        "qwen_loaded": False,
        "planner_used": False,
        "grasp_height_offset_m": 0.0,
        "training_performed": False,
        "server_result": server_result,
        "valid_for_final_evaluation": False,
    }
    write_json_atomic(output_dir / "smoke_result.json", summary)
    if not success:
        raise RuntimeError(
            "Scanned-basket collision grasp smoke failed: "
            f"returncode={server.returncode}, grasp={grasp}"
        )
    print(
        "SCANNED_BASKET_COLLISION_GRASP_SMOKE="
        f"{output_dir / 'smoke_result.json'}"
    )


if __name__ == "__main__":
    main()
