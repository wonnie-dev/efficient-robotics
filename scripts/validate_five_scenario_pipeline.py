#!/usr/bin/env python3
"""Finish the nonreserved five-scenario capture, perception, and audit stages."""

from __future__ import annotations

import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
CAPTURE_CONFIG = ROOT / "configs/research/five_scenario_validation_episodes.json"
CAPTURE_ROOT = ROOT / "outputs/live_pipeline/five_scenario_validation_episodes"
PERCEPTION_ROOT = ROOT / "outputs/calibration/five_scenario_perception"
AUDIT_ROOT = ROOT / "outputs/development/five_scenario_grounding_audit"
SELECTION_ROOT = ROOT / "outputs/development/five_scenario_candidate_selection"
STATE_PATH = ROOT / "outputs/development/five_scenario_pipeline/status.json"
GPU_IDS = (0, 2, 4, 5)


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def run(command: list[str], log_name: str) -> None:
    log_root = STATE_PATH.parent / "logs"
    log_root.mkdir(parents=True, exist_ok=True)
    completed = subprocess.run(
        command,
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    (log_root / f"{log_name}_stdout.log").write_text(
        completed.stdout, encoding="utf-8"
    )
    (log_root / f"{log_name}_stderr.log").write_text(
        completed.stderr, encoding="utf-8"
    )
    if completed.returncode != 0:
        raise RuntimeError(
            f"{log_name} failed with return code {completed.returncode}"
        )


def wait_for_capture(state: dict[str, Any]) -> None:
    capture_status_path = CAPTURE_ROOT / "batch_status.json"
    while True:
        if capture_status_path.is_file():
            capture = load_json(capture_status_path)
            state["capture_status"] = capture.get("status")
            state["capture_completed_episode_count"] = len(
                capture.get("completed_seeds", [])
            )
            write_json(STATE_PATH, state)
            if capture.get("status") == "completed":
                if set(int(seed) for seed in capture["completed_seeds"]) != set(
                    range(1334, 1354)
                ):
                    raise RuntimeError("Five-scenario capture seed set changed")
                return
            if capture.get("status") == "failed":
                raise RuntimeError("Five-scenario capture failed")
        time.sleep(30.0)


def main() -> None:
    state: dict[str, Any] = {
        "schema_version": "five-scenario-pipeline-status-v1",
        "status": "waiting_for_capture",
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "gpu_ids": list(GPU_IDS),
        "one_model_instance_per_gpu": True,
        "distributed": False,
        "ddp_used": False,
        "nccl_used": False,
        "training_performed": False,
        "testing_performed": False,
        "reserved_test_seeds_used": False,
        "valid_for_final_evaluation": False,
    }
    write_json(STATE_PATH, state)
    try:
        wait_for_capture(state)
        state["status"] = "running_perception"
        write_json(STATE_PATH, state)
        run(
            [
                sys.executable,
                "scripts/run_calibration_perception_batch.py",
                "--gpu-ids",
                *(str(gpu) for gpu in GPU_IDS),
                "--capture-config",
                str(CAPTURE_CONFIG),
                "--capture-root",
                str(CAPTURE_ROOT),
                "--output-root",
                str(PERCEPTION_ROOT),
            ],
            "perception",
        )
        state["status"] = "running_candidate_audit"
        write_json(STATE_PATH, state)
        run(
            [
                sys.executable,
                "scripts/audit_grounding_candidate_selection.py",
                "--perception-root",
                str(PERCEPTION_ROOT),
                "--policy-result",
                str(STATE_PATH.parent / "policy_result_not_generated_yet.json"),
                "--output-root",
                str(AUDIT_ROOT),
            ],
            "candidate_audit",
        )
        run(
            [
                sys.executable,
                "scripts/evaluate_candidate_selection.py",
                "--audit-root",
                str(AUDIT_ROOT),
                "--output-root",
                str(SELECTION_ROOT),
            ],
            "candidate_selection",
        )
        state.update(
            {
                "status": "completed",
                "completed_at_utc": datetime.now(timezone.utc).isoformat(),
                "perception_result": str((PERCEPTION_ROOT / "COMPLETED.json").resolve()),
                "candidate_audit": str((AUDIT_ROOT / "audit.json").resolve()),
                "candidate_selection": str((SELECTION_ROOT / "result.json").resolve()),
                "next_required_stage": "inspect_condition_wise_failures_then_fit_and_freeze_calibration",
            }
        )
        write_json(STATE_PATH, state)
    except Exception as error:
        state.update(
            {
                "status": "failed",
                "completed_at_utc": datetime.now(timezone.utc).isoformat(),
                "failure_type": type(error).__name__,
                "failure_reason": str(error),
            }
        )
        write_json(STATE_PATH, state)
        raise


if __name__ == "__main__":
    main()
