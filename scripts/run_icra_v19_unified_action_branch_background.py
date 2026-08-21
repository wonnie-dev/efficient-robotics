#!/usr/bin/env python3
"""Run live unified-policy branches without forcing the planner's first action."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/research/icra_v19_unified_action_branch_live_smoke.json"
RUNNER = ROOT / "scripts/run_icra_v13_joint_live_development.py"
STATUS = ROOT / "outputs/live_pipeline/icra_v19_unified_action_branch_live_smoke/status.json"

sys.path.insert(0, str(ROOT / "scripts"))
from run_icra_v13_joint_live_stress_batch import (  # noqa: E402
    gpu_device_minors,
    isolated_command,
    require_gpu_devices,
)


def utc_now() -> str:
    """Return an ISO-8601 UTC timestamp for experiment records."""
    return datetime.now(timezone.utc).isoformat()


def load_json(path: Path) -> dict[str, Any]:
    """Load one experiment configuration or result artifact."""
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    """Atomically replace a JSON status artifact."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def action_kind(action: str | None) -> str | None:
    """Collapse parameterized grasp actions to their shared action kind."""
    if action is None:
        return None
    return "grasp" if action.startswith("grasp:") else action


def run_job(
    job: dict[str, Any],
    base: dict[str, Any],
    output_root: Path,
    minor: int,
) -> dict[str, Any]:
    """Run one isolated live episode and evaluate its declared conditions."""
    seed = int(job["seed"])
    gpu = int(job["physical_gpu"])
    job_root = output_root / str(job["headline_scenario"])
    effective = {
        **base,
        "schema_version": "icra-v19-unified-action-branch-job-v1",
        "status": "nonreserved_live_action_branch_diagnostic",
        "experiment_id": f"v19_{job['headline_scenario']}_seed{seed}",
        "seed": seed,
        "scene_variant": str(job["scene_variant"]),
        "initial_task_state": str(job["initial_task_state"]),
        "record_debug_video": False,
        "valid_for_final_evaluation": False,
    }
    config_path = output_root / "job_configs" / f"seed{seed}.json"
    write_json(config_path, effective)
    log_root = output_root / "logs"
    log_root.mkdir(parents=True, exist_ok=True)
    stdout_path = log_root / f"seed{seed}_stdout.log"
    stderr_path = log_root / f"seed{seed}_stderr.log"
    environment = dict(os.environ)
    environment.update(
        {
            "CUDA_VISIBLE_DEVICES": "0",
            "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
            "NVIDIA_VISIBLE_DEVICES": "0",
            "PHYSICAL_GPU": "0",
            "EFFICIENT_ROBOTICS_HOST_PHYSICAL_GPU": str(gpu),
            "EFFICIENT_ROBOTICS_ISAAC_PYTHON": environment.get(
                "EFFICIENT_ROBOTICS_ISAAC_PYTHON",
                str(ROOT / ".venv-isaac/bin/python"),
            ),
            "EFFICIENT_ROBOTICS_PERCEPTION_PYTHON": environment.get(
                "EFFICIENT_ROBOTICS_PERCEPTION_PYTHON",
                str(ROOT / ".venv-perception/bin/python"),
            ),
            "EFFICIENT_ROBOTICS_LIBERO_ROOT": environment.get(
                "EFFICIENT_ROBOTICS_LIBERO_ROOT",
                str(ROOT / "third_party/LIBERO"),
            ),
            "EFFICIENT_ROBOTICS_MODELS_ROOT": environment.get(
                "EFFICIENT_ROBOTICS_MODELS_ROOT",
                str(ROOT / "models"),
            ),
        }
    )
    for name in ("RANK", "LOCAL_RANK", "WORLD_SIZE", "MASTER_ADDR", "MASTER_PORT"):
        environment.pop(name, None)
    command = [
        sys.executable,
        str(RUNNER),
        "--config",
        str(config_path),
        "--seed",
        str(seed),
        "--output-root",
        str(job_root),
        "--timeout-seconds",
        "3600",
        "--startup-timeout-seconds",
        "1200",
    ]
    with stdout_path.open("w", encoding="utf-8") as stdout, stderr_path.open(
        "w", encoding="utf-8"
    ) as stderr:
        completed = subprocess.run(
            isolated_command(command, minor),
            cwd=ROOT,
            env=environment,
            stdout=stdout,
            stderr=stderr,
            text=True,
            check=False,
        )
    result_paths = sorted(job_root.glob(f"seed{seed:03d}/run*/icra_v13_joint_live_result.json"))
    result_path = result_paths[-1] if result_paths else None
    result = load_json(result_path) if result_path else {}
    sequence = list(result.get("action_sequence") or [])
    sequence_kinds = [action_kind(action) for action in sequence]
    first_kind = action_kind(sequence[0] if sequence else None)
    terminal_kind = action_kind(result.get("terminal_joint_action"))
    first_expected = job.get("expected_first_action_kind")
    terminal_expected = job.get("expected_terminal_action_kind")
    required_action = job.get("required_action_kind")
    require_scientific_success = bool(
        job.get("require_scientific_episode_success", False)
    )
    passed = bool(
        completed.returncode == 0
        and result.get("status") == "completed"
        and (first_expected is None or first_kind == first_expected)
        and (terminal_expected is None or terminal_kind == terminal_expected)
        and (required_action is None or required_action in sequence_kinds)
        and result.get("root_action_forced") is False
        and (
            not require_scientific_success
            or result.get("scientific_episode_success") is True
        )
    )
    return {
        **job,
        "returncode": completed.returncode,
        "passed": passed,
        "result": str(result_path.resolve()) if result_path else None,
        "action_sequence": sequence,
        "observed_first_action_kind": first_kind,
        "observed_terminal_action_kind": terminal_kind,
        "required_action_observed": (
            None if required_action is None else required_action in sequence_kinds
        ),
        "root_action_forced": result.get("root_action_forced"),
        "scientific_episode_success": result.get("scientific_episode_success"),
        "stdout": str(stdout_path.resolve()),
        "stderr": str(stderr_path.resolve()),
    }


def main(config_path: Path = CONFIG) -> None:
    """Schedule one sequential job queue per physical GPU."""
    config = load_json(config_path)
    base = load_json(ROOT / str(config["base_config"]))
    jobs = list(config["jobs"])
    gpu_ids = list(config["gpu_ids"])
    require_gpu_devices(gpu_ids)
    minors = gpu_device_minors(gpu_ids)
    output_root = ROOT / str(config["output_root"])
    status_path = output_root / "status.json"
    queues = {gpu: [] for gpu in gpu_ids}
    for job in jobs:
        queues[int(job["physical_gpu"])].append(job)
    state: dict[str, Any] = {
        "schema_version": "icra-v19-unified-action-branch-status-v1",
        "status": "running",
        "started_at_utc": utc_now(),
        "gpu_ids": gpu_ids,
        "forbidden_gpu_ids": list(config.get("forbidden_gpu_ids", [])),
        "device_isolation": "bubblewrap_single_nvidia_device",
        "root_action_forced": False,
        "reserved_test_seeds_used": False,
        "training_performed": False,
        "testing_performed": False,
        "valid_for_final_evaluation": False,
        "jobs": [],
    }
    write_json(status_path, state)
    wave_count = max(len(queue) for queue in queues.values())
    for wave in range(wave_count):
        processes = []
        for gpu, queue in queues.items():
            if wave >= len(queue):
                continue
            job = queue[wave]
            # A short-lived worker process keeps each Isaac/VLM instance isolated.
            worker_output = output_root / "worker_results" / f"seed{int(job['seed'])}.json"
            worker_config = output_root / "worker_inputs" / f"seed{int(job['seed'])}.json"
            write_json(worker_config, {"job": job, "base": base, "output_root": str(output_root), "minor": minors[gpu]})
            process = subprocess.Popen(
                [sys.executable, __file__, "--worker", str(worker_config), "--worker-output", str(worker_output)],
                cwd=ROOT,
            )
            processes.append((job, process, worker_output))
        wave_failed = False
        for job, process, worker_output in processes:
            returncode = process.wait()
            record = load_json(worker_output) if worker_output.is_file() else {**job, "passed": False, "worker_returncode": returncode}
            state["jobs"].append(record)
            wave_failed = wave_failed or not bool(record.get("passed"))
            write_json(status_path, state)
        if wave_failed and bool(config.get("stop_on_wave_failure", True)):
            break
    first_actions = sorted(
        {str(row.get("observed_first_action_kind")) for row in state["jobs"] if row.get("observed_first_action_kind")}
    )
    state.update(
        {
            "status": "completed" if len(state["jobs"]) == len(jobs) and all(row.get("passed") for row in state["jobs"]) else "failed",
            "completed_at_utc": utc_now(),
            "distinct_observed_first_action_kinds": first_actions,
            "first_action_diversity_count": len(first_actions),
        }
    )
    write_json(status_path, state)
    if state["status"] != "completed":
        raise SystemExit(1)


if __name__ == "__main__":
    if "--worker" in sys.argv:
        worker_index = sys.argv.index("--worker")
        output_index = sys.argv.index("--worker-output")
        payload = load_json(Path(sys.argv[worker_index + 1]))
        record = run_job(
            payload["job"],
            payload["base"],
            Path(payload["output_root"]),
            int(payload["minor"]),
        )
        write_json(Path(sys.argv[output_index + 1]), record)
        raise SystemExit(0 if record["passed"] else 1)
    config_path = CONFIG
    if "--config" in sys.argv:
        config_index = sys.argv.index("--config")
        config_path = Path(sys.argv[config_index + 1]).resolve()
    main(config_path)
