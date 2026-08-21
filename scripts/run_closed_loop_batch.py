#!/usr/bin/env python3
"""Run the configured closed-loop episode batch across GPU shards."""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import argparse
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/research/joint_live_stress_episodes.json"
RUNNER = ROOT / "scripts/run_closed_loop_episode.py"
NVIDIA_SHARED_DEVICES = (
    "/dev/nvidiactl",
    "/dev/nvidia-uvm",
    "/dev/nvidia-uvm-tools",
    "/dev/nvidia-modeset",
)


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def resolve_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def completed_result_exists(output_root: Path, seed: int) -> bool:
    """Treat a saved scientific result as authoritative over transport exit codes."""
    candidates = sorted(
        (output_root / "episodes" / f"seed{seed}").glob(
            "run*/closed_loop_result.json"
        )
    )
    for path in reversed(candidates):
        result = load_json(path)
        if result.get("scientific_episode_success") is True:
            return True
    return False


def require_gpu_devices(gpu_ids: list[int]) -> None:
    for gpu, minor in gpu_device_minors(gpu_ids).items():
        if not Path(f"/dev/nvidia{minor}").exists():
            raise RuntimeError(
                f"GPU index {gpu} maps to missing /dev/nvidia{minor}; batch not started"
            )


def gpu_device_minors(gpu_ids: list[int]) -> dict[int, int]:
    """Map nvidia-smi indices to driver device minors on this host."""
    query = subprocess.check_output(
        [
            "nvidia-smi",
            "--query-gpu=index,pci.bus_id",
            "--format=csv,noheader,nounits",
        ],
        text=True,
    )
    bus_by_index = {}
    for line in query.splitlines():
        index_text, bus_id = (part.strip() for part in line.split(",", 1))
        domain, bus, device = bus_id.lower().split(":", 2)
        bus_by_index[int(index_text)] = f"{domain[-4:]}:{bus}:{device}"

    minors = {}
    for gpu in gpu_ids:
        if gpu not in bus_by_index:
            raise RuntimeError(f"nvidia-smi GPU index {gpu} is unavailable")
        information = Path(
            f"/proc/driver/nvidia/gpus/{bus_by_index[gpu]}/information"
        ).read_text(encoding="utf-8")
        match = re.search(r"^Device Minor:\s+(\d+)\s*$", information, re.MULTILINE)
        if match is None:
            raise RuntimeError(f"Device Minor is missing for GPU index {gpu}")
        minors[gpu] = int(match.group(1))
    return minors


def isolated_command(command: list[str], device_minor: int) -> list[str]:
    """Expose one physical NVIDIA device and shared driver nodes to a child."""
    wrapped = [
        "bwrap",
        "--die-with-parent",
        "--bind", "/", "/",
        "--dev", "/dev",
        "--proc", "/proc",
        "--dev-bind", f"/dev/nvidia{device_minor}", f"/dev/nvidia{device_minor}",
    ]
    for path in NVIDIA_SHARED_DEVICES:
        if Path(path).exists():
            wrapped.extend(["--dev-bind", path, path])
    return [*wrapped, *command]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=CONFIG)
    args = parser.parse_args()
    config = load_json(args.config.resolve())
    gpu_ids = [int(value) for value in config["gpu_ids"]]
    if not gpu_ids or len(gpu_ids) != len(set(gpu_ids)) or any(
        gpu < 0 for gpu in gpu_ids
    ):
        raise ValueError("gpu_ids must contain unique non-negative indices")
    require_gpu_devices(gpu_ids)
    device_minors = gpu_device_minors(gpu_ids)
    assignments = list(config["assignments"])
    base = load_json(resolve_path(config["base_episode_config"]))
    base["collect_development_failures"] = True
    output_root = resolve_path(config["output_root"])
    status_path = output_root / "batch_status.json"
    previous = load_json(status_path) if status_path.is_file() else {}
    completed = {int(seed) for seed in previous.get("completed_seeds", [])}
    completed.update(
        int(row["seed"])
        for row in assignments
        if completed_result_exists(output_root, int(row["seed"]))
    )
    pending = [row for row in assignments if int(row["seed"]) not in completed]
    queues = {
        gpu: [row for row in pending if int(row["physical_gpu"]) == gpu]
        for gpu in gpu_ids
    }
    state = {
        "schema_version": "closed-loop-batch-status-v1",
        "status": "running",
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "gpu_ids": gpu_ids,
        "gpu_device_minors": {str(key): value for key, value in device_minors.items()},
        "device_isolation": "bubblewrap_single_nvidia_device",
        "one_process_per_gpu": True,
        "multi_gpu_inside_process": False,
        "resumed_from_previous_status": bool(previous),
        "completed_seeds": sorted(completed),
        "failed_seeds": [],
        "failed_attempts": list(previous.get("failed_attempts", []))
        + list(previous.get("failed_seeds", [])),
        "training_performed": False,
        "testing_performed": False,
    }
    write_json(status_path, state)
    config_root = output_root / "episode_configs"
    for wave in range(max((len(queue) for queue in queues.values()), default=0)):
        jobs = []
        for gpu, queue in queues.items():
            if wave >= len(queue):
                continue
            assignment = queue[wave]
            seed = int(assignment["seed"])
            episode = dict(base)
            episode.update(
                {
                    "seed": seed,
                    "scene_variant": assignment["scene_variant"],
                    "development_family": assignment["family"],
                    "host_physical_gpu": gpu,
                    "host_nvidia_device_minor": device_minors[gpu],
                    "collect_development_failures": True,
                }
            )
            episode_config = config_root / f"seed{seed}.json"
            write_json(episode_config, episode)
            environment = dict(os.environ)
            # A single exposed device is enumerated as logical GPU 0 inside
            # the isolated process, including by Vulkan.
            gpu_text = "0"
            environment.update(
                {
                    "CUDA_VISIBLE_DEVICES": gpu_text,
                    "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
                    "NVIDIA_VISIBLE_DEVICES": gpu_text,
                    "PHYSICAL_GPU": gpu_text,
                    "EFFICIENT_ROBOTICS_HOST_PHYSICAL_GPU": str(gpu),
                    "EFFICIENT_ROBOTICS_ISAAC_PYTHON": environment.get(
                        "EFFICIENT_ROBOTICS_ISAAC_PYTHON",
                        str(ROOT / ".venv-isaac/bin/python"),
                    ),
                    "EFFICIENT_ROBOTICS_LIBERO_ROOT": environment.get(
                        "EFFICIENT_ROBOTICS_LIBERO_ROOT",
                        str(ROOT / "third_party/LIBERO"),
                    ),
                    "EFFICIENT_ROBOTICS_PERCEPTION_PYTHON": environment.get(
                        "EFFICIENT_ROBOTICS_PERCEPTION_PYTHON",
                        str(ROOT / ".venv-perception/bin/python"),
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
                "--config", str(episode_config),
                "--seed", str(seed),
                "--output-root", str(output_root / "episodes"),
                "--startup-timeout-seconds", "1200",
                "--timeout-seconds", "3600",
            ]
            process = subprocess.Popen(
                isolated_command(command, device_minors[gpu]),
                cwd=ROOT,
                env=environment,
            )
            jobs.append((seed, gpu, process))
        for seed, gpu, process in jobs:
            returncode = process.wait()
            if returncode == 0 or completed_result_exists(output_root, seed):
                state["completed_seeds"].append(seed)
                if returncode != 0:
                    state.setdefault("transport_exit_warnings", []).append(
                        {
                            "seed": seed,
                            "physical_gpu": gpu,
                            "returncode": returncode,
                            "saved_scientific_result_authoritative": True,
                        }
                    )
            else:
                failure = {"seed": seed, "physical_gpu": gpu, "returncode": returncode}
                state["failed_seeds"].append(failure)
                state["failed_attempts"].append(failure)
            write_json(status_path, state)
    state["completed_at_utc"] = datetime.now(timezone.utc).isoformat()
    state["status"] = "completed" if not state["failed_seeds"] else "failed"
    write_json(status_path, state)
    if state["failed_seeds"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
