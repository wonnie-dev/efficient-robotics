"""Capture relation-preserving benchmark seeds sequentially on physical GPU 5."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
ISAAC_PYTHON = Path("/data/wonheekoh/isaacsim_venv/bin/python")
VIEWS = ("left", "center", "right", "close_high")


def capture_complete(seed: int) -> bool:
    observation_root = (
        ROOT
        / "outputs"
        / "seeded_pilot"
        / f"benchmark_seed{seed:03d}"
        / "observations"
    )
    required = ("rgb.png", "depth_m.npy", "objects.json", "metadata.json")
    return all(
        (observation_root / view / filename).is_file()
        for view in VIEWS
        for filename in required
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed-start", type=int, default=0)
    parser.add_argument("--seed-stop", type=int, default=10)
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "outputs" / "seeded_pilot" / "capture_summary.json",
    )
    args = parser.parse_args()
    expected = os.environ.get("PHYSICAL_GPU", "5")
    if os.environ.get("CUDA_VISIBLE_DEVICES") != expected:
        raise RuntimeError(f"CUDA_VISIBLE_DEVICES must be exactly {expected}")
    if args.seed_start < 0 or args.seed_stop <= args.seed_start:
        raise ValueError("Expected 0 <= seed-start < seed-stop")
    if not ISAAC_PYTHON.is_file():
        raise FileNotFoundError(ISAAC_PYTHON)

    log_root = ROOT / "outputs" / "seeded_pilot" / "logs"
    log_root.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    results = []
    for seed in range(args.seed_start, args.seed_stop):
        episode_id = f"benchmark_seed{seed:03d}"
        if capture_complete(seed):
            results.append(
                {
                    "episode_id": episode_id,
                    "status": "completed_existing",
                    "wall_seconds": 0.0,
                }
            )
            print(f"SKIP_COMPLETE={episode_id}", flush=True)
            continue
        command = [
            str(ISAAC_PYTHON),
            str(ROOT / "scripts" / "open_minimal_scene.py"),
            "--scene-profile",
            "benchmark",
            "--headless",
            "--renderer-gpu",
            os.environ.get("PHYSICAL_GPU", "5"),
            "--physics-gpu",
            "0",
            "--seeded-pilot-capture",
            "--seed",
            str(seed),
        ]
        episode_started = time.perf_counter()
        completed = subprocess.run(
            command,
            cwd=ROOT,
            text=True,
            capture_output=True,
            check=False,
            env={
                **os.environ,
                "CUDA_VISIBLE_DEVICES": os.environ.get("PHYSICAL_GPU", "5"),
                "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
                "NVIDIA_VISIBLE_DEVICES": "5",
            },
        )
        stdout_path = log_root / f"{episode_id}.stdout.log"
        stderr_path = log_root / f"{episode_id}.stderr.log"
        stdout_path.write_text(completed.stdout, encoding="utf-8")
        stderr_path.write_text(completed.stderr, encoding="utf-8")
        status = (
            "completed"
            if completed.returncode == 0 and capture_complete(seed)
            else "failed"
        )
        results.append(
            {
                "episode_id": episode_id,
                "status": status,
                "returncode": completed.returncode,
                "wall_seconds": time.perf_counter() - episode_started,
                "stdout_log": str(stdout_path),
                "stderr_log": str(stderr_path),
            }
        )
        print(f"{status.upper()}={episode_id}", flush=True)

    summary = {
        "schema_version": "seeded-pilot-capture-summary-v1",
        "purpose": "pipeline_validation_only_not_final_paper_evidence",
        "gpu_policy": {
            "physical_gpu": 5,
            "physics_cuda_device": 0,
            "multi_gpu": False,
            "parallel_jobs": False,
        },
        "views": list(VIEWS),
        "seed_start": args.seed_start,
        "seed_stop": args.seed_stop,
        "successful_episode_count": sum(
            item["status"].startswith("completed") for item in results
        ),
        "failed_episode_count": sum(
            item["status"] == "failed" for item in results
        ),
        "wall_seconds": time.perf_counter() - started,
        "results": results,
        "valid_for_final_evaluation": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(summary, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"WROTE={args.output}", flush=True)
    if summary["failed_episode_count"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
