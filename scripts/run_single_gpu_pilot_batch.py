"""Run available pilot episodes sequentially and write a failure-safe summary."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = ROOT / "outputs" / "vlm_dataset" / "manifest.json"
DEFAULT_OUTPUT = ROOT / "outputs" / "single_gpu_pilot" / "batch_summary.json"


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def available_episodes(manifest_path: Path) -> list[str]:
    manifest = load_json(manifest_path)
    episode_ids = set()
    for sample in manifest["samples"]:
        input_path = (ROOT / sample["input"]).resolve()
        episode_ids.add(load_json(input_path)["episode_id"])
    return sorted(episode_ids)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--episode-output-root",
        type=Path,
        default=ROOT / "outputs" / "single_gpu_pilot",
    )
    parser.add_argument("--maximum-episodes", type=int, default=10)
    parser.add_argument("--allow-cache-miss-inference", action="store_true")
    args = parser.parse_args()
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "5":
        raise RuntimeError("CUDA_VISIBLE_DEVICES must be exactly 5")
    if args.maximum_episodes <= 0:
        raise ValueError("--maximum-episodes must be positive")

    episodes = available_episodes(args.manifest.resolve())[
        : args.maximum_episodes
    ]
    started = time.perf_counter()
    results = []
    for episode_id in episodes:
        command = [
            sys.executable,
            str(ROOT / "scripts" / "run_single_gpu_pilot.py"),
            "--manifest",
            str(args.manifest.resolve()),
            "--episode-id",
            episode_id,
            "--output-root",
            str(args.episode_output_root.resolve()),
        ]
        if args.allow_cache_miss_inference:
            command.append("--allow-cache-miss-inference")
        episode_started = time.perf_counter()
        completed = subprocess.run(
            command,
            cwd=ROOT,
            check=False,
            capture_output=True,
            text=True,
        )
        results.append(
            {
                "episode_id": episode_id,
                "status": (
                    "completed" if completed.returncode == 0 else "failed"
                ),
                "returncode": completed.returncode,
                "wall_seconds": time.perf_counter() - episode_started,
                "stdout": completed.stdout,
                "stderr": completed.stderr,
            }
        )

    summary = {
        "schema_version": "single-gpu-pilot-batch-summary-v1",
        "purpose": "pipeline_validation_only_not_final_paper_evidence",
        "execution": {
            "sequential": True,
            "batch_size": 1,
            "parallel_jobs": False,
            "distributed": False,
            "physical_gpu": 5,
        },
        "available_episode_count": len(episodes),
        "successful_episode_count": sum(
            result["status"] == "completed" for result in results
        ),
        "failed_episode_count": sum(
            result["status"] == "failed" for result in results
        ),
        "wall_seconds": time.perf_counter() - started,
        "results": results,
        "valid_for_final_evaluation": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(f"AVAILABLE_EPISODES={summary['available_episode_count']}")
    print(f"SUCCESSFUL_EPISODES={summary['successful_episode_count']}")
    print(f"FAILED_EPISODES={summary['failed_episode_count']}")
    print(f"WROTE={args.output}")
    if summary["failed_episode_count"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
