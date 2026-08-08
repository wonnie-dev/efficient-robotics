"""Summarize seeded pilot artifacts without making final-evaluation claims."""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ROOT = ROOT / "outputs" / "seeded_pilot"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pilot-root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    pilot_root = args.pilot_root.resolve()
    batch = json.loads(
        (pilot_root / "batch_summary.json").read_text(encoding="utf-8")
    )
    episode_paths = sorted(
        (pilot_root / "episodes").glob("benchmark_seed*/episode.json")
    )
    if not episode_paths:
        raise RuntimeError("No seeded pilot episode results were found")

    episodes = []
    inference_seconds = []
    model_load_seconds = []
    peak_memory_gib = []
    cache_hits = 0
    for path in episode_paths:
        episode = json.loads(path.read_text(encoding="utf-8"))
        observations = [episode["initial_observation"]]
        observations.extend(
            item
            for item in (
                episode["new_observation"],
                episode["second_new_observation"],
            )
            if item is not None
        )
        for observation in observations:
            metrics = observation["recorded_inference_metrics"]
            inference_seconds.append(metrics["inference_seconds"])
            model_load_seconds.append(metrics["model_load_seconds"])
            peak_memory_gib.append(metrics["peak_gpu_memory_gib"])
            cache_hits += int(observation["cache_hit"])
        debug = episode["debug_ground_truth_metrics"]
        final_debug = next(
            (
                debug[name]
                for name in ("second_post_action", "post_action", "initial")
                if debug[name].get("available")
            ),
            {"target_correct": None},
        )
        episodes.append(
            {
                "episode_id": episode["episode_id"],
                "first_action": episode["first_action"]["type"],
                "final_action": episode["final_action"]["type"],
                "consumed_observation_count": len(observations),
                "final_target_correct_debug_only": final_debug["target_correct"],
            }
        )

    batch_walls = [item["wall_seconds"] for item in batch["results"]]
    report = {
        "schema_version": "seeded-pilot-report-v1",
        "purpose": "pipeline_validation_only_not_final_paper_evidence",
        "training": "not_performed",
        "calibration": "not_performed",
        "testing": "not_performed",
        "episode_count": len(episodes),
        "successful_episode_count": batch["successful_episode_count"],
        "failed_episode_count": batch["failed_episode_count"],
        "consumed_observation_count": len(inference_seconds),
        "active_view_episode_count": sum(
            item["first_action"].startswith("viewpoint_")
            for item in episodes
        ),
        "direct_grasp_episode_count": sum(
            item["first_action"] == "grasp" for item in episodes
        ),
        "final_grasp_episode_count": sum(
            item["final_action"] == "grasp" for item in episodes
        ),
        "final_target_correct_episode_count_debug_only": sum(
            item["final_target_correct_debug_only"] is True
            for item in episodes
        ),
        "runtime": {
            "capture_batch_seconds": json.loads(
                (pilot_root / "capture_summary.json").read_text(encoding="utf-8")
            )["wall_seconds"],
            "vlm_batch_seconds": batch["wall_seconds"],
            "episode_wall_seconds_mean_including_cache": statistics.mean(
                batch_walls
            ),
            "episode_wall_seconds_min": min(batch_walls),
            "episode_wall_seconds_max": max(batch_walls),
            "inference_seconds_per_observation_mean": statistics.mean(
                inference_seconds
            ),
            "inference_seconds_per_observation_min": min(inference_seconds),
            "inference_seconds_per_observation_max": max(inference_seconds),
            "model_load_seconds_per_observation_mean": statistics.mean(
                model_load_seconds
            ),
        },
        "gpu": {
            "physical_gpu": 5,
            "visible_device_count": 1,
            "batch_size": 1,
            "parallel_jobs": False,
            "distributed": False,
            "peak_memory_gib": max(peak_memory_gib),
        },
        "cache": {
            "root": str(
                ROOT / "outputs" / "pilot_cache" / "qwen3_vl"
            ),
            "format": [
                "request.json",
                "output.json",
                "metrics.json",
                "inference_stdout.log",
                "inference_stderr.log",
            ],
            "cache_hits_in_reported_batch": cache_hits,
        },
        "episodes": episodes,
        "limitations": [
            "Simulator ground truth is read only after planning for debugging.",
            "Scores and confidence thresholds are uncalibrated.",
            "Observations are deterministic pre-captured simulation replay.",
            "Grasp selection is not grasp execution or contact-physics evidence.",
            "Ten pilot seeds do not provide a statistical guarantee.",
        ],
        "valid_for_final_evaluation": False,
    }
    output_path = args.output or pilot_root / "pilot_report.json"
    output_path.write_text(
        json.dumps(report, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"EPISODES={report['episode_count']}")
    print(f"FAILED={report['failed_episode_count']}")
    print(f"ACTIVE_VIEW={report['active_view_episode_count']}")
    print(f"WROTE={output_path}")


if __name__ == "__main__":
    main()
