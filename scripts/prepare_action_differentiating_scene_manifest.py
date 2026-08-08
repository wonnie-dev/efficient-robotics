"""Prepare a GPU-free manifest for causal view-action scene captures."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from scanned_basket_scene import (  # noqa: E402
    ACTION_DIFFERENTIATING_SCENE_VARIANTS,
    compute_action_differentiating_layout,
    factorized_calibration_ground_truth,
)


DEFAULT_CONFIG = (
    ROOT
    / "configs"
    / "research"
    / "action_differentiating_scene_pilot.json"
)


def resolve_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def build_scene_manifest(config: dict[str, Any]) -> dict[str, Any]:
    seed_start = int(config["seed_start"])
    seed_stop = int(config["seed_stop_exclusive"])
    if seed_start < 0 or seed_stop <= seed_start:
        raise ValueError("Expected 0 <= seed_start < seed_stop_exclusive")
    seeds = list(range(seed_start, seed_stop))
    reserved = {int(seed) for seed in config["reserved_test_seeds"]}
    overlap = sorted(set(seeds) & reserved)
    if overlap:
        raise ValueError(f"Pilot uses reserved test seeds: {overlap}")
    cycle = tuple(str(value) for value in config["variant_cycle"])
    if set(cycle) != set(ACTION_DIFFERENTIATING_SCENE_VARIANTS):
        raise ValueError(
            "variant_cycle must contain each action-differentiating "
            "variant exactly once"
        )
    if len(cycle) != len(set(cycle)):
        raise ValueError("variant_cycle contains duplicates")
    if len(seeds) % len(cycle) != 0:
        raise ValueError("Pilot seed count must be balanced across variants")
    basket_center = tuple(
        float(value)
        for value in config["nominal_basket_center_world_m"]
    )
    scenes = []
    for offset, seed in enumerate(seeds):
        variant = cycle[offset % len(cycle)]
        layout = compute_action_differentiating_layout(
            basket_center,
            variant,
            seed,
        )
        ground_truth = factorized_calibration_ground_truth(variant)
        scenes.append(
            {
                "seed": seed,
                "variant": variant,
                "layout_preview": layout,
                "intended_action_outcome": ground_truth[
                    "action_outcome_design"
                ],
                "generator_ground_truth": ground_truth,
                "future_capture_command": [
                    "python",
                    str(
                        ROOT
                        / config["future_capture"]["script"]
                    ),
                    "--seed",
                    str(seed),
                    "--scanned-basket-perception-pilot",
                    "--calibration-scene-variant",
                    variant,
                    "--requested-views",
                    *config["future_capture"]["requested_views"],
                    "--output-dir",
                    str(
                        resolve_path(config["output_root"])
                        / variant
                        / f"seed{seed:03d}"
                        / "run001"
                    ),
                ],
                "render_status": "not_run_gpu_unavailable",
                "render_validation_passed": None,
                "eligible_for_calibration": False,
                "eligible_for_testing": False,
            }
        )
    counts = Counter(scene["variant"] for scene in scenes)
    return {
        "schema_version": "action-differentiating-scene-manifest-v1",
        "experiment_id": config["experiment_id"],
        "status": "prepared_not_rendered",
        "purpose": config["purpose"],
        "scene_count": len(scenes),
        "variant_counts": dict(sorted(counts.items())),
        "seeds": seeds,
        "reserved_test_seeds_used": False,
        "required_observations": config["required_observations"],
        "candidate_actions": config["candidate_actions"],
        "render_acceptance": config["render_acceptance"],
        "future_capture_policy": config["future_capture"],
        "scenes": scenes,
        "gpu_used": False,
        "gpu_memory_gib": 0.0,
        "isaac_sim_launched": False,
        "training_performed": False,
        "calibration_performed": False,
        "testing_performed": False,
        "valid_for_final_evaluation": False,
        "limitations": [
            "Analytic layouts are only initializations.",
            "No scene is accepted until objective rendered masks pass its causal action gate.",
            "Cover removal is represented as a required future interaction but is not executed.",
            "This 12-scene batch is a geometry smoke plan, not calibration or testing.",
        ],
    }


def run(config_path: Path) -> Path:
    config = load_json(config_path)
    manifest = build_scene_manifest(config)
    output_root = resolve_path(config["output_root"])
    output_root.mkdir(parents=True, exist_ok=True)
    output_path = output_root / "manifest.json"
    output_path.write_text(
        json.dumps(manifest, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"ACTION_SCENE_MANIFEST={output_path}")
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    args = parser.parse_args()
    run(args.config.resolve())


if __name__ == "__main__":
    main()
