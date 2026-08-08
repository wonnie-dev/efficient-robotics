"""Validate the deterministic open-container scene for integration debugging.

This post-capture check may read simulator instance labels.  Its output is
debug evidence about scene construction only and is never consumed by the
pre-action planner.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OBSERVATIONS = (
    ROOT
    / "outputs"
    / "seeded_pilot"
    / "benchmark_seed000"
    / "observations"
)
DEFAULT_LAYOUT = DEFAULT_OBSERVATIONS.parent / "scene_layout.json"
DEFAULT_OUTPUT = (
    ROOT
    / "outputs"
    / "first_belief_mpc_integration"
    / "scene_validation.json"
)


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--observations", type=Path, default=DEFAULT_OBSERVATIONS)
    parser.add_argument("--layout", type=Path, default=DEFAULT_LAYOUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    layout = load_json(args.layout)
    objects = {
        view: load_json(args.observations / view / "objects.json")
        for view in ("center", "left", "right")
    }
    target_pixels = {
        view: int(payload["target_red"]["pixel_count"])
        for view, payload in objects.items()
    }
    rear_pixels = {
        view: int(payload["rear_red_candidate"]["pixel_count"])
        for view, payload in objects.items()
    }
    checks = {
        "two_red_target_candidates_exist": all(
            candidate in objects["center"]
            for candidate in ("target_red", "rear_red_candidate")
        ),
        "target_relation_is_inside": (
            layout["relations_preserved"]["target_red"] == "inside"
        ),
        "rear_candidate_relation_is_behind": (
            layout["relations_preserved"]["rear_red_candidate"]
            == "behind_container"
        ),
        "center_target_is_partially_visible": (
            0 < target_pixels["center"] < 500
        ),
        "left_hides_inside_target": target_pixels["left"] == 0,
        "right_reveals_more_inside_target": (
            target_pixels["right"] >= target_pixels["center"]
            and target_pixels["right"] >= 25
        ),
        "candidate_views_have_different_outcomes": (
            target_pixels["left"] != target_pixels["right"]
        ),
    }
    result = {
        "schema_version": "first-belief-mpc-scene-validation-v1",
        "status": "completed" if all(checks.values()) else "failed",
        "scenario": "open_container_two_candidate_active_reobservation",
        "seed": layout["seed"],
        "target_pixel_counts": target_pixels,
        "rear_candidate_pixel_counts": rear_pixels,
        "checks": checks,
        "provenance": {
            "simulator_instance_labels_used": True,
            "usage": "post_capture_scene_debug_validation_only",
            "planner_input": False,
            "valid_for_final_evaluation": False,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    print(f"SCENE_VALIDATION={result['status']}")
    print(f"WROTE={args.output}")
    raise SystemExit(0 if result["status"] == "completed" else 2)


if __name__ == "__main__":
    main()
