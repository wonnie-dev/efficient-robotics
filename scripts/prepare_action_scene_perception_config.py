"""Build a learned-perception config from validated causal-view renders."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
BASE_CONFIG = (
    ROOT
    / "configs"
    / "perception"
    / "action_differentiating_scene_pair_seed185_186.json"
)
BATCH_STATUS = (
    ROOT
    / "outputs"
    / "live_pipeline"
    / "action_differentiating_scene_pilot"
    / "batch_seed187_196"
    / "status.json"
)
DEFAULT_OUTPUT = (
    ROOT
    / "outputs"
    / "perception_grounding_pilot"
    / "action_differentiating_seed185_196"
    / "perception_config.json"
)
VIEWS = ("center", "close_high", "right")


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def relative(path: Path) -> str:
    return str(path.resolve().relative_to(ROOT)).replace("\\", "/")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-config", type=Path, default=BASE_CONFIG)
    parser.add_argument("--batch-status", type=Path, default=BATCH_STATUS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    base = load_json(args.base_config.resolve())
    batch = load_json(args.batch_status.resolve())
    if batch.get("status") != "completed":
        raise ValueError("Render batch must be completed")
    config = copy.deepcopy(base)
    config["experiment_id"] = "action_differentiating_seed185_196"
    config["output_root"] = (
        "outputs/perception_grounding_pilot/action_differentiating_seed185_196"
    )
    samples = list(base["samples"])
    scene_labels = dict(
        base["evaluation"]["scene_labels_not_used_for_inference"]
    )
    excluded = []
    for scene in batch["scenes"]:
        seed = int(scene["seed"])
        variant = str(scene["variant"])
        if not scene["eligible_for_calibration"]:
            excluded.append(
                {
                    "seed": seed,
                    "variant": variant,
                    "failure_reasons": scene["failure_reasons"],
                }
            )
            continue
        run_dir = Path(scene["run_dir"]).resolve()
        ground_truth = run_dir / "calibration_ground_truth.json"
        if not ground_truth.is_file():
            raise FileNotFoundError(ground_truth)
        scene_labels[f"seed{seed:03d}"] = variant
        for view in VIEWS:
            observation_dir = run_dir / "observations" / view
            for required in ("rgb.png", "depth_m.npy", "camera_calibration.json"):
                if not (observation_dir / required).is_file():
                    raise FileNotFoundError(observation_dir / required)
            samples.append(
                {
                    "sample_id": f"seed{seed:03d}_{view}",
                    "observation_dir": relative(observation_dir),
                    "seed": seed,
                    "calibration_scene_variant": variant,
                    "calibration_ground_truth_file": relative(ground_truth),
                }
            )
    config["samples"] = samples
    concepts = config["task"]["open_vocabulary_concepts"]
    if "lid or cover" not in concepts:
        concepts.append("lid or cover")
    config["task"]["grounding_dino_prompt"] = ". ".join(concepts) + "."
    config["evaluation"]["scene_labels_not_used_for_inference"] = (
        scene_labels
    )
    config["source_render_batch"] = relative(args.batch_status.resolve())
    config["excluded_render_scenes"] = excluded
    config["episode_count"] = len(scene_labels)
    config["observation_count"] = len(samples)
    config["limitations"] = [
        (
            f"This development calibration collection contains "
            f"{len(scene_labels)} episodes and {len(samples)} saved views."
        ),
        "Qwen choice scores are not final calibrated success probabilities.",
        (
            "Simulator masks and scene labels are used only for post-hoc "
            "evaluation and calibration, never as inference inputs."
        ),
        "These results are not reserved testing or final paper evidence.",
    ]
    config["training_performed"] = False
    config["calibration_performed"] = False
    config["valid_for_final_evaluation"] = False
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "config": str(output),
                "episode_count": config["episode_count"],
                "observation_count": config["observation_count"],
                "excluded_render_scenes": excluded,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
