#!/usr/bin/env python3
"""Revalidate saved post-action RGB-D with relation-neutral perception prompts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from audit_saved_learned_relation import audit_relation
from run_live_single_gpu_pipeline import write_json_atomic
from run_negative_evidence_live_smoke import learned_post_remove_outcome
from run_remove_cover_live_smoke import learned_post_remove_localization
from run_single_gpu_pilot import require_single_gpu_policy


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = (
    ROOT / "configs" / "research" / "negative_evidence_live_development.json"
)


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def resolve_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def process_view(
    source_run: Path,
    output_dir: Path,
    config: dict,
    view: str,
    index: int,
) -> tuple[dict, dict]:
    observation_dir = source_run / "observations" / view
    perception = learned_post_remove_localization(
        output_dir,
        observation_dir,
        observation_index=index,
        view=view,
        task_overrides=config["perception_task_overrides"],
    )
    relation = audit_relation(
        observation_dir,
        Path(perception["ranking"]["input_path"]),
        Path(perception["ranking_path"]),
        resolve_path(config["rgbd_relation_config"]),
    )
    write_json_atomic(output_dir / f"{view}_learned_relation.json", relation)
    return perception, relation


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source_run", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    args = parser.parse_args()
    require_single_gpu_policy()
    source_run = args.source_run.resolve()
    output_dir = args.output_dir.resolve()
    config = load_json(args.config.resolve())
    output_dir.mkdir(parents=True, exist_ok=True)

    post_perception, post_relation = process_view(
        source_run, output_dir, config, "post_remove", 1
    )
    right_perception, right_relation = process_view(
        source_run, output_dir, config, "right", 2
    )
    post_symbol = learned_post_remove_outcome(post_relation)
    right_qwen = right_relation["qwen_relation_top_label"]
    right_rgbd = right_relation["rgbd_relation"][
        "membership_world_evidence"
    ]["label"]
    success = bool(
        post_symbol == "empty_container"
        and right_qwen == "outside"
        and right_rgbd == "outside"
    )
    result = {
        "schema_version": "saved-negative-evidence-neutral-prompt-audit-v1",
        "status": "completed" if success else "failed",
        "source_run": str(source_run),
        "instruction": config["perception_task_overrides"]["instruction"],
        "relation_words_removed_from_instruction": True,
        "post_remove": {
            "selected_candidate_id": post_perception["selected_candidate_id"],
            "qwen_relation": post_relation["qwen_relation_top_label"],
            "rgbd_relation": post_relation["rgbd_relation"][
                "membership_world_evidence"
            ]["label"],
            "belief_symbol": post_symbol,
        },
        "right": {
            "selected_candidate_id": right_perception["selected_candidate_id"],
            "qwen_relation": right_qwen,
            "rgbd_relation": right_rgbd,
        },
        "simulator_ground_truth_used_for_prediction": False,
        "training_performed": False,
        "valid_for_final_evaluation": False,
        "audit_scope": "development_protocol_regression_before_freeze",
    }
    write_json_atomic(output_dir / "result.json", result)
    print(json.dumps(result, indent=2))
    if not success:
        raise RuntimeError(f"Neutral-prompt perception audit failed: {result}")


if __name__ == "__main__":
    main()
