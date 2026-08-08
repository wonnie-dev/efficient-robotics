#!/usr/bin/env python3
"""Prepare a zero-copy Qwen replay with a relation-neutral instruction."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
INSTRUCTION = "Find and pick up the red mug with the white logo."
TARGET_DESCRIPTION = "the red mug with the white logo"
DIRECT_PROMPT = (
    "Find the single red mug with the white logo. If it cannot be located "
    "reliably, return []. Return JSON only."
)
RELATION_WORDS = ("inside", "outside", "behind", "near", "covered")


def resolve_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def prepare(source_root: Path, destination_root: Path) -> dict:
    source_root = source_root.resolve()
    destination_root = destination_root.resolve()
    allowed = (ROOT / "outputs").resolve()
    if not destination_root.is_relative_to(allowed):
        raise ValueError(f"Destination must stay below {allowed}")
    lowered = f"{INSTRUCTION} {DIRECT_PROMPT}".lower()
    leaked = [word for word in RELATION_WORDS if word in lowered]
    if leaked:
        raise ValueError(f"Neutral prompt leaks relation labels: {leaked}")

    source_input_root = source_root / "grounded_sam2_qwen_inputs"
    source_manifest = json.loads(
        (source_input_root / "manifest.json").read_text(encoding="utf-8")
    )
    destination_input_root = destination_root / "grounded_sam2_qwen_inputs"
    samples = []
    for item in source_manifest["samples"]:
        source_input = resolve_path(item["input_path"])
        model_input = json.loads(source_input.read_text(encoding="utf-8"))
        model_input["instruction"] = INSTRUCTION
        model_input["target_description"] = TARGET_DESCRIPTION
        destination_input = (
            destination_input_root / item["sample_id"] / "input.json"
        )
        write_json(destination_input, model_input)
        samples.append(
            {
                **item,
                "input_path": str(destination_input),
                "source_input_path": str(source_input),
            }
        )
    write_json(
        destination_input_root / "manifest.json",
        {
            "schema_version": "grounded-sam2-qwen-input-manifest-v1",
            "samples": samples,
            "asset_policy": "zero_copy_assets_reference_source_input_tree",
        },
    )

    source_config = json.loads(
        (source_root / "perception_config.json").read_text(encoding="utf-8")
    )
    source_config["experiment_id"] = (
        f"{source_config['experiment_id']}_neutral_prompt_replay"
    )
    source_config["output_root"] = str(destination_root)
    source_config["task"].update(
        {
            "instruction": INSTRUCTION,
            "target_description": TARGET_DESCRIPTION,
            "qwen_direct_prompt": DIRECT_PROMPT,
        }
    )
    source_config["neutral_prompt_replay"] = {
        "source_root": str(source_root),
        "relation_answer_in_instruction": False,
        "grounding_and_segmentation_reused": True,
        "qwen_outputs_recomputed": True,
    }
    config_path = destination_root / "perception_config.json"
    write_json(config_path, source_config)
    result = {
        "schema_version": "neutral-prompt-qwen-replay-preparation-v1",
        "status": "prepared",
        "source_root": str(source_root),
        "destination_root": str(destination_root),
        "config_path": str(config_path),
        "sample_count": len(samples),
        "instruction": INSTRUCTION,
        "relation_answer_in_instruction": False,
        "training_performed": False,
        "testing_performed": False,
        "reserved_test_seeds_used": False,
    }
    write_json(destination_root / "preparation.json", result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source_root", type=Path)
    parser.add_argument("destination_root", type=Path)
    args = parser.parse_args()
    print(
        json.dumps(
            prepare(args.source_root, args.destination_root), indent=2
        )
    )


if __name__ == "__main__":
    main()
