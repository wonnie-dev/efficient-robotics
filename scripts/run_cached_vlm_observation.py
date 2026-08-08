"""Run or replay one batch-size-one Qwen observation through the pilot cache."""

import argparse
import json
from pathlib import Path

from run_single_gpu_pilot import (
    DEFAULT_CACHE_ROOT,
    DEFAULT_MODEL,
    cached_inference,
    output_belief,
    require_single_gpu_policy,
    select_action,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)
    parser.add_argument("--cache-root", type=Path, default=DEFAULT_CACHE_ROOT)
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--max-pixels", type=int, default=512 * 28 * 28)
    parser.add_argument("--allow-cache-miss-inference", action="store_true")
    parser.add_argument("--summary-output", type=Path)
    args = parser.parse_args()

    require_single_gpu_policy()
    result = cached_inference(
        args.input.resolve(),
        args.cache_root.resolve(),
        args.model_path.resolve(),
        args.max_pixels,
        args.allow_cache_miss_inference,
    )
    belief = output_belief(result["output"])
    predicted_target = max(belief["target"], key=belief["target"].get)
    model_input = json.loads(args.input.read_text(encoding="utf-8"))
    view_id = model_input["view_id"]
    pilot_action = select_action(
        belief,
        current_view=view_id,
        available_views={view_id},
        visited_views={view_id},
    )
    summary = {
        "schema_version": "cached-vlm-observation-summary-v1",
        "sample_id": result["output"]["sample_id"],
        "predicted_target": predicted_target,
        "target_belief": belief["target"],
        "relations": belief["relations"],
        "pilot_action": pilot_action,
        "action_execution_authorized": False,
        "action_execution_blocker": (
            "RG6 grasp/contact physics and real-robot calibration are not validated"
        ),
        "cache_key": result["cache_key"],
        "cache_dir": result["cache_dir"],
        "cache_hit": result["cache_hit"],
        "cache_source": result["cache_source"],
        "metrics": result["metrics"],
        "training_performed": False,
        "calibration_performed": False,
        "valid_for_final_evaluation": False,
    }
    output_path = args.summary_output or args.input.with_name(
        "cached_qwen_summary.json"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(summary, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"PREDICTED_TARGET={predicted_target}")
    print(f"CACHE_SOURCE={result['cache_source']}")
    print(f"CACHE_DIR={result['cache_dir']}")
    print(f"SUMMARY={output_path}")


if __name__ == "__main__":
    main()
