"""Export all seeded benchmark views into one pilot-only VLM manifest."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from export_vlm_dataset import export_view, relative


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "outputs" / "seeded_pilot" / "vlm_dataset"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed-start", type=int, default=0)
    parser.add_argument("--seed-stop", type=int, default=10)
    parser.add_argument(
        "--views",
        nargs="+",
        default=["left", "center", "right", "close_high"],
    )
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    if args.seed_start < 0 or args.seed_stop <= args.seed_start:
        raise ValueError("Expected 0 <= seed-start < seed-stop")

    output_root = args.output_root.resolve()
    manifest = {
        "schema_version": "vlm-dataset-manifest-v1",
        "purpose": "seeded_pipeline_validation_only_not_final_evaluation",
        "samples": [],
    }
    for seed in range(args.seed_start, args.seed_stop):
        episode_id = f"benchmark_seed{seed:03d}"
        observation_root = (
            ROOT / "outputs" / "seeded_pilot" / episode_id / "observations"
        )
        for view in args.views:
            input_path, ground_truth_path = export_view(
                view,
                output_root,
                observation_root=observation_root,
                episode_id=episode_id,
            )
            manifest["samples"].append(
                {
                    "input": relative(input_path),
                    "ground_truth": relative(ground_truth_path),
                    "split": "pilot_debug_only",
                }
            )
    manifest_path = output_root / "manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        json.dumps(manifest, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"EPISODES={args.seed_stop - args.seed_start}")
    print(f"SAMPLES={len(manifest['samples'])}")
    print(f"MANIFEST={manifest_path}")


if __name__ == "__main__":
    main()
