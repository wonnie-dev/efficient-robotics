#!/usr/bin/env python3
"""Run pretrained GroundingDINO, SAM2, and Qwen on V15 calibration views."""

from __future__ import annotations

import argparse
import copy
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from run_covered_action_calibration_perception import (  # noqa: E402
    SOURCE_CONFIG,
    load_json,
    run_stage,
    sha256,
    write_json,
)
from run_icra_v15_calibration_capture_batch import (  # noqa: E402
    assignments,
    completed,
)


CAPTURE_CONFIG = ROOT / "configs/research/icra_v15_calibration_36episode.json"
CAPTURE_ROOT = ROOT / "outputs/live_pipeline/icra_v15_calibration_36episode"
OUTPUT_ROOT = ROOT / "outputs/calibration/icra_v15_calibration_36episode_perception"
SMOKE_ROOT = ROOT / "outputs/calibration/icra_v15_calibration_perception_smoke"
MODEL_PATH = Path(
    os.environ.get(
        "EFFICIENT_ROBOTICS_QWEN_MODEL",
        ROOT / "models/Qwen3-VL-8B-Instruct",
    )
)
SAM2_SOURCE_ROOT = Path(
    os.environ.get(
        "EFFICIENT_ROBOTICS_SAM2_ROOT",
        ROOT / "third_party/Grounded-SAM-2",
    )
)
OBSERVATION_FILES = (
    "rgb.png",
    "depth_m.npy",
    "camera_calibration.json",
    "instance_ids.npy",
    "instance_labels.json",
)


def views_for(row: dict[str, Any]) -> tuple[str, ...]:
    if row["collector"] == "remove_cover_counterfactual":
        return ("center", "post_remove", "close_high", "right")
    return ("center", "close_high", "right")


def select_sample_shard(
    samples: list[dict[str, Any]],
    shard_index: int,
    shard_count: int,
) -> list[dict[str, Any]]:
    """Return a deterministic modulo shard without duplicating samples."""
    if shard_count <= 0:
        raise ValueError("shard_count must be positive")
    if shard_index < 0 or shard_index >= shard_count:
        raise ValueError("shard_index must be in [0, shard_count)")
    return [
        sample
        for index, sample in enumerate(samples)
        if index % shard_count == shard_index
    ]


def build_config(
    output_root: Path,
    sample_limit: int | None = None,
    *,
    capture_config_path: Path = CAPTURE_CONFIG,
    capture_root: Path = CAPTURE_ROOT,
    shard_index: int | None = None,
    shard_count: int | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    capture = load_json(capture_config_path)
    data_split = str(capture.get("data_split", "calibration"))
    if data_split not in {"calibration", "test"}:
        raise ValueError(f"Unsupported data split: {data_split}")
    is_test = data_split == "test"
    source = load_json(SOURCE_CONFIG)
    samples: list[dict[str, Any]] = []
    episodes: list[dict[str, Any]] = []
    for row in assignments(capture):
        result_path = completed(capture_root, row)
        if result_path is None:
            raise FileNotFoundError(
                f"No accepted V15 capture for seed {row['seed']}"
            )
        run_dir = result_path.parent.resolve()
        episode = {
            "seed": int(row["seed"]),
            "family": row["family"],
            "scene_variant": row["scene_variant"],
            "collector": row["collector"],
            "initial_task_state": row.get(
                "initial_task_state",
                "covered"
                if row["collector"] == "remove_cover_counterfactual"
                else "open",
            ),
            "resolving_view": row["resolving_view"],
            "blocked_view": row["blocked_view"],
            "expected_membership": row["expected_membership"],
            "source_result": {
                "path": str(result_path.resolve()),
                "sha256": sha256(result_path),
            },
            "observations": {},
        }
        for view in views_for(row):
            observation = run_dir / "observations" / view
            for name in OBSERVATION_FILES:
                path = observation / name
                if not path.is_file() or path.stat().st_size == 0:
                    raise FileNotFoundError(path)
            sample_id = f"seed{int(row['seed']):03d}_{view}"
            samples.append(
                {
                    "sample_id": sample_id,
                    "observation_dir": str(observation.resolve()),
                    "split": data_split,
                    "seed": int(row["seed"]),
                    "view_id": view,
                }
            )
            episode["observations"][view] = {
                name: {
                    "path": str((observation / name).resolve()),
                    "sha256": sha256(observation / name),
                }
                for name in OBSERVATION_FILES
            }
        episodes.append(episode)

    expected_sample_count = sum(len(views_for(row)) for row in assignments(capture))
    if len(samples) != expected_sample_count:
        raise RuntimeError(
            f"Expected {expected_sample_count} calibration observations, "
            f"found {len(samples)}"
        )
    if (shard_index is None) != (shard_count is None):
        raise ValueError("shard_index and shard_count must be specified together")
    if shard_count is not None:
        samples = select_sample_shard(samples, shard_index, shard_count)
    if sample_limit is not None:
        if sample_limit <= 0:
            raise ValueError("--sample-limit must be positive")
        if shard_count is not None:
            raise ValueError("sample_limit and sharding are mutually exclusive")
        samples = samples[:sample_limit]

    config = copy.deepcopy(source)
    config.update(
        {
            "experiment_id": f"{capture['schema_version']}_perception",
            "output_root": str(output_root.resolve()),
            "motion_result": None,
            "samples": samples,
            "training_performed": False,
            "calibration_performed": False,
            "testing_performed": is_test,
            "reserved_test_seeds_used": is_test,
            "valid_for_final_evaluation": is_test,
        }
    )
    config["task"].update(
        {
            "instruction": "Find the red mug with the white rectangular logo.",
            "target_description": "red mug with a white rectangular logo",
            "qwen_direct_prompt": (
                "Find the red mug with the white rectangular logo. Report its "
                "tight visible bounding box using relative image coordinates "
                "from 0 to 1000. Output exactly one JSON list containing the "
                "bbox and label target, or [] when it cannot be located reliably."
            ),
        }
    )
    config["evaluation"]["scene_labels_not_used_for_inference"] = {}
    config["limitations"] = [
        "Pretrained inference only; no training or fine-tuning.",
        "Scene-family and simulator labels are excluded from model inference.",
        "Raw Qwen logits are uncalibrated and are not success probabilities.",
        "Calibration fitting and all reserved testing occur after this cache is fixed.",
    ]
    manifest = {
        "schema_version": "icra-v15-observation-manifest-v1",
        "data_split": data_split,
        "episodes": episodes,
        "full_sample_count": expected_sample_count,
        "inference_sample_count": len(samples),
        "shard_index": shard_index,
        "shard_count": shard_count,
        "labels_excluded_from_model_input": True,
        "training_performed": False,
        "calibration_performed": False,
        "testing_performed": is_test,
        "reserved_test_seeds_used": is_test,
        "valid_for_final_evaluation": is_test,
    }
    return config, manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--physical-gpu", type=int, required=True, choices=range(6))
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--sample-limit", type=int)
    parser.add_argument("--shard-index", type=int)
    parser.add_argument("--shard-count", type=int)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--capture-config", type=Path, default=CAPTURE_CONFIG)
    parser.add_argument("--capture-root", type=Path, default=CAPTURE_ROOT)
    args = parser.parse_args()
    gpu = str(args.physical_gpu)
    if os.environ.get("CUDA_VISIBLE_DEVICES") != gpu:
        raise RuntimeError("CUDA_VISIBLE_DEVICES must match --physical-gpu")
    if os.environ.get("PHYSICAL_GPU") != gpu:
        raise RuntimeError("PHYSICAL_GPU must match --physical-gpu")
    if not MODEL_PATH.is_dir():
        raise FileNotFoundError(MODEL_PATH)
    if not (SAM2_SOURCE_ROOT / "sam2").is_dir():
        raise FileNotFoundError(SAM2_SOURCE_ROOT / "sam2")

    output_root = (
        args.output_root.resolve()
        if args.output_root is not None
        else (SMOKE_ROOT if args.sample_limit is not None else OUTPUT_ROOT)
    )
    config, manifest = build_config(
        output_root,
        args.sample_limit,
        capture_config_path=args.capture_config.resolve(),
        capture_root=args.capture_root.resolve(),
        shard_index=args.shard_index,
        shard_count=args.shard_count,
    )
    output_root.mkdir(parents=True, exist_ok=True)
    config_path = output_root / "inference_config.json"
    manifest_path = output_root / "observation_manifest.json"
    write_json(config_path, config)
    write_json(manifest_path, manifest)
    if args.prepare_only:
        print(f"ICRA_V15_PERCEPTION_PREPARED={config_path}")
        return

    environment = dict(os.environ)
    environment.update(
        {
            "CUDA_VISIBLE_DEVICES": gpu,
            "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
            "NVIDIA_VISIBLE_DEVICES": gpu,
            "PHYSICAL_GPU": gpu,
            "EFFICIENT_ROBOTICS_QWEN_MODEL": str(MODEL_PATH),
        }
    )
    existing_pythonpath = environment.get("PYTHONPATH", "")
    environment["PYTHONPATH"] = str(SAM2_SOURCE_ROOT) + (
        os.pathsep + existing_pythonpath if existing_pythonpath else ""
    )
    for name in ("RANK", "LOCAL_RANK", "WORLD_SIZE", "MASTER_ADDR", "MASTER_PORT"):
        environment.pop(name, None)

    sample_count = len(config["samples"])
    is_test = bool(config.get("testing_performed"))
    state: dict[str, Any] = {
        "schema_version": "icra-v15-perception-status-v1",
        "status": "running",
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "physical_gpu": args.physical_gpu,
        "single_model_instance_at_a_time": True,
        "batch_size": 1,
        "sample_count": sample_count,
        "training_performed": False,
        "calibration_performed": False,
        "testing_performed": is_test,
        "reserved_test_seeds_used": is_test,
        "valid_for_final_evaluation": is_test,
        "stages": [],
    }
    write_json(output_root / "status.json", state)
    python = sys.executable
    commands = (
        ("gdino_detect", [python, str(ROOT / "scripts/run_perception_grounding_pilot.py"), "gdino_detect", "--config", str(config_path)], max(900.0, sample_count * 15.0)),
        ("sam2_segment", [python, str(ROOT / "scripts/run_perception_grounding_pilot.py"), "sam2_segment", "--config", str(config_path)], max(900.0, sample_count * 15.0)),
        ("export_qwen_inputs", [python, str(ROOT / "scripts/export_grounded_sam2_qwen_inputs.py"), "--config", str(config_path)], max(300.0, sample_count * 3.0)),
        ("qwen_ranking", [python, str(ROOT / "scripts/run_grounded_proposal_qwen_ranking.py"), "--config", str(config_path), "--model-path", str(MODEL_PATH), "--max-pixels", "401408"], max(1800.0, sample_count * 40.0)),
    )
    started = time.perf_counter()
    try:
        for name, command, timeout in commands:
            state["stages"].append(
                run_stage(name, command, environment, output_root, timeout)
            )
            write_json(output_root / "status.json", state)
        ranking_count = len(
            list((output_root / "grounded_sam2_qwen_rankings").glob("*/result.json"))
        )
        if ranking_count != sample_count:
            raise RuntimeError(
                f"Expected {sample_count} ranking files, found {ranking_count}"
            )
        state.update(
            {
                "status": "completed",
                "completed_at_utc": datetime.now(timezone.utc).isoformat(),
                "runtime_seconds": time.perf_counter() - started,
                "ranking_result_count": ranking_count,
            }
        )
        write_json(output_root / "status.json", state)
        write_json(output_root / "COMPLETED.json", state)
        print(f"ICRA_V15_PERCEPTION_COMPLETED={output_root / 'COMPLETED.json'}")
    except Exception as error:
        state.update(
            {
                "status": "failed",
                "completed_at_utc": datetime.now(timezone.utc).isoformat(),
                "runtime_seconds": time.perf_counter() - started,
                "error_type": type(error).__name__,
                "error": str(error),
            }
        )
        write_json(output_root / "status.json", state)
        write_json(output_root / "FAILED.json", state)
        raise


if __name__ == "__main__":
    main()
