#!/usr/bin/env python3
"""Run sequential pretrained perception on covered-action calibration views."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
PROFILES = {
    "seed221_222": {
        "capture_root": ROOT
        / "outputs/live_pipeline/covered_action_calibration_smoke",
        "output_root": ROOT
        / "outputs/calibration_pilot/covered_action_differentiating_seed221_222_gpu0",
        "experiment_id": "covered_action_differentiating_seed221_222",
        "episodes": (
            (221, "covered_then_close_high_only"),
            (222, "covered_then_right_only"),
        ),
    },
    "center_ambiguous_seed224_225": {
        "capture_root": ROOT
        / "outputs/live_pipeline/covered_action_center_ambiguous_calibration",
        "output_root": ROOT
        / "outputs/calibration_pilot/covered_action_center_ambiguous_seed224_225_gpu0",
        "experiment_id": "covered_action_center_ambiguous_seed224_225",
        "episodes": (
            (224, "covered_center_ambiguous_then_close_high_only"),
            (225, "covered_center_ambiguous_then_right_only"),
        ),
    },
    "center_ambiguous_seed226_229": {
        "capture_root": ROOT
        / "outputs/live_pipeline/covered_action_center_ambiguous_calibration",
        "output_root": ROOT
        / "outputs/calibration_pilot/covered_action_center_ambiguous_seed226_229_gpu0",
        "experiment_id": "covered_action_center_ambiguous_seed226_229",
        "episodes": (
            (226, "covered_center_ambiguous_then_close_high_only"),
            (227, "covered_center_ambiguous_then_right_only"),
            (228, "covered_center_ambiguous_then_close_high_only"),
            (229, "covered_center_ambiguous_then_right_only"),
        ),
    },
    "center_ambiguous_semantic_seed230": {
        "capture_root": ROOT
        / "outputs/live_pipeline/covered_action_center_ambiguous_calibration",
        "output_root": ROOT
        / "outputs/calibration_pilot/covered_action_center_ambiguous_semantic_seed230_gpu0",
        "experiment_id": "covered_action_center_ambiguous_semantic_seed230",
        "episodes": (
            (230, "covered_center_ambiguous_then_close_high_only"),
        ),
    },
    "center_ambiguous_semantic_seed230_234": {
        "capture_root": ROOT
        / "outputs/live_pipeline/covered_action_center_ambiguous_calibration",
        "output_root": ROOT
        / "outputs/calibration_pilot/covered_action_center_ambiguous_semantic_seed230_234_gpu0",
        "experiment_id": "covered_action_center_ambiguous_semantic_seed230_234",
        "episodes": (
            (230, "covered_center_ambiguous_then_close_high_only"),
            (232, "covered_center_ambiguous_then_close_high_only"),
            (234, "covered_center_ambiguous_then_close_high_only"),
        ),
    },
    "center_ambiguous_right_seed231_233": {
        "capture_root": ROOT
        / "outputs/live_pipeline/covered_action_center_ambiguous_calibration",
        "output_root": ROOT
        / "outputs/calibration_pilot/covered_action_center_ambiguous_right_seed231_233_gpu0",
        "experiment_id": "covered_action_center_ambiguous_right_seed231_233",
        "episodes": (
            (231, "covered_center_ambiguous_then_right_only"),
            (233, "covered_center_ambiguous_then_right_only"),
        ),
    },
    "center_ambiguous_right_seed231_235": {
        "capture_root": ROOT
        / "outputs/live_pipeline/covered_action_center_ambiguous_calibration",
        "output_root": ROOT
        / "outputs/calibration_pilot/covered_action_center_ambiguous_right_seed231_235_gpu0",
        "experiment_id": "covered_action_center_ambiguous_right_seed231_235",
        "episodes": (
            (231, "covered_center_ambiguous_then_right_only"),
            (233, "covered_center_ambiguous_then_right_only"),
            (235, "covered_center_ambiguous_then_right_only"),
        ),
    },
    "center_ambiguous_balanced_seed236_237": {
        "capture_root": ROOT
        / "outputs/live_pipeline/covered_action_center_ambiguous_calibration",
        "output_root": ROOT
        / "outputs/calibration_pilot/covered_action_center_ambiguous_balanced_seed236_237_gpu0",
        "experiment_id": "covered_action_center_ambiguous_balanced_seed236_237",
        "episodes": (
            (236, "covered_center_ambiguous_then_close_high_only"),
            (237, "covered_center_ambiguous_then_right_only"),
        ),
    },
    "center_ambiguous_balanced_seed238_239": {
        "capture_root": ROOT
        / "outputs/live_pipeline/covered_action_center_ambiguous_calibration",
        "output_root": ROOT
        / "outputs/calibration_pilot/covered_action_center_ambiguous_balanced_seed238_239_gpu0",
        "experiment_id": "covered_action_center_ambiguous_balanced_seed238_239",
        "episodes": (
            (238, "covered_center_ambiguous_then_close_high_only"),
            (239, "covered_center_ambiguous_then_right_only"),
        ),
    },
}
SOURCE_CONFIG = (
    ROOT
    / "outputs/perception_grounding_pilot"
    / "action_differentiating_neutral_seed185_196/perception_config.json"
)
VIEWS = ("center", "post_remove", "close_high", "right")
OBSERVATION_FILES = ("rgb.png", "depth_m.npy", "camera_calibration.json")


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def calibration_run(seed: int, variant: str, capture_root: Path) -> Path:
    seed_root = capture_root / variant / f"seed{seed:03d}"
    candidates = sorted(seed_root.glob("run*/calibration_validation.json"))
    for validation_path in reversed(candidates):
        validation = load_json(validation_path)
        if (
            validation.get("status") == "passed"
            and validation.get("calibration_performed") is True
            and validation.get("testing_performed") is False
            and validation.get("valid_for_final_evaluation") is False
        ):
            run_dir = validation_path.parent.resolve()
            for view in VIEWS:
                observation = run_dir / "observations" / view
                for name in OBSERVATION_FILES:
                    if not (observation / name).is_file():
                        raise FileNotFoundError(observation / name)
            return run_dir
    raise FileNotFoundError(
        f"No passed calibration capture for seed {seed} variant {variant}"
    )


def build_config(
    output_root: Path, profile: dict[str, Any]
) -> tuple[dict[str, Any], dict[str, Any]]:
    source = load_json(SOURCE_CONFIG)
    episodes = []
    samples = []
    for seed, variant in profile["episodes"]:
        run_dir = calibration_run(seed, variant, profile["capture_root"])
        validation_path = run_dir / "calibration_validation.json"
        episodes.append(
            {
                "seed": seed,
                "variant": variant,
                "run_dir": str(run_dir),
                "validation": {
                    "path": str(validation_path),
                    "sha256": sha256(validation_path),
                },
            }
        )
        for view in VIEWS:
            observation_dir = run_dir / "observations" / view
            samples.append(
                {
                    "sample_id": f"seed{seed:03d}_{view}",
                    "observation_dir": str(observation_dir),
                    "split": "calibration",
                    "seed": seed,
                    "view_id": view,
                    "scenario_variant": variant,
                }
            )
            episodes[-1].setdefault("observations", {})[view] = {
                name: {
                    "path": str(observation_dir / name),
                    "sha256": sha256(observation_dir / name),
                }
                for name in OBSERVATION_FILES
            }
    config = copy.deepcopy(source)
    config.update(
        {
            "experiment_id": profile["experiment_id"],
            "output_root": str(output_root.resolve()),
            "motion_result": None,
            "samples": samples,
            "training_performed": False,
            "calibration_performed": True,
            "testing_performed": False,
            "reserved_test_seeds_used": False,
            "valid_for_final_evaluation": False,
        }
    )
    config["limitations"] = [
        "Pretrained inference only; no training or fine-tuning.",
        "Calibration seeds only; reserved test seeds 240-259 are unopened.",
        "Scores are raw and uncalibrated and are not success probabilities.",
        "Simulator labels, masks, and depth are excluded from model inference.",
    ]
    manifest = {
        "schema_version": "covered-action-calibration-observation-manifest-v1",
        "episodes": episodes,
        "sample_count": len(samples),
        "views": list(VIEWS),
        "training_performed": False,
        "calibration_performed": True,
        "testing_performed": False,
        "reserved_test_seeds_used": False,
        "valid_for_final_evaluation": False,
    }
    return config, manifest


def run_stage(
    name: str,
    command: list[str],
    environment: dict[str, str],
    output_root: Path,
    timeout_seconds: float,
) -> dict[str, Any]:
    log_root = output_root / "logs"
    log_root.mkdir(parents=True, exist_ok=True)
    stdout_path = log_root / f"{name}.stdout.log"
    stderr_path = log_root / f"{name}.stderr.log"
    started = time.perf_counter()
    with stdout_path.open("w", encoding="utf-8") as stdout, stderr_path.open(
        "w", encoding="utf-8"
    ) as stderr:
        completed = subprocess.run(
            command,
            cwd=ROOT,
            env=environment,
            stdout=stdout,
            stderr=stderr,
            text=True,
            timeout=timeout_seconds,
            check=False,
        )
    result = {
        "name": name,
        "command": command,
        "returncode": completed.returncode,
        "runtime_seconds": time.perf_counter() - started,
        "stdout": str(stdout_path.resolve()),
        "stderr": str(stderr_path.resolve()),
    }
    if completed.returncode != 0:
        raise RuntimeError(f"Calibration perception stage failed: {result}")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--physical-gpu", type=int, required=True)
    parser.add_argument(
        "--profile", choices=tuple(PROFILES), default="seed221_222"
    )
    args = parser.parse_args()
    gpu = str(args.physical_gpu)
    if os.environ.get("CUDA_VISIBLE_DEVICES") != gpu:
        raise RuntimeError("CUDA_VISIBLE_DEVICES must match --physical-gpu")
    if os.environ.get("PHYSICAL_GPU") != gpu:
        raise RuntimeError("PHYSICAL_GPU must match --physical-gpu")

    environment = dict(os.environ)
    environment.update(
        {
            "CUDA_VISIBLE_DEVICES": gpu,
            "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
            "NVIDIA_VISIBLE_DEVICES": gpu,
            "PHYSICAL_GPU": gpu,
            "EFFICIENT_ROBOTICS_QWEN_MODEL": (
                environment.get(
                    "EFFICIENT_ROBOTICS_QWEN_MODEL",
                    str(ROOT / "models/Qwen3-VL-8B-Instruct"),
                )
            ),
        }
    )
    for name in ("RANK", "LOCAL_RANK", "WORLD_SIZE", "MASTER_ADDR", "MASTER_PORT"):
        environment.pop(name, None)

    profile = PROFILES[args.profile]
    output_root = profile["output_root"]
    config, manifest = build_config(output_root, profile)
    output_root.mkdir(parents=True, exist_ok=True)
    config_path = output_root / "inference_config.json"
    manifest_path = output_root / "observation_manifest.json"
    write_json(config_path, config)
    write_json(manifest_path, manifest)
    completed_path = output_root / "COMPLETED.json"
    if completed_path.is_file():
        completed = load_json(completed_path)
        if (
            completed.get("inference_config_sha256") == sha256(config_path)
            and completed.get("observation_manifest_sha256")
            == sha256(manifest_path)
            and completed.get("status") == "completed"
        ):
            print(f"CALIBRATION_PERCEPTION_CACHE_HIT={completed_path}")
            return

    state: dict[str, Any] = {
        "schema_version": "covered-action-calibration-perception-status-v1",
        "status": "running",
        "physical_gpu": args.physical_gpu,
        "single_model_instance_at_a_time": True,
        "sample_count": len(config["samples"]),
        "inference_config": str(config_path.resolve()),
        "inference_config_sha256": sha256(config_path),
        "observation_manifest": str(manifest_path.resolve()),
        "observation_manifest_sha256": sha256(manifest_path),
        "training_performed": False,
        "calibration_performed": True,
        "testing_performed": False,
        "reserved_test_seeds_used": False,
        "valid_for_final_evaluation": False,
        "stages": [],
    }
    write_json(output_root / "status.json", state)
    python = sys.executable
    commands = (
        (
            "gdino_detect",
            [
                python,
                str(ROOT / "scripts/run_perception_grounding_pilot.py"),
                "gdino_detect",
                "--config",
                str(config_path),
            ],
            1800.0,
        ),
        (
            "sam2_segment",
            [
                python,
                str(ROOT / "scripts/run_perception_grounding_pilot.py"),
                "sam2_segment",
                "--config",
                str(config_path),
            ],
            1800.0,
        ),
        (
            "export_qwen_inputs",
            [
                python,
                str(ROOT / "scripts/export_grounded_sam2_qwen_inputs.py"),
                "--config",
                str(config_path),
            ],
            600.0,
        ),
        (
            "qwen_ranking",
            [
                python,
                str(ROOT / "scripts/run_grounded_proposal_qwen_ranking.py"),
                "--config",
                str(config_path),
                "--model-path",
                environment["EFFICIENT_ROBOTICS_QWEN_MODEL"],
                "--max-pixels",
                "401408",
            ],
            1800.0,
        ),
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
        if ranking_count != len(config["samples"]):
            raise RuntimeError(
                f"Expected {len(config['samples'])} rankings, found {ranking_count}"
            )
        state.update(
            {
                "status": "completed",
                "runtime_seconds": time.perf_counter() - started,
                "ranking_result_count": ranking_count,
            }
        )
        write_json(output_root / "status.json", state)
        write_json(completed_path, state)
        print(f"CALIBRATION_PERCEPTION_COMPLETED={completed_path}")
    except Exception as error:
        state.update(
            {
                "status": "failed",
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
