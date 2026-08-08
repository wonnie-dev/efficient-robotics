"""Build a held-out simulator calibration pilot with no training or test use."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
from pathlib import Path

import numpy as np
from PIL import Image

from calibrated_belief import (
    fit_temperature_grid,
    negative_log_likelihood,
    softmax_temperature,
)
from evaluate_perception_grounding_pilot import mask_iou, semantic_class_masks
from run_live_single_gpu_pipeline import ROOT, write_json_atomic
from run_single_gpu_pilot import configured_physical_gpu, require_single_gpu_policy


PERCEPTION_PYTHON = Path(
    "/data/wonheekoh/venvs/efficient-robotics-perception/bin/python"
)
SOURCE_CONFIG = (
    ROOT
    / "configs"
    / "perception"
    / "scanned_basket_occlusion_two_step_seed000.json"
)
DEFAULT_ROOT = (
    ROOT
    / "outputs"
    / "calibration_pilot"
    / "scanned_basket_seed100_109"
)
CAPTURE_ROOT = (
    ROOT
    / "outputs"
    / "live_pipeline"
    / "factorized_calibration_capture"
)
VIEWS = ("center", "close_high", "right")


def single_gpu_environment() -> dict[str, str]:
    physical_gpu = configured_physical_gpu()
    environment = dict(os.environ)
    environment.update(
        {
            "CUDA_VISIBLE_DEVICES": str(physical_gpu),
            "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
            "NVIDIA_VISIBLE_DEVICES": str(physical_gpu),
            "PHYSICAL_GPU": str(physical_gpu),
        }
    )
    for name in ("RANK", "LOCAL_RANK", "WORLD_SIZE", "MASTER_ADDR", "MASTER_PORT"):
        environment.pop(name, None)
    return environment


def completed_capture(
    path: Path, seed: int, expected_variant: str
) -> bool:
    result_path = path / "smoke_result.json"
    if not result_path.is_file():
        return False
    result = json.loads(result_path.read_text(encoding="utf-8"))
    ground_truth_path = path / "calibration_ground_truth.json"
    household_path = path / "household_scene.json"
    generator_revision_ok = True
    if expected_variant in (
        "behind_ambiguous",
        "behind_boundary_unknown",
    ):
        if not household_path.is_file():
            generator_revision_ok = False
        else:
            household = json.loads(
                household_path.read_text(encoding="utf-8")
            )
            expected_revision = {
                "behind_ambiguous": "behind-ambiguous-ray-jitter-v2",
                "behind_boundary_unknown": "behind-boundary-unknown-v1",
            }[expected_variant]
            generator_revision_ok = household["reference"].get(
                "calibration_generator_revision"
            ) == expected_revision
    objective_behind_ok = False
    if ground_truth_path.is_file():
        ground_truth = json.loads(
            ground_truth_path.read_text(encoding="utf-8")
        )
        objective_behind = ground_truth.get(
            "objective_camera_relative_behind_ground_truth", {}
        )
        objective_behind_ok = all(
            isinstance(objective_behind.get(view), dict)
            and objective_behind[view].get("valid") is True
            and objective_behind[view].get("label")
            in {"yes", "no", "unknown"}
            for view in VIEWS
        )
    return (
        result.get("status") == "completed"
        and result.get("seed") == seed
        and result.get("calibration_scene_variant") == expected_variant
        and result.get("sequence") == ["center", "close_high", "right"]
        and ground_truth_path.is_file()
        and objective_behind_ok
        and generator_revision_ok
        and all(
            (path / "observations" / view / filename).is_file()
            for view in VIEWS
            for filename in (
                "rgb.png",
                "depth_m.npy",
                "camera_calibration.json",
                "instance_ids.npy",
                "instance_labels.json",
                "objective_occlusion.json",
                "objective_reference_occlusion.json",
                "objective_camera_relative_behind.json",
            )
        )
    )


def capture_path(seed: int, variant: str) -> tuple[Path, bool]:
    seed_root = CAPTURE_ROOT / variant / f"seed{seed:03d}"
    seed_root.mkdir(parents=True, exist_ok=True)
    indices = []
    for path in seed_root.glob("run*"):
        try:
            indices.append(int(path.name.removeprefix("run")))
        except ValueError:
            continue
        if completed_capture(path, seed, variant):
            return path, True
    return seed_root / f"run{max(indices, default=0) + 1:03d}", False


def run_logged(
    command: list[str],
    *,
    log_root: Path,
    name: str,
    timeout_seconds: float,
) -> dict:
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
            env=single_gpu_environment(),
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
        "stdout": str(stdout_path),
        "stderr": str(stderr_path),
    }
    if completed.returncode != 0:
        raise RuntimeError(f"Calibration pilot stage failed: {result}")
    return result


def capture_episodes(
    seed_start: int,
    seed_stop: int,
    output_root: Path,
    state: dict,
    *,
    collision_physics_basket: bool,
    ambiguous_seed_start: int | None,
    forced_calibration_scene_variant: str | None,
) -> dict[int, Path]:
    from scanned_basket_scene import calibration_variant_for_seed

    captures = {}
    for seed in range(seed_start, seed_stop):
        variant = (
            forced_calibration_scene_variant
            if forced_calibration_scene_variant is not None
            else "behind_ambiguous"
            if (
                ambiguous_seed_start is not None
                and seed >= ambiguous_seed_start
            )
            else calibration_variant_for_seed(seed)
        )
        path, existing = capture_path(seed, variant)
        if existing:
            result = {
                "seed": seed,
                "status": "completed_existing",
                "capture_dir": str(path),
                "calibration_scene_variant": variant,
                "runtime_seconds": 0.0,
            }
        else:
            command = [
                str(PERCEPTION_PYTHON),
                str(ROOT / "scripts" / "run_actual_view_motion_smoke.py"),
                "--seed",
                str(seed),
                "--scanned-basket-perception-pilot",
                "--calibration-scene-variant",
                variant,
                "--requested-views",
                "close_high",
                "right",
                "--output-dir",
                str(path),
            ]
            if collision_physics_basket:
                command.append("--basket-collision-physics-pilot")
            stage = run_logged(
                command,
                log_root=output_root / "logs",
                name=f"capture_seed{seed:03d}",
                timeout_seconds=420.0,
            )
            if not completed_capture(path, seed, variant):
                raise RuntimeError(f"Capture validation failed for seed {seed}")
            result = {
                "seed": seed,
                "status": "completed",
                "capture_dir": str(path),
                "calibration_scene_variant": variant,
                "runtime_seconds": stage["runtime_seconds"],
                "stdout": stage["stdout"],
                "stderr": stage["stderr"],
            }
        captures[seed] = path
        state["capture_results"].append(result)
        state["last_completed_seed"] = seed
        write_json_atomic(output_root / "status.json", state)
        print(f"CALIBRATION_CAPTURE_COMPLETE=seed{seed:03d}", flush=True)
    return captures


def load_existing_captures(
    manifest_paths: list[Path],
    seed_start: int,
    seed_stop: int,
    state: dict,
) -> dict[int, Path]:
    episodes = {}
    for manifest_path in manifest_paths:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        for item in manifest["episodes"]:
            seed = int(item["seed"])
            if seed in episodes:
                raise ValueError(
                    f"Duplicate seed {seed} across capture manifests"
                )
            episodes[seed] = item
    expected_seeds = set(range(seed_start, seed_stop))
    if set(episodes) != expected_seeds:
        raise ValueError(
            "Existing capture manifest seeds do not match requested range: "
            f"{sorted(episodes)} != {sorted(expected_seeds)}"
        )
    captures = {}
    for seed in range(seed_start, seed_stop):
        item = episodes[seed]
        variant = str(item["variant"])
        path = Path(item["capture_dir"]).resolve()
        if not completed_capture(path, seed, variant):
            raise RuntimeError(
                f"Existing capture validation failed: seed={seed} path={path}"
            )
        if not all(
            (
                path
                / "observations"
                / view_id
                / filename
            ).is_file()
            for view_id in VIEWS
            for filename in (
                "objective_occlusion.json",
                "objective_reference_occlusion.json",
            )
        ):
            raise RuntimeError(
                f"Objective occlusion ground truth is missing: {path}"
            )
        captures[seed] = path
        state["capture_results"].append(
            {
                "seed": seed,
                "status": "completed_existing_manifest",
                "capture_dir": str(path),
                "calibration_scene_variant": variant,
                "runtime_seconds": 0.0,
            }
        )
        state["last_completed_seed"] = seed
    write_json_atomic(Path(state["output_root"]) / "status.json", state)
    return captures


def build_perception_config(
    captures: dict[int, Path],
    output_root: Path,
    seed_start: int,
    seed_stop: int,
    minimum_candidate_proposals: int,
) -> Path:
    config = json.loads(SOURCE_CONFIG.read_text(encoding="utf-8"))
    config.update(
        {
            "experiment_id": (
                f"scanned_basket_calibration_seed{seed_start:03d}_"
                f"{seed_stop - 1:03d}"
            ),
            "output_root": str(output_root / "perception"),
            "motion_result": None,
            "samples": [
                {
                    "sample_id": f"seed{seed:03d}_{view}",
                    "observation_dir": str(
                        captures[seed] / "observations" / view
                    ),
                    "split": "calibration",
                    "seed": seed,
                    "calibration_scene_variant": json.loads(
                        (
                            captures[seed]
                            / "calibration_ground_truth.json"
                        ).read_text(encoding="utf-8")
                    )["variant"],
                    "calibration_ground_truth_file": str(
                        captures[seed]
                        / "calibration_ground_truth.json"
                    ),
                }
                for seed in range(seed_start, seed_stop)
                for view in VIEWS
            ],
            "training_performed": False,
            "calibration_performed": False,
            "valid_for_final_evaluation": False,
        }
    )
    config["task"]["minimum_candidate_proposals"] = minimum_candidate_proposals
    config["task"].update(
        {
            "instruction": "Find the red mug with the white logo.",
            "target_description": "the red mug with the white logo",
            "qwen_reference_concept": "basket",
            "open_vocabulary_concepts": ["red mug", "basket"],
            "grounding_dino_prompt": "red mug. basket.",
            "factorized_relations": True,
            "membership_label_space": [
                "inside",
                "outside",
                "unknown",
            ],
            "independent_relation_label_spaces": {
                "behind": ["yes", "no", "unknown"],
                "occluded_by": ["yes", "no"],
            },
        }
    )
    config["task"].pop("relation_label_space", None)
    config["limitations"] = [
        "Held-out calibration pilot only; no training or final testing.",
        "Calibration seeds are episode-disjoint from reserved test seeds.",
        "Membership, behind, and occluded_by are scored as separate factors.",
        "Covered target proposal misses calibrate observation missingness, not an invented VLM score.",
        "Grounding thresholds and proposal deduplication are still provisional.",
    ]
    path = output_root / "perception_config.json"
    write_json_atomic(path, config)
    write_json_atomic(
        output_root / "split_manifest.json",
        {
            "schema_version": "calibration-pilot-split-v1",
            "training": {
                "performed": False,
                "seeds": [],
            },
            "calibration": {
                "seeds": list(range(seed_start, seed_stop)),
                "views": list(VIEWS),
                "automatically_labeled_by_simulator": True,
                "scene_variants": sorted(
                    {
                        json.loads(
                            (
                                captures[seed]
                                / "calibration_ground_truth.json"
                            ).read_text(encoding="utf-8")
                        )["variant"]
                        for seed in range(seed_start, seed_stop)
                    }
                ),
                "factorized_relation_labels": True,
            },
            "testing": {
                "performed": False,
                "reserved_seed_range": [200, 210],
                "used_during_calibration": False,
            },
            "view_leakage_prevention": (
                "all views from one seed remain in the calibration split"
            ),
        },
    )
    return path


def binary_metrics(
    logits: list[list[float]], labels: list[int], temperature: float
) -> dict:
    probabilities = [
        softmax_temperature(values, temperature) for values in logits
    ]
    predictions = [
        max(range(len(values)), key=values.__getitem__)
        for values in probabilities
    ]
    brier = float(
        np.mean(
            [
                sum(
                    (
                        probability
                        - (1.0 if index == label else 0.0)
                    )
                    ** 2
                    for index, probability in enumerate(values)
                )
                for values, label in zip(probabilities, labels)
            ]
        )
    )
    return {
        "negative_log_likelihood": negative_log_likelihood(
            logits, labels, temperature
        ),
        "brier_score": brier,
        "accuracy": sum(
            prediction == label
            for prediction, label in zip(predictions, labels)
        )
        / len(labels),
    }


RELATION_FACTORS = {
    "membership": ("inside", "outside", "unknown"),
    "behind": ("yes", "no", "unknown"),
    "occluded_by": ("yes", "no"),
}


def entity_view_ground_truth(
    calibration_ground_truth: dict,
    view_id: str,
    entity_id: str,
) -> dict:
    view = calibration_ground_truth["view_observable_intent"][view_id]
    entities = view.get("entities", {})
    if entity_id not in entities:
        raise KeyError(
            f"Missing factorized view ground truth: {view_id}/{entity_id}"
        )
    return entities[entity_id]


def factor_label(entity_ground_truth: dict, factor: str) -> str:
    if factor == "membership":
        return entity_ground_truth["membership_observable"]
    if factor == "behind":
        return entity_ground_truth["behind"]
    if factor == "occluded_by":
        return entity_ground_truth["occluded_by"]["label"]
    raise ValueError(f"Unknown relation factor: {factor}")


def objective_occlusion_label(
    calibration_ground_truth: dict,
    view_id: str,
) -> str:
    measurement = calibration_ground_truth[
        "objective_reference_occlusion_ground_truth"
    ][view_id]
    if not measurement["valid"]:
        raise ValueError(
            f"Invalid objective occlusion ground truth for view {view_id}"
        )
    return "no" if measurement["severity"] == "no" else "yes"


def objective_behind_label(
    calibration_ground_truth: dict, view_id: str
) -> str:
    measurement = calibration_ground_truth[
        "objective_camera_relative_behind_ground_truth"
    ][view_id]
    if not measurement["valid"]:
        raise ValueError(
            f"Invalid objective behind ground truth for view {view_id}"
        )
    return str(measurement["label"])


def fit_factor(
    logits: list[list[float]],
    labels: list[int],
    observed_labels: set[str],
    *,
    factor: str,
    calibration_seed_count: int,
) -> dict:
    blocking_reasons = []
    missing = sorted(set(RELATION_FACTORS[factor]) - observed_labels)
    if calibration_seed_count < 20:
        blocking_reasons.append(
            "fewer_than_20_episode_disjoint_calibration_scenes"
        )
    if missing:
        blocking_reasons.append("missing_labels:" + ",".join(missing))
    if not logits or len(set(labels)) < 2:
        blocking_reasons.append("fewer_than_two_observed_classes")
        fit = None
        uncalibrated_metrics = None
        calibrated_metrics = None
        class_diagnostics = {}
    else:
        predictions = [
            max(range(len(values)), key=values.__getitem__)
            for values in logits
        ]
        class_diagnostics = {}
        zero_recall_labels = []
        for index, label_name in enumerate(RELATION_FACTORS[factor]):
            support = sum(label == index for label in labels)
            correct = sum(
                label == index and prediction == index
                for label, prediction in zip(labels, predictions)
            )
            recall = correct / support if support else None
            class_diagnostics[label_name] = {
                "support": support,
                "correct": correct,
                "recall": recall,
            }
            if support and correct == 0:
                zero_recall_labels.append(label_name)
        if zero_recall_labels:
            blocking_reasons.append(
                "zero_recall_labels:" + ",".join(zero_recall_labels)
            )
        fit = fit_temperature_grid(
            logits,
            labels,
            minimum=0.25,
            maximum=8.0,
            steps=311,
        )
        if fit["temperature"] in {0.25, 8.0}:
            blocking_reasons.append(
                "temperature_grid_boundary_solution"
            )
        uncalibrated_metrics = binary_metrics(logits, labels, 1.0)
        calibrated_metrics = binary_metrics(
            logits, labels, fit["temperature"]
        )
    return {
        "record_count": len(logits),
        "observed_ground_truth_labels": sorted(observed_labels),
        "required_ground_truth_labels": list(RELATION_FACTORS[factor]),
        "uncalibrated_metrics": uncalibrated_metrics,
        "fit": fit,
        "calibrated_metrics": calibrated_metrics,
        "uncalibrated_class_diagnostics": class_diagnostics,
        "deployable_as_final_calibration": not blocking_reasons,
        "blocking_reasons": blocking_reasons,
    }


def fit_calibration(config_path: Path, output_root: Path) -> dict:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    perception_root = Path(config["output_root"])
    target_logits: list[list[float]] = []
    target_labels: list[int] = []
    factor_logits: dict[str, list[list[float]]] = {
        factor: [] for factor in RELATION_FACTORS
    }
    factor_labels: dict[str, list[int]] = {
        factor: [] for factor in RELATION_FACTORS
    }
    factor_ground_truth_names: dict[str, set[str]] = {
        factor: set() for factor in RELATION_FACTORS
    }
    records = []
    for sample in config["samples"]:
        sample_id = sample["sample_id"]
        observation_dir = Path(sample["observation_dir"])
        view_id = observation_dir.name
        calibration_ground_truth = json.loads(
            Path(sample["calibration_ground_truth_file"]).read_text(
                encoding="utf-8"
            )
        )
        ranking_path = (
            perception_root
            / "grounded_sam2_qwen_rankings"
            / sample_id
            / "result.json"
        )
        ranking = json.loads(ranking_path.read_text(encoding="utf-8"))
        model_input = json.loads(
            Path(ranking["input_path"]).read_text(encoding="utf-8")
        )
        ground_truth_masks = semantic_class_masks(observation_dir)
        target_ground_truth = ground_truth_masks["target_red"]
        distractor_ground_truth = ground_truth_masks["rear_red_candidate"]
        candidate_records = []
        relations_by_source_factor = {
            (item["source_id"], item["relation_type"]): item
            for item in ranking["relations"]
        }
        target_proposal_present = False
        for candidate, raw_match_logit in zip(
            model_input["candidates"], ranking["raw_match_logits"]
        ):
            candidate_mask_path = Path(candidate["mask_path"])
            if not candidate_mask_path.is_absolute():
                candidate_mask_path = ROOT / candidate_mask_path
            candidate_mask = np.asarray(
                Image.open(candidate_mask_path).convert("L")
            ) > 0
            target_overlap = mask_iou(candidate_mask, target_ground_truth)
            distractor_overlap = mask_iou(
                candidate_mask, distractor_ground_truth
            )
            is_target = (
                target_overlap >= 0.5
                and target_overlap >= distractor_overlap
            )
            if is_target:
                target_proposal_present = True
            target_logits.append([float(raw_match_logit), 0.0])
            target_labels.append(0 if is_target else 1)
            entity_id = None
            if is_target:
                entity_id = "target_red"
            elif distractor_overlap >= 0.5:
                entity_id = "rear_red_candidate"
            relation_ground_truth = {}
            relation_ground_truth_sources = {}
            relation_scores = {}
            if entity_id is not None:
                entity_ground_truth = entity_view_ground_truth(
                    calibration_ground_truth,
                    view_id,
                    entity_id,
                )
                objective_behind_available = (
                    view_id
                    in calibration_ground_truth.get(
                        "objective_camera_relative_behind_ground_truth",
                        {},
                    )
                    and calibration_ground_truth[
                        "objective_camera_relative_behind_ground_truth"
                    ][view_id]
                    is not None
                )
                for factor in RELATION_FACTORS:
                    relation = relations_by_source_factor[
                        (candidate["candidate_id"], factor)
                    ]
                    if (
                        factor == "occluded_by"
                        and entity_id == "target_red"
                    ):
                        ground_truth_label = objective_occlusion_label(
                            calibration_ground_truth, view_id
                        )
                        ground_truth_source = (
                            "rendered_reference_removed_amodal_fraction"
                        )
                    elif (
                        factor == "behind"
                        and entity_id == "target_red"
                        and objective_behind_available
                    ):
                        ground_truth_label = objective_behind_label(
                            calibration_ground_truth, view_id
                        )
                        ground_truth_source = (
                            "simulator_geometry_and_rendered_projection"
                        )
                    else:
                        ground_truth_label = factor_label(
                            entity_ground_truth, factor
                        )
                        ground_truth_source = (
                            "deterministic_scene_generator_legacy_view_intent"
                        )
                    relation_index = relation["labels"].index(
                        ground_truth_label
                    )
                    include_in_factor_fit = not (
                        factor == "occluded_by"
                        and entity_id != "target_red"
                    )
                    if (
                        factor == "behind"
                        and objective_behind_available
                        and entity_id != "target_red"
                    ):
                        include_in_factor_fit = False
                    if include_in_factor_fit:
                        factor_logits[factor].append(
                            [
                                float(value)
                                for value in relation["raw_logits"]
                            ]
                        )
                        factor_labels[factor].append(relation_index)
                        factor_ground_truth_names[factor].add(
                            ground_truth_label
                        )
                    relation_ground_truth[factor] = ground_truth_label
                    relation_ground_truth_sources[
                        factor
                    ] = ground_truth_source
                    relation_scores[factor] = {
                        "labels": relation["labels"],
                        "raw_logits": relation["raw_logits"],
                        "top_label": relation["top_label"],
                    }
            candidate_records.append(
                {
                    "candidate_id": candidate["candidate_id"],
                    "target_mask_iou": target_overlap,
                    "distractor_mask_iou": distractor_overlap,
                    "target_label": bool(is_target),
                    "matched_simulator_entity": entity_id,
                    "relation_ground_truth": relation_ground_truth,
                    "relation_ground_truth_sources": (
                        relation_ground_truth_sources
                    ),
                    "raw_match_logit": float(raw_match_logit),
                    "factorized_relation_scores": relation_scores,
                }
            )
        target_visible_pixels = int(
            calibration_ground_truth["observed_target_visibility"][view_id][
                "target_visible_pixel_count"
            ]
        )
        records.append(
            {
                "sample_id": sample_id,
                "seed": sample["seed"],
                "view": view_id,
                "calibration_scene_variant": calibration_ground_truth[
                    "variant"
                ],
                "target_visible_pixel_count": target_visible_pixels,
                "target_proposal_present": target_proposal_present,
                "candidates": candidate_records,
            }
        )
    if not target_logits or len(set(target_labels)) < 2:
        raise RuntimeError(
            "Target calibration requires both match and nonmatch examples"
        )
    target_fit = fit_temperature_grid(
        target_logits,
        target_labels,
        minimum=0.25,
        maximum=8.0,
        steps=311,
    )
    calibration_seeds = sorted(
        {int(sample["seed"]) for sample in config["samples"]}
    )
    target_uncalibrated = binary_metrics(
        target_logits, target_labels, 1.0
    )
    target_blocking_reasons = []
    if len(calibration_seeds) < 20:
        target_blocking_reasons.append(
            "fewer_than_20_episode_disjoint_calibration_scenes"
        )
    if target_fit["temperature"] in {0.25, 8.0}:
        target_blocking_reasons.append(
            "temperature_grid_boundary_solution"
        )
    if target_uncalibrated["accuracy"] >= 1.0:
        target_blocking_reasons.append(
            "no_identity_errors_or_hard_negatives_to_calibrate"
        )
    factor_results = {
        factor: fit_factor(
            factor_logits[factor],
            factor_labels[factor],
            factor_ground_truth_names[factor],
            factor=factor,
            calibration_seed_count=len(calibration_seeds),
        )
        for factor in RELATION_FACTORS
    }
    missing_target_observations = [
        record
        for record in records
        if not record["target_proposal_present"]
    ]
    visible_target_observations = [
        record
        for record in records
        if record["target_visible_pixel_count"] > 0
    ]
    visible_target_misses = [
        record
        for record in visible_target_observations
        if not record["target_proposal_present"]
    ]
    all_factor_deployable = all(
        item["deployable_as_final_calibration"]
        for item in factor_results.values()
    )
    component_fit_acceptance = {
        "target_identity": not target_blocking_reasons,
        **{
            factor: item["deployable_as_final_calibration"]
            for factor, item in factor_results.items()
        },
    }
    mpc_deployment_blocking_reasons = []
    if not all_factor_deployable:
        mpc_deployment_blocking_reasons.append(
            "incomplete_factorized_relation_calibration"
        )
    mpc_deployment_blocking_reasons.extend(
        [
            "task_risk_gate_not_calibrated",
            "action_conditioned_observation_model_not_calibrated",
        ]
    )
    result = {
        "schema_version": "grounded-qwen-factorized-calibration-pilot-v2",
        "split": "calibration_only",
        "target_identity": {
            "record_count": len(target_logits),
            "positive_count": sum(label == 0 for label in target_labels),
            "negative_count": sum(label == 1 for label in target_labels),
            "uncalibrated_temperature": 1.0,
            "uncalibrated_metrics": target_uncalibrated,
            "fitted_temperature": target_fit["temperature"],
            "calibrated_metrics": binary_metrics(
                target_logits,
                target_labels,
                target_fit["temperature"],
            ),
            "fit": target_fit,
            "deployable_as_final_calibration": not target_blocking_reasons,
            "blocking_reasons": target_blocking_reasons,
        },
        "factorized_relations": factor_results,
        "objective_occlusion_calibration": {
            "definition": (
                "target-only: yes when hiding the reference reveals at least "
                "10% of the amodal target; no otherwise"
            ),
            "minimum_yes_fraction": 0.10,
            "non_target_candidates_included": False,
            "reason": (
                "Reference-attributed amodal simulator ground truth is "
                "currently rendered only for the instruction target."
            ),
        },
        "objective_behind_calibration": {
            "definition": (
                "target-only camera-relative relation from simulator world "
                "bounds, current camera ray, and rendered projected overlap"
            ),
            "non_target_candidates_included_when_objective_available": False,
            "legacy_view_intent_used_only_when_objective_measurement_missing": (
                True
            ),
        },
        "proposal_observation_model": {
            "observation_count": len(records),
            "visible_target_observation_count": len(
                visible_target_observations
            ),
            "target_proposal_present_count": (
                len(records) - len(missing_target_observations)
            ),
            "target_proposal_missing_count": len(
                missing_target_observations
            ),
            "visible_target_miss_count": len(visible_target_misses),
            "covered_or_zero_pixel_observation_count": sum(
                record["target_visible_pixel_count"] == 0
                for record in records
            ),
            "calibration_note": (
                "A hidden target with no proposal calibrates observation "
                "missingness; no artificial Qwen candidate score is created."
            ),
        },
        "deployment_decision": {
            "component_fit_acceptance": component_fit_acceptance,
            "apply_target_temperature_to_mpc": False,
            "apply_factor_temperatures_to_mpc": False,
            "calibrate_task_risk_gate": False,
            "calibrate_action_conditioned_observation_model": False,
            "mpc_deployment_blocking_reasons": (
                mpc_deployment_blocking_reasons
            ),
            "reason": (
                "Accepted component fits are retained as calibration "
                "candidates, but no MPC configuration is changed until every "
                "required factor, task-risk gate, and action-conditioned "
                "observation model is calibrated."
            ),
        },
        "calibration_seed_count": len(calibration_seeds),
        "training_performed": False,
        "calibration_performed": True,
        "testing_performed": False,
        "simulator_ground_truth_used_for_calibration_only": True,
        "valid_for_final_evaluation": False,
    }
    write_json_atomic(output_root / "calibration_records.json", {"records": records})
    write_json_atomic(output_root / "calibration_fit.json", result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed-start", type=int, default=100)
    parser.add_argument("--seed-stop", type=int, default=110)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument(
        "--minimum-candidate-proposals",
        type=int,
        default=1,
    )
    parser.add_argument(
        "--perception-only-basket",
        action="store_true",
        help="Use the smaller perception basket without grasp collision proxies.",
    )
    parser.add_argument(
        "--ambiguous-seed-start",
        type=int,
        help=(
            "Use behind_ambiguous for this seed and every later seed in the "
            "requested range; earlier seeds retain the balanced four-way cycle."
        ),
    )
    parser.add_argument(
        "--forced-calibration-scene-variant",
        choices=(
            "inside_clear",
            "outside",
            "rim_occluded",
            "covered_unknown",
            "behind_ambiguous",
            "behind_boundary_unknown",
        ),
        help="Use one deterministic calibration scene variant for every seed.",
    )
    parser.add_argument(
        "--existing-capture-manifest",
        type=Path,
        nargs="+",
        help=(
            "Reuse validated saved RGB-D captures instead of launching "
            "Isaac Sim again."
        ),
    )
    args = parser.parse_args()
    require_single_gpu_policy()
    if args.seed_start < 0 or args.seed_stop <= args.seed_start:
        raise ValueError("Expected 0 <= seed-start < seed-stop")
    if set(range(args.seed_start, args.seed_stop)) & set(range(200, 210)):
        raise ValueError("Reserved test seeds 200--209 cannot be calibrated")
    if args.minimum_candidate_proposals < 1:
        raise ValueError("minimum-candidate-proposals must be positive")
    if args.ambiguous_seed_start is not None and not (
        args.seed_start <= args.ambiguous_seed_start < args.seed_stop
    ):
        raise ValueError(
            "ambiguous-seed-start must be inside [seed-start, seed-stop)"
        )
    if (
        args.ambiguous_seed_start is not None
        and args.forced_calibration_scene_variant is not None
    ):
        raise ValueError(
            "ambiguous-seed-start and forced-calibration-scene-variant "
            "are mutually exclusive"
        )
    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    state = {
        "schema_version": "grounded-qwen-calibration-job-v1",
        "status": "running",
        "seed_start": args.seed_start,
        "seed_stop": args.seed_stop,
        "capture_results": [],
        "stages": [],
        "last_completed_seed": None,
        "physical_gpu": configured_physical_gpu(),
        "minimum_candidate_proposals": args.minimum_candidate_proposals,
        "perception_only_basket": args.perception_only_basket,
        "ambiguous_seed_start": args.ambiguous_seed_start,
        "forced_calibration_scene_variant": (
            args.forced_calibration_scene_variant
        ),
        "existing_capture_manifest": (
            [
                str(path.resolve())
                for path in args.existing_capture_manifest
            ]
            if args.existing_capture_manifest is not None
            else None
        ),
        "output_root": str(output_root),
        "training_performed": False,
        "testing_performed": False,
    }
    (output_root / "COMPLETED.json").unlink(missing_ok=True)
    (output_root / "FAILED.json").unlink(missing_ok=True)
    write_json_atomic(output_root / "status.json", state)
    started = time.perf_counter()
    try:
        if args.existing_capture_manifest is not None:
            captures = load_existing_captures(
                [
                    path.resolve()
                    for path in args.existing_capture_manifest
                ],
                args.seed_start,
                args.seed_stop,
                state,
            )
        else:
            captures = capture_episodes(
                args.seed_start,
                args.seed_stop,
                output_root,
                state,
                collision_physics_basket=not args.perception_only_basket,
                ambiguous_seed_start=args.ambiguous_seed_start,
                forced_calibration_scene_variant=(
                    args.forced_calibration_scene_variant
                ),
            )
        config_path = build_perception_config(
            captures,
            output_root,
            args.seed_start,
            args.seed_stop,
            args.minimum_candidate_proposals,
        )
        for stage in ("gdino_detect", "sam2_segment"):
            result = run_logged(
                [
                    str(PERCEPTION_PYTHON),
                    str(ROOT / "scripts" / "run_perception_grounding_pilot.py"),
                    stage,
                    "--config",
                    str(config_path),
                ],
                log_root=output_root / "logs",
                name=stage,
                timeout_seconds=1800.0,
            )
            state["stages"].append(result)
            write_json_atomic(output_root / "status.json", state)
        for name, command in (
            (
                "export_qwen_inputs",
                [
                    str(PERCEPTION_PYTHON),
                    str(ROOT / "scripts" / "export_grounded_sam2_qwen_inputs.py"),
                    "--config",
                    str(config_path),
                ],
            ),
            (
                "qwen_ranking",
                [
                    str(PERCEPTION_PYTHON),
                    str(ROOT / "scripts" / "run_grounded_proposal_qwen_ranking.py"),
                    "--config",
                    str(config_path),
                ],
            ),
            (
                "posthoc_evaluation",
                [
                    str(PERCEPTION_PYTHON),
                    str(ROOT / "scripts" / "evaluate_perception_grounding_pilot.py"),
                    "--config",
                    str(config_path),
                ],
            ),
        ):
            result = run_logged(
                command,
                log_root=output_root / "logs",
                name=name,
                timeout_seconds=2400.0,
            )
            state["stages"].append(result)
            write_json_atomic(output_root / "status.json", state)
        calibration = fit_calibration(config_path, output_root)
        state.update(
            {
                "status": "completed",
                "runtime_seconds": time.perf_counter() - started,
                "perception_config": str(config_path),
                "calibration_fit": str(output_root / "calibration_fit.json"),
                "target_fitted_temperature": calibration[
                    "target_identity"
                ]["fitted_temperature"],
            }
        )
        write_json_atomic(output_root / "status.json", state)
        write_json_atomic(
            output_root / "COMPLETED.json",
            {
                "status": "completed",
                "runtime_seconds": state["runtime_seconds"],
                "calibration_fit": state["calibration_fit"],
            },
        )
        print(f"CALIBRATION_PILOT_COMPLETED={output_root}", flush=True)
    except Exception as error:
        state.update(
            {
                "status": "failed",
                "runtime_seconds": time.perf_counter() - started,
                "error_type": type(error).__name__,
                "error": str(error),
            }
        )
        write_json_atomic(output_root / "status.json", state)
        write_json_atomic(
            output_root / "FAILED.json",
            {
                "status": "failed",
                "error_type": type(error).__name__,
                "error": str(error),
            },
        )
        raise


if __name__ == "__main__":
    main()
