"""Run one causal learned-perception-to-contact-grasp single-GPU pilot.

The Isaac process remains alive while GroundingDINO, SAM2.1, and Qwen3-VL are
loaded one at a time in separate subprocesses.  Each next view is requested
only after the current observation has been processed and a pre-action plan has
been written.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import itertools
import json
import math
import os
import shutil
import subprocess
import time
from pathlib import Path

import numpy as np
from PIL import Image

from rgbd_target_localization import localize_mask_files
from run_live_single_gpu_pipeline import (
    ISAAC_PYTHON,
    ROOT,
    wait_for_path,
    write_json_atomic,
)
from run_non_oracle_hybrid_planner import plan
from run_qwen_belief_mpc_replay import weighted_log_belief_update
from run_scanned_basket_occlusion_belief_pilot import (
    candidate_centers,
    normalize,
    softmax,
)
from run_scanned_basket_two_step_belief_pilot import planner_config
from run_single_gpu_pilot import require_single_gpu_policy
from run_single_gpu_pilot import configured_physical_gpu


PERCEPTION_PYTHON = Path(
    os.environ.get(
        "EFFICIENT_ROBOTICS_PERCEPTION_PYTHON",
        ROOT / ".venv-perception" / "bin" / "python",
    )
)
MODEL_CONFIG_SOURCE = (
    ROOT
    / "configs"
    / "perception"
    / "scanned_basket_occlusion_two_step_seed000.json"
)
PLANNER_CONFIG_SOURCE = (
    ROOT
    / "configs"
    / "research"
    / "scanned_basket_occlusion_belief_mpc_pilot.json"
)
OUTPUT_ROOT = (
    ROOT
    / "outputs"
    / "live_pipeline"
    / "learned_scanned_basket_e2e"
)
GROUNDED_QWEN_CACHE_ROOT = (
    ROOT / "outputs" / "pilot_cache" / "grounded_qwen_factorized"
)


def next_output_dir(seed: int) -> Path:
    seed_root = OUTPUT_ROOT / f"seed{seed:03d}"
    seed_root.mkdir(parents=True, exist_ok=True)
    indices = []
    for path in seed_root.glob("run*"):
        try:
            indices.append(int(path.name.removeprefix("run")))
        except ValueError:
            continue
    return seed_root / f"run{max(indices, default=0) + 1:03d}"


def single_gpu_environment() -> dict[str, str]:
    physical_gpu = str(configured_physical_gpu())
    environment = dict(os.environ)
    environment.update(
        {
            "CUDA_VISIBLE_DEVICES": physical_gpu,
            "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
            "NVIDIA_VISIBLE_DEVICES": physical_gpu,
            "PHYSICAL_GPU": physical_gpu,
        }
    )
    for name in ("RANK", "LOCAL_RANK", "WORLD_SIZE", "MASTER_ADDR", "MASTER_PORT"):
        environment.pop(name, None)
    return environment


def run_stage(
    command: list[str],
    *,
    session_dir: Path,
    step_index: int,
    stage_name: str,
    timeout_seconds: float,
) -> dict:
    log_root = session_dir / "learned_perception" / "logs"
    log_root.mkdir(parents=True, exist_ok=True)
    stdout_path = log_root / f"{step_index:03d}_{stage_name}_stdout.log"
    stderr_path = log_root / f"{step_index:03d}_{stage_name}_stderr.log"
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
        "stage": stage_name,
        "command": command,
        "returncode": completed.returncode,
        "runtime_seconds": time.perf_counter() - started,
        "stdout": str(stdout_path),
        "stderr": str(stderr_path),
    }
    if completed.returncode != 0:
        raise RuntimeError(f"Learned perception stage failed: {result}")
    return result


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def grounded_qwen_cache_key(input_path: Path, config_path: Path) -> str:
    model_input = json.loads(input_path.read_text(encoding="utf-8"))
    config = json.loads(config_path.read_text(encoding="utf-8"))
    normalized = {
        "schema_version": model_input["schema_version"],
        "instruction": model_input["instruction"],
        "target_description": model_input["target_description"],
        "image_sha256": file_sha256(
            resolve_input_asset(model_input, model_input["image"]["rgb_path"])
        ),
        "candidates": [],
        "references": [],
        "relation_queries": model_input.get("relation_queries", []),
        "model": config["models"]["qwen"],
        "prompt_version": (
            "grounded-qwen-factorized-identity-relation-v1-rgb-box-crop"
        ),
    }
    for candidate in model_input["candidates"]:
        normalized["candidates"].append(
            {
                "candidate_id": candidate["candidate_id"],
                "bbox_xyxy": candidate["bbox_xyxy"],
                "crop_sha256": file_sha256(
                    resolve_input_asset(model_input, candidate["crop_path"])
                ),
                "mask_sha256": file_sha256(
                    resolve_input_asset(model_input, candidate["mask_path"])
                ),
                "context_sha256": file_sha256(
                    resolve_input_asset(model_input, candidate["context_path"])
                ),
            }
        )
    for reference in model_input["reference_entities"]:
        normalized["references"].append(
            {
                "reference_id": reference["reference_id"],
                "bbox_xyxy": reference["bbox_xyxy"],
                "mask_sha256": file_sha256(
                    resolve_input_asset(model_input, reference["mask_path"])
                ),
                "overlay_sha256": file_sha256(
                    resolve_input_asset(model_input, reference["overlay_path"])
                ),
            }
        )
    return hashlib.sha256(
        json.dumps(
            normalized,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()


def qwen_ranking_with_cache(
    *,
    session_dir: Path,
    config_path: Path,
    sample_id: str,
    step_index: int,
) -> dict:
    learned_root = session_dir / "learned_perception"
    input_path = (
        learned_root
        / "grounded_sam2_qwen_inputs"
        / sample_id
        / "input.json"
    )
    result_path = (
        learned_root
        / "grounded_sam2_qwen_rankings"
        / sample_id
        / "result.json"
    )
    result_path.parent.mkdir(parents=True, exist_ok=True)
    cache_key = grounded_qwen_cache_key(input_path, config_path)
    cache_dir = GROUNDED_QWEN_CACHE_ROOT / cache_key
    cached_result_path = cache_dir / "result.json"
    if cached_result_path.is_file():
        result = json.loads(cached_result_path.read_text(encoding="utf-8"))
        result["sample_id"] = sample_id
        result["input_path"] = str(input_path)
        result["cache"] = {
            "hit": True,
            "key": cache_key,
            "source": str(cached_result_path),
        }
        write_json_atomic(result_path, result)
        return {
            "stage": "qwen_ranking",
            "command": [],
            "returncode": 0,
            "runtime_seconds": 0.0,
            "stdout": None,
            "stderr": None,
            "cache_hit": True,
            "cache_key": cache_key,
        }

    stage_result = run_stage(
        [
            str(PERCEPTION_PYTHON),
            str(ROOT / "scripts" / "run_grounded_proposal_qwen_ranking.py"),
            "--config",
            str(config_path),
            "--force",
        ],
        session_dir=session_dir,
        step_index=step_index,
        stage_name="qwen_ranking",
        timeout_seconds=420.0,
    )
    result = json.loads(result_path.read_text(encoding="utf-8"))
    result["cache"] = {
        "hit": False,
        "key": cache_key,
        "source": str(result_path),
    }
    write_json_atomic(result_path, result)
    cache_dir.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(result_path, cached_result_path)
    write_json_atomic(
        cache_dir / "metadata.json",
        {
            "schema_version": "grounded-qwen-factorized-cache-v1",
            "cache_key": cache_key,
            "source_input": str(input_path),
            "model": result["model"],
            "prompt_version": result["prompt_version"],
        },
    )
    return {
        **stage_result,
        "cache_hit": False,
        "cache_key": cache_key,
    }


def make_perception_config(
    *,
    session_dir: Path,
    observation_dir: Path,
    sample_id: str,
    step_index: int,
    task_overrides: dict | None = None,
) -> Path:
    source = json.loads(MODEL_CONFIG_SOURCE.read_text(encoding="utf-8"))
    config = {
        **source,
        "experiment_id": f"{session_dir.name}_live_step_{step_index:03d}",
        "output_root": str(session_dir / "learned_perception"),
        "motion_result": None,
        "samples": [
            {
                "sample_id": sample_id,
                "observation_dir": str(observation_dir),
            }
        ],
    }
    # A re-observation may legitimately see only one of the two persistent
    # hypotheses.  The live tracker keeps an unseen hypothesis with neutral
    # likelihood instead of forcing a detector hallucination.
    config["task"]["minimum_candidate_proposals"] = 1
    if task_overrides:
        unknown = sorted(set(task_overrides) - set(config["task"]))
        if unknown:
            raise ValueError(f"Unknown perception task overrides: {unknown}")
        config["task"].update(copy.deepcopy(task_overrides))
    config["limitations"] = [
        "Single deterministic live integration pilot only.",
        "Grounding thresholds, Qwen scores, and belief update are uncalibrated.",
        "This output is not final evaluation evidence.",
    ]
    config_path = session_dir / f"perception_config_{step_index:03d}.json"
    write_json_atomic(config_path, config)
    return config_path


def run_current_observation_perception(
    *,
    session_dir: Path,
    observation_dir: Path,
    view: str,
    step_index: int,
    task_overrides: dict | None = None,
) -> tuple[dict, Path, Path, list[dict]]:
    sample_id = f"{session_dir.name}_{step_index:03d}_{view}"
    config_path = make_perception_config(
        session_dir=session_dir,
        observation_dir=observation_dir,
        sample_id=sample_id,
        step_index=step_index,
        task_overrides=task_overrides,
    )
    stage_results = []
    for stage in ("gdino_detect", "sam2_segment"):
        stage_results.append(
            run_stage(
                [
                    str(PERCEPTION_PYTHON),
                    str(ROOT / "scripts" / "run_perception_grounding_pilot.py"),
                    stage,
                    "--config",
                    str(config_path),
                    "--force",
                ],
                session_dir=session_dir,
                step_index=step_index,
                stage_name=stage,
                timeout_seconds=300.0,
            )
        )
    stage_results.append(
        run_stage(
            [
                str(PERCEPTION_PYTHON),
                str(ROOT / "scripts" / "export_grounded_sam2_qwen_inputs.py"),
                "--config",
                str(config_path),
            ],
            session_dir=session_dir,
            step_index=step_index,
            stage_name="export_qwen_inputs",
            timeout_seconds=120.0,
        )
    )
    stage_results.append(
        qwen_ranking_with_cache(
            session_dir=session_dir,
            config_path=config_path,
            sample_id=sample_id,
            step_index=step_index,
        )
    )
    learned_root = session_dir / "learned_perception"
    ranking_path = (
        learned_root
        / "grounded_sam2_qwen_rankings"
        / sample_id
        / "result.json"
    )
    ranking = json.loads(ranking_path.read_text(encoding="utf-8"))
    return ranking, ranking_path, config_path, stage_results


def resolve_input_asset(_container: dict, value: str) -> Path:
    path = Path(value)
    if not path.is_absolute():
        path = ROOT / path
    path = path.resolve()
    if not path.is_relative_to(ROOT):
        raise ValueError(f"Perception asset escapes project root: {path}")
    return path


def selected_target_mask(
    ranking: dict,
    candidate_to_track: dict[str, str],
    selected_track: str,
) -> Path:
    candidates = [
        candidate_id
        for candidate_id, track_id in candidate_to_track.items()
        if track_id == selected_track
    ]
    if len(candidates) != 1:
        raise ValueError(
            f"Expected one candidate for {selected_track}, found {candidates}"
        )
    model_input = json.loads(Path(ranking["input_path"]).read_text(encoding="utf-8"))
    by_id = {
        candidate["candidate_id"]: candidate for candidate in model_input["candidates"]
    }
    return resolve_input_asset(ranking, by_id[candidates[0]]["mask_path"])


def partial_track_mapping(
    reference_centers: dict[str, list[float]],
    current_centers: dict[str, list[float]],
    *,
    maximum_track_distance_m: float = 0.08,
) -> tuple[dict[str, str], dict]:
    reference_ids = list(reference_centers)
    current_ids = list(current_centers)
    if not current_ids:
        raise ValueError("At least one current candidate is required")
    if len(current_ids) > len(reference_ids):
        raise ValueError(
            "Current detector produced more candidates than persistent tracks"
        )
    reference_tracks = {
        candidate_id: f"track_{index:03d}"
        for index, candidate_id in enumerate(reference_ids, start=1)
    }
    best = None
    for reference_subset in itertools.permutations(
        reference_ids, len(current_ids)
    ):
        pairs = list(zip(current_ids, reference_subset))
        distances = [
            float(
                np.linalg.norm(
                    np.asarray(current_centers[current_id])
                    - np.asarray(reference_centers[reference_id])
                )
            )
            for current_id, reference_id in pairs
        ]
        total = sum(distances)
        if best is None or total < best[0]:
            best = (total, pairs, distances)
    assert best is not None
    accepted = [
        (current_id, reference_id, distance)
        for (current_id, reference_id), distance in zip(best[1], best[2])
        if distance <= maximum_track_distance_m
    ]
    if not accepted:
        raise ValueError(
            "No current candidate is within the RGB-D track-distance gate"
        )
    mapping = {
        current_id: reference_tracks[reference_id]
        for current_id, reference_id, _distance in accepted
    }
    return mapping, {
        "method": "minimum_sum_partial_rgbd_world_center_distance",
        "maximum_track_distance_m": maximum_track_distance_m,
        "persistent_track_count": len(reference_ids),
        "visible_candidate_count": len(current_ids),
        "accepted_candidate_count": len(accepted),
        "unobserved_tracks": sorted(set(reference_tracks.values()) - set(mapping.values())),
        "unmatched_current_candidates": sorted(
            set(current_ids) - set(mapping)
        ),
        "total_distance_m": best[0],
        "pairs": [
            {
                "current_candidate_id": current_id,
                "reference_candidate_id": reference_id,
                "track_id": reference_tracks[reference_id],
                "distance_m": distance,
                "accepted": distance <= maximum_track_distance_m,
            }
            for (current_id, reference_id), distance in zip(best[1], best[2])
        ],
        "simulator_ground_truth_used": False,
    }


def partial_ranking_belief(
    ranking: dict,
    candidate_to_track: dict[str, str],
    persistent_tracks: list[str],
    temperature: float,
) -> dict:
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    logits_by_candidate = dict(
        zip(ranking["candidate_ids"], ranking["raw_match_logits"])
    )
    # A match-minus-nonmatch logit is a log-likelihood-ratio-like score.
    # Unobserved tracks receive zero log evidence, so their likelihood ratio is
    # one.  When all tracks are visible this is exactly softmax(logit/T).
    track_logits = {track_id: 0.0 for track_id in persistent_tracks}
    for candidate_id, track_id in candidate_to_track.items():
        track_logits[track_id] = float(logits_by_candidate[candidate_id])
    maximum = max(value / temperature for value in track_logits.values())
    target_weights = {
        track_id: math.exp(value / temperature - maximum)
        for track_id, value in track_logits.items()
    }

    relation = ranking["selected_candidate_relation"]
    relation_scores = softmax(relation["raw_logits"], temperature)
    by_label = dict(zip(relation["labels"], relation_scores))
    return {
        "target": normalize(target_weights),
        "relation": normalize(
            {
                "inside": by_label.get("inside", 0.0),
                "outside": by_label.get("outside", 0.0),
                "unknown": (
                    by_label.get("behind", 0.0)
                    + by_label.get("unknown", 0.0)
                ),
            }
        ),
        "selected_candidate_id": ranking["selected_candidate_id"],
        "selected_track_id": candidate_to_track[
            ranking["selected_candidate_id"]
        ],
        "raw_logit_temperature": temperature,
        "unobserved_track_log_evidence": 0.0,
        "calibrated": False,
    }


def mask_iou(first: Path, second: Path) -> float:
    first_mask = np.asarray(Image.open(first).convert("L")) > 0
    second_mask = np.asarray(Image.open(second).convert("L")) > 0
    intersection = int(np.logical_and(first_mask, second_mask).sum())
    union = int(np.logical_or(first_mask, second_mask).sum())
    return intersection / union if union else 0.0


def occluder_mask(ranking: dict, target_mask: Path) -> Path:
    sample_id = ranking["sample_id"]
    segmentation_path = (
        Path(ranking["input_path"]).resolve().parents[2]
        / "grounded_sam2"
        / sample_id
        / "segmentations.json"
    )
    segmentations = json.loads(segmentation_path.read_text(encoding="utf-8"))
    candidates = [
        annotation
        for annotation in segmentations["annotations"]
        if annotation["label"] == "orange cylinder"
    ]
    if not candidates:
        raise ValueError("Grounded-SAM2 did not produce an orange-cylinder mask")
    non_target_candidates = [
        annotation
        for annotation in candidates
        if mask_iou(
            segmentation_path.parent / annotation["mask_path"],
            target_mask,
        )
        < 0.10
    ]
    if not non_target_candidates:
        raise ValueError(
            "Every orange-cylinder proposal overlaps the selected target mask"
        )
    selected = max(
        non_target_candidates,
        key=lambda annotation: (
            float(annotation["score"]),
            int(annotation["mask_pixel_count"]),
        ),
    )
    return segmentation_path.parent / selected["mask_path"]


def validate_server_result(result: dict, terminal_action: str) -> None:
    if result.get("status") != "completed":
        raise RuntimeError(f"Isaac server did not complete: {result}")
    if terminal_action == "grasp":
        grasp = result.get("grasp_execution") or {}
        required = (
            result.get("grasp_executed"),
            grasp.get("lift_verified"),
            grasp.get("bilateral_contact_before_lift"),
            not grasp.get("unexpected_environment_pairs"),
            grasp.get("contact_force_within_limit"),
            grasp.get("contact_penetration_within_limit"),
        )
        if not all(required):
            raise RuntimeError(f"Persistent contact grasp failed safety gates: {grasp}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--maximum-observations", type=int, default=3)
    parser.add_argument(
        "--planner-config",
        type=Path,
        default=PLANNER_CONFIG_SOURCE,
        help=(
            "Belief-space planner configuration. The file is copied into "
            "the episode provenance before execution."
        ),
    )
    parser.add_argument(
        "--grasp-timeout-seconds",
        type=float,
        default=1800.0,
        help=(
            "Maximum time to wait for contact-gated grasp physics and video "
            "encoding after a terminal grasp request. Shared GPUs can make "
            "this substantially slower than the simulated motion duration."
        ),
    )
    parser.add_argument("--execute-grasp", action="store_true")
    args = parser.parse_args()
    if args.grasp_timeout_seconds <= 0.0:
        parser.error("--grasp-timeout-seconds must be positive")
    require_single_gpu_policy()
    allow_pilot_grasp = os.environ.get("LEARNED_ALLOW_PILOT_GRASP") == "1"
    if args.execute_grasp and args.seed != 0 and not allow_pilot_grasp:
        raise ValueError(
            "The learned contact-grasp path is restricted to validated seed 0"
        )
    if args.maximum_observations < 1:
        raise ValueError("--maximum-observations must be positive")
    if not ISAAC_PYTHON.is_file() or not PERCEPTION_PYTHON.is_file():
        raise FileNotFoundError("Required Isaac or perception Python is unavailable")

    session_dir = next_output_dir(args.seed)
    session_dir.mkdir(parents=True, exist_ok=False)
    planner_config_source = args.planner_config.resolve()
    if not planner_config_source.is_file():
        raise FileNotFoundError(
            f"Planner config does not exist: {planner_config_source}"
        )
    base_planner = json.loads(
        planner_config_source.read_text(encoding="utf-8")
    )
    server_method = json.loads(
        (
            ROOT / "configs" / "research" / "first_belief_mpc_integration.json"
        ).read_text(encoding="utf-8")
    )
    server_method["viewpoint_execution"]["mode"] = "interpolated_joint_physics"
    server_method["viewpoint_execution"]["debug_ee_positions_world_m"] = {}
    method_path = session_dir / "effective_method_config.json"
    write_json_atomic(method_path, server_method)
    physical_gpu = configured_physical_gpu()

    command = [
        str(ISAAC_PYTHON),
        str(ROOT / "scripts" / "open_minimal_scene.py"),
        "--scene-profile",
        "benchmark",
        "--headless",
        "--renderer-gpu",
        str(physical_gpu),
        "--physics-gpu",
        "0",
        "--live-pipeline-server",
        "--actual-view-motion",
        "--live-session-dir",
        str(session_dir),
        "--method-config",
        str(method_path),
        "--seed",
        str(args.seed),
        "--household-perception-pilot",
        "--scanned-basket-perception-pilot",
        "--active-occlusion-pilot",
    ]
    if args.execute_grasp:
        command.append("--basket-collision-physics-pilot")
        command.append("--execute-persistent-composite-grasp")

    isaac_stdout_path = session_dir / "isaac_stdout.log"
    isaac_stderr_path = session_dir / "isaac_stderr.log"
    isaac_stdout = isaac_stdout_path.open("w", encoding="utf-8")
    isaac_stderr = isaac_stderr_path.open("w", encoding="utf-8")
    started = time.perf_counter()
    server = subprocess.Popen(
        command,
        cwd=ROOT,
        env=single_gpu_environment(),
        stdout=isaac_stdout,
        stderr=isaac_stderr,
        text=True,
    )

    steps = []
    reference_centers = None
    persistent_tracks: list[str] = []
    belief = None
    executed_actions: list[str] = []
    terminal_action = "stop"
    try:
        for step_index in range(args.maximum_observations):
            event_path = session_dir / f"observation_ready_{step_index:03d}.json"
            wait_for_path(event_path, server, timeout=180.0)
            event = json.loads(event_path.read_text(encoding="utf-8"))
            view = event["view"]
            observation_dir = Path(event["observation_dir"]).resolve()
            ranking, ranking_path, perception_config_path, stages = (
                run_current_observation_perception(
                    session_dir=session_dir,
                    observation_dir=observation_dir,
                    view=view,
                    step_index=step_index,
                )
            )
            centers = candidate_centers(ranking, observation_dir)
            if reference_centers is None:
                reference_centers = centers
                mapping = {
                    candidate_id: f"track_{index:03d}"
                    for index, candidate_id in enumerate(centers, start=1)
                }
                persistent_tracks = list(mapping.values())
                tracking = {
                    "method": "initial_rgbd_world_center_tracks",
                    "simulator_ground_truth_used": False,
                }
            else:
                mapping, tracking = partial_track_mapping(
                    reference_centers, centers
                )

            adapter = base_planner["qwen_belief_adapter"]
            observation_belief = partial_ranking_belief(
                ranking,
                mapping,
                persistent_tracks,
                float(adapter["raw_logit_temperature"]),
            )
            belief_before = copy.deepcopy(belief)
            if belief is None:
                belief = {
                    "target": observation_belief["target"],
                    "relation": observation_belief["relation"],
                }
            else:
                belief = weighted_log_belief_update(
                    belief,
                    {
                        "target": observation_belief["target"],
                        "relation": observation_belief["relation"],
                    },
                    float(adapter["observation_log_weight"]),
                )

            current_config = planner_config(
                base_planner,
                belief,
                completed_reobservations=step_index,
                executed_actions=executed_actions,
                perception_config_path=perception_config_path,
            )
            current_plan = plan(current_config)
            action_type = current_plan["action_request"]["type"]
            if (
                step_index == args.maximum_observations - 1
                and action_type.startswith("viewpoint_")
            ):
                action_type = "defer"
            if action_type == "grasp" and not args.execute_grasp:
                action_type = "stop"

            localization_path = None
            selected_track = max(belief["target"], key=belief["target"].get)
            if action_type == "grasp":
                target_mask = selected_target_mask(
                    ranking, mapping, selected_track
                )
                localization = localize_mask_files(
                    observation_dir,
                    {
                        "selected_target": target_mask,
                    },
                )
                localization["selection"] = {
                    "posterior_selected_track": selected_track,
                    "source_ranking": str(ranking_path),
                    "source_view": view,
                    "simulator_ground_truth_used": False,
                }
                localization_path = session_dir / "terminal_rgbd_localization.json"
                write_json_atomic(localization_path, localization)

            step = {
                "index": step_index,
                "view": view,
                "event": event,
                "perception_config": str(perception_config_path),
                "ranking_path": str(ranking_path),
                "perception_stages": stages,
                "candidate_tracking": tracking,
                "candidate_to_track": mapping,
                "observation_belief": observation_belief,
                "belief_before_update": belief_before,
                "belief_after_update": belief,
                "pre_action_plan": current_plan,
                "selected_action": action_type,
                "selected_track": selected_track,
                "terminal_rgbd_localization": (
                    str(localization_path) if localization_path else None
                ),
            }
            steps.append(step)
            decision_path = session_dir / f"decision_{step_index:03d}.json"
            write_json_atomic(decision_path, step)
            write_json_atomic(
                session_dir / f"action_request_{step_index:03d}.json",
                {
                    "schema_version": "live-learned-action-request-v1",
                    "index": step_index,
                    "type": action_type,
                    "selected_candidate": selected_track,
                    "rgbd_localization_path": (
                        str(localization_path) if localization_path else None
                    ),
                    "source_decision": str(decision_path),
                },
            )
            print(
                f"LEARNED_LIVE_STEP={step_index} VIEW={view} "
                f"ACTION={action_type}",
                flush=True,
            )
            if action_type.startswith("viewpoint_"):
                executed_actions.append(action_type)
                continue
            terminal_action = action_type
            break

        wait_for_path(
            session_dir / "server_result.json",
            server,
            timeout=(
                args.grasp_timeout_seconds
                if terminal_action == "grasp"
                else 180.0
            ),
        )
        server.wait(timeout=60.0)
        if server.returncode != 0:
            raise RuntimeError(f"Isaac server failed with code {server.returncode}")
        server_result = json.loads(
            (session_dir / "server_result.json").read_text(encoding="utf-8")
        )
        validate_server_result(server_result, terminal_action)
        result = {
            "schema_version": "live-learned-scanned-basket-e2e-v1",
            "status": "completed",
            "purpose": "single_seed_pipeline_validation_only",
            "session_dir": str(session_dir),
            "seed": args.seed,
            "steps": steps,
            "terminal_action": terminal_action,
            "server_result": server_result,
            "runtime_seconds": time.perf_counter() - started,
            "gpu_policy": {
                "physical_gpu": physical_gpu,
                "renderer_active_gpu": physical_gpu,
                "physics_cuda_device": 0,
                "visible_cuda_devices": 1,
                "single_model_instance_at_a_time": True,
                "batch_size": 1,
                "multi_gpu": False,
                "distributed": False,
            },
            "perception_path": (
                "GroundingDINO-Base -> SAM2.1-Large -> anonymous masks -> "
                "Qwen3-VL-8B target/relation scores"
            ),
            "future_capture_files_read_before_plan": [],
            "pre_captured_candidate_replay": False,
            "planner_config_source": str(planner_config_source),
            "training_performed": False,
            "calibration_performed": False,
            "actual_mpc_solver": bool(
                steps
                and all(
                    step["pre_action_plan"]["provenance"][
                        "actual_mpc_solver"
                    ]
                    for step in steps
                )
            ),
            "grasp_executed": bool(server_result.get("grasp_executed")),
            "valid_for_final_evaluation": False,
        }
        write_json_atomic(session_dir / "pipeline_result.json", result)
        print(f"LEARNED_LIVE_PIPELINE_RESULT={session_dir / 'pipeline_result.json'}")
    except Exception as error:
        write_json_atomic(
            session_dir / "pipeline_error.json",
            {
                "status": "failed",
                "error_type": type(error).__name__,
                "error": str(error),
                "steps": steps,
            },
        )
        if server.poll() is None:
            index = len(steps)
            write_json_atomic(
                session_dir / f"action_request_{index:03d}.json",
                {
                    "schema_version": "live-learned-action-request-v1",
                    "index": index,
                    "type": "stop",
                    "reason": "external_pipeline_error",
                },
            )
            try:
                server.wait(timeout=20.0)
            except subprocess.TimeoutExpired:
                server.terminate()
                server.wait(timeout=20.0)
        raise
    finally:
        if server.poll() is None:
            server.terminate()
            server.wait(timeout=30.0)
        isaac_stdout.close()
        isaac_stderr.close()


if __name__ == "__main__":
    main()
