"""Run one contact-gated cover-removal validation in a persistent Isaac stage."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
from pathlib import Path

import numpy as np

from run_closed_loop_pipeline import (
    ISAAC_PYTHON,
    ROOT,
    wait_for_path,
    write_json_atomic,
)
from single_gpu_runtime import configured_physical_gpu
from single_gpu_runtime import require_single_gpu_policy
from final_evaluation_authorization import (
    DEFAULT_PROTOCOL,
    validate_output_authorization,
)
from run_cover_search_belief_mpc import (
    execute_observation_action,
    normalize,
    plan,
    validate_config,
)
from run_post_cover_view_calibration import select_post_cover_action
from run_scene_conditioned_future_belief_calibration import (
    FEATURE_NAMES,
    bbox_area_fraction,
    bbox_aspect,
    predict_variant_distribution,
)


OUTPUT_ROOT = ROOT / "outputs" / "live_pipeline" / "cover_removal_validation"
CALIBRATION_RESULT = (
    ROOT
    / "outputs"
    / "offline_mpc"
    / "scene_conditioned_future_belief_seed185_196"
    / "result.json"
)
FROZEN_COVER_PLANNER = (
    ROOT / "outputs" / "final_evaluation" / "icra_protocol_v1"
    / "frozen_cover_planner.json"
)
FINAL_PERCEPTION_TASK_OVERRIDES = {
    "instruction": "Find and pick up the red mug with the white logo.",
    "target_description": "the red mug with the white logo",
    "qwen_direct_prompt": (
        "Find the single red mug with the white logo. If it cannot be "
        "located reliably, return []. Return JSON only."
    ),
    "qwen_candidate_concept": "red object",
    "qwen_reference_concept": "open container",
    "minimum_candidate_proposals": 1,
    "factorized_relations": True,
    "membership_label_space": ["inside", "outside", "unknown"],
    "independent_relation_label_spaces": {
        "near": ["yes", "no", "unknown"],
        "behind": ["yes", "no", "unknown"],
        "occluded_by": ["yes", "no", "unknown"],
    },
    "open_vocabulary_concepts": [
        "red object",
        "orange object",
        "yellow object",
        "blue object",
        "green object",
        "purple object",
        "open container",
        "lid or cover",
    ],
    "grounding_dino_prompt": (
        "red object. orange object. yellow object. blue object. green object. "
        "purple object. open container. lid or cover."
    ),
}
FINAL_PERCEPTION_MODEL_OVERRIDES = {
    "grounding_dino": {
        "box_threshold": 0.25,
        "text_threshold": 0.2,
    }
}
POST_COVER_VIEW_CALIBRATION = (
    ROOT / "outputs/offline_mpc/post_cover_view_calibration_seed224_239/result.json"
)
COVERED_SCENE_VARIANTS = (
    "cover_removal_required",
    "empty_cover_then_right",
    "covered_then_close_high_only",
    "covered_then_right_only",
    "covered_then_either_view",
    "covered_center_ambiguous_then_close_high_only",
    "covered_center_ambiguous_then_close_high_logo_v2",
    "covered_center_ambiguous_then_right_only",
    "target_absent_covered",
    "covered_target_outside_visible_no_gain",
)


def expected_visibility_outcome_passed(
    scene_variant: str,
    initial_pixels: int,
    post_pixels: int,
) -> tuple[str, bool]:
    """Check the intervention outcome specified by each calibration family."""
    if scene_variant == "target_absent_covered":
        return "target_absent_before_and_after", bool(
            initial_pixels == 0 and post_pixels == 0
        )
    if scene_variant == "covered_target_outside_visible_no_gain":
        return "outside_target_visible_before_and_after", bool(
            initial_pixels >= 100 and post_pixels >= 100
        )
    return "target_visibility_increases_after_removal", bool(
        post_pixels > initial_pixels
    )


def counterfactual_cache_step(
    views: list[str], view_offset: int
) -> tuple[int, str, dict]:
    """Return the expected event and follow-up request for one cached view."""
    if not views:
        raise ValueError("At least one counterfactual view is required")
    if view_offset < 0 or view_offset >= len(views):
        raise IndexError("Counterfactual view offset is out of range")
    if len(views) != len(set(views)):
        raise ValueError("Counterfactual views must be unique")
    expected_view = views[view_offset]
    next_view = views[view_offset + 1] if view_offset + 1 < len(views) else None
    event_index = 2 + view_offset
    request = {
        "schema_version": "counterfactual-cache-request-v1",
        "index": event_index,
        "type": f"viewpoint_{next_view}" if next_view is not None else "stop",
        "counterfactual_cache_only": True,
        "physical_execution_requested": False,
    }
    return event_index, expected_view, request


def next_output_dir(seed: int, output_root: Path = OUTPUT_ROOT) -> Path:
    """Allocate a new per-seed output directory."""
    seed_root = output_root / f"seed{seed:03d}"
    seed_root.mkdir(parents=True, exist_ok=True)
    indices = []
    for path in seed_root.glob("run*"):
        try:
            indices.append(int(path.name.removeprefix("run")))
        except ValueError:
            continue
    return seed_root / f"run{max(indices, default=0) + 1:03d}"


def target_pixel_count(observation_dir: Path) -> int:
    """Count visible target pixels in a saved benchmark instance mask."""
    labels = json.loads(
        (observation_dir / "instance_labels.json").read_text(encoding="utf-8")
    )
    target_ids = [
        int(instance_id)
        for instance_id, value in labels.items()
        if value.get("class") == "target_red"
    ]
    ids = np.load(observation_dir / "instance_ids.npy")
    return int(np.isin(ids, target_ids).sum())


def learned_post_remove_localization(
    output_dir: Path,
    observation_dir: Path,
    *,
    observation_index: int,
    view: str = "post_remove",
    task_overrides: dict | None = None,
) -> dict:
    """Run learned perception, then backproject its mask with aligned RGB-D."""
    from rgbd_target_localization import localize_mask_files
    from run_scanned_basket_pipeline import (
        resolve_input_asset,
        run_current_observation_perception,
    )

    ranking, ranking_path, perception_config_path, stages = (
        run_current_observation_perception(
            session_dir=output_dir,
            observation_dir=observation_dir,
            view=view,
            step_index=observation_index,
            task_overrides=task_overrides,
            model_overrides=(
                FINAL_PERCEPTION_MODEL_OVERRIDES
                if task_overrides is FINAL_PERCEPTION_TASK_OVERRIDES
                else None
            ),
        )
    )
    selected_candidate = ranking.get("selected_candidate_id")
    if not selected_candidate:
        raise RuntimeError(
            "Learned post-remove perception did not select a target candidate"
        )
    model_input = json.loads(
        Path(ranking["input_path"]).read_text(encoding="utf-8")
    )
    candidates = {
        candidate["candidate_id"]: candidate
        for candidate in model_input["candidates"]
    }
    if selected_candidate not in candidates:
        raise ValueError(
            "Qwen selected a candidate that is absent from its input: "
            f"{selected_candidate}"
        )
    target_mask = resolve_input_asset(
        model_input,
        candidates[selected_candidate]["mask_path"],
    )
    localization = localize_mask_files(
        observation_dir,
        {"selected_target": target_mask},
    )
    localization.update(
        {
            "selection": {
                "selected_candidate_id": selected_candidate,
                "source_ranking": str(ranking_path.resolve()),
                "source_perception_config": str(
                    perception_config_path.resolve()
                ),
                "source_view": view,
                "mask_path": str(target_mask.resolve()),
                "simulator_ground_truth_used": False,
            },
            "perception_stages": stages,
            "learned_perception_executed": True,
            "simulator_ground_truth_used_for_estimate": False,
        }
    )
    localization_name = (
        "post_remove_rgbd_localization.json"
        if view == "post_remove"
        else f"post_remove_{view}_rgbd_localization.json"
    )
    localization_path = output_dir / localization_name
    write_json_atomic(localization_path, localization)
    return {
        "ranking": ranking,
        "ranking_path": str(ranking_path.resolve()),
        "perception_config_path": str(perception_config_path.resolve()),
        "selected_candidate_id": selected_candidate,
        "target_mask_path": str(target_mask.resolve()),
        "localization": localization,
        "localization_path": str(localization_path.resolve()),
        "stages": stages,
    }


def learned_scene_features(perception: dict) -> dict:
    """Extract the frozen post-cover selector inputs from one live result."""
    from PIL import Image

    ranking = perception["ranking"]
    config = json.loads(
        Path(perception["perception_config_path"]).read_text(encoding="utf-8")
    )
    output_root = Path(config["output_root"])
    detections_path = (
        output_root
        / "grounded_sam2"
        / ranking["sample_id"]
        / "detections.json"
    )
    detections = json.loads(detections_path.read_text(encoding="utf-8"))
    selected_index = ranking["candidate_ids"].index(
        ranking["selected_candidate_id"]
    )
    with Image.open(detections["image_path"]) as image:
        image_area = float(image.width * image.height)
    orange = [
        item for item in detections["annotations"]
        if item["label"] == "orange object"
    ]
    covers = [
        item for item in detections["annotations"]
        if item["label"] == "lid or cover"
    ]
    values = [
        float(ranking["raw_match_logits"][selected_index]),
        float(len(ranking["candidate_ids"])),
        max((bbox_aspect(item) for item in orange), default=0.0),
        max((float(item["score"]) for item in covers), default=0.0),
        max(
            (bbox_area_fraction(item, image_area) for item in covers),
            default=0.0,
        ),
    ]
    return {
        "sample_id": ranking["sample_id"],
        "values": values,
        "named_values": dict(zip(FEATURE_NAMES, values)),
        "inference_inputs": [
            str(detections_path.resolve()),
            str(Path(perception["ranking_path"]).resolve()),
        ],
    }


def select_live_post_cover_view(perception: dict) -> dict:
    """Apply the calibration-frozen future-observation selector live."""
    calibration = json.loads(
        POST_COVER_VIEW_CALIBRATION.read_text(encoding="utf-8")
    )
    query = learned_scene_features(perception)
    prediction = predict_variant_distribution(
        query,
        calibration["frozen_full_calibration_model"]["episodes"],
        k=int(calibration["frozen_full_calibration_model"]["neighbor_count"]),
    )
    decision = select_post_cover_action(prediction["variant_probabilities"])
    selected = decision["selected_action"]
    if selected not in {"viewpoint_close_high", "viewpoint_right"}:
        raise RuntimeError(f"Frozen selector did not choose a view: {selected}")
    return {
        "schema_version": "live-post-cover-view-decision-v1",
        "features": query,
        "prediction": prediction,
        "decision": decision,
        "calibration_source": str(POST_COVER_VIEW_CALIBRATION.resolve()),
        "future_observation_used_for_selection": False,
    }


def qwen_grasp_gate(perception: dict) -> dict:
    """Require positive identity evidence and an inside membership judgment."""
    ranking = perception["ranking"]
    selected = ranking["selected_candidate_id"]
    selected_index = ranking["candidate_ids"].index(selected)
    target_logit = float(ranking["raw_match_logits"][selected_index])
    membership = next(
        (
            relation
            for relation in ranking.get("relations", [])
            if relation.get("source_id") == selected
            and relation.get("relation_type") == "membership"
        ),
        None,
    )
    membership_label = membership.get("top_label") if membership else None
    authorized = target_logit > 0.0 and membership_label == "inside"
    return {
        "authorized": authorized,
        "selected_candidate_id": selected,
        "selected_target_logit": target_logit,
        "membership_label": membership_label,
        "thresholds": {
            "target_logit_strictly_greater_than": 0.0,
            "required_membership": "inside",
        },
        "calibrated_probability_used": False,
    }


def removal_contact_success(removal: dict) -> bool:
    """Apply the contact contract appropriate to a held or released cover.

    A released cover no longer has finger contact by design, so success shifts
    to verified support, retreat clearance, and post-release stability.
    """
    if removal.get("cover_placed_and_released"):
        placement = removal.get("supported_placement") or {}
        return bool(
            removal.get("bilateral_contact_before_release")
            and removal.get("contact_maintained_before_release")
            and placement.get("release_executed")
            and placement.get("retreat_executed")
            and placement.get("finger_contact_cleared_after_retreat")
            and placement.get("stable_after_release")
        )
    return bool(removal.get("bilateral_contact_after_lift"))


def contact_grasp_success(server_result: dict) -> bool:
    """Require physical lift and all contact/collision safety gates.

    Lift distance alone is insufficient because a wedged or over-penetrating
    grasp can move the target without being a safe bilateral grasp.
    """
    grasp = server_result.get("grasp_execution") or {}
    return bool(
        server_result.get("grasp_executed")
        and grasp.get("lift_verified")
        and grasp.get("bilateral_contact_before_lift")
        and not grasp.get("unexpected_environment_pairs")
        and grasp.get("contact_force_within_limit")
        and grasp.get("contact_penetration_within_limit")
    )


def transport_or_saved_result_succeeded(
    returncode: int | None, server_result: dict
) -> bool:
    """Do not discard a complete result because Kit timed out while closing."""
    return bool(
        returncode == 0 or server_result.get("status") == "completed"
    )


def calibration_authorizes_remove_cover(
    seed: int,
    *,
    scene_variant: str = "cover_removal_required",
    forced_observation_calibration: bool = False,
    final_evaluation_authorized: bool = False,
    protocol_path: Path = DEFAULT_PROTOCOL,
) -> dict:
    """Authorize calibration interventions without opening reserved test seeds."""
    if forced_observation_calibration:
        protocol = json.loads(protocol_path.resolve().read_text(encoding="utf-8"))
        split = protocol["data_split"]
        if "reserved_test_seeds" in split:
            reserved = {int(value) for value in split["reserved_test_seeds"]}
        else:
            first, last = [int(value) for value in split["reserved_test_seed_range"]]
            reserved = set(range(first, last + 1))
        if seed in reserved:
            raise ValueError(
                f"Seed {seed} is reserved for final testing and cannot be "
                "used for forced calibration"
            )
        return {
            "authorization_mode": "forced_observation_calibration_intervention",
            "seed": seed,
            "scene_variant": scene_variant,
            "future_observation_used_for_selection": False,
            "testing_performed": False,
            "valid_for_final_evaluation": False,
        }
    if final_evaluation_authorized:
        protocol = json.loads(protocol_path.resolve().read_text(encoding="utf-8"))
        frozen_path = ROOT / protocol["calibration_freeze_requirements"][
            "frozen_parameters_path"
        ]
        frozen = json.loads(frozen_path.read_text(encoding="utf-8"))
        if frozen.get("status") not in {
            "frozen_before_reserved_test",
            "frozen_ready_for_reserved_test",
        }:
            raise RuntimeError("Final parameters are not frozen")
        if not frozen.get("reserved_test_launch_authorized"):
            raise RuntimeError("Reserved test launch is not authorized")
        return {
            "authorization_mode": "frozen_reserved_test_policy",
            "seed": seed,
            "selected_action": "remove_cover",
            "source": str(frozen_path.resolve()),
            "future_observation_used_for_selection": False,
            "testing_performed": True,
            "valid_for_final_evaluation": True,
        }
    result = json.loads(CALIBRATION_RESULT.read_text(encoding="utf-8"))
    matches = [
        fold for fold in result["folds"] if int(fold["held_out_seed"]) == seed
    ]
    if len(matches) != 1:
        raise ValueError(f"Expected one calibration fold for seed {seed}")
    fold = matches[0]
    if fold["selected_action"] != "remove_cover":
        raise RuntimeError(
            f"Calibration did not authorize remove_cover for seed {seed}: {fold}"
        )
    return fold


def main() -> None:
    """Run one persistent cover-removal, re-observation, and grasp episode."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=188)
    parser.add_argument(
        "--scene-variant",
        choices=COVERED_SCENE_VARIANTS,
        default="cover_removal_required",
    )
    parser.add_argument("--timeout-seconds", type=float, default=1800.0)
    parser.add_argument(
        "--rg6-lid-calibration-config",
        type=Path,
        help="Completed lab-measured RG6/lid calibration file.",
    )
    parser.add_argument(
        "--require-transfer-ready-physics",
        action="store_true",
        help="Refuse provisional RG6/lid physics before launching Isaac Sim.",
    )
    parser.add_argument(
        "--allow-provisional-rg6-lid-physics",
        action="store_true",
        help=(
            "Development only: use an explicit public-spec/simulation proxy "
            "without claiming transfer readiness."
        ),
    )
    parser.add_argument(
        "--final-evaluation-authorized",
        action="store_true",
        help="Write a reserved test episode under the frozen final protocol.",
    )
    parser.add_argument(
        "--protocol",
        type=Path,
        default=DEFAULT_PROTOCOL,
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=OUTPUT_ROOT,
    )
    parser.add_argument(
        "--rg6-coupling-mode",
        choices=("passive_mimic", "coordinated_drives"),
        default="passive_mimic",
    )
    parser.add_argument(
        "--coordinated-rg6-total-drive-effort-limit-nm",
        type=float,
    )
    parser.add_argument(
        "--replan-after-remove-cover",
        action="store_true",
        help=(
            "Consume the physical post-remove observation, update the "
            "cover-search belief, and send the replanned second action while "
            "the same Isaac process remains alive."
        ),
    )
    parser.add_argument(
        "--execute-post-remove-grasp",
        action="store_true",
        help=(
            "After cover removal, run GroundingDINO, SAM2.1, and Qwen on "
            "the new RGB-D observation, replan, and physically execute the "
            "selected target grasp in the same Isaac process."
        ),
    )
    parser.add_argument(
        "--execute-post-remove-active-view-grasp",
        action="store_true",
        help=(
            "Run frozen post-cover view selection, physically move to that "
            "view, infer again, and execute a contact-gated target grasp."
        ),
    )
    parser.add_argument(
        "--counterfactual-post-remove-viewpoint-right-cache",
        action="store_true",
        help=(
            "Cache the reachable right observation after physical cover "
            "removal, send stop, and do not count the run as a policy result."
        ),
    )
    parser.add_argument(
        "--counterfactual-post-remove-views",
        nargs="+",
        choices=("left", "close_high", "right"),
        help=(
            "Cache one or both reachable views after physical cover removal, "
            "then stop without counting the run as a policy result."
        ),
    )
    parser.add_argument(
        "--force-remove-cover-for-observation-calibration",
        action="store_true",
        help=(
            "Calibration only: execute the remove-cover intervention without "
            "claiming that MPC selected it. Reserved test seeds are rejected."
        ),
    )
    parser.add_argument(
        "--disable-manipulation-video",
        action="store_true",
        help=(
            "Skip per-step manipulation frames and MP4 encoding while "
            "retaining RGB-D observations and all physics safety checks."
        ),
    )
    parser.add_argument(
        "--physics-only-manipulation-steps",
        action="store_true",
        help=(
            "Use renderer-free PhysX stepping during manipulation while "
            "retaining rendered RGB-D observations at decision points."
        ),
    )
    args = parser.parse_args()
    if (
        args.counterfactual_post_remove_viewpoint_right_cache
        and args.counterfactual_post_remove_views
    ):
        parser.error("Use only one counterfactual post-remove view option")
    counterfactual_views = list(
        args.counterfactual_post_remove_views
        or (
            ["right"]
            if args.counterfactual_post_remove_viewpoint_right_cache
            else []
        )
    )
    if len(counterfactual_views) != len(set(counterfactual_views)):
        parser.error("Counterfactual post-remove views must be unique")
    counterfactual_cache_only = bool(counterfactual_views)
    if args.timeout_seconds <= 0.0:
        parser.error("--timeout-seconds must be positive")
    execute_final_grasp = bool(
        args.execute_post_remove_grasp
        or args.execute_post_remove_active_view_grasp
    )
    if execute_final_grasp and not args.replan_after_remove_cover:
        parser.error(
            "Post-remove grasp execution requires "
            "--replan-after-remove-cover"
        )
    if (
        args.execute_post_remove_grasp
        and args.execute_post_remove_active_view_grasp
    ):
        parser.error("Choose direct grasp or active-view grasp, not both")
    if (
        counterfactual_cache_only
        and not args.replan_after_remove_cover
    ):
        parser.error(
            "--counterfactual-post-remove-viewpoint-right-cache requires "
            "--replan-after-remove-cover"
        )
    if (
        counterfactual_cache_only
        and execute_final_grasp
    ):
        parser.error(
            "Counterfactual cache capture cannot execute a terminal grasp"
        )
    if (
        counterfactual_cache_only
        and not (
            args.final_evaluation_authorized
            or args.force_remove_cover_for_observation_calibration
        )
    ):
        parser.error(
            "Counterfactual cache capture requires final-evaluation "
            "authorization or an explicit calibration intervention"
        )
    if args.require_transfer_ready_physics and not args.rg6_lid_calibration_config:
        parser.error(
            "--require-transfer-ready-physics requires "
            "--rg6-lid-calibration-config"
        )
    if (
        args.allow_provisional_rg6_lid_physics
        and not args.rg6_lid_calibration_config
    ):
        parser.error(
            "--allow-provisional-rg6-lid-physics requires "
            "--rg6-lid-calibration-config"
        )
    if (
        args.allow_provisional_rg6_lid_physics
        and args.require_transfer_ready_physics
    ):
        parser.error(
            "Provisional and transfer-ready physics modes are mutually exclusive"
        )
    if args.rg6_coupling_mode == "coordinated_drives" and not (
        args.allow_provisional_rg6_lid_physics
        and args.coordinated_rg6_total_drive_effort_limit_nm is not None
    ):
        parser.error(
            "coordinated_drives requires provisional physics and an explicit "
            "aggregate drive-effort limit"
        )
    if (
        args.rg6_coupling_mode == "passive_mimic"
        and args.coordinated_rg6_total_drive_effort_limit_nm is not None
    ):
        parser.error(
            "coordinated drive-effort limit requires coordinated_drives"
        )
    calibration_path = None
    calibration_report = None
    if args.rg6_lid_calibration_config is not None:
        from rg6_lid_calibration import load_json, validate_calibration

        calibration_path = args.rg6_lid_calibration_config
        if not calibration_path.is_absolute():
            calibration_path = ROOT / calibration_path
        calibration_report = validate_calibration(load_json(calibration_path))
        calibration_authorized = bool(
            calibration_report["transfer_ready"]
            or (
                args.allow_provisional_rg6_lid_physics
                and calibration_report["development_proxy_usable"]
            )
        )
        if not calibration_authorized:
            parser.error(
                "RG6/lid calibration is not transfer-ready: "
                f"{calibration_report}"
            )
    require_single_gpu_policy()
    physical_gpu = configured_physical_gpu()
    fold = calibration_authorizes_remove_cover(
        args.seed,
        scene_variant=args.scene_variant,
        forced_observation_calibration=(
            args.force_remove_cover_for_observation_calibration
        ),
        final_evaluation_authorized=args.final_evaluation_authorized,
        protocol_path=args.protocol,
    )
    output_root = args.output_root.resolve()
    output_dir = next_output_dir(args.seed, output_root)
    validate_output_authorization(
        seed=args.seed,
        output_dir=output_dir.resolve(),
        final_evaluation_authorized=args.final_evaluation_authorized,
        protocol_path=args.protocol.resolve(),
    )
    output_dir.mkdir(parents=True, exist_ok=False)

    base_config = json.loads(
        (
            ROOT / "configs" / "sim" / "closed_loop_execution.json"
        ).read_text(encoding="utf-8")
    )
    base_config["viewpoint_execution"]["mode"] = "interpolated_joint_physics"
    base_config["viewpoint_execution"]["debug_ee_positions_world_m"] = {}
    method_path = output_dir / "effective_method_config.json"
    write_json_atomic(method_path, base_config)

    command = [
        str(ISAAC_PYTHON),
        str(ROOT / "scripts" / "isaac_sim_server.py"),
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
        str(output_dir),
        "--method-config",
        str(method_path),
        "--seed",
        str(args.seed),
        "--tabletop-scene",
        "--scanned-basket-scene",
        "--calibration-scene-variant",
        args.scene_variant,
        "--basket-contact-physics",
        "--execute-persistent-remove-cover",
    ]
    if calibration_path is not None:
        command.extend(
            ["--rg6-lid-calibration-config", str(calibration_path)]
        )
    if args.require_transfer_ready_physics:
        command.append("--require-transfer-ready-physics")
    if args.allow_provisional_rg6_lid_physics:
        command.append("--allow-provisional-rg6-lid-physics")
    if args.rg6_coupling_mode == "coordinated_drives":
        command.extend(
            [
                "--rg6-coupling-mode",
                "coordinated_drives",
                "--coordinated-rg6-total-drive-effort-limit-nm",
                str(args.coordinated_rg6_total_drive_effort_limit_nm),
            ]
        )
    if args.disable_manipulation_video:
        command.append("--disable-manipulation-video")
    if args.physics_only_manipulation_steps:
        command.append("--physics-only-manipulation-steps")
    if args.replan_after_remove_cover:
        command.append("--continue-after-remove-cover")
    if args.final_evaluation_authorized:
        command.extend(
            [
                "--final-evaluation-authorized",
                "--final-evaluation-protocol",
                str(args.protocol.resolve()),
            ]
        )
    environment = dict(os.environ)
    # The renderer keeps the host's physical index, while the one exposed CUDA
    # device is numbered 0 inside the process for PhysX.
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

    stdout_path = output_dir / "isaac_stdout.log"
    stderr_path = output_dir / "isaac_stderr.log"
    stdout = stdout_path.open("w", encoding="utf-8")
    stderr = stderr_path.open("w", encoding="utf-8")
    started = time.perf_counter()
    # One server owns capture, removal, reobservation, and any follow-up grasp;
    # keeping it alive preserves the released cover and articulation state.
    server = subprocess.Popen(
        command,
        cwd=ROOT,
        env=environment,
        stdout=stdout,
        stderr=stderr,
        text=True,
    )
    try:
        initial_event_path = output_dir / "observation_ready_000.json"
        # Concurrent RTX startup on the shared server can exceed ten minutes.
        # Keep startup bounded, but do not confuse shader/extension contention
        # with an episode failure.
        wait_for_path(
            initial_event_path,
            server,
            timeout=min(args.timeout_seconds, 1200.0),
        )
        initial_event = json.loads(initial_event_path.read_text(encoding="utf-8"))
        write_json_atomic(
            output_dir / "action_request_000.json",
            {
                "schema_version": "live-remove-cover-request-v1",
                "index": 0,
                "type": "remove_cover",
                "interface_target": "cover_01",
                "source_calibration_result": (
                    None
                    if args.force_remove_cover_for_observation_calibration
                    else str(CALIBRATION_RESULT)
                ),
                "source_calibration_fold": fold,
                "forced_observation_calibration_intervention": (
                    args.force_remove_cover_for_observation_calibration
                ),
                "future_observation_used_for_selection": False,
            },
        )
        replanning = None
        learned_perception = None
        active_view_decision = None
        final_view_perception = None
        final_grasp_gate = None
        if args.replan_after_remove_cover:
            post_event_path = output_dir / "observation_ready_001.json"
            wait_for_path(
                post_event_path,
                server,
                timeout=args.timeout_seconds,
            )
            post_event_live = json.loads(
                post_event_path.read_text(encoding="utf-8")
            )
            post_observation_dir = Path(post_event_live["observation_dir"])
            post_pixels_live = target_pixel_count(post_observation_dir)
            if execute_final_grasp:
                learned_perception = learned_post_remove_localization(
                    output_dir,
                    post_observation_dir,
                    observation_index=1,
                    task_overrides=(
                        FINAL_PERCEPTION_TASK_OVERRIDES
                        if (
                            args.final_evaluation_authorized
                            or args.execute_post_remove_active_view_grasp
                        )
                        else None
                    ),
                )
                outcome = "target_detected"
            else:
                outcome = (
                    "target_detected"
                    if post_pixels_live >= 100
                    else "empty_container"
                )
            if args.execute_post_remove_active_view_grasp:
                active_view_decision = select_live_post_cover_view(
                    learned_perception
                )
                replanned_policy = active_view_decision["decision"]
                belief_before = None
                belief_update = {"posterior": None}
            elif args.final_evaluation_authorized:
                protocol = json.loads(
                    args.protocol.resolve().read_text(encoding="utf-8")
                )
                planner_path = ROOT / protocol.get(
                    "frozen_cover_planner_path",
                    str(FROZEN_COVER_PLANNER.relative_to(ROOT)),
                )
            else:
                planner_path = (
                    ROOT / "configs/research/cover_search_planner.json"
                )
            if not args.execute_post_remove_active_view_grasp:
                planner_config = json.loads(
                    planner_path.read_text(encoding="utf-8")
                )
                validate_config(planner_config)
                belief_before = normalize(planner_config["initial_belief"])
                belief_update = execute_observation_action(
                    belief_before,
                    "remove_cover",
                    outcome,
                    planner_config,
                )
                replanned_policy = plan(
                    belief_update["posterior"], planner_config
                )
            replanning = {
                "schema_version": "live-post-remove-replan-v1",
                "observation_index": 1,
                "post_action_observation": outcome,
                "post_action_target_visible_pixels": post_pixels_live,
                "belief_before": belief_before,
                "belief_after": belief_update["posterior"],
                "planner": replanned_policy,
                "selected_action": replanned_policy["selected_action"],
                "active_view_decision": active_view_decision,
                "simulator_instance_mask_used_for_validation_control": (
                    not execute_final_grasp
                ),
                "learned_perception_executed": bool(learned_perception),
                "learned_perception": learned_perception,
                "valid_for_final_evaluation": bool(
                    args.final_evaluation_authorized
                    and not counterfactual_cache_only
                ),
            }
            dispatched_action = (
                f"viewpoint_{counterfactual_views[0]}"
                if counterfactual_cache_only
                else replanned_policy["selected_action"]
            )
            replanning["dispatched_action"] = dispatched_action
            replanning["counterfactual_cache_only"] = bool(
                counterfactual_cache_only
            )
            replanning["counterfactual_views"] = counterfactual_views
            write_json_atomic(
                output_dir / "post_remove_replan.json", replanning
            )
            write_json_atomic(
                output_dir / "action_request_001.json",
                {
                    "schema_version": "live-post-remove-action-request-v1",
                    "index": 1,
                    "type": dispatched_action,
                    "source_replan": str(
                        output_dir / "post_remove_replan.json"
                    ),
                    "rgbd_localization_path": (
                        learned_perception["localization_path"]
                        if learned_perception is not None
                        else None
                    ),
                    "physical_execution_requested": bool(
                        args.execute_post_remove_grasp
                        or args.execute_post_remove_active_view_grasp
                    ),
                },
            )
            if args.execute_post_remove_active_view_grasp:
                view_event_path = output_dir / "observation_ready_002.json"
                wait_for_path(
                    view_event_path,
                    server,
                    timeout=args.timeout_seconds,
                )
                view_event = json.loads(
                    view_event_path.read_text(encoding="utf-8")
                )
                selected_view = dispatched_action.removeprefix("viewpoint_")
                if view_event.get("view") != selected_view:
                    raise RuntimeError(
                        f"Unexpected selected view: {view_event.get('view')} "
                        f"!= {selected_view}"
                    )
                final_view_perception = learned_post_remove_localization(
                    output_dir,
                    Path(view_event["observation_dir"]),
                    observation_index=2,
                    view=selected_view,
                    task_overrides=FINAL_PERCEPTION_TASK_OVERRIDES,
                )
                final_grasp_gate = qwen_grasp_gate(final_view_perception)
                terminal_type = (
                    "grasp_inside"
                    if final_grasp_gate["authorized"]
                    else "stop"
                )
                write_json_atomic(
                    output_dir / "action_request_002.json",
                    {
                        "schema_version": "live-post-view-action-request-v1",
                        "index": 2,
                        "type": terminal_type,
                        "source_view_decision": str(
                            output_dir / "post_remove_replan.json"
                        ),
                        "qwen_grasp_gate": final_grasp_gate,
                        "rgbd_localization_path": (
                            final_view_perception["localization_path"]
                            if final_grasp_gate["authorized"]
                            else None
                        ),
                        "physical_execution_requested": bool(
                            final_grasp_gate["authorized"]
                        ),
                    },
                )
            if counterfactual_cache_only:
                for view_offset in range(len(counterfactual_views)):
                    event_index, expected_view, request = (
                        counterfactual_cache_step(
                            counterfactual_views, view_offset
                        )
                    )
                    view_event_path = (
                        output_dir
                        / f"observation_ready_{event_index:03d}.json"
                    )
                    wait_for_path(
                        view_event_path,
                        server,
                        timeout=args.timeout_seconds,
                    )
                    view_event = json.loads(
                        view_event_path.read_text(encoding="utf-8")
                    )
                    if view_event.get("view") != expected_view:
                        raise RuntimeError(
                            "Unexpected counterfactual view: "
                            f"{view_event.get('view')} != {expected_view}"
                        )
                    write_json_atomic(
                        output_dir
                        / f"action_request_{event_index:03d}.json",
                        request,
                    )
        wait_for_path(
            output_dir / "server_result.json",
            server,
            timeout=args.timeout_seconds,
        )
        server.wait(timeout=90.0)
    finally:
        if server.poll() is None:
            server.terminate()
            server.wait(timeout=30.0)
        stdout.close()
        stderr.close()

    server_result = json.loads(
        (output_dir / "server_result.json").read_text(encoding="utf-8")
    )
    post_event_path = output_dir / "observation_ready_001.json"
    post_event = (
        json.loads(post_event_path.read_text(encoding="utf-8"))
        if post_event_path.is_file()
        else None
    )
    initial_dir = Path(initial_event["observation_dir"])
    post_dir = Path(post_event["observation_dir"]) if post_event else None
    initial_pixels = target_pixel_count(initial_dir)
    post_pixels = target_pixel_count(post_dir) if post_dir else 0
    expected_visibility_outcome, visibility_outcome_passed = (
        expected_visibility_outcome_passed(
            args.scene_variant,
            initial_pixels,
            post_pixels,
        )
    )
    removal = server_result.get("cover_removal_execution") or {}
    final_grasp = server_result.get("grasp_execution") or {}
    final_grasp_success = contact_grasp_success(server_result)
    # Accept the execution only when manipulation evidence and the expected scene
    # change agree. Contact alone cannot prove the cover revealed the target,
    # and a visibility increase cannot prove the motion was physically safe.
    success = bool(
        transport_or_saved_result_succeeded(server.returncode, server_result)
        and server_result.get("status") == "completed"
        and server_result.get("cover_removal_executed")
        and server_result.get("post_removal_observation_generated")
        and removal.get("removal_verified")
        and removal.get("bilateral_contact_before_lift")
        and removal_contact_success(removal)
        and not removal.get("unexpected_environment_pairs")
        and removal.get("contact_force_within_limit")
        and removal.get("contact_penetration_within_limit")
        and visibility_outcome_passed
        and (
            not execute_final_grasp
            or final_grasp_success
        )
    )
    result = {
        "schema_version": "cover-removal-execution-v1",
        "status": "completed" if success else "failed",
        "seed": args.seed,
        "runtime_seconds": time.perf_counter() - started,
        "transport_returncode": server.returncode,
        "transport_exit_warning": bool(
            server.returncode not in (0, None)
            and server_result.get("status") == "completed"
        ),
        "root_action": "remove_cover",
        "root_action_source": (
            "forced_observation_calibration_intervention"
            if args.force_remove_cover_for_observation_calibration
            else (
                "frozen_counterfactual_cache_policy"
                if counterfactual_cache_only
                else (
                    "frozen_reserved_test_policy"
                    if args.final_evaluation_authorized
                    else "scene_conditioned_calibration_model"
                )
            )
        ),
        "future_observation_used_for_root_selection": False,
        "initial_target_visible_pixel_count": initial_pixels,
        "post_removal_target_visible_pixel_count": post_pixels,
        "target_visibility_gain_pixels": post_pixels - initial_pixels,
        "expected_visibility_outcome": expected_visibility_outcome,
        "visibility_outcome_passed": visibility_outcome_passed,
        "cover_removal_execution": removal,
        "post_removal_observation": str(post_dir) if post_dir else None,
        "gpu_policy": {
            "physical_gpu": physical_gpu,
            "renderer_active_gpu": physical_gpu,
            "physics_cuda_device": 0,
            "multi_gpu": False,
        },
        "rg6_lid_physics": {
            "mode": (
                "lab_measured_calibration"
                if (
                    calibration_report is not None
                    and calibration_report["transfer_ready"]
                )
                else (
                    "explicit_provisional_public_spec"
                    if calibration_report is not None
                    else "implicit_provisional_development"
                )
            ),
            "calibration_config": (
                str(calibration_path) if calibration_path is not None else None
            ),
            "calibration_report": calibration_report,
            "transfer_ready_required": args.require_transfer_ready_physics,
            "provisional_explicitly_allowed": (
                args.allow_provisional_rg6_lid_physics
            ),
        },
        "training_performed": False,
        "calibration_performed": not args.final_evaluation_authorized,
        "testing_performed": bool(
            args.final_evaluation_authorized
            and not counterfactual_cache_only
        ),
        "reserved_test_seeds_used": args.final_evaluation_authorized,
        "negative_evidence_update_performed": False,
        "replanning_performed": replanning is not None,
        "post_remove_replan": replanning,
        "post_remove_replanned_action": (
            replanning["selected_action"]
            if replanning is not None
            else None
        ),
        "post_remove_learned_perception": learned_perception,
        "post_remove_active_view_decision": active_view_decision,
        "final_view_learned_perception": final_view_perception,
        "final_grasp_gate": final_grasp_gate,
        "post_remove_replanned_action_physical_execution": bool(
            execute_final_grasp and final_grasp_success
        ),
        "final_grasp_executed": final_grasp_success,
        "final_grasp_execution": final_grasp,
        "counterfactual_cache_only": bool(
            counterfactual_cache_only
        ),
        "counterfactual_views": counterfactual_views,
        "valid_for_final_evaluation": bool(
            args.final_evaluation_authorized
            and success
            and not counterfactual_cache_only
        ),
    }
    write_json_atomic(output_dir / "cover_removal_result.json", result)
    print(f"REMOVE_COVER_RESULT={output_dir / 'cover_removal_result.json'}")
    if not success:
        raise RuntimeError(f"Remove-cover execution failed: {result}")


if __name__ == "__main__":
    main()
