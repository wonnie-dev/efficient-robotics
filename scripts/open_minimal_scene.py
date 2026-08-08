import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
from isaacsim import SimulationApp


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RUNTIME_LOG = PROJECT_ROOT / "runtime_scene_status.log"
ACTION_REQUEST_PATH = PROJECT_ROOT / "outputs" / "active_view_controller" / "action_request.json"
ACTION_EXECUTION_PATH = (
    PROJECT_ROOT / "outputs" / "active_view_controller" / "action_execution.json"
)

parser = argparse.ArgumentParser()
parser.add_argument(
    "--execute-action-request",
    action="store_true",
    help="Capture center, run the controller, execute its selected view, and recapture.",
)
parser.add_argument(
    "--execute-non-oracle-plan",
    action="store_true",
    help=(
        "Benchmark only: plan from center without future captures, execute the "
        "selected viewpoint, update belief from the new observation, and replan."
    ),
)
parser.add_argument(
    "--method-config",
    type=Path,
    default=(
        PROJECT_ROOT
        / "configs"
        / "research"
        / "initial_method_design.json"
    ),
    help="Research-method config used by --execute-non-oracle-plan.",
)
parser.add_argument(
    "--movement-demo",
    action="store_true",
    help="Run a slow center-left-center-right-center motion for visual inspection.",
)
parser.add_argument(
    "--meeting-demo",
    action="store_true",
    help=(
        "Execute the cached deterministic Qwen viewpoint decision, capture the "
        "new RGB-D observation, replan, and save a continuous overview MP4."
    ),
)
parser.add_argument(
    "--candidate-view-demo",
    action="store_true",
    help=(
        "Benchmark only: execute center-to-close_high as a provisional "
        "simulation wrist-view candidate and save RGB-D plus an overview MP4."
    ),
)
parser.add_argument(
    "--seeded-pilot-capture",
    action="store_true",
    help=(
        "Benchmark only: apply a deterministic relation-preserving scene seed "
        "and capture left/center/right/close_high RGB-D."
    ),
)
parser.add_argument(
    "--household-perception-pilot",
    action="store_true",
    help=(
        "Seeded benchmark only: replace the colored target/container visuals "
        "with procedural red mugs and an open slatted basket, then save the "
        "capture under a separate household-perception pilot directory."
    ),
)
parser.add_argument(
    "--scanned-basket-perception-pilot",
    action="store_true",
    help=(
        "Household pilot only: replace the procedural slatted basket with the "
        "locally available textured LIBERO scanned-basket mesh."
    ),
)
parser.add_argument(
    "--active-occlusion-pilot",
    action="store_true",
    help=(
        "Scanned-basket pilot only: enlarge and raise the orange occluder so "
        "the initial center observation is deliberately less informative."
    ),
)
parser.add_argument(
    "--calibration-scene-variant",
    choices=(
        "auto",
        "inside_clear",
        "outside",
        "rim_occluded",
        "covered_unknown",
        "behind_ambiguous",
        "behind_boundary_unknown",
        "close_high_only",
        "right_only",
        "either_view",
        "cover_removal_required",
        "empty_cover_then_right",
    ),
    help=(
        "Scanned-basket pilot only: deterministic factorized calibration scene. "
        "'auto' cycles variants by seed."
    ),
)
parser.add_argument(
    "--basket-collision-physics-pilot",
    action="store_true",
    help=(
        "Scanned-basket pilot only: add a documented static bottom-and-four-"
        "walls collision approximation for RG6 physics validation."
    ),
)
parser.add_argument(
    "--live-pipeline-server",
    action="store_true",
    help=(
        "Benchmark only: keep Isaac Sim alive while an external VLM process "
        "requests viewpoints and replanning through a session directory."
    ),
)
parser.add_argument(
    "--actual-view-motion",
    action="store_true",
    help=(
        "Live pipeline only: author the initial center JointState before the "
        "first physics frame and execute requested views with UR10e joints."
    ),
)
parser.add_argument(
    "--execute-persistent-composite-grasp",
    action="store_true",
    help=(
        "After a live terminal grasp request, execute the validated UR10e+RG6 "
        "bilateral-contact grasp in this same Isaac process and stage."
    ),
)
parser.add_argument(
    "--execute-persistent-remove-cover",
    action="store_true",
    help=(
        "After a live remove_cover request, use the existing UR10e+RG6 "
        "articulation to grasp the dynamic cover handle, lift it, move it "
        "beside the basket, and capture a new RGB-D observation."
    ),
)
parser.add_argument(
    "--rg6-lid-calibration-config",
    type=Path,
    help=(
        "Completed lab-measured RG6/lid calibration. Supplying this file "
        "enables measured cover geometry and control parameters only if its "
        "transfer-readiness gate passes before Isaac Sim starts."
    ),
)
parser.add_argument(
    "--require-transfer-ready-physics",
    action="store_true",
    help=(
        "Fail before starting Isaac Sim unless a completed, validated "
        "--rg6-lid-calibration-config is supplied."
    ),
)
parser.add_argument(
    "--allow-provisional-rg6-lid-physics",
    action="store_true",
    help=(
        "Development only: allow a validated provisional_public_spec file. "
        "Outputs remain invalid for transfer-ready or final-evaluation claims."
    ),
)
parser.add_argument(
    "--rg6-coupling-mode",
    choices=("passive_mimic", "coordinated_drives"),
    default="passive_mimic",
    help=(
        "Development coupling for persistent RG6 manipulation. Coordinated "
        "drives are provisional and cannot be transfer-ready."
    ),
)
parser.add_argument(
    "--coordinated-rg6-total-drive-effort-limit-nm",
    type=float,
    help=(
        "Development-only aggregate six-joint drive effort; not real RG6 "
        "motor torque or commanded grip force."
    ),
)
parser.add_argument(
    "--continue-after-remove-cover",
    action="store_true",
    help=(
        "After physical cover removal and post-action RGB-D capture, wait for "
        "one replanned action request before closing the live server. The "
        "replanned action is recorded but is not physically executed yet."
    ),
)
parser.add_argument(
    "--persistent-grasp-height-offset-m",
    type=float,
    default=0.0,
    help=(
        "Debug physics pilot only: stop at a validated higher descent waypoint "
        "to grasp the upper part of a tall target inside a narrow container."
    ),
)
parser.add_argument(
    "--rg6-physics-smoke",
    action="store_true",
    help=(
        "Attach the RG6 physical joint graph to the UR10e end effector, "
        "exercise its finger DOFs, save diagnostics, and exit."
    ),
)
parser.add_argument("--live-session-dir", type=Path)
parser.add_argument("--seed", type=int, default=0)
parser.add_argument(
    "--pilot-result",
    type=Path,
    default=(
        PROJECT_ROOT
        / "outputs"
        / "single_gpu_pilot"
        / "benchmark_seed000"
        / "episode.json"
    ),
)
parser.add_argument(
    "--scene-profile",
    choices=("minimal", "benchmark"),
    default="minimal",
)
parser.add_argument(
    "--headless",
    action="store_true",
    help="Run without a GUI and exit after the requested capture work completes.",
)
parser.add_argument(
    "--renderer-gpu",
    type=int,
    default=0,
    help="Physical Vulkan/Omniverse GPU index used by the renderer.",
)
parser.add_argument(
    "--physics-gpu",
    type=int,
    default=0,
    help="CUDA-visible GPU index used by PhysX; normally cuda:0 after masking.",
)
parser.add_argument(
    "--capture-video",
    action="store_true",
    help="Build a short CPU-encoded MP4 from the captured observation RGB frames.",
)
parser.add_argument(
    "--render-quality",
    choices=("preview", "paper"),
    default="preview",
    help=(
        "preview preserves the normal experiment outputs; paper renders a "
        "separate 1080p overview with higher sampling and video quality."
    ),
)
args, _unknown = parser.parse_known_args()
if args.scene_profile == "benchmark" and args.execute_action_request:
    parser.error(
        "Benchmark active-view execution is disabled until its multi-object "
        "segmentation and graph pipeline are implemented."
    )
if args.execute_non_oracle_plan and args.scene_profile != "benchmark":
    parser.error("--execute-non-oracle-plan requires --scene-profile benchmark")
if args.execute_non_oracle_plan and args.execute_action_request:
    parser.error("Select only one execution mode")
if args.movement_demo and (
    args.execute_non_oracle_plan
    or args.execute_action_request
    or args.meeting_demo
    or args.candidate_view_demo
    or args.seeded_pilot_capture
    or args.live_pipeline_server
):
    parser.error("Movement demo cannot be combined with another execution mode")
if args.meeting_demo and (
    args.execute_non_oracle_plan
    or args.execute_action_request
    or args.candidate_view_demo
    or args.seeded_pilot_capture
    or args.live_pipeline_server
):
    parser.error("Meeting demo cannot be combined with another execution mode")
if args.meeting_demo and args.scene_profile != "benchmark":
    parser.error("--meeting-demo requires --scene-profile benchmark")
if args.candidate_view_demo and (
    args.execute_non_oracle_plan
    or args.execute_action_request
    or args.seeded_pilot_capture
    or args.live_pipeline_server
):
    parser.error("Candidate-view demo cannot be combined with another execution mode")
if args.candidate_view_demo and args.scene_profile != "benchmark":
    parser.error("--candidate-view-demo requires --scene-profile benchmark")
if args.seeded_pilot_capture and (
    args.execute_non_oracle_plan
    or args.execute_action_request
    or args.live_pipeline_server
):
    parser.error("Seeded pilot capture cannot be combined with another execution mode")
if args.seeded_pilot_capture and args.scene_profile != "benchmark":
    parser.error("--seeded-pilot-capture requires --scene-profile benchmark")
if args.household_perception_pilot and not (
    args.seeded_pilot_capture or args.live_pipeline_server
):
    parser.error(
        "--household-perception-pilot requires --seeded-pilot-capture "
        "or --live-pipeline-server"
    )
if (
    args.scanned_basket_perception_pilot
    and not args.household_perception_pilot
):
    parser.error(
        "--scanned-basket-perception-pilot requires "
        "--household-perception-pilot"
    )
if args.active_occlusion_pilot and not args.scanned_basket_perception_pilot:
    parser.error(
        "--active-occlusion-pilot requires "
        "--scanned-basket-perception-pilot"
    )
if (
    args.calibration_scene_variant is not None
    and not args.scanned_basket_perception_pilot
):
    parser.error(
        "--calibration-scene-variant requires "
        "--scanned-basket-perception-pilot"
    )
if args.active_occlusion_pilot and args.calibration_scene_variant is not None:
    parser.error(
        "--active-occlusion-pilot and --calibration-scene-variant "
        "are mutually exclusive"
    )
if (
    args.basket_collision_physics_pilot
    and not args.scanned_basket_perception_pilot
):
    parser.error(
        "--basket-collision-physics-pilot requires "
        "--scanned-basket-perception-pilot"
    )
if args.live_pipeline_server and (
    args.execute_non_oracle_plan or args.execute_action_request
):
    parser.error("Live pipeline server cannot be combined with another execution mode")
if args.live_pipeline_server and args.scene_profile != "benchmark":
    parser.error("--live-pipeline-server requires --scene-profile benchmark")
if args.live_pipeline_server and args.live_session_dir is None:
    parser.error("--live-pipeline-server requires --live-session-dir")
if args.live_session_dir is not None and not args.live_pipeline_server:
    parser.error("--live-session-dir requires --live-pipeline-server")
if args.actual_view_motion and not args.live_pipeline_server:
    parser.error("--actual-view-motion requires --live-pipeline-server")
if args.execute_persistent_composite_grasp and not args.live_pipeline_server:
    parser.error(
        "--execute-persistent-composite-grasp requires --live-pipeline-server"
    )
if args.execute_persistent_remove_cover and not args.live_pipeline_server:
    parser.error(
        "--execute-persistent-remove-cover requires --live-pipeline-server"
    )
if args.continue_after_remove_cover and not args.execute_persistent_remove_cover:
    parser.error(
        "--continue-after-remove-cover requires "
        "--execute-persistent-remove-cover"
    )
if (
    args.execute_persistent_composite_grasp
    and args.execute_persistent_remove_cover
):
    parser.error("Select only one persistent manipulation executor")
if args.execute_persistent_remove_cover and (
    args.calibration_scene_variant
    not in ("cover_removal_required", "empty_cover_then_right")
):
    parser.error(
        "--execute-persistent-remove-cover requires "
        "--calibration-scene-variant cover_removal_required or "
        "empty_cover_then_right"
    )
if args.rg6_lid_calibration_config and not args.execute_persistent_remove_cover:
    parser.error(
        "--rg6-lid-calibration-config requires "
        "--execute-persistent-remove-cover"
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
        "Provisional and transfer-ready RG6/lid physics modes are mutually exclusive"
    )
coordinated_cover_proxy = (
    args.execute_persistent_remove_cover
    and args.allow_provisional_rg6_lid_physics
)
coordinated_target_physics_smoke = (
    args.execute_persistent_composite_grasp
    and not args.execute_persistent_remove_cover
)
if args.rg6_coupling_mode == "coordinated_drives" and not (
    args.coordinated_rg6_total_drive_effort_limit_nm is not None
    and (coordinated_cover_proxy or coordinated_target_physics_smoke)
):
    parser.error(
        "--rg6-coupling-mode coordinated_drives requires either provisional "
        "remove-cover execution or a target-only persistent physics smoke, "
        "plus an explicit aggregate effort limit"
    )
if (
    args.rg6_coupling_mode == "passive_mimic"
    and args.coordinated_rg6_total_drive_effort_limit_nm is not None
):
    parser.error(
        "--coordinated-rg6-total-drive-effort-limit-nm requires "
        "--rg6-coupling-mode coordinated_drives"
    )
if (
    args.persistent_grasp_height_offset_m
    and not args.execute_persistent_composite_grasp
):
    parser.error(
        "--persistent-grasp-height-offset-m requires "
        "--execute-persistent-composite-grasp"
    )
if not 0.0 <= args.persistent_grasp_height_offset_m <= 0.08:
    parser.error("--persistent-grasp-height-offset-m must be in [0.0, 0.08]")
if args.rg6_physics_smoke and args.scene_profile != "benchmark":
    parser.error("--rg6-physics-smoke requires --scene-profile benchmark")
if args.rg6_physics_smoke and (
    args.execute_non_oracle_plan
    or args.execute_action_request
    or args.movement_demo
    or args.meeting_demo
    or args.candidate_view_demo
    or args.seeded_pilot_capture
    or args.live_pipeline_server
    or args.capture_video
):
    parser.error("--rg6-physics-smoke cannot be combined with another mode")
if args.seed < 0:
    parser.error("--seed must be non-negative")
if args.capture_video and (
    args.movement_demo
    or args.meeting_demo
    or args.candidate_view_demo
    or args.seeded_pilot_capture
    or args.live_pipeline_server
    or args.execute_non_oracle_plan
    or args.execute_action_request
):
    parser.error(
        "--capture-video requires the default static left/center/right capture mode"
    )
if args.render_quality == "paper" and args.scene_profile != "minimal":
    parser.error("--render-quality paper currently supports the minimal scene only")
if args.render_quality == "paper" and (
    args.movement_demo
    or args.meeting_demo
    or args.execute_non_oracle_plan
    or args.execute_action_request
):
    parser.error("--render-quality paper supports only static capture")

rg6_lid_transfer_parameters = None
if args.rg6_lid_calibration_config is not None:
    from rg6_lid_calibration import load_json, simulation_parameters

    calibration_path = args.rg6_lid_calibration_config
    if not calibration_path.is_absolute():
        calibration_path = PROJECT_ROOT / calibration_path
    try:
        rg6_lid_transfer_parameters = simulation_parameters(
            load_json(calibration_path),
            allow_provisional=args.allow_provisional_rg6_lid_physics,
        )
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        parser.error(f"RG6 lid calibration is not transfer-ready: {exc}")

RENDER_QUALITY = {
    "preview": {
        "renderer": "RaytracedLighting",
        "anti_aliasing": 3,
        "samples_per_pixel_per_frame": 1,
        "capture_steps": 2,
        "rt_subframes": 4,
        "video_crf": 23,
        "video_preset": "medium",
    },
    "paper": {
        "renderer": "PathTracing",
        "anti_aliasing": 4,
        "samples_per_pixel_per_frame": 64,
        "capture_steps": 4,
        "rt_subframes": 8,
        "video_crf": 16,
        "video_preset": "slow",
    },
}
quality_settings = RENDER_QUALITY[args.render_quality]
if args.meeting_demo or args.candidate_view_demo or args.live_pipeline_server:
    quality_settings = {
        **quality_settings,
        "anti_aliasing": 4,
        "capture_steps": 3,
        "rt_subframes": 6,
        "video_crf": 16,
        "video_preset": "slow",
    }
SCENE_PATH = (
    PROJECT_ROOT
    / "assets"
    / "scenes"
    / (
        "open_container_benchmark.usda"
        if args.scene_profile == "benchmark"
        else "open_container_minimal.usda"
    )
)


def record(message: str) -> None:
    with RUNTIME_LOG.open("a", encoding="utf-8") as stream:
        stream.write(message + "\n")


RUNTIME_LOG.write_text("START\n", encoding="utf-8")
record(
    "RUNTIME_CONFIG "
    f"headless={args.headless} "
    f"cuda_visible_devices={os.environ.get('CUDA_VISIBLE_DEVICES')} "
    f"renderer_gpu={args.renderer_gpu} "
    f"physics_gpu={args.physics_gpu} "
    f"render_quality={args.render_quality} "
    f"renderer={quality_settings['renderer']} "
    "multi_gpu=False"
)

simulation_app = SimulationApp(
    {
        "headless": args.headless,
        "active_gpu": args.renderer_gpu,
        "physics_gpu": args.physics_gpu,
        "multi_gpu": False,
        "max_gpu_count": 1,
        "extra_args": [
            "--/renderer/multiGpu/autoEnable=false",
        ],
        "renderer": quality_settings["renderer"],
        "anti_aliasing": quality_settings["anti_aliasing"],
        "samples_per_pixel_per_frame": quality_settings[
            "samples_per_pixel_per_frame"
        ],
        "denoiser": True,
        "fast_shutdown": True,
        "shutdown_watchdog_timeout": 30.0,
    }
)
record("SIMULATION_APP_READY")

import omni.usd
import carb
import isaacsim.core.experimental.utils.app as app_utils
from isaacsim.core.utils.stage import add_reference_to_stage
from isaacsim.storage.native import get_assets_root_path
from omni.kit.viewport.utility import frame_viewport_prims, get_active_viewport
from omni.kit.viewport.utility.camera_state import ViewportCameraState
from pxr import Gf, PhysxSchema, Sdf, Usd, UsdGeom, UsdPhysics
from isaacsim.core.experimental.prims import Articulation, GeomPrim, RigidPrim
from isaacsim.core.simulation_manager import SimulationManager

from observation_capture import (
    BENCHMARK_SEMANTIC_OBJECTS,
    add_scene_labels,
    align_world_camera_to_ee,
    align_tool_and_camera,
    camera_calibration,
    configure_camera,
    create_capture_pipeline,
    create_fixed_overview_camera,
    create_rgb_capture_pipeline,
    load_observation_config,
    make_gripper_kinematic,
    move_pose_interpolated,
    render_benchmark_id_pass,
    render_reference_removed_target_id_pass,
    render_target_amodal_id_pass,
    save_capture,
    set_pose,
)


context = omni.usd.get_context()
if not context.open_stage(str(SCENE_PATH)):
    simulation_app.close()
    raise RuntimeError(f"Failed to open stage: {SCENE_PATH}")
record("STAGE_OPEN_REQUESTED")

for _ in range(20):
    simulation_app.update()

stage = context.get_stage()
record("STAGE_READY")
seeded_layout = None
household_scene = None
if (
    args.seeded_pilot_capture
    or args.live_pipeline_server
    or args.execute_non_oracle_plan
):
    from seeded_benchmark import apply_layout, generate_layout

    seeded_layout = generate_layout(args.seed)
    apply_layout(stage, seeded_layout)
    record(f"SEEDED_BENCHMARK_LAYOUT_APPLIED={args.seed}")
    if args.household_perception_pilot:
        from household_pilot_scene import apply_household_pilot_visuals

        household_scene = apply_household_pilot_visuals(
            stage, seeded_layout
        )
        record("HOUSEHOLD_PERCEPTION_VISUALS_APPLIED")
        if args.scanned_basket_perception_pilot:
            from scanned_basket_scene import (
                PHYSICS_CLEARANCE_OUTSIDE_MUG_POSITION_WORLD_M,
                VISIBLE_OUTSIDE_MUG_POSITION_WORLD_M,
                calibration_variant_for_seed,
                replace_procedural_basket_with_scan,
            )

            calibration_scene_variant = args.calibration_scene_variant
            if calibration_scene_variant == "auto":
                calibration_scene_variant = calibration_variant_for_seed(
                    args.seed
                )
            household_scene["reference"] = (
                replace_procedural_basket_with_scan(
                    stage,
                    active_occlusion_pilot=args.active_occlusion_pilot,
                    collision_physics_pilot=(
                        args.basket_collision_physics_pilot
                    ),
                    calibration_scene_variant=calibration_scene_variant,
                    calibration_seed=args.seed,
                    cover_physics_calibration=(
                        rg6_lid_transfer_parameters
                    ),
                )
            )
            household_scene["target_distractor"]["relation"] = "outside"
            outside_mug_position = (
                PHYSICS_CLEARANCE_OUTSIDE_MUG_POSITION_WORLD_M
                if args.basket_collision_physics_pilot
                else VISIBLE_OUTSIDE_MUG_POSITION_WORLD_M
            )
            household_scene["target_distractor"]["position_world_m"] = list(
                outside_mug_position
            )
            seeded_layout["positions_world_m"]["rear_red_candidate"] = list(
                outside_mug_position
            )
            if calibration_scene_variant is not None:
                scanned_reference = household_scene["reference"]
                seeded_layout["positions_world_m"]["target_red"] = list(
                    scanned_reference["target_position_world_m"]
                )
                seeded_layout["calibration_ground_truth"] = (
                    scanned_reference["calibration_ground_truth"]
                )
                household_scene["calibration_ground_truth"] = (
                    scanned_reference["calibration_ground_truth"]
                )
                household_scene["target"]["relation"] = (
                    scanned_reference["calibration_ground_truth"][
                        "world_ground_truth"
                    ]["membership"]
                )
                calibration_active_occlusion = scanned_reference.get(
                    "active_occlusion", {}
                )
                action_scene_layout = calibration_active_occlusion.get(
                    "action_scene_layout"
                )
                if action_scene_layout is not None:
                    action_occluder_position = action_scene_layout.get(
                        "action_occluder_position_world_m"
                    )
                    if action_occluder_position is not None:
                        seeded_layout["positions_world_m"][
                            "occluder_orange"
                        ] = list(action_occluder_position)
                    seeded_layout.setdefault(
                        "geometry_overrides_world_m", {}
                    )["action_occluder"] = action_scene_layout.get(
                        "action_occluder_geometry"
                    )
                    seeded_layout["relations_preserved"][
                        "occluder_orange"
                    ] = (
                        "action-conditioned occluder; see "
                        "geometry_overrides_world_m"
                    )
            if args.active_occlusion_pilot:
                active_occlusion = household_scene["reference"][
                    "active_occlusion"
                ]
                seeded_layout["positions_world_m"]["target_red"] = list(
                    active_occlusion["target_position_world_m"]
                )
                action_scene_layout = active_occlusion.get(
                    "action_scene_layout"
                )
                if action_scene_layout is not None:
                    action_occluder_position = action_scene_layout.get(
                        "action_occluder_position_world_m"
                    )
                    if action_occluder_position is not None:
                        seeded_layout["positions_world_m"][
                            "occluder_orange"
                        ] = list(action_occluder_position)
                    seeded_layout.setdefault(
                        "geometry_overrides_world_m", {}
                    )["action_occluder"] = action_scene_layout.get(
                        "action_occluder_geometry"
                    )
                    seeded_layout["relations_preserved"][
                        "occluder_orange"
                    ] = (
                        "action-conditioned occluder; see "
                        "geometry_overrides_world_m"
                    )
                else:
                    seeded_layout["relations_preserved"][
                        "occluder_orange"
                    ] = "disabled; scanned basket rim is the occluder"
            household_scene["schema_version"] = (
                "household-scanned-basket-scene-v1"
            )
            record("SCANNED_BASKET_VISUAL_APPLIED")
            if args.basket_collision_physics_pilot:
                record("SCANNED_BASKET_STATIC_COLLISION_APPLIED")

robot_prim_path = "/World/RobotSystem/UR10e"
robot_joint_root = f"{robot_prim_path}/joints"
actual_world_shift = np.asarray([0.20, -0.32, -0.76])
if args.actual_view_motion:
    from persistent_composite_grasp import ENVIRONMENT_ROOTS

    for environment_path in ENVIRONMENT_ROOTS:
        environment_prim = stage.GetPrimAtPath(environment_path)
        if not environment_prim.IsValid():
            continue
        xformable = UsdGeom.Xformable(environment_prim)
        translate_ops = [
            op
            for op in xformable.GetOrderedXformOps()
            if op.GetOpType() == UsdGeom.XformOp.TypeTranslate
        ]
        if translate_ops:
            current = np.asarray(
                translate_ops[0].Get(), dtype=np.float64
            )
            translate_ops[0].Set(
                Gf.Vec3d(*(current + actual_world_shift))
            )
        else:
            xformable.AddTranslateOp().Set(
                Gf.Vec3d(*actual_world_shift)
            )
    # Place the household mug's *bottom origin* on the support surface before
    # live observation starts.  The older code treated this Xform origin as
    # the object center and added half the mug height, visibly floating it.
    target_prim = stage.GetPrimAtPath("/World/TargetRed")
    container_bottom = stage.GetPrimAtPath(
        "/World/OpenContainer/Bottom"
    )
    if (
        seeded_layout is not None
        and target_prim.IsValid()
        and container_bottom.IsValid()
    ):
        target_world = np.asarray(
            omni.usd.get_world_transform_matrix(
                target_prim
            ).ExtractTranslation(),
            dtype=np.float64,
        )
        bottom_world_z = float(
            omni.usd.get_world_transform_matrix(
                container_bottom
            ).ExtractTranslation()[2]
        )
        scanned_reference = (
            household_scene.get("reference", {})
            if household_scene is not None
            else {}
        )
        target_support = scanned_reference.get("target_support", {})
        active_occlusion = scanned_reference.get("active_occlusion", {})
        if target_support.get("surface") == "table":
            desired_target_base_z = float(
                target_support["base_z_world_m"]
            ) + float(actual_world_shift[2])
            support_source = target_support["validation"]
        elif target_support.get("surface") == "scanned_basket_interior":
            support_offset_z = float(
                target_support["base_z_offset_from_reference_m"]
            )
            desired_target_base_z = bottom_world_z + support_offset_z
            support_source = target_support["validation"]
        elif active_occlusion.get("enabled"):
            support_offset_z = float(
                active_occlusion["support_surface_z_offset_m"]
            )
            desired_target_base_z = bottom_world_z + support_offset_z
            support_source = active_occlusion["support_validation"]
        else:
            desired_target_base_z = bottom_world_z + 0.018 * 0.5
            support_source = "procedural_container_bottom_top"
        target_xform = UsdGeom.Xformable(target_prim)
        target_translate_ops = [
            op
            for op in target_xform.GetOrderedXformOps()
            if op.GetOpType() == UsdGeom.XformOp.TypeTranslate
        ]
        if not target_translate_ops:
            raise RuntimeError("Seeded target translate op is missing")
        authored_translation = np.asarray(
            target_translate_ops[0].Get(), dtype=np.float64
        )
        authored_translation[2] += desired_target_base_z - target_world[2]
        target_translate_ops[0].Set(
            Gf.Vec3d(*authored_translation)
        )
        seeded_layout.setdefault(
            "physical_positions_world_m", {}
        )["target_red_bottom_contact"] = [
            float(target_world[0]),
            float(target_world[1]),
            desired_target_base_z,
        ]
        # Keep the established center-position field for RGB-D evaluation.
        # The actual procedural mug is 102 mm tall.
        seeded_layout["physical_positions_world_m"][
            "target_red_settled"
        ] = [
            float(target_world[0]),
            float(target_world[1]),
            desired_target_base_z + 0.102 * 0.5,
        ]
        seeded_layout["physical_positions_world_m"][
            "target_red_support_source"
        ] = support_source
        record(
            "ACTUAL_VIEW_TARGET_BOTTOM_CONTACT_Z="
            f"{desired_target_base_z:.6f}"
        )
    legacy_robot_system = stage.GetPrimAtPath("/World/RobotSystem")
    if legacy_robot_system.IsValid():
        legacy_robot_system.SetActive(False)
    import_result = json.loads(
        (
            PROJECT_ROOT
            / "assets"
            / "robots"
            / "ur10e_rg6"
            / "isaac6_import"
            / "import_result.json"
        ).read_text(encoding="utf-8")
    )
    composite_asset = Path(import_result["output_usd"]).resolve()
    robot_prim_path = "/World/UR10eRG6"
    robot_joint_root = f"{robot_prim_path}/Physics"
    composite_prim = stage.DefinePrim(robot_prim_path, "Xform")
    composite_prim.GetReferences().AddReference(str(composite_asset))
    composite_variant = composite_prim.GetVariantSets().GetVariantSet(
        "Physics"
    )
    if composite_variant.IsValid():
        composite_variant.SetVariantSelection("physx")
    for arm_joint_name in (
        "shoulder_pan_joint",
        "shoulder_lift_joint",
        "elbow_joint",
        "wrist_1_joint",
        "wrist_2_joint",
        "wrist_3_joint",
    ):
        arm_joint = stage.GetPrimAtPath(
            f"{robot_joint_root}/{arm_joint_name}"
        )
        drive = UsdPhysics.DriveAPI.Get(arm_joint, "angular")
        if drive:
            drive.CreateStiffnessAttr().Set(1000.0)
            drive.CreateDampingAttr().Set(50.0)
            drive.CreateMaxForceAttr().Set(400.0)
    record(f"ACTUAL_VIEW_COMPOSITE_ASSET={composite_asset}")

assets_root = get_assets_root_path()
if assets_root is None:
    raise RuntimeError("Isaac Sim production asset root is unavailable")
if not args.actual_view_motion:
    ur10e_asset = (
        assets_root
        + "/Isaac/Robots/UniversalRobots/ur10e/ur10e.usd"
    )
    add_reference_to_stage(ur10e_asset, robot_prim_path)
    record("UR10E_ASSET=" + ur10e_asset)
for _ in range(30):
    simulation_app.update()
camera_prim = (
    UsdGeom.Camera.Define(stage, "/World/CompositeWristCamera").GetPrim()
    if args.actual_view_motion
    else stage.GetPrimAtPath("/World/ObservationCamera")
)
if not camera_prim.IsValid():
    simulation_app.close()
    raise RuntimeError("Observation camera is missing from the stage")

ur10e_ee = None
if args.actual_view_motion:
    ur10e_ee = next(
        (
            prim
            for prim in stage.Traverse()
            if str(prim.GetPath()).startswith(robot_prim_path)
            and prim.GetName() == "rg6_onrobot_rg6_base_link"
        ),
        None,
    )
    ur10e_ee_candidates = ()
else:
    ur10e_ee_candidates = (
        "/World/RobotSystem/UR10e/ee_link",
        "/World/RobotSystem/UR10e/wrist_3_link/flange",
        "/World/RobotSystem/UR10e/tool0",
    )
for candidate_path in ur10e_ee_candidates:
    candidate = stage.GetPrimAtPath(candidate_path)
    if candidate.IsValid():
        ur10e_ee = candidate
        record("UR10E_EE=" + candidate_path)
        break
rg6_prim = (
    ur10e_ee
    if args.actual_view_motion
    else stage.GetPrimAtPath("/World/RobotSystem/RG6")
)
robot_system_prim = stage.GetPrimAtPath("/World/RobotSystem")
if ur10e_ee is None or not rg6_prim.IsValid():
    matching_paths = [
        str(prim.GetPath())
        for prim in stage.Traverse()
        if "ur10" in str(prim.GetPath()).lower()
        or "ee" in str(prim.GetPath()).lower()
        or "rg6" in str(prim.GetPath()).lower()
    ]
    record("ROBOT_PATHS=" + "|".join(matching_paths))
    raise RuntimeError("UR10e or RG6 prim failed to load")
record("ROBOT_PRIMS_VALID")
zivid_camera = camera_prim if args.actual_view_motion else stage.GetPrimAtPath(
    "/World/RobotSystem/RG6/Zivid2Camera"
)
if not zivid_camera.IsValid():
    raise RuntimeError("Zivid 2 wrist camera prim is missing")

observation_config = load_observation_config(PROJECT_ROOT)
if args.household_perception_pilot:
    observation_config["camera"]["resolution"] = [960, 720]
if args.actual_view_motion:
    observation_config["camera"]["look_at_world_m"] = (
        np.asarray(
            observation_config["camera"]["look_at_world_m"],
            dtype=np.float64,
        )
        + actual_world_shift
    ).tolist()
    observation_config["overview_camera"]["position_world_m"] = (
        np.asarray(
            observation_config["overview_camera"]["position_world_m"],
            dtype=np.float64,
        )
        + actual_world_shift
    ).tolist()
    observation_config["overview_camera"]["look_at_world_m"] = (
        np.asarray(
            observation_config["overview_camera"]["look_at_world_m"],
            dtype=np.float64,
        )
        + actual_world_shift
    ).tolist()
    # Use the validated close high-detail full-manipulator framing rather than
    # the older distant overview during actual composite episodes.
    observation_config["overview_camera"]["position_world_m"] = (
        np.asarray([1.94, -2.50, 2.10], dtype=np.float64)
        + actual_world_shift
    ).tolist()
    observation_config["overview_camera"]["look_at_world_m"] = (
        np.asarray([0.46, 0.02, 1.20], dtype=np.float64)
        + actual_world_shift
    ).tolist()
    # Keep the wrist roll on the already validated seed-0 grasp branch. This
    # avoids a later near-pi wrist command when the terminal manipulation
    # starts, while leaving the shoulder/elbow motions that produce the
    # re-observation viewpoints unchanged.
    for pose_values in observation_config["poses_rad"].values():
        pose_values[5] = -3.141247102
capture_pose_names = list(observation_config["capture"]["poses"])
if (
    args.seeded_pilot_capture
    or args.live_pipeline_server
    or args.execute_non_oracle_plan
):
    capture_pose_names.extend(observation_config["candidate_views"]["poses"])
if args.household_perception_pilot and args.seeded_pilot_capture:
    # This first household-shaped pilot intentionally evaluates exactly the
    # three wrist-reachable views approved in the immediate plan. The current
    # overhead entry is a synthetic diagnostic pose and has no UR10e joint
    # configuration in observation_poses.json.
    capture_pose_names = ["left", "center", "right"]
if args.render_quality == "paper":
    paper_overview = observation_config["overview_camera"]["paper_quality"]
    observation_config["overview_camera"]["resolution"] = paper_overview[
        "resolution"
    ]
    observation_config["overview_camera"]["focal_length_mm"] = paper_overview[
        "focal_length_mm"
    ]
elif args.meeting_demo or args.candidate_view_demo or args.live_pipeline_server:
    meeting_overview = observation_config["overview_camera"][
        "meeting_quality"
    ]
    observation_config["overview_camera"]["resolution"] = meeting_overview[
        "resolution"
    ]
    observation_config["overview_camera"]["focal_length_mm"] = (
        meeting_overview["focal_length_mm"]
    )
overview_camera = create_fixed_overview_camera(stage, observation_config)
record(
    "OVERVIEW_CAMERA_CREATED="
    + observation_config["overview_camera"]["path"]
)
add_scene_labels(
    stage,
    BENCHMARK_SEMANTIC_OBJECTS if args.scene_profile == "benchmark" else None,
)
for _ in range(10):
    simulation_app.update()
record(
    "SEMANTIC_LABELS_ADDED="
    + "|".join(
        (
            BENCHMARK_SEMANTIC_OBJECTS
            if args.scene_profile == "benchmark"
            else {
                "/World/TargetRed": "target_red",
                "/World/DistractorBlue": "distractor_blue",
                "/World/OpenContainer": "container",
            }
        ).values()
    )
)
configure_camera(zivid_camera, observation_config)
if args.rg6_physics_smoke:
    rg6_base_path = (
        "/World/RobotSystem/RG6/Geometry/onrobot_rg6_base_link"
    )
    rg6_base_prim = stage.GetPrimAtPath(rg6_base_path)
    rg6_root_joint = UsdPhysics.FixedJoint(
        stage.GetPrimAtPath("/World/RobotSystem/RG6/Physics/root_joint")
    )
    if not rg6_base_prim.IsValid() or not rg6_root_joint.GetPrim().IsValid():
        raise RuntimeError("RG6 physical base or root joint is missing")
    record(f"RG6_STANDALONE_PHYSICAL_ROOT={rg6_base_path}")
elif not args.actual_view_motion:
    physics_variant = rg6_prim.GetVariantSets().GetVariantSet("Physics")
    if physics_variant.IsValid():
        physics_variant.SetVariantSelection("none")
        record("RG6_IMPORTED_PHYSICS_DISABLED_FOR_VISUAL_OBSERVATION_MOUNT")
    make_gripper_kinematic(rg6_prim)
    record("RG6_KINEMATIC_FOR_PROVISIONAL_MOUNT")
resolution = tuple(observation_config["camera"]["resolution"])
wrist_camera_path = (
    "/World/CompositeWristCamera"
    if args.actual_view_motion
    else "/World/RobotSystem/RG6/Zivid2Camera"
)
rep, render_product, rgb_annotator, depth_annotator = create_capture_pipeline(
    wrist_camera_path, resolution
)
overview_settings = observation_config["overview_camera"]
overview_render_product, overview_rgb_annotator = create_rgb_capture_pipeline(
    overview_settings["path"],
    tuple(overview_settings["resolution"]),
)
camera_provenance = {
    "wrist_sensor": {
        "path": wrist_camera_path,
        "role": "vlm_rgb_and_metric_depth_sensor",
        "resolution": list(resolution),
        "modalities": ["rgb", "metric_depth"],
        "used_by_vlm_or_planner": True,
    },
    "external_overview": {
        "path": overview_settings["path"],
        "role": overview_settings["purpose"],
        "resolution": list(overview_settings["resolution"]),
        "position_world_m": overview_settings["position_world_m"],
        "look_at_world_m": overview_settings["look_at_world_m"],
        "modalities": ["rgb"],
        "used_by_vlm_or_planner": overview_settings["used_by_vlm_or_planner"],
    },
}
output_root = PROJECT_ROOT / observation_config["capture"]["output_directory"]
if args.scene_profile == "benchmark":
    output_root = PROJECT_ROOT / "outputs" / "benchmark_observations"
elif args.render_quality == "paper":
    output_root = PROJECT_ROOT / "outputs" / "paper_visualization" / "minimal"
if args.meeting_demo:
    output_root = (
        PROJECT_ROOT
        / "outputs"
        / "meeting_demo"
        / "benchmark_seed000"
        / "observations"
    )
elif args.candidate_view_demo:
    output_root = (
        PROJECT_ROOT
        / "outputs"
        / "candidate_view_demo"
        / "benchmark_seed000"
        / "observations"
    )
elif args.seeded_pilot_capture:
    output_family = (
        "household_perception_pilot"
        if args.household_perception_pilot
        else "seeded_pilot"
    )
    output_root = (
        PROJECT_ROOT
        / "outputs"
        / output_family
        / f"benchmark_seed{args.seed:03d}"
        / "observations"
    )
    output_root.parent.mkdir(parents=True, exist_ok=True)
    (output_root.parent / "scene_layout.json").write_text(
        json.dumps(seeded_layout, indent=2) + "\n",
        encoding="utf-8",
    )
    if args.household_perception_pilot:
        (output_root.parent / "household_scene.json").write_text(
            json.dumps(household_scene, indent=2) + "\n",
            encoding="utf-8",
        )
elif args.live_pipeline_server:
    allowed_live_root = (PROJECT_ROOT / "outputs" / "live_pipeline").resolve()
    live_session_dir = args.live_session_dir.resolve()
    if not live_session_dir.is_relative_to(allowed_live_root):
        raise ValueError(
            f"Live session must be under {allowed_live_root}: {live_session_dir}"
        )
    output_root = live_session_dir / "observations"
    output_root.mkdir(parents=True, exist_ok=True)
    (live_session_dir / "scene_layout.json").write_text(
        json.dumps(seeded_layout, indent=2) + "\n",
        encoding="utf-8",
    )
    if args.household_perception_pilot:
        (live_session_dir / "household_scene.json").write_text(
            json.dumps(household_scene, indent=2) + "\n",
            encoding="utf-8",
        )
record(
    "CAPTURE_PIPELINE_READY="
    "wrist_rgb|wrist_depth|overview_rgb|rgb_color_key_instance_fallback"
)
if args.actual_view_motion:
    # Author the live center pose and open RG6 state before PhysX creates the
    # composite articulation. All subsequent observation motions use drives.
    center_by_name = dict(
        zip(
            observation_config["joint_order"],
            observation_config["poses_rad"]["center"],
        )
    )
    open_master = -0.20
    initial_by_name = {
        **center_by_name,
        "rg6_finger_joint": open_master,
        "rg6_left_inner_knuckle_joint": -open_master,
        "rg6_left_inner_finger_joint": open_master,
        "rg6_right_outer_knuckle_joint": -open_master,
        "rg6_right_inner_knuckle_joint": -open_master,
        "rg6_right_inner_finger_joint": open_master,
    }
    for joint_name, position_rad in initial_by_name.items():
        joint = stage.GetPrimAtPath(f"{robot_joint_root}/{joint_name}")
        if not joint.IsValid():
            raise RuntimeError(
                f"Composite initialization joint is missing: {joint_name}"
            )
        state = PhysxSchema.JointStateAPI.Get(joint, "angular")
        if not state:
            state = PhysxSchema.JointStateAPI.Apply(joint, "angular")
        state.CreatePositionAttr().Set(
            float(np.degrees(position_rad))
        )
        state.CreateVelocityAttr().Set(0.0)
        drive = UsdPhysics.DriveAPI.Get(joint, "angular")
        if drive:
            drive.CreateTargetPositionAttr().Set(
                float(np.degrees(position_rad))
            )
            drive.CreateTargetVelocityAttr().Set(0.0)
    record("ACTUAL_VIEW_INITIAL_CENTER_AND_OPEN_RG6_STATE_AUTHORED")
SimulationManager.setup_simulation(dt=1.0 / 60.0)
robot = Articulation(robot_prim_path)
simulation_app.update()
record("UR10E_DOFS=" + "|".join(robot.dof_names))
app_utils.play()
simulation_app.update()
record("PHYSICS_PLAYING")
record("UR10E_LINKS=" + "|".join(robot.link_names))
if args.rg6_physics_smoke:
    diagnostics_root = PROJECT_ROOT / "outputs" / "rg6_physics"
    diagnostics_root.mkdir(parents=True, exist_ok=True)
    rg6_robot = Articulation(rg6_base_path)
    simulation_app.update()
    finger_dof_names = [
        name
        for name in rg6_robot.dof_names
        if "finger" in name or "knuckle" in name
    ]
    finger_indices = [
        rg6_robot.dof_names.index(name) for name in finger_dof_names
    ]
    master_finger_index = rg6_robot.dof_names.index("finger_joint")
    zero_positions = [0.0] * len(rg6_robot.dof_names)
    rg6_robot.set_dof_positions(zero_positions)
    rg6_robot.set_dof_position_targets(zero_positions)
    for _ in range(30):
        simulation_app.update()
    initial_array = rg6_robot.get_dof_positions().numpy()
    initial_vector = initial_array[0] if initial_array.ndim > 1 else initial_array
    initial_positions = initial_vector.tolist()
    sequence = []
    for label, target in (("open", -0.45), ("close", 0.45), ("reopen", -0.45)):
        rg6_robot.set_dof_position_targets(
            [target],
            dof_indices=[master_finger_index],
        )
        for _ in range(120):
            simulation_app.update()
        measured_array = rg6_robot.get_dof_positions().numpy()
        measured = (
            measured_array[0] if measured_array.ndim > 1 else measured_array
        )
        sequence.append(
            {
                "label": label,
                "requested_rad": target,
                "measured_rad": {
                    name: float(measured[index])
                    for name, index in zip(finger_dof_names, finger_indices)
                },
            }
        )
    diagnostics = {
        "schema_version": "rg6-physical-articulation-smoke-v1",
        "status": "completed" if finger_dof_names else "failed",
        "articulation_mode": (
            "rg6_physics_articulation_with_world_fixed_base_for_smoke_test"
        ),
        "rg6_articulation_path": rg6_base_path,
        "ur10e_end_effector_path": str(ur10e_ee.GetPath()),
        "rg6_base_path": rg6_base_path,
        "all_dof_names": list(rg6_robot.dof_names),
        "finger_dof_names": finger_dof_names,
        "initial_positions_rad": initial_positions,
        "sequence": sequence,
        "gpu_policy": {
            "renderer_physical_gpu": args.renderer_gpu,
            "physics_cuda_device": args.physics_gpu,
            "multi_gpu": False,
        },
    }
    diagnostics_path = diagnostics_root / "articulation_smoke.json"
    diagnostics_path.write_text(
        json.dumps(diagnostics, indent=2) + "\n",
        encoding="utf-8",
    )
    record(f"RG6_PHYSICS_SMOKE_RESULT={diagnostics_path}")
    print(f"RG6_PHYSICS_SMOKE_RESULT={diagnostics_path}", flush=True)
    app_utils.stop()
    simulation_app.close()
    raise SystemExit(0 if finger_dof_names else 2)
if args.actual_view_motion:
    ee_geom = GeomPrim(paths=str(ur10e_ee.GetPath()))
    physical_ee_name = ur10e_ee.GetName()
else:
    physical_ee_candidates = ("ee_link", "wrist_3_link", "tool0")
    physical_ee_name = next(
        name for name in physical_ee_candidates if name in robot.link_names
    )
    ee_link_index = robot.link_names.index(physical_ee_name)
    ee_geom = GeomPrim(paths=robot.link_paths[0][ee_link_index])
record("PHYSX_EE_LINK=" + physical_ee_name)
if args.actual_view_motion:
    # Articulation link prims are nested. A world bound computed on a parent
    # link therefore contains all downstream links and creates false obstacle
    # overlaps. Check the leaf collision shapes instead, excluding only the
    # fixed base collision.
    moving_body_names = {
        "shoulder_link",
        "upper_arm_link",
        "forearm_link",
        "wrist_1_link",
        "wrist_2_link",
        "wrist_3_link",
        "rg6_onrobot_rg6_base_link",
        "rg6_left_outer_knuckle",
        "rg6_left_inner_knuckle",
        "rg6_right_outer_knuckle",
        "rg6_right_inner_knuckle",
        "rg6_left_inner_finger",
        "rg6_right_inner_finger",
    }
    moving_link_paths = []
    for prim in Usd.PrimRange.Stage(
        stage, Usd.TraverseInstanceProxies()
    ):
        path = str(prim.GetPath())
        path_components = set(path.split("/"))
        if (
            path.startswith(f"{robot_prim_path}/Geometry/")
            and path_components.intersection(moving_body_names)
            and (
                prim.HasAPI(UsdPhysics.CollisionAPI)
                or "PhysicsCollisionAPI" in prim.GetAppliedSchemas()
            )
        ):
            moving_link_paths.append(path)
    if not moving_link_paths:
        raise RuntimeError(
            "No moving composite collision shapes were discovered"
        )
else:
    moving_link_names = [
        name
        for name in robot.link_names
        if name
        in {
            "shoulder_link",
            "upper_arm_link",
            "forearm_link",
            "wrist_1_link",
            "wrist_2_link",
            "wrist_3_link",
            "ee_link",
            "tool0",
        }
    ]
    moving_link_paths = [
        robot.link_paths[0][robot.link_names.index(name)]
        for name in moving_link_names
    ]
record(f"MOVING_COLLISION_PRIM_COUNT={len(moving_link_paths)}")
contact_monitor = None
record(
    "CONTACT_TENSOR_MONITOR_SKIPPED="
    "configured_filter_paths_do_not_resolve_to_supported_rigid_contact_entries"
)


def current_ee_pose():
    positions, orientations = ee_geom.get_world_poses()
    return positions.numpy()[0], orientations.numpy()[0]


def align_active_wrist_camera(ee_pose) -> None:
    if args.actual_view_motion:
        align_world_camera_to_ee(
            zivid_camera, observation_config, ee_pose
        )
    else:
        align_tool_and_camera(
            stage,
            ur10e_ee,
            rg6_prim,
            zivid_camera,
            observation_config,
            ee_pose,
        )

if args.actual_view_motion:
    center_values = observation_config["poses_rad"]["center"]
    center_indices = [
        robot.dof_names.index(name)
        for name in observation_config["joint_order"]
    ]
    robot.set_dof_positions(center_values, dof_indices=center_indices)
    robot.set_dof_velocities(
        [0.0] * len(center_indices), dof_indices=center_indices
    )
    robot.set_dof_position_targets(center_values, dof_indices=center_indices)
    for _ in range(30):
        simulation_app.update()
    center_measured_array = robot.get_dof_positions().numpy()
    center_measured_vector = (
        center_measured_array[0]
        if center_measured_array.ndim > 1
        else center_measured_array
    )
    center_measured = np.asarray(
        center_measured_vector[center_indices], dtype=np.float64
    )
    if (
        not np.all(np.isfinite(center_measured))
        or np.max(
            np.abs(
                center_measured
                - np.asarray(center_values, dtype=np.float64)
            )
        )
        > observation_config["trajectory"]["final_tolerance_rad"]
    ):
        raise RuntimeError(
            "Pre-authored UR10e center state failed initial hold: "
            f"{center_measured.tolist()}"
        )
    record("ACTUAL_VIEW_INITIAL_CENTER_HOLD_VERIFIED")
else:
    set_pose(robot, observation_config, "home", simulation_app.update, 1)
app_utils.pause()
simulation_app.update()
align_active_wrist_camera(current_ee_pose())
record("HOME_POSE_APPLIED")


def objective_behind_geometry_from_stage() -> dict | None:
    """Read simulator-only geometry for post-hoc behind evaluation."""
    if household_scene is None:
        return None
    calibration_ground_truth = household_scene.get(
        "calibration_ground_truth"
    )
    if calibration_ground_truth is None:
        return None
    bbox_cache = UsdGeom.BBoxCache(
        Usd.TimeCode.Default(),
        [UsdGeom.Tokens.default_, UsdGeom.Tokens.render],
        useExtentsHint=True,
    )

    def aligned_bounds(path: str) -> tuple[np.ndarray, np.ndarray]:
        prim = stage.GetPrimAtPath(path)
        if not prim.IsValid():
            raise RuntimeError(
                f"Objective-behind ground-truth prim is missing: {path}"
            )
        aligned = bbox_cache.ComputeWorldBound(prim).ComputeAlignedBox()
        return (
            np.asarray(aligned.GetMin(), dtype=np.float64),
            np.asarray(aligned.GetMax(), dtype=np.float64),
        )

    target_lower, target_upper = aligned_bounds("/World/TargetRed")
    reference_lower, reference_upper = aligned_bounds(
        "/World/OpenContainer"
    )
    return {
        "target_center_world_m": (
            0.5 * (target_lower + target_upper)
        ).tolist(),
        "reference_bounds_world_m": {
            "lower": reference_lower.tolist(),
            "upper": reference_upper.tolist(),
        },
        "membership": calibration_ground_truth["world_ground_truth"][
            "entities"
        ]["target_red"]["membership"],
        "source": "simulator_world_aligned_bounds",
        "exposed_to_model_or_planner": False,
    }


def capture_observation(
    pose_name: str,
    debug_ee_position_world_m: list[float] | None = None,
    *,
    set_robot_to_configured_pose: bool = True,
) -> None:
    app_utils.play()
    simulation_app.update()
    if (
        debug_ee_position_world_m is None
        and set_robot_to_configured_pose
    ):
        set_pose(
            robot,
            observation_config,
            pose_name,
            simulation_app.update,
            1,
        )
    app_utils.pause()
    simulation_app.update()
    if debug_ee_position_world_m is None:
        pose_position, pose_orientation = current_ee_pose()
    else:
        pose_position = np.asarray(debug_ee_position_world_m, dtype=np.float64)
        pose_orientation = np.asarray([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
        record(
            f"DEBUG_SYNTHETIC_EE_POSE {pose_name} "
            f"position={pose_position.tolist()}"
        )
    record(f"EE_POSE {pose_name} position={pose_position.tolist()}")
    align_active_wrist_camera((pose_position, pose_orientation))
    for _ in range(8):
        simulation_app.update()
    wrist_camera_calibration = camera_calibration(
        zivid_camera, resolution
    )
    for _ in range(quality_settings["capture_steps"]):
        rep.orchestrator.step(
            rt_subframes=quality_settings["rt_subframes"]
        )
    rgb_data = np.asarray(rgb_annotator.get_data()).copy()
    depth_data = np.asarray(depth_annotator.get_data()).copy()
    overview_rgb_data = np.asarray(overview_rgb_annotator.get_data()).copy()
    instance_override = (
        render_benchmark_id_pass(stage, rep, rgb_annotator)
        if args.scene_profile == "benchmark"
        else None
    )
    target_amodal_override = (
        render_target_amodal_id_pass(stage, rep, rgb_annotator)
        if args.scene_profile == "benchmark"
        else None
    )
    target_reference_removed_override = (
        render_reference_removed_target_id_pass(
            stage, rep, rgb_annotator
        )
        if args.scene_profile == "benchmark"
        else None
    )
    save_capture(
        output_root,
        pose_name,
        rgb_data,
        depth_data,
        instance_override=instance_override,
        target_amodal_override=target_amodal_override,
        target_reference_removed_override=(
            target_reference_removed_override
        ),
        overview_rgb_data=overview_rgb_data,
        camera_provenance=camera_provenance,
        camera_calibration_data=wrist_camera_calibration,
        objective_behind_geometry=(
            objective_behind_geometry_from_stage()
        ),
    )
    record("CAPTURED_POSE=" + pose_name)


meeting_demo_result = None
candidate_view_result = None
live_pipeline_result = None
if args.live_pipeline_server:
    live_session_dir = args.live_session_dir.resolve()
    live_method_config = json.loads(
        args.method_config.resolve().read_text(encoding="utf-8")
    )
    live_debug_positions = live_method_config.get(
        "viewpoint_execution", {}
    ).get("debug_ee_positions_world_m", {})
    live_viewpoint_execution_mode = live_method_config.get(
        "viewpoint_execution", {}
    ).get("mode", "interpolated_joint_physics")
    live_aabb_collision_pairs: list[list[str]] = []
    live_obstacle_paths = [
        "/World/WorkBench",
        "/World/OpenContainer",
        "/World/TargetRed",
        "/World/OccluderOrange",
        "/World/DistractorYellow",
        "/World/DistractorBlue",
        "/World/DistractorGreen",
        "/World/BoundaryPurple",
        "/World/RearRedCandidate",
    ]

    def live_world_aabb_collision() -> bool:
        cache = UsdGeom.BBoxCache(
            Usd.TimeCode.Default(), [UsdGeom.Tokens.default_]
        )
        mover_paths = list(moving_link_paths)
        for mover_path in mover_paths:
            mover_prim = stage.GetPrimAtPath(mover_path)
            if not mover_prim.IsValid():
                continue
            mover_range = cache.ComputeWorldBound(
                mover_prim
            ).ComputeAlignedRange()
            mover_min, mover_max = mover_range.GetMin(), mover_range.GetMax()
            for obstacle_path in live_obstacle_paths:
                obstacle_prim = stage.GetPrimAtPath(obstacle_path)
                if not obstacle_prim.IsValid():
                    continue
                obstacle_range = cache.ComputeWorldBound(
                    obstacle_prim
                ).ComputeAlignedRange()
                obstacle_min = obstacle_range.GetMin()
                obstacle_max = obstacle_range.GetMax()
                overlaps = all(
                    mover_min[axis] <= obstacle_max[axis]
                    and mover_max[axis] >= obstacle_min[axis]
                    for axis in range(3)
                )
                if overlaps:
                    pair = [mover_path, obstacle_path]
                    if pair not in live_aabb_collision_pairs:
                        live_aabb_collision_pairs.append(pair)
                    return True
        return False

    def write_live_json(path: Path, payload: dict) -> None:
        temporary_path = path.with_suffix(path.suffix + ".tmp")
        temporary_path.write_text(
            json.dumps(payload, indent=2) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary_path, path)

    def wait_for_live_request(index: int, timeout_seconds: float = 600.0) -> dict:
        request_path = live_session_dir / f"action_request_{index:03d}.json"
        deadline = time.monotonic() + timeout_seconds
        while time.monotonic() < deadline:
            if request_path.is_file():
                return json.loads(request_path.read_text(encoding="utf-8"))
            simulation_app.update()
            time.sleep(0.05)
        raise TimeoutError(f"Timed out waiting for {request_path}")

    initial_debug_position = live_debug_positions.get("center")
    capture_observation(
        "center",
        debug_ee_position_world_m=initial_debug_position,
        set_robot_to_configured_pose=not args.actual_view_motion,
    )
    live_events = [
        {
            "index": 0,
            "type": "observation_ready",
            "view": "center",
            "observation_dir": str((output_root / "center").resolve()),
            "trajectory": {
                "status": "completed",
                "mode": (
                    "debug_synthetic_wrist_pose_capture"
                    if initial_debug_position is not None
                    else "configured_center_pose_capture"
                ),
                "continuous_physics_trajectory": False,
                "collision_checked": False,
                "actual_robot_motion_executed": False,
                "debug_ee_position_world_m": initial_debug_position,
            },
        }
    ]
    write_live_json(
        live_session_dir / "observation_ready_000.json",
        live_events[0],
    )
    record("LIVE_PIPELINE_OBSERVATION_READY=0:center")

    terminal_action = None
    terminal_request = None
    for request_index in range(4):
        request = wait_for_live_request(request_index)
        action_type = request.get("type", "")
        if action_type.startswith("viewpoint_"):
            selected_pose = action_type.removeprefix("viewpoint_")
            if selected_pose not in capture_pose_names:
                raise RuntimeError(f"Live request selected unknown view: {selected_pose}")

            debug_position = live_debug_positions.get(selected_pose)
            if debug_position is not None:
                capture_observation(
                    selected_pose,
                    debug_ee_position_world_m=debug_position,
                )
                trajectory = {
                    "status": "completed",
                    "mode": "debug_synthetic_wrist_pose_capture",
                    "continuous_physics_trajectory": False,
                    "collision_checked": False,
                    "actual_robot_motion_executed": False,
                    "debug_ee_position_world_m": debug_position,
                }
            elif live_viewpoint_execution_mode == "debug_pose_capture":
                capture_observation(selected_pose)
                trajectory = {
                    "status": "completed",
                    "mode": "debug_pose_capture",
                    "continuous_physics_trajectory": False,
                    "collision_checked": False,
                    "actual_robot_motion_executed": False,
                    "debug_ee_position_world_m": None,
                }
            else:
                app_utils.play()

                def live_motion_update() -> None:
                    simulation_app.update()
                    pose_position, pose_orientation = current_ee_pose()
                    align_active_wrist_camera(
                        (pose_position, pose_orientation)
                    )

                # A paused tensor view can return stale values immediately
                # after timeline resume in Isaac Sim 6.0. Advance and validate
                # the held articulation before reading the trajectory start.
                current_pose = live_events[-1]["view"]
                current_target = observation_config["poses_rad"].get(
                    current_pose
                )
                joint_indices = [
                    robot.dof_names.index(name)
                    for name in observation_config["joint_order"]
                ]
                if current_target is not None:
                    robot.set_dof_position_targets(
                        current_target,
                        dof_indices=joint_indices,
                    )
                for _ in range(4):
                    live_motion_update()
                resumed_array = robot.get_dof_positions().numpy()
                resumed_vector = (
                    resumed_array[0]
                    if resumed_array.ndim > 1
                    else resumed_array
                )
                resumed = np.asarray(
                    resumed_vector[joint_indices], dtype=np.float64
                )
                lower_raw, upper_raw = robot.get_dof_limits(
                    dof_indices=joint_indices
                )
                lower_array = lower_raw.numpy()
                upper_array = upper_raw.numpy()
                lower = (
                    lower_array[0]
                    if lower_array.ndim > 1
                    else lower_array
                )
                upper = (
                    upper_array[0]
                    if upper_array.ndim > 1
                    else upper_array
                )
                if (
                    not np.all(np.isfinite(resumed))
                    or np.any(resumed < lower)
                    or np.any(resumed > upper)
                ):
                    raise RuntimeError(
                        "UR10e state invalid after live timeline resume: "
                        f"{resumed.tolist()}"
                    )

                live_aabb_collision_pairs.clear()
                trajectory = move_pose_interpolated(
                    robot,
                    observation_config,
                    selected_pose,
                    live_motion_update,
                    collision_checker=live_world_aabb_collision,
                )
                trajectory["actual_robot_motion_executed"] = True
                trajectory["continuous_physics_trajectory"] = True
                trajectory["collision_checked"] = True
                trajectory["collision_scope"] = (
                    "world_aabb_composite_collision_shape_vs_scene_obstacle"
                )
                trajectory["aabb_collision_pairs"] = list(
                    live_aabb_collision_pairs
                )
                app_utils.pause()
                simulation_app.update()
                if trajectory["status"] != "completed":
                    raise RuntimeError(
                        f"Live viewpoint motion failed: {trajectory}"
                    )
                capture_observation(
                    selected_pose,
                    set_robot_to_configured_pose=False,
                )
            event_index = len(live_events)
            event = {
                "index": event_index,
                "type": "observation_ready",
                "view": selected_pose,
                "observation_dir": str((output_root / selected_pose).resolve()),
                "trajectory": trajectory,
            }
            live_events.append(event)
            write_live_json(
                live_session_dir / f"observation_ready_{event_index:03d}.json",
                event,
            )
            record(
                f"LIVE_PIPELINE_OBSERVATION_READY={event_index}:{selected_pose}"
            )
            continue
        if action_type in {"grasp", "remove_cover", "defer", "stop"}:
            terminal_action = action_type
            terminal_request = request
            break
        raise RuntimeError(f"Unsupported live pipeline action: {action_type!r}")

    root_terminal_action = terminal_action
    persistent_grasp_result = None
    persistent_remove_cover_result = None
    post_remove_replan_request = None
    post_remove_replan_requests = []
    if terminal_action == "grasp" and args.execute_persistent_composite_grasp:
        from persistent_composite_grasp import (
            PROVISIONAL_TARGET_COORDINATED_DRIVE_EFFORT_LIMIT_NM,
            PROVISIONAL_TARGET_FOLLOWER_REQUEST_BLEND,
            PROVISIONAL_TARGET_MINIMUM_GRIP_FORCE_PER_FINGER_N,
            PROVISIONAL_TARGET_PASSIVE_FORCE_CONTROLLER_MAX_TORQUE_NM,
            execute_persistent_composite_grasp,
        )

        live_arm_positions = None
        terminal_rgbd_localization = None
        if args.actual_view_motion:
            measured_array = robot.get_dof_positions().numpy()
            measured_vector = (
                measured_array[0]
                if measured_array.ndim > 1
                else measured_array
            )
            live_arm_positions = [
                float(measured_vector[robot.dof_names.index(name)])
                for name in observation_config["joint_order"]
            ]
            localization_value = (
                terminal_request or {}
            ).get("rgbd_localization_path")
            if localization_value:
                localization_path = Path(
                    localization_value
                ).resolve()
                if not localization_path.is_relative_to(
                    live_session_dir
                ):
                    raise ValueError(
                        "Terminal RGB-D localization must be inside the "
                        f"live session: {localization_path}"
                    )
                terminal_rgbd_localization = json.loads(
                    localization_path.read_text(encoding="utf-8")
                )
        persistent_grasp_result = execute_persistent_composite_grasp(
            project_root=PROJECT_ROOT,
            stage=stage,
            simulation_app=simulation_app,
            rep=rep,
            overview_rgb_annotator=overview_rgb_annotator,
            overview_camera_path=overview_settings["path"],
            output_root=live_session_dir / "persistent_grasp",
            seed=args.seed,
            reuse_existing_composite=args.actual_view_motion,
            initial_arm_positions_rad=live_arm_positions,
            rgbd_localization=terminal_rgbd_localization,
            grasp_height_offset_m=args.persistent_grasp_height_offset_m,
            rg6_coupling_mode="passive_mimic",
            coordinated_total_drive_effort_limit_nm=None,
            coordinated_follower_request_blend=(
                PROVISIONAL_TARGET_FOLLOWER_REQUEST_BLEND
            ),
            minimum_grip_force_per_finger_n=(
                PROVISIONAL_TARGET_MINIMUM_GRIP_FORCE_PER_FINGER_N
            ),
            force_controller_max_torque_nm=(
                PROVISIONAL_TARGET_PASSIVE_FORCE_CONTROLLER_MAX_TORQUE_NM
            ),
        )
        if (
            persistent_grasp_result.get("status") != "completed"
            or not persistent_grasp_result.get("lift_verified")
        ):
            raise RuntimeError(
                "Persistent terminal composite grasp failed: "
                f"{persistent_grasp_result.get('status')}"
            )

    if terminal_action == "remove_cover" and args.execute_persistent_remove_cover:
        from persistent_composite_grasp import (
            PROVISIONAL_LID_COMBINED_GRIP_FORCE_N,
            PROVISIONAL_LID_GRIP_FORCE_PER_FINGER_N,
            PROVISIONAL_EPDM_LID_DYNAMIC_FRICTION,
            PROVISIONAL_EPDM_LID_STATIC_FRICTION,
            PROVISIONAL_TARGET_COORDINATED_DRIVE_EFFORT_LIMIT_NM,
            PROVISIONAL_TARGET_FOLLOWER_REQUEST_BLEND,
            PROVISIONAL_TARGET_MINIMUM_GRIP_FORCE_PER_FINGER_N,
            PROVISIONAL_TARGET_PASSIVE_FORCE_CONTROLLER_MAX_TORQUE_NM,
            execute_persistent_composite_grasp,
            grasp_yaw_from_pinch_axis_world,
        )

        measured_array = robot.get_dof_positions().numpy()
        measured_vector = (
            measured_array[0]
            if measured_array.ndim > 1
            else measured_array
        )
        live_arm_positions = [
            float(measured_vector[robot.dof_names.index(name)])
            for name in observation_config["joint_order"]
        ]
        cover_path = "/World/OpenContainer/CalibrationCover"
        cover_handle_path = f"{cover_path}/Handle"
        cover_handle_prim = stage.GetPrimAtPath(cover_handle_path)
        if not cover_handle_prim.IsValid():
            raise RuntimeError(
                f"Dynamic cover handle is missing: {cover_handle_path}"
            )
        handle_world_transform = omni.usd.get_world_transform_matrix(
            cover_handle_prim
        )
        handle_world = np.asarray(
            handle_world_transform.ExtractTranslation(),
            dtype=np.float64,
        )
        handle_pinch_axis_world = np.asarray(
            handle_world_transform.TransformDir(Gf.Vec3d(0.0, 1.0, 0.0)),
            dtype=np.float64,
        )
        handle_grasp_yaw_rad = grasp_yaw_from_pinch_axis_world(
            handle_pinch_axis_world
        )
        persistent_remove_cover_result = execute_persistent_composite_grasp(
            project_root=PROJECT_ROOT,
            stage=stage,
            simulation_app=simulation_app,
            rep=rep,
            overview_rgb_annotator=overview_rgb_annotator,
            overview_camera_path=overview_settings["path"],
            output_root=live_session_dir / "persistent_remove_cover",
            seed=args.seed,
            reuse_existing_composite=True,
            initial_arm_positions_rad=live_arm_positions,
            rgbd_localization=None,
            manipulation_target_path=cover_path,
            contact_target_path=cover_handle_path,
            manipulation_label="cover",
            planning_target_world_m=handle_world.tolist(),
            planning_grasp_yaw_rad=handle_grasp_yaw_rad,
            # Move diagonally beyond the combined cover/basket footprint onto
            # the raised WorkMat, lower onto the support, release,
            # and retreat before collecting the post-removal observation.
            # The former -X staging pose brought the wide cover plate into the
            # UR10e shoulder while lowering (run044).  Keeping the cover's X
            # coordinate change avoids the former shoulder collision while
            # keeping the cover center and at least 75% of its footprint on
            # the declared raised support instead of straddling two heights.
            transfer_offset_world_m=[-0.42, -0.20, 0.16],
            placement_support_path="/World/WorkMat",
            release_after_placement=True,
            minimum_verified_lift_m=0.10,
            pregrasp_settle_steps=240,
            # The RG6 datasheet gives 25 N as its minimum adjustable gripping
            # force.  Use it as a combined contact-force proxy while retaining
            # an 8 N minimum on each side for bilateral load sharing.  The
            # torque-to-force mapping remains provisional until real-tool
            # calibration is available.
            # Run016 reached the force gate at 1.75 Nm but slipped during
            # lift. Use the executor's 2.0 Nm ceiling and an EPDM-like
            # provisional friction pair; the official RG6 standard fingertips
            # use EPDM rubber, but the exact lab value still requires
            # real-tool calibration.
            rg6_master_max_torque_nm=(
                rg6_lid_transfer_parameters["initial_drive_torque_nm"]
                if rg6_lid_transfer_parameters is not None
                else 2.0
            ),
            minimum_grip_force_per_finger_n=(
                rg6_lid_transfer_parameters["minimum_force_per_finger_n"]
                if rg6_lid_transfer_parameters is not None
                else PROVISIONAL_LID_GRIP_FORCE_PER_FINGER_N
            ),
            minimum_combined_grip_force_n=(
                rg6_lid_transfer_parameters["minimum_combined_force_n"]
                if rg6_lid_transfer_parameters is not None
                else PROVISIONAL_LID_COMBINED_GRIP_FORCE_N
            ),
            grip_static_friction=(
                rg6_lid_transfer_parameters["static_friction"]
                if rg6_lid_transfer_parameters is not None
                else PROVISIONAL_EPDM_LID_STATIC_FRICTION
            ),
            grip_dynamic_friction=(
                rg6_lid_transfer_parameters["dynamic_friction"]
                if rg6_lid_transfer_parameters is not None
                else PROVISIONAL_EPDM_LID_DYNAMIC_FRICTION
            ),
            grip_compliant_contact_stiffness_n_m=(
                rg6_lid_transfer_parameters[
                    "compliant_contact_stiffness_n_m"
                ]
                if rg6_lid_transfer_parameters is not None
                else 0.0
            ),
            grip_compliant_contact_damping_n_s_m=(
                rg6_lid_transfer_parameters[
                    "compliant_contact_damping_n_s_m"
                ]
                if rg6_lid_transfer_parameters is not None
                else 0.0
            ),
            enable_micro_lift_force_validation=True,
            # Development-only torque ceiling for the force-maintenance
            # proxy. Physical contact remains bounded by the predeclared
            # 60 N-per-finger and 3 mm penetration fail-closed gates. Replace
            # this mapping with the measured lab RG6 controller response.
            force_controller_max_torque_nm=(
                rg6_lid_transfer_parameters["maximum_drive_torque_nm"]
                if rg6_lid_transfer_parameters is not None
                else 6.0
            ),
            rg6_coupling_mode=args.rg6_coupling_mode,
            coordinated_total_drive_effort_limit_nm=(
                args.coordinated_rg6_total_drive_effort_limit_nm
            ),
        )
        if (
            persistent_remove_cover_result.get("status") != "completed"
            or not persistent_remove_cover_result.get("removal_verified")
            or not persistent_remove_cover_result.get(
                "cover_placed_and_released"
            )
        ):
            raise RuntimeError(
                "Persistent remove-cover execution failed: "
                f"{persistent_remove_cover_result.get('status')}"
            )
        capture_observation(
            "post_remove",
            set_robot_to_configured_pose=False,
        )
        post_remove_event = {
            "index": len(live_events),
            "type": "observation_ready",
            "view": "post_remove",
            "observation_dir": str(
                (output_root / "post_remove").resolve()
            ),
            "trajectory": {
                "status": "completed",
                "mode": "contact_gated_ur10e_rg6_cover_removal",
                "actual_robot_motion_executed": True,
                "continuous_physics_trajectory": True,
                "collision_checked": True,
            },
        }
        live_events.append(post_remove_event)
        write_live_json(
            live_session_dir
            / f"observation_ready_{post_remove_event['index']:03d}.json",
            post_remove_event,
        )
        record(
            "LIVE_PIPELINE_OBSERVATION_READY="
            f"{post_remove_event['index']}:post_remove"
        )
        if args.continue_after_remove_cover:
            post_remove_replan_request = wait_for_live_request(
                post_remove_event["index"], timeout_seconds=600.0
            )
            post_remove_replan_requests.append(
                post_remove_replan_request
            )
            replanned_type = str(
                post_remove_replan_request.get("type", "")
            )
            supported_replanned_actions = {
                "grasp",
                "grasp_inside",
                "grasp_outside",
                "viewpoint_close_high",
                "viewpoint_right",
                "defer",
                "stop",
            }
            if replanned_type not in supported_replanned_actions:
                raise RuntimeError(
                    "Unsupported post-remove replanned action: "
                    f"{replanned_type!r}"
                )
            terminal_action = replanned_type
            terminal_request = post_remove_replan_request
            record(
                "LIVE_PIPELINE_POST_REMOVE_REPLAN="
                f"{post_remove_event['index']}:{replanned_type}"
            )
            if replanned_type.startswith("viewpoint_"):
                selected_pose = replanned_type.removeprefix(
                    "viewpoint_"
                )
                if selected_pose not in capture_pose_names:
                    raise RuntimeError(
                        "Post-remove replan selected unknown view: "
                        f"{selected_pose}"
                    )
                app_utils.play()

                def post_remove_motion_update() -> None:
                    simulation_app.update()
                    pose_position, pose_orientation = current_ee_pose()
                    align_active_wrist_camera(
                        (pose_position, pose_orientation)
                    )

                for _ in range(4):
                    post_remove_motion_update()
                live_aabb_collision_pairs.clear()
                trajectory = move_pose_interpolated(
                    robot,
                    observation_config,
                    selected_pose,
                    post_remove_motion_update,
                    collision_checker=live_world_aabb_collision,
                )
                trajectory["actual_robot_motion_executed"] = True
                trajectory["continuous_physics_trajectory"] = True
                trajectory["collision_checked"] = True
                trajectory["collision_scope"] = (
                    "world_aabb_composite_collision_shape_vs_scene_obstacle"
                )
                trajectory["aabb_collision_pairs"] = list(
                    live_aabb_collision_pairs
                )
                app_utils.pause()
                simulation_app.update()
                if trajectory["status"] != "completed":
                    raise RuntimeError(
                        "Post-remove viewpoint motion failed: "
                        f"{trajectory}"
                    )
                capture_observation(
                    selected_pose,
                    set_robot_to_configured_pose=False,
                )
                post_view_event = {
                    "index": len(live_events),
                    "type": "observation_ready",
                    "view": selected_pose,
                    "observation_dir": str(
                        (output_root / selected_pose).resolve()
                    ),
                    "trajectory": trajectory,
                    "after_cover_removal": True,
                }
                live_events.append(post_view_event)
                write_live_json(
                    live_session_dir
                    / f"observation_ready_{post_view_event['index']:03d}.json",
                    post_view_event,
                )
                record(
                    "LIVE_PIPELINE_OBSERVATION_READY="
                    f"{post_view_event['index']}:{selected_pose}"
                )
                post_remove_replan_request = wait_for_live_request(
                    post_view_event["index"], timeout_seconds=600.0
                )
                post_remove_replan_requests.append(
                    post_remove_replan_request
                )
                replanned_type = str(
                    post_remove_replan_request.get("type", "")
                )
                if replanned_type not in {
                    "grasp",
                    "grasp_inside",
                    "grasp_outside",
                    "defer",
                    "stop",
                }:
                    raise RuntimeError(
                        "Unsupported second post-remove replanned action: "
                        f"{replanned_type!r}"
                    )
                terminal_action = replanned_type
                terminal_request = post_remove_replan_request
                record(
                    "LIVE_PIPELINE_POST_REMOVE_REPLAN="
                    f"{post_view_event['index']}:{replanned_type}"
                )
            if (
                replanned_type
                in {"grasp", "grasp_inside", "grasp_outside"}
                and post_remove_replan_request.get(
                    "physical_execution_requested", False
                )
            ):
                localization_value = post_remove_replan_request.get(
                    "rgbd_localization_path"
                )
                if not localization_value:
                    raise ValueError(
                        "Physical post-remove grasp requires an RGB-D "
                        "localization path"
                    )
                localization_path = Path(localization_value).resolve()
                if not localization_path.is_relative_to(live_session_dir):
                    raise ValueError(
                        "Post-remove RGB-D localization must be inside the "
                        f"live session: {localization_path}"
                    )
                post_remove_localization = json.loads(
                    localization_path.read_text(encoding="utf-8")
                )
                if post_remove_localization.get(
                    "simulator_ground_truth_used_for_estimate"
                ):
                    raise ValueError(
                        "Post-remove physical grasp localization reports "
                        "simulator ground-truth leakage"
                    )
                measured_array = robot.get_dof_positions().numpy()
                measured_vector = (
                    measured_array[0]
                    if measured_array.ndim > 1
                    else measured_array
                )
                post_remove_arm_positions = [
                    float(measured_vector[robot.dof_names.index(name)])
                    for name in observation_config["joint_order"]
                ]
                persistent_grasp_result = execute_persistent_composite_grasp(
                    project_root=PROJECT_ROOT,
                    stage=stage,
                    simulation_app=simulation_app,
                    rep=rep,
                    overview_rgb_annotator=overview_rgb_annotator,
                    overview_camera_path=overview_settings["path"],
                    output_root=(
                        live_session_dir / "persistent_post_remove_grasp"
                    ),
                    seed=args.seed,
                    reuse_existing_composite=True,
                    initial_arm_positions_rad=post_remove_arm_positions,
                    rgbd_localization=post_remove_localization,
                    # The cover needs the development coordinated-drive proxy,
                    # but target transport is stable only with the authored
                    # RG6 Newton linkage.  The executor restores that schema
                    # in memory without rebuilding the stage or resetting the
                    # already released cover pose.
                    rg6_coupling_mode="passive_mimic",
                    coordinated_total_drive_effort_limit_nm=None,
                    coordinated_follower_request_blend=(
                        PROVISIONAL_TARGET_FOLLOWER_REQUEST_BLEND
                    ),
                    minimum_grip_force_per_finger_n=(
                        PROVISIONAL_TARGET_MINIMUM_GRIP_FORCE_PER_FINGER_N
                    ),
                    force_controller_max_torque_nm=(
                        PROVISIONAL_TARGET_PASSIVE_FORCE_CONTROLLER_MAX_TORQUE_NM
                    ),
                    # Require a verified 1 cm contact micro-lift before the
                    # full target lift, just as the cover manipulation does.
                    # This keeps the same force, penetration, slip, and
                    # bilateral-contact gates active at lift onset.
                    enable_micro_lift_force_validation=True,
                )
                if (
                    persistent_grasp_result.get("status") != "completed"
                    or not persistent_grasp_result.get("lift_verified")
                ):
                    raise RuntimeError(
                        "Persistent post-remove target grasp failed: "
                        f"{persistent_grasp_result.get('status')}"
                    )
                record(
                    "LIVE_PIPELINE_POST_REMOVE_GRASP_COMPLETED="
                    f"{post_remove_event['index']}:{replanned_type}"
                )

    live_pipeline_result = {
        "schema_version": "live-isaac-pipeline-server-v1",
        "status": "completed",
        "seed": args.seed,
        "events": live_events,
        "root_terminal_action": root_terminal_action,
        "terminal_action": terminal_action,
        "grasp_executed": bool(
            persistent_grasp_result
            and persistent_grasp_result.get("lift_verified")
        ),
        "grasp_execution": persistent_grasp_result,
        "cover_removal_executed": bool(
            persistent_remove_cover_result
            and persistent_remove_cover_result.get("removal_verified")
        ),
        "cover_removal_execution": persistent_remove_cover_result,
        "post_removal_observation_generated": bool(
            persistent_remove_cover_result
            and persistent_remove_cover_result.get("removal_verified")
        ),
        "post_remove_replan_request_received": bool(
            post_remove_replan_request is not None
        ),
        "post_remove_replan_requests": post_remove_replan_requests,
        "post_remove_replanned_action": (
            post_remove_replan_request.get("type")
            if post_remove_replan_request is not None
            else None
        ),
        "post_remove_replanned_action_physical_execution": bool(
            post_remove_replan_request is not None
            and post_remove_replan_request.get(
                "physical_execution_requested", False
            )
            and persistent_grasp_result
            and persistent_grasp_result.get("lift_verified")
        ),
        "execution_mode": (
            "single_persistent_isaac_process_and_stage_with_external_vlm_ipc"
            if (
                args.execute_persistent_composite_grasp
                or args.execute_persistent_remove_cover
            )
            else "single_persistent_isaac_process_with_external_vlm_ipc"
        ),
        "pre_captured_candidate_replay": False,
        "gpu_policy": {
            "renderer_physical_gpu": args.renderer_gpu,
            "physics_cuda_device": args.physics_gpu,
            "multi_gpu": False,
        },
        "valid_for_final_evaluation": False,
    }
    write_live_json(
        live_session_dir / "server_result.json",
        live_pipeline_result,
    )
    record(f"LIVE_PIPELINE_SERVER_COMPLETED={terminal_action}")
elif args.candidate_view_demo:
    from PIL import Image, ImageDraw, ImageFont

    from build_observation_video import build_frame_sequence_video

    candidate_pose = "close_high"
    if candidate_pose not in observation_config["candidate_views"]["poses"]:
        raise RuntimeError(f"Unknown candidate viewpoint: {candidate_pose}")
    demo_root = PROJECT_ROOT / "outputs" / "candidate_view_demo" / "benchmark_seed000"
    frame_root = demo_root / "overview_frames"
    frame_root.mkdir(parents=True, exist_ok=True)
    frame_paths = []
    frame_counter = {"value": 0}

    try:
        candidate_font = ImageFont.truetype(
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 40
        )
    except OSError:
        candidate_font = ImageFont.load_default()

    def save_candidate_frame(label: str) -> None:
        rep.orchestrator.step(rt_subframes=2)
        rgba = np.asarray(overview_rgb_annotator.get_data()).copy()
        image = Image.fromarray(rgba[:, :, :3].astype(np.uint8), "RGB")
        draw = ImageDraw.Draw(image)
        text_box = draw.textbbox((0, 0), label, font=candidate_font)
        text_width = text_box[2] - text_box[0]
        text_height = text_box[3] - text_box[1]
        draw.rounded_rectangle(
            (18, 18, 42 + text_width, 38 + text_height),
            radius=8,
            fill=(0, 0, 0),
        )
        draw.text((30, 26), label, fill=(255, 255, 255), font=candidate_font)
        frame_path = frame_root / f"frame_{frame_counter['value']:04d}.png"
        image.save(frame_path)
        frame_paths.append(frame_path)
        frame_counter["value"] += 1

    capture_observation("center")
    for _ in range(12):
        save_candidate_frame("Center wrist RGB-D")

    app_utils.play()
    motion_update_counter = {"value": 0}

    def candidate_motion_update() -> None:
        simulation_app.update()
        pose_position, pose_orientation = current_ee_pose()
        align_tool_and_camera(
            stage,
            ur10e_ee,
            rg6_prim,
            zivid_camera,
            observation_config,
            (pose_position, pose_orientation),
        )
        if motion_update_counter["value"] % 3 == 0:
            save_candidate_frame("Moving to provisional close/high wrist view")
        motion_update_counter["value"] += 1

    trajectory_result = move_pose_interpolated(
        robot,
        observation_config,
        candidate_pose,
        candidate_motion_update,
    )
    app_utils.pause()
    simulation_app.update()
    if trajectory_result["status"] != "completed":
        raise RuntimeError(f"Candidate-view trajectory failed: {trajectory_result}")
    capture_observation(candidate_pose)
    candidate_ee_position, _ = current_ee_pose()
    for _ in range(15):
        save_candidate_frame("Close/high wrist RGB-D captured")

    video_result = build_frame_sequence_video(
        frame_paths,
        demo_root / "close_high_candidate_demo.mp4",
        fps=5,
        crf=quality_settings["video_crf"],
        preset=quality_settings["video_preset"],
        purpose="provisional_simulation_wrist_view_validation_not_final_evaluation",
    )
    candidate_view_result = {
        "schema_version": "candidate-wrist-view-demo-v1",
        "status": "completed",
        "initial_observation": "center",
        "executed_viewpoint": candidate_pose,
        "candidate_ee_position_world_m": candidate_ee_position.tolist(),
        "trajectory": trajectory_result,
        "video": video_result,
        "rgbd_output_root": str(output_root.resolve()),
        "training_performed": False,
        "fresh_vlm_inference_performed": False,
        "valid_for_real_robot_execution": False,
        "valid_for_final_evaluation": False,
        "limitations": (
            "The candidate joint pose is simulation-provisional. Physical camera "
            "mount verification, hand-eye calibration, lab workspace checks, and "
            "a real collision-free trajectory are required before robot use."
        ),
    }
    candidate_result_path = demo_root / "candidate_view_demo.json"
    candidate_result_path.write_text(
        json.dumps(candidate_view_result, indent=2) + "\n",
        encoding="utf-8",
    )
    record(
        "CANDIDATE_VIEW_DEMO_COMPLETED="
        f"pose={candidate_pose} frames={len(frame_paths)} "
        f"video={video_result['output_path']}"
    )
elif args.meeting_demo:
    from PIL import Image, ImageDraw, ImageFont

    from build_observation_video import build_frame_sequence_video

    pilot_result = json.loads(
        args.pilot_result.resolve().read_text(encoding="utf-8")
    )
    if pilot_result.get("status") != "completed":
        raise RuntimeError("Pilot result is not completed")
    first_action = pilot_result["first_action"]["type"]
    if not first_action.startswith("viewpoint_"):
        raise RuntimeError(f"Meeting demo requires a viewpoint action: {first_action}")
    selected_pose = first_action.removeprefix("viewpoint_")
    if selected_pose not in observation_config["capture"]["poses"]:
        raise RuntimeError(f"Unknown meeting-demo viewpoint: {selected_pose}")
    replanned_action = pilot_result["replan"]["type"]

    demo_root = PROJECT_ROOT / "outputs" / "meeting_demo" / "benchmark_seed000"
    frame_root = demo_root / "overview_frames"
    frame_root.mkdir(parents=True, exist_ok=True)
    frame_paths = []
    frame_counter = {"value": 0}

    try:
        meeting_font = ImageFont.truetype(
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 40
        )
    except OSError:
        meeting_font = ImageFont.load_default()

    def save_meeting_frame(label: str) -> None:
        rep.orchestrator.step(rt_subframes=2)
        rgba = np.asarray(overview_rgb_annotator.get_data()).copy()
        image = Image.fromarray(rgba[:, :, :3].astype(np.uint8), "RGB")
        draw = ImageDraw.Draw(image)
        text_box = draw.textbbox((0, 0), label, font=meeting_font)
        text_width = text_box[2] - text_box[0]
        text_height = text_box[3] - text_box[1]
        draw.rounded_rectangle(
            (18, 18, 42 + text_width, 38 + text_height),
            radius=8,
            fill=(0, 0, 0),
        )
        draw.text((30, 26), label, fill=(255, 255, 255), font=meeting_font)
        frame_path = frame_root / f"frame_{frame_counter['value']:04d}.png"
        image.save(frame_path)
        frame_paths.append(frame_path)
        frame_counter["value"] += 1

    capture_observation("center")
    for _ in range(15):
        save_meeting_frame(
            f"Center RGB-D | cached Qwen action: {first_action}"
        )

    app_utils.play()
    motion_update_counter = {"value": 0}
    current_motion_action = {"value": first_action}

    def meeting_motion_update() -> None:
        simulation_app.update()
        pose_position, pose_orientation = current_ee_pose()
        align_tool_and_camera(
            stage,
            ur10e_ee,
            rg6_prim,
            zivid_camera,
            observation_config,
            (pose_position, pose_orientation),
        )
        if motion_update_counter["value"] % 3 == 0:
            save_meeting_frame(
                f"Executing {current_motion_action['value']}"
            )
        motion_update_counter["value"] += 1

    trajectory_result = move_pose_interpolated(
        robot,
        observation_config,
        selected_pose,
        meeting_motion_update,
    )
    app_utils.pause()
    simulation_app.update()
    if trajectory_result["status"] != "completed":
        raise RuntimeError(f"Meeting-demo trajectory failed: {trajectory_result}")
    capture_observation(selected_pose)
    for _ in range(12):
        save_meeting_frame(
            f"New {selected_pose} RGB-D | replanned: {replanned_action}"
        )

    replanned_trajectory = None
    replanned_pose = None
    if replanned_action.startswith("viewpoint_"):
        replanned_pose = replanned_action.removeprefix("viewpoint_")
        if replanned_pose not in observation_config["capture"]["poses"]:
            raise RuntimeError(
                f"Unknown replanned meeting-demo viewpoint: {replanned_pose}"
            )
        current_motion_action["value"] = replanned_action
        motion_update_counter["value"] = 0
        app_utils.play()
        replanned_trajectory = move_pose_interpolated(
            robot,
            observation_config,
            replanned_pose,
            meeting_motion_update,
        )
        app_utils.pause()
        simulation_app.update()
        if replanned_trajectory["status"] != "completed":
            raise RuntimeError(
                "Meeting-demo replanned trajectory failed: "
                f"{replanned_trajectory}"
            )
        capture_observation(replanned_pose)
        for _ in range(15):
            save_meeting_frame(
                f"Executed replan: {replanned_action} | new RGB-D saved"
            )

    video_result = build_frame_sequence_video(
        frame_paths,
        demo_root / "meeting_active_view_demo.mp4",
        fps=5,
        crf=quality_settings["video_crf"],
        preset=quality_settings["video_preset"],
        purpose=(
            "cached_qwen_active_view_pipeline_meeting_demo_"
            "not_final_evaluation"
        ),
    )
    meeting_demo_result = {
        "schema_version": "meeting-active-view-demo-v1",
        "status": "completed",
        "pilot_result": str(args.pilot_result.resolve()),
        "initial_observation": "center",
        "cached_qwen_action": first_action,
        "executed_viewpoint": selected_pose,
        "new_observation": selected_pose,
        "replanned_action": replanned_action,
        "executed_replanned_viewpoint": replanned_pose,
        "trajectory": trajectory_result,
        "replanned_trajectory": replanned_trajectory,
        "video": video_result,
        "training_performed": False,
        "fresh_vlm_inference_performed": False,
        "valid_for_final_evaluation": False,
        "limitations": (
            "The VLM decision and belief update are replayed from the cached "
            "deterministic pilot. Both robot viewpoint motions and all RGB-D "
            "captures are executed live in Isaac Sim. No third Qwen inference "
            "or post-left replan is performed."
        ),
    }
    meeting_result_path = demo_root / "meeting_demo.json"
    meeting_result_path.write_text(
        json.dumps(meeting_demo_result, indent=2) + "\n",
        encoding="utf-8",
    )
    record(
        "MEETING_DEMO_COMPLETED="
        f"action={first_action} replan={replanned_action} "
        f"frames={len(frame_paths)} video={video_result['output_path']}"
    )
elif args.movement_demo:
    demo_config = observation_config
    set_pose(robot, demo_config, "center", simulation_app.update, 1)
    pose_position, pose_orientation = current_ee_pose()
    align_tool_and_camera(
        stage,
        ur10e_ee,
        rg6_prim,
        zivid_camera,
        demo_config,
        (pose_position, pose_orientation),
    )
    record("MOVEMENT_DEMO_READY_WAIT=5s")
    for _ in range(300):
        simulation_app.update()
    demo_results = []
    for pose_name in ("left", "center", "right", "center"):
        app_utils.pause()
        joint_indices = [
            robot.dof_names.index(name) for name in demo_config["joint_order"]
        ]
        current_array = robot.get_dof_positions().numpy()
        current_vector = current_array[0] if current_array.ndim > 1 else current_array
        start = np.asarray(current_vector[joint_indices], dtype=np.float64)
        target = np.asarray(demo_config["poses_rad"][pose_name], dtype=np.float64)
        visual_steps = 90
        for step in range(1, visual_steps + 1):
            alpha = step / visual_steps
            waypoint = start + alpha * (target - start)
            robot.set_dof_positions(
                waypoint.tolist(), dof_indices=joint_indices
            )
            robot.set_dof_position_targets(
                waypoint.tolist(), dof_indices=joint_indices
            )
            for _ in range(2):
                simulation_app.update()
        measured_array = robot.get_dof_positions().numpy()
        measured_vector = (
            measured_array[0] if measured_array.ndim > 1 else measured_array
        )
        measured = np.asarray(measured_vector[joint_indices], dtype=np.float64)
        result = {
            "status": "completed",
            "pose_name": pose_name,
            "motion_mode": "paused_visual_direct_joint_interpolation",
            "visual_steps": visual_steps,
            "maximum_joint_error_rad": float(
                np.max(np.abs(measured - target))
            ),
            "physics_result_valid": False,
        }
        pose_position, pose_orientation = current_ee_pose()
        align_tool_and_camera(
            stage,
            ur10e_ee,
            rg6_prim,
            zivid_camera,
            demo_config,
            (pose_position, pose_orientation),
        )
        demo_results.append(result)
        record(
            f"MOVEMENT_DEMO_POSE={pose_name} "
            f"visual_steps={result['visual_steps']} "
            f"max_error={result['maximum_joint_error_rad']}"
        )
        for _ in range(120):
            simulation_app.update()
    demo_output = PROJECT_ROOT / "outputs" / "movement_demo.json"
    demo_output.parent.mkdir(parents=True, exist_ok=True)
    demo_output.write_text(
        json.dumps(
            {
                "status": "completed",
                "sequence": ["center", "left", "center", "right", "center"],
                "startup_wait_seconds": 5,
                "hold_seconds_per_pose": 2,
                "results": demo_results,
                "purpose": "visual_inspection_only_not_physics_or_mpc",
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    record("MOVEMENT_DEMO_COMPLETED")
elif args.execute_non_oracle_plan:
    import copy

    from run_non_oracle_hybrid_planner import load_json as load_method_json
    from run_non_oracle_hybrid_planner import plan as make_non_oracle_plan
    from update_belief_from_executed_observation import update_from_observation

    method_config_path = args.method_config.resolve()
    method_config = load_method_json(method_config_path)
    non_oracle_output = PROJECT_ROOT / method_config["output_root"]
    non_oracle_output.mkdir(parents=True, exist_ok=True)

    capture_observation("center")
    initial_plan = make_non_oracle_plan(method_config)
    (non_oracle_output / "pre_action_plan.json").write_text(
        json.dumps(initial_plan, indent=2) + "\n", encoding="utf-8"
    )
    action_name = initial_plan["action_request"]["type"]
    if not action_name.startswith("viewpoint_"):
        raise RuntimeError(f"Expected a viewpoint action, got: {action_name}")
    selected_pose = action_name.removeprefix("viewpoint_")
    if selected_pose not in capture_pose_names:
        raise RuntimeError(f"Unknown non-oracle observation pose: {selected_pose}")
    record(f"NON_ORACLE_PRE_ACTION_PLAN={action_name}")

    def maximum_contact_force_non_oracle() -> float:
        forces = contact_monitor.get_net_contact_forces(dt=1.0 / 60.0).numpy()
        return float((forces**2).sum(axis=1).max() ** 0.5)

    collision_obstacle_paths = [
        "/World/WorkBench",
        "/World/OpenContainer",
        "/World/TargetRed",
        "/World/OccluderOrange",
        "/World/DistractorBlue",
    ]

    def world_aabb_collision_non_oracle() -> bool:
        cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_])
        obstacle_ranges = [
            cache.ComputeWorldBound(stage.GetPrimAtPath(path)).ComputeAlignedRange()
            for path in collision_obstacle_paths
            if stage.GetPrimAtPath(path).IsValid()
        ]
        for link_path in moving_link_paths:
            link_range = cache.ComputeWorldBound(
                stage.GetPrimAtPath(link_path)
            ).ComputeAlignedRange()
            link_min, link_max = link_range.GetMin(), link_range.GetMax()
            for obstacle_range in obstacle_ranges:
                obstacle_min, obstacle_max = obstacle_range.GetMin(), obstacle_range.GetMax()
                if all(
                    link_min[axis] <= obstacle_max[axis]
                    and link_max[axis] >= obstacle_min[axis]
                    for axis in range(3)
                ):
                    return True
        return False

    viewpoint_execution_mode = method_config.get(
        "viewpoint_execution", {}
    ).get("mode", "interpolated_joint_physics")

    def capture_debug_or_robot_view(pose_name: str) -> dict:
        debug_positions = method_config.get("viewpoint_execution", {}).get(
            "debug_ee_positions_world_m", {}
        )
        debug_position = debug_positions.get(pose_name)
        capture_observation(
            pose_name,
            debug_ee_position_world_m=debug_position,
        )
        return {
            "status": "completed",
            "mode": (
                "debug_synthetic_wrist_pose_capture"
                if debug_position is not None
                else "debug_pose_capture"
            ),
            "continuous_physics_trajectory": False,
            "collision_checked": False,
            "actual_robot_motion_executed": False,
            "debug_ee_position_world_m": debug_position,
        }

    if viewpoint_execution_mode == "debug_pose_capture":
        # The first integration validates the closed-loop information flow.
        # It deliberately uses the existing simulation pose-capture action
        # because the official UR10e USD ee_joint becomes unstable during the
        # provisional continuous transition.  Do not call this physical MPC.
        trajectory_result = capture_debug_or_robot_view(selected_pose)
    else:
        app_utils.play()
        set_pose(
            robot,
            observation_config,
            "center",
            simulation_app.update,
            1,
        )
        trajectory_result = move_pose_interpolated(
            robot,
            observation_config,
            selected_pose,
            simulation_app.update,
            maximum_contact_force_non_oracle
            if contact_monitor is not None
            else None,
            world_aabb_collision_non_oracle,
        )
        app_utils.pause()
        simulation_app.update()
        if trajectory_result["status"] != "completed":
            raise RuntimeError(
                f"Non-oracle trajectory failed: {trajectory_result}"
            )
        capture_observation(selected_pose)
    actual_objects_path = output_root / selected_pose / "objects.json"
    actual_objects = json.loads(actual_objects_path.read_text(encoding="utf-8"))
    belief_update = update_from_observation(
        initial_plan["current_belief"],
        action_name,
        actual_objects,
        method_config,
    )
    (non_oracle_output / "belief_update.json").write_text(
        json.dumps(belief_update, indent=2) + "\n", encoding="utf-8"
    )

    replanning_config = copy.deepcopy(method_config)
    replanning_config["initial_belief"]["target"] = belief_update["posterior"]["target"]
    replanning_config["initial_belief"]["relation"] = belief_update["posterior"][
        "relation"
    ]
    replanning_config["initial_belief"]["source"] = (
        "post_action_simulator_observation_adapter"
    )
    replanning_config["completed_reobservations"] = (
        int(method_config.get("completed_reobservations", 0)) + 1
    )
    for candidate_name, candidate_action in replanning_config["actions"].items():
        enable_after = candidate_action.get(
            "enable_after_completed_reobservations"
        )
        if (
            enable_after is not None
            and replanning_config["completed_reobservations"] >= enable_after
        ):
            candidate_action["enabled"] = True
            candidate_action.pop("disabled_until", None)
    current_joints = observation_config["poses_rad"][selected_pose]
    for pose_name in observation_config["capture"]["poses"]:
        action_key = f"viewpoint_{pose_name}"
        if action_key not in replanning_config["actions"]:
            continue
        candidate_joints = observation_config["poses_rad"][pose_name]
        replanning_config["actions"][action_key]["motion_cost"] = (
            sum(
                abs(current - candidate)
                for current, candidate in zip(current_joints, candidate_joints)
            )
            / len(current_joints)
        )
    replanning_config["actions"][action_name]["enabled"] = False
    replanning_config["actions"][action_name]["disabled_until"] = (
        "a_new_physical_viewpoint_change"
    )
    replanned = make_non_oracle_plan(replanning_config)
    (non_oracle_output / "post_action_replan.json").write_text(
        json.dumps(replanned, indent=2) + "\n", encoding="utf-8"
    )

    reobservation_steps = [
        {
            "index": 1,
            "action": action_name,
            "pose": selected_pose,
            "observation": str(actual_objects_path.relative_to(PROJECT_ROOT)),
            "belief_update": str(
                (non_oracle_output / "belief_update.json").relative_to(
                    PROJECT_ROOT
                )
            ),
            "trajectory": trajectory_result,
        }
    ]
    final_replan = replanned
    second_action = replanned["action_request"]["type"]
    if second_action.startswith("viewpoint_"):
        if viewpoint_execution_mode != "debug_pose_capture":
            raise RuntimeError(
                "Two-step integration currently supports debug pose capture only"
            )
        second_pose = second_action.removeprefix("viewpoint_")
        if second_pose not in capture_pose_names:
            raise RuntimeError(
                f"Unknown second non-oracle observation pose: {second_pose}"
            )
        second_trajectory = capture_debug_or_robot_view(second_pose)
        second_objects_path = output_root / second_pose / "objects.json"
        second_objects = json.loads(
            second_objects_path.read_text(encoding="utf-8")
        )
        second_update = update_from_observation(
            replanned["current_belief"],
            second_action,
            second_objects,
            replanning_config,
        )
        second_update_path = non_oracle_output / "belief_update_002.json"
        second_update_path.write_text(
            json.dumps(second_update, indent=2) + "\n",
            encoding="utf-8",
        )
        final_config = copy.deepcopy(replanning_config)
        final_config["initial_belief"]["target"] = second_update["posterior"][
            "target"
        ]
        final_config["initial_belief"]["relation"] = second_update["posterior"][
            "relation"
        ]
        final_config["initial_belief"]["source"] = (
            "second_post_action_simulator_observation_adapter"
        )
        final_config["completed_reobservations"] = (
            replanning_config["completed_reobservations"] + 1
        )
        final_config["actions"][second_action]["enabled"] = False
        final_config["actions"][second_action]["disabled_until"] = (
            "a_new_physical_viewpoint_change"
        )
        final_replan = make_non_oracle_plan(final_config)
        final_replan_path = non_oracle_output / "post_action_replan_002.json"
        final_replan_path.write_text(
            json.dumps(final_replan, indent=2) + "\n",
            encoding="utf-8",
        )
        reobservation_steps.append(
            {
                "index": 2,
                "action": second_action,
                "pose": second_pose,
                "observation": str(
                    second_objects_path.relative_to(PROJECT_ROOT)
                ),
                "belief_update": str(
                    second_update_path.relative_to(PROJECT_ROOT)
                ),
                "trajectory": second_trajectory,
            }
        )

    joint_indices = [
        robot.dof_names.index(name)
        for name in observation_config["joint_order"]
    ]
    measured_array = robot.get_dof_positions().numpy()
    measured_vector = (
        measured_array[0] if measured_array.ndim > 1 else measured_array
    )
    measured = measured_vector[joint_indices].tolist()
    requested = observation_config["poses_rad"][selected_pose]
    finite_measured = bool(np.all(np.isfinite(measured)))
    continuous_execution = (
        viewpoint_execution_mode != "debug_pose_capture"
    )
    absolute_errors = (
        [
            abs(actual - target)
            for actual, target in zip(measured, requested)
        ]
        if finite_measured and continuous_execution
        else None
    )
    maximum_joint_error = (
        max(absolute_errors) if absolute_errors is not None else None
    )
    execution = {
        "status": (
            "completed"
            if (
                not continuous_execution
                or (
                    maximum_joint_error is not None
                    and maximum_joint_error
                    <= observation_config["trajectory"][
                        "final_tolerance_rad"
                    ]
                )
            )
            else "joint_verification_failed"
        ),
        "pre_action_plan": str(
            (non_oracle_output / "pre_action_plan.json").relative_to(PROJECT_ROOT)
        ).replace("\\", "/"),
        "executed_action": action_name,
        "actual_observation": str(
            actual_objects_path.relative_to(PROJECT_ROOT)
        ).replace("\\", "/"),
        "belief_update": str(
            (non_oracle_output / "belief_update.json").relative_to(PROJECT_ROOT)
        ).replace("\\", "/"),
        "post_action_replan": str(
            (non_oracle_output / "post_action_replan.json").relative_to(
                PROJECT_ROOT
            )
        ).replace("\\", "/"),
        "post_action_first_action": replanned["action_request"]["type"],
        "final_replanned_action": final_replan["action_request"]["type"],
        "final_commitment_gate": final_replan["commitment_gate"],
        "completed_reobservations": len(reobservation_steps),
        "reobservation_steps": reobservation_steps,
        "trajectory": trajectory_result,
        "maximum_joint_error_rad": maximum_joint_error,
        "future_capture_used_in_pre_action_plan": False,
        "actual_capture_consumed_only_after_motion": True,
        "simulated_viewpoint_action_executed": True,
        "actual_robot_motion_executed": continuous_execution,
        "continuous_physics_trajectory_executed": continuous_execution,
        "planner_label": "non_oracle_receding_horizon_engineering_prototype",
        "mpc_claim_allowed": False,
        "limitations": (
            "The observation model is hand specified, the post-action adapter "
            "uses simulator instance labels, the overhead view uses a synthetic "
            "debug wrist pose, and no final continuous MPC trajectory is run."
        ),
    }
    (non_oracle_output / "execution.json").write_text(
        json.dumps(execution, indent=2) + "\n", encoding="utf-8"
    )
    record(
        f"NON_ORACLE_EXECUTION status={execution['status']} "
        f"executed={action_name} "
        f"replan={execution['post_action_first_action']} "
        f"final={execution['final_replanned_action']}"
    )
elif args.execute_action_request:
    capture_observation("center")
    record("ACTIVE_VIEW_INITIAL_CAPTURE=center")

    from build_scene_graphs import main as build_scene_graphs
    from build_uncertainty_stub_graphs import main as build_uncertainty_stub_graphs
    from run_active_view_controller_stub import main as run_active_view_controller

    original_argv = sys.argv
    try:
        sys.argv = [sys.argv[0]]
        build_scene_graphs()
        build_uncertainty_stub_graphs()
        run_active_view_controller()
    finally:
        sys.argv = original_argv
    action_request = json.loads(ACTION_REQUEST_PATH.read_text(encoding="utf-8"))
    if action_request["type"] != "move_to_observation_pose":
        raise RuntimeError(f"Unsupported active-view action: {action_request}")
    selected_pose = action_request["pose_name"]
    if selected_pose not in observation_config["capture"]["poses"]:
        raise RuntimeError(f"Unknown observation pose requested: {selected_pose}")

    def maximum_contact_force() -> float:
        forces = contact_monitor.get_net_contact_forces(dt=1.0 / 60.0).numpy()
        return float((forces**2).sum(axis=1).max() ** 0.5)

    collision_obstacle_paths = observation_config["trajectory"]["contact_filter_paths"]

    def world_aabb_collision() -> bool:
        cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_])
        obstacle_ranges = [
            cache.ComputeWorldBound(stage.GetPrimAtPath(path)).ComputeAlignedRange()
            for path in collision_obstacle_paths
        ]
        for link_path in moving_link_paths:
            link_range = cache.ComputeWorldBound(
                stage.GetPrimAtPath(link_path)
            ).ComputeAlignedRange()
            link_min, link_max = link_range.GetMin(), link_range.GetMax()
            for obstacle_range in obstacle_ranges:
                obstacle_min, obstacle_max = obstacle_range.GetMin(), obstacle_range.GetMax()
                if all(
                    link_min[axis] <= obstacle_max[axis]
                    and link_max[axis] >= obstacle_min[axis]
                    for axis in range(3)
                ):
                    return True
        return False

    app_utils.play()
    trajectory_result = move_pose_interpolated(
        robot,
        observation_config,
        selected_pose,
        simulation_app.update,
        maximum_contact_force if contact_monitor is not None else None,
        world_aabb_collision,
    )
    app_utils.pause()
    simulation_app.update()
    if trajectory_result["status"] != "completed":
        raise RuntimeError(f"Active-view trajectory failed: {trajectory_result}")
    pose_position, pose_orientation = current_ee_pose()
    align_tool_and_camera(
        stage,
        ur10e_ee,
        rg6_prim,
        zivid_camera,
        observation_config,
        (pose_position, pose_orientation),
    )
    for _ in range(8):
        simulation_app.update()
    rep.orchestrator.step(rt_subframes=4)
    rep.orchestrator.step(rt_subframes=4)
    save_capture(
        output_root,
        selected_pose,
        rgb_annotator.get_data(),
        depth_annotator.get_data(),
        overview_rgb_data=overview_rgb_annotator.get_data(),
        camera_provenance=camera_provenance,
    )
    record("CAPTURED_POSE=" + selected_pose)
    original_argv = sys.argv
    try:
        sys.argv = [sys.argv[0]]
        build_scene_graphs()
        build_uncertainty_stub_graphs()
    finally:
        sys.argv = original_argv
    record("ACTIVE_VIEW_ACTION_EXECUTED=" + selected_pose)

    joint_indices = [robot.dof_names.index(name) for name in observation_config["joint_order"]]
    measured_array = robot.get_dof_positions().numpy()
    measured_vector = measured_array[0] if measured_array.ndim > 1 else measured_array
    measured = measured_vector[joint_indices].tolist()
    requested = observation_config["poses_rad"][selected_pose]
    absolute_errors = [abs(actual - target) for actual, target in zip(measured, requested)]
    execution = {
        "status": "completed" if max(absolute_errors) <= 0.02 else "joint_verification_failed",
        "action_request": action_request,
        "initial_capture": "center",
        "selected_capture": selected_pose,
        "requested_joint_positions_rad": requested,
        "measured_joint_positions_rad": measured,
        "absolute_joint_errors_rad": absolute_errors,
        "maximum_joint_error_rad": max(absolute_errors),
        "verification_tolerance_rad": observation_config["trajectory"]["final_tolerance_rad"],
        "actual_robot_motion_executed": True,
        "motion_mode": "interpolated_joint_position_targets",
        "continuous_trajectory_executed": True,
        "trajectory": trajectory_result,
        "capture_files": [
            f"outputs/observations/{selected_pose}/rgb.png",
            f"outputs/observations/{selected_pose}/depth_m.npy",
            f"outputs/observations/{selected_pose}/objects.json",
            f"outputs/observations/{selected_pose}/scene_graph.json",
            f"outputs/observations/{selected_pose}/uncertainty_scene_graph_stub.json",
        ],
        "limitations": (
            "The candidate outcome predictor still uses offline ground-truth-derived replay. "
            "The transition is deterministic joint-space interpolation, not MPC. "
            "When the PhysX contact view is unavailable, collision monitoring falls back "
            "to conservative world-AABB overlap; the kinematic RG6 visual mount is not "
            "included in swept-volume checking."
        ),
    }
    ACTION_EXECUTION_PATH.parent.mkdir(parents=True, exist_ok=True)
    ACTION_EXECUTION_PATH.write_text(
        json.dumps(execution, indent=2), encoding="utf-8"
    )
    record(
        f"ACTION_VERIFICATION status={execution['status']} "
        f"max_joint_error_rad={execution['maximum_joint_error_rad']}"
    )
else:
    for pose_name in capture_pose_names:
        capture_observation(pose_name)
    if args.scene_profile == "benchmark" and not args.seeded_pilot_capture:
        from build_benchmark_scene_graphs import main as build_benchmark_scene_graphs
        from build_benchmark_uncertainty_graphs import (
            main as build_benchmark_uncertainty_graphs,
        )
        from run_benchmark_active_view_stub import (
            main as run_benchmark_active_view_stub,
        )

        build_benchmark_scene_graphs()
        build_benchmark_uncertainty_graphs()
        run_benchmark_active_view_stub()
        record(
            "BENCHMARK_UNCERTAINTY_AND_ACTIVE_VIEW_BUILT=left|center|right"
        )
    app_utils.play()
    simulation_app.update()
    set_pose(robot, observation_config, "center", simulation_app.update, 1)
    app_utils.pause()
    simulation_app.update()
    align_tool_and_camera(
        stage, ur10e_ee, rg6_prim, zivid_camera, observation_config, current_ee_pose()
    )
    record("RG6_AND_CAMERA_ALIGNED_TO_EE")

table_prim_path = "/World/WorkBench" if args.scene_profile == "benchmark" else "/World/Table"
bbox_cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_])
for bbox_path in (
    "/World/RobotSystem/UR10e",
    "/World/RobotSystem/RG6",
    "/World/Ground",
    table_prim_path,
    "/World/OpenContainer",
    "/World/TargetRed",
    "/World/DistractorBlue",
):
    bbox_prim = stage.GetPrimAtPath(bbox_path)
    if not bbox_prim.IsValid() or not bbox_prim.IsActive():
        record(f"BBOX_SKIPPED_INVALID_OR_INACTIVE={bbox_path}")
        continue
    bbox_range = bbox_cache.ComputeWorldBound(bbox_prim).ComputeAlignedRange()
    record(f"BBOX {bbox_path} min={bbox_range.GetMin()} max={bbox_range.GetMax()}")

viewport = get_active_viewport()
if viewport is not None:
    viewport.camera_path = Sdf.Path("/OmniverseKit_Persp")
    for _ in range(10):
        simulation_app.update()
    frame_viewport_prims(
        viewport,
        [
            table_prim_path,
            "/World/OpenContainer",
            "/World/TargetRed",
            "/World/DistractorBlue",
            "/World/RobotSystem/UR10e",
            "/World/RobotSystem/RG6",
        ],
    )
    for _ in range(5):
        simulation_app.update()
    camera_state = ViewportCameraState("/OmniverseKit_Persp", viewport=viewport)
    camera_state.set_position_world(Gf.Vec3d(2.35, -2.35, 2.05), False)
    camera_state.set_target_world(Gf.Vec3d(0.25, 0.0, 0.72), True)

settings = carb.settings.get_settings()
settings.set_bool("/app/runLoops/main/rateLimitEnabled", True)
settings.set_int("/app/runLoops/main/rateLimitFrequency", 60)

print(f"OPENED_SCENE={SCENE_PATH}")
print(
    "GPU_CONFIGURATION="
    f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')},"
    f"renderer_physical_gpu={args.renderer_gpu},"
    f"physics_cuda_device={args.physics_gpu},"
    "multi_gpu=False"
)
print(
    "ACTIVE_CAMERA="
    + (
        "/World/RobotSystem/RG6/Zivid2Camera"
        if args.headless
        else "/OmniverseKit_Persp"
    )
)
print("ROBOT_STACK=UR10e+OnRobot_RG6+Zivid2_provisional")
if meeting_demo_result is not None:
    print(
        "MEETING_DEMO_VIDEO="
        + meeting_demo_result["video"]["output_path"]
    )
    print(
        "MEETING_DEMO_REPLAN="
        + meeting_demo_result["replanned_action"]
    )
if candidate_view_result is not None:
    print(
        "CANDIDATE_VIEW_VIDEO="
        + candidate_view_result["video"]["output_path"]
    )
    print(
        "CANDIDATE_VIEW_RGBD="
        + candidate_view_result["rgbd_output_root"]
    )
if live_pipeline_result is not None:
    print(
        "LIVE_PIPELINE_SERVER_RESULT="
        + str((args.live_session_dir.resolve() / "server_result.json"))
    )

if args.capture_video:
    from build_observation_video import build_observation_video

    wrist_video_name = (
        "benchmark_wrist_observations.mp4"
        if args.scene_profile == "benchmark"
        else "minimal_observations.mp4"
    )
    overview_video_name = (
        "benchmark_overview_observations.mp4"
        if args.scene_profile == "benchmark"
        else "overview_observations.mp4"
    )
    wrist_video_result = build_observation_video(
        output_root=output_root,
        output_path=output_root / wrist_video_name,
        poses=tuple(observation_config["capture"]["poses"]),
        frame_name="rgb.png",
        purpose="wrist_rgb_headless_visual_verification",
    )
    overview_video_result = build_observation_video(
        output_root=output_root,
        output_path=output_root / overview_video_name,
        poses=tuple(observation_config["capture"]["poses"]),
        frame_name="overview_rgb.png",
        purpose="external_overview_human_debugging_and_paper_video",
        crf=quality_settings["video_crf"],
        preset=quality_settings["video_preset"],
    )
    print(f"WRIST_CAPTURE_VIDEO={wrist_video_result['output_path']}")
    print(f"OVERVIEW_CAPTURE_VIDEO={overview_video_result['output_path']}")
    record(
        "CAPTURE_VIDEOS_CREATED="
        f"wrist={wrist_video_result['output_path']} "
        f"overview={overview_video_result['output_path']} "
        f"duration_seconds={overview_video_result['duration_seconds']}"
    )

if args.headless:
    print("HEADLESS_CAPTURE_COMPLETED")
    record("HEADLESS_CAPTURE_COMPLETED")
    simulation_app.close(wait_for_replicator=False, skip_cleanup=True)
else:
    record("ENTERING_GUI_LOOP")
    while simulation_app.is_running():
        simulation_app.update()
    record("GUI_LOOP_EXITED")
    simulation_app.close()
    record("SIMULATION_APP_CLOSED")
