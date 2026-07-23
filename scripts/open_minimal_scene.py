from pathlib import Path

from isaacsim import SimulationApp


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCENE_PATH = PROJECT_ROOT / "assets" / "scenes" / "open_container_minimal.usda"
RUNTIME_LOG = PROJECT_ROOT / "runtime_scene_status.log"


def record(message: str) -> None:
    with RUNTIME_LOG.open("a", encoding="utf-8") as stream:
        stream.write(message + "\n")


RUNTIME_LOG.write_text("START\n", encoding="utf-8")

simulation_app = SimulationApp(
    {
        "headless": False,
        "active_gpu": 0,
        "physics_gpu": 0,
        "multi_gpu": False,
        "renderer": "RaytracedLighting",
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
from pxr import Gf, Sdf, Usd, UsdGeom
from isaacsim.core.experimental.prims import Articulation, GeomPrim
from isaacsim.core.simulation_manager import SimulationManager

from observation_capture import (
    add_scene_labels,
    align_tool_and_camera,
    configure_camera,
    create_capture_pipeline,
    load_observation_config,
    make_gripper_kinematic,
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

assets_root = get_assets_root_path()
if assets_root is None:
    raise RuntimeError("Isaac Sim production asset root is unavailable")
ur10e_asset = assets_root + "/Isaac/Robots/UniversalRobots/ur10e/ur10e.usd"
add_reference_to_stage(ur10e_asset, "/World/RobotSystem/UR10e")
record("UR10E_ASSET=" + ur10e_asset)
for _ in range(30):
    simulation_app.update()
camera_prim = stage.GetPrimAtPath("/World/ObservationCamera")
if not camera_prim.IsValid():
    simulation_app.close()
    raise RuntimeError("Observation camera is missing from the stage")

ur10e_ee = None
for candidate_path in (
    "/World/RobotSystem/UR10e/ee_link",
    "/World/RobotSystem/UR10e/wrist_3_link/flange",
    "/World/RobotSystem/UR10e/tool0",
):
    candidate = stage.GetPrimAtPath(candidate_path)
    if candidate.IsValid():
        ur10e_ee = candidate
        record("UR10E_EE=" + candidate_path)
        break
rg6_prim = stage.GetPrimAtPath("/World/RobotSystem/RG6")
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
zivid_camera = stage.GetPrimAtPath("/World/RobotSystem/RG6/Zivid2Camera")
if not zivid_camera.IsValid():
    raise RuntimeError("Zivid 2 wrist camera prim is missing")

observation_config = load_observation_config(PROJECT_ROOT)
add_scene_labels(stage)
for _ in range(10):
    simulation_app.update()
record("SEMANTIC_LABELS_ADDED=target_red|distractor_blue|container")
configure_camera(zivid_camera, observation_config)
make_gripper_kinematic(rg6_prim)
record("RG6_KINEMATIC_FOR_PROVISIONAL_MOUNT")
resolution = tuple(observation_config["camera"]["resolution"])
rep, render_product, rgb_annotator, depth_annotator = create_capture_pipeline(
    "/World/RobotSystem/RG6/Zivid2Camera", resolution
)
output_root = PROJECT_ROOT / observation_config["capture"]["output_directory"]
record("CAPTURE_PIPELINE_READY=rgb|depth|rgb_color_key_instance_fallback")
SimulationManager.setup_simulation(dt=1.0 / 60.0)
robot = Articulation("/World/RobotSystem/UR10e")
simulation_app.update()
record("UR10E_DOFS=" + "|".join(robot.dof_names))
app_utils.play()
simulation_app.update()
record("PHYSICS_PLAYING")
record("UR10E_LINKS=" + "|".join(robot.link_names))
physical_ee_name = next(
    name for name in ("ee_link", "wrist_3_link", "tool0") if name in robot.link_names
)
ee_link_index = robot.link_names.index(physical_ee_name)
ee_geom = GeomPrim(paths=robot.link_paths[0][ee_link_index])
record("PHYSX_EE_LINK=" + physical_ee_name)


def current_ee_pose():
    positions, orientations = ee_geom.get_world_poses()
    return positions.numpy()[0], orientations.numpy()[0]

set_pose(robot, observation_config, "home", simulation_app.update, 1)
app_utils.pause()
simulation_app.update()
align_tool_and_camera(stage, ur10e_ee, rg6_prim, zivid_camera, observation_config, current_ee_pose())
record("HOME_POSE_APPLIED")

for pose_name in observation_config["capture"]["poses"]:
    app_utils.play()
    simulation_app.update()
    set_pose(
        robot,
        observation_config,
        pose_name,
        simulation_app.update,
        1,
    )
    app_utils.pause()
    simulation_app.update()
    pose_position, pose_orientation = current_ee_pose()
    record(f"EE_POSE {pose_name} position={pose_position.tolist()}")
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
        pose_name,
        rgb_annotator.get_data(),
        depth_annotator.get_data(),
    )
    record("CAPTURED_POSE=" + pose_name)

app_utils.play()
simulation_app.update()
set_pose(robot, observation_config, "center", simulation_app.update, 1)
app_utils.pause()
simulation_app.update()
align_tool_and_camera(stage, ur10e_ee, rg6_prim, zivid_camera, observation_config, current_ee_pose())
record("RG6_AND_CAMERA_ALIGNED_TO_EE")

bbox_cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_])
for bbox_path in (
    "/World/RobotSystem/UR10e",
    "/World/RobotSystem/RG6",
    "/World/Ground",
    "/World/Table",
    "/World/OpenContainer",
    "/World/TargetRed",
    "/World/DistractorBlue",
):
    bbox_prim = stage.GetPrimAtPath(bbox_path)
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
            "/World/Table",
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
print("ACTIVE_CAMERA=/OmniverseKit_Persp")
print("ROBOT_STACK=UR10e+OnRobot_RG6+Zivid2_provisional")
record("ENTERING_GUI_LOOP")

while simulation_app.is_running():
    simulation_app.update()

record("GUI_LOOP_EXITED")
simulation_app.close()
record("SIMULATION_APP_CLOSED")
