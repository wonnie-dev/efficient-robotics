# Real-Robot Transfer

## Lab hardware currently reported

- Universal Robots UR10 arm; exact revision must be confirmed
- OnRobot RG6 gripper
- lab camera configuration not yet measured
- 3D-printed cylindrical container and matching lid

Simulation currently uses UR10e and a rectangular scanned-basket setup. The robot model, tool transform, camera transform, and container geometry must remain configurable.

## Measurements required before motion

| Item | Required information |
| --- | --- |
| Robot | exact model and revision, base frame, controller interface, joint limits |
| RG6 | serial or asset ID, fingertip type and dimensions, opening range, force command behavior |
| Camera | model, intrinsics, depth scale, frame rate, image encoding, hand-eye transform |
| Table | dimensions, height, robot-base transform, keep-out region |
| Container | mesh or dimensions, mass, pose convention, interior clearance |
| Lid | mass, dimensions, inertia estimate, friction, handle pose and dimensions |
| Control | command rate, trajectory interface, emergency stop, force and speed limits |

Enter measured values in `configs/hardware/rg6_lid_transfer_calibration.json`. Validate the worksheet without commanding hardware:

```bash
bash scripts/validate_real_robot_configuration.sh
```

## Interface boundary

The research code does not require ROS 2 internally. The lab adapter may use ROS 2, RTDE, a UR controller API, or another supported interface. It must implement the same boundary.

Input:

```text
RGB image
metric depth image or point cloud
camera intrinsics
camera-to-robot transform
joint state
gripper state
timestamp and frame IDs
```

Output:

```text
semantic action request
target or reference object ID
desired camera or grasp pose
preconditions and safety limits
```

The hardware adapter must return measured execution status and a new observation. The planner must not infer success from a command acknowledgment.

## Transfer sequence

1. Validate configuration and frame transforms with no robot motion.
2. Replay recorded lab RGB-D through perception and the planner.
3. Verify camera-only motion at reduced speed with no objects in reach.
4. Verify empty-table RG6 commands and force limits.
5. Run isolated lid-handle contact trials and record at least five trials.
6. Run isolated target grasps.
7. Run one supervised closed-loop episode with emergency-stop access.
8. Expand only after failure logs and safety gates pass.

## Transfer constraints

No real-robot motion has been executed from this repository. The supplied transfer calibration contains placeholders or public-spec proxies and is not transfer-ready.
