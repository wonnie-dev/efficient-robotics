# UR10e asset provenance

The visual and collision meshes in this directory come from the official
Universal Robots ROS 2 description repository:

- repository: `UniversalRobots/Universal_Robots_ROS2_Description`
- release tag: `4.3.1`
- commit: `ae333289875f9ba5a9ea6649a54036efb5ccabee`
- source subdirectory: `meshes/ur10e`
- retrieved: `2026-07-25`

The flattened `ur10e_lula_source.urdf` is the UR10e model distributed with the
installed Isaac Sim 6.0.1 Lula motion-generation configuration. Its kinematic
parameters match the project motion solver. The composite builder removes ROS
control-only elements and rewrites mesh references to the official local
meshes above.

The upstream BSD-3-Clause license is preserved as
`LICENSE-BSD-3-Clause`.
