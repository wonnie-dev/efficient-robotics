"""CPU tests for the guarded remove-cover primitive compiler."""

import copy
import json
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from plan_remove_cover_primitive import (  # noqa: E402
    compile_plan,
    load_json,
    resolve_path,
)


class RemoveCoverPrimitiveTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.config = load_json(
            ROOT
            / "configs"
            / "research"
            / "remove_cover_primitive_cpu_contract.json"
        )
        cls.request = load_json(
            resolve_path(cls.config["source_action_request"])
        )
        cls.graph = load_json(
            resolve_path(cls.config["source_scene_graph"])
        )

    def test_valid_request_compiles_eight_guarded_phases(self) -> None:
        plan = compile_plan(self.config, self.request, self.graph)
        self.assertEqual(
            plan["status"], "ready_for_live_ik_and_physics_validation"
        )
        self.assertEqual(len(plan["phases"]), 8)
        self.assertTrue(
            plan["geometry_validation"]["static_geometry_gate_passed"]
        )
        self.assertFalse(plan["physical_execution_authorized"])
        self.assertFalse(plan["gpu_used"])

    def test_request_hash_mismatch_is_rejected(self) -> None:
        request = copy.deepcopy(self.request)
        request["source_scene_graph_sha256"] = "0" * 64
        with self.assertRaisesRegex(ValueError, "source graph"):
            compile_plan(self.config, request, self.graph)

    def test_non_remove_cover_action_is_rejected(self) -> None:
        request = copy.deepcopy(self.request)
        request["type"] = "viewpoint_right"
        with self.assertRaisesRegex(ValueError, "remove_cover"):
            compile_plan(self.config, request, self.graph)

    def test_static_or_handleless_cover_is_rejected(self) -> None:
        static = copy.deepcopy(self.config)
        static["cover"]["rigid_body"] = False
        with self.assertRaisesRegex(ValueError, "dynamic rigid body"):
            compile_plan(static, self.request, self.graph)
        handleless = copy.deepcopy(self.config)
        handleless["cover"]["handle"] = None
        with self.assertRaisesRegex(ValueError, "no RG6 grasp handle"):
            compile_plan(handleless, self.request, self.graph)

    def test_staging_collision_is_rejected(self) -> None:
        colliding = copy.deepcopy(self.config)
        colliding["primitive"]["staging_cover_center_local_m"] = [
            0.0,
            -0.1,
            0.007,
        ]
        with self.assertRaisesRegex(ValueError, "clear the basket"):
            compile_plan(colliding, self.request, self.graph)

    def test_live_gates_block_lift_without_bilateral_contact(self) -> None:
        plan = compile_plan(self.config, self.request, self.graph)
        lift = next(
            phase
            for phase in plan["phases"]
            if phase["phase"] == "vertical_cover_lift"
        )
        self.assertIn("bilateral_handle_contact", lift["required_gates"])
        self.assertIn(
            "lost_bilateral_contact",
            plan["failure_to_observation_mapping"],
        )


if __name__ == "__main__":
    unittest.main()
