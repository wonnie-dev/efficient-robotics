"""Tests for causal replanning after physical remove-cover output."""

import json
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from replan_after_physical_remove_cover import (  # noqa: E402
    physical_outcome,
    run_replan,
)


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


class ReplanAfterPhysicalRemoveCoverTests(unittest.TestCase):
    def make_run(self, root: Path, post_pixels: int) -> Path:
        write_json(
            root / "server_result.json",
            {
                "seed": 188,
                "cover_removal_executed": True,
                "cover_removal_execution": {
                    "status": "completed",
                    "removal_verified": True,
                },
            },
        )
        write_json(root / "action_request_000.json", {"type": "remove_cover"})
        for view, pixels in (("center", 0), ("post_remove", post_pixels)):
            observation = root / "observations" / view
            observation.mkdir(parents=True, exist_ok=True)
            (observation / "rgb.png").touch()
            (observation / "depth_m.npy").touch()
            write_json(
                observation / "objects.json",
                {"target_red": {"pixel_count": pixels}},
            )
        return root

    def test_positive_evidence_replans_to_inside_grasp(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = self.make_run(root / "source", 8515)
            result = run_replan(source, root / "output")
            self.assertEqual(result["post_action_observation"], "target_detected")
            self.assertEqual(result["next_action"], "grasp_inside")
            self.assertTrue(result["root_action_physical_execution_verified"])
            self.assertFalse(result["valid_for_final_evaluation"])

    def test_empty_observation_is_negative_evidence(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = self.make_run(root / "source", 0)
            result = run_replan(source, root / "output")
            self.assertEqual(result["post_action_observation"], "empty_container")
            self.assertEqual(result["next_action"], "grasp_outside")
            self.assertTrue(result["negative_evidence_observed_this_episode"])
            self.assertGreater(
                result["posterior_joint_belief"]["outside_near|open"],
                result["initial_joint_belief"]["outside_near|covered"],
            )

    def test_failed_physical_action_maps_to_action_failed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            post = Path(directory) / "post"
            write_json(post / "objects.json", {"target_red": {"pixel_count": 900}})
            outcome, pixels = physical_outcome(
                {
                    "cover_removal_executed": False,
                    "cover_removal_execution": {"status": "failed"},
                },
                post,
                minimum_target_pixels=100,
            )
            self.assertEqual(outcome, "action_failed")
            self.assertEqual(pixels, 0)


if __name__ == "__main__":
    unittest.main()
