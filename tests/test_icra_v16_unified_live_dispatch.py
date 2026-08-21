"""CPU checks for the V16 root-action physical dispatcher."""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from run_icra_v13_joint_live_development import (  # noqa: E402
    live_request_payload,
    posthoc_semantic_audit,
    unified_action_kind,
    unified_physical_task_success,
    update_live_semantic_belief,
)


class IcraV16UnifiedLiveDispatchTests(unittest.TestCase):
    def setUp(self) -> None:
        self.policy = {
            "selected_action": "remove_cover",
            "action_values": [],
        }
        self.belief = {
            "track_center_selected|inside": 0.7,
            "track_center_selected|outside": 0.2,
            "target_absent|not_applicable": 0.1,
        }

    @staticmethod
    def _semantic_model() -> dict:
        hypotheses = (
            "track_center_selected|inside",
            "track_center_selected|outside",
            "track_other_target|inside",
            "track_other_target|outside",
            "target_absent|not_applicable",
        )

        def action_model(target_absent_likelihood: float) -> dict:
            return {
                "outcomes": ["no_target_evidence", "unseen"],
                "likelihood": {
                    hypothesis: {
                        "no_target_evidence": (
                            target_absent_likelihood
                            if hypothesis == "target_absent|not_applicable"
                            else 0.05
                        ),
                        "unseen": 0.5,
                    }
                    for hypothesis in hypotheses
                },
            }

        return {
            "observation_model": {
                "initial_observation": action_model(0.9),
                "viewpoint_right": action_model(0.9),
                "viewpoint_close_high": action_model(0.9),
                "remove_cover": action_model(0.9),
            }
        }

    def test_non_exhaustive_view_miss_does_not_create_global_absence(self) -> None:
        prior = {
            "track_center_selected|inside": 0.2,
            "track_center_selected|outside": 0.2,
            "track_other_target|inside": 0.2,
            "track_other_target|outside": 0.2,
            "target_absent|not_applicable": 0.2,
        }
        posterior, provenance = update_live_semantic_belief(
            prior,
            self._semantic_model(),
            "viewpoint_right",
            "no_target_evidence",
        )
        self.assertEqual(posterior, prior)
        self.assertEqual(
            provenance["source"],
            "non_exhaustive_view_no_global_absence_update",
        )

    def test_verified_post_removal_miss_can_support_target_absence(self) -> None:
        prior = {
            "track_center_selected|inside": 0.2,
            "track_center_selected|outside": 0.2,
            "track_other_target|inside": 0.2,
            "track_other_target|outside": 0.2,
            "target_absent|not_applicable": 0.2,
        }
        posterior, provenance = update_live_semantic_belief(
            prior,
            self._semantic_model(),
            "remove_cover",
            "no_target_evidence",
        )
        self.assertGreater(
            posterior["target_absent|not_applicable"],
            prior["target_absent|not_applicable"],
        )
        self.assertEqual(
            provenance["source"], "action_conditioned_exact_symbol"
        )

    def test_every_root_action_has_a_physical_dispatch_kind(self) -> None:
        self.assertEqual(unified_action_kind("remove_cover"), "remove_cover")
        self.assertEqual(
            unified_action_kind("viewpoint_close_high"), "viewpoint"
        )
        self.assertEqual(unified_action_kind("viewpoint_right"), "viewpoint")
        self.assertEqual(
            unified_action_kind("grasp:track_center_selected:outside"),
            "grasp",
        )
        self.assertEqual(unified_action_kind("defer"), "defer")

    def test_remove_cover_is_forwarded_without_becoming_a_forced_root(self) -> None:
        payload = live_request_payload(
            index=2,
            joint_action="remove_cover",
            policy=self.policy,
            belief=self.belief,
        )
        self.assertEqual(payload["index"], 2)
        self.assertEqual(payload["type"], "remove_cover")
        self.assertEqual(payload["joint_action"], "remove_cover")
        self.assertTrue(payload["physical_execution_requested"])

    def test_root_grasp_uses_server_grasp_protocol_and_requires_localization(self) -> None:
        with self.assertRaisesRegex(ValueError, "requires RGB-D localization"):
            live_request_payload(
                index=0,
                joint_action="grasp:track_center_selected:outside",
                policy=self.policy,
                belief=self.belief,
            )
        payload = live_request_payload(
            index=0,
            joint_action="grasp:track_center_selected:outside",
            policy=self.policy,
            belief=self.belief,
            localization_path=Path("/tmp/localization.json"),
        )
        self.assertEqual(payload["type"], "grasp")
        self.assertEqual(
            payload["joint_action"],
            "grasp:track_center_selected:outside",
        )

    def test_defer_is_sent_to_server_without_physical_manipulation(self) -> None:
        payload = live_request_payload(
            index=1,
            joint_action="defer",
            policy=self.policy,
            belief=self.belief,
        )
        self.assertEqual(payload["type"], "defer")
        self.assertFalse(payload["physical_execution_requested"])

    def test_direct_root_grasp_does_not_require_unselected_cover_removal(self) -> None:
        result = {
            "status": "completed",
            "terminal_action": "grasp",
            "grasp_executed": True,
            "grasp_execution": {
                "lift_verified": True,
                "bilateral_contact_before_lift": True,
                "unexpected_environment_pairs": [],
                "contact_force_within_limit": True,
                "contact_penetration_within_limit": True,
            },
            "cover_removal_execution": None,
        }
        self.assertTrue(
            unified_physical_task_success(
                "grasp:track_center_selected:outside",
                ["grasp:track_center_selected:outside"],
                result,
            )
        )

    def test_selected_remove_cover_must_pass_its_contact_contract(self) -> None:
        result = {
            "status": "completed",
            "terminal_action": "grasp",
            "grasp_executed": True,
            "grasp_execution": {
                "lift_verified": True,
                "bilateral_contact_before_lift": True,
                "unexpected_environment_pairs": [],
                "contact_force_within_limit": True,
                "contact_penetration_within_limit": True,
            },
            "cover_removal_execution": {
                "removal_verified": False,
            },
        }
        self.assertFalse(
            unified_physical_task_success(
                "grasp:track_center_selected:inside",
                ["remove_cover", "grasp:track_center_selected:inside"],
                result,
            )
        )

    def test_target_absent_defer_is_a_semantic_success(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "household_scene.json").write_text(
                json.dumps(
                    {
                        "calibration_ground_truth": {
                            "world_ground_truth": {"target_exists": False},
                            "action_outcome_design": {
                                "required_action_sequence": [
                                    "remove_cover",
                                    "defer",
                                ]
                            },
                        }
                    }
                ),
                encoding="utf-8",
            )
            audit = posthoc_semantic_audit(
                root,
                "defer",
                ["remove_cover", "defer"],
                {},
            )
        self.assertTrue(audit["passed"])
        self.assertTrue(audit["safe_absent_deferral"])

    def test_live_smoke_config_no_longer_declares_remove_cover_only_scope(self) -> None:
        config = json.loads(
            (
                ROOT
                / "configs/research/icra_v16_unified_live_pretest_smoke.json"
            ).read_text(encoding="utf-8")
        )
        self.assertEqual(
            config["live_smoke_branch_scope"],
            "unified_root_action_dispatch",
        )

    def test_server_accepts_each_terminal_action_at_the_root(self) -> None:
        source = (ROOT / "scripts/open_minimal_scene.py").read_text(
            encoding="utf-8"
        )
        for action in (
            '"grasp"',
            '"grasp_inside"',
            '"grasp_outside"',
            '"remove_cover"',
            '"defer"',
        ):
            self.assertIn(action, source)
        self.assertIn(
            "args.execute_persistent_composite_grasp\n"
            "            or args.execute_persistent_remove_cover",
            source,
        )


if __name__ == "__main__":
    unittest.main()
