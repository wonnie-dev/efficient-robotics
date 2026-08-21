import json
import copy
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from audit_icra_evaluation_protocol import audit  # noqa: E402


class IcraEvaluationProtocolTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = json.loads(
            (
                ROOT
                / "configs"
                / "research"
                / "icra_simulation_evaluation_protocol_v1.json"
            ).read_text(encoding="utf-8")
        )

    def test_protocol_is_structurally_complete_and_frozen(self):
        result = audit(self.config)
        self.assertEqual(result["structural_failures"], [])
        self.assertEqual(result["status"], "passed")
        self.assertTrue(result["reserved_test_launch_authorized"])

    def test_protocol_declares_paper_scale(self):
        result = audit(self.config)
        self.assertEqual(result["planned_policy_evaluation_count"], 260)
        self.assertEqual(result["scenario_family_count"], 2)
        self.assertEqual(result["supplemental_physics_calibration_seed_count"], 2)

    def test_audit_rejects_supplemental_calibration_test_overlap(self):
        config = copy.deepcopy(self.config)
        config["data_split"]["supplemental_physics_calibration_seeds"] = [200]
        result = audit(config)
        self.assertIn(
            "supplemental_calibration_and_test_seed_overlap",
            result["structural_failures"],
        )

    def test_relation_answer_is_not_in_instruction(self):
        instruction = self.config["instruction_protocol"]["template"].lower()
        for relation in ("inside", "outside", "behind", "near", "covered"):
            self.assertNotIn(relation, instruction)

    def test_audit_rejects_frozen_flag_without_matching_artifact(self):
        config = copy.deepcopy(self.config)
        frozen = json.loads(
            (
                ROOT
                / "configs"
                / "research"
                / "icra_frozen_parameters_v1.json"
            ).read_text(encoding="utf-8")
        )
        frozen["relation_calibration"]["frozen"] = False
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "invalid_frozen.json"
            path.write_text(json.dumps(frozen), encoding="utf-8")
            config["calibration_freeze_requirements"][
                "frozen_parameters_path"
            ] = str(path)
            result = audit(config)
            self.assertIn(
                "relation_likelihoods_frozen_artifact_invalid",
                result["unresolved_freeze_requirements"],
            )

    def test_audit_rejects_relation_source_hash_mismatch(self):
        config = copy.deepcopy(self.config)
        frozen = json.loads(
            (
                ROOT
                / "configs"
                / "research"
                / "icra_frozen_parameters_v1.json"
            ).read_text(encoding="utf-8")
        )
        frozen["relation_calibration"]["source_sha256"] = "invalid"
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "invalid_hash.json"
            path.write_text(json.dumps(frozen), encoding="utf-8")
            config["calibration_freeze_requirements"][
                "frozen_parameters_path"
            ] = str(path)
            result = audit(config)
            self.assertIn(
                "relation_likelihoods_frozen_artifact_invalid",
                result["unresolved_freeze_requirements"],
            )

    def test_audit_rejects_action_model_source_hash_mismatch(self):
        config = copy.deepcopy(self.config)
        frozen = json.loads(
            (
                ROOT
                / "configs"
                / "research"
                / "icra_frozen_parameters_v1.json"
            ).read_text(encoding="utf-8")
        )
        frozen["action_conditioned_observation_model"][
            "joint_freeze_source_sha256"
        ] = "invalid"
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "invalid_action_hash.json"
            path.write_text(json.dumps(frozen), encoding="utf-8")
            config["calibration_freeze_requirements"][
                "frozen_parameters_path"
            ] = str(path)
            result = audit(config)
            self.assertIn(
                "action_conditioned_observation_models_frozen_artifact_invalid",
                result["unresolved_freeze_requirements"],
            )

    def test_audit_rejects_task_cost_that_differs_from_nested_selection(self):
        config = copy.deepcopy(self.config)
        frozen = json.loads(
            (
                ROOT
                / "configs"
                / "research"
                / "icra_frozen_parameters_v1.json"
            ).read_text(encoding="utf-8")
        )
        frozen["task_cost_and_commitment_gate"]["noncompletion_cost"] = 9.0
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "invalid_task_cost.json"
            path.write_text(json.dumps(frozen), encoding="utf-8")
            config["calibration_freeze_requirements"][
                "frozen_parameters_path"
            ] = str(path)
            result = audit(config)
            self.assertIn(
                "task_cost_or_commitment_gate_artifact_invalid",
                result["unresolved_freeze_requirements"],
            )


if __name__ == "__main__":
    unittest.main()
