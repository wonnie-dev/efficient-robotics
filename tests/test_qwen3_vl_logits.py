"""CPU-only tests for Qwen3-VL prompt and path-independent scoring setup."""

import os
import sys
import unittest
from pathlib import Path
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from qwen3_vl_logits import (  # noqa: E402
    cyclic_mappings,
    joint_candidate_question,
    letter_mapping,
    relation_question,
    require_single_gpu_only,
    target_question,
)


class Qwen3VlLogitTests(unittest.TestCase):
    def test_target_mapping_preserves_candidate_order(self) -> None:
        model_input = {
            "instruction": "Retrieve the red object in the container.",
            "candidates": [
                {"candidate_id": "object_003"},
                {"candidate_id": "object_001"},
            ],
        }
        prompt, mapping = target_question(model_input)
        self.assertEqual(mapping, [("A", "object_003"), ("B", "object_001")])
        self.assertIn("Robot instruction:", prompt)

    def test_relation_mapping_preserves_label_order(self) -> None:
        query = {
            "source_id": "object_001",
            "target_id": "container_001",
            "label_space": ["inside", "outside", "unknown"],
        }
        prompt, mapping = relation_question(query)
        self.assertEqual(
            mapping,
            [("A", "inside"), ("B", "outside"), ("C", "unknown")],
        )
        self.assertIn("basket/container", prompt)
        self.assertIn("INNER walls", prompt)
        self.assertIn("resting on the container floor is inside", prompt)

    def test_factorized_relation_prompt_does_not_conflate_labels(self) -> None:
        membership_prompt, _ = relation_question(
            {
                "source_id": "candidate_001",
                "target_id": "container_001",
                "relation_type": "membership",
                "label_space": ["inside", "outside", "unknown"],
            }
        )
        behind_prompt, mapping = relation_question(
            {
                "source_id": "candidate_001",
                "target_id": "container_001",
                "relation_type": "behind",
                "label_space": ["yes", "no", "unknown"],
            }
        )
        self.assertIn("Do not encode behind or occlusion", membership_prompt)
        self.assertIn("Relation factor: behind", behind_prompt)
        self.assertEqual(
            mapping,
            [("A", "yes"), ("B", "no"), ("C", "unknown")],
        )

    def test_joint_candidate_requires_identity_and_inside_relation(self) -> None:
        model_input = {
            "instruction": "Retrieve the red object in the container.",
            "reference_entities": [
                {
                    "reference_id": "container_001",
                    "bbox_xyxy": [10, 20, 100, 120],
                }
            ],
        }
        prompt, mapping = joint_candidate_question(
            model_input, "object_001"
        )
        self.assertEqual(
            mapping,
            [
                ("A", "matches_instruction"),
                ("B", "does_not_match"),
            ],
        )
        self.assertIn("BOTH requirements", prompt)
        self.assertIn("object_001", prompt)

    def test_choice_mapping_rejects_single_class(self) -> None:
        with self.assertRaises(ValueError):
            letter_mapping(["only"])

    def test_cyclic_mapping_assigns_every_letter_to_every_value(self) -> None:
        mappings = cyclic_mappings(
            [("A", "inside"), ("B", "outside"), ("C", "unknown")]
        )
        assignments = {
            value: {letter for mapping in mappings for letter, item in mapping if item == value}
            for value in ("inside", "outside", "unknown")
        }
        self.assertEqual(
            assignments,
            {
                "inside": {"A", "B", "C"},
                "outside": {"A", "B", "C"},
                "unknown": {"A", "B", "C"},
            },
        )

    def test_gpu_guard_accepts_one_visible_gpu(self) -> None:
        with mock.patch.dict(
            os.environ,
            {"CUDA_VISIBLE_DEVICES": "5"},
            clear=True,
        ):
            require_single_gpu_only()
        with mock.patch.dict(
            os.environ,
            {"CUDA_VISIBLE_DEVICES": "0"},
            clear=True,
        ):
            require_single_gpu_only()

    def test_gpu_guard_rejects_physical_visible_mismatch(self) -> None:
        with mock.patch.dict(
            os.environ,
            {"PHYSICAL_GPU": "4", "CUDA_VISIBLE_DEVICES": "4"},
            clear=True,
        ):
            require_single_gpu_only()
        with mock.patch.dict(
            os.environ,
            {"PHYSICAL_GPU": "4", "CUDA_VISIBLE_DEVICES": "5"},
            clear=True,
        ):
            with self.assertRaises(RuntimeError):
                require_single_gpu_only()


if __name__ == "__main__":
    unittest.main()
