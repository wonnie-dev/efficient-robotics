import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from fit_action_conditioned_observation_likelihood import (  # noqa: E402
    build_transition_rows,
    fit_categorical_table,
)


class ActionConditionedObservationLikelihoodTest(unittest.TestCase):
    def test_dirichlet_probabilities_sum_to_one(self):
        rows = [
            {"action": "right", "latent": "yes", "outcome": "yes"},
            {"action": "right", "latent": "yes", "outcome": "unknown"},
        ]
        result = fit_categorical_table(
            rows,
            condition_keys=("action", "latent"),
            outcome_key="outcome",
            outcomes=("yes", "no", "unknown"),
            alpha=1.0,
            minimum_cell_count=5,
        )
        cell = result["cells"][0]
        self.assertAlmostEqual(
            sum(cell["dirichlet_smoothed_probabilities"].values()),
            1.0,
        )
        self.assertEqual(cell["counts"]["yes"], 1)
        self.assertEqual(cell["counts"]["no"], 0)
        self.assertTrue(cell["sparse"])

    def test_rejects_unknown_outcome(self):
        with self.assertRaises(ValueError):
            fit_categorical_table(
                [{"action": "right", "outcome": "invalid"}],
                condition_keys=("action",),
                outcome_key="outcome",
                outcomes=("yes", "no"),
                alpha=1.0,
                minimum_cell_count=1,
            )

    def test_transition_rows_pair_actions_with_center(self):
        rows = [
            {
                "seed": "1",
                "action": "initial_observation",
                "reference_occlusion_state": "yes",
                "total_visibility_state": "visible",
            },
            {
                "seed": "1",
                "action": "viewpoint_close_high",
                "reference_occlusion_state": "no",
                "total_visibility_state": "visible",
            },
            {
                "seed": "1",
                "action": "viewpoint_right",
                "reference_occlusion_state": "unknown",
                "total_visibility_state": "visible",
            },
        ]
        transitions = build_transition_rows(rows)
        self.assertEqual(len(transitions), 2)
        self.assertEqual(
            transitions[0]["current_reference_occlusion_state"], "yes"
        )
        self.assertEqual(
            transitions[0]["next_reference_occlusion_state"], "no"
        )


if __name__ == "__main__":
    unittest.main()
