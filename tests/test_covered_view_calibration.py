from __future__ import annotations

import copy
import json
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from run_closed_loop_episode import add_covered_view_likelihoods


class CoveredViewCalibrationTests(unittest.TestCase):
    def test_supplement_does_not_overlap_reserved_splits(self) -> None:
        config = json.loads(
            (
                ROOT
                / "configs/research/covered_view_calibration_episodes.json"
            ).read_text(encoding="utf-8")
        )
        seeds = set(
            range(config["seed_start"], config["seed_start"] + config["episode_count"])
        )
        self.assertFalse(seeds & set(range(1000, 1060)))
        self.assertFalse(seeds & set(range(1100, 1160)))
        self.assertTrue(all(row["initial_task_state"] == "covered" for row in config["family_cycle"]))

    def test_covered_rows_are_added_without_replacing_open_rows(self) -> None:
        base = json.loads(
            (
                ROOT / "artifacts/calibration/calibration_candidate_model.json"
            ).read_text(encoding="utf-8")
        )
        calibration = {"schema_version": "test", "episode_count": 4, "actions": {}}
        for action_name in ("viewpoint_close_high", "viewpoint_right"):
            action = base["information_actions"][action_name]
            outcomes = list(action["outcomes"])
            row = {outcome: 1.0 / len(outcomes) for outcome in outcomes}
            calibration["actions"][action_name] = {
                "outcomes": outcomes,
                "likelihood": {
                    hypothesis: copy.deepcopy(row)
                    for hypothesis in base["semantic_hypotheses"]
                },
            }
        original_open = copy.deepcopy(
            base["information_actions"]["viewpoint_right"]["observation_likelihood"]["open"]
        )
        updated = add_covered_view_likelihoods(base, calibration)
        self.assertIn(
            "covered",
            updated["information_actions"]["viewpoint_right"]["allowed_task_states"],
        )
        self.assertEqual(
            original_open,
            updated["information_actions"]["viewpoint_right"]["observation_likelihood"]["open"],
        )


if __name__ == "__main__":
    unittest.main()
