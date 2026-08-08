"""CPU-only tests for nested belief-MPC calibration."""

import json
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from run_nested_action_conditioned_mpc_calibration import (  # noqa: E402
    action_difference_audit,
    load_fold_target_temperatures,
    tune_noncompletion_cost,
    validate_outer_folds,
)
from run_offline_action_conditioned_mpc_replay import (  # noqa: E402
    build_episode_rows,
)


class NestedActionConditionedMpcCalibrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.nested_config = json.loads(
            (
                ROOT
                / "configs"
                / "research"
                / "nested_action_conditioned_mpc_calibration_seed165_184.json"
            ).read_text(encoding="utf-8")
        )
        cls.base_config = json.loads(
            (
                ROOT
                / cls.nested_config["base_replay_config"]
            ).read_text(encoding="utf-8")
        )
        cls.episodes = build_episode_rows(cls.base_config)

    def test_outer_folds_are_disjoint_and_cover_all_episodes(self) -> None:
        validate_outer_folds(
            self.episodes,
            self.nested_config["outer_folds"],
        )
        held_out = [
            seed
            for fold in self.nested_config["outer_folds"]
            for seed in fold["held_out_seeds"]
        ]
        self.assertEqual(len(held_out), 20)
        self.assertEqual(len(set(held_out)), 20)

    def test_inner_tuning_excludes_validation_seed(self) -> None:
        outer_held_out = set(
            self.nested_config["outer_folds"][0]["held_out_seeds"]
        )
        training = {
            seed: episode
            for seed, episode in self.episodes.items()
            if seed not in outer_held_out
        }
        result = tune_noncompletion_cost(
            training,
            self.base_config,
            self.nested_config,
            action_agnostic=False,
        )
        self.assertNotIn(
            result["selected_task_noncompletion_cost"],
            [],
        )
        for candidate in result["candidate_results"]:
            for row in candidate["validation_rows"]:
                self.assertFalse(
                    row["inner_validation_seed_used_for_fit"]
                )
                self.assertNotIn(
                    row["inner_validation_seed"],
                    row["inner_fit_seeds"],
                )
                self.assertTrue(
                    set(row["inner_fit_seeds"]).isdisjoint(
                        outer_held_out
                    )
                )

    def test_selected_cost_comes_from_declared_grid(self) -> None:
        outer_held_out = set(
            self.nested_config["outer_folds"][1]["held_out_seeds"]
        )
        training = {
            seed: episode
            for seed, episode in self.episodes.items()
            if seed not in outer_held_out
        }
        result = tune_noncompletion_cost(
            training,
            self.base_config,
            self.nested_config,
            action_agnostic=True,
        )
        self.assertIn(
            result["selected_task_noncompletion_cost"],
            self.nested_config["candidate_task_noncompletion_costs"],
        )
        self.assertFalse(
            result["outer_held_out_data_used_for_selection"]
        )

    def test_action_difference_audit_counts_all_episodes(self) -> None:
        audit = action_difference_audit(
            self.episodes,
            self.nested_config,
        )
        self.assertEqual(audit["episode_count"], 20)
        self.assertGreater(
            audit["learned_observation_signature_difference_count"],
            audit["posthoc_latent_state_difference_count"],
        )
        self.assertEqual(
            sum(audit["tracked_target_preference_counts"].values()),
            20,
        )

    def test_fold_target_temperatures_exclude_matching_held_out_seeds(
        self,
    ) -> None:
        calibrated_config = json.loads(
            (
                ROOT
                / "configs"
                / "research"
                / "nested_calibrated_perception_mpc_seed165_184.json"
            ).read_text(encoding="utf-8")
        )
        temperatures = load_fold_target_temperatures(calibrated_config)
        self.assertEqual(
            temperatures,
            {0: 5.025, 1: 5.875, 2: 6.25, 3: 6.125},
        )

    def test_episode_rows_record_temperature_override(self) -> None:
        episodes = build_episode_rows(
            self.base_config,
            target_temperature=5.025,
        )
        for episode in episodes.values():
            for row in episode.values():
                self.assertEqual(row["target_temperature_applied"], 5.025)


if __name__ == "__main__":
    unittest.main()
