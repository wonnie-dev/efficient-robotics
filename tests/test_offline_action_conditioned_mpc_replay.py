"""CPU-only tests for the cached action-conditioned MPC replay."""

import json
import math
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from run_offline_action_conditioned_mpc_replay import (  # noqa: E402
    build_episode_rows,
    confidence_bin,
    fit_action_model,
    forecast_view_action,
    initial_belief,
    replay_mpc,
    update_with_observation,
)


class OfflineActionConditionedMpcReplayTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.config_path = (
            ROOT
            / "configs"
            / "research"
            / "offline_action_conditioned_mpc_replay_seed165_184.json"
        )
        cls.config = json.loads(
            cls.config_path.read_text(encoding="utf-8")
        )
        cls.episodes = build_episode_rows(cls.config)

    def test_confidence_bins_have_explicit_boundaries(self) -> None:
        settings = self.config["identity_confidence_bins"]
        self.assertEqual(confidence_bin(0.0, settings), "very_low")
        self.assertEqual(confidence_bin(0.2, settings), "low")
        self.assertEqual(confidence_bin(0.5, settings), "medium")
        self.assertEqual(confidence_bin(0.8, settings), "high")

    def test_cached_episode_rows_cover_twenty_disjoint_seeds(self) -> None:
        self.assertEqual(len(self.episodes), 20)
        self.assertEqual(set(self.episodes), set(range(165, 185)))
        for episode in self.episodes.values():
            self.assertEqual(
                set(episode),
                {
                    "initial_observation",
                    "viewpoint_close_high",
                    "viewpoint_right",
                },
            )

    def test_leave_one_out_model_excludes_held_out_seed(self) -> None:
        held_out_seed = 165
        model = fit_action_model(
            {
                seed: episode
                for seed, episode in self.episodes.items()
                if seed != held_out_seed
            },
            self.config,
            action_agnostic=False,
        )
        self.assertNotIn(held_out_seed, model["training_seeds"])
        self.assertEqual(model["training_episode_count"], 19)
        for state_table in model["perception_observation"].values():
            for probabilities in state_table.values():
                self.assertTrue(
                    math.isclose(sum(probabilities.values()), 1.0)
                )

    def test_forecast_is_normalized_and_does_not_read_future_cache(
        self,
    ) -> None:
        held_out_seed = 165
        model = fit_action_model(
            {
                seed: episode
                for seed, episode in self.episodes.items()
                if seed != held_out_seed
            },
            self.config,
            action_agnostic=False,
        )
        center = self.episodes[held_out_seed]["initial_observation"]
        belief, _ = update_with_observation(
            initial_belief(model),
            center,
            model,
            action="initial_observation",
        )
        forecast = forecast_view_action(
            belief,
            model,
            "viewpoint_close_high",
            self.config,
        )
        self.assertFalse(
            forecast[
                "future_held_out_observation_used_for_forecast"
            ]
        )
        self.assertEqual(forecast["observation_branch_count"], 576)
        self.assertTrue(
            math.isclose(
                forecast["all_branch_probability_sum"],
                1.0,
            )
        )

    def test_mpc_replay_records_non_oracle_action_order(self) -> None:
        held_out_seed = 165
        model = fit_action_model(
            {
                seed: episode
                for seed, episode in self.episodes.items()
                if seed != held_out_seed
            },
            self.config,
            action_agnostic=False,
        )
        result = replay_mpc(
            "action_conditioned_belief_mpc",
            held_out_seed,
            self.episodes[held_out_seed],
            model,
            self.config,
        )
        self.assertFalse(
            result["simulator_ground_truth_used_during_action_selection"]
        )
        root = result["planning_details"]["root_policy"]
        self.assertFalse(
            root[
                "future_held_out_observation_used_for_action_selection"
            ]
        )
        self.assertNotIn(
            held_out_seed,
            result["planning_details"][
                "leave_one_episode_out_model_training_seeds"
            ],
        )


if __name__ == "__main__":
    unittest.main()
