import json
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from run_offline_action_conditioned_mpc_replay import (  # noqa: E402
    build_episode_rows,
    fit_action_model,
)
from run_offline_full_baseline_ablation import (  # noqa: E402
    configure_relation_likelihood,
    configure_relation_observations,
)


class RelationFactorMpcAblationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.config = json.loads(
            (
                ROOT
                / "configs"
                / "research"
                / "offline_action_conditioned_mpc_replay_seed165_184.json"
            ).read_text(encoding="utf-8")
        )
        cls.episodes = build_episode_rows(cls.config)

    def test_membership_only_observation_preserves_membership(self) -> None:
        configured = configure_relation_observations(
            self.episodes,
            membership_enabled=True,
            occlusion_enabled=False,
        )
        for seed, episode in configured.items():
            for action, row in episode.items():
                original = self.episodes[seed][action]
                self.assertEqual(
                    row["membership_observation"],
                    original["membership_observation"],
                )
                self.assertEqual(
                    row["reference_occlusion_observation"], "missing"
                )

    def test_occlusion_only_likelihood_neutralizes_membership(self) -> None:
        model = fit_action_model(
            self.episodes, self.config, action_agnostic=False
        )
        configured = configure_relation_likelihood(
            model,
            membership_enabled=False,
            occlusion_enabled=True,
        )
        for action in configured["membership_observation"].values():
            for distribution in action.values():
                self.assertEqual(distribution["missing"], 1.0)
        self.assertTrue(
            any(
                distribution["high|yes"] > 0.0
                for action in configured["perception_observation"].values()
                for distribution in action.values()
            )
        )


if __name__ == "__main__":
    unittest.main()
