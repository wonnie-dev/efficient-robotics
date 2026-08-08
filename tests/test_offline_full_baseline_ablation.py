import json
import math
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
    make_relation_likelihood_uninformative,
    remove_hybrid_relation_observations,
    risk_neutral_config,
)


class OfflineFullBaselineAblationTests(unittest.TestCase):
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

    def test_relation_ablation_replaces_runtime_observations(self) -> None:
        ablated = remove_hybrid_relation_observations(self.episodes)
        for episode in ablated.values():
            for row in episode.values():
                self.assertEqual(row["membership_observation"], "missing")
                self.assertEqual(
                    row["reference_occlusion_observation"], "missing"
                )
                self.assertTrue(
                    row["perception_observation"].endswith("|missing")
                )

    def test_uninformative_membership_likelihood_does_not_update(self) -> None:
        model = fit_action_model(
            self.episodes, self.config, action_agnostic=False
        )
        ablated = make_relation_likelihood_uninformative(model)
        for action in ablated["membership_observation"].values():
            distributions = list(action.values())
            self.assertTrue(
                all(distribution == distributions[0] for distribution in distributions)
            )
            self.assertEqual(distributions[0]["missing"], 1.0)

    def test_identity_only_perception_likelihood_is_normalized(self) -> None:
        model = fit_action_model(
            self.episodes, self.config, action_agnostic=False
        )
        ablated = make_relation_likelihood_uninformative(model)
        for action in ablated["perception_observation"].values():
            for distribution in action.values():
                self.assertTrue(
                    math.isclose(sum(distribution.values()), 1.0)
                )
                self.assertTrue(
                    all(
                        probability == 0.0
                        for outcome, probability in distribution.items()
                        if not outcome.endswith("|missing")
                    )
                )

    def test_risk_neutral_ablation_disables_risk_terms_and_gates(self) -> None:
        selected = risk_neutral_config(self.config)
        self.assertEqual(selected["task_cost"]["wrong_target_weight"], 0.0)
        self.assertEqual(
            selected["commitment_gate"][
                "minimum_selected_target_correct_probability"
            ],
            0.0,
        )
        self.assertEqual(
            selected["commitment_gate"][
                "maximum_reference_occlusion_probability"
            ],
            1.0,
        )


if __name__ == "__main__":
    unittest.main()
