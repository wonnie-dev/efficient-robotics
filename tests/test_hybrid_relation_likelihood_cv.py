import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from run_hybrid_relation_likelihood_cv import (  # noqa: E402
    decision_metric_summary,
    fit_likelihood,
    load_configured_rows,
    missing_true_labels,
    posterior_from_likelihood,
    uniform_posterior,
    validate_folds,
)


class HybridRelationLikelihoodCvTest(unittest.TestCase):
    def test_likelihood_is_normalized_for_each_true_state(self):
        examples = [
            {"truth": "inside", "observation": "inside"},
            {"truth": "inside", "observation": "unknown"},
            {"truth": "outside", "observation": "outside"},
        ]
        likelihood = fit_likelihood(
            examples,
            ["inside", "outside"],
            ["inside", "outside", "unknown"],
            pseudocount=1.0,
        )
        for distribution in likelihood.values():
            self.assertAlmostEqual(sum(distribution.values()), 1.0)

    def test_posterior_uses_observation_likelihood(self):
        likelihood = {
            "yes": {"yes": 0.8, "no": 0.2},
            "no": {"yes": 0.1, "no": 0.9},
        }
        posterior = posterior_from_likelihood(
            "yes", likelihood, ["yes", "no"]
        )
        self.assertGreater(posterior["yes"], posterior["no"])
        self.assertAlmostEqual(sum(posterior.values()), 1.0)

    def test_uniform_posterior_is_normalized(self):
        posterior = uniform_posterior(
            ["inside", "outside"]
        )
        self.assertEqual(posterior, {"inside": 0.5, "outside": 0.5})

    def test_hard_unknown_is_counted_as_abstention(self):
        metrics = decision_metric_summary(
            [{"truth": "inside", "observation": "unknown"}],
            ["inside", "outside"],
            ["inside", "outside", "unknown"],
        )
        self.assertEqual(metrics["coverage"], 0.0)
        self.assertEqual(
            metrics["accuracy_with_abstentions_counted_as_incorrect"], 0.0
        )

    def test_fold_validation_rejects_duplicates(self):
        with self.assertRaises(ValueError):
            validate_folds(
                [
                    {"fold_id": 0, "held_out_seeds": [1]},
                    {"fold_id": 1, "held_out_seeds": [1]},
                ],
                {1},
            )

    def test_missing_truth_labels_are_reported(self):
        examples = [
            {"truth": "yes", "observation": "yes"},
            {"truth": "no", "observation": "no"},
        ]
        self.assertEqual(
            missing_true_labels(examples, ["yes", "no", "unknown"]),
            ["unknown"],
        )

    def test_multiple_audit_csvs_are_combined(self):
        with tempfile.TemporaryDirectory() as directory:
            first = Path(directory) / "first.csv"
            second = Path(directory) / "second.csv"
            first.write_text("seed,sample_id\n1,a\n", encoding="utf-8")
            second.write_text("seed,sample_id\n2,b\n", encoding="utf-8")
            paths, rows = load_configured_rows(
                {"audit_rows_csvs": [str(first), str(second)]}
            )
            self.assertEqual(paths, [first, second])
            self.assertEqual([row["seed"] for row in rows], ["1", "2"])


if __name__ == "__main__":
    unittest.main()
