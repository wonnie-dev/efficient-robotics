import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from aggregate_policy_results import (  # noqa: E402
    exact_mcnemar,
    paired_bootstrap_difference,
    wilson,
)


class AggregatePolicyResultsTests(unittest.TestCase):
    def test_wilson_interval_contains_estimate(self):
        interval = wilson(8, 10)
        self.assertAlmostEqual(interval["estimate"], 0.8)
        self.assertLess(interval["lower"], interval["estimate"])
        self.assertGreater(interval["upper"], interval["estimate"])

    def test_exact_mcnemar_counts_paired_disagreements(self):
        result = exact_mcnemar(
            [True, True, True, False],
            [False, False, True, True],
        )
        self.assertEqual(result["first_only"], 2)
        self.assertEqual(result["second_only"], 1)
        self.assertEqual(result["discordant_count"], 3)

    def test_paired_bootstrap_is_deterministic(self):
        first = paired_bootstrap_difference([1.0, 2.0], [2.0, 4.0], resamples=100)
        second = paired_bootstrap_difference([1.0, 2.0], [2.0, 4.0], resamples=100)
        self.assertEqual(first, second)
        self.assertAlmostEqual(first["mean_difference_first_minus_second"], -1.5)


if __name__ == "__main__":
    unittest.main()
