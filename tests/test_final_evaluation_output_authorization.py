import unittest
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from final_evaluation_authorization import validate_output_authorization  # noqa: E402


class FinalEvaluationOutputAuthorizationTests(unittest.TestCase):
    def test_development_capture_rejects_final_output_root(self):
        path = ROOT / "outputs" / "final_evaluation" / "icra_protocol_v1" / "x"
        with self.assertRaises(ValueError):
            validate_output_authorization(
                seed=197,
                output_dir=path.resolve(),
                final_evaluation_authorized=False,
            )

    def test_final_capture_rejects_nonreserved_seed(self):
        path = ROOT / "outputs" / "final_evaluation" / "icra_protocol_v1" / "x"
        with self.assertRaises(ValueError):
            validate_output_authorization(
                seed=197,
                output_dir=path.resolve(),
                final_evaluation_authorized=True,
            )

    def test_final_capture_accepts_reserved_seed_under_final_root(self):
        path = ROOT / "outputs" / "final_evaluation" / "icra_protocol_v1" / "x"
        allowed = validate_output_authorization(
            seed=200,
            output_dir=path.resolve(),
            final_evaluation_authorized=True,
        )
        self.assertEqual(
            allowed,
            (ROOT / "outputs" / "final_evaluation" / "icra_protocol_v1").resolve(),
        )


if __name__ == "__main__":
    unittest.main()
