import copy
import json
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from final_evaluation_authorization import validate_output_authorization  # noqa: E402


class FinalEvaluationOutputAuthorizationTests(unittest.TestCase):
    def frozen_protocol(self, directory: str) -> Path:
        protocol = json.loads(
            (ROOT / "configs/research/final_evaluation_protocol.json").read_text()
        )
        protocol["status"] = "frozen_before_untouched_test"
        protocol["reserved_test_launch_authorized"] = True
        path = Path(directory) / "protocol.json"
        path.write_text(json.dumps(protocol), encoding="utf-8")
        return path

    def test_development_capture_rejects_final_output_root(self) -> None:
        path = ROOT / "outputs/final_evaluation/reserved_test/x"
        with self.assertRaises(ValueError):
            validate_output_authorization(
                seed=1099,
                output_dir=path.resolve(),
                final_evaluation_authorized=False,
            )

    def test_final_capture_checks_seed_and_output_root(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            protocol = self.frozen_protocol(directory)
            allowed_path = ROOT / "outputs/final_evaluation/reserved_test/x"
            with self.assertRaises(ValueError):
                validate_output_authorization(
                    seed=1099,
                    output_dir=allowed_path.resolve(),
                    final_evaluation_authorized=True,
                    protocol_path=protocol,
                )
            allowed = validate_output_authorization(
                seed=1100,
                output_dir=allowed_path.resolve(),
                final_evaluation_authorized=True,
                protocol_path=protocol,
            )
            self.assertEqual(
                allowed,
                (ROOT / "outputs/final_evaluation/reserved_test").resolve(),
            )


if __name__ == "__main__":
    unittest.main()
