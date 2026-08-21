import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from audit_evaluation_readiness import check  # noqa: E402


class EvaluationReadinessTests(unittest.TestCase):
    def test_check_keeps_machine_readable_evidence(self) -> None:
        row = check("example", True, {"episodes": 48})
        self.assertEqual(
            row,
            {
                "name": "example",
                "passed": True,
                "evidence": {"episodes": 48},
            },
        )


if __name__ == "__main__":
    unittest.main()
