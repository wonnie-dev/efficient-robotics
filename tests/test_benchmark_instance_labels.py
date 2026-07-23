"""Tests for benchmark color-ID component disambiguation."""

import sys
import unittest
from pathlib import Path

import numpy as np


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from observation_capture import _split_id_pair_by_horizontal_components  # noqa: E402


class BenchmarkInstanceLabelTests(unittest.TestCase):
    def test_disconnected_pair_is_assigned_left_and_right(self) -> None:
        ids = np.zeros((20, 40), dtype=np.uint32)
        ids[2:12, 3:10] = 2
        ids[4:16, 25:34] = 3
        _split_id_pair_by_horizontal_components(
            ids, pair=(2, 3), left_id=3, right_id=2
        )
        self.assertTrue(np.all(ids[2:12, 3:10] == 3))
        self.assertTrue(np.all(ids[4:16, 25:34] == 2))


if __name__ == "__main__":
    unittest.main()
