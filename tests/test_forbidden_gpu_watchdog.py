import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from run_with_forbidden_gpu_watchdog import (  # noqa: E402
    parse_pmon,
    single_gpu_environment,
)


class ForbiddenGpuWatchdogTests(unittest.TestCase):
    def test_parse_compute_and_graphics_rows(self) -> None:
        output = """
# gpu         pid   type     fb   ccpm    command
    0       1001     G      4      0    python
    1       1002   C+G   2321      0    python
    2          -     -      -      -    -
"""
        self.assertEqual(
            parse_pmon(output),
            [
                {
                    "gpu": 0,
                    "pid": 1001,
                    "type": "G",
                    "memory_mib": 4,
                },
                {
                    "gpu": 1,
                    "pid": 1002,
                    "type": "C+G",
                    "memory_mib": 2321,
                },
            ],
        )

    def test_single_gpu_environment_hides_distributed_state(self) -> None:
        environment = single_gpu_environment(
            {
                "PATH": "/usr/bin",
                "CUDA_VISIBLE_DEVICES": "0,1,2",
                "WORLD_SIZE": "8",
                "LOCAL_RANK": "3",
            },
            0,
        )
        self.assertEqual(environment["PHYSICAL_GPU"], "0")
        self.assertEqual(environment["CUDA_VISIBLE_DEVICES"], "0")
        self.assertEqual(environment["NVIDIA_VISIBLE_DEVICES"], "0")
        self.assertEqual(environment["CUDA_DEVICE_ORDER"], "PCI_BUS_ID")
        self.assertNotIn("WORLD_SIZE", environment)
        self.assertNotIn("LOCAL_RANK", environment)
        self.assertEqual(environment["PATH"], "/usr/bin")

    def test_single_gpu_environment_rejects_negative_index(self) -> None:
        with self.assertRaises(ValueError):
            single_gpu_environment({}, -1)


if __name__ == "__main__":
    unittest.main()
