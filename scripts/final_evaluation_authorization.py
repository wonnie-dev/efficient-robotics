"""Fail-closed output authorization for frozen reserved-test captures."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from audit_icra_evaluation_protocol import audit, load_json


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PROTOCOL = (
    ROOT / "configs" / "research" / "icra_simulation_evaluation_protocol_v1.json"
)


def _v16_reserved_seeds(protocol: dict[str, Any]) -> set[int]:
    split = protocol["data_split"]
    start, stop = [int(value) for value in split["reserved_test_seed_range"]]
    if stop < start:
        raise ValueError("Invalid V16 reserved-test range")
    seeds = set(range(start, stop + 1))
    if len(seeds) != int(split["reserved_test_episode_count"]):
        raise ValueError("V16 reserved-test count does not match its seed range")
    return seeds


def validate_output_authorization(
    *,
    seed: int,
    output_dir: Path,
    final_evaluation_authorized: bool,
    protocol_path: Path = DEFAULT_PROTOCOL,
) -> Path:
    """Return the allowed output root after checking the frozen protocol."""
    if not final_evaluation_authorized:
        allowed = (ROOT / "outputs" / "live_pipeline").resolve()
        if not output_dir.is_relative_to(allowed):
            raise ValueError(f"Output directory must be under {allowed}: {output_dir}")
        return allowed
    protocol = load_json(protocol_path.resolve())
    if str(protocol.get("schema_version", "")).startswith("icra-v16"):
        if (
            protocol.get("status") != "frozen_before_untouched_test"
            or protocol.get("reserved_test_launch_authorized") is not True
            or protocol.get("testing_performed") is not False
            or protocol.get("reserved_test_seeds_used") is not False
        ):
            raise RuntimeError("The V16 final protocol is not authorized and unopened")
        reserved = _v16_reserved_seeds(protocol)
    else:
        preflight = audit(protocol)
        reserved = {
            int(value) for value in protocol["data_split"]["reserved_test_seeds"]
        }
        if preflight["status"] != "passed":
            raise RuntimeError(f"Final-evaluation preflight failed: {preflight}")
    if seed not in reserved:
        raise ValueError(f"Final-evaluation seed is not reserved: {seed}")
    allowed = (
        ROOT
        / protocol.get(
            "final_output_root",
            "outputs/final_evaluation/icra_protocol_v1",
        )
    ).resolve()
    if not output_dir.is_relative_to(allowed):
        raise ValueError(f"Final output directory must be under {allowed}: {output_dir}")
    return allowed
