#!/usr/bin/env python3
"""Audit a completed remove-cover run after success-gate changes."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from run_live_single_gpu_pipeline import write_json_atomic
from run_remove_cover_live_smoke import removal_contact_success


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", type=Path)
    args = parser.parse_args()
    run_dir = args.run_dir.resolve()
    original = json.loads(
        (run_dir / "remove_cover_smoke_result.json").read_text(
            encoding="utf-8"
        )
    )
    server = json.loads(
        (run_dir / "server_result.json").read_text(encoding="utf-8")
    )
    removal = server.get("cover_removal_execution") or {}
    gates = {
        "server_result_completed": server.get("status") == "completed",
        "cover_removal_executed": bool(server.get("cover_removal_executed")),
        "post_removal_observation_generated": bool(
            server.get("post_removal_observation_generated")
        ),
        "removal_verified": bool(removal.get("removal_verified")),
        "bilateral_contact_before_lift": bool(
            removal.get("bilateral_contact_before_lift")
        ),
        "release_aware_contact_contract": removal_contact_success(removal),
        "zero_unexpected_robot_contacts": not bool(
            removal.get("unexpected_environment_pairs")
        ),
        "zero_unexpected_target_environment_contacts": not bool(
            removal.get("unexpected_target_environment_pairs")
        ),
        "contact_force_within_limit": bool(
            removal.get("contact_force_within_limit")
        ),
        "contact_penetration_within_limit": bool(
            removal.get("contact_penetration_within_limit")
        ),
        "target_visibility_increased": (
            original["post_removal_target_visible_pixel_count"]
            > original["initial_target_visible_pixel_count"]
        ),
    }
    audit = {
        "schema_version": "remove-cover-smoke-revalidation-v1",
        "status": "completed" if all(gates.values()) else "failed",
        "source_run": str(run_dir),
        "source_original_status": original.get("status"),
        "gate_revision": (
            "intentional release uses pre-release bilateral contact plus "
            "release-retreat-stability instead of final bilateral contact"
        ),
        "gates": gates,
        "all_gates_passed": all(gates.values()),
        "isaac_server_exit_code_persisted": False,
        "gpu_watchdog_violation": None,
        "valid_for_final_evaluation": False,
    }
    output = run_dir / "remove_cover_smoke_revalidation.json"
    write_json_atomic(output, audit)
    print(output)
    if not audit["all_gates_passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
