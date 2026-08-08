"""Validate lab measurements before claiming RG6 lid-transfer readiness.

This module is deliberately independent of Isaac Sim so the handoff package
and calibration records can be checked on a CPU-only machine.  It separates a
valid *worksheet* from a completed, lab-measured calibration that may supply
parameters to the simulator.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = (
    ROOT / "configs" / "hardware" / "rg6_lid_transfer_calibration.json"
)
SUPPORTED_SCHEMA = "rg6-lid-transfer-calibration-v1"
READY_STATUS = "lab_measured"
PROVISIONAL_STATUS = "provisional_public_spec"


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _finite_number(value: Any) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
    )


def _positive_vector(value: Any, length: int) -> bool:
    return (
        isinstance(value, list)
        and len(value) == length
        and all(_finite_number(item) and float(item) > 0.0 for item in value)
    )


def _nested(payload: dict[str, Any], dotted_path: str) -> Any:
    value: Any = payload
    for key in dotted_path.split("."):
        if not isinstance(value, dict) or key not in value:
            return None
        value = value[key]
    return value


READY_REQUIRED_FIELDS = (
    "measurement.lab_name",
    "measurement.measured_by",
    "measurement.measured_at",
    "hardware.gripper_serial_or_asset_id",
    "hardware.fingertip_model",
    "hardware.fingertip_material",
    "hardware.fingertip_geometry_confirmed",
    "lid.mass_kg",
    "lid.plate_center_local_m",
    "lid.plate_full_extents_m",
    "lid.handle_full_extents_m",
    "lid.handle_center_local_m",
    "lid.handle_material",
    "controller.commanded_grip_force_n",
    "simulation_mapping.initial_drive_torque_nm",
    "simulation_mapping.maximum_drive_torque_nm",
    "simulation_mapping.minimum_force_per_finger_n",
    "simulation_mapping.minimum_combined_force_n",
    "simulation_mapping.static_friction",
    "simulation_mapping.dynamic_friction",
    "simulation_mapping.compliant_contact_stiffness_n_m",
    "simulation_mapping.compliant_contact_damping_n_s_m",
    "simulation_mapping.mapping_source",
)

DEVELOPMENT_REQUIRED_FIELDS = (
    "hardware.arm_model",
    "hardware.gripper_model",
    "hardware.fingertip_model",
    "hardware.fingertip_material",
    "lid.mass_kg",
    "lid.plate_center_local_m",
    "lid.plate_full_extents_m",
    "lid.handle_full_extents_m",
    "lid.handle_center_local_m",
    "lid.handle_material",
    "controller.commanded_grip_force_n",
    "simulation_mapping.initial_drive_torque_nm",
    "simulation_mapping.maximum_drive_torque_nm",
    "simulation_mapping.minimum_force_per_finger_n",
    "simulation_mapping.minimum_combined_force_n",
    "simulation_mapping.static_friction",
    "simulation_mapping.dynamic_friction",
    "simulation_mapping.compliant_contact_stiffness_n_m",
    "simulation_mapping.compliant_contact_damping_n_s_m",
    "simulation_mapping.mapping_source",
)


def _trial_passes(trial: dict[str, Any], acceptance: dict[str, Any]) -> bool:
    return bool(
        trial.get("bilateral_contact_maintained", False)
        and _finite_number(trial.get("measured_lift_m"))
        and float(trial["measured_lift_m"])
        >= float(acceptance["minimum_measured_lift_m"])
        and _finite_number(trial.get("relative_translation_m"))
        and float(trial["relative_translation_m"])
        <= float(acceptance["maximum_relative_translation_m"])
        and not trial.get("unexpected_environment_collision", True)
    )


def validate_calibration(payload: dict[str, Any]) -> dict[str, Any]:
    """Return a deterministic readiness report without mutating the input."""
    errors: list[str] = []
    if payload.get("schema_version") != SUPPORTED_SCHEMA:
        errors.append("unsupported schema_version")
    status = payload.get("calibration_status")
    if status not in {
        "pending_lab_measurement",
        PROVISIONAL_STATUS,
        READY_STATUS,
    }:
        errors.append(
            "calibration_status must be pending_lab_measurement, "
            "provisional_public_spec, or lab_measured"
        )
    if payload.get("training_performed") is not False:
        errors.append("training_performed must be false")
    if payload.get("valid_for_final_evaluation") is not False:
        errors.append("hardware calibration must not be labeled final evaluation")
    if status == READY_STATUS and payload.get("calibration_performed") is not True:
        errors.append("lab_measured status requires calibration_performed=true")

    acceptance = payload.get("acceptance")
    if not isinstance(acceptance, dict):
        errors.append("acceptance must be an object")
        acceptance = {}
    expected_limits = {
        "requested_micro_lift_m": 0.010,
        "minimum_measured_lift_m": 0.007,
        "maximum_relative_translation_m": 0.005,
        "maximum_penetration_m": 0.003,
        "maximum_contact_force_per_finger_n": 60.0,
    }
    for name, expected in expected_limits.items():
        value = acceptance.get(name)
        if not _finite_number(value) or not math.isclose(
            float(value), expected, rel_tol=0.0, abs_tol=1.0e-12
        ):
            errors.append(f"acceptance.{name} must remain {expected}")
    minimum_trials = acceptance.get("minimum_trial_count")
    minimum_rate = acceptance.get("minimum_pass_rate")
    if not isinstance(minimum_trials, int) or minimum_trials < 5:
        errors.append("acceptance.minimum_trial_count must be at least 5")
    if not _finite_number(minimum_rate) or not 0.8 <= float(minimum_rate) <= 1.0:
        errors.append("acceptance.minimum_pass_rate must be in [0.8, 1.0]")

    missing = [
        field
        for field in READY_REQUIRED_FIELDS
        if _nested(payload, field) in (None, "", [])
    ]
    fingertip_confirmed = _nested(
        payload, "hardware.fingertip_geometry_confirmed"
    )
    if fingertip_confirmed is not True:
        missing.append("hardware.fingertip_geometry_confirmed=true")

    lid = payload.get("lid", {})
    if lid.get("mass_kg") is not None and (
        not _finite_number(lid["mass_kg"]) or float(lid["mass_kg"]) <= 0.0
    ):
        errors.append("lid.mass_kg must be positive")
    for field in ("plate_full_extents_m", "handle_full_extents_m"):
        value = lid.get(field)
        if value is not None and not _positive_vector(value, 3):
            errors.append(f"lid.{field} must contain three positive values")
    for field in ("plate_center_local_m", "handle_center_local_m"):
        center = lid.get(field)
        if center is not None and not (
            isinstance(center, list)
            and len(center) == 3
            and all(_finite_number(value) for value in center)
        ):
            errors.append(f"lid.{field} must contain three finite values")

    commanded_force = _nested(payload, "controller.commanded_grip_force_n")
    if commanded_force is not None and (
        not _finite_number(commanded_force)
        or not 25.0 <= float(commanded_force) <= 120.0
    ):
        errors.append("controller.commanded_grip_force_n must be in [25, 120]")
    mapping = payload.get("simulation_mapping", {})
    initial_torque = mapping.get("initial_drive_torque_nm")
    maximum_torque = mapping.get("maximum_drive_torque_nm")
    if initial_torque is not None and (
        not _finite_number(initial_torque)
        or not 0.0 < float(initial_torque) <= 2.0
    ):
        errors.append("initial_drive_torque_nm must be in (0, 2]")
    if maximum_torque is not None and (
        not _finite_number(maximum_torque)
        or initial_torque is None
        or not float(initial_torque) <= float(maximum_torque) <= 12.0
    ):
        errors.append("maximum_drive_torque_nm must be between initial torque and 12")
    static_friction = mapping.get("static_friction")
    dynamic_friction = mapping.get("dynamic_friction")
    if static_friction is not None or dynamic_friction is not None:
        if not (
            _finite_number(static_friction)
            and _finite_number(dynamic_friction)
            and 0.0 < float(dynamic_friction) <= float(static_friction) <= 2.0
        ):
            errors.append("friction must satisfy 0 < dynamic <= static <= 2")
    compliant_stiffness = mapping.get("compliant_contact_stiffness_n_m")
    compliant_damping = mapping.get("compliant_contact_damping_n_s_m")
    if compliant_stiffness is not None and (
        not _finite_number(compliant_stiffness)
        or float(compliant_stiffness) <= 0.0
    ):
        errors.append("compliant_contact_stiffness_n_m must be positive")
    if compliant_damping is not None and (
        not _finite_number(compliant_damping)
        or float(compliant_damping) < 0.0
    ):
        errors.append("compliant_contact_damping_n_s_m must be non-negative")
    mapping_source = mapping.get("mapping_source")
    allowed_mapping_sources = (None, "lab_fit", "provisional_simulation_fit")
    if mapping_source not in allowed_mapping_sources:
        errors.append(
            "simulation_mapping.mapping_source must be lab_fit or "
            "provisional_simulation_fit"
        )
    if status == READY_STATUS and mapping_source != "lab_fit":
        errors.append("lab_measured status requires mapping_source=lab_fit")
    if (
        status == PROVISIONAL_STATUS
        and mapping_source != "provisional_simulation_fit"
    ):
        errors.append(
            "provisional_public_spec status requires "
            "mapping_source=provisional_simulation_fit"
        )
    minimum_per_finger = mapping.get("minimum_force_per_finger_n")
    minimum_combined = mapping.get("minimum_combined_force_n")
    if minimum_per_finger is not None and (
        not _finite_number(minimum_per_finger)
        or not 0.0 < float(minimum_per_finger) < 60.0
    ):
        errors.append("minimum_force_per_finger_n must be in (0, 60)")
    if minimum_combined is not None and (
        not _finite_number(minimum_combined)
        or minimum_per_finger is None
        or not 2.0 * float(minimum_per_finger)
        <= float(minimum_combined)
        <= 120.0
    ):
        errors.append(
            "minimum_combined_force_n must be at least twice the per-finger floor and at most 120"
        )

    trials = payload.get("micro_lift_trials", [])
    if not isinstance(trials, list):
        errors.append("micro_lift_trials must be a list")
        trials = []
    trial_results = [
        _trial_passes(trial, acceptance)
        for trial in trials
        if isinstance(trial, dict) and not errors
    ]
    pass_count = sum(trial_results)
    pass_rate = pass_count / len(trial_results) if trial_results else 0.0
    enough_trials = isinstance(minimum_trials, int) and len(trials) >= minimum_trials
    enough_passes = (
        _finite_number(minimum_rate)
        and pass_rate >= float(minimum_rate)
    )
    ready = bool(
        not errors
        and status == READY_STATUS
        and not missing
        and enough_trials
        and enough_passes
    )
    missing_development_fields = [
        field
        for field in DEVELOPMENT_REQUIRED_FIELDS
        if _nested(payload, field) in (None, "", [])
    ]
    development_proxy_usable = bool(
        not errors
        and status == PROVISIONAL_STATUS
        and not missing_development_fields
    )
    return {
        "schema_version": "rg6-lid-transfer-readiness-report-v1",
        "calibration_status": status,
        "structure_valid": not errors,
        "transfer_ready": ready,
        "development_proxy_usable": development_proxy_usable,
        "missing_ready_fields": sorted(set(missing)),
        "missing_development_fields": sorted(
            set(missing_development_fields)
        ),
        "errors": errors,
        "trial_count": len(trials),
        "trial_pass_count": pass_count,
        "trial_pass_rate": pass_rate,
        "minimum_trial_count": minimum_trials,
        "minimum_pass_rate": minimum_rate,
        "training_performed": False,
        "calibration_only": True,
        "valid_for_final_evaluation": False,
    }


def simulation_parameters(
    payload: dict[str, Any],
    *,
    allow_provisional: bool = False,
) -> dict[str, Any]:
    """Extract parameters only from a completed, validated lab calibration."""
    report = validate_calibration(payload)
    provisional_allowed = bool(
        allow_provisional and report["development_proxy_usable"]
    )
    if not report["transfer_ready"] and not provisional_allowed:
        raise ValueError(f"RG6 lid calibration is not transfer-ready: {report}")
    lid = payload["lid"]
    controller = payload["controller"]
    mapping = payload["simulation_mapping"]
    return {
        "cover_mass_kg": float(lid["mass_kg"]),
        "cover_plate_full_extents_m": [
            float(value) for value in lid["plate_full_extents_m"]
        ],
        "cover_plate_center_local_m": [
            float(value) for value in lid["plate_center_local_m"]
        ],
        "cover_handle_full_extents_m": [
            float(value) for value in lid["handle_full_extents_m"]
        ],
        "cover_handle_center_local_m": [
            float(value) for value in lid["handle_center_local_m"]
        ],
        "commanded_grip_force_n": float(controller["commanded_grip_force_n"]),
        "initial_drive_torque_nm": float(mapping["initial_drive_torque_nm"]),
        "maximum_drive_torque_nm": float(mapping["maximum_drive_torque_nm"]),
        "minimum_force_per_finger_n": float(
            mapping["minimum_force_per_finger_n"]
        ),
        "minimum_combined_force_n": float(
            mapping["minimum_combined_force_n"]
        ),
        "static_friction": float(mapping["static_friction"]),
        "dynamic_friction": float(mapping["dynamic_friction"]),
        "compliant_contact_stiffness_n_m": float(
            mapping["compliant_contact_stiffness_n_m"]
        ),
        "compliant_contact_damping_n_s_m": float(
            mapping["compliant_contact_damping_n_s_m"]
        ),
        "physical_parameters_status": (
            "lab_measured_calibration"
            if report["transfer_ready"]
            else "provisional_public_spec_and_simulation_assumptions"
        ),
        "transfer_ready": report["transfer_ready"],
        "development_proxy": provisional_allowed,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--require-transfer-ready", action="store_true")
    parser.add_argument("--allow-provisional", action="store_true")
    args = parser.parse_args()
    config_path = args.config if args.config.is_absolute() else ROOT / args.config
    report = validate_calibration(load_json(config_path))
    rendered = json.dumps(report, indent=2, sort_keys=True)
    if args.output:
        output_path = args.output if args.output.is_absolute() else ROOT / args.output
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)
    if not report["structure_valid"] or (
        args.require_transfer_ready and not report["transfer_ready"]
    ) or (
        not args.require_transfer_ready
        and args.allow_provisional
        and not report["development_proxy_usable"]
    ):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
