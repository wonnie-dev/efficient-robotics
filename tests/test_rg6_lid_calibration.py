"""CPU tests for the RG6/lid lab-calibration readiness gate."""

from __future__ import annotations

import copy
import json
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from rg6_lid_calibration import (  # noqa: E402
    DEFAULT_CONFIG,
    simulation_parameters,
    validate_calibration,
)

DEVELOPMENT_PROXY = (
    ROOT / "configs" / "hardware" / "rg6_lid_development_proxy.json"
)


def completed_payload() -> dict:
    payload = json.loads(DEFAULT_CONFIG.read_text(encoding="utf-8"))
    payload["calibration_status"] = "lab_measured"
    payload["calibration_performed"] = True
    payload["measurement"] = {
        "lab_name": "transfer_test_lab",
        "measured_by": "operator",
        "measured_at": "2026-08-04T12:00:00-07:00",
        "notes": "synthetic unit-test fixture",
    }
    payload["hardware"].update(
        {
            "gripper_serial_or_asset_id": "fixture_rg6",
            "fingertip_model": "fixture_pad",
            "fingertip_material": "EPDM",
            "fingertip_geometry_confirmed": True,
        }
    )
    payload["lid"] = {
        "mass_kg": 0.55,
        "plate_center_local_m": [0.0, 0.0, 0.166],
        "plate_full_extents_m": [0.362, 0.338, 0.014],
        "handle_full_extents_m": [0.080, 0.030, 0.035],
        "handle_center_local_m": [0.0, 0.0, 0.1905],
        "handle_material": "fixture_polymer",
    }
    payload["controller"].update(
        {
            "commanded_grip_force_n": 40.0,
            "closing_speed_m_s": 0.02,
            "command_interface": "fixture",
        }
    )
    payload["simulation_mapping"] = {
        "initial_drive_torque_nm": 1.5,
        "maximum_drive_torque_nm": 4.0,
        "minimum_force_per_finger_n": 8.0,
        "minimum_combined_force_n": 25.0,
        "static_friction": 0.9,
        "dynamic_friction": 0.7,
        "compliant_contact_stiffness_n_m": 30000.0,
        "compliant_contact_damping_n_s_m": 90.0,
        "mapping_source": "lab_fit",
        "fit_notes": "synthetic unit-test fixture",
    }
    payload["micro_lift_trials"] = [
        {
            "trial_id": f"fixture_{index:03d}",
            "measured_lift_m": 0.009,
            "relative_translation_m": 0.002,
            "bilateral_contact_maintained": True,
            "peak_contact_force_per_finger_n": [18.0, 19.0],
            "maximum_penetration_m": 0.0002,
            "unexpected_environment_collision": False,
            "notes": "synthetic unit-test fixture",
        }
        for index in range(5)
    ]
    return payload


class RG6LidCalibrationTests(unittest.TestCase):
    def test_repository_template_is_valid_but_deliberately_not_ready(self) -> None:
        payload = json.loads(DEFAULT_CONFIG.read_text(encoding="utf-8"))
        report = validate_calibration(payload)
        self.assertTrue(report["structure_valid"])
        self.assertFalse(report["transfer_ready"])
        self.assertGreater(len(report["missing_ready_fields"]), 0)
        self.assertEqual(report["trial_count"], 0)

    def test_completed_lab_fixture_is_ready(self) -> None:
        payload = completed_payload()
        report = validate_calibration(payload)
        self.assertTrue(report["transfer_ready"])
        self.assertEqual(report["trial_pass_count"], 5)
        parameters = simulation_parameters(payload)
        self.assertEqual(parameters["cover_mass_kg"], 0.55)
        self.assertEqual(parameters["maximum_drive_torque_nm"], 4.0)
        self.assertEqual(
            parameters["physical_parameters_status"],
            "lab_measured_calibration",
        )

    def test_public_spec_proxy_is_usable_but_not_transfer_ready(self) -> None:
        payload = json.loads(DEVELOPMENT_PROXY.read_text(encoding="utf-8"))
        report = validate_calibration(payload)
        self.assertTrue(report["structure_valid"])
        self.assertTrue(report["development_proxy_usable"])
        self.assertFalse(report["transfer_ready"])
        with self.assertRaises(ValueError):
            simulation_parameters(payload)
        parameters = simulation_parameters(payload, allow_provisional=True)
        self.assertTrue(parameters["development_proxy"])
        self.assertFalse(parameters["transfer_ready"])
        self.assertEqual(parameters["commanded_grip_force_n"], 25.0)
        self.assertEqual(
            parameters["compliant_contact_stiffness_n_m"], 25000.0
        )
        self.assertEqual(
            parameters["compliant_contact_damping_n_s_m"], 80.0
        )

    def test_micro_lift_slip_failure_blocks_readiness(self) -> None:
        payload = completed_payload()
        for trial in payload["micro_lift_trials"][:2]:
            trial["relative_translation_m"] = 0.006
        report = validate_calibration(payload)
        self.assertFalse(report["transfer_ready"])
        self.assertEqual(report["trial_pass_count"], 3)

    def test_safety_limits_cannot_be_relaxed_in_calibration_file(self) -> None:
        payload = completed_payload()
        payload["acceptance"]["maximum_relative_translation_m"] = 0.015
        report = validate_calibration(payload)
        self.assertFalse(report["structure_valid"])
        self.assertFalse(report["transfer_ready"])

    def test_provisional_mapping_cannot_be_called_lab_fit(self) -> None:
        payload = completed_payload()
        payload["simulation_mapping"]["mapping_source"] = "guessed"
        report = validate_calibration(payload)
        self.assertFalse(report["structure_valid"])
        with self.assertRaises(ValueError):
            simulation_parameters(payload)

    def test_force_outside_rg6_range_is_rejected(self) -> None:
        payload = copy.deepcopy(completed_payload())
        payload["controller"]["commanded_grip_force_n"] = 121.0
        report = validate_calibration(payload)
        self.assertFalse(report["structure_valid"])

    def test_invalid_compliant_contact_mapping_is_rejected(self) -> None:
        payload = completed_payload()
        payload["simulation_mapping"][
            "compliant_contact_stiffness_n_m"
        ] = -1.0
        report = validate_calibration(payload)
        self.assertFalse(report["structure_valid"])


if __name__ == "__main__":
    unittest.main()
