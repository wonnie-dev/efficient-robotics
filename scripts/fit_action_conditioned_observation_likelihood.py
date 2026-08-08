"""Fit a calibration-only discrete action-conditioned observation pilot."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = (
    ROOT
    / "configs"
    / "research"
    / "action_conditioned_observation_likelihood_reference_seed165_173.json"
)


def resolve_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def fit_categorical_table(
    rows: list[dict[str, str]],
    *,
    condition_keys: tuple[str, ...],
    outcome_key: str,
    outcomes: tuple[str, ...],
    alpha: float,
    minimum_cell_count: int,
) -> dict[str, Any]:
    """Return explicit-count Dirichlet-smoothed categorical tables."""
    if alpha <= 0.0:
        raise ValueError("Dirichlet alpha must be positive")
    grouped: dict[tuple[str, ...], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        outcome = row[outcome_key]
        if outcome not in outcomes:
            raise ValueError(
                f"Unexpected {outcome_key}={outcome!r}; expected {outcomes}"
            )
        grouped[tuple(row[key] for key in condition_keys)].append(row)
    cells = []
    for condition, items in sorted(grouped.items()):
        counts = {
            outcome: sum(item[outcome_key] == outcome for item in items)
            for outcome in outcomes
        }
        denominator = len(items) + alpha * len(outcomes)
        probabilities = {
            outcome: (counts[outcome] + alpha) / denominator
            for outcome in outcomes
        }
        cells.append(
            {
                "condition": dict(zip(condition_keys, condition)),
                "episode_count": len(items),
                "counts": counts,
                "dirichlet_smoothed_probabilities": probabilities,
                "sparse": len(items) < minimum_cell_count,
            }
        )
    return {
        "condition_keys": list(condition_keys),
        "outcome_key": outcome_key,
        "outcomes": list(outcomes),
        "dirichlet_alpha": alpha,
        "cells": cells,
        "cell_count": len(cells),
        "sparse_cell_count": sum(cell["sparse"] for cell in cells),
    }


def reference_occlusion_state(ground_truth: dict, view_id: str) -> str:
    measurement = ground_truth[
        "objective_reference_occlusion_ground_truth"
    ][view_id]
    if not measurement["valid"]:
        return "unknown"
    return "no" if measurement["severity"] == "no" else "yes"


def total_visibility_state(ground_truth: dict, view_id: str) -> str:
    measurement = ground_truth["objective_occlusion_ground_truth"][view_id]
    if not measurement["valid"]:
        return "unknown"
    return "fully_hidden" if measurement["fully_hidden"] else "visible"


def build_rows(
    calibration_root: Path,
    hybrid_audit_csv: Path,
    view_to_action: dict[str, str],
) -> tuple[list[dict[str, str]], list[int]]:
    perception_config = load_json(
        calibration_root / "perception_config.json"
    )
    records = {
        item["sample_id"]: item
        for item in load_json(
            calibration_root / "calibration_records.json"
        )["records"]
    }
    with hybrid_audit_csv.open("r", encoding="utf-8", newline="") as stream:
        audit_rows = list(csv.DictReader(stream))
    target_audit = {
        row["sample_id"]: row
        for row in audit_rows
        if row["matched_entity_posthoc"] == "target_red"
    }
    rows = []
    seeds = set()
    for sample in perception_config["samples"]:
        sample_id = sample["sample_id"]
        seed = int(sample["seed"])
        seeds.add(seed)
        view_id = Path(sample["observation_dir"]).name
        if view_id not in view_to_action:
            raise KeyError(f"No action mapping for view {view_id}")
        ground_truth = load_json(
            resolve_path(sample["calibration_ground_truth_file"])
        )
        record = records[sample_id]
        target_row = target_audit.get(sample_id)
        target_candidate = next(
            (
                candidate
                for candidate in record["candidates"]
                if candidate["target_label"]
            ),
            None,
        )
        proposal_observation = (
            "present" if target_candidate is not None else "missing"
        )
        rows.append(
            {
                "sample_id": sample_id,
                "seed": str(seed),
                "action": view_to_action[view_id],
                "world_membership": ground_truth["world_ground_truth"][
                    "entities"
                ]["target_red"]["membership"],
                "reference_occlusion_state": reference_occlusion_state(
                    ground_truth, view_id
                ),
                "total_visibility_state": total_visibility_state(
                    ground_truth, view_id
                ),
                "proposal_observation": proposal_observation,
                "membership_observation": (
                    target_row["hybrid_membership"]
                    if target_row is not None
                    else "missing"
                ),
                "reference_occlusion_observation": (
                    target_row["hybrid_occluded_by"]
                    if target_row is not None
                    else "missing"
                ),
            }
        )
    return rows, sorted(seeds)


def build_transition_rows(
    rows: list[dict[str, str]],
) -> list[dict[str, str]]:
    """Pair each reachable re-observation with its episode's center state."""
    by_seed: dict[str, dict[str, dict[str, str]]] = defaultdict(dict)
    for row in rows:
        by_seed[row["seed"]][row["action"]] = row
    transitions = []
    for seed, action_rows in sorted(by_seed.items(), key=lambda item: int(item[0])):
        if "initial_observation" not in action_rows:
            raise ValueError(f"Seed {seed} has no initial observation")
        initial = action_rows["initial_observation"]
        for action in ("viewpoint_close_high", "viewpoint_right"):
            if action not in action_rows:
                raise ValueError(f"Seed {seed} has no {action} observation")
            next_row = action_rows[action]
            transitions.append(
                {
                    "seed": seed,
                    "action": action,
                    "current_reference_occlusion_state": initial[
                        "reference_occlusion_state"
                    ],
                    "next_reference_occlusion_state": next_row[
                        "reference_occlusion_state"
                    ],
                    "current_total_visibility_state": initial[
                        "total_visibility_state"
                    ],
                    "next_total_visibility_state": next_row[
                        "total_visibility_state"
                    ],
                }
            )
    return transitions


def run(config_path: Path) -> dict[str, Any]:
    config = load_json(config_path)
    if config.get("training_performed") is not False:
        raise ValueError("This fit is calibration, not training")
    if config.get("testing_performed") is not False:
        raise ValueError("Testing must remain false")
    if config.get("apply_to_mpc") is not False:
        raise ValueError("Pilot likelihood must not be applied to MPC")
    calibration_root = resolve_path(config["calibration_root"])
    rows, seeds = build_rows(
        calibration_root,
        resolve_path(config["hybrid_audit_csv"]),
        config["view_to_action"],
    )
    transition_rows = build_transition_rows(rows)
    alpha = float(config["dirichlet_alpha"])
    minimum_cell_count = int(config["minimum_cell_episode_count"])
    tables = {
        "proposal_observation": fit_categorical_table(
            rows,
            condition_keys=("action", "total_visibility_state"),
            outcome_key="proposal_observation",
            outcomes=("present", "missing"),
            alpha=alpha,
            minimum_cell_count=minimum_cell_count,
        ),
        "membership_observation": fit_categorical_table(
            rows,
            condition_keys=("action", "world_membership"),
            outcome_key="membership_observation",
            outcomes=("inside", "outside", "unknown", "missing"),
            alpha=alpha,
            minimum_cell_count=minimum_cell_count,
        ),
        "reference_occlusion_observation": fit_categorical_table(
            rows,
            condition_keys=("action", "reference_occlusion_state"),
            outcome_key="reference_occlusion_observation",
            outcomes=("yes", "no", "unknown", "missing"),
            alpha=alpha,
            minimum_cell_count=minimum_cell_count,
        ),
    }
    transition_tables = {
        "reference_occlusion_transition": fit_categorical_table(
            transition_rows,
            condition_keys=(
                "action",
                "current_reference_occlusion_state",
            ),
            outcome_key="next_reference_occlusion_state",
            outcomes=("yes", "no", "unknown"),
            alpha=alpha,
            minimum_cell_count=minimum_cell_count,
        ),
        "total_visibility_transition": fit_categorical_table(
            transition_rows,
            condition_keys=(
                "action",
                "current_total_visibility_state",
            ),
            outcome_key="next_total_visibility_state",
            outcomes=("visible", "fully_hidden", "unknown"),
            alpha=alpha,
            minimum_cell_count=minimum_cell_count,
        ),
    }
    minimum_episodes = int(config["minimum_calibration_episode_count"])
    blocking_reasons = []
    if len(seeds) < minimum_episodes:
        blocking_reasons.append(
            "fewer_than_required_episode_disjoint_calibration_scenes"
        )
    if any(table["sparse_cell_count"] for table in tables.values()):
        blocking_reasons.append("sparse_action_latent_observation_cells")
    if any(
        table["sparse_cell_count"]
        for table in transition_tables.values()
    ):
        blocking_reasons.append("sparse_action_state_transition_cells")
    blocking_reasons.extend(
        [
            "discrete_scene_family_support_only",
            "task_risk_gate_not_calibrated",
            "final_test_not_performed",
        ]
    )
    result = {
        "schema_version": "action-conditioned-observation-likelihood-fit-v1",
        "experiment_id": config["experiment_id"],
        "split": "calibration_only",
        "episode_count": len(seeds),
        "seeds": seeds,
        "observation_count": len(rows),
        "transition_count": len(transition_rows),
        "actions": sorted({row["action"] for row in rows}),
        "tables": tables,
        "transition_tables": transition_tables,
        "deployment_decision": {
            "apply_to_mpc": False,
            "blocking_reasons": blocking_reasons,
        },
        "interpretation": {
            "probabilities_are_dirichlet_smoothed_calibration_estimates": True,
            "simulator_ground_truth_used_only_for_latent_conditioning": True,
            "learned_observations_used_as_outcomes": True,
            "state_transitions_pair_center_with_reachable_views": True,
            "valid_for_final_evaluation": False,
        },
        "training_performed": False,
        "calibration_performed": True,
        "testing_performed": False,
        "reserved_test_seeds_used": False,
        "valid_for_final_evaluation": False,
    }
    output_path = resolve_path(config["output_path"])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    rows_path = output_path.with_name("rows.csv")
    with rows_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    transition_rows_path = output_path.with_name("transition_rows.csv")
    with transition_rows_path.open(
        "w", encoding="utf-8", newline=""
    ) as stream:
        writer = csv.DictWriter(
            stream, fieldnames=list(transition_rows[0])
        )
        writer.writeheader()
        writer.writerows(transition_rows)
    print(f"ACTION_CONDITIONED_OBSERVATION_FIT={output_path}")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    args = parser.parse_args()
    run(args.config.resolve())


if __name__ == "__main__":
    main()
