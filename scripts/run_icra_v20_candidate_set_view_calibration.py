#!/usr/bin/env python3
"""Calibrate post-remove view outcomes using all planner-visible candidates."""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any

from run_icra_v15b_scene_conditioned_view_calibration import (
    candidate_set_feature_names,
    expected_calibration_error,
    extract_candidate_set_features,
    load_json,
    predict,
    select_k,
    view_mode,
    write_json,
)


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ROOTS = (
    ROOT / "outputs/calibration/icra_v16_calibration_perception",
    ROOT / "outputs/calibration/icra_v15b_supplemental_calibration_perception",
)
DEFAULT_OUTPUT = (
    ROOT
    / "outputs/calibration/icra_v20_candidate_set_view_model_candidate"
    / "scene_conditioned_view_model_candidate.json"
)


def load_rows(roots: tuple[Path, ...]) -> list[dict[str, Any]]:
    rows = []
    for root in roots:
        manifest = load_json(root / "observation_manifest.json")
        for episode in manifest["episodes"]:
            if "post_remove" not in episode["observations"]:
                continue
            family = str(episode["family"])
            rows.append(
                {
                    "seed": int(episode["seed"]),
                    "family": family,
                    "view_mode": view_mode(family),
                    "features": extract_candidate_set_features(
                        root, int(episode["seed"])
                    ),
                    "automatic_simulator_label_used_for_calibration_only": True,
                }
            )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--perception-root", type=Path, action="append", default=[])
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    roots = tuple(path.resolve() for path in args.perception_root) or DEFAULT_ROOTS
    rows = load_rows(roots)
    grid = (1, 3, 5, 7, 9)
    beta = 0.1
    folds = []
    for held_out in rows:
        outer = [row for row in rows if row["seed"] != held_out["seed"]]
        inner = select_k(outer, grid, beta)
        prediction = predict(
            held_out["features"],
            outer,
            k=int(inner["selected_k"]),
            beta=beta,
        )
        predicted = max(prediction["probabilities"], key=prediction["probabilities"].get)
        folds.append(
            {
                "seed": held_out["seed"],
                "family": held_out["family"],
                "true_view_mode": held_out["view_mode"],
                "predicted_view_mode": predicted,
                "probabilities": prediction["probabilities"],
                "confidence": prediction["probabilities"][predicted],
                "correct": predicted == held_out["view_mode"],
                "neighbors": prediction["neighbors"],
                "inner_k_selection": inner,
                "held_out_episode_used_for_fit": False,
                "held_out_future_view_used": False,
            }
        )
    full = select_k(rows, grid, beta)
    accuracy = sum(row["correct"] for row in folds) / len(folds)
    nll = -sum(
        math.log(max(1e-12, row["probabilities"][row["true_view_mode"]]))
        for row in folds
    ) / len(folds)
    model = {
        "schema_version": "icra-v20-candidate-set-scene-conditioned-view-model-v1",
        "status": "calibration_candidate_not_frozen",
        "feature_extractor": "candidate_set_x_sorted_v1",
        "feature_names": list(candidate_set_feature_names()),
        "episodes": rows,
        "neighbor_count": int(full["selected_k"]),
        "probability_pseudocount": beta,
        "k_selection": full,
        "nested_leave_one_episode_out": {
            "episode_count": len(folds),
            "accuracy": accuracy,
            "negative_log_likelihood": nll,
            "expected_calibration_error": expected_calibration_error(folds),
            "view_mode_support": dict(Counter(row["view_mode"] for row in rows)),
            "folds": folds,
        },
        "role": "condition action observation likelihoods inside unified MPC",
        "policy_override_used": False,
        "simulator_ground_truth_used_for_features": False,
        "simulator_ground_truth_used_for_calibration_labels": True,
        "training_performed": False,
        "calibration_performed": True,
        "testing_performed": False,
        "reserved_test_seeds_used": False,
        "valid_for_final_evaluation": False,
        "gpu_used": False,
    }
    write_json(args.output.resolve(), model)
    print(json.dumps({"episode_count": len(rows), "accuracy": accuracy, "negative_log_likelihood": nll, "expected_calibration_error": model["nested_leave_one_episode_out"]["expected_calibration_error"], "selected_k": full["selected_k"]}, indent=2))


if __name__ == "__main__":
    main()
