#!/usr/bin/env python3
"""Fit the V16 joint observation model and replay the unified MPC by CV."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from run_icra_v15_joint_calibration_cv import (  # noqa: E402
    build_rows,
    load_json,
    run,
    write_json,
)
from run_icra_v15b_integrated_scene_conditioned_mpc_cv import (  # noqa: E402
    fitted_resolution_likelihoods,
    replace_symbols,
)
from run_icra_v15b_scene_conditioned_view_calibration import (  # noqa: E402
    load_rows as load_view_rows,
    select_k,
)


DEFAULT_PERCEPTION_ROOT = (
    ROOT / "outputs/calibration/icra_v16_calibration_perception"
)
DEFAULT_OUTPUT_ROOT = ROOT / "outputs/calibration/icra_v16_joint_calibration_cv"
DEFAULT_PLANNER_TEMPLATE = (
    ROOT / "configs/research/icra_v16_unified_task_belief_template.json"
)


def write_auxiliary_calibration_candidates(
    perception_roots: tuple[Path, ...],
    output_root: Path,
    *,
    minimum_iou: float,
    maximum_track_distance_m: float,
) -> None:
    """Write the view model used inside the planner; it is not a policy override."""
    result = load_json(output_root / "result.json")
    alpha = float(
        result["full_calibration_alpha_selection"]["selected_alpha"]
    )
    view_rows = load_view_rows(perception_roots)
    k_selection = select_k(view_rows, (1, 3, 5, 7, 9), 0.1)
    write_json(
        output_root / "scene_conditioned_view_model_candidate.json",
        {
            "schema_version": "icra-v16-scene-conditioned-view-model-candidate-v1",
            "status": "calibration_candidate_not_frozen",
            "episodes": view_rows,
            "neighbor_count": int(k_selection["selected_k"]),
            "probability_pseudocount": 0.1,
            "k_selection": k_selection,
            "role": "condition action observation likelihoods inside unified MPC",
            "policy_override_used": False,
            "training_performed": False,
            "calibration_performed": True,
            "testing_performed": False,
            "valid_for_final_evaluation": False,
        },
    )
    episodes = {}
    for perception_root in perception_roots:
        current = build_rows(
            perception_root,
            minimum_iou=minimum_iou,
            maximum_track_distance_m=maximum_track_distance_m,
        )
        overlap = set(episodes) & set(current)
        if overlap:
            raise ValueError(f"Duplicate calibration seeds: {sorted(overlap)}")
        episodes.update(current)
    episodes = replace_symbols(episodes)
    write_json(
        output_root / "resolution_likelihoods_candidate.json",
        {
            "schema_version": "icra-v16-resolution-likelihoods-candidate-v1",
            "status": "calibration_candidate_not_frozen",
            "dirichlet_alpha": alpha,
            "actions": {
                action: fitted_resolution_likelihoods(episodes, action, alpha)
                for action in ("viewpoint_close_high", "viewpoint_right")
            },
            "policy_override_used": False,
            "training_performed": False,
            "calibration_performed": True,
            "testing_performed": False,
            "valid_for_final_evaluation": False,
        },
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--perception-root",
        type=Path,
        action="append",
        default=[],
        help="Calibration perception root; repeat to combine predeclared supplements.",
    )
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--planner-template", type=Path, default=DEFAULT_PLANNER_TEMPLATE)
    parser.add_argument("--dirichlet-alpha", type=float, default=0.5)
    parser.add_argument(
        "--alpha-grid",
        default="0.05,0.1,0.25,0.5,1.0,2.0,4.0",
    )
    parser.add_argument("--minimum-target-iou", type=float, default=0.25)
    parser.add_argument("--maximum-track-distance-m", type=float, default=0.12)
    parser.add_argument("--defer-cost", type=float)
    parser.add_argument("--wrong-commitment-cost", type=float)
    args = parser.parse_args()
    perception_roots = tuple(path.resolve() for path in args.perception_root)
    if not perception_roots:
        perception_roots = (DEFAULT_PERCEPTION_ROOT.resolve(),)
    result = run(
        perception_roots[0],
        args.output_root.resolve(),
        additional_perception_roots=perception_roots[1:],
        alpha=float(args.dirichlet_alpha),
        alpha_grid=tuple(
            float(value) for value in args.alpha_grid.split(",") if value.strip()
        ),
        minimum_iou=float(args.minimum_target_iou),
        maximum_track_distance_m=float(args.maximum_track_distance_m),
        defer_cost=args.defer_cost,
        wrong_commitment_cost=args.wrong_commitment_cost,
        planner_template=args.planner_template.resolve(),
        persistent_semantic_symbols=True,
    )
    write_auxiliary_calibration_candidates(
        perception_roots,
        args.output_root.resolve(),
        minimum_iou=float(args.minimum_target_iou),
        maximum_track_distance_m=float(args.maximum_track_distance_m),
    )
    print(
        json.dumps(
            {
                "status": result["status"],
                "summary": result["summary"],
                "blocking_reasons": result["blocking_reasons"],
                "runtime_seconds": result["runtime_seconds"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
