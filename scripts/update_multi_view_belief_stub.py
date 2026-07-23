"""Fuse per-view rule-based beliefs into a provisional temporal belief trace.

The normalized-product update is only an interface stub. It assumes
conditionally independent observations and is not a calibrated final method.
"""

import argparse
import csv
import json
import math
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs/scene_graph/multi_view_belief_stub.json"


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as stream:
        return json.load(stream)


def normalize(distribution: dict[str, float], floor: float) -> dict[str, float]:
    clipped = {key: max(floor, value) for key, value in distribution.items()}
    total = sum(clipped.values())
    return {key: value / total for key, value in clipped.items()}


def fuse_distributions(
    prior: dict[str, float] | None,
    observation: dict[str, float],
    floor: float,
) -> dict[str, float]:
    observation = normalize(observation, floor)
    if prior is None:
        return observation
    if prior.keys() != observation.keys():
        raise ValueError("Prior and observation distributions must use identical labels")
    product = {
        label: max(floor, prior[label]) * max(floor, observation[label])
        for label in prior
    }
    return normalize(product, floor)


def binary_target_distribution(graph: dict) -> dict[str, float]:
    candidates = {
        node["id"]: node["belief"]["target_probability"]
        for node in graph["nodes"]
        if node["type"] == "object"
    }
    if len(candidates) < 2:
        raise ValueError("At least two object candidates are required")
    return candidates


def relation_distribution(graph: dict) -> dict[str, float]:
    if len(graph["edges"]) != 1:
        raise ValueError("The current stub expects exactly one task-relevant relation edge")
    return graph["edges"][0]["belief"]["relation_distribution"]


def entropy(distribution: dict[str, float]) -> float:
    return -sum(value * math.log(value) for value in distribution.values() if value > 0.0)


def rounded(distribution: dict[str, float]) -> dict[str, float]:
    return {key: round(value, 8) for key, value in distribution.items()}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    args = parser.parse_args()

    config = load_json(args.config)
    floor = config["fusion"]["probability_floor"]
    gate = config["temporary_execution_gate"]
    target_belief = None
    relation_belief = None
    trace = []

    for index, view in enumerate(config["view_order"], start=1):
        input_path = ROOT / config["input_root"] / view / config["input_filename"]
        graph = load_json(input_path)
        observed_target = binary_target_distribution(graph)
        observed_relation = relation_distribution(graph)
        previous_target_entropy = entropy(target_belief) if target_belief else None
        previous_relation_entropy = entropy(relation_belief) if relation_belief else None

        target_belief = fuse_distributions(target_belief, observed_target, floor)
        relation_belief = fuse_distributions(relation_belief, observed_relation, floor)
        target_id = max(target_belief, key=target_belief.get)
        relation_id = max(relation_belief, key=relation_belief.get)
        target_entropy = entropy(target_belief)
        relation_entropy = entropy(relation_belief)
        ready = (
            target_belief[target_id] >= gate["target_probability_minimum"]
            and relation_belief[gate["required_relation"]]
            >= gate["relation_probability_minimum"]
        )
        trace.append(
            {
                "step": index,
                "view_id": view,
                "input_graph": str(input_path.relative_to(ROOT)).replace("\\", "/"),
                "observed_target_distribution": observed_target,
                "observed_relation_distribution": observed_relation,
                "fused_target_distribution": rounded(target_belief),
                "fused_relation_distribution": rounded(relation_belief),
                "most_likely_target": target_id,
                "most_likely_relation": relation_id,
                "target_entropy_nats": round(target_entropy, 8),
                "relation_entropy_nats": round(relation_entropy, 8),
                "target_entropy_reduction": (
                    None
                    if previous_target_entropy is None
                    else round(previous_target_entropy - target_entropy, 8)
                ),
                "relation_entropy_reduction": (
                    None
                    if previous_relation_entropy is None
                    else round(previous_relation_entropy - relation_entropy, 8)
                ),
                "temporary_execution_gate_passed": ready,
            }
        )

    output_dir = ROOT / config["output_root"]
    output_dir.mkdir(parents=True, exist_ok=True)
    result = {
        "status": config["status"],
        "method": config["fusion"]["method"],
        "assumption": config["fusion"]["assumption"],
        "calibrated": False,
        "allowed_for_final_evaluation": config["allowed_for_final_evaluation"],
        "view_order": config["view_order"],
        "temporary_execution_gate": gate,
        "trace": trace,
        "provenance": {
            "ground_truth_used": True,
            "notes": (
                "Inputs come from the rule-based stub. This trace tests temporal "
                "data flow and must not be reported as final model performance."
            ),
        },
    }
    json_path = output_dir / "multi_view_belief_trace.json"
    with json_path.open("w", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2, ensure_ascii=False)
        stream.write("\n")

    csv_path = output_dir / "multi_view_belief_trace.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=[
                "step",
                "view_id",
                "target_probability",
                "inside_probability",
                "target_entropy_nats",
                "relation_entropy_nats",
                "target_entropy_reduction",
                "relation_entropy_reduction",
                "temporary_execution_gate_passed",
            ],
        )
        writer.writeheader()
        for step in trace:
            writer.writerow(
                {
                    "step": step["step"],
                    "view_id": step["view_id"],
                    "target_probability": step["fused_target_distribution"][
                        step["most_likely_target"]
                    ],
                    "inside_probability": step["fused_relation_distribution"]["inside"],
                    "target_entropy_nats": step["target_entropy_nats"],
                    "relation_entropy_nats": step["relation_entropy_nats"],
                    "target_entropy_reduction": step["target_entropy_reduction"],
                    "relation_entropy_reduction": step["relation_entropy_reduction"],
                    "temporary_execution_gate_passed": step[
                        "temporary_execution_gate_passed"
                    ],
                }
            )

    print(f"WROTE={json_path}")
    print(f"WROTE={csv_path}")
    for step in trace:
        target_p = step["fused_target_distribution"][step["most_likely_target"]]
        inside_p = step["fused_relation_distribution"]["inside"]
        print(
            f"STEP={step['step']} VIEW={step['view_id']} "
            f"TARGET_P={target_p:.6f} INSIDE_P={inside_p:.6f} "
            f"READY={step['temporary_execution_gate_passed']}"
        )


if __name__ == "__main__":
    main()
