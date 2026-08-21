"""Validate project-specific invariants in an uncertainty-aware Scene Graph JSON."""

import json
import math
import sys
from pathlib import Path


REQUIRED_TOP_LEVEL = {
    "schema_version",
    "schema_status",
    "scene_id",
    "task",
    "observation",
    "nodes",
    "edges",
    "graph_belief",
    "provenance",
}


def validate_distribution(name: str, distribution: dict) -> None:
    if not distribution:
        raise ValueError(f"{name} must not be empty")
    for label, probability in distribution.items():
        if not isinstance(probability, (int, float)) or not 0.0 <= probability <= 1.0:
            raise ValueError(f"{name}.{label} is not a probability: {probability}")
    total = sum(distribution.values())
    if not math.isclose(total, 1.0, abs_tol=1e-6):
        raise ValueError(f"{name} must sum to 1.0, got {total}")


def validate_uncertainty(name: str, record: dict) -> None:
    if record["status"] == "pending_definition" and record["value"] is not None:
        raise ValueError(f"{name} cannot have a value while pending definition")
    if record["value"] is not None and not record["method"]:
        raise ValueError(f"{name} requires a named method when value is populated")
    if record["calibrated"] and record["value"] is None:
        raise ValueError(f"{name} cannot be calibrated without a value")


def validate(graph: dict) -> None:
    missing = REQUIRED_TOP_LEVEL - graph.keys()
    if missing:
        raise ValueError(f"Missing top-level fields: {sorted(missing)}")
    if graph["schema_version"] != "0.2.0-draft":
        raise ValueError("Unexpected schema_version")

    node_ids = set()
    for node in graph["nodes"]:
        if node["id"] in node_ids:
            raise ValueError(f"Duplicate node id: {node['id']}")
        node_ids.add(node["id"])
        validate_distribution(
            f"node[{node['id']}].class_distribution", node["belief"]["class_distribution"]
        )
        target_probability = node["belief"]["target_probability"]
        if not 0.0 <= target_probability <= 1.0:
            raise ValueError(f"Invalid target probability on {node['id']}")
        validate_uncertainty(
            f"node[{node['id']}].target_uncertainty", node["belief"]["target_uncertainty"]
        )

    for edge in graph["edges"]:
        if edge["source"] not in node_ids or edge["target"] not in node_ids:
            raise ValueError(f"Edge references missing node: {edge['id']}")
        validate_distribution(
            f"edge[{edge['id']}].relation_distribution", edge["belief"]["relation_distribution"]
        )
        validate_uncertainty(
            f"edge[{edge['id']}].relation_uncertainty", edge["belief"]["relation_uncertainty"]
        )

    for field in ("target_uncertainty", "relation_uncertainty", "task_failure_risk"):
        validate_uncertainty(f"graph_belief.{field}", graph["graph_belief"][field])
    joint = graph["graph_belief"].get("joint_task_state_distribution")
    if joint is not None:
        validate_distribution(
            "graph_belief.joint_task_state_distribution", joint
        )


def main() -> None:
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("examples/uncertainty_scene_graph.example.json")
    with path.open("r", encoding="utf-8") as stream:
        graph = json.load(stream)
    validate(graph)
    print(f"VALID={path}")


if __name__ == "__main__":
    main()
