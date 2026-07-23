"""Add provisional multi-entity beliefs to benchmark Scene Graphs.

The probabilities are deterministic engineering stubs for interface testing.
They are not calibrated VLM outputs or an approved paper method.
"""

import argparse
import copy
import json
import math
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs/scene_graph/benchmark_probability_stub.json"
VIEWS = ("left", "center", "right")


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def normalize(values: dict[str, float], floor: float) -> dict[str, float]:
    clipped = {key: max(floor, value) for key, value in values.items()}
    total = sum(clipped.values())
    return {key: value / total for key, value in clipped.items()}


def entropy(distribution: dict[str, float]) -> float:
    return -sum(p * math.log(p) for p in distribution.values() if p > 0.0)


def uncertainty(distribution: dict[str, float], method: str) -> dict:
    return {
        "method": method,
        "value": round(entropy(distribution), 8),
        "calibrated": False,
    }


def confidence_from_pixels(pixel_count: int, scale: float) -> float:
    return 0.5 + 0.49 * (1.0 - math.exp(-pixel_count / scale))


def build_uncertainty_graph(graph: dict, config: dict) -> dict:
    result = copy.deepcopy(graph)
    node_by_id = {node["id"]: node for node in result["nodes"]}
    settings = config["visibility"]
    raw_target_scores = {}
    for object_id in config["target_candidates"]:
        pixels = node_by_id[object_id]["observation"]["pixel_count"]
        evidence = settings["minimum_evidence"] + min(
            1.0, pixels / settings["target_pixel_saturation"]
        )
        raw_target_scores[object_id] = (
            config["appearance_prior"][object_id]
            * config["task_relation_compatibility"][object_id]
            * evidence
        )
    target_distribution = normalize(raw_target_scores, config["probability_floor"])

    for object_id in config["target_candidates"] + ["container"]:
        node = node_by_id[object_id]
        pixels = node["observation"]["pixel_count"]
        existence_probability = confidence_from_pixels(
            pixels, settings["existence_pixel_scale"]
        )
        node["belief"] = {
            "existence_probability": round(existence_probability, 8),
            "existence_uncertainty": uncertainty(
                {"exists": existence_probability, "absent": 1.0 - existence_probability},
                config["uncertainty_method"],
            ),
            "target_probability": round(target_distribution.get(object_id, 0.0), 8),
            "target_uncertainty": uncertainty(
                target_distribution, config["uncertainty_method"]
            ),
            "source": "rule_based_visibility_and_config_stub",
        }

    relation_edges = []
    for edge in result["edges"]:
        if edge["relation"] == "visible_from":
            continue
        source_observation = node_by_id[edge["source"]]["observation"]
        confidence = confidence_from_pixels(
            source_observation["pixel_count"], settings["relation_pixel_scale"]
        )
        distribution = {
            edge["relation"]: confidence,
            "unknown": 1.0 - confidence,
        }
        edge["belief"] = {
            "relation_distribution": {
                key: round(value, 8) for key, value in distribution.items()
            },
            "relation_uncertainty": uncertainty(
                distribution, config["uncertainty_method"]
            ),
            "source": "rule_based_visibility_stub",
        }
        relation_edges.append(edge)

    target_relation_edge = next(
        edge for edge in relation_edges if edge["source"] == "target_red"
    )
    inside_probability = target_relation_edge["belief"]["relation_distribution"]["inside"]
    target_probability = target_distribution["target_red"]
    result["schema_version"] = "0.2.0-benchmark-uncertainty-stub"
    result["graph_belief"] = {
        "target_distribution": {
            key: round(value, 8) for key, value in target_distribution.items()
        },
        "target_uncertainty": uncertainty(
            target_distribution, config["uncertainty_method"]
        ),
        "required_relation": "inside",
        "required_relation_probability": inside_probability,
        "required_relation_uncertainty": target_relation_edge["belief"][
            "relation_uncertainty"
        ],
        "task_failure_risk": round(1.0 - target_probability * inside_probability, 8),
        "risk_definition": "1-P(target_red)*P(target_red inside container)_stub",
    }
    result["provenance"]["probability_source"] = (
        "deterministic_rule_based_engineering_stub"
    )
    result["provenance"]["valid_for_final_evaluation"] = False
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    args, _unknown = parser.parse_known_args()
    config = load_json(args.config)
    root = ROOT / config["observation_root"]
    for view in VIEWS:
        graph = build_uncertainty_graph(
            load_json(root / view / "scene_graph.json"), config
        )
        output = root / view / "uncertainty_scene_graph_stub.json"
        output.write_text(json.dumps(graph, indent=2) + "\n", encoding="utf-8")
        belief = graph["graph_belief"]
        print(
            f"VIEW={view} TARGET_P={belief['target_distribution']['target_red']:.6f} "
            f"INSIDE_P={belief['required_relation_probability']:.6f} "
            f"RISK={belief['task_failure_risk']:.6f}"
        )


if __name__ == "__main__":
    main()
