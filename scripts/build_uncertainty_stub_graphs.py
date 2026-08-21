"""Build provisional rule-based uncertainty graphs for interface testing.

This uses configured object identities and the configured nominal target
relation. Outputs are not learned, calibrated, or final research results.
"""

import argparse
import json
from pathlib import Path

from validate_uncertainty_scene_graph import validate


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs/scene_graph/rule_based_stub.json"
SCENE_CONFIG = ROOT / "configs/sim/open_container_minimal.json"
SCHEMA_PATH = "configs/scene_graph/uncertainty_aware_scene_graph.schema.json"


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as stream:
        return json.load(stream)


def probability_from_visibility(visible_fraction: float, settings: dict) -> float:
    value = settings["base"] + settings["visible_fraction_gain"] * visible_fraction
    return round(max(settings["minimum"], min(settings["maximum"], value)), 6)


def uncertainty(confidence: float, settings: dict) -> dict:
    return {
        "status": "available",
        "method": settings["method"],
        "value": round(1.0 - confidence, 6),
        "calibrated": settings["calibrated"],
        "units": settings["units"],
    }


def pending_uncertainty() -> dict:
    return {
        "status": "pending_definition",
        "method": None,
        "value": None,
        "calibrated": False,
        "units": None,
    }


def node_observation(observation: dict) -> dict:
    return {
        "visible": observation["visible"],
        "visible_fraction": observation["visible_fraction"],
        "bbox_xyxy": observation["bbox_xyxy"],
        "depth_mean_m": observation["depth_mean_m"],
    }


def relation_distribution(primary: str, confidence: float, labels: list[str]) -> dict:
    others = [label for label in labels if label != primary]
    share = (1.0 - confidence) / len(others)
    distribution = {label: share for label in others}
    distribution[primary] = confidence
    return distribution


def build_graph(view: str, sequence_index: int, config: dict, scene: dict, objects: dict) -> dict:
    target_id, distractor_id = config["target_probability"]["candidate_ids"]
    target_observation = objects[target_id]
    target_probability = probability_from_visibility(
        target_observation["visible_fraction"], config["target_probability"]
    )
    distractor_probability = round(1.0 - target_probability, 6)

    relation_settings = config["relation_probability"]
    relation_probability = probability_from_visibility(
        target_observation["visible_fraction"], relation_settings
    )
    configured_target = next(item for item in scene["objects"] if item["id"] == target_id)
    nominal_relation = configured_target["relation"]
    uncertainty_settings = config["uncertainty"]

    nodes = [
        {
            "id": target_id,
            "type": "object",
            "observation": node_observation(objects[target_id]),
            "belief": {
                "class_distribution": {"red_cube": 0.95, "other": 0.05},
                "target_probability": target_probability,
                "target_uncertainty": uncertainty(target_probability, uncertainty_settings),
                "source": "rule_based_stub",
            },
            "last_observed_view": view,
        },
        {
            "id": distractor_id,
            "type": "object",
            "observation": node_observation(objects[distractor_id]),
            "belief": {
                "class_distribution": {"blue_cube": 0.95, "other": 0.05},
                "target_probability": distractor_probability,
                "target_uncertainty": uncertainty(
                    1.0 - distractor_probability, uncertainty_settings
                ),
                "source": "rule_based_stub",
            },
            "last_observed_view": view,
        },
        {
            "id": "container",
            "type": "container",
            "observation": node_observation(objects["container"]),
            "belief": {
                "class_distribution": {"open_container": 1.0},
                "target_probability": 0.0,
                "target_uncertainty": uncertainty(1.0, uncertainty_settings),
                "source": "ground_truth_stub",
            },
            "last_observed_view": view,
        },
    ]
    edge = {
        "id": f"{target_id}_to_container_relation",
        "source": target_id,
        "target": "container",
        "type": "spatial_relation_belief",
        "belief": {
            "relation_distribution": relation_distribution(
                nominal_relation, relation_probability, relation_settings["labels"]
            ),
            "relation_uncertainty": uncertainty(relation_probability, uncertainty_settings),
            "source": "rule_based_stub",
        },
        "last_updated_view": view,
    }
    return {
        "schema_version": "0.2.0-draft",
        "schema_status": "provisional_pending_overleaf_method_definition",
        "scene_id": scene["scene_id"],
        "episode_id": f"stub_{view}_seed_{scene['seed']}",
        "seed": scene["seed"],
        "task": config["task"],
        "observation": {
            "view_id": view,
            "sequence_index": sequence_index,
            "rgb_path": f"outputs/observations/{view}/rgb.png",
            "depth_path": f"outputs/observations/{view}/depth_m.npy",
            "camera_prim": scene["robot"]["camera_prim"],
            "timestamp_seconds": None,
        },
        "nodes": nodes,
        "edges": [edge],
        "graph_belief": {
            "most_likely_target": target_id,
            "target_uncertainty": uncertainty(target_probability, uncertainty_settings),
            "relation_uncertainty": uncertainty(relation_probability, uncertainty_settings),
            "task_failure_risk": pending_uncertainty(),
            "observation_count": 1,
        },
        "provenance": {
            "schema_file": SCHEMA_PATH,
            "perception_source": (
                "rule-based visible_fraction stub with configured object identity "
                "and configured nominal relation"
            ),
            "ground_truth_used_for_control": True,
            "notes": (
                "Interface-test output only; not learned, calibrated, or valid for final evaluation."
            ),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    args = parser.parse_args()

    config = load_json(args.config)
    scene = load_json(SCENE_CONFIG)
    for view in config["views"]:
        observation_dir = ROOT / config["input_root"] / view
        graph = build_graph(
            view,
            config["sequence_indices"][view],
            config,
            scene,
            load_json(observation_dir / "objects.json"),
        )
        validate(graph)
        output_path = observation_dir / "uncertainty_scene_graph_stub.json"
        with output_path.open("w", encoding="utf-8") as stream:
            json.dump(graph, stream, indent=2, ensure_ascii=False)
            stream.write("\n")
        relation_p = graph["edges"][0]["belief"]["relation_distribution"]["inside"]
        print(
            f"WROTE={output_path} "
            f"target_p={graph['nodes'][0]['belief']['target_probability']:.6f} "
            f"relation_p={relation_p:.6f}"
        )


if __name__ == "__main__":
    main()
