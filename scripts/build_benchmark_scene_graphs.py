"""Build deterministic multi-object Scene Graphs for benchmark observations."""

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "configs/sim/open_container_benchmark.json"
OBSERVATION_ROOT = ROOT / "outputs/benchmark_observations"
VIEWS = ("left", "center", "right")


def read_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as stream:
        return json.load(stream)


def observation_node(object_config: dict, observation: dict) -> dict:
    return {
        "id": object_config["id"],
        "type": object_config["role"],
        "shape": object_config["shape"],
        "observation": {
            "visible": observation["visible"],
            "pixel_count": observation["pixel_count"],
            "visible_fraction": observation["visible_fraction"],
            "bbox_xyxy": observation["bbox_xyxy"],
            "depth_mean_m": observation["depth_mean_m"],
            "depth_min_m": observation["depth_min_m"],
            "depth_max_m": observation["depth_max_m"],
        },
        "ground_truth": {
            "prim": object_config["prim"],
            "position_m": object_config["position_m"],
        },
    }


def build_graph(view: str, config: dict, observations: dict) -> dict:
    nodes = [
        {"id": f"camera_view_{view}", "type": "observation_view", "pose_name": view},
        {
            "id": "container",
            "type": "open_container",
            "observation": observations["container"],
            "ground_truth": {"prim": "/World/OpenContainer"},
        },
    ]
    nodes.extend(
        observation_node(object_config, observations[object_config["id"]])
        for object_config in config["objects"]
    )

    edges = []
    for object_config in config["objects"]:
        relation = object_config["relation"]
        if object_config["id"] == "occluder_orange":
            edges.append(
                {
                    "source": "occluder_orange",
                    "relation": "occludes",
                    "target": "target_red",
                    "value": True,
                    "source_type": "ground_truth_config",
                }
            )
        else:
            edges.append(
                {
                    "source": object_config["id"],
                    "relation": relation,
                    "target": "container",
                    "value": True,
                    "source_type": "ground_truth_config",
                }
            )
    for object_id, observation in observations.items():
        edges.append(
            {
                "source": object_id,
                "relation": "visible_from",
                "target": f"camera_view_{view}",
                "value": observation["visible"],
                "source_type": "benchmark_color_id_observation",
            }
        )

    return {
        "schema_version": "0.1.0-benchmark",
        "scene_id": config["scene_id"],
        "scenario": config["scenario"],
        "view": view,
        "seed": config["seed"],
        "nodes": nodes,
        "edges": edges,
        "provenance": {
            "observations": f"outputs/benchmark_observations/{view}/objects.json",
            "relations": "configs/sim/open_container_benchmark.json",
            "segmentation_source": "temporary_unique_color_id_render_pass",
            "ground_truth_relations_used": True,
            "valid_for_final_evaluation": False,
        },
    }


def main() -> None:
    config = read_json(CONFIG_PATH)
    for view in VIEWS:
        view_dir = OBSERVATION_ROOT / view
        graph = build_graph(view, config, read_json(view_dir / "objects.json"))
        output_path = view_dir / "scene_graph.json"
        output_path.write_text(json.dumps(graph, indent=2), encoding="utf-8")
        print(f"BENCHMARK_SCENE_GRAPH={output_path}")


if __name__ == "__main__":
    main()
