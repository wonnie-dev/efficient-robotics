"""Build one observation Scene Graph JSON for each captured camera view."""

import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = PROJECT_ROOT / "configs" / "sim" / "open_container_minimal.json"
OBSERVATION_ROOT = PROJECT_ROOT / "outputs" / "observations"
VIEWS = ("left", "center", "right")


def read_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as stream:
        return json.load(stream)


def make_object_node(object_id: str, object_type: str, observation: dict, config_object=None) -> dict:
    node = {
        "id": object_id,
        "type": object_type,
        "observation": {
            "visible": observation["visible"],
            "pixel_count": observation["pixel_count"],
            "visible_fraction": observation["visible_fraction"],
            "bbox_xyxy": observation["bbox_xyxy"],
            "depth_mean_m": observation["depth_mean_m"],
            "depth_min_m": observation["depth_min_m"],
            "depth_max_m": observation["depth_max_m"],
        },
    }
    if config_object is not None:
        node["ground_truth"] = {
            "shape": config_object["shape"],
            "size_m": config_object["size_m"],
            "position_m": config_object["position_m"],
        }
    return node


def build_graph(view: str, scene_config: dict, observations: dict) -> dict:
    config_objects = {item["id"]: item for item in scene_config["objects"]}
    nodes = [
        {
            "id": f"camera_view_{view}",
            "type": "observation_view",
            "pose_name": view,
        },
        make_object_node("container", "open_container", observations["container"]),
        make_object_node(
            "target_red", "target_candidate", observations["target_red"], config_objects["target_red"]
        ),
        make_object_node(
            "distractor_blue",
            "distractor",
            observations["distractor_blue"],
            config_objects["distractor_blue"],
        ),
    ]

    edges = [
        {
            "source": "target_red",
            "relation": "inside",
            "target": "container",
            "value": True,
            "source_type": "ground_truth_config",
        },
        {
            "source": "distractor_blue",
            "relation": "outside_near",
            "target": "container",
            "value": True,
            "source_type": "ground_truth_config",
        },
    ]
    for object_id in ("target_red", "distractor_blue", "container"):
        edges.append(
            {
                "source": object_id,
                "relation": "visible_from",
                "target": f"camera_view_{view}",
                "value": observations[object_id]["visible"],
                "source_type": "camera_observation",
            }
        )

    return {
        "schema_version": "0.1.0",
        "scene_id": scene_config["scene_id"],
        "view": view,
        "seed": scene_config["seed"],
        "nodes": nodes,
        "edges": edges,
        "provenance": {
            "observations": f"outputs/observations/{view}/objects.json",
            "relations": "configs/sim/open_container_minimal.json",
            "segmentation_source": "rgb_color_key_fallback",
        },
    }


def main() -> None:
    scene_config = read_json(CONFIG_PATH)
    for view in VIEWS:
        view_dir = OBSERVATION_ROOT / view
        graph = build_graph(view, scene_config, read_json(view_dir / "objects.json"))
        output_path = view_dir / "scene_graph.json"
        output_path.write_text(json.dumps(graph, indent=2), encoding="utf-8")
        print(f"SCENE_GRAPH={output_path}")


if __name__ == "__main__":
    main()
