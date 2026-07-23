"""Score target visibility and select an information-gathering viewpoint."""

import csv
import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
POLICY_PATH = PROJECT_ROOT / "configs" / "policy" / "viewpoint_selection.json"
OBSERVATION_ROOT = PROJECT_ROOT / "outputs" / "observations"
OUTPUT_ROOT = PROJECT_ROOT / "outputs" / "viewpoint_selection"
VIEWS = ("left", "center", "right")


def read_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as stream:
        return json.load(stream)


def target_observation(graph: dict, target_id: str) -> dict:
    return next(node["observation"] for node in graph["nodes"] if node["id"] == target_id)


def score_view(view: str, graph: dict, objects: dict, policy: dict) -> dict:
    observation = target_observation(graph, policy["target_node_id"])
    raw_object = objects[policy["target_node_id"]]
    pixel_count = int(observation["pixel_count"])
    depth_valid_pixels = int(raw_object["depth_valid_pixels"])
    area_score = min(pixel_count / float(policy["reference_target_pixels"]), 1.0)
    depth_validity = depth_valid_pixels / pixel_count if pixel_count else 0.0
    visible_gate = 1.0 if observation["visible"] else 0.0
    score = visible_gate * (
        policy["weights"]["target_area"] * area_score
        + policy["weights"]["depth_validity"] * depth_validity
    )
    return {
        "view": view,
        "score": round(score, 6),
        "uncertainty": round(1.0 - score, 6),
        "visible": observation["visible"],
        "pixel_count": pixel_count,
        "target_area_score": round(area_score, 6),
        "depth_validity": round(depth_validity, 6),
        "depth_mean_m": observation["depth_mean_m"],
    }


def make_decision(scores: dict, policy: dict) -> dict:
    initial_view = policy["initial_view"]
    initial_score = scores[initial_view]["score"]
    threshold = policy["sufficient_score_threshold"]
    if initial_score >= threshold:
        action = "keep_initial_view"
        selected_view = initial_view
        reason = "initial_view_score_is_sufficient"
    else:
        action = "change_viewpoint"
        selected_view = max(policy["candidate_views"], key=lambda view: scores[view]["score"])
        reason = "initial_view_uncertain_select_highest_scoring_candidate"
    return {
        "action": action,
        "initial_view": initial_view,
        "selected_view": selected_view,
        "threshold": threshold,
        "initial_score": initial_score,
        "selected_score": scores[selected_view]["score"],
        "expected_score_gain": round(scores[selected_view]["score"] - initial_score, 6),
        "reason": reason,
    }


def write_csv(scores: dict) -> None:
    with (OUTPUT_ROOT / "view_scores.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(scores["left"].keys()))
        writer.writeheader()
        writer.writerows(scores[view] for view in VIEWS)


def write_svg(scores: dict, decision: dict, threshold: float) -> None:
    width, height = 720, 440
    chart_top, chart_bottom = 60, 350
    bar_width = 110
    xs = {"left": 120, "center": 305, "right": 490}
    colors = {"left": "#4c78a8", "center": "#f2a541", "right": "#59a14f"}
    threshold_y = chart_bottom - threshold * (chart_bottom - chart_top)
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        '<text x="360" y="30" text-anchor="middle" font-size="20" font-family="sans-serif">Target visibility score by viewpoint</text>',
        f'<line x1="70" y1="{chart_bottom}" x2="660" y2="{chart_bottom}" stroke="#333"/>',
        f'<line x1="70" y1="{threshold_y:.1f}" x2="660" y2="{threshold_y:.1f}" stroke="#d62728" stroke-dasharray="7 5"/>',
        f'<text x="665" y="{threshold_y + 5:.1f}" font-size="12" font-family="sans-serif" fill="#d62728">threshold {threshold:.2f}</text>',
    ]
    for view in VIEWS:
        score = scores[view]["score"]
        bar_height = score * (chart_bottom - chart_top)
        y = chart_bottom - bar_height
        selected_marker = " [selected]" if view == decision["selected_view"] else ""
        parts.extend(
            [
                f'<rect x="{xs[view]}" y="{y:.1f}" width="{bar_width}" height="{bar_height:.1f}" fill="{colors[view]}"/>',
                f'<text x="{xs[view] + bar_width / 2}" y="{y - 8:.1f}" text-anchor="middle" font-size="16" font-family="sans-serif">{score:.3f}</text>',
                f'<text x="{xs[view] + bar_width / 2}" y="380" text-anchor="middle" font-size="16" font-family="sans-serif">{view}{selected_marker}</text>',
            ]
        )
    parts.append(
        f'<text x="360" y="420" text-anchor="middle" font-size="14" font-family="sans-serif">decision: {decision["action"]} to {decision["selected_view"]}</text>'
    )
    parts.append("</svg>")
    (OUTPUT_ROOT / "view_scores.svg").write_text("\n".join(parts), encoding="utf-8")


def main() -> None:
    policy = read_json(POLICY_PATH)
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    scores = {}
    for view in VIEWS:
        graph = read_json(OBSERVATION_ROOT / view / "scene_graph.json")
        objects = read_json(OBSERVATION_ROOT / view / "objects.json")
        scores[view] = score_view(view, graph, objects, policy)
    decision = make_decision(scores, policy)
    result = {
        "status": policy["status"],
        "policy": policy,
        "scores": scores,
        "decision": decision,
        "input_scene_graphs": [f"outputs/observations/{view}/scene_graph.json" for view in VIEWS],
    }
    (OUTPUT_ROOT / "decision.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    log_lines = [
        f"{view}: score={scores[view]['score']:.6f}, uncertainty={scores[view]['uncertainty']:.6f}, pixels={scores[view]['pixel_count']}"
        for view in VIEWS
    ]
    log_lines.append(
        f"decision: {decision['action']} from={decision['initial_view']} to={decision['selected_view']} gain={decision['expected_score_gain']:.6f}"
    )
    (OUTPUT_ROOT / "decision.log").write_text("\n".join(log_lines) + "\n", encoding="utf-8")
    write_csv(scores)
    write_svg(scores, decision, policy["sufficient_score_threshold"])
    print(json.dumps(decision, indent=2))


if __name__ == "__main__":
    main()
