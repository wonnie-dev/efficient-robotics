"""Validate cross-file invariants in the VLM input/output contract."""

import argparse
import json
from pathlib import Path


FORBIDDEN_INPUT_TOKENS = (
    "target_red",
    "rear_red_candidate",
    "distractor_",
    "occluder_orange",
    "boundary_purple",
)


def validate_contract(model_input: dict, model_output: dict) -> None:
    if model_input["schema_version"] != "vlm-input-v1":
        raise ValueError("Unsupported VLM input schema")
    if model_output["schema_version"] != "vlm-output-v1":
        raise ValueError("Unsupported VLM output schema")
    if model_input["sample_id"] != model_output["sample_id"]:
        raise ValueError("Input and output sample IDs differ")
    serialized_input = json.dumps(model_input).lower()
    leaked = [token for token in FORBIDDEN_INPUT_TOKENS if token in serialized_input]
    if leaked:
        raise ValueError(f"Semantic ground-truth leakage in model input: {leaked}")
    candidate_ids = [
        candidate["candidate_id"] for candidate in model_input["candidates"]
    ]
    target = model_output["target"]
    if target["candidate_ids"] != candidate_ids:
        raise ValueError("Target output candidate order differs from input")
    if len(target["raw_logits"]) != len(candidate_ids):
        raise ValueError("Target logit dimension differs from candidate count")
    query_by_id = {
        query["query_id"]: query for query in model_input["relation_queries"]
    }
    output_query_ids = set()
    for relation in model_output["relations"]:
        query_id = relation["query_id"]
        if query_id not in query_by_id:
            raise ValueError(f"Unknown relation query: {query_id}")
        if relation["labels"] != query_by_id[query_id]["label_space"]:
            raise ValueError(f"Relation label order differs for {query_id}")
        if len(relation["raw_logits"]) != len(relation["labels"]):
            raise ValueError(f"Relation logit dimension differs for {query_id}")
        output_query_ids.add(query_id)
    if output_query_ids != set(query_by_id):
        raise ValueError("VLM output does not cover every relation query")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    validate_contract(
        json.loads(args.input.read_text(encoding="utf-8")),
        json.loads(args.output.read_text(encoding="utf-8")),
    )
    print("VLM_CONTRACT_VALID")


if __name__ == "__main__":
    main()
