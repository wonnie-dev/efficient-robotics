"""Generate deterministic contract-test logits without reading ground truth."""

import argparse
import hashlib
import json
from pathlib import Path


def stable_logit(sample_id: str, key: str) -> float:
    digest = hashlib.sha256(f"{sample_id}|{key}".encode("utf-8")).digest()
    integer = int.from_bytes(digest[:4], "big")
    return round((integer / (2**32 - 1)) * 4.0 - 2.0, 6)


def build_mock_output(model_input: dict) -> dict:
    sample_id = model_input["sample_id"]
    candidate_ids = [
        candidate["candidate_id"] for candidate in model_input["candidates"]
    ]
    return {
        "schema_version": "vlm-output-v1",
        "sample_id": sample_id,
        "model": {"name": "deterministic_contract_mock", "checkpoint": "none"},
        "target": {
            "candidate_ids": candidate_ids,
            "raw_logits": [
                stable_logit(sample_id, f"target:{candidate_id}")
                for candidate_id in candidate_ids
            ],
        },
        "relations": [
            {
                "query_id": query["query_id"],
                "labels": query["label_space"],
                "raw_logits": [
                    stable_logit(sample_id, f"{query['query_id']}:{label}")
                    for label in query["label_space"]
                ],
            }
            for query in model_input["relation_queries"]
        ],
        "provenance": {
            "prompt_version": "contract-mock-v1",
            "weights_hash": "not_applicable",
            "device": "cpu",
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    model_input = json.loads(args.input.read_text(encoding="utf-8"))
    output = args.output or args.input.with_name("mock_output.json")
    output.write_text(
        json.dumps(build_mock_output(model_input), indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"WROTE={output}")


if __name__ == "__main__":
    main()
