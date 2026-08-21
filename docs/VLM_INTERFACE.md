# VLM Interface

## Purpose

The interface allows different pretrained VLMs to score the same anonymous
object candidates and relation choices without changing the Scene Graph or
planner code.

## Leakage boundary

Inference inputs use anonymous candidate IDs such as `object_001`. Simulator
names that reveal semantic identity are excluded. Ground truth is stored in a
separate file and is available only to calibration and evaluation code.

The language instruction is part of the task definition and is provided to the
model.

## Input

Each `input.json` contains:

- an RGB observation;
- the natural-language instruction;
- anonymous candidate IDs and bounding boxes;
- candidate crops and binary masks;
- reference entities;
- relation queries with ordered categorical label spaces.

The schema is `configs/vlm/vlm_input.schema.json`.

## Output

Each implementation returns:

- one raw target-choice score for every candidate in input order;
- one raw score for every label in each relation query;
- model repository and checkpoint revision;
- prompt version and input hashes;
- runtime, device, and parse status.

Pre-softmax logits are preferred because calibration requires a stable score
space. Generated statements such as “80% confident” are not treated as
probabilities. The schema is `configs/vlm/vlm_output.schema.json`.

## Ground truth

The separate ground-truth record contains the correct anonymous target and one
categorical label per relation query. It is never included in the model prompt
or root-action selection. Its schema is
`configs/vlm/vlm_ground_truth.schema.json`.

## Contract validation

Export a sample, produce an output with either a real adapter or the synthetic
interface fixture, and validate it:

```bash
python scripts/export_vlm_dataset.py
python scripts/synthetic_vlm_output.py \
  outputs/vlm_dataset/samples/example/input.json
python scripts/validate_vlm_contract.py \
  outputs/vlm_dataset/samples/example/input.json \
  outputs/vlm_dataset/samples/example/synthetic_output.json
```

The synthetic fixture checks file compatibility only and is not a perception
baseline.

## Qwen3-VL adapter

The reference implementation uses `Qwen/Qwen3-VL-8B-Instruct` and
permutation-averaged forced-choice logits. See
[Qwen3-VL Integration](QWEN3_VL_INTEGRATION.md) for the scoring procedure and
checkpoint revision.

## Data splits

Model weights remain frozen. Project-specific simulator episodes are used for
calibration and evaluation, with all views from one episode kept in the same
split. Ground-truth labels are generated from simulator state; a manually
annotated VLM training dataset is not required.
