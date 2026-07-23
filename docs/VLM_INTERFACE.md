# VLM Interface and Dataset Contract

## Purpose

This contract lets independent VLM implementations produce interchangeable
raw logits for the same Isaac Sim samples. The user's model and Hansol's model
must consume the same input files and emit the same output structure.

## Leakage boundary

Model inputs use anonymous candidate IDs such as `object_001`. Semantic
simulator names such as `target_red`, `rear_red_candidate`, and
`occluder_orange` are forbidden in inference inputs.

Ground truth is stored in a separate `ground_truth.json`. It must be used only
by training, calibration, and evaluation code and must never be passed to the
model inference prompt or planner.

The natural-language instruction is not leakage. Color and spatial constraints
in the instruction are the task specification the VLM is expected to ground.

## Input

Each `input.json` contains:

- the full RGB image;
- the natural-language instruction;
- anonymous candidate IDs;
- candidate bounding boxes;
- masked RGB crop paths;
- binary mask paths;
- relation queries and their ordered categorical label spaces.

The input schema is `configs/vlm/vlm_input.schema.json`.

## Output

Each model must output:

- one raw target logit for every candidate, in the exact input order;
- one raw relation logit for every label of every relation query;
- model name and checkpoint;
- prompt version, weight hash, and device.

Do not output only softmax probabilities. Temperature scaling requires
pre-softmax logits. The output schema is
`configs/vlm/vlm_output.schema.json`.

## Ground truth

The separate file contains the correct anonymous target candidate and one
categorical label per relation query. Its schema is
`configs/vlm/vlm_ground_truth.schema.json`.

## Export and contract test

```powershell
D:\isaac-sim\python.bat scripts\export_vlm_dataset.py
D:\isaac-sim\python.bat scripts\mock_vlm_logits.py `
  outputs\vlm_dataset\samples\benchmark_seed000_center\input.json
D:\isaac-sim\python.bat scripts\validate_vlm_contract.py `
  outputs\vlm_dataset\samples\benchmark_seed000_center\input.json `
  outputs\vlm_dataset\samples\benchmark_seed000_center\mock_output.json
```

Generated data is under `outputs/vlm_dataset/` and is intentionally ignored by
Git. The current three left/center/right samples are development examples only.
They are not a valid train/calibration/test dataset because they come from one
fixed scene and seed.

## Handoff requirement

Any VLM implementation is compatible when this command succeeds:

```powershell
D:\isaac-sim\python.bat scripts\validate_vlm_contract.py INPUT_JSON OUTPUT_JSON
```

The deterministic mock adapter tests file compatibility only. It has no
perception ability and must not be used as a baseline result.

## Data strategy

The project will not manually build and label a large VLM training dataset.
Model training or adaptation should start from pretrained weights and suitable
existing public datasets.

Project-specific data is still required for calibration and evaluation because
the robot, camera, object candidates, relations, and active-view actions differ
from public datasets. These samples will be generated and labeled
automatically by Isaac Sim rather than manually annotated. They must be divided
by scene seed and episode into disjoint train/adaptation (if needed),
calibration, validation, and test groups. Different views of the same episode
must never be split across calibration and test.
