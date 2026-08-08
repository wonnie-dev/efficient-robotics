# Qwen3-VL-8B-Instruct Integration

## Scope

This integration connects the pretrained
`Qwen/Qwen3-VL-8B-Instruct` checkpoint to the shared `vlm-input-v1` and
`vlm-output-v1` contract. It is an initial perception implementation, not a
calibrated model or a final paper result.

The Isaac Sim environment remains unchanged. Model inference uses the separate
environment:

```text
/data/wonheekoh/venvs/efficient-robotics-vlm
```

The downloaded checkpoint is:

```text
/data/wonheekoh/models/Qwen3-VL-8B-Instruct
Hugging Face revision: 0c351dd01ed87e9c1b53cbc748cba10e6187ff3b
```

## GPU boundary

Always use the launcher rather than calling the adapter directly:

```bash
scripts/run_qwen3_vl_gpu5.sh INPUT_JSON
```

The launcher enforces:

- `CUDA_VISIBLE_DEVICES=5`;
- physical GPU 5 appears inside PyTorch as `cuda:0`;
- exactly one CUDA device is visible;
- no `device_map="auto"`, DataParallel, DDP, or distributed initialization;
- BF16 inference with PyTorch SDPA.

## Raw-score definition

The adapter does not use generated confidence text or post-softmax
probabilities. It maps every requested class to a single-token uppercase
letter and reads the corresponding next-token language-model logit.

Choice letters have non-semantic prior biases. To avoid treating those biases
as class evidence, the adapter cyclically permutes the labels so every class
occupies every letter position once, then averages its pre-softmax logits.
This score is still uncalibrated. Temperature scaling and conformal fitting
must use held-out, episode-separated calibration data.

Prompt version:

```text
qwen3-vl-forced-choice-v3-permutation-debiased
```

The visual prompt contains:

- the unmodified full RGB scene;
- a derived full-scene overlay of anonymous candidate IDs and input bboxes;
- every anonymous masked candidate crop;
- normalized bbox coordinates;
- the instruction or ordered relation query and its exact label space.

No `ground_truth.json` file is opened by the inference adapter.

## Reproduce the development smoke test

Generate benchmark observations on physical GPU 5:

```bash
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=5 \
  /data/wonheekoh/isaacsim_venv/bin/python \
  scripts/open_minimal_scene.py \
  --scene-profile benchmark \
  --headless \
  --renderer-gpu 5 \
  --physics-gpu 0
```

Export anonymous samples:

```bash
/data/wonheekoh/venvs/efficient-robotics-vlm/bin/python \
  scripts/export_vlm_dataset.py
```

Run one sample and validate the output:

```bash
scripts/run_qwen3_vl_gpu5.sh \
  outputs/vlm_dataset/samples/benchmark_seed000_right/input.json

/data/wonheekoh/venvs/efficient-robotics-vlm/bin/python \
  scripts/validate_vlm_contract.py \
  outputs/vlm_dataset/samples/benchmark_seed000_right/input.json \
  outputs/vlm_dataset/samples/benchmark_seed000_right/qwen3_vl_output.json
```

## Measured smoke-test result

On the RTX A6000, the final permutation-debiased adapter measured:

| View | Peak allocated VRAM | Scoring time | Target result | Relation result |
| --- | ---: | ---: | --- | ---: |
| center | 16.698 GiB | 12.808 s | `object_007` (incorrect) | 2/7 |
| right | 16.700 GiB | 13.067 s | `object_007` (incorrect) | 3/7 |

Model loading took approximately 8.3 seconds in each fresh process. On the
right view, the correct `object_001` target score rose to within 1.0 raw-logit
point of `object_007`, and the `object_001 inside container` relation became
correct. These two views come from one fixed seed and are only an interface
smoke test. They must not be reported as model accuracy, a baseline, or
calibration evidence.

## Next data step

Generate a 50--100 episode pilot with episode-level split isolation and
variation in candidate identity, object layout, viewpoint, height, distance,
occlusion severity, and lighting. Use that pilot to audit prompts and label
balance before producing the held-out calibration, validation, IID test, and
viewpoint-shift test sets.
