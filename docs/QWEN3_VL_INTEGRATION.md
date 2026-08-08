# Qwen3-VL Integration

## Model

The language-conditioned target selector uses
`Qwen/Qwen3-VL-8B-Instruct` with pretrained inference only. The reference
checkpoint revision is:

```text
0c351dd01ed87e9c1b53cbc748cba10e6187ff3b
```

Model weights and caches are stored outside Git. The model path is supplied at
runtime with `--model-path` or `EFFICIENT_ROBOTICS_QWEN_MODEL`.

## Input

The adapter consumes the shared VLM contract:

- the unmodified RGB observation;
- an overlay with anonymous candidate IDs and bounding boxes;
- anonymous candidate crops and masks;
- the language instruction;
- an ordered set of target or relation choices.

Simulator object names and ground-truth labels are not provided to the model.

## Forced-choice scoring

Each requested class is mapped to a single-token letter. The adapter records
the corresponding next-token language-model logit before softmax. To reduce
letter-position bias, it cyclically permutes the labels across the available
letters and averages each class score across permutations.

Prompt version:

```text
qwen3-vl-forced-choice-v3-permutation-debiased
```

These values are raw ranking scores, not probabilities. Calibration is fitted
only on episode-disjoint calibration data.

## Inference

Run one model instance with a single visible CUDA device:

```bash
python scripts/qwen3_vl_logits.py \
  outputs/vlm_dataset/samples/example/input.json \
  --model-path /path/to/Qwen3-VL-8B-Instruct
```

The adapter rejects distributed initialization and more than one visible CUDA
device. It uses BF16 inference, PyTorch SDPA, and batch size one.

## Output and caching

The output follows `configs/vlm/vlm_output.schema.json` and records model
revision, prompt version, raw logits, input hashes, runtime, and peak CUDA
memory. Cache keys include the image and mask hashes, normalized prompt,
checkpoint revision, and inference settings so identical observations are not
processed twice.

See [VLM Interface](VLM_INTERFACE.md) for the complete data contract.
