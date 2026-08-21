# Model Setup

## Models

| Role | Model | Use |
| --- | --- | --- |
| Target ranking and language-conditioned choice | `Qwen/Qwen3-VL-8B-Instruct` | pretrained inference only |
| Open-vocabulary proposals | GroundingDINO-Base | pretrained inference only |
| Candidate masks | SAM2.1-Large | pretrained inference only |

The Qwen revision used by the current calibration is:

```text
0c351dd01ed87e9c1b53cbc748cba10e6187ff3b
```

The current grounding checkpoints are pinned to:

```text
IDEA-Research/grounding-dino-base@12bdfa3120f3e7ec7b434d90674b3396eccf88eb
facebook/sam2.1-hiera-large@665f8e2ad61cf5f53d65644ff27c8ee525124610
Grounded-SAM-2 source@b7a9c29f196edff0eb54dbe14588d7ae5e3dde28
```

No fine-tuning, LoRA, or foundation-model weight update is used.

## Environments

Keep Isaac Sim, Qwen, and grounding dependencies separate.

Qwen environment:

```bash
python3 -m venv /path/to/efficient-robotics-vlm
source /path/to/efficient-robotics-vlm/bin/activate
pip install --upgrade pip
pip install -r requirements/vlm-qwen3-vl.txt
```

Grounding environment:

```bash
python3 -m venv /path/to/efficient-robotics-perception
source /path/to/efficient-robotics-perception/bin/activate
pip install --upgrade pip
pip install -r requirements/perception.txt
```

Install the official Grounded-SAM-2 source at the pinned commit into the grounding environment. Record local checkpoint hashes in each paper run because weights are stored outside Git.

## Inference

Model paths, caches, and compute-device assignments are deployment-specific and
must not be hard-coded or committed. Qwen runs in BF16 with one model instance,
batch size one, and no distributed runtime.

On an RTX A6000, Qwen scoring has used approximately `16.7-17.2 GiB` of peak allocated memory and roughly `10-18 s` per observation after model loading. Grounding and segmentation add separate runtime and memory use because the models run sequentially.

## Input and output

The VLM receives an RGB image, anonymous candidate IDs, candidate masks or boxes, a target instruction, reference entities, and a closed relation vocabulary. It does not receive simulator object labels or ground truth.

The output contains raw target-choice logits, relation-choice logits, selected candidate ID, model revision, prompt version, input hash, and cache metadata. Raw logits are not probabilities. See [VLM Interface](VLM_INTERFACE.md).

Cache identity includes the image and mask hashes, normalized prompt payload, model revision, prompt version, and inference settings. Repeated requests reuse an exact cache match.
