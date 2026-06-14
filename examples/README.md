# Examples

Real, end-to-end examples driving HuggingFace models through the single
`use_transformers` tool. Every example here has been run against live model
inference on GPU — not mocked.

Run any example with the repo on your path:

```bash
PYTHONPATH=. python examples/<name>.py
```

## Two layers, one tool

`use_transformers` exposes the entire transformers library two ways:

| Layer | Action | Use when | Examples |
|-------|--------|----------|----------|
| **High-level** | `run` | The task is a standard transformers pipeline (text, image, audio, zero-shot, detection, ASR, TTS, …). Inputs are paths/URLs/PIL/arrays; outputs are structured + media saved to disk. | `multimodal_pipelines.py` |
| **Low-level** | `call` | The model has a custom API (e.g. VLA `predict_action`) or you need raw `AutoProcessor` + `AutoModel` control. Load components, cache them, chain calls. | `molmoact_vla.py`, `openvla_vla.py` |

## The examples

### `multimodal_pipelines.py` — the `run` path
Text generation, sentiment, image classification, and zero-shot classification
with tiny models. The fastest way to see the tool work across modalities.

### `molmoact_vla.py` — Vision-Language-Action (MolmoAct2-SO100_101)
[allenai/MolmoAct2-SO100_101](https://huggingface.co/allenai/MolmoAct2-SO100_101)
predicts continuous robot actions from two camera views + a language task + joint
state. Driven through the `call` layer:

```python
use_transformers(action="call", target="AutoProcessor.from_pretrained",
                 parameters={"pretrained_model_name_or_path": REPO, "trust_remote_code": True},
                 cache_key="proc")
use_transformers(action="call", target="AutoModelForImageTextToText.from_pretrained",
                 parameters={..., "dtype": "bfloat16", "device_map": "cuda"}, cache_key="molmo")
use_transformers(action="call", target="cached:molmo.predict_action",
                 parameters={"processor": "cached:proc", "images": [...], "state": [...],
                             "norm_tag": "so100_so101_molmoact2", ...})
# → MolmoAct2ActionOutput.actions, shape [1, 30, 6]
```

### `openvla_vla.py` — Vision-Language-Action (OpenVLA-7b)
[openvla/openvla-7b](https://huggingface.co/openvla/openvla-7b) predicts a 7-DoF
action from a wrist-cam image + instruction. Notable because OpenVLA shipped for
transformers **4.40.1** and needs the compat layer (below) to run on 5.x. Shows
the `"**"` unpack key feeding a processed batch into `predict_action`:

```python
use_transformers(action="call", target="cached:ovla.predict_action",
                 parameters={"**": "cached:ovla_batch", "unnorm_key": "bridge_orig"})
# → 7-DoF action vector
```

## Legacy model compatibility (`strands_transformers.core.compat`)

Many `trust_remote_code` models were written for transformers 4.x and break on
5.x. `compat.apply()` (invoked automatically by the tool) patches the gaps so the
model's own code runs unchanged — no version pinning:

- **Moved tokenizer symbols** — `PaddingStrategy`, `TruncationStrategy`, … are
  re-exposed on `transformers.tokenization_utils`.
- **Removed `AutoModelForVision2Seq`** — recreated as an alias of
  `AutoModelForImageTextToText`, registered everywhere `auto_map` dispatch and
  `register_for_auto_class()` validation look.
- **`tie_weights()` signature drift** — legacy overrides are made kwarg-tolerant.
- **Hard `timm` version pins** — `compat.spoof_timm_version()` context manager.

These shims are generic: they help any 4.x-era custom-code model, not just OpenVLA.

## Using models as the agent's brain

You can also use a local HuggingFace causal-LM as the Strands model provider and
give it `use_transformers` as a tool — see the repo root `agent.py`.
