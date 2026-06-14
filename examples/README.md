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

## ⭐ Multimodal agent brain examples

These drive `TransformerModel` — a **local** HF model as the agent's brain —
through real content blocks. All GPU-verified with live inference.

### `multimodal_agent.py` — image content block → VLM agent
`Agent(model=TransformerModel("SmolVLM-256M"))` answers `"Green."` to a green
image passed as an `{"image": {...}}` content block. Multimodal is auto-detected.

### `multimodal_advanced.py` — video + tool-result image
- **Video**: a `{"video": {"fps": 2.0, "source": {"bytes": frames}}}` block →
  SmolVLM2 answers `"BRIGHTER."` (real frame timestamps via `VideoMetadata`).
- **Tool-result image** (the agentic-vision loop): a tool returns an image inside
  a `toolResult`; the VLM reasons over it next turn → `"Blue."`.

### `document_and_audio.py` — document block + audio round-trip
- **Document**: a `{"document": {...}}` block is flattened to text; the LM
  recovers a passphrase from it.
- **Audio (tool path)**: a real TTS→ASR round-trip (mms-tts → whisper-tiny)
  transcribes "the quick brown fox…" back word-for-word.

### `audio_content_block.py` — audio content block → audio-native model
Extends the Strands taxonomy with an `audio` block (`make_audio_block`) and feeds
it to a Qwen2-Audio model's feature extractor — audio *inside the conversation*.

### `omni_audio.py` — Qwen2.5-Omni: audio-in AND speech-out
The frontier. One any-to-any model hears audio and **speaks** its reply:
- audio-in (440 Hz tone) → `"It's a pure tone."`
- text-in → text + a real 24 kHz speech waveform via `model.get_last_audio()`;
  whisper re-transcribes Omni's own speech to confirm it's intelligible.

---

## `use_transformers` tool examples

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

### `multimodal_pipelines.py` — text, image & audio via `run`
text-generation, text/token classification, fill-mask, feature-extraction,
zero-shot, image-classification, **text-to-audio**, and a full **ASR round-trip**
(TTS → .wav → transcription) — all with tiny models, no big downloads.

### `vision_tasks.py` — structured vision outputs via `run`
object-detection (boxes), image-feature-extraction (embeddings),
image-classification, depth-estimation (depth-map PNG artifact), and
image-segmentation (masks).

### `smolvlm_image_text.py` — a real VLM via `run`
HuggingFaceTB/SmolVLM-256M-Instruct image-text-to-text; images are auto-folded
into chat content.

### `local_model_agent.py` — local brain via `TransformerModel`
Runs a local HF causal-LM as the agent's reasoning engine, paired with
`use_transformers` as a tool.

### `smoke.py` — fast E2E health gate
12 real checks (discovery + text/image/audio round-trip), no large downloads,
non-zero exit on failure:

```bash
PYTHONPATH=. python examples/smoke.py
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
