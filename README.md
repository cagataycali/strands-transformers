<div align="center">
  <h1>🤗 Strands Transformers</h1>
  <h3>The universal entrypoint to HuggingFace transformers for Strands agents.</h3>
  <p><b>100% task & modality coverage. Zero hardcoding. Agents that see, hear, and speak — fully local.</b></p>

  <div>
    <a href="https://github.com/cagataycali/strands-transformers/issues"><img alt="issues" src="https://img.shields.io/github/issues/cagataycali/strands-transformers"/></a>
    <a href="https://python.org"><img alt="python" src="https://img.shields.io/badge/python-3.10+-blue"/></a>
    <img alt="transformers" src="https://img.shields.io/badge/🤗_transformers-24_tasks-yellow"/>
    <img alt="modalities" src="https://img.shields.io/badge/modalities-text·image·video·audio-orange"/>
    <img alt="license" src="https://img.shields.io/badge/license-MIT-green"/>
  </div>

  <p>
    <a href="#why"><b>Why</b></a> ·
    <a href="#60-second-hello-multimodal"><b>60-sec hello</b></a> ·
    <a href="#-multimodal-agent-brain-transformermodel"><b>Agent brain</b></a> ·
    <a href="#the-one-tool-use_transformers"><b>The tool</b></a> ·
    <a href="#robotics--vision-language-action-vla"><b>Robotics / VLA</b></a> ·
    <a href="#examples-at-a-glance"><b>Examples</b></a> ·
    <a href="#how-100-coverage-works"><b>How it works</b></a>
  </p>
</div>

---

`use_aws` wraps **all of boto3**. `use_lerobot` wraps **all of lerobot**.
**`use_transformers` wraps all of HuggingFace transformers** — every task, every
modality, in one tool. It reads transformers' own task taxonomy at runtime, so
the day HuggingFace ships a new task or model, you support it with **no code change**.

And `TransformerModel` lets a **local** HF model *be the agent's brain* — now
genuinely multimodal: pass image, video, audio, and document content blocks and
the model actually consumes them. With Qwen2.5-Omni it even **speaks back**.

```
  ┌────────────────── two ways to use transformers ──────────────────┐
  │                                                                   │
  │  use_transformers  (tool)   image·video·audio·text·robot-state    │
  │                              ─▶ text·audio·image·labels·actions   │
  │                                                                   │
  │  TransformerModel  (brain)  Agent(model=…) consumes image/video/  │
  │                              audio/document blocks → text + speech │
  └───────────────────────────────────────────────────────────────────┘
```

## Why

Wiring HuggingFace into an agent usually means a pile of per-model glue: look up
the right `AutoModel*` class, build the matching processor, format inputs,
decode outputs, special-case every new architecture. Multiply that by every task
and modality and it never ends.

`use_transformers` collapses all of it into **one tool** that reads transformers'
*own* task taxonomy at runtime — so coverage tracks upstream automatically. And
`TransformerModel` makes any local HF model a **drop-in Strands brain** that
speaks the full content-block protocol (image / video / audio / document),
no servers and no API keys.

|                        | Hand-rolled glue | **strands-transformers** |
|------------------------|------------------|--------------------------|
| New task / model       | write an adapter | works, **no code change** |
| Discovery              | read model cards | `action="tasks"` / `"inspect"` |
| Multimodal inputs      | format per model | content blocks, handled for you |
| Local model as brain   | custom provider  | `TransformerModel(model_path=…)` |
| Audio in / speech out  | bolt on TTS+ASR  | native via Qwen2.5-Omni |

## Install

```bash
pip install -e .
# optional extras:
pip install -e ".[audio]"     # soundfile, librosa  (mp3/flac/ogg decode)
pip install -e ".[vision]"    # pillow, opencv, av  (video)
pip install -e ".[training]"  # trl, peft, accelerate
```

## Quick Start — the tool

```python
from strands import Agent
from strands_transformers import use_transformers

agent = Agent(tools=[use_transformers])

agent("Transcribe recording.wav")                  # automatic-speech-recognition
agent("What's in scene.jpg?")                       # image-text-to-text
agent("Say 'hello from strands' as audio")          # text-to-audio
agent("Detect objects in https://.../street.jpg")   # object-detection
```

The agent discovers the right task, loads the model, runs it, and hands back
text plus paths to any generated media — all natively.

---

## 60-second hello, multimodal

A full local **vision agent** in a dozen lines — no API key, no server. Copy, run,
watch it name a color it has never been told:

```python
import io
from PIL import Image
from strands import Agent
from strands_transformers import TransformerModel

# a green square as PNG bytes
buf = io.BytesIO(); Image.new("RGB", (64, 64), (20, 200, 40)).save(buf, "PNG")

model = TransformerModel(model_path="HuggingFaceTB/SmolVLM-256M-Instruct")
agent = Agent(model=model, system_prompt="You are concise.")

print(agent([
    {"image": {"format": "png", "source": {"bytes": buf.getvalue()}}},
    {"text": "Color? One word."},
]))
# → Green.
```

That's a 256M-param model running on CPU or GPU, wired into the standard Strands
agent loop, *seeing* pixels through a content block. Swap the `model_path` for any
HF vision-language model and it just works.

## ⭐ Multimodal agent brain (`TransformerModel`)

Make a **local** HuggingFace model the agent's reasoning engine. The provider
implements the full Strands content-block taxonomy — text, image, video,
document — and adds an `audio` block for audio-native models. No servers, no API
keys, no cloud.

```python
from strands import Agent
from strands_transformers import TransformerModel

# Auto-detects a vision-language model and loads it with its AutoProcessor.
model = TransformerModel(model_path="HuggingFaceTB/SmolVLM-256M-Instruct")
agent = Agent(model=model, system_prompt="You are a concise vision assistant.")

# Pass an image content block — the Bedrock/Strands shape — and the model SEES it:
result = agent([
    {"image": {"format": "png", "source": {"bytes": png_bytes}}},
    {"text": "What color is this image? One word."},
])
print(result)   # → "Green."
```

> **Every snippet below maps to a runnable, GPU-verified example in
> [`examples/`](examples/).** Outputs shown are real model outputs.

### 🖼️ Image — `examples/multimodal_agent.py`
A green PNG in → **`"Green."`** out. Multimodal is auto-detected from the
processor; text-only causal LMs keep the original fast tokenizer path.

### 🎬 Video + 🧰 tool-result images — `examples/multimodal_advanced.py`
```python
# Video block: a list of frames (or a (T,H,W,C) array / container bytes).
# Provide fps so the model builds real frame timestamps.
agent_or_model.stream([{"role": "user", "content": [
    {"video": {"format": "mp4", "fps": 2.0, "source": {"bytes": frames}}},
    {"text": "Does this video get brighter or darker?"},
]}])                                          # → "BRIGHTER."

# The agentic-vision loop: a tool returns an image *inside a toolResult*,
# and the VLM reasons over it on the next turn.
{"toolResult": {"toolUseId": "t1", "status": "success", "content": [
    {"text": "Here is the captured screen:"},
    {"image": {"format": "png", "source": {"bytes": blue_png}}},
]}}                                           # → "Blue."
```

### 📄 Document — `examples/document_and_audio.py`
A `document` content block is flattened to text and fed to a plain text LM:
```python
{"document": {"name": "secret", "format": "txt",
              "source": {"bytes": b"...the passphrase is BANANA-42..."}}}
# "What is the passphrase?" → recovers "BANANA-42"
```

### 🔊 Audio in → text — `examples/audio_content_block.py`
The Strands schema has **no audio block**, so we extend it. `make_audio_block()`
builds one shaped exactly like `image`/`video`, and the provider routes it
through an audio-native model's feature extractor:
```python
from strands_transformers import TransformerModel, make_audio_block

model = TransformerModel(model_path="Qwen/Qwen2-Audio-7B-Instruct")
model.stream([{"role": "user", "content": [
    make_audio_block(waveform, "wav", 16000),
    {"text": "Describe what you hear."},
]}])
```

### 🎙️ Audio in **and** audio out — one model — `examples/omni_audio.py`
[Qwen2.5-Omni](https://huggingface.co/Qwen/Qwen2.5-Omni-3B) is any-to-any: it
*hears* audio in the conversation **and speaks its reply**. Unlike a TTS→ASR tool
chain (two pipeline models), text **and** a real 24 kHz speech waveform come from
a single `generate()`. The provider handles Omni's non-standard interface for you:
```python
from strands_transformers import TransformerModel, make_audio_block

model = TransformerModel(model_path="Qwen/Qwen2.5-Omni-3B")

# audio-in → text-out
model.stream([{"role": "user", "content": [
    make_audio_block(tone, "wav", 16000),
    {"text": "Is this a pure tone or human speech?"},
]}])                                          # → "It's a pure tone."

# text-in → text + SPEECH out
model.update_config(speak=True)               # enables the Talker
# ...stream a turn...
wav, sr = model.get_last_audio()              # (np.float32 waveform, 24000)
```
**Verified end-to-end:** Omni's spoken reply, re-transcribed by whisper, reads
back the words it was asked to say.

| Content block | Handled by | Example | Verified output |
|---------------|-----------|---------|-----------------|
| `text`        | tokenizer fast-path | every example | — |
| `image`       | AutoProcessor (vision) | `multimodal_agent.py` | `"Green."` |
| `video`       | processor + `VideoMetadata` (fps) | `multimodal_advanced.py` | `"BRIGHTER."` |
| `image` in `toolResult` | folded back into the turn | `multimodal_advanced.py` | `"Blue."` |
| `document`    | flattened to text | `document_and_audio.py` | recovers `BANANA-42` |
| `audio` *(our extension)* | feature extractor | `audio_content_block.py` | audio→text |
| `audio` in/out | Qwen2.5-Omni Thinker+Talker | `omni_audio.py` | hears + **speaks** |

Streaming, tool-calling, and Qwen3 `<think>` reasoning are all supported — see
[`models/transformers.py`](strands_transformers/models/transformers.py).

### Choosing a model

`device="auto"` picks cuda → mps → cpu, and bf16 on GPU. Pick by what you need:

| Want | Try | Notes |
|------|-----|-------|
| Tiny vision agent (laptop/CPU) | `HuggingFaceTB/SmolVLM-256M-Instruct` | the 60-sec demo; fast, runs anywhere |
| Video understanding | `HuggingFaceTB/SmolVLM2-500M-Video-Instruct` | accepts `video` blocks |
| Audio in → text | `Qwen/Qwen2-Audio-7B-Instruct` | hears audio in the conversation |
| Audio in **and** speech out | `Qwen/Qwen2.5-Omni-3B` | ~12 GB; `speak=True` for voice |
| Text-only reasoning brain | `Qwen/Qwen3-0.6B` … `Qwen3-8B` | `<think>` mode, tool-calling |

Multimodal is detected automatically from the model's processor — you don't flag
it. Text-only models keep the fast tokenizer path with zero overhead.

---

## The one tool: `use_transformers`

### Discover (never guess)

```python
use_transformers(action="tasks")        # 24 tasks + modality + auto-model + default
use_transformers(action="modalities")   # tasks grouped: text/image/audio/video/multimodal
use_transformers(action="task_info", task="image-text-to-text")
use_transformers(action="classes")      # all Auto* entrypoints
use_transformers(action="inspect", target="pipeline")   # signature + docs of anything
```

### Run — high level (native multimodal pipelines)

Inputs accept **file paths, URLs, base64 data-URIs, raw text, dicts, or arrays**.

```python
# ASR
use_transformers(action="run", task="automatic-speech-recognition", inputs="clip.wav")

# Vision-language
use_transformers(action="run", task="image-text-to-text",
                 model="HuggingFaceTB/SmolVLM-256M-Instruct",
                 inputs={"images": "scene.jpg", "text": "describe this"})

# Text-to-speech → .wav path returned in `artifacts`
use_transformers(action="run", task="text-to-audio",
                 model="facebook/mms-tts-eng", inputs="Hello!")

# Object detection from a URL
use_transformers(action="run", task="object-detection",
                 inputs="https://images.cocodataset.org/val2017/000000039769.jpg")
```

### Call — low level (any class / function / method)

For VLA / robot-action models, or anything pipelines don't cover. Load components
dynamically and cache them across calls:

```python
use_transformers(action="call", target="AutoProcessor.from_pretrained",
                 parameters={"pretrained_model_name_or_path": "model_id"}, cache_key="proc")
use_transformers(action="call", target="AutoModelForImageTextToText.from_pretrained",
                 parameters={"pretrained_model_name_or_path": "model_id"}, cache_key="vla")
use_transformers(action="call", target="cached:vla.predict_action", parameters={...})
```

## How "100% coverage" works

The source of truth is transformers' own `SUPPORTED_TASKS` registry, read at
runtime in [`core/registry.py`](strands_transformers/core/registry.py):

| Layer | File | Responsibility |
|-------|------|----------------|
| **Registry** | `core/registry.py` | Reads transformers' task taxonomy → modality → AutoModel. Dynamic class/fn resolution. |
| **Engine** | `core/engine.py` | Loads & caches pipelines/models. Auto device (cuda/mps/cpu) + dtype. |
| **I/O** | `core/io.py` | Coerces multimodal inputs; serializes outputs; saves audio/images to disk. |
| **Provider** | `models/transformers.py` | `TransformerModel` — local model as the agent brain, multimodal content blocks. |
| **Types** | `types/audio.py` | Audio content-block extension to the Strands taxonomy. |
| **Tool** | `tools/use_transformers.py` | The single `@tool` agents call. Discovery + run + call. |

Nothing is hardcoded per-task. New transformers task ⇒ instantly available.

```text
$ use_transformers(action="tasks")
🤗 transformers supports 24 tasks (100% coverage):
  • any-to-any                      [multimodal]  default: google/gemma-3n-E4B-it
  • automatic-speech-recognition    [multimodal]  auto: AutoModelForCTC, …SpeechSeq2Seq
  • image-text-to-text              [multimodal]  auto: AutoModelForImageTextToText
  • object-detection                [multimodal]  auto: AutoModelForObjectDetection
  • text-to-audio                   [text]        auto: AutoModelForTextToWaveform
  … 19 more — the list is read from transformers at runtime, so it's never stale.
```

## Robotics / Vision-Language-Action (VLA)

VLA models take camera images + a language instruction (+ robot state) and emit
**robot actions** via a custom `predict_action`, so they're driven through the
low-level `call` layer. Two worked, GPU-verified examples:

| Model | Example | Output |
|-------|---------|--------|
| [MolmoAct2-SO100_101](https://huggingface.co/allenai/MolmoAct2-SO100_101) | [`examples/molmoact_vla.py`](examples/molmoact_vla.py) | continuous actions `[1, 30, 6]` |
| [OpenVLA-7b](https://huggingface.co/openvla/openvla-7b) | [`examples/openvla_vla.py`](examples/openvla_vla.py) | 7-DoF action vector |

Helpers that make this ergonomic:
- `cached:key[.attr]` references resolve to live cached objects, including inside
  `parameters` (so `processor="cached:proc"` works).
- A `"**"` parameter key unpacks a cached mapping into kwargs — the idiomatic
  `model.predict_action(**processor(prompt, image))`.

### Legacy models on new transformers

Some VLA models (OpenVLA) shipped for transformers 4.x and break on 5.x. The
built-in [`core/compat.py`](strands_transformers/core/compat.py) shims patch the
gaps automatically (moved tokenizer symbols, removed `AutoModelForVision2Seq`,
`tie_weights` signature drift, hard `timm` pins) so the model's own code runs
unchanged. Trigger explicitly with `use_transformers(action="compat", ...)`.

## Examples at a glance

| Example | Layer | What it proves |
|---------|-------|----------------|
| [`multimodal_agent.py`](examples/multimodal_agent.py) | brain | image block → VLM agent (`"Green."`) |
| [`multimodal_advanced.py`](examples/multimodal_advanced.py) | brain | video round-trip + tool-result image |
| [`document_and_audio.py`](examples/document_and_audio.py) | brain+tool | document block + real TTS→ASR round-trip |
| [`audio_content_block.py`](examples/audio_content_block.py) | brain | audio content block → audio-native model |
| [`omni_audio.py`](examples/omni_audio.py) | brain | Qwen2.5-Omni audio-in **and** speech-out |
| [`smolvlm_image_text.py`](examples/smolvlm_image_text.py) | tool | real VLM via the `run` path |
| [`multimodal_pipelines.py`](examples/multimodal_pipelines.py) | tool | text/image/audio pipelines + ASR round-trip |
| [`vision_tasks.py`](examples/vision_tasks.py) | tool | detection, embeddings, depth, segmentation |
| [`molmoact_vla.py`](examples/molmoact_vla.py) | tool | VLA robot actions `[1,30,6]` |
| [`openvla_vla.py`](examples/openvla_vla.py) | tool | 7-DoF VLA + legacy compat |
| [`local_model_agent.py`](examples/local_model_agent.py) | brain | local causal-LM brain + tool |
| [`smoke.py`](examples/smoke.py) | — | fast 12-check E2E gate (no big downloads) |

```bash
PYTHONPATH=. python examples/<name>.py
```

## Supported modalities

| Modality | Example tasks |
|----------|---------------|
| **text** | text-generation, fill-mask, token/text-classification, feature-extraction, table-qa |
| **image** | image-classification, depth-estimation, image-feature-extraction, keypoint-matching |
| **audio** | automatic-speech-recognition, audio-classification, text-to-audio |
| **video** | video-classification |
| **multimodal** | image-text-to-text, visual/document-qa, object-detection, segmentation, zero-shot-*, any-to-any |

Run `use_transformers(action="tasks")` for the live, complete list on your install.

## License

MIT — built with [Strands Agents SDK](https://github.com/strands-agents/sdk-python)
and [HuggingFace Transformers](https://github.com/huggingface/transformers).
