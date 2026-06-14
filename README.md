<div align="center">
  <h1>🤗 Strands Transformers</h1>
  <h3>The universal entrypoint to HuggingFace transformers for Strands agents.</h3>
  <p><b>100% task & modality coverage. Zero hardcoding.</b></p>

  <div>
    <a href="https://github.com/cagataycali/strands-transformers/issues"><img alt="issues" src="https://img.shields.io/github/issues/cagataycali/strands-transformers"/></a>
    <a href="https://python.org"><img alt="python" src="https://img.shields.io/badge/python-3.10+-blue"/></a>
  </div>
</div>

---

`use_aws` wraps **all of boto3**. `use_lerobot` wraps **all of lerobot**.
**`use_transformers` wraps all of HuggingFace transformers** — every task, every
modality, in one tool. It reads transformers' own task taxonomy at runtime, so
the day HuggingFace ships a new task or model, you support it with **no code change**.

```
   image · video · audio · text · robot-state   ─▶  use_transformers  ─▶   text · audio · image · labels · actions
```

## Install

```bash
pip install -e .
# optional extras:
pip install -e ".[audio]"     # soundfile, librosa
pip install -e ".[vision]"    # opencv, av (video)
pip install -e ".[training]"  # trl, peft, accelerate
```

## Quick Start

```python
from strands import Agent
from strands_transformers import use_transformers

agent = Agent(tools=[use_transformers])

agent("Transcribe recording.wav")              # automatic-speech-recognition
agent("What's in scene.jpg?")                  # image-text-to-text
agent("Say 'hello from strands' as audio")     # text-to-audio
agent("Detect objects in https://.../street.jpg")  # object-detection
```

The agent discovers the right task, loads the model, runs it, and hands back
text plus paths to any generated media — all natively.

## The one tool: `use_transformers`

### Discover (never guess)

```python
use_transformers(action="tasks")                  # 24 tasks + modality + auto-model + default
use_transformers(action="modalities")             # tasks grouped: text/image/audio/video/multimodal
use_transformers(action="task_info", task="image-text-to-text")
use_transformers(action="classes")                # all Auto* entrypoints (AutoModelForImageTextToText, ...)
use_transformers(action="inspect", target="pipeline")   # signature + docs of anything
```

### Run — high level (native multimodal pipelines)

Inputs accept **file paths, URLs, base64 data-URIs, raw text, dicts, or arrays**.

```python
# ASR
use_transformers(action="run", task="automatic-speech-recognition", inputs="clip.wav")

# Vision-language (e.g. MolmoAct, LLaVA, Qwen-VL)
use_transformers(action="run", task="image-text-to-text",
                 model="allenai/MolmoAct2-SO100_101",
                 inputs={"images": "scene.jpg", "text": "pick up the red cube"})

# Text-to-speech → saved as .wav, path returned in `artifacts`
use_transformers(action="run", task="text-to-audio",
                 model="suno/bark-small", inputs="Hello!")

# Object detection from a URL
use_transformers(action="run", task="object-detection",
                 inputs="https://images.cocodataset.org/val2017/000000039769.jpg")
```

### Call — low level (any class / function / method)

For VLA / robot-action models, or anything pipelines don't cover. Load components
dynamically and cache them across calls:

```python
# load processor + model once, cache them
use_transformers(action="call", target="AutoProcessor.from_pretrained",
                 parameters={"pretrained_model_name_or_path": "model_id"}, cache_key="proc")
use_transformers(action="call", target="AutoModelForImageTextToText.from_pretrained",
                 parameters={"pretrained_model_name_or_path": "model_id"}, cache_key="vla")

# preprocess → generate actions/tokens using the cached objects
use_transformers(action="call", target="cached:proc",
                 parameters={"images": "scene.jpg", "text": "grasp", "return_tensors": "pt"},
                 cache_key="batch")
use_transformers(action="call", target="cached:vla.generate", parameters={...})
```

## How "100% coverage" works

The source of truth is transformers' own `SUPPORTED_TASKS` registry. We read it at
runtime in [`core/registry.py`](strands_transformers/core/registry.py):

| Layer | File | Responsibility |
|-------|------|----------------|
| **Registry** | `core/registry.py` | Reads transformers' task taxonomy → modality → AutoModel. Dynamic resolution of any class/fn. |
| **Engine** | `core/engine.py` | Loads & caches pipelines/models. Auto device (cuda/mps/cpu) + dtype. |
| **I/O** | `core/io.py` | Coerces multimodal inputs; serializes outputs; saves audio/images to disk. |
| **Tool** | `tools/use_transformers.py` | The single `@tool` agents call. Discovery + run + call. |

Nothing is hardcoded per-task. New transformers task ⇒ instantly available.

## Robotics / Vision-Language-Action (VLA)

VLA models take camera images + a language instruction (+ robot state) and emit
**robot actions**. They usually expose a custom `predict_action` method rather
than a standard pipeline, so they're driven through the low-level `call` layer.
Two worked, GPU-verified examples live in [`examples/`](examples/):

| Model | Example | Output |
|-------|---------|--------|
| [MolmoAct2-SO100_101](https://huggingface.co/allenai/MolmoAct2-SO100_101) | [`examples/molmoact_vla.py`](examples/molmoact_vla.py) | continuous actions `[1, 30, 6]` |
| [OpenVLA-7b](https://huggingface.co/openvla/openvla-7b) | [`examples/openvla_vla.py`](examples/openvla_vla.py) | 7-DoF action vector |

```python
use_transformers(action="call", target="AutoProcessor.from_pretrained",
                 parameters={"pretrained_model_name_or_path": REPO, "trust_remote_code": True},
                 cache_key="proc")
use_transformers(action="call", target="AutoModelForImageTextToText.from_pretrained",
                 parameters={"pretrained_model_name_or_path": REPO, "trust_remote_code": True,
                             "dtype": "bfloat16", "device_map": "cuda"}, cache_key="vla")
use_transformers(action="call", target="cached:vla.predict_action",
                 parameters={"processor": "cached:proc", "images": [...], "state": [...]})
```

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
unchanged. You can also trigger it explicitly:

```python
use_transformers(action="compat", parameters={"timm_version": "0.9.16"})
```

## Local model as the agent's brain

Use any HuggingFace causal-LM as the Strands model provider:

```python
from strands import Agent
from strands_transformers import TransformerModel, use_transformers

brain = TransformerModel(model_path="Qwen/Qwen3-1.7B", device="auto", enable_thinking=True)
agent = Agent(model=brain, tools=[use_transformers])
```

Streaming, tool-calling, and Qwen3 `<think>` reasoning are supported — see
[`models/transformers.py`](strands_transformers/models/transformers.py).

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
