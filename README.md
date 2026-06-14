<div align="center">
  <h1>🤗 Strands Transformers</h1>
  <h3>One tool wraps <i>all</i> of HuggingFace transformers. One provider makes any local model a multimodal agent brain.</h3>
  <p><b>Agents that see, hear, and speak — 100% task coverage, zero hardcoding, fully local.</b></p>

  <div>
    <a href="https://github.com/cagataycali/strands-transformers/actions/workflows/docs.yml"><img alt="docs" src="https://github.com/cagataycali/strands-transformers/actions/workflows/docs.yml/badge.svg"/></a>
    <a href="https://github.com/cagataycali/strands-transformers/issues"><img alt="issues" src="https://img.shields.io/github/issues/cagataycali/strands-transformers"/></a>
    <img alt="python" src="https://img.shields.io/badge/python-3.10+-blue"/>
    <img alt="transformers" src="https://img.shields.io/badge/🤗_transformers-24_tasks-yellow"/>
    <img alt="modalities" src="https://img.shields.io/badge/modalities-text·image·video·audio-orange"/>
    <img alt="license" src="https://img.shields.io/badge/license-MIT-green"/>
  </div>
</div>

---

`use_aws` wraps all of boto3. `use_lerobot` wraps all of lerobot.
**`use_transformers` wraps all of HuggingFace transformers** — every task, every
modality, in one tool that reads transformers' own taxonomy at runtime (new task
upstream ⇒ supported here with **no code change**). And **`TransformerModel`** makes
any **local** HF model a drop-in Strands brain that speaks the full content-block
protocol — image, video, audio, document. With Qwen2.5-Omni it even **speaks back**.

```
  use_transformers  (tool)    image · video · audio · text · robot-state  ──▶  text · audio · image · labels · actions
  TransformerModel  (brain)   Agent(model=…) consumes image/video/audio/document blocks  ──▶  text · reasoning · speech
```

📖 **[Full documentation →](https://cagataycali.github.io/strands-transformers/)** &nbsp;·&nbsp; built with MkDocs (`docs/`)

## Install

```bash
uv pip install -e .            # or: pip install -e .
PYTHONPATH=. python examples/smoke.py     # verify → "12/12 checks passed"
```

<details>
<summary>Optional extras (audio · vision · training · docs)</summary>

```bash
uv pip install -e ".[audio]"      # soundfile, librosa  (mp3/flac/ogg decode)
uv pip install -e ".[vision]"     # opencv, av  (video)
uv pip install -e ".[training]"   # trl, peft, accelerate
uv pip install -e ".[docs]"       # mkdocs-material, mkdocstrings
uv pip install -e ".[all]"        # everything
```
WAV audio works without extras. `device="auto"` picks cuda → mps → cpu (bf16 on GPU).
</details>

## 60-second hello — a local vision agent

```python
import io
from PIL import Image
from strands import Agent
from strands_transformers import TransformerModel

buf = io.BytesIO(); Image.new("RGB", (64, 64), (20, 200, 40)).save(buf, "PNG")  # green square

model = TransformerModel(model_path="HuggingFaceTB/SmolVLM-256M-Instruct")
agent = Agent(model=model, system_prompt="You are concise.")

print(agent([
    {"image": {"format": "png", "source": {"bytes": buf.getvalue()}}},
    {"text": "Color? One word."},
]))
# → Green.
```

A 256M-param model in the standard Strands loop, *seeing* pixels through a content
block — no API key, no server. Swap `model_path` for any HF VLM.

## Two ways to use it

<details open>
<summary><b>As a tool</b> — <code>use_transformers</code> (discover · run · call)</summary>

```python
from strands import Agent
from strands_transformers import use_transformers

agent = Agent(tools=[use_transformers])
agent("Transcribe recording.wav")                  # automatic-speech-recognition
agent("What's in scene.jpg?")                       # image-text-to-text
agent("Say 'hello from strands' as audio")          # text-to-audio
agent("Detect objects in https://.../street.jpg")   # object-detection
```

Discover everything at runtime (`action="tasks" | "modalities" | "inspect" | …`),
run high-level pipelines, or `call` any class/fn/method for custom models.
→ **[The tool guide](https://cagataycali.github.io/strands-transformers/guide/the-tool/)**
</details>

<details>
<summary><b>As the agent's brain</b> — <code>TransformerModel</code> (multimodal content blocks)</summary>

Pass `image` / `video` / `audio` / `document` content blocks (and media inside a
`toolResult`) — the provider auto-detects the model's processor and routes them.
All outputs below are **real** results (CUDA, transformers 5.12 / torch 2.10):

| Content block | Example | Verified output |
|---|---|---|
| `image` | `multimodal_agent.py` | `"Green."` |
| `video` (with `fps`) | `multimodal_advanced.py` | `"BRIGHTER."` |
| `image` in `toolResult` | `multimodal_advanced.py` | `"Blue."` |
| `document` | `document_and_audio.py` | recovers `BANANA-42` |
| `audio` *(our schema extension)* | `audio_content_block.py` | audio → text |
| `audio` in **and** speech out | `omni_audio.py` | hears + **speaks** (Qwen2.5-Omni) |

→ **[Agent brain](https://cagataycali.github.io/strands-transformers/guide/agent-brain/)** ·
**[Content blocks](https://cagataycali.github.io/strands-transformers/guide/content-blocks/)** ·
**[Audio](https://cagataycali.github.io/strands-transformers/guide/audio/)**
</details>

<details>
<summary><b>Robotics / VLA</b> — camera + instruction → robot actions</summary>

VLA models expose a custom `predict_action`, driven through the `call` layer.
Verified: [MolmoAct2](https://huggingface.co/allenai/MolmoAct2-SO100_101) → actions
`[1,30,6]`; [OpenVLA-7b](https://huggingface.co/openvla/openvla-7b) → 7-DoF vector
(with automatic 4.x→5.x compat shims).
→ **[Robotics guide](https://cagataycali.github.io/strands-transformers/guide/robotics/)**
</details>

## How it works

Nothing is hardcoded per task — `core/registry.py` reads transformers' own
`SUPPORTED_TASKS` at runtime, so coverage tracks upstream automatically.

<details>
<summary>Project layout</summary>

```
strands_transformers/
├── tools/use_transformers.py   # the one @tool: discover · run · call
├── models/transformers.py      # TransformerModel — local multimodal agent brain
├── types/audio.py              # audio content-block extension
└── core/{registry,engine,io,compat}.py   # taxonomy · load/cache · I/O · legacy shims
```
→ **[Architecture](https://cagataycali.github.io/strands-transformers/reference/architecture/)** ·
**[API reference](https://cagataycali.github.io/strands-transformers/reference/transformer-model/)**
</details>

## Examples

12 runnable, GPU-verified examples in [`examples/`](examples/) — image, video,
audio, document, Omni speech, VLA, and pipelines. Run any:

```bash
PYTHONPATH=. python examples/<name>.py
```

→ **[Examples & FAQ](https://cagataycali.github.io/strands-transformers/reference/examples/)**

## License

MIT — built with [Strands Agents SDK](https://github.com/strands-agents/sdk-python)
and [HuggingFace Transformers](https://github.com/huggingface/transformers).

<div align="center">
  <sub>If this saved you a pile of per-model glue code, consider giving it a ⭐</sub>
</div>
