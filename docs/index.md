# 🤗 Strands Transformers

**One tool wraps *all* of HuggingFace transformers. One provider makes any local
model a multimodal agent brain.** Agents that see, hear, and speak — 100% task
coverage, zero hardcoding, fully local.

`use_aws` wraps all of boto3. `use_lerobot` wraps all of lerobot.
**`use_transformers` wraps all of HuggingFace transformers** — every task, every
modality, in one tool that reads transformers' own task taxonomy at runtime. The
day HuggingFace ships a new task or model, you support it with **no code change**.

And `TransformerModel` lets a **local** HF model *be the agent's brain* —
genuinely multimodal: pass image, video, audio, and document content blocks and
the model consumes them. With Qwen2.5-Omni it even **speaks back**.

```
  use_transformers  (tool)    image · video · audio · text · robot-state
                          ──▶  text · audio · image · labels · actions

  TransformerModel  (brain)   Agent(model=…) consumes image/video/audio/
                          ──▶  document blocks  →  text · reasoning · speech
```

## What you can build

- 🗣️ **Voice assistant** — speak to it, it speaks back, one local model (Qwen2.5-Omni).
- 🤖 **Robot controller** — camera frames + an instruction → joint actions (MolmoAct, OpenVLA).
- 👁️ **Screen-watcher agent** — a tool returns a screenshot; the VLM reasons over it.
- 📄 **Document Q&A** — drop a doc content block in the conversation, ask about it.
- 🎬 **Video understander** — pass frames, ask what changes over time.
- 🔌 **Any HF task on tap** — ASR, detection, segmentation, embeddings… via one tool.

All **local** — no API keys, no servers, no per-model glue.

## Why

Wiring HuggingFace into an agent usually means a pile of per-model glue: look up
the right `AutoModel*` class, build the matching processor, format inputs, decode
outputs, special-case every new architecture. Multiply that by every task and
modality and it never ends.

| | Hand-rolled glue | **strands-transformers** |
|---|---|---|
| New task / model | write an adapter | works, **no code change** |
| Discovery | read model cards | `action="tasks"` / `"inspect"` |
| Multimodal inputs | format per model | content blocks, handled for you |
| Local model as brain | custom provider | `TransformerModel(model_path=…)` |
| Audio in / speech out | bolt on TTS+ASR | native via Qwen2.5-Omni |

## Where to next

- New here? → **[Installation](guide/installation.md)** then **[Quickstart](guide/quickstart.md)**
- Want the agent to see/hear → **[Agent brain](guide/agent-brain.md)**, **[Content blocks](guide/content-blocks.md)**, **[Audio](guide/audio.md)**
- Robotics → **[Robotics / VLA](guide/robotics.md)**
- Internals → **[Architecture](reference/architecture.md)** and the **[API reference](reference/transformer-model.md)**
