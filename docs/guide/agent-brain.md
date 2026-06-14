# The agent brain: `TransformerModel`

Make a **local** HuggingFace model the agent's reasoning engine. The provider
implements the full Strands content-block taxonomy — text, image, video,
document — and adds an `audio` block for audio-native models. No servers, no API
keys, no cloud.

```
        content blocks                TransformerModel                 out
   ┌───────────────────────┐      ┌──────────────────────┐     ┌──────────────┐
   │ text                  │      │  auto-detect:        │     │ text          │
   │ image  {format,bytes} │ ───▶ │   tokenizer (text)   │ ──▶ │ + reasoning   │
   │ video  {fps,frames}   │      │   AutoProcessor (👁) │     │   (<think>)   │
   │ audio  {bytes,sr}     │      │   feature_extr. (🔊) │     │ + tool calls  │
   │ document {bytes}      │      │   Omni Thinker+Talker│     │ + speech 🔊   │
   │ toolResult(image/…)   │      └──────────────────────┘     │   (Omni)      │
   └───────────────────────┘         standard Strands loop     └──────────────┘
```

```python
from strands import Agent
from strands_transformers import TransformerModel

model = TransformerModel(model_path="HuggingFaceTB/SmolVLM-256M-Instruct")
agent = Agent(model=model, system_prompt="You are a concise vision assistant.")

result = agent([
    {"image": {"format": "png", "source": {"bytes": png_bytes}}},
    {"text": "What color is this image? One word."},
])
print(result)   # → "Green."
```

Streaming, tool-calling, and Qwen3 `<think>` reasoning are all supported.
Multimodal is **auto-detected** from the model's processor — you don't flag it.
Text-only models keep the fast tokenizer path with zero overhead.

## Choosing a model

| Want | Try | Notes |
|------|-----|-------|
| Tiny vision agent (laptop/CPU) | `HuggingFaceTB/SmolVLM-256M-Instruct` | the 60-sec demo; fast, runs anywhere |
| Video understanding | `HuggingFaceTB/SmolVLM2-500M-Video-Instruct` | accepts `video` blocks |
| Audio in → text | `Qwen/Qwen2-Audio-7B-Instruct` | hears audio in the conversation |
| Audio in **and** speech out | `Qwen/Qwen2.5-Omni-3B` | ~12 GB; `speak=True` for voice |
| Text-only reasoning brain | `Qwen/Qwen3-0.6B` … `Qwen3-8B` | `<think>` mode, tool-calling |

See **[Content blocks](content-blocks.md)** for every modality, **[Audio](audio.md)**
for speech in/out, and the **[API reference](../reference/transformer-model.md)**.
