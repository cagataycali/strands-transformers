# Quickstart

Two ways to use transformers in a Strands agent.

## As a tool

Give the agent `use_transformers` and it discovers the right task, loads the
model, runs it, and returns text plus paths to any generated media.

```python
from strands import Agent
from strands_transformers import use_transformers

agent = Agent(tools=[use_transformers])

agent("Transcribe recording.wav")                  # automatic-speech-recognition
agent("What's in scene.jpg?")                       # image-text-to-text
agent("Say 'hello from strands' as audio")          # text-to-audio
agent("Detect objects in https://.../street.jpg")   # object-detection
```

## As the agent's brain — 60-second multimodal hello

A full local **vision agent** in a dozen lines — no API key, no server. Watch it
name a color it was never told:

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

That's a 256M-param model wired into the standard Strands agent loop, *seeing*
pixels through a content block. Swap `model_path` for any HF vision-language
model and it just works.

Next: **[The tool](the-tool.md)** · **[The agent brain](agent-brain.md)**
