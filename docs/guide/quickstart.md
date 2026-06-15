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

## As the agent's brain - 60-second multimodal hello

A full local **vision agent** - no API key, no server. The snippet below is a
self-contained [PEP 723](https://peps.python.org/pep-0723/) script: its
dependencies live in the header, so `uv run hello.py` installs everything into a
throwaway env and runs it. Save as `hello.py`:

```python
# /// script
# requires-python = ">=3.10"
# dependencies = ["strands-transformers[vision]"]
# ///
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
```

```console
$ uv run hello.py
Green.
```

A 256M-param model wired into the standard Strands agent loop, *seeing* pixels
through a content block. Swap `model_path` for any HF vision-language model
([pick a model](agent-brain.md#pick-a-model)) - same code.

Next: **[The tool](the-tool.md)** · **[The agent brain](agent-brain.md)**
