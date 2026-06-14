# Installation

Requires Python 3.10+. We recommend [uv](https://docs.astral.sh/uv/).

=== "uv"

    ```bash
    uv pip install -e .
    # optional extras:
    uv pip install -e ".[audio]"      # soundfile, librosa  (mp3/flac/ogg decode)
    uv pip install -e ".[vision]"     # pillow, opencv, av  (video)
    uv pip install -e ".[training]"   # trl, peft, accelerate
    uv pip install -e ".[docs]"       # mkdocs-material, mkdocstrings
    uv pip install -e ".[all]"        # everything
    ```

=== "pip"

    ```bash
    pip install -e .
    pip install -e ".[audio]"
    pip install -e ".[vision]"
    pip install -e ".[training]"
    pip install -e ".[all]"
    ```

## Verify your install

Fast, no big downloads — 12 real checks:

```bash
PYTHONPATH=. python examples/smoke.py     # → "12/12 checks passed"
```

## Optional extras

| Extra | Pulls in | Needed for |
|-------|----------|-----------|
| `audio` | soundfile, librosa | mp3/flac/ogg decode (WAV works without it) |
| `vision` | pillow, opencv, av | video container decode |
| `training` | trl, peft, accelerate | fine-tuning workflows |
| `docs` | mkdocs-material, mkdocstrings | building this site |

!!! tip "Device selection"
    `device="auto"` (the default) picks **cuda → mps → cpu**, and uses bf16 on GPU.
