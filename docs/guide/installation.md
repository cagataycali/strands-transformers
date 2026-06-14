# Installation

Requires Python 3.10+. We recommend [uv](https://docs.astral.sh/uv/).

=== "uv"

    ```bash
    uv pip install strands-transformers   # from PyPI
    # from source:
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
    pip install strands-transformers      # from PyPI
    # from source:
    pip install -e .
    pip install -e ".[audio]"
    pip install -e ".[vision]"
    pip install -e ".[training]"
    pip install -e ".[all]"
    ```

## Verify your install

Fast, no big downloads — 16 real checks:

```bash
PYTHONPATH=. python examples/smoke.py     # → "16/16 checks passed"
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


## Releasing (maintainers)

Versioning is derived from git tags via
[setuptools-scm](https://setuptools-scm.readthedocs.io/) — there is **no hardcoded
version**. To cut a release, just tag and push:

```bash
git tag v0.3.0
git push origin v0.3.0
```

The `Release` workflow then builds the sdist + wheel (the version comes from the
tag), publishes to PyPI using the `PYPI_API_TOKEN` repo secret, and creates a
GitHub Release with auto-generated notes.

!!! note "One-time PyPI setup"
    Add a PyPI API token as the `PYPI_API_TOKEN` secret in the repo
    (Settings → Secrets and variables → Actions). After that, releases are fully
    automated from tags.
