# AGENTS.md

Guidance for AI agents and contributors working in this repository.

## What this is

`strands-transformers` connects HuggingFace **transformers** to **Strands agents**
two ways:

- **`use_transformers`** - one tool exposing every transformers task. Reads the
  task taxonomy at runtime (no hardcoded per-task code). Actions: `tasks`,
  `modalities`, `task_info`, `classes`, `inspect`, `run`, `call`, `compat`,
  `cache`, `clear_cache`.
- **`TransformerModel`** - a local model provider for `Agent(model=...)`. Speaks
  the Strands content-block protocol (`text`, `image`, `video`, `audio`,
  `document`), so multimodal models receive media directly. Audio-native models
  (Qwen2-Audio, Qwen2.5-Omni) hear and, for Omni, speak back.

## Layout

```
strands_transformers/
  tools/use_transformers.py   the one @tool: discover / run / call
  models/transformers.py      TransformerModel provider (multimodal brain)
  types/audio.py              audio content-block extension (not in upstream Strands)
  core/
    registry.py               transformers task taxonomy -> modality -> AutoModel
    engine.py                 load + cache models/pipelines; device + dtype
    io.py                     coerce inputs; serialize outputs; save media
    compat.py                 shims so 4.x-era custom-code models run on 5.x
examples/                     runnable, real-inference examples (no mocks)
docs/                         MkDocs site (Material theme)
```

## Golden rules

1. **Verify on a real path, never mock.** Import the actual code, run real
   inference or real serialization, and check `status == "success"` plus real
   data shapes/text before claiming anything works. Most bugs in this repo were
   found by *using* it, not reading it.
2. **Keep `examples/smoke.py` green.** It is the fast regression gate (no large
   downloads). Run it after any change:
   ```
   PYTHONPATH=. python examples/smoke.py     # -> "N/N checks passed"
   ```
   When you fix a bug, add a guard to smoke so it can't regress.
3. **Zero hardcoding per task.** The source of truth is transformers'
   `SUPPORTED_TASKS` / `TASK_ALIASES`, read at runtime in `core/registry.py`.
   A new upstream task should work with no code change.
4. **Fail loudly, not silently.** Prefer a clear, actionable error
   (e.g. "install strands-transformers[vision]") over degrading to a path that
   crashes cryptically later.
5. **No emdashes** in code, docs, or commit messages. Use ` - ` or `-`.
6. **Examples use tiny/cached models** so they run in seconds; the same code
   scales to SOTA by changing `model_path`.

## Conventions

- **Docs snippets are runnable** PEP 723 scripts (deps in a `# /// script` header)
  so `uv run demo.py` just works. Pair each snippet with a real ` ```console `
  result block. Generated media (audio/video/images) live under `docs/assets/`.
- **Theme**: dark-first ink canvas with Strands violet (#8B5CFF) -> cyan (#22D3EE)
  accents; defined in `docs/assets/extra.css` and `mkdocs.yml`.
- **Mermaid** for diagrams (color-coded: violet inputs, amber core, cyan out).
  Use `flowchart`; avoid `<br/>` inside `sequenceDiagram` participant aliases.
- **Versioning** is git-tag driven via setuptools-scm (no hardcoded version).
  Release = `git tag vX.Y.Z && git push origin vX.Y.Z` (CI builds + publishes).

## Dependencies

Core: `strands-agents`, `transformers`, `torch`, `accelerate`, `pillow`, `numpy`.
Extras: `[audio]` (soundfile, librosa), `[vision]` (torchvision, opencv, av -
**vision-language models need this**), `[training]`, `[docs]`.

## Hard-won lessons (do not regress)

- 8-bit WAV is **unsigned** PCM (silence at 128); only 16/32-bit are signed.
- `device_map="cuda"` needs `accelerate`; transformers 5.x vision processors
  need `torchvision`. Missing either gives a clear error, not a silent fallback.
- Sanitize NaN/Inf before int16 audio casts; cap recursion/serialization breadth.
- The `call` path routes `AutoX.from_pretrained` through `engine.load_object`
  (auto device/dtype/trust_remote_code); explicit user params always win.
- `pad_token_id ... got 128002` is a transformers-internal warning about a
  model's own config (e.g. SmolVLM), not a bug in this package.

## Before you push

```
PYTHONPATH=. python examples/smoke.py        # all checks pass
PYTHONPATH=. python -m pytest tests/ -q       # if tests present
mkdocs build --strict                         # docs build clean
```
