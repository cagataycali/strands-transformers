# Architecture

`strands-transformers` is the universal entrypoint to HuggingFace transformers for
Strands agents — 100% task & modality coverage with zero hardcoding. It reads
transformers' own task taxonomy at runtime, so new tasks/models work without code
changes (the same philosophy as `use_aws` wrapping boto3 and `use_lerobot`
wrapping lerobot).

## Layout

```
strands_transformers/
├── core/
│   ├── registry.py   # task taxonomy + dynamic class/attr resolution
│   ├── engine.py     # load/cache pipelines & models, device/dtype selection
│   ├── io.py         # multimodal input coercion + JSON-safe output serialization
│   └── compat.py     # backward-compat shims for legacy trust_remote_code models
├── models/
│   └── transformers.py  # TransformerModel — a Strands model provider (local brain)
└── tools/
    └── use_transformers.py  # the single @tool agents call
examples/              # runnable, GPU-verified examples (see examples/README.md)
```

## Data flow

```
agent → use_transformers(action=...) ─┬─ discovery   → registry
                                       ├─ run(task)   → engine.get_pipeline → pipeline(inputs) → io.serialize_output
                                       └─ call(target)→ registry.resolve_attr / cached: → obj(**params) → io.serialize_output
```

### `core/registry.py` — the source of truth
- `supported_tasks()` reads transformers' `SUPPORTED_TASKS` → `{task: {type, auto_models, default_model, pipeline_class}}`.
- `tasks_by_modality()`, `task_info()`, `resolve_task()` (tolerant of underscores/hyphens).
- `auto_model_classes()` lists every `Auto*` entrypoint.
- `resolve_attr(dotted)` resolves any dotted path into transformers (class, fn,
  method), with a root-getattr fast path so transformers' lazy `__getattr__`
  (which raises `AttributeError` on submodule-import attempts) never aborts
  resolution.
- `describe(obj)` introspects signatures/docstrings for the `inspect` action.

### `core/engine.py` — load, cache, run
- `select_device()` / `select_dtype()` auto-pick cuda/mps/cpu and bf16/fp16.
- `get_pipeline(task, model, ...)` builds & caches a `transformers.pipeline`.
  Image-output tasks (depth-estimation, segmentation, image-to-image,
  mask-generation) are kept in **float32** — half precision breaks PIL/numpy
  post-processing.
- `load_object(auto_class, model_path, ...)` loads any `AutoModel*` / `AutoProcessor`
  / `AutoTokenizer` via `from_pretrained` for the low-level `call` layer.
- `_CACHE` holds pipelines/models/processors keyed by `cache_key` for the session.

### `core/io.py` — multimodal I/O
- **In:** `coerce_input` decodes base64 data-URIs to PIL/bytes; paths/URLs/arrays
  pass through natively. `decode_wav` / `maybe_decode_audio_path` pre-decode WAV
  files for audio tasks with the stdlib `wave` module (no ffmpeg needed).
- **Out:** `serialize_output` converts any result to JSON-safe form — audio dicts
  → `.wav` artifacts, PIL images → `.png` artifacts, torch/numpy tensors → lists
  (bf16/fp16 upcast to float32 first), with `_ensure_json_safe` as a final guard.

### `core/compat.py` — legacy model support
Patches transformers 4.x→5.x gaps so old `trust_remote_code` models (e.g. OpenVLA)
run unchanged. Idempotent + re-entrant (`force=True`) because remote code can
re-import transformers mid-load:
- moved tokenizer symbols (`PaddingStrategy`, …) re-exposed on a real
  file-backed `tokenization_utils` module;
- `AutoModelForVision2Seq` recreated as an `AutoModelForImageTextToText` alias,
  asserted everywhere `auto_map` dispatch and `register_for_auto_class()` look;
- `tie_weights()` signature drift made kwarg-tolerant via an `init_weights` wrap;
- broken-torchcodec detection disabled so audio pipelines take the array path;
- `spoof_timm_version()` for models with hard timm pins.

### `tools/use_transformers.py` — the one tool
Two layers + discovery:
- **run** — high-level pipelines (native multimodal). Folds separate images into
  chat content for `image-text-to-text`; pre-decodes WAV for audio tasks.
- **call** — dynamic dispatch to any class/fn/method. `cached:key[.attr]` refs
  resolve to live cached objects (including inside `parameters`); a `"**"` param
  key unpacks a cached mapping into kwargs (e.g. `model.predict_action(**batch)`).
- **discovery** — `tasks`, `modalities`, `task_info`, `classes`, `inspect`,
  `cache`, `clear_cache`, `compat`.

### `models/transformers.py` — local brain
`TransformerModel` is a Strands model provider running any local HF causal-LM as
the agent's reasoning engine (streaming, chat templates, Qwen3 `<think>`, XML
tool-calling). Pair it with `use_transformers` for a fully local multimodal agent.

## Testing philosophy

Every change is verified **end-to-end against the real implementation** — actual
model inference / pipelines, not mocks. `examples/smoke.py` is a fast (no large
downloads) 12-check gate across discovery + text/image/audio that exits non-zero
on failure.
