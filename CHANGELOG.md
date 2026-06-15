# Changelog

All notable changes to **strands-transformers** are documented here.
The format follows [Keep a Changelog](https://keepachangelog.com/),
and this project adheres to [Semantic Versioning](https://semver.org/).

## [0.4.0] - 2026-06-15

First public, open-source release.

### Highlights
- **`use_transformers`** - one tool over every HuggingFace transformers task
  (`tasks` / `modalities` / `task_info` / `classes` / `inspect` / `run` /
  `call` / `compat` / `cache` / `clear_cache`). The task taxonomy is read from
  transformers' `SUPPORTED_TASKS` at runtime - new upstream tasks need no code
  change here.
- **`TransformerModel`** - a local model provider for `Agent(model=...)` that
  speaks the Strands content-block protocol (text / image / video / audio /
  document). VLMs see, audio-native models hear, and Qwen2.5-Omni speaks back.
- **`make_audio_block`** - audio content-block extension (not in upstream
  Strands) for passing waveforms straight to audio-native models.

### Verified end-to-end (no mocks)
- text generation / classification, zero-shot
- image classification / detection / embeddings / depth / segmentation
- TTS and ASR (stdlib WAV, no ffmpeg required), text-to-music (MusicGen)
- SmolVLM image-text-to-text; Qwen2.5-Omni audio-in and speech-out
- MolmoAct and OpenVLA vision-language-action (real robot action tensors)
- `smoke.py` gate: 18/18 checks green

### Compatibility
- `core/compat.py` shims transformers 4.x-era custom-code models onto 5.x:
  tokenizer symbol moves, `AutoModelForVision2Seq` alias, `tie_weights` kwarg,
  timm version-pin spoof, and disabling a broken torchcodec path.

## [0.3.0] - 2026-06-14
- Internal pre-release (PyPI).

## [0.2.0]
- Internal pre-release (PyPI).
