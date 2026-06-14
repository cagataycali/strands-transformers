# Examples

Real, end-to-end examples — every one driven against live model inference, not
mocked. Run any with the repo on your path:

```bash
PYTHONPATH=. python examples/<name>.py
```

| Example | Layer | What it proves |
|---------|-------|----------------|
| `multimodal_agent.py` | brain | image block → VLM agent (`"Green."`) |
| `multimodal_advanced.py` | brain | video round-trip + tool-result image (`"BRIGHTER."`, `"Blue."`) |
| `document_and_audio.py` | brain+tool | document block + real TTS→ASR round-trip |
| `audio_content_block.py` | brain | audio content block → audio-native model |
| `omni_audio.py` | brain | Qwen2.5-Omni audio-in **and** speech-out |
| `smolvlm_image_text.py` | tool | real VLM via the `run` path |
| `multimodal_pipelines.py` | tool | text/image/audio pipelines + ASR round-trip |
| `vision_tasks.py` | tool | detection, embeddings, depth, segmentation |
| `cosmos_reason_embodied.py` | tool | Cosmos-Reason2 embodied scene reasoning |
| `robot_reason_act_agent.py` | tool | two-step robot agent: Cosmos-Reason plans → MolmoAct acts |
| `molmoact_vla.py` | tool | VLA robot actions `[1,30,6]` |
| `openvla_vla.py` | tool | 7-DoF VLA + legacy compat |
| `local_model_agent.py` | brain | local causal-LM brain + tool |
| `smoke.py` | — | fast 12-check E2E gate (no big downloads) |

- **brain** = uses `TransformerModel` as the agent's model provider.
- **tool** = uses `use_transformers` as a tool the agent calls.

## FAQ & troubleshooting

??? question "A Qwen3 reply came back empty / all reasoning."
    Qwen3 spends tokens inside `<think>…</think>` first. Raise `max_tokens`, or
    set `enable_thinking=False`. Reasoning streams separately as `reasoningContent`.

??? question "`mp3` / `flac` / `ogg` audio won't decode."
    WAV works out of the box (stdlib). For compressed formats install the extra:
    `uv pip install -e \".[audio]\"` (pulls in `soundfile`). Raw numpy waveforms
    always work.

??? question "`trust_remote_code` models (VLA, Omni)."
    `TransformerModel` and the `call` path pass `trust_remote_code=True` by
    default. Legacy 4.x-era models are auto-patched by `core/compat.py`.

??? question "Qwen2.5-Omni didn't speak."
    Speech is off by default (keeps text fast). Set
    `model.update_config(speak=True)`, then read the waveform with
    `model.get_last_audio()` → `(np.float32, 24000)`.

??? question "Out of memory."
    Drop to a smaller model (see *Choosing a model*), or force `device=\"cpu\"`.
    The provider uses bf16 on GPU automatically.

??? question "Where do generated images / audio go?"
    The `run` path writes media to disk and returns the path in the result's
    `artifacts` list (e.g. a TTS `.wav`).
