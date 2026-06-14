"""Audio content block → audio-native model (Qwen2-Audio / Qwen2.5-Omni).

THE NOVEL BIT: the Strands / harness-sdk message schema has no `audio` content
block. We extend the taxonomy in `strands_transformers.types.audio` with one
shaped exactly like `image`/`video`:

    {"audio": {"format": "wav", "source": {"bytes": <wav | np waveform>}}}

`TransformerModel` now:
  - detects audio-native models via the processor's `feature_extractor`
    (no image_processor) and loads Qwen2_5Omni / Qwen2Audio classes;
  - decodes the audio payload (stdlib WAV, or soundfile for mp3/flac/ogg, or a
    bare numpy waveform), resamples to the feature-extractor rate;
  - emits an audio chat part that triggers the model's <|AUDIO|> tokens and
    passes the raw waveform via processor(audio=..., sampling_rate=...).

So an `audio` content block (including audio returned inside a toolResult) now
reaches an audio-native model *inside the conversation* — not via a separate
ASR tool. Audio also works as a turn the model reasons over alongside text.

Default model is a tiny-random Qwen2-Audio so this runs fast without a large
download; its weights are random so the *text* is gibberish, but the full
audio→features→model→text path is exercised end to end. Point `--model` at
`Qwen/Qwen2-Audio-7B-Instruct` or `Qwen/Qwen2.5-Omni-3B` for real answers.

    PYTHONPATH=. python examples/audio_content_block.py
"""

import asyncio
import sys

import numpy as np

from strands_transformers.models.transformers import TransformerModel
from strands_transformers.types import make_audio_block

MODEL = "yujiepan/qwen2-audio-tiny-random"


async def _collect(model, messages) -> str:
    out = ""
    async for ev in model.stream(messages):
        delta = ev.get("contentBlockDelta", {}).get("delta", {})
        if "text" in delta:
            out += delta["text"]
    return out.strip()


def main(model_id: str = MODEL) -> int:
    model = TransformerModel(model_path=model_id, params={"max_tokens": 16, "do_sample": False})
    print("is_multimodal:", model.is_multimodal, "| has_audio_input:", model.has_audio_input)
    print("model class:", type(model.model).__name__)
    if not model.has_audio_input:
        print("status: unexpected (model is not audio-native)")
        return 1

    # 1-second 440 Hz tone as a mono waveform → audio content block
    sr = 16000
    tone = np.sin(2 * np.pi * 440 * np.arange(int(sr)) / sr).astype(np.float32)

    messages = [{"role": "user", "content": [
        make_audio_block(tone, "wav", sr),
        {"text": "Describe what you hear."},
    ]}]

    answer = asyncio.run(_collect(model, messages))
    print("audio-block answer:", repr(answer[:120]))

    # Success = the audio path produced *some* text (tiny-random => gibberish,
    # but the audio tower + LM ran end to end on the waveform).
    ok = bool(answer)
    print("status:", "success" if ok else "unexpected")
    return 0 if ok else 1


if __name__ == "__main__":
    mid = sys.argv[1] if len(sys.argv) > 1 else MODEL
    raise SystemExit(main(mid))
