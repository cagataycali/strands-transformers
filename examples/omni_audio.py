"""Qwen2.5-Omni-3B — real audio-in AND audio-out through ONE model.

This is the frontier: Qwen2.5-Omni is an any-to-any model. Unlike the
TTS→ASR *tool* round-trip (two separate pipeline models), here a SINGLE model
both *hears* audio in the conversation and *speaks* its reply — text and a real
24 kHz speech waveform come out of one generate() call.

`TransformerModel` handles Omni's non-standard interface transparently:
  - auto-detects the Qwen2_5Omni architecture and loads the right class;
  - audio content blocks (our taxonomy extension) feed the model's audio tower;
  - a dedicated _stream_omni path drives generate(thinker_/talker_max_new_tokens,
    return_audio=...) and streams the decoded text;
  - set config `speak=True` and the synthesized speech is available via
    `model.get_last_audio()` as (waveform float32, sample_rate).

    PYTHONPATH=. python examples/omni_audio.py

Requires Qwen/Qwen2.5-Omni-3B (~12 GB). E2E verified on NVIDIA Thor:
  - audio-in (a 440 Hz tone) -> "It's a pure tone." (correctly NOT speech)
  - text-in -> text "Strands transformers can speak." + 4.1 s of speech;
    whisper-tiny re-transcribes that speech as "Transformers can speak."
"""

import asyncio
import os
import wave

import numpy as np

from strands_transformers.models.transformers import TransformerModel
from strands_transformers.types import make_audio_block

MODEL = "Qwen/Qwen2.5-Omni-3B"
OUT_WAV = "/tmp/omni/omni_reply.wav"


async def _collect(model, messages) -> str:
    out = ""
    async for ev in model.stream(messages):
        delta = ev.get("contentBlockDelta", {}).get("delta", {})
        if "text" in delta:
            out += delta["text"]
    return out.strip()


def save_wav(wav: np.ndarray, sr: int, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    pcm = (np.clip(wav, -1, 1) * 32767).astype("int16")
    with wave.open(path, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(pcm.tobytes())


def main(model_id: str = MODEL) -> int:
    model = TransformerModel(model_path=model_id, params={"max_tokens": 48})
    print("is_omni:", model.is_omni, "| model class:", type(model.model).__name__)
    if not model.is_omni:
        print("status: unexpected (not an omni model)")
        return 1

    # 1) AUDIO IN -> TEXT OUT (a 440 Hz pure tone)
    sr = 16000
    tone = np.sin(2 * np.pi * 440 * np.arange(int(sr)) / sr).astype(np.float32)
    msgs_in = [
        {"role": "system", "content": [{"text": "You are Qwen, a helpful assistant."}]},
        {"role": "user", "content": [
            make_audio_block(tone, "wav", sr),
            {"text": "Is this a pure tone or human speech? One short sentence."},
        ]},
    ]
    heard = asyncio.run(_collect(model, msgs_in))
    print("audio-in -> text:", repr(heard[-160:]))

    # 2) TEXT IN -> TEXT + SPEECH OUT
    model.update_config(speak=True, params={"max_tokens": 48, "talker_max_tokens": 512})
    msgs_out = [
        {"role": "system", "content": [
            {"text": "You are Qwen, a helpful assistant capable of generating speech."}
        ]},
        {"role": "user", "content": [{"text": "Say exactly: Strands transformers can speak."}]},
    ]
    said = asyncio.run(_collect(model, msgs_out))
    print("text-in  -> text:", repr(said[-120:]))

    spoke = False
    audio = model.get_last_audio()
    if audio is not None:
        wav, sr_out = audio
        save_wav(wav, sr_out, OUT_WAV)
        dur = len(wav) / sr_out
        print(f"text-in  -> speech: {wav.shape} @ {sr_out} Hz | {dur:.2f}s -> {OUT_WAV}")
        spoke = wav.size > 1000

    ok = bool(heard) and bool(said) and spoke
    print("status:", "success" if ok else "unexpected")
    return 0 if ok else 1


if __name__ == "__main__":
    import sys

    raise SystemExit(main(sys.argv[1] if len(sys.argv) > 1 else MODEL))
