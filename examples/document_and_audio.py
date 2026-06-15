"""Document content blocks + a real audio round-trip.

DOCUMENT (model-provider content block):
    A `document` content block is flattened to text by `TransformerModel` and
    fed to the model as part of the prompt - so a plain text causal LM can
    answer questions about an attached document. Verified: a txt document
    containing a passphrase → the model recovers the passphrase.

AUDIO (tool path, not a content block):
    Note: the Strands / harness-sdk message schema has NO audio content block
    (the arms are text/image/video/document/tool-*). Audio is therefore an
    *I/O modality*, handled by the `use_transformers` tool via the
    text-to-audio (TTS) and automatic-speech-recognition (ASR) tasks - not by
    the model provider. This demo proves a real, intelligible round-trip:
    mms-tts synthesizes speech to a .wav, whisper-tiny transcribes it back.

    PYTHONPATH=. python examples/document_and_audio.py
"""

import asyncio
import re

from strands_transformers import use_transformers
from strands_transformers.models.transformers import TransformerModel

PHRASE = "the quick brown fox jumps over the lazy dog"


async def _collect(model, messages) -> str:
    out = ""
    async for ev in model.stream(messages):
        delta = ev.get("contentBlockDelta", {}).get("delta", {})
        if "text" in delta:
            out += delta["text"]
    return out.strip()


def test_document() -> bool:
    """A `document` content block flows into a text LM's prompt."""
    model = TransformerModel(
        model_path="Qwen/Qwen3-0.6B",
        enable_thinking=False,
        params={"max_tokens": 64, "do_sample": False},
    )
    body = b"The secret passphrase for the vault is BANANA-42. Keep it safe."
    messages = [
        {
            "role": "user",
            "content": [
                {"document": {"name": "secret", "format": "txt", "source": {"bytes": body}}},
                {
                    "text": "What is the secret passphrase in the document? Answer with just the passphrase."
                },
            ],
        }
    ]
    ans = asyncio.run(_collect(model, messages))
    print("[document] answer:", repr(ans[:100]))
    return "banana" in ans.lower()


def _words(x: str):
    return set(re.sub(r"[^a-z ]", "", x.lower()).split())


def test_audio_round_trip() -> bool:
    """text -> speech (.wav) -> text, with real models. Word overlap >= 4."""
    tts = use_transformers(
        action="run",
        task="text-to-audio",
        model="facebook/mms-tts-eng",
        inputs=PHRASE,
        label="mms-tts",
    )
    wav = tts.get("artifacts", [None])[0]
    print("[audio] tts status:", tts["status"], "| wav:", wav)
    if tts["status"] != "success" or not wav:
        return False

    asr = use_transformers(
        action="run",
        task="automatic-speech-recognition",
        model="openai/whisper-tiny",
        inputs=wav,
        label="whisper",
    )
    transcript = asr["content"][0]["text"]
    print("[audio] asr status:", asr["status"])
    overlap = _words(PHRASE) & _words(transcript)
    print("[audio] word overlap:", len(overlap), "/", len(_words(PHRASE)))
    return asr["status"] == "success" and len(overlap) >= 4


def main() -> int:
    ok_doc = test_document()
    ok_audio = test_audio_round_trip()
    ok = ok_doc and ok_audio
    print("status:", "success" if ok else "unexpected")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
