"""A Strands Agent whose brain is a LOCAL vision-language model.

This is the payoff of the multimodal `TransformerModel` provider: you build a
normal `strands.Agent`, hand it a `TransformerModel` pointing at any HF
vision-language model, and pass **image content blocks** in the conversation.
The provider routes image/video/document blocks (including media returned
inside a `toolResult`) through the model's `AutoProcessor` so the model
actually *sees* the pixels — no servers, no API keys, fully local.

Multimodal is auto-detected: if the model ships an `AutoProcessor` with an
`image_processor`, the provider loads it as `AutoModelForImageTextToText`
(falling back to `AutoModelForVision2Seq`) and uses the processor chat
template. Text-only models keep the original tokenizer fast-path unchanged.

    PYTHONPATH=. python examples/multimodal_agent.py

Verified E2E on HuggingFaceTB/SmolVLM-256M-Instruct (Idefics3Processor):
solid-color test images are correctly named by the agent.
"""

import io

from PIL import Image
from strands import Agent

from strands_transformers.models.transformers import TransformerModel

MODEL = "HuggingFaceTB/SmolVLM-256M-Instruct"


def solid_png(rgb, size=224) -> bytes:
    """A solid-color PNG as raw bytes (Strands ImageContent.source.bytes)."""
    buf = io.BytesIO()
    Image.new("RGB", (size, size), rgb).save(buf, format="PNG")
    return buf.getvalue()


def main() -> int:
    # Build a local VLM-backed agent. Multimodal is auto-detected.
    model = TransformerModel(
        model_path=MODEL,
        params={"max_tokens": 32, "do_sample": False},
    )
    assert model.is_multimodal, "expected a multimodal processor to be detected"

    agent = Agent(model=model, system_prompt="You are a concise vision assistant.")

    # Pass an image content block + a text block — Bedrock/Strands shape.
    result = agent(
        [
            {"image": {"format": "png", "source": {"bytes": solid_png((20, 200, 40))}}},
            {"text": "What color is this image? Answer in one word."},
        ]
    )

    answer = str(result).strip()
    print("agent answer:", repr(answer))
    ok = bool(answer) and "green" in answer.lower()
    print("status:", "success" if ok else "unexpected")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
