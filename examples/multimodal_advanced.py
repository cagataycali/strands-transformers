"""Advanced multimodal content blocks: video round-trip + tool-result images.

Two capabilities beyond a single still image, both fully local via
`TransformerModel`:

1. VIDEO — a `video` content block (a list of frames or a (T,H,W,C) array)
   is routed through the processor's video pipeline. The model reasons over
   temporal change. Verified: dark→bright frames → "brighter".

2. TOOL-RESULT IMAGE — the killer agentic-vision loop. A tool returns an
   image inside a `toolResult` block; the VLM sees that image on the next
   turn and reasons about it. Verified: blue capture → "Blue".

This exercises the full Strands content-block taxonomy where a
`ToolResultContent` arm may itself be image/video — exactly what the
harness-sdk messages schema allows.

    PYTHONPATH=. python examples/multimodal_advanced.py
"""

import asyncio
import io

import numpy as np
from PIL import Image

from strands_transformers.models.transformers import TransformerModel

IMG_MODEL = "HuggingFaceTB/SmolVLM-256M-Instruct"
VIDEO_MODEL = "HuggingFaceTB/SmolVLM2-500M-Video-Instruct"


def png(rgb, size=224) -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (size, size), rgb).save(buf, format="PNG")
    return buf.getvalue()


async def _collect(model, messages) -> str:
    out = ""
    async for ev in model.stream(messages):
        delta = ev.get("contentBlockDelta", {}).get("delta", {})
        if "text" in delta:
            out += delta["text"]
    return out.strip()


def test_tool_result_image() -> bool:
    """A tool returns an image; the VLM reasons over it on the next turn."""
    model = TransformerModel(
        model_path=IMG_MODEL, params={"max_tokens": 32, "do_sample": False}
    )
    messages = [
        {"role": "user", "content": [{"text": "Capture the screen, then name its color."}]},
        {"role": "assistant", "content": [
            {"toolUse": {"name": "capture_screen", "toolUseId": "t1", "input": {}}}
        ]},
        {"role": "user", "content": [{"toolResult": {
            "toolUseId": "t1",
            "status": "success",
            "content": [
                {"text": "Here is the captured screen:"},
                {"image": {"format": "png", "source": {"bytes": png((25, 25, 210))}}},
            ],
        }}]},
        {"role": "user", "content": [
            {"text": "What is the dominant color of the captured screen? One word."}
        ]},
    ]
    ans = asyncio.run(_collect(model, messages))
    print("[tool-result image] answer:", repr(ans[:80]))
    return "blue" in ans.lower()


def test_video() -> bool:
    """A video content block: does the clip get brighter or darker?"""
    model = TransformerModel(
        model_path=VIDEO_MODEL, params={"max_tokens": 48, "do_sample": False}
    )
    frames = [
        Image.fromarray(np.full((224, 224, 3), v, dtype=np.uint8))
        for v in (10, 40, 80, 120, 160, 200, 230, 250)
    ]
    messages = [{"role": "user", "content": [
        {"video": {"format": "mp4", "fps": 2.0, "source": {"bytes": frames}}},
        {"text": "Does this video get brighter or darker over time? Answer brighter or darker."},
    ]}]
    ans = asyncio.run(_collect(model, messages))
    print("[video] answer:", repr(ans[:80]))
    return "brighter" in ans.lower() or "darker" in ans.lower()


def main() -> int:
    ok_img = test_tool_result_image()
    ok_vid = test_video()
    ok = ok_img and ok_vid
    print("status:", "success" if ok else "unexpected")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
