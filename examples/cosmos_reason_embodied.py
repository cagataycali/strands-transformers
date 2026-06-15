"""Cosmos-Reason2 - embodied/physical-AI reasoning over a robot scene.

NVIDIA's Cosmos-Reason2-2B is a transformers-native vision-language model
(`Qwen3VLForConditionalGeneration`) tuned for *physical* reasoning: given a
camera view and a goal, it reasons about where things are and what a robot
should do. Unlike a VLA model (which emits raw joint actions via a custom
`predict_action`), Cosmos-Reason is a standard `image-text-to-text` VLM, so it
runs through the high-level `run` path - and slots straight into an agent that
then calls a low-level VLA for the actual motor commands.

    PYTHONPATH=. python examples/cosmos_reason_embodied.py

Requires nvidia/Cosmos-Reason2-2B (~9 GB). E2E verified on NVIDIA (CUDA):
a red cube on the lower-left → "...the robot arm should first move to the
bottom left corner to grasp it."
"""

import numpy as np
from PIL import Image

from strands_transformers import use_transformers

MODEL = "nvidia/Cosmos-Reason2-2B"


def scene_with_red_cube() -> Image.Image:
    """A gray tabletop with a red cube in the lower-left quadrant."""
    arr = np.full((256, 256, 3), (180, 180, 180), dtype=np.uint8)
    arr[150:210, 40:100] = (200, 30, 30)  # red cube, lower-left
    return Image.fromarray(arr)


def reason(image: Image.Image, goal: str) -> dict:
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": goal},
            ],
        }
    ]
    return use_transformers(
        action="run",
        task="image-text-to-text",
        model=MODEL,
        inputs={"text": messages},
        parameters={"max_new_tokens": 96, "do_sample": False},
        label="Cosmos-Reason embodied",
    )


def _assistant_text(result: dict) -> str:
    """Pull the assistant's reply out of an image-text-to-text run result."""
    import json
    import re

    blob = result["content"][0]["text"]
    # The pipeline returns a chat structure; grab the assistant content.
    m = re.search(r'"role":\s*"assistant",\s*"content":\s*"((?:[^"\\]|\\.)*)"', blob)
    if m:
        return json.loads('"' + m.group(1) + '"')
    return blob


def main() -> int:
    result = reason(
        scene_with_red_cube(),
        "A robot arm must pick up the red cube. Where is it and what should the "
        "arm do first? Answer in one sentence.",
    )
    print("status:", result["status"])
    answer = _assistant_text(result)
    print("reasoning:", answer[:400])
    ok = result["status"] == "success" and ("left" in answer.lower() or "red" in answer.lower())
    print("verdict:", "success" if ok else "unexpected")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
