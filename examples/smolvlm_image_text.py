"""SmolVLM image-text-to-text via the run path — a real VLM, no custom code.

Unlike the VLA models (which need the low-level `call` layer for predict_action),
a vision-language *chat* model like HuggingFaceTB/SmolVLM-256M-Instruct is a
standard `image-text-to-text` pipeline — so it runs through the high-level `run`
action. Images go inside the chat message content; the model returns generated
text describing/answering about the image.

    PYTHONPATH=. python examples/smolvlm_image_text.py
"""

import numpy as np
from PIL import Image

from strands_transformers import use_transformers

MODEL = "HuggingFaceTB/SmolVLM-256M-Instruct"


def sample_image(seed: int = 0) -> Image.Image:
    rng = np.random.default_rng(seed)
    return Image.fromarray(rng.integers(0, 255, (224, 224, 3), dtype=np.uint8))


def describe(image: Image.Image, question: str = "Describe this image in one word."):
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": question},
            ],
        }
    ]
    return use_transformers(
        action="run",
        task="image-text-to-text",
        model=MODEL,
        inputs={"text": messages},
        parameters={"max_new_tokens": 32},
        label="SmolVLM image-text-to-text",
    )


if __name__ == "__main__":
    result = describe(sample_image())
    print("status:", result["status"])
    print(result["content"][0]["text"][:600])
