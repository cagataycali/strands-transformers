"""Multimodal pipelines via use_transformers — text, image & audio, the easy way.

Where the VLA examples (molmoact_vla.py, openvla_vla.py) use the low-level `call`
layer for custom model APIs, MOST transformers tasks need no such ceremony: the
high-level `run` action wraps transformers.pipeline(), which natively accepts
file paths / URLs / PIL images / numpy arrays and returns structured results.

This example exercises several modalities with small, fast models so it runs
end-to-end quickly:

  • text-generation        (text → text)
  • text-classification    (text → label)
  • image-classification   (image → labels)
  • zero-shot-classification (text + candidate labels → ranked labels)

Run directly:   python examples/multimodal_pipelines.py
"""

import numpy as np
from PIL import Image

from strands_transformers import use_transformers


def demo_text_generation():
    return use_transformers(
        action="run",
        task="text-generation",
        model="sshleifer/tiny-gpt2",
        inputs="The robot picked up the",
        parameters={"max_new_tokens": 8},
        label="tiny text-generation",
    )


def demo_text_classification():
    return use_transformers(
        action="run",
        task="text-classification",
        model="hf-internal-testing/tiny-random-distilbert",
        inputs="Strands makes agents easy.",
        label="tiny sentiment",
    )


def demo_image_classification():
    img = Image.fromarray(
        np.random.default_rng(0).integers(0, 255, (64, 64, 3), dtype=np.uint8)
    )
    return use_transformers(
        action="run",
        task="image-classification",
        model="hf-internal-testing/tiny-random-vit",
        inputs=img,
        label="tiny image-classification",
    )


def demo_zero_shot():
    return use_transformers(
        action="run",
        task="zero-shot-classification",
        model="hf-internal-testing/tiny-random-bart",
        inputs="The arm moved the cube into the bowl.",
        parameters={"candidate_labels": ["robotics", "cooking", "finance"]},
        label="tiny zero-shot",
    )


DEMOS = {
    "text-generation": demo_text_generation,
    "text-classification": demo_text_classification,
    "image-classification": demo_image_classification,
    "zero-shot-classification": demo_zero_shot,
}


if __name__ == "__main__":
    for name, fn in DEMOS.items():
        result = fn()
        status = result["status"]
        print(f"\n=== {name}: {status} ===")
        print(result["content"][0]["text"][:500])
