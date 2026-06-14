"""Vision tasks via use_transformers run path — detection, embeddings, classification.

Shows that structured vision outputs (bounding boxes, embedding tensors, label
scores) all serialize to JSON-safe results automatically. Uses tiny models so it
runs end-to-end fast.

    PYTHONPATH=. python examples/vision_tasks.py
"""

import numpy as np
from PIL import Image

from strands_transformers import use_transformers


def _img(seed: int = 0, size=(64, 64)) -> Image.Image:
    rng = np.random.default_rng(seed)
    return Image.fromarray(rng.integers(0, 255, (*size, 3), dtype=np.uint8))


def demo_object_detection():
    return use_transformers(
        action="run",
        task="object-detection",
        model="hf-internal-testing/tiny-detr-mobilenetsv3",
        inputs=_img(1),
        label="tiny object-detection",
    )


def demo_image_feature_extraction():
    return use_transformers(
        action="run",
        task="image-feature-extraction",
        model="hf-internal-testing/tiny-random-vit",
        inputs=_img(2),
        label="tiny image embeddings",
    )


def demo_image_classification():
    return use_transformers(
        action="run",
        task="image-classification",
        model="hf-internal-testing/tiny-random-vit",
        inputs=_img(3),
        label="tiny image-classification",
    )


def demo_depth_estimation():
    # dense image output → depth map saved as a PNG artifact
    return use_transformers(
        action="run",
        task="depth-estimation",
        model="Intel/dpt-hybrid-midas",
        inputs=_img(4, size=(96, 96)),
        label="depth-estimation",
    )


def demo_image_segmentation():
    return use_transformers(
        action="run",
        task="image-segmentation",
        model="facebook/detr-resnet-50-panoptic",
        inputs=_img(5, size=(96, 96)),
        label="image-segmentation",
    )


DEMOS = {
    "object-detection": demo_object_detection,
    "image-feature-extraction": demo_image_feature_extraction,
    "image-classification": demo_image_classification,
    "depth-estimation": demo_depth_estimation,
    "image-segmentation": demo_image_segmentation,
}


if __name__ == "__main__":
    for name, fn in DEMOS.items():
        r = fn()
        print(f"\n=== {name}: {r['status']} ===")
        print(r["content"][0]["text"][:300])
