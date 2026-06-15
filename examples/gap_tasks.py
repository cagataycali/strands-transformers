"""More transformers tasks through the `run` path - zero-shot + audio + video.

Rounds out the example coverage with tasks not shown elsewhere, all via the
high-level `run` action (no custom code):

  • zero-shot-image-classification  - CLIP: classify into labels you invent
  • zero-shot-object-detection      - OWL-ViT: detect arbitrary prompts
  • audio-classification            - wav2vec2: label a waveform
  • video-classification            - VideoMAE: label a clip (frame list → (T,H,W,C))

Tiny/standard models so it runs without huge downloads.

    PYTHONPATH=. python examples/gap_tasks.py

E2E verified: each returns status=="success" with real structured data.
"""

import numpy as np
from PIL import Image

from strands_transformers import use_transformers


def _img(seed: int = 0) -> Image.Image:
    rng = np.random.default_rng(seed)
    return Image.fromarray(rng.integers(0, 255, (120, 120, 3), dtype=np.uint8))


def zero_shot_image():
    return use_transformers(
        action="run",
        task="zero-shot-image-classification",
        model="openai/clip-vit-base-patch32",
        inputs=_img(),
        parameters={"candidate_labels": ["a cat", "a dog", "random noise"]},
        label="zero-shot image-cls",
    )


def zero_shot_detect():
    return use_transformers(
        action="run",
        task="zero-shot-object-detection",
        model="google/owlvit-base-patch32",
        inputs=_img(),
        parameters={"candidate_labels": ["a cat", "a remote"]},
        label="zero-shot detection",
    )


def audio_classify():
    sr = 16000
    tone = np.sin(2 * np.pi * 440 * np.arange(sr) / sr).astype("float32")
    return use_transformers(
        action="run",
        task="audio-classification",
        model="hf-internal-testing/tiny-random-wav2vec2",
        inputs={"raw": tone, "sampling_rate": sr},
        label="audio-cls",
    )


def video_classify():
    frames = [
        np.random.default_rng(i).integers(0, 255, (224, 224, 3), dtype=np.uint8) for i in range(16)
    ]
    return use_transformers(
        action="run",
        task="video-classification",
        model="MCG-NJU/videomae-base-finetuned-kinetics",
        inputs=frames,
        label="video-cls",
    )


DEMOS = {
    "zero-shot-image-classification": zero_shot_image,
    "zero-shot-object-detection": zero_shot_detect,
    "audio-classification": audio_classify,
    "video-classification": video_classify,
}


def main() -> int:
    ok = True
    for name, fn in DEMOS.items():
        r = fn()
        passed = r["status"] == "success"
        ok = ok and passed
        data = r.get("data")
        n = len(data) if isinstance(data, list) else "?"
        print(f"  [{'PASS' if passed else 'FAIL'}] {name} → {n} result(s)")
    print("status:", "success" if ok else "unexpected")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
