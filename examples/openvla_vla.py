"""OpenVLA-7B — vision-language-action via use_transformers (no hardcoding).

openvla/openvla-7b is a 7-DoF VLA: given a wrist-camera image + an instruction,
it predicts a normalized robot action that you un-normalize per embodiment
(`unnorm_key`, e.g. "bridge_orig").

OpenVLA exposes a custom `predict_action` method, so we drive it through
use_transformers' low-level `call` layer — the exact pattern from the model card,
`action = vla.predict_action(**processor(prompt, image), unnorm_key=...)`,
expressed with cached objects and the "**" unpack key:

  1. load AutoProcessor                        → cache_key="ovla_proc"
  2. load the VLA model                         → cache_key="ovla"
  3. processor(text=prompt, images=image)       → cache_key="ovla_batch"
  4. predict_action(**ovla_batch, unnorm_key)   via {"**": "cached:ovla_batch"}

Note: OpenVLA shipped for transformers 4.x where the auto-class was
`AutoModelForVision2Seq`; on 5.x that role is `AutoModelForImageTextToText`.
We pick whichever exists.

Run directly (real inference):   python examples/openvla_vla.py
"""

import numpy as np
from PIL import Image

from strands_transformers import use_transformers
from strands_transformers.core import compat

# OpenVLA's auto_map registers under "AutoModelForVision2Seq" (a transformers 4.x
# class removed in 5.x). compat.apply() recreates it as an alias so the model's
# remote code resolves. The auto-class name must match the auto_map key exactly.
compat.apply()

REPO = "openvla/openvla-7b"
UNNORM_KEY = "bridge_orig"
MODEL_AUTO_CLASS = "AutoModelForVision2Seq"


def sample_image(seed: int = 0) -> Image.Image:
    """Placeholder 224×224 RGB observation (swap for a real wrist-cam frame)."""
    rng = np.random.default_rng(seed)
    return Image.fromarray(rng.integers(0, 255, (224, 224, 3), dtype=np.uint8))


def predict_action(image: Image.Image, instruction: str):
    prompt = f"In: What action should the robot take to {instruction}?\nOut:"
    # OpenVLA hard-pins timm to 0.9.x in its remote code; newer timm is generally
    # inference-compatible, so we spoof the version string during load.
    _timm_ctx = compat.spoof_timm_version()
    _timm_ctx.__enter__()

    use_transformers(
        action="call",
        target="AutoProcessor.from_pretrained",
        parameters={"pretrained_model_name_or_path": REPO, "trust_remote_code": True},
        cache_key="ovla_proc",
    )
    use_transformers(
        action="call",
        target=f"{MODEL_AUTO_CLASS}.from_pretrained",
        parameters={
            "pretrained_model_name_or_path": REPO,
            "trust_remote_code": True,
            "dtype": "bfloat16",
            "low_cpu_mem_usage": True,
            "device_map": "cuda",
            "attn_implementation": "eager",
        },
        cache_key="ovla",
    )
    _timm_ctx.__exit__(None, None, None)
    use_transformers(
        action="call",
        target="cached:ovla_proc",
        parameters={"text": prompt, "images": image, "return_tensors": "pt"},
        cache_key="ovla_batch",
    )
    # Move the processed batch onto the model's device (cuda) + dtype.
    use_transformers(
        action="call",
        target="cached:ovla_batch.to",
        parameters={"device": "cuda"},
        cache_key="ovla_batch",
    )
    return use_transformers(
        action="call",
        target="cached:ovla.predict_action",
        parameters={"**": "cached:ovla_batch", "unnorm_key": UNNORM_KEY, "do_sample": False},
        label="OpenVLA predict_action",
    )


if __name__ == "__main__":
    print("auto-class:", MODEL_AUTO_CLASS)
    result = predict_action(sample_image(), "pick up the red block")
    print("status:", result["status"])
    print(result["content"][0]["text"][:1500])
