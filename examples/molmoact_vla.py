"""MolmoAct2-SO100_101 — vision-language-action via use_transformers (no hardcoding).

allenai/MolmoAct2-SO100_101 is a VLA robotics model: it takes camera images + a
language task + the robot's joint state, and predicts continuous robot actions.

It exposes a custom `predict_action` method (not a standard pipeline), so we use
the LOW-LEVEL `call` layer of use_transformers to:
  1. load the AutoProcessor   (cache_key="molmo_proc")
  2. load the AutoModelForImageTextToText  (cache_key="molmo")
  3. call cached:molmo.predict_action(processor=cached:molmo_proc, images=..., state=...)

This is the same dynamic-dispatch pattern as use_aws/use_lerobot — the model's
own API is discovered and driven through one universal tool.

Run directly (real inference):   python examples/molmoact_vla.py
"""

import numpy as np
from huggingface_hub import hf_hub_download
from PIL import Image

from strands_transformers import use_transformers

REPO = "allenai/MolmoAct2-SO100_101"
NORM_TAG = "so100_so101_molmoact2"


def load_sample():
    top = Image.open(hf_hub_download(REPO, "assets/sample_realsense_top_rgb.png")).convert("RGB")
    side = Image.open(hf_hub_download(REPO, "assets/sample_realsense_side_rgb.png")).convert("RGB")
    task = "Move the arm towards the lemon, grasp it, lift it up, and drop it into the red bowl."
    state = np.array(
        [-0.52734375, 189.140625, 181.40625, 60.64453125, -3.603515625, 1.0971786975860596],
        dtype=np.float32,
    )
    return top, side, task, state


def predict_action(top, side, task, state, num_steps: int = 10):
    """Drive MolmoAct's predict_action entirely through use_transformers."""
    # 1. processor
    use_transformers(
        action="call",
        target="AutoProcessor.from_pretrained",
        parameters={"pretrained_model_name_or_path": REPO, "trust_remote_code": True},
        cache_key="molmo_proc",
    )
    # 2. model (float32 for broad compatibility; bf16 also supported on CUDA)
    use_transformers(
        action="call",
        target="AutoModelForImageTextToText.from_pretrained",
        parameters={
            "pretrained_model_name_or_path": REPO,
            "trust_remote_code": True,
            "dtype": "float32",
        },
        cache_key="molmo",
    )
    # 3. predict actions — processor passed by cached reference
    return use_transformers(
        action="call",
        target="cached:molmo.predict_action",
        parameters={
            "processor": "cached:molmo_proc",
            "images": [top, side],
            "task": task,
            "state": state.tolist(),
            "norm_tag": NORM_TAG,
            "inference_action_mode": "continuous",
            "enable_depth_reasoning": False,
            "num_steps": num_steps,
            "normalize_language": True,
        },
        label="MolmoAct predict_action",
    )


if __name__ == "__main__":
    top, side, task, state = load_sample()
    print(f"Task: {task}\nState: {state}")
    result = predict_action(top, side, task, state)
    print("status:", result["status"])
    print(result["content"][0]["text"][:1500])
