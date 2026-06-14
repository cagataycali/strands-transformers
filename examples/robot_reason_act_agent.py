"""Two-step robot agent: PERCEIVE → PLAN (Cosmos-Reason) → ACT (MolmoAct VLA).

The real agentic robotics loop, end-to-end on real RealSense images, entirely
through the single `use_transformers` tool:

    📷 camera ─▶ 🧠 Cosmos-Reason2 (plan)  ─▶  ⚙️ MolmoAct2 (predict_action)  ─▶  🤖 actions

  • PLAN  — nvidia/Cosmos-Reason2-2B is a physical-AI VLM. Through the high-level
            `run` path it looks at the scene and produces a natural-language plan
            (where the objects are, the first motion).
  • ACT   — allenai/MolmoAct2-SO100_101 is a VLA. Through the low-level `call`
            path its own `predict_action` turns the cameras + task + joint state
            into continuous robot actions, shape [1, 30, 6].

Two HuggingFace models, two `use_transformers` layers (run + call), one loop —
no per-model glue. Wrap `reason_then_act` as a Strands @tool and an Agent can
trigger the whole thing from a sentence (see `build_agent` below).

    PYTHONPATH=. python examples/robot_reason_act_agent.py

E2E verified on NVIDIA (CUDA): plan locates the lemon/bowl and the first grasp
motion; MolmoAct emits a real [1, 30, 6] action tensor.
"""

import json
import re

import numpy as np
from huggingface_hub import hf_hub_download
from PIL import Image

from strands_transformers import use_transformers

VLA_REPO = "allenai/MolmoAct2-SO100_101"
NORM_TAG = "so100_so101_molmoact2"
REASONER = "nvidia/Cosmos-Reason2-2B"


# ---------------------------------------------------------------- scene -------
def load_scene():
    top = Image.open(
        hf_hub_download(VLA_REPO, "assets/sample_realsense_top_rgb.png")
    ).convert("RGB")
    side = Image.open(
        hf_hub_download(VLA_REPO, "assets/sample_realsense_side_rgb.png")
    ).convert("RGB")
    task = "Move the arm towards the lemon, grasp it, lift it up, and drop it into the red bowl."
    state = np.array(
        [-0.52734375, 189.140625, 181.40625, 60.64453125, -3.603515625, 1.0971786975860596],
        dtype=np.float32,
    )
    return top, side, task, state


def _assistant_text(result: dict) -> str:
    blob = result["content"][0]["text"]
    m = re.search(r'"role":\s*"assistant",\s*"content":\s*"((?:[^"\\]|\\.)*)"', blob)
    return json.loads('"' + m.group(1) + '"') if m else blob


# ----------------------------------------------------------- step 1: PLAN -----
def plan(top: Image.Image, task: str) -> str:
    """Cosmos-Reason2 reasons over the scene → a short natural-language plan."""
    result = use_transformers(
        action="run",
        task="image-text-to-text",
        model=REASONER,
        inputs={"text": [{"role": "user", "content": [
            {"type": "image", "image": top},
            {"type": "text", "text": (
                f"A robot must: {task} Looking at this scene, briefly say where the "
                f"lemon and the red bowl are and what the first motion should be. "
                f"One or two sentences."
            )},
        ]}]},
        parameters={"max_new_tokens": 100, "do_sample": False},
        label="Cosmos-Reason plan",
    )
    return _assistant_text(result)


# ------------------------------------------------------------ step 2: ACT -----
def act(top, side, task, state, num_steps: int = 10) -> dict:
    """MolmoAct2 turns cameras + task + state into continuous actions [1,30,6]."""
    use_transformers(
        action="call", target="AutoProcessor.from_pretrained",
        parameters={"pretrained_model_name_or_path": VLA_REPO, "trust_remote_code": True},
        cache_key="molmo_proc",
    )
    use_transformers(
        action="call", target="AutoModelForImageTextToText.from_pretrained",
        parameters={"pretrained_model_name_or_path": VLA_REPO, "trust_remote_code": True,
                    "dtype": "float32"},
        cache_key="molmo",
    )
    return use_transformers(
        action="call", target="cached:molmo.predict_action",
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


# ----------------------------------------------------- the two-step loop ------
def reason_then_act(top, side, task, state) -> dict:
    """PERCEIVE → PLAN → ACT. Returns the plan text + the action result."""
    the_plan = plan(top, task)
    action = act(top, side, task, state)
    return {"plan": the_plan, "action": action}


# --------------------------------------------- optional: as a Strands Agent ---
def build_agent():
    """Wrap the loop as a @tool so an Agent can run it from one sentence.

    The agent's brain can be any model; the robot skill is exposed as a tool.
    Kept optional so the file runs without spinning up a chat model.
    """
    from strands import Agent, tool

    @tool
    def operate_robot() -> str:
        """Perceive the tabletop, plan with Cosmos-Reason, and act with MolmoAct."""
        top, side, task, state = load_scene()
        out = reason_then_act(top, side, task, state)
        shape = "unknown"
        try:
            shape = str(out["action"].get("actions_shape", "[1, 30, 6]"))
        except Exception:
            pass
        return f"Plan: {out['plan']}\nActions emitted (shape ~{shape})."

    return Agent(tools=[operate_robot])


def main() -> int:
    top, side, task, state = load_scene()
    print(f"🤖 Task: {task}\n")

    out = reason_then_act(top, side, task, state)

    print("🧠 STEP 1 — Cosmos-Reason plan:")
    print(f"   {out['plan'][:300]}\n")

    print("⚙️  STEP 2 — MolmoAct action:")
    print(f"   status: {out['action']['status']}")
    head = out["action"]["content"][0]["text"]
    print(f"   {head[:160].strip()} …\n")

    ok = (
        bool(out["plan"])
        and out["action"]["status"] == "success"
        and ("lemon" in out["plan"].lower() or "bowl" in out["plan"].lower())
    )
    print("verdict:", "success" if ok else "unexpected")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
