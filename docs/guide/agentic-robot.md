# Agentic robot loop: reason → act

The showcase: a **real two-step robot agent** on real RealSense camera images,
entirely through the single `use_transformers` tool. One model *plans*, another
*acts* — perception to motor commands, no per-model glue.

```mermaid
sequenceDiagram
    autonumber
    participant C as 📷 cameras + task
    participant R as 🧠 Cosmos-Reason2<br/>(run · image-text-to-text)
    participant A as ⚙️ MolmoAct2<br/>(call · predict_action)
    participant Rb as 🤖 robot

    C->>R: top view + "pick the lemon, drop in bowl"
    R-->>C: plan — "lemon center, bowl right; move arm to lemon first"
    C->>A: top+side views + task + joint state
    A-->>Rb: actions [1, 30, 6]
```

```python
# examples/robot_reason_act_agent.py  (abridged)
from strands_transformers import use_transformers

# STEP 1 — PLAN: Cosmos-Reason2 reasons over the scene (high-level run path)
plan = use_transformers(action="run", task="image-text-to-text",
    model="nvidia/Cosmos-Reason2-2B",
    inputs={"text": [{"role": "user", "content": [
        {"type": "image", "image": top_view},
        {"type": "text", "text": f"A robot must: {task} Where are the objects and what is the first motion?"},
    ]}]},
    parameters={"max_new_tokens": 100, "do_sample": False})

# STEP 2 — ACT: MolmoAct2's predict_action emits joint actions (low-level call path)
use_transformers(action="call", target="AutoProcessor.from_pretrained",
    parameters={"pretrained_model_name_or_path": VLA, "trust_remote_code": True}, cache_key="proc")
use_transformers(action="call", target="AutoModelForImageTextToText.from_pretrained",
    parameters={"pretrained_model_name_or_path": VLA, "trust_remote_code": True, "dtype": "float32"}, cache_key="vla")
actions = use_transformers(action="call", target="cached:vla.predict_action",
    parameters={"processor": "cached:proc", "images": [top_view, side_view],
                "task": task, "state": joint_state, "norm_tag": "so100_so101_molmoact2"})
# → MolmoAct2ActionOutput.actions, shape [1, 30, 6]
```

## Real run

`PYTHONPATH=. python examples/robot_reason_act_agent.py`

| Step | Model | Real output |
|------|-------|-------------|
| 🧠 plan | Cosmos-Reason2-2B | *"The lemon is near the center of the table … the first motion should be moving the robot arm towards the lemon to prepare for grasping it."* |
| ⚙️ act | MolmoAct2-SO100_101 | `MolmoAct2ActionOutput.actions` — shape `[1, 30, 6]` |

## As a Strands Agent

The whole loop wraps as a `@tool`, so an agent triggers it from one sentence:

```python
from strands import Agent, tool
from examples.robot_reason_act_agent import load_scene, reason_then_act

@tool
def operate_robot() -> str:
    """Perceive the tabletop, plan with Cosmos-Reason, act with MolmoAct."""
    top, side, task, state = load_scene()
    out = reason_then_act(top, side, task, state)
    return f"Plan: {out['plan']}\nActions emitted [1,30,6]."

agent = Agent(tools=[operate_robot])
agent("Clean up the table.")   # → agent calls operate_robot → plan + actions
```

!!! tip "Why two models"
    Cosmos-Reason is a general physical-AI reasoner (great at *what to do*);
    MolmoAct is a trained VLA policy (great at *the exact joint trajectory*).
    Splitting reason from act is the standard pattern for reliable robot agents —
    and here both run locally through one tool.

See **[Robotics / VLA](robotics.md)** for the model landscape.
