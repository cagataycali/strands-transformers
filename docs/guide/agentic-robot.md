# Agentic robot loop: reason → act

The showcase: a **real two-step robot agent** on real RealSense camera images,
entirely through the single `use_transformers` tool. One model *plans*, another
*acts* — perception to motor commands, no per-model glue.

```mermaid
flowchart LR
    C["📷 cameras + task"]
    R["🧠 Cosmos-Reason2<br/>run · image-text-to-text"]
    A["⚙️ MolmoAct2<br/>call · predict_action"]
    Rb["🤖 robot"]

    C -->|"top view + 'pick the lemon'"| R
    R -->|"plan: lemon center, move arm there first"| A
    C -->|"top+side views + task + joint state"| A
    A -->|"actions 1×30×6"| Rb

    classDef in fill:#7C4DFF,stroke:#5b34d6,color:#fff;
    classDef mid fill:#FFD21E,stroke:#E68A00,color:#3a2d00;
    classDef out fill:#00E5FF,stroke:#00b3cc,color:#003844;
    class C in;
    class R,A mid;
    class Rb out;
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
