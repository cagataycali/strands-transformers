# Strands Transformers

Use local HuggingFace models as Strands providers + fine-tuning infrastructure.

---

## Quick Start

```bash
git clone git@github.com:cagataycali/strands-transformers.git
cd strands-transformers

python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

python3 agent.py  # Test trained model
```

---

## What's Here

```
models/transformers.py          # HuggingFace → Strands provider
tools/
  ├── dataset_generator.py      # Generate training datasets
  ├── model_trainer.py          # Train & merge with LoRA
  └── template.py               # Manage chat templates
templates/*.j2                  # Qwen3, Llama 3.1/3.2, GPT-OSS
strands.jsonl                   # 50 training examples
```

---

## Use Trained Model

```python
from models.transformers import TransformerModel
from strands import Agent

model = TransformerModel(model_path="./qwen3_1.7b_strands_final", device="auto")
agent = Agent(model=model)
agent("What is Strands Agents?")
```

---

## Generate Dataset

**Option 1: Manual (dataset_generator)**
```python
from tools.dataset_generator import dataset_generator

dataset_generator(
    action="generate",
    template_name="qwen3",
    output_file="my_data.jsonl",
    examples=[{"instruction": "Q", "response": "A"}]
)
```

**Option 2: Automatic (JsonlSessionManager)**
```python
from jsonl_session_manager import JsonlSessionManager

# Auto-collect training data from conversations
session = JsonlSessionManager(session_id="training", template_name="qwen3")
agent = Agent(model=model, session_manager=session)
agent("What is Python?")  # Auto-saved to ~/.strands/training_data/training.jsonl
```

---

## Train Model

```python
from tools.model_trainer import model_trainer

model_trainer(action="train", model_name="Qwen/Qwen3-1.7B", 
              dataset="my_data.jsonl", output_dir="./trained", use_lora=True)
model_trainer(action="load_for_inference", model_name="./trained", 
              output_dir="./merged")
```

---

## Key Features

- ✅ TransformerModel provider for any HF model
- ✅ LoRA fine-tuning (297MB adapter vs 6.4GB full model)
- ✅ 4 chat templates (Qwen3, Llama 3.1/3.2, GPT-OSS)
- ✅ Tool calling & thinking mode support
- ✅ 50 Strands SDK examples included

---

**Built by research agent • Foundation ready • Iterate to improve**
