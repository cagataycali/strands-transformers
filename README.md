<div align="center">

  <h1>
    Strands Transformers
  </h1>

  <h2>
    Use local HuggingFace models with Strands + automatic training data collection
  </h2>

  <div align="center">
    <a href="https://github.com/cagataycali/strands-transformers/issues"><img alt="GitHub open issues" src="https://img.shields.io/github/issues/cagataycali/strands-transformers"/></a>
    <a href="https://github.com/cagataycali/strands-transformers/blob/main/LICENSE"><img alt="License" src="https://img.shields.io/github/license/cagataycali/strands-transformers"/></a>
    <a href="https://python.org"><img alt="Python versions" src="https://img.shields.io/badge/python-3.10+-blue"/></a>
  </div>
</div>

Strands Transformers lets you use any HuggingFace model as a Strands model provider and automatically collect training data from your conversations. Train domain-specific agents that understand your tools and workflows using LoRA fine-tuning.

## Feature Overview

- **TransformerModel Provider**: Use any HuggingFace model (Qwen, Llama, Mistral, etc.)
- **Automatic Training Collection**: JsonlSessionManager captures conversations with tool calls
- **LoRA Fine-Tuning**: Efficient training with TRL SFTTrainer (297MB vs 6.4GB)
- **Template System**: Built-in support for Qwen3, Llama 3.1/3.2, GPT-OSS chat formats
- **Thinking Mode**: Qwen3 internal reasoning with `<think>` tags
- **Tool Calling**: Automatic XML-based tool calling and response formatting

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Or install as package
pip install -e .
```

```python
from strands import Agent
from strands_transformers import TransformerModel, JsonlSessionManager
from strands_tools import shell, calculator

# 1. Load local model
model = TransformerModel(
    model_path="Qwen/Qwen3-1.7B",
    device="auto"
)

# 2. Enable auto-training collection (optional)
session = JsonlSessionManager(
    session_id="my_agent",
    template_name="qwen3"
)

# 3. Create agent
agent = Agent(
    model=model,
    tools=[shell, calculator],
    session_manager=session  # Conversations → training data
)

# 4. Use agent (data collected automatically!)
agent("What tools do you have?")
agent("Calculate 25 * 17")
```

> **Note**: This project is experimental. We're actively training models to understand Strands - best results so far with 50+ examples and 70+ training steps.

## Installation

```bash
git clone git@github.com:cagataycali/strands-transformers.git
cd strands-transformers

# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

## Core Components

### TransformerModel - HuggingFace as Strands Provider

```python
from strands_transformers import TransformerModel

model = TransformerModel(
    model_path="Qwen/Qwen3-1.7B",  # Or local path
    device="auto",                  # cuda/mps/cpu
    enable_thinking=True,           # Qwen3 thinking mode
    params={
        "max_tokens": 500,
        "temperature": 0.7
    }
)
```

**Features:**
- ✅ Streaming responses
- ✅ Tool calling with XML tags
- ✅ Qwen3 thinking mode (`<think>` tags)
- ✅ Chat template auto-detection
- ✅ Merged LoRA weights support

### JsonlSessionManager - Automatic Training Data

```python
from strands_transformers import JsonlSessionManager

session = JsonlSessionManager(
    session_id="production",
    template_name="qwen3",           # qwen3, llama3.1, llama3.2, gpt-oss
    storage_dir="./training_data"
)

agent = Agent(model=model, tools=[...], session_manager=session)

# Every conversation automatically saved to JSONL!
```

**Captures:**
- System prompts with tool definitions
- User messages
- Tool calls (`<tool_call>` tags)
- Tool responses (`<tool_response>` tags)
- Complete conversation context

### Training Pipeline

```python
from strands_transformers import model_trainer

# Train with LoRA
model_trainer(
    action="train",
    model_name="Qwen/Qwen3-1.7B",
    dataset="./training_data/production.jsonl",
    output_dir="./trained",
    use_lora=True,
    max_steps=150
)

# Merge weights
model_trainer(
    action="load_for_inference",
    model_name="./trained",
    output_dir="./merged"
)

# Use fine-tuned model
fine_tuned = TransformerModel(model_path="./merged", device="auto")
agent = Agent(model=fine_tuned, tools=[...])
```

## Available Templates

| Template | Model Type | Features |
|----------|------------|----------|
| `qwen3.j2` | Qwen3 | Thinking mode, tool calling |
| `llama3.1.j2` | Llama 3.1 (8B+) | Tool calling, ipython format |
| `llama3.2.j2` | Llama 3.2 (3B) | Optimized for smaller models |
| `gpt-oss.j2` | GPT-OSS | Multi-channel reasoning |

## Example: Training Flow

```python
# Phase 1: Collect data
session = JsonlSessionManager(session_id="my_app", template_name="qwen3")
agent = Agent(model=model, tools=[...], session_manager=session)
# ... use agent in production ...

# Phase 2: Train
model_trainer(action="train", dataset="~/.strands/training_data/my_app.jsonl", ...)

# Phase 3: Deploy
trained_model = TransformerModel(model_path="./merged", device="auto")
agent = Agent(model=trained_model, tools=[...], session_manager=session)  # Continue learning!
```

## Documentation

- **Installation**: Clone repo and `pip install -r requirements.txt`
- **TransformerModel**: See `strands_transformers/models/transformers.py`
- **JsonlSessionManager**: See `strands_transformers/session/jsonl_session_manager.py`
- **Training Tools**: See `strands_transformers/tools/`
- **Templates**: See `strands_transformers/templates/`

## Contributing

We welcome contributions! 

1. Fork the repository
2. Create feature branch: `git checkout -b feature/my-feature`
3. Make changes and test
4. Commit: `git commit -m "feat: add feature"`
5. Push and open Pull Request

## License

MIT License - see [LICENSE](LICENSE) file for details

---

**Built with** [Strands Agents SDK](https://github.com/strands-agents/sdk-python) • [HuggingFace Transformers](https://huggingface.co/transformers) • [PEFT](https://github.com/huggingface/peft)
