# Strands Transformers

**Local HuggingFace models as Strands providers + comprehensive fine-tuning infrastructure.**

Train custom AI agents that understand your domain, tools, and workflows using local transformers models with automatic training data collection.

---

## 🎯 Overview

Strands Transformers provides three core capabilities:

1. **TransformerModel Provider** - Use any HuggingFace model as a Strands model provider
2. **Automatic Training Data Collection** - Capture conversations with tool calls in training-ready format
3. **Complete Training Pipeline** - LoRA fine-tuning, merging, and template management

**Perfect for:**
- Training domain-specific agents
- Building tool-using models
- Local/private AI deployments
- Continuous model improvement

---

## 📦 Package Structure

```
strands_transformers/
├── models/
│   └── transformers.py          # HuggingFace model provider
├── session/
│   └── jsonl_session_manager.py # Auto-capture training data
├── tools/
│   ├── dataset_generator.py     # Manual dataset creation
│   ├── model_trainer.py         # LoRA training & merging
│   └── template.py              # Chat template manager
└── templates/
    ├── qwen3.j2                 # Qwen3 format (thinking mode)
    ├── llama3.1.j2              # Llama 3.1 (8B+)
    ├── llama3.2.j2              # Llama 3.2 (3B)
    └── gpt-oss.j2               # GPT-OSS multi-channel
```

---

## 🚀 Installation

```bash
git clone git@github.com:cagataycali/strands-transformers.git
cd strands-transformers

# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Install in development mode (optional)
pip install -e .
```

**Dependencies:**
- `strands-agents` - Strands SDK
- `transformers` - HuggingFace transformers
- `torch` - PyTorch
- `peft` - LoRA training
- `jinja2` - Template rendering
- `datasets` - Dataset utilities

---

## 🔧 Core Components

### 1. TransformerModel Provider

```python
from strands_transformers.models.transformers import TransformerModel
from strands import Agent

# Load any HuggingFace model
model = TransformerModel(
    model_path="Qwen/Qwen3-1.7B",  # Or local path: "./my_model"
    device="auto",                  # cuda/mps/cpu
    enable_thinking=True,           # Qwen3 thinking mode
    params={
        "max_tokens": 500,
        "temperature": 0.7,
        "top_p": 0.9,
        "repetition_penalty": 1.2
    }
)

agent = Agent(model=model)
```

**Features:**
- ✅ Automatic device selection (CUDA/MPS/CPU)
- ✅ Streaming responses
- ✅ Tool calling support
- ✅ Qwen3 thinking mode (`<think>` tags)
- ✅ Chat template formatting
- ✅ Merged LoRA weights

### 2. JsonlSessionManager (Auto-Training)

**Automatically collect training data from every conversation:**

```python
from strands_transformers.session.jsonl_session_manager import JsonlSessionManager
from strands import Agent
from strands_tools import shell, calculator

# Create session manager
session = JsonlSessionManager(
    session_id="production_data",
    template_name="qwen3",           # qwen3, llama3.1, llama3.2, gpt-oss
    storage_dir="./training_data"    # Default: ~/.strands/training_data
)

# Create agent with session manager
agent = Agent(
    model=model,
    tools=[shell, calculator],
    session_manager=session
)

# Every conversation is automatically saved!
agent("What tools do you have?")
agent("Run ls -la command")
# → Saved to: ./training_data/production_data.jsonl
```

**Captures:**
- ✅ System prompts with tool definitions
- ✅ User messages
- ✅ Tool calls (`<tool_call>` tags)
- ✅ Tool responses (`<tool_response>` tags)
- ✅ Assistant responses
- ✅ Complete conversation context

**Format:**
```json
{"text": "<|im_start|>system\n...\n<tools>...</tools>\n<|im_end|>\n<|im_start|>user\n...<|im_end|>\n<|im_start|>assistant\n<tool_call>{...}</tool_call><|im_end|>..."}
```

### 3. Model Trainer (LoRA Fine-Tuning)

```python
from strands_transformers.tools.model_trainer import model_trainer

# Train with LoRA (efficient!)
model_trainer(
    action="train",
    model_name="Qwen/Qwen3-1.7B",
    dataset="./training_data/production_data.jsonl",
    output_dir="./qwen3_trained",
    use_lora=True,              # LoRA = 297MB vs 6.4GB full model
    num_epochs=3,
    learning_rate=2e-4,
    lora_r=16,
    lora_alpha=32
)

# Merge LoRA weights for inference
model_trainer(
    action="load_for_inference",
    model_name="./qwen3_trained",
    output_dir="./qwen3_merged"
)
```

**Training Features:**
- ✅ LoRA fine-tuning (low memory)
- ✅ Automatic train/eval split
- ✅ Loss tracking
- ✅ Weight merging
- ✅ GPU/CPU support

### 4. Dataset Generator (Manual)

```python
from strands_transformers.tools.dataset_generator import dataset_generator

# Generate training examples
dataset_generator(
    action="generate",
    template_name="qwen3",
    output_file="custom_data.jsonl",
    examples=[
        {
            "system_prompt": "You are a Python expert.",
            "instruction": "How do I create a list?",
            "response": "Use square brackets: my_list = [1, 2, 3]",
            "tools": [...],  # Optional
            "thinking_content": "..."  # Optional
        }
    ]
)

# Preview format
dataset_generator(action="preview", template_name="qwen3", examples=[...], count=1)

# List available templates
dataset_generator(action="list_formats")
```

### 5. Template Manager

```python
from strands_transformers.tools.template import template

# List templates
template(action="list")

# Create custom template
template(
    action="create",
    template_name="my_model",
    content="<|start|>{{ role }}<|message|>{{ content }}<|end|>"
)

# Render template
template(
    action="render",
    template_name="qwen3",
    variables={"system_prompt": "...", "instruction": "...", "response": "..."}
)
```

---

## 📚 Available Templates

| Template | Model | Features | Use Case |
|----------|-------|----------|----------|
| `qwen3.j2` | Qwen3 | Thinking mode, tool calling | General purpose |
| `llama3.1.j2` | Llama 3.1 (8B+) | Tool calling, ipython results | Large models |
| `llama3.2.j2` | Llama 3.2 (3B) | Tool calling, optimized | Small models |
| `gpt-oss.j2` | GPT-OSS | Multi-channel (analysis, commentary) | Advanced |

**Template Features:**
- System prompts with tool definitions
- Tool calling format (`<tool_call>` tags)
- Tool responses (`<tool_response>` tags)
- Thinking mode support (`<think>` tags for Qwen3)
- Jinja2 syntax for flexibility

---

## 🔄 Complete Training Workflow

### Option 1: Automatic (Recommended)

```python
from strands import Agent
from strands_tools import shell, calculator
from strands_transformers.models.transformers import TransformerModel
from strands_transformers.session.jsonl_session_manager import JsonlSessionManager
from strands_transformers.tools.model_trainer import model_trainer

# 1. Setup base model
model = TransformerModel(model_path="Qwen/Qwen3-1.7B", device="auto")

# 2. Enable auto-training data collection
session = JsonlSessionManager(
    session_id="my_agent",
    template_name="qwen3",
    storage_dir="./training_data"
)

# 3. Create agent
agent = Agent(model=model, tools=[shell, calculator], session_manager=session)

# 4. Use agent normally - data collected automatically!
agent("What tools do you have?")
agent("Calculate 25 * 17")
agent("Run ls command")
# → Saved to: ./training_data/my_agent.jsonl

# 5. Train on collected data
model_trainer(
    action="train",
    model_name="Qwen/Qwen3-1.7B",
    dataset="./training_data/my_agent.jsonl",
    output_dir="./qwen3_my_agent_trained",
    use_lora=True,
    num_epochs=3
)

# 6. Merge weights
model_trainer(
    action="load_for_inference",
    model_name="./qwen3_my_agent_trained",
    output_dir="./qwen3_my_agent_merged"
)

# 7. Use improved model
improved_model = TransformerModel(
    model_path="./qwen3_my_agent_merged",
    device="auto"
)
improved_agent = Agent(model=improved_model, tools=[shell, calculator])
```

### Option 2: Manual Dataset Creation

```python
from strands_transformers.tools.dataset_generator import dataset_generator
from strands_transformers.tools.model_trainer import model_trainer

# 1. Generate dataset manually
examples = [
    {
        "system_prompt": "You are a helpful assistant.",
        "instruction": "What is Python?",
        "response": "Python is a programming language..."
    },
    {
        "instruction": "How do I create a function?",
        "response": "Use def keyword: def my_function():"
    }
]

dataset_generator(
    action="generate",
    template_name="qwen3",
    output_file="manual_data.jsonl",
    examples=examples
)

# 2. Train on manual dataset
model_trainer(
    action="train",
    model_name="Qwen/Qwen3-1.7B",
    dataset="manual_data.jsonl",
    output_dir="./trained",
    use_lora=True
)
```

---

## 🎓 Example: Training a Tool-Using Agent

```python
from strands import Agent
from strands_tools import shell, calculator, weather
from strands_transformers.models.transformers import TransformerModel
from strands_transformers.session.jsonl_session_manager import JsonlSessionManager
from strands_transformers.tools.model_trainer import model_trainer

# Phase 1: Data Collection (1-2 weeks)
session = JsonlSessionManager(session_id="tool_expert", template_name="qwen3")
base_model = TransformerModel(model_path="Qwen/Qwen3-1.7B", device="auto")
agent = Agent(model=base_model, tools=[shell, calculator, weather], session_manager=session)

# Use agent extensively to collect diverse examples
# ... (agent handles real user requests)

# Phase 2: Training
model_trainer(
    action="train",
    model_name="Qwen/Qwen3-1.7B",
    dataset="~/.strands/training_data/tool_expert.jsonl",
    output_dir="./tool_expert_trained",
    use_lora=True,
    num_epochs=5,
    learning_rate=1e-4
)

model_trainer(
    action="load_for_inference",
    model_name="./tool_expert_trained",
    output_dir="./tool_expert_merged"
)

# Phase 3: Deploy
production_model = TransformerModel(
    model_path="./tool_expert_merged",
    device="auto",
    enable_thinking=True
)
production_agent = Agent(
    model=production_model,
    tools=[shell, calculator, weather],
    session_manager=session  # Continue collecting data!
)
```

---

## 🛠️ Development

### Project Setup

```bash
# Clone repository
git clone git@github.com:cagataycali/strands-transformers.git
cd strands-transformers

# Install in editable mode
pip install -e .

# Run example agent
python agent.py
```

### Code Structure

```
strands_transformers/
├── models/
│   ├── __init__.py
│   └── transformers.py           # Model provider implementation
│       ├── TransformerModel       # Main model class
│       ├── _load_model()          # Load HF model & tokenizer
│       ├── _format_messages()     # Chat template formatting
│       ├── stream()               # Streaming generation
│       └── structured_output()    # JSON schema support
│
├── session/
│   └── jsonl_session_manager.py  # Training data collection
│       ├── JsonlSessionManager    # Session manager class
│       ├── _format_conversation() # Format to chat template
│       ├── sync_agent()           # Save on completion
│       └── _serialize_tool_spec() # Tool definition export
│
├── tools/
│   ├── dataset_generator.py      # Dataset creation from examples
│   ├── model_trainer.py          # LoRA training & merging
│   ├── template.py               # Template management
│   └── use_transformers.py       # Strands tool wrapper
│
└── templates/                    # Chat format templates
    ├── qwen3.j2
    ├── llama3.1.j2
    ├── llama3.2.j2
    └── gpt-oss.j2
```

### Adding New Templates

```python
# 1. Create template file: strands_transformers/templates/my_model.j2
# 2. Use Jinja2 syntax with variables:
#    - system_prompt
#    - instruction
#    - response
#    - tools (optional)
#    - thinking_content (optional)
#    - tool_calls (optional)

# 3. Test template
from strands_transformers.tools.template import template

template(
    action="render",
    template_name="my_model",
    variables={
        "system_prompt": "Test system",
        "instruction": "Test input",
        "response": "Test output"
    }
)
```

---

## 🧪 Testing

```python
# Test TransformerModel
from strands_transformers.models.transformers import TransformerModel

model = TransformerModel(model_path="Qwen/Qwen3-1.7B", device="cpu")
print(model.get_config())

# Test JsonlSessionManager
from strands_transformers.session.jsonl_session_manager import JsonlSessionManager

session = JsonlSessionManager(session_id="test", template_name="qwen3")
print(f"Output: {session.get_jsonl_path()}")

# Test dataset generator
from strands_transformers.tools.dataset_generator import dataset_generator

dataset_generator(
    action="preview",
    template_name="qwen3",
    examples=[{"instruction": "Test", "response": "Output"}],
    count=1
)
```

---

## 📊 Training Metrics

**Example training run (Qwen3-1.7B, 50 examples, 3 epochs):**

| Metric | Value |
|--------|-------|
| Training Loss | 2.2185 |
| Runtime | 199s (~3.3 min) |
| LoRA Checkpoint | 297 MB |
| Merged Model | 6.4 GB |
| Train/Eval Split | 45/5 |

**LoRA vs Full Fine-Tuning:**
- LoRA adapter: ~297 MB
- Full model weights: ~6.4 GB
- Training speed: 2-3x faster with LoRA
- Memory usage: ~70% less

---

## 🐛 Troubleshooting

### Issue: Model loading fails

```python
# Solution: Enable trust_remote_code
model = TransformerModel(
    model_path="model_path",
    trust_remote_code=True
)
```

### Issue: CUDA out of memory

```python
# Solution 1: Use smaller model
model = TransformerModel(model_path="Qwen/Qwen3-0.6B", device="cuda")

# Solution 2: Use CPU
model = TransformerModel(model_path="Qwen/Qwen3-1.7B", device="cpu")

# Solution 3: Lower batch size in training
model_trainer(..., params={"per_device_train_batch_size": 1})
```

### Issue: Training data not saved

```python
# Check:
# 1. Session manager attached to agent?
agent = Agent(model=model, session_manager=session)

# 2. Complete conversation (assistant response with text)?
agent("prompt")  # Wait for full response

# 3. Check file
print(session.get_jsonl_path())
print(f"Examples: {session.get_example_count()}")
```

---

## 🤝 Contributing

1. Fork the repository
2. Create feature branch: `git checkout -b feature/my-feature`
3. Make changes and test
4. Commit: `git commit -m "feat: add my feature"`
5. Push: `git push origin feature/my-feature`
6. Open Pull Request

**Areas for contribution:**
- New chat templates
- Additional model providers
- Training optimizations
- Documentation improvements
- Example notebooks

---

## 📄 License

MIT License - see LICENSE file for details

---

## 🔗 Links

- **Strands SDK:** https://github.com/strands-agents/sdk-python
- **HuggingFace Transformers:** https://huggingface.co/docs/transformers
- **LoRA Paper:** https://arxiv.org/abs/2106.09685

---

## 🙏 Acknowledgments

Built using:
- [Strands Agents SDK](https://github.com/strands-agents/sdk-python)
- [HuggingFace Transformers](https://huggingface.co/transformers)
- [PEFT (LoRA)](https://github.com/huggingface/peft)

**Author:** Research Agent  
**Repository:** https://github.com/cagataycali/strands-transformers
