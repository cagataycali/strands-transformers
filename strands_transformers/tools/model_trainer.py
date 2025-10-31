"""
Model Trainer - Tool for fine-tuning language models.

- TRL SFTTrainer for supervised fine-tuning
- Proper chat template handling
- LoRA/PEFT with expert layer targeting
- Modern hyperparameters (cosine with min_lr)
- Merge and unload pattern for inference
- 4-bit quantization support

Supports models like Qwen3-1.7B, gpt-oss-20b, Llama, Mistral, etc.
Built with the modern @tool decorator pattern.
"""

import traceback
from strands import tool
from typing import Dict, Any, Optional, List, Union
import os
import json
import torch
from pathlib import Path
from datetime import datetime

# Import transformers and TRL components
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
)
from datasets import load_dataset, Dataset as HFDataset
from peft import LoraConfig, get_peft_model, PeftModel

from transformers import TextIteratorStreamer
from threading import Thread

# TRL imports - optional, will be imported when needed
try:
    from trl import SFTConfig, SFTTrainer

    HAS_TRL = True
except ImportError:
    HAS_TRL = False
    SFTConfig = None
    SFTTrainer = None


@tool
def model_trainer(
    action: str,
    model_name: Optional[str] = None,
    dataset: Optional[str] = None,
    output_dir: Optional[str] = None,
    use_lora: bool = False,
    quantize_4bit: bool = False,
    batch_size: int = 4,
    learning_rate: float = 2e-4,
    num_epochs: int = 3,
    max_steps: int = -1,
    gradient_accumulation_steps: int = 4,
    warmup_steps: int = 100,
    eval_steps: int = 500,
    save_steps: int = 500,
    max_seq_length: int = 512,
    lora_r: int = 8,
    lora_alpha: int = 32,
    lora_dropout: float = 0.1,
    lora_target_modules: Optional[List[str]] = None,
    dataset_text_field: str = "text",
    dataset_split: str = "train",
    test_size: float = 0.1,
    prompt: Optional[str] = None,
    max_new_tokens: int = 100,
    temperature: float = 0.7,
    top_p: float = 0.9,
    stream: bool = False,
    enable_thinking: bool = True,
    top_k: int = 20,
) -> Dict[str, Any]:
    """
    Model training tool for agents.

    Supports thinking mode - models can use internal reasoning before responding.

    Actions:
    - train: Fine-tune a model with TRL SFTTrainer
    - load_dataset: Load and preview a dataset
    - generate: Generate text with a trained model
    - evaluate: Evaluate a trained model
    - export: Export model for deployment
    - info: Get model information
    - load_for_inference: Merge LoRA weights for fast inference
    - export_to_ollama: Export merged model to Ollama (use prompt field for model name)
    - convert_to_gguf: Convert HuggingFace model to GGUF format (use prompt for quantization type)

    Args:
        action: Action to perform
        model_name: HuggingFace model name or path (e.g., "Qwen/Qwen3-1.7B")
        dataset: Dataset name/path (HF Hub, file, or directory)
        output_dir: Output directory for trained/merged model
        use_lora: Use LoRA for efficient training (recommended!)
        quantize_4bit: Use 4-bit quantization with bitsandbytes
        batch_size: Training batch size per device
        learning_rate: Learning rate (2e-4 good for LoRA)
        num_epochs: Number of training epochs
        max_steps: Maximum training steps (overrides epochs if > 0)
        gradient_accumulation_steps: Gradient accumulation steps
        warmup_steps: Number of warmup steps (deprecated, use warmup_ratio)
        eval_steps: Steps between evaluations
        save_steps: Steps between checkpoints
        max_seq_length: Maximum sequence length for tokenization
        lora_r: LoRA rank (8-32, higher = more parameters)
        lora_alpha: LoRA alpha parameter (typically 2x rank)
        lora_dropout: LoRA dropout rate
        lora_target_modules: Target modules for LoRA (auto: "all-linear")
        dataset_text_field: Field name containing text in dataset
        dataset_split: Dataset split to use
        test_size: Proportion of data for validation (0 = no eval)
        prompt: Text prompt for generation
        max_new_tokens: Maximum new tokens to generate
        temperature: Sampling temperature
        top_p: Nucleus sampling parameter
        stream: Stream output token by token (default: False)
        enable_thinking: Enable thinking mode for Qwen3 models (default: True)
        top_k: Top-k sampling parameter (default: 20)

    Returns:
        Dict with status and results

    Examples:
        # Train Qwen3-1.7B with LoRA (OpenAI cookbook pattern)
        model_trainer(
            action="train",
            model_name="Qwen/Qwen3-1.7B",
            dataset="my_data.txt",
            output_dir="./qwen3_finetuned",
            use_lora=True,
            num_epochs=3,
            learning_rate=2e-4
        )

        # Merge LoRA weights for fast inference
        model_trainer(
            action="load_for_inference",
            model_name="./qwen3_finetuned",
            output_dir="./qwen3_merged"
        )

        # Generate with merged model
        model_trainer(
            action="generate",
            model_name="./qwen3_merged",
            prompt="Hello! I am"
        )

        # Export to Ollama
        model_trainer(
            action="export_to_ollama",
            model_name="./qwen3_merged",
            prompt="my-custom-model"  # Ollama model name
        )
    """

    try:
        if action == "train":
            return _train_model(
                model_name=model_name,
                dataset=dataset,
                output_dir=output_dir,
                use_lora=use_lora,
                quantize_4bit=quantize_4bit,
                batch_size=batch_size,
                learning_rate=learning_rate,
                num_epochs=num_epochs,
                max_steps=max_steps,
                gradient_accumulation_steps=gradient_accumulation_steps,
                warmup_steps=warmup_steps,
                eval_steps=eval_steps,
                save_steps=save_steps,
                max_seq_length=max_seq_length,
                lora_r=lora_r,
                lora_alpha=lora_alpha,
                lora_dropout=lora_dropout,
                lora_target_modules=lora_target_modules,
                dataset_text_field=dataset_text_field,
                dataset_split=dataset_split,
                test_size=test_size,
            )

        elif action == "load_dataset":
            return _load_and_preview_dataset(
                dataset=dataset,
                dataset_text_field=dataset_text_field,
                dataset_split=dataset_split,
            )

        elif action == "generate":
            return _generate_text(
                model_name=model_name,
                prompt=prompt,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                stream=stream,
                enable_thinking=enable_thinking,
            )

        elif action == "evaluate":
            return _evaluate_model(
                model_name=model_name,
                dataset=dataset,
                max_seq_length=max_seq_length,
                batch_size=batch_size,
                dataset_text_field=dataset_text_field,
            )

        elif action == "export":
            return _export_model(
                model_name=model_name,
                output_dir=output_dir,
            )

        elif action == "info":
            return _get_model_info(model_name=model_name)

        elif action == "load_for_inference":
            return _load_merged_model(
                model_name=model_name,
                output_dir=output_dir,
            )

        elif action == "export_to_ollama":
            return _export_to_ollama(
                model_name=model_name,
                ollama_model_name=prompt,  # Use prompt field for Ollama model name
            )

        elif action == "convert_to_gguf":
            return _convert_to_gguf(
                model_name=model_name,
                output_dir=output_dir,
                quantization=prompt,  # Use prompt field for quantization type
            )

        else:
            return {
                "status": "error",
                "content": [
                    {
                        "text": f"Unknown action: {action}. Valid actions: train, load_dataset, generate, evaluate, export, info, load_for_inference, export_to_ollama, convert_to_gguf"
                    }
                ],
            }

    except Exception as e:
        tb = traceback.format_exc()
        return {
            "status": "error",
            "content": [
                {"text": f"Error in model_trainer: {str(e)}\n\nTraceback:\n{tb}"}
            ],
        }


def _load_dataset_from_source(
    dataset: str,
    dataset_text_field: str = "text",
    dataset_split: str = "train",
) -> HFDataset:
    """Load dataset from various sources."""
    dataset_path = Path(dataset)

    # Single file
    if dataset_path.is_file():
        with open(dataset_path, "r", encoding="utf-8") as f:
            texts = [line.strip() for line in f if line.strip()]
        return HFDataset.from_dict({dataset_text_field: texts})

    # Directory of text files
    elif dataset_path.is_dir():
        texts = []
        for file_path in dataset_path.rglob("*.txt"):
            with open(file_path, "r", encoding="utf-8") as f:
                texts.extend([line.strip() for line in f if line.strip()])
        return HFDataset.from_dict({dataset_text_field: texts})

    # HuggingFace Hub
    else:
        return load_dataset(dataset, split=dataset_split)


def _train_model(
    model_name: str,
    dataset: str,
    output_dir: Optional[str],
    use_lora: bool,
    quantize_4bit: bool,
    batch_size: int,
    learning_rate: float,
    num_epochs: int,
    max_steps: int,
    gradient_accumulation_steps: int,
    warmup_steps: int,
    eval_steps: int,
    save_steps: int,
    max_seq_length: int,
    lora_r: int,
    lora_alpha: int,
    lora_dropout: float,
    lora_target_modules: Optional[List[str]],
    dataset_text_field: str,
    dataset_split: str,
    test_size: float,
) -> Dict[str, Any]:
    """Train a model using TRL SFTTrainer (OpenAI cookbook pattern)."""

    # Check if TRL is available
    if not HAS_TRL:
        return {
            "status": "error",
            "content": [
                {
                    "text": "Error: TRL library not installed. Install with: pip install 'trl>=0.20.0'"
                }
            ],
        }

    # Setup output directory
    if output_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_basename = model_name.split("/")[-1]
        output_dir = f"./trained_models/{model_basename}_{timestamp}"

    output_dir = os.path.expanduser(output_dir)
    os.makedirs(output_dir, exist_ok=True)

    results = []
    results.append(f"🚀 Starting training: {model_name}")
    results.append(f"📁 Output directory: {output_dir}")
    results.append(f"🔧 LoRA: {use_lora}, 4-bit: {quantize_4bit}")

    # Load tokenizer
    results.append("\n📝 Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

    # Set padding token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load model with optional quantization
    results.append("🤖 Loading model...")

    # Detect device
    if torch.cuda.is_available():
        device = "cuda"
        device_config = "auto"
    else:
        device = "cpu"
        device_config = {"": "cpu"}  # Explicitly map to CPU to avoid meta device

    results.append(f"   Device: {device}")

    model_kwargs = {
        "trust_remote_code": True,
        "torch_dtype": torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        "use_cache": False,  # Required for gradient checkpointing
        "device_map": device_config,
    }

    if quantize_4bit:
        if not torch.cuda.is_available():
            results.append("   ⚠️  4-bit quantization requires CUDA, skipping...")
        else:
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
            )
            model_kwargs["quantization_config"] = bnb_config
            results.append("   Using 4-bit quantization")

    model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)

    # Apply LoRA if requested
    peft_config = None
    if use_lora:
        results.append("🔗 Applying LoRA configuration...")

        # Auto-detect target modules if not provided
        if lora_target_modules is None:
            # Use "all-linear" for modern approach (targets all linear layers)
            # For specific models, can override
            if "qwen" in model_name.lower():
                target_modules = "all-linear"
                # For MoE models, also target expert layers
                target_parameters = None
                if hasattr(model.config, "num_experts"):
                    # Target expert projection layers
                    target_parameters = [
                        "7.mlp.experts.gate_up_proj",
                        "7.mlp.experts.down_proj",
                        "15.mlp.experts.gate_up_proj",
                        "15.mlp.experts.down_proj",
                        "23.mlp.experts.gate_up_proj",
                        "23.mlp.experts.down_proj",
                    ]
            elif "llama" in model_name.lower():
                target_modules = "all-linear"
                target_parameters = None
            else:
                target_modules = "all-linear"
                target_parameters = None
        else:
            target_modules = lora_target_modules
            target_parameters = None

        peft_config = LoraConfig(
            r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            target_modules=target_modules,
            target_parameters=target_parameters if target_parameters else None,
            bias="none",
        )

        model = get_peft_model(model, peft_config)
        model.print_trainable_parameters()
        results.append(f"   Target modules: {target_modules}")
        results.append(f"   LoRA rank: {lora_r}, alpha: {lora_alpha}")

    # Load dataset
    results.append("\n📊 Loading dataset...")
    dataset_full = _load_dataset_from_source(dataset, dataset_text_field, dataset_split)
    results.append(f"   Total examples: {len(dataset_full)}")

    # TRL SFTTrainer can handle chat templates automatically
    # For non-conversational data, we use dataset_text_field

    # Split into train/eval if needed
    if test_size > 0:
        results.append("🔀 Creating train/validation split...")
        split_dataset = dataset_full.train_test_split(test_size=test_size)
        train_dataset = split_dataset["train"]
        eval_dataset = split_dataset["test"]
        results.append(f"   Train: {len(train_dataset)}, Eval: {len(eval_dataset)}")
    else:
        train_dataset = dataset_full
        eval_dataset = None
        results.append(f"   Train: {len(train_dataset)} (no eval split)")

    # Training configuration using SFTConfig
    results.append("\n⚙️ Configuring training with SFTConfig...")

    training_args = SFTConfig(
        output_dir=output_dir,
        # Learning & optimization
        learning_rate=learning_rate,
        num_train_epochs=num_epochs,
        max_steps=max_steps if max_steps > 0 else -1,
        # Batch configuration
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        # Scheduler - modern cosine with min_lr
        lr_scheduler_type="cosine_with_min_lr",
        lr_scheduler_kwargs={"min_lr_rate": 0.1},
        warmup_ratio=0.03,  # Modern warmup ratio instead of steps
        # Optimization
        optim="adamw_torch",
        weight_decay=0.01,
        # Checkpointing
        gradient_checkpointing=True,
        # Logging & evaluation
        logging_steps=1,
        eval_strategy="steps" if eval_dataset else "no",
        eval_steps=eval_steps if eval_dataset else None,
        # Saving
        save_strategy="steps",
        save_steps=save_steps,
        save_total_limit=3,
        load_best_model_at_end=True if eval_dataset else False,
        # SFT-specific (note: parameter name is max_length in newer TRL)
        max_length=max_seq_length,
        dataset_text_field=dataset_text_field,
        # Reporting (disabled to avoid tensorboard dependency)
        report_to=[],
        # Mixed precision
        bf16=True if torch.cuda.is_available() else False,
        fp16=False,
    )

    # Create SFTTrainer
    results.append("🏋️ Creating SFTTrainer...")
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,  # SFTTrainer uses processing_class
        peft_config=peft_config if use_lora else None,
    )

    # Train
    results.append("\n🎯 Training started...\n")
    train_result = trainer.train()

    # Save model
    results.append("\n💾 Saving model...")
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)

    # Save training config
    config_file = os.path.join(output_dir, "training_config.json")
    with open(config_file, "w") as f:
        json.dump(
            {
                "model_name": model_name,
                "use_lora": use_lora,
                "quantize_4bit": quantize_4bit,
                "lora_config": (
                    {
                        "r": lora_r,
                        "alpha": lora_alpha,
                        "dropout": lora_dropout,
                        "target_modules": target_modules if use_lora else None,
                    }
                    if use_lora
                    else None
                ),
                "training_args": {
                    "batch_size": batch_size,
                    "learning_rate": learning_rate,
                    "num_epochs": num_epochs,
                    "max_steps": max_steps,
                    "scheduler": "cosine_with_min_lr",
                },
                "metrics": (
                    train_result.metrics if hasattr(train_result, "metrics") else {}
                ),
            },
            f,
            indent=2,
        )

    # Format metrics
    results.append("\n📈 Training complete! Metrics:")
    if hasattr(train_result, "metrics"):
        for key, value in train_result.metrics.items():
            if isinstance(value, float):
                results.append(f"   {key}: {value:.4f}")
            else:
                results.append(f"   {key}: {value}")

    results.append(f"\n✅ Model saved to: {output_dir}")
    results.append(
        f"\n💡 Use action='load_for_inference' to load with merged LoRA weights"
    )

    return {"status": "success", "content": [{"text": "\n".join(results)}]}


def _load_and_preview_dataset(
    dataset: str,
    dataset_text_field: str,
    dataset_split: str,
) -> Dict[str, Any]:
    """Load and preview a dataset."""

    results = []
    results.append(f"📊 Loading dataset: {dataset}")

    dataset_obj = _load_dataset_from_source(dataset, dataset_text_field, dataset_split)

    results.append(f"\n✅ Dataset loaded successfully")
    results.append(f"   Total examples: {len(dataset_obj)}")
    results.append(f"   Columns: {dataset_obj.column_names}")

    # Show first few examples
    results.append("\n📝 First 3 examples:")
    for i, example in enumerate(dataset_obj.select(range(min(3, len(dataset_obj))))):
        text = example.get(dataset_text_field, str(example))
        preview = text[:200] + "..." if len(text) > 200 else text
        results.append(f"\n   Example {i+1}: {preview}")

    return {"status": "success", "content": [{"text": "\n".join(results)}]}


def _generate_text(
    model_name: str,
    prompt: str,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    top_k: int,
    stream: bool = False,
    enable_thinking: bool = True,
) -> Dict[str, Any]:
    """Generate text with a trained model (handles PEFT models and Qwen3 thinking mode)."""

    if prompt is None:
        return {
            "status": "error",
            "content": [{"text": "Error: prompt is required for generate action"}],
        }

    results = []
    results.append(f"🤖 Loading model: {model_name}")

    model_path = Path(model_name)

    # Check if this is a PEFT model
    is_peft_model = (model_path / "adapter_config.json").exists()

    if is_peft_model:
        results.append("   Detected PEFT/LoRA model")

        # Load training config to get base model
        config_file = model_path / "training_config.json"
        if config_file.exists():
            with open(config_file, "r") as f:
                training_config = json.load(f)
            base_model_name = training_config.get("model_name")
            results.append(f"   Base model: {base_model_name}")
        else:
            return {
                "status": "error",
                "content": [
                    {
                        "text": "Error: PEFT model found but training_config.json missing. Cannot determine base model."
                    }
                ],
            }

        # Load tokenizer from adapter directory
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

        # Determine device (use CPU to avoid meta tensor issues on MPS)
        device = (
            "cuda" if torch.cuda.is_available() else "cpu"
        )  # Force CPU for non-CUDA
        results.append(f"   Device: {device}")

        # Load base model on CPU (prevent meta device with low_cpu_mem_usage=False)
        results.append("   Loading base model...")
        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            trust_remote_code=True,
            low_cpu_mem_usage=False,  # Critical: prevents meta device initialization
        )

        # Move to CUDA if available
        if device == "cuda":
            base_model = base_model.to(device)

        # Load PEFT adapter with is_trainable=False to fix meta tensor issue
        results.append("   Loading LoRA adapter...")
        model = PeftModel.from_pretrained(base_model, model_name, is_trainable=False)
    else:
        # Regular model loading
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

        # Determine device (use CPU to avoid meta tensor issues on MPS)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        results.append(f"   Device: {device}")

        # Load on CPU
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            trust_remote_code=True,
            low_cpu_mem_usage=False,  # Critical: prevents meta device initialization
        )
        if device == "cuda":
            model = model.to(device)

    results.append(f"💬 Prompt: {prompt}")
    results.append(
        f"🎯 Generating ({max_new_tokens} tokens, thinking={'enabled' if enable_thinking else 'disabled'})...\n"
    )

    # For Qwen3 models with thinking mode, use chat template
    is_qwen3 = "qwen3" in model_name.lower() or (
        hasattr(model.config, "model_type") and model.config.model_type == "qwen3"
    )

    if is_qwen3:
        # Use chat template for Qwen3
        messages = [{"role": "user", "content": prompt}]

        # Apply chat template with enable_thinking parameter
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=enable_thinking,
        )
        inputs = tokenizer([text], return_tensors="pt")
    else:
        # Regular tokenization for non-Qwen3 models
        inputs = tokenizer(prompt, return_tensors="pt")

    # Move inputs to same device as model
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    # Print initial results
    print("\n".join(results))

    if stream:
        # Create streamer
        streamer = TextIteratorStreamer(
            tokenizer, skip_prompt=True, skip_special_tokens=True
        )

        # Generation kwargs
        generation_kwargs = dict(
            inputs=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
            streamer=streamer,
        )

        # Start generation in a thread
        thread = Thread(target=model.generate, kwargs=generation_kwargs)
        thread.start()

        # Stream output
        print("✨ Generated:", end=" ", flush=True)
        generated_text = ""
        for new_text in streamer:
            print(new_text, end="", flush=True)
            generated_text += new_text
        print()  # Newline at end

        thread.join()

        return {
            "status": "success",
            "content": [{"text": f"✨ Generated:\n{generated_text}"}],
        }
    else:
        # Non-streaming generation
        outputs = model.generate(
            inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
        )

        # Extract only the generated tokens (not the input)
        output_ids = outputs[0][len(inputs["input_ids"][0]) :].tolist()

        # Parse thinking content for Qwen3 models
        if is_qwen3 and enable_thinking:
            try:
                # Find </think> token (151668)
                index = len(output_ids) - output_ids[::-1].index(151668)
            except ValueError:
                # No thinking content found
                index = 0

            thinking_content = tokenizer.decode(
                output_ids[:index], skip_special_tokens=True
            ).strip("\n")
            generated_text = tokenizer.decode(
                output_ids[index:], skip_special_tokens=True
            ).strip("\n")

            result_text = []
            if thinking_content:
                result_text.append(f"🧠 Thinking:\n{thinking_content}\n")
            result_text.append(f"✨ Response:\n{generated_text}")

            return {"status": "success", "content": [{"text": "\n".join(result_text)}]}
        else:
            # Regular output for non-thinking models
            generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

            return {
                "status": "success",
                "content": [{"text": f"✨ Generated:\n{generated_text}"}],
            }


def _evaluate_model(
    model_name: str,
    dataset: str,
    max_seq_length: int,
    batch_size: int,
    dataset_text_field: str,
) -> Dict[str, Any]:
    """Evaluate a model on a dataset."""

    if dataset is None:
        return {
            "status": "error",
            "content": [{"text": "Error: dataset is required for evaluate action"}],
        }

    results = []
    results.append(f"📊 Evaluating model: {model_name}")

    # Load model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

    # Set padding token (IMPORTANT!)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True)

    # Load dataset
    eval_dataset = _load_dataset_from_source(dataset, dataset_text_field, "test")
    results.append(f"   Evaluation examples: {len(eval_dataset)}")

    # Tokenize
    def tokenize_function(examples):
        return tokenizer(
            examples[dataset_text_field],
            truncation=True,
            max_length=max_seq_length,
            padding="max_length",
        )

    tokenized_dataset = eval_dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=eval_dataset.column_names,
    )

    # Create trainer for evaluation
    training_args = TrainingArguments(
        output_dir="./eval_tmp",
        per_device_eval_batch_size=batch_size,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        eval_dataset=tokenized_dataset,
        data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False),
    )

    # Evaluate
    results.append("\n🎯 Running evaluation...")
    eval_results = trainer.evaluate()

    results.append("\n📈 Evaluation results:")
    for key, value in eval_results.items():
        if isinstance(value, float):
            results.append(f"   {key}: {value:.4f}")
        else:
            results.append(f"   {key}: {value}")

    return {"status": "success", "content": [{"text": "\n".join(results)}]}


def _export_model(
    model_name: str,
    output_dir: Optional[str],
) -> Dict[str, Any]:
    """Export model for deployment."""

    if output_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = f"./exported_models/{timestamp}"

    output_dir = os.path.expanduser(output_dir)
    os.makedirs(output_dir, exist_ok=True)

    results = []
    results.append(f"📦 Exporting model: {model_name}")
    results.append(f"📁 Output: {output_dir}")

    # Load model and tokenizer
    model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

    # Save model
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    # Save config
    config_file = os.path.join(output_dir, "export_info.json")
    with open(config_file, "w") as f:
        json.dump(
            {
                "source_model": model_name,
                "export_date": datetime.now().isoformat(),
                "model_type": model.config.model_type,
            },
            f,
            indent=2,
        )

    results.append(f"\n✅ Model exported successfully")

    return {"status": "success", "content": [{"text": "\n".join(results)}]}


def _get_model_info(model_name: str) -> Dict[str, Any]:
    """Get information about a model."""

    results = []
    results.append(f"ℹ️ Model Information: {model_name}\n")

    try:
        from transformers import AutoConfig

        config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)

        results.append(f"Model Type: {config.model_type}")
        results.append(f"Hidden Size: {config.hidden_size}")
        results.append(f"Num Layers: {config.num_hidden_layers}")
        results.append(f"Num Attention Heads: {config.num_attention_heads}")
        results.append(f"Vocab Size: {config.vocab_size}")

        if hasattr(config, "max_position_embeddings"):
            results.append(f"Max Position Embeddings: {config.max_position_embeddings}")

        # Try to load model to get parameter count
        try:
            model = AutoModelForCausalLM.from_pretrained(
                model_name, trust_remote_code=True
            )
            num_params = sum(p.numel() for p in model.parameters())
            results.append(
                f"\nTotal Parameters: {num_params:,} ({num_params/1e9:.2f}B)"
            )
            trainable_params = sum(
                p.numel() for p in model.parameters() if p.requires_grad
            )
            results.append(f"Trainable Parameters: {trainable_params:,}")
        except:
            results.append("\n(Could not load model to count parameters)")

        return {"status": "success", "content": [{"text": "\n".join(results)}]}

    except Exception as e:
        return {
            "status": "error",
            "content": [{"text": f"Error loading model info: {str(e)}"}],
        }


def _load_merged_model(
    model_name: str,
    output_dir: Optional[str],
) -> Dict[str, Any]:
    """Load LoRA model and merge weights for fast inference (OpenAI cookbook pattern)."""

    if output_dir is None:
        return {
            "status": "error",
            "content": [
                {"text": "Error: output_dir is required for load_for_inference action"}
            ],
        }

    output_dir = os.path.expanduser(output_dir)

    results = []
    results.append(f"🔄 Loading and merging LoRA weights...")
    results.append(f"📁 Fine-tuned model: {model_name}")

    # Check if training config exists
    config_file = os.path.join(model_name, "training_config.json")
    if not os.path.exists(config_file):
        return {
            "status": "error",
            "content": [
                {"text": f"Error: training_config.json not found in {model_name}"}
            ],
        }

    # Load training config
    with open(config_file, "r") as f:
        training_config = json.load(f)

    base_model_name = training_config.get("model_name")
    use_lora = training_config.get("use_lora", False)

    if not use_lora:
        return {
            "status": "error",
            "content": [
                {"text": "Error: Model was not trained with LoRA. Nothing to merge."}
            ],
        }

    results.append(f"📖 Base model: {base_model_name}")
    results.append("\n🤖 Loading base model...")

    # Determine device (use CPU to avoid meta tensor issues on MPS)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    results.append(f"   Device: {device}")

    # Load the base model on CPU first (prevent meta device)
    model_kwargs = {
        "trust_remote_code": True,
        "low_cpu_mem_usage": False,  # Critical: prevents meta device initialization
    }

    base_model = AutoModelForCausalLM.from_pretrained(base_model_name, **model_kwargs)

    # Move to CUDA if available
    if device == "cuda":
        base_model = base_model.to(device)

    # Load LoRA weights with is_trainable=False to fix meta tensor issue
    results.append("🔗 Loading LoRA weights...")
    peft_model = PeftModel.from_pretrained(base_model, model_name, is_trainable=False)

    # Merge and unload
    results.append("🔀 Merging LoRA weights into base model...")
    merged_model = peft_model.merge_and_unload()

    # Save merged model
    os.makedirs(output_dir, exist_ok=True)
    results.append(f"\n💾 Saving merged model to: {output_dir}")

    merged_model.save_pretrained(output_dir)

    # Also save tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    tokenizer.save_pretrained(output_dir)

    # Save merged info
    merged_info_file = os.path.join(output_dir, "merged_info.json")
    with open(merged_info_file, "w") as f:
        json.dump(
            {
                "base_model": base_model_name,
                "lora_model": model_name,
                "merged_date": datetime.now().isoformat(),
                "training_config": training_config,
            },
            f,
            indent=2,
        )

    results.append("\n✅ Model merged successfully!")
    results.append(
        f"\n💡 Use this merged model for fast inference without LoRA overhead"
    )
    results.append(
        f"\n🚀 Load with: AutoModelForCausalLM.from_pretrained('{output_dir}')"
    )

    return {"status": "success", "content": [{"text": "\n".join(results)}]}


def _export_to_ollama(
    model_name: str,
    ollama_model_name: Optional[str],
) -> Dict[str, Any]:
    """Export fine-tuned model to Ollama.

    This creates a Modelfile and imports the model to Ollama.
    Requires ollama CLI to be installed and running.
    """

    if ollama_model_name is None:
        # Auto-generate name from model path
        ollama_model_name = f"finetuned-{Path(model_name).name}"

    results = []
    results.append(f"🦙 Exporting to Ollama: {ollama_model_name}")
    results.append(f"📁 Source model: {model_name}")

    model_path = Path(model_name).expanduser().resolve()

    # Check if model exists
    if not model_path.exists():
        return {
            "status": "error",
            "content": [{"text": f"Error: Model path does not exist: {model_path}"}],
        }

    # Check if this is a merged model or needs merging
    merged_info_file = model_path / "merged_info.json"
    training_config_file = model_path / "training_config.json"
    is_lora_model = training_config_file.exists() and not merged_info_file.exists()

    if is_lora_model:
        results.append("\n⚠️  Detected LoRA model - needs merging first!")
        results.append(
            "   Run: model_trainer(action='load_for_inference', model_name='...', output_dir='...')"
        )
        return {"status": "error", "content": [{"text": "\n".join(results)}]}

    # Create temporary Modelfile
    modelfile_dir = Path.home() / ".cache" / "model_trainer_ollama"
    modelfile_dir.mkdir(parents=True, exist_ok=True)
    modelfile_path = modelfile_dir / f"Modelfile.{ollama_model_name}"

    results.append(f"\n📝 Creating Modelfile...")

    # Read model config to get parameters
    try:
        from transformers import AutoConfig

        config = AutoConfig.from_pretrained(str(model_path), trust_remote_code=True)

        # Create Modelfile content
        modelfile_content = f"""# Modelfile for {ollama_model_name}
FROM {model_path}

# Model parameters
PARAMETER temperature 0.7
PARAMETER top_p 0.9
PARAMETER top_k 40

# System prompt (customize as needed)
SYSTEM You are a helpful AI assistant.
"""

        with open(modelfile_path, "w") as f:
            f.write(modelfile_content)

        results.append(f"   Modelfile: {modelfile_path}")
        results.append(f"\n📦 Importing to Ollama...")

        # Import to Ollama using shell command
        import subprocess

        try:
            # Run ollama create
            cmd = ["ollama", "create", ollama_model_name, "-f", str(modelfile_path)]
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=300,  # 5 minute timeout for large models
            )

            if result.returncode == 0:
                results.append("   ✅ Successfully imported to Ollama!")
                results.append(f"\n🚀 Usage:")
                results.append(f"   ollama run {ollama_model_name}")
                results.append(f"\n💡 In Python (Strands):")
                results.append(f"   from strands.models.ollama import OllamaModel")
                results.append(
                    f"   model = OllamaModel(host='http://localhost:11434', model_id='{ollama_model_name}')"
                )

                return {"status": "success", "content": [{"text": "\n".join(results)}]}
            else:
                error_msg = result.stderr if result.stderr else result.stdout
                results.append(f"\n❌ Ollama import failed:")
                results.append(f"   {error_msg}")
                return {"status": "error", "content": [{"text": "\n".join(results)}]}

        except FileNotFoundError:
            results.append("\n❌ Ollama CLI not found!")
            results.append("   Install: https://ollama.com/download")
            return {"status": "error", "content": [{"text": "\n".join(results)}]}

        except subprocess.TimeoutExpired:
            results.append("\n❌ Ollama import timed out (> 5 minutes)")
            return {"status": "error", "content": [{"text": "\n".join(results)}]}

    except Exception as e:
        tb = traceback.format_exc()
        return {
            "status": "error",
            "content": [
                {"text": f"Error creating Modelfile: {str(e)}\n\nTraceback:\n{tb}"}
            ],
        }


def _convert_to_gguf(
    model_name: str,
    output_dir: Optional[str],
    quantization: Optional[str],
) -> Dict[str, Any]:
    """Convert HuggingFace model to GGUF format for Ollama.

    This uses llama.cpp's conversion script to create GGUF files.
    Requires llama.cpp to be cloned and accessible.
    """

    if output_dir is None:
        output_dir = f"{model_name}_gguf"

    output_dir = os.path.expanduser(output_dir)
    os.makedirs(output_dir, exist_ok=True)

    # Default quantization
    if quantization is None:
        quantization = "Q4_K_M"  # Good balance of quality and size

    results = []
    results.append(f"🔄 Converting to GGUF: {model_name}")
    results.append(f"📁 Output directory: {output_dir}")
    results.append(f"⚙️ Quantization: {quantization}")

    model_path = Path(model_name).expanduser().resolve()

    # Check if model exists
    if not model_path.exists():
        return {
            "status": "error",
            "content": [{"text": f"Error: Model path does not exist: {model_path}"}],
        }

    # Check if this is a merged model
    merged_info_file = model_path / "merged_info.json"
    training_config_file = model_path / "training_config.json"
    is_lora_model = training_config_file.exists() and not merged_info_file.exists()

    if is_lora_model:
        results.append("\n⚠️  Detected LoRA model - needs merging first!")
        results.append(
            "   Run: model_trainer(action='load_for_inference', model_name='...', output_dir='...')"
        )
        return {"status": "error", "content": [{"text": "\n".join(results)}]}

    # Check for llama.cpp
    llama_cpp_dir = Path.home() / "llama.cpp"
    convert_script = llama_cpp_dir / "convert_hf_to_gguf.py"
    quantize_binary = llama_cpp_dir / "build" / "bin" / "llama-quantize"

    if not convert_script.exists():
        results.append("\n❌ llama.cpp not found!")
        results.append(f"   Expected: {convert_script}")
        results.append("\n📝 Installation instructions:")
        results.append(
            "   git clone https://github.com/ggerganov/llama.cpp ~/llama.cpp"
        )
        results.append("   cd ~/llama.cpp")
        results.append("   make")
        results.append(f"   pip install -r requirements.txt")
        return {"status": "error", "content": [{"text": "\n".join(results)}]}

    import subprocess
    import sys

    try:
        # Step 1: Convert to FP16 GGUF
        results.append("\n📝 Step 1: Converting to FP16 GGUF...")
        fp16_output = Path(output_dir) / "model-f16.gguf"

        cmd_convert = [
            sys.executable,
            str(convert_script),
            str(model_path),
            "--outfile",
            str(fp16_output),
            "--outtype",
            "f16",
        ]

        result = subprocess.run(
            cmd_convert,
            capture_output=True,
            text=True,
            timeout=600,  # 10 minute timeout
        )

        if result.returncode != 0:
            results.append(f"\n❌ Conversion failed:")
            results.append(f"   {result.stderr}")
            return {"status": "error", "content": [{"text": "\n".join(results)}]}

        results.append(f"   ✅ FP16 GGUF created: {fp16_output}")

        # Step 2: Quantize
        if quantization != "f16":
            results.append(f"\n📝 Step 2: Quantizing to {quantization}...")
            quantized_output = Path(output_dir) / f"model-{quantization.lower()}.gguf"

            if not quantize_binary.exists():
                results.append(f"\n⚠️  llama-quantize not found at {quantize_binary}")
                results.append("   Skipping quantization step")
                results.append(f"   You can use the FP16 model: {fp16_output}")
                final_output = fp16_output
            else:
                cmd_quantize = [
                    str(quantize_binary),
                    str(fp16_output),
                    str(quantized_output),
                    quantization,
                ]

                result = subprocess.run(
                    cmd_quantize,
                    capture_output=True,
                    text=True,
                    timeout=600,
                )

                if result.returncode != 0:
                    results.append(f"\n❌ Quantization failed:")
                    results.append(f"   {result.stderr}")
                    results.append(f"   You can use the FP16 model: {fp16_output}")
                    final_output = fp16_output
                else:
                    results.append(f"   ✅ Quantized GGUF created: {quantized_output}")
                    final_output = quantized_output
        else:
            final_output = fp16_output

        # Step 3: Create Modelfile for Ollama
        results.append(f"\n📝 Step 3: Creating Ollama Modelfile...")
        modelfile_path = Path(output_dir) / "Modelfile"

        ollama_model_name = f"{model_path.name}-gguf"

        modelfile_content = f"""# Modelfile for {ollama_model_name}
FROM {final_output.resolve()}

# Model parameters
PARAMETER temperature 0.7
PARAMETER top_p 0.9
PARAMETER top_k 40
PARAMETER stop "<|im_start|>"
PARAMETER stop "<|im_end|>"

# System prompt
SYSTEM You are a helpful AI assistant.
"""

        with open(modelfile_path, "w") as f:
            f.write(modelfile_content)

        results.append(f"   ✅ Modelfile created: {modelfile_path}")
        results.append(f"\n✅ Conversion complete!")
        results.append(f"\n📦 GGUF model: {final_output}")
        results.append(f"\n🚀 Import to Ollama:")
        results.append(f"   ollama create {ollama_model_name} -f {modelfile_path}")
        results.append(f"\n💡 Or use model_trainer:")
        results.append(
            f"   model_trainer(action='export_to_ollama', model_name='{model_name}', prompt='{ollama_model_name}')"
        )

        return {"status": "success", "content": [{"text": "\n".join(results)}]}

    except subprocess.TimeoutExpired:
        results.append("\n❌ Conversion timed out (> 10 minutes)")
        return {"status": "error", "content": [{"text": "\n".join(results)}]}

    except Exception as e:
        tb = traceback.format_exc()
        return {
            "status": "error",
            "content": [
                {"text": f"Error during conversion: {str(e)}\n\nTraceback:\n{tb}"}
            ],
        }
