"""
Dataset Generator - Create training datasets for LLM fine-tuning.

Supports multiple chat formats:
- Qwen
- Llama
- Alpaca
- ShareGPT
- Custom templates from ./templates/ directory

Built with the @tool decorator pattern.
"""

from strands import tool
from typing import Dict, Any, Optional, List
import json
from pathlib import Path
from jinja2 import Template


# Pre-defined templates for popular formats
# Qwen3 official template format
QWEN_TEMPLATE = """{%- if tools -%}
<|im_start|>system
{{ system_prompt }}

# Tools

You may call one or more functions to assist with the user query.

You are provided with function signatures within <tools></tools> XML tags:
<tools>
{%- for tool in tools %}
{"type": "function", "function": {{ tool | tojson }}}
{%- endfor %}
</tools>

For each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:
<tool_call>
{"name": <function-name>, "arguments": <args-json-object>}
</tool_call><|im_end|>
{%- else -%}
<|im_start|>system
{{ system_prompt }}<|im_end|>
{%- endif %}
<|im_start|>user
{{ instruction }}
{%- if enable_thinking is defined %}
{%- if enable_thinking %} /think{% else %} /no_think{% endif -%}
{%- endif -%}
<|im_end|>
{%- if tool_calls %}
<|im_start|>assistant
{%- for tool_call in tool_calls %}
<tool_call>
{"name": "{{ tool_call.name }}", "arguments": {{ tool_call.arguments | tojson }}}
</tool_call>
{%- endfor -%}
<|im_end|>
<|im_start|>user
<tool_response>
{%- for tool_response in tool_responses %}
{{ tool_response }}
{%- endfor -%}
</tool_response><|im_end|>
<|im_start|>assistant
{% if thinking_content -%}
<think>
{{ thinking_content }}
</think>

{% endif -%}
{{ response }}<|im_end|>
{%- else %}
<|im_start|>assistant
{% if thinking_content -%}
<think>
{{ thinking_content }}
</think>

{% endif -%}
{{ response }}<|im_end|>
{%- endif -%}"""


LLAMA_TEMPLATE = """<|begin_of_text|><|start_header_id|>system<|end_header_id|>

{{ system_prompt }}<|eot_id|><|start_header_id|>user<|end_header_id|>

{{ instruction }}<|eot_id|><|start_header_id|>assistant<|end_header_id|>

{{ response }}<|eot_id|>"""


ALPACA_TEMPLATE = """Below is an instruction that describes a task. Write a response that appropriately completes the request.

### Instruction:
{{ instruction }}

### Response:
{{ response }}"""


SHAREGPT_TEMPLATE = """{{ system_prompt }}

USER: {{ instruction }}

ASSISTANT: {{ response }}"""


@tool
def dataset_generator(
    action: str,
    examples: Optional[List[Dict[str, Any]]] = None,
    output_file: Optional[str] = None,
    format: str = "qwen",
    template_name: Optional[str] = None,
    custom_template: Optional[str] = None,
    template_file: Optional[str] = None,
    count: int = None,
) -> Dict[str, Any]:
    """
    Generate training datasets for LLM fine-tuning with multiple format support.

    Actions:
    - generate: Generate dataset from examples
    - preview: Preview formatted examples without saving
    - list_formats: Show available formats
    - validate: Validate dataset structure

    Args:
        action: Action to perform
        examples: List of example dictionaries with fields like:
            - system_prompt: System prompt text
            - instruction: User instruction
            - response: Expected response
            - enable_thinking: Optional flag to add /think or /no_think control (Qwen3)
            - thinking_content: Optional thinking process for Qwen3 (wrapped in <think> tags)
            - tools: Optional list of tool definitions (must include "type": "function", "function": {...})
            - tool_calls: Optional list of tool call dicts
            - tool_responses: Optional list of tool response strings
        output_file: Output JSONL file path
        format: Dataset format (qwen, llama, alpaca, sharegpt, custom)
        template_name: Name of template from ./templates/ directory (e.g., "llama3.1", "llama3.2", "gpt-oss")
        custom_template: Custom Jinja2 template string
        template_file: Path to custom template file
        count: Number of examples to show in preview

    Returns:
        Dict with status and results

    Examples:
        # Use custom template from ./templates/
        dataset_generator(
            action="generate",
            template_name="llama3.1",
            output_file="llama_data.jsonl",
            examples=[...]
        )

        # Generate Qwen dataset with thinking mode (Qwen3)
        dataset_generator(
            action="generate",
            format="qwen",
            output_file="training_data.jsonl",
            examples=[
                {
                    "system_prompt": "You are a helpful math tutor.",
                    "instruction": "What is 15 * 7?",
                    "enable_thinking": True,  # Add /think flag
                    "thinking_content": "I need to multiply 15 by 7. Let me break this down: 15 * 7 = (10 * 7) + (5 * 7) = 70 + 35 = 105",
                    "response": "The answer is 105."
                }
            ]
        )

        # Preview without saving
        dataset_generator(
            action="preview",
            template_name="gpt-oss",
            examples=[...],
            count=2
        )
    """

    try:
        if action == "list_formats":
            return _list_formats()

        elif action == "generate":
            return _generate_dataset(
                examples=examples,
                output_file=output_file,
                format=format,
                template_name=template_name,
                custom_template=custom_template,
                template_file=template_file,
            )

        elif action == "preview":
            return _preview_dataset(
                examples=examples,
                format=format,
                template_name=template_name,
                custom_template=custom_template,
                template_file=template_file,
                count=count or 3,
            )

        elif action == "validate":
            return _validate_dataset(examples=examples, format=format)

        else:
            return {
                "status": "error",
                "content": [
                    {
                        "text": f"Unknown action: {action}. Valid: generate, preview, list_formats, validate"
                    }
                ],
            }

    except Exception as e:
        import traceback

        tb = traceback.format_exc()
        return {
            "status": "error",
            "content": [{"text": f"Error in dataset_generator: {str(e)}\n\n{tb}"}],
        }


def _get_template(
    format: str,
    template_name: Optional[str],
    custom_template: Optional[str],
    template_file: Optional[str],
) -> Template:
    """Get Jinja2 template for the specified format."""

    # Template from ./templates/ directory (highest priority)
    if template_name:
        template_dir = Path.cwd() / "templates"
        template_path = template_dir / f"{template_name}.j2"
        if template_path.exists():
            with open(template_path, "r") as f:
                template_str = f.read()
            return Template(template_str)
        else:
            raise ValueError(f"Template not found: {template_path}")

    # Custom template file
    if template_file:
        with open(template_file, "r") as f:
            template_str = f.read()
        return Template(template_str)

    # Custom template string
    if custom_template:
        return Template(custom_template)

    # Pre-defined formats
    format_map = {
        "qwen": QWEN_TEMPLATE,
        "llama": LLAMA_TEMPLATE,
        "alpaca": ALPACA_TEMPLATE,
        "sharegpt": SHAREGPT_TEMPLATE,
    }

    if format not in format_map:
        raise ValueError(
            f"Unknown format: {format}. Use 'custom_template' for custom formats."
        )

    return Template(format_map[format])


def _generate_dataset(
    examples: List[Dict[str, Any]],
    output_file: str,
    format: str,
    template_name: Optional[str],
    custom_template: Optional[str],
    template_file: Optional[str],
) -> Dict[str, Any]:
    """Generate dataset and save to JSONL."""

    if not examples:
        return {
            "status": "error",
            "content": [{"text": "Error: examples list is required"}],
        }

    if not output_file:
        return {
            "status": "error",
            "content": [{"text": "Error: output_file is required for generate action"}],
        }

    results = []
    format_name = template_name if template_name else format
    results.append(f"🎯 Generating dataset with format: {format_name}")
    results.append(f"📊 Number of examples: {len(examples)}")

    # Get template
    template = _get_template(format, template_name, custom_template, template_file)

    # Format examples
    formatted_examples = []
    for i, example in enumerate(examples):
        try:
            formatted_text = template.render(**example)
            formatted_examples.append({"text": formatted_text})
        except Exception as e:
            results.append(f"⚠️  Warning: Failed to format example {i+1}: {str(e)}")

    results.append(f"✅ Successfully formatted {len(formatted_examples)} examples")

    # Save to JSONL
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        for example in formatted_examples:
            f.write(json.dumps(example) + "\n")

    results.append(f"\n💾 Saved to: {output_file}")
    results.append(f"📁 File size: {output_path.stat().st_size / 1024:.2f} KB")

    # Show first example preview
    if formatted_examples:
        preview = formatted_examples[0]["text"][:300]
        results.append(f"\n📄 First example preview:\n{preview}...")

    return {"status": "success", "content": [{"text": "\n".join(results)}]}


def _preview_dataset(
    examples: List[Dict[str, Any]],
    format: str,
    template_name: Optional[str],
    custom_template: Optional[str],
    template_file: Optional[str],
    count: int,
) -> Dict[str, Any]:
    """Preview formatted examples without saving."""

    if not examples:
        return {
            "status": "error",
            "content": [{"text": "Error: examples list is required"}],
        }

    results = []
    format_name = template_name if template_name else format
    results.append(f"👀 Preview of {format_name} format\n")

    # Get template
    template = _get_template(format, template_name, custom_template, template_file)

    # Format and show examples
    for i, example in enumerate(examples[:count]):
        try:
            formatted_text = template.render(**example)
            results.append(f"{'='*60}")
            results.append(f"Example {i+1}:")
            results.append(f"{'='*60}")
            results.append(formatted_text)
            results.append("")
        except Exception as e:
            results.append(f"⚠️  Failed to format example {i+1}: {str(e)}\n")

    return {"status": "success", "content": [{"text": "\n".join(results)}]}


def _validate_dataset(examples: List[Dict[str, Any]], format: str) -> Dict[str, Any]:
    """Validate dataset structure."""

    if not examples:
        return {
            "status": "error",
            "content": [{"text": "Error: examples list is required"}],
        }

    results = []
    results.append(f"🔍 Validating {len(examples)} examples for {format} format\n")

    # Required fields by format
    required_fields = {
        "qwen": ["system_prompt", "instruction", "response"],
        "llama": ["system_prompt", "instruction", "response"],
        "alpaca": ["instruction", "response"],
        "sharegpt": ["system_prompt", "instruction", "response"],
    }

    required = required_fields.get(format, ["instruction", "response"])

    errors = []
    warnings = []

    for i, example in enumerate(examples):
        # Check required fields
        missing = [field for field in required if field not in example]
        if missing:
            errors.append(f"Example {i+1}: Missing required fields: {missing}")

        # Check tool calling format (Qwen)
        if format == "qwen" and "tools" in example:
            if "tool_calls" in example and not isinstance(example["tool_calls"], list):
                errors.append(f"Example {i+1}: tool_calls must be a list")
            if "tool_responses" in example and not isinstance(
                example["tool_responses"], list
            ):
                errors.append(f"Example {i+1}: tool_responses must be a list")

            # Validate tool call structure
            if "tool_calls" in example:
                for j, tc in enumerate(example["tool_calls"]):
                    if "name" not in tc or "arguments" not in tc:
                        errors.append(
                            f"Example {i+1}, tool_call {j+1}: Must have 'name' and 'arguments'"
                        )

    if errors:
        results.append("❌ Validation failed:\n")
        results.extend([f"  - {err}" for err in errors])
    else:
        results.append("✅ All examples valid!")

    if warnings:
        results.append("\n⚠️  Warnings:\n")
        results.extend([f"  - {warn}" for warn in warnings])

    results.append(f"\n📊 Summary:")
    results.append(f"  Total examples: {len(examples)}")
    results.append(f"  Errors: {len(errors)}")
    results.append(f"  Warnings: {len(warnings)}")

    return {
        "status": "success" if not errors else "error",
        "content": [{"text": "\n".join(results)}],
    }


def _list_formats() -> Dict[str, Any]:
    """List available dataset formats."""

    results = []
    results.append("📋 Available Dataset Formats:\n")

    # Check for custom templates in ./templates/
    template_dir = Path.cwd() / "templates"
    custom_templates = []
    if template_dir.exists():
        custom_templates = [f.stem for f in template_dir.glob("*.j2")]

    if custom_templates:
        results.append("🎨 **Custom Templates** (from ./templates/):")
        for tmpl in sorted(custom_templates):
            results.append(f"   - **{tmpl}** (use template_name='{tmpl}')")
        results.append("")

    results.append("📦 **Built-in Formats:**")
    results.append("")
    results.append("1. **qwen** - Qwen chat format with tool calling support")
    results.append("   Required: system_prompt, instruction, response")
    results.append(
        "   Optional: enable_thinking, thinking_content, tools, tool_calls, tool_responses"
    )
    results.append(
        "   Format: <|im_start|>...<|im_end|> with <think>, /think, /no_think, and XML tool tags\n"
    )

    results.append("2. **llama** - Llama 3 chat format")
    results.append("   Required: system_prompt, instruction, response")
    results.append("   Format: <|begin_of_text|>...<|eot_id|>\n")

    results.append("3. **alpaca** - Alpaca instruction format")
    results.append("   Required: instruction, response")
    results.append("   Format: ### Instruction / ### Response\n")

    results.append("4. **sharegpt** - ShareGPT conversation format")
    results.append("   Required: system_prompt, instruction, response")
    results.append("   Format: USER: / ASSISTANT:\n")

    results.append("5. **custom** - Use custom_template or template_file")
    results.append("   Provide Jinja2 template with your format\n")

    results.append("💡 Use template_name to load templates from ./templates/ directory")
    results.append("💡 Use action='preview' to see formatted examples before saving")

    return {"status": "success", "content": [{"text": "\n".join(results)}]}
