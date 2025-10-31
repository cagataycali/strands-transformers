"""HuggingFace Transformers model provider.

Direct integration with locally loaded transformers models, supporting
fine-tuned models with merged LoRA weights.

- Docs: https://huggingface.co/docs/transformers
"""

import json
import logging
import time
from typing import (
    Any,
    AsyncGenerator,
    Dict,
    Optional,
    Type,
    TypeVar,
    Union,
    cast,
)

import torch
from pydantic import BaseModel
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TextIteratorStreamer,
)
from threading import Thread
from typing_extensions import TypedDict, Unpack, override

from strands.types.content import ContentBlock, Messages
from strands.types.streaming import StreamEvent
from strands.types.tools import ToolChoice, ToolSpec
from strands.models._validation import (
    validate_config_keys,
    warn_on_tool_choice_not_supported,
)
from strands.models.model import Model

logger = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)


class TransformerModel(Model):
    """HuggingFace Transformers model provider implementation.

    Loads models directly from HuggingFace transformers without requiring
    external servers or format conversion. Ideal for fine-tuned models with
    merged LoRA weights.

    The implementation handles:
    - Local model loading with device management
    - Streaming responses with TextIteratorStreamer
    - Tool/function calling
    - Qwen3 thinking mode
    - Chat template formatting

    Example:
        Basic usage:
        >>> model = TransformerModel(model_path="./my_finetuned_model")
        >>> model.update_config(params={"temperature": 0.7, "max_tokens": 100})

        With custom device:
        >>> model = TransformerModel(
        ...     model_path="Qwen/Qwen3-1.7B",
        ...     device="cuda"
        ... )

        Qwen3 with thinking mode:
        >>> model = TransformerModel(
        ...     model_path="Qwen/Qwen3-1.7B",
        ...     enable_thinking=True
        ... )
    """

    class TransformerConfig(TypedDict, total=False):
        """Configuration options for HuggingFace Transformers models.

        Attributes:
            model_path: Path to model directory or HuggingFace model ID.
            device: Device to use ("cpu", "cuda", "mps", or "auto").
                Default is "auto" which selects cuda if available, else cpu.
            params: Model parameters for generation:
                - max_tokens: Maximum number of tokens to generate
                - temperature: Sampling temperature (0.0 to 2.0)
                - top_p: Nucleus sampling parameter (0.0 to 1.0)
                - top_k: Top-k sampling (default: 20)
                - do_sample: Whether to use sampling (default: True)
                - repetition_penalty: Penalize repeat tokens (default: 1.0)
            enable_thinking: Enable Qwen3 thinking mode (default: True)
            trust_remote_code: Trust remote code when loading model (default: True)
            low_cpu_mem_usage: Use low CPU memory mode (default: False)
        """

        model_path: str
        device: Optional[str]
        params: Optional[Dict[str, Any]]
        enable_thinking: bool
        trust_remote_code: bool
        low_cpu_mem_usage: bool

    def __init__(
        self,
        model_path: str,
        device: Optional[str] = None,
        **model_config: Unpack[TransformerConfig],
    ) -> None:
        """Initialize transformers provider instance.

        Args:
            model_path: Path to model directory or HuggingFace model ID.
            device: Device to use ("cpu", "cuda", "mps", or "auto").
            **model_config: Configuration options for the transformers model.
        """
        validate_config_keys(model_config, self.TransformerConfig)

        # Set defaults
        if "model_path" not in model_config:
            model_config["model_path"] = model_path
        if "device" not in model_config:
            model_config["device"] = device or "auto"
        if "enable_thinking" not in model_config:
            model_config["enable_thinking"] = True
        if "trust_remote_code" not in model_config:
            model_config["trust_remote_code"] = True
        if "low_cpu_mem_usage" not in model_config:
            model_config["low_cpu_mem_usage"] = False

        self.config = dict(model_config)
        logger.debug("config=<%s> | initializing", self.config)

        # Determine device
        device_config = self.config["device"]
        if device_config == "auto":
            if torch.cuda.is_available():
                self.device = "cuda"
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                self.device = "mps"
            else:
                self.device = "cpu"
        else:
            self.device = device_config

        logger.debug("device=<%s> | selected", self.device)

        # Load model and tokenizer
        self._load_model()

    def _load_model(self) -> None:
        """Load model and tokenizer from path."""
        model_path = self.config["model_path"]
        logger.debug("model_path=<%s> | loading", model_path)

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=self.config["trust_remote_code"],
        )

        # Set padding token
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Load model
        model_kwargs = {
            "trust_remote_code": self.config["trust_remote_code"],
            "low_cpu_mem_usage": self.config["low_cpu_mem_usage"],
        }

        # Use bfloat16 if available
        if self.device == "cuda":
            model_kwargs["torch_dtype"] = torch.bfloat16

        self.model = AutoModelForCausalLM.from_pretrained(model_path, **model_kwargs)

        # Move to device
        if self.device == "cuda":
            self.model = self.model.to(self.device)
        # Note: For CPU and MPS, model stays where it was loaded

        logger.debug("model loaded successfully")

        # Detect Qwen3 for thinking mode
        self.is_qwen3 = "qwen3" in model_path.lower() or (
            hasattr(self.model.config, "model_type")
            and self.model.config.model_type == "qwen3"
        )

    @override
    def update_config(self, **model_config: Unpack[TransformerConfig]) -> None:  # type: ignore[override]
        """Update the transformers model configuration with provided arguments.

        Args:
            **model_config: Configuration overrides.
        """
        validate_config_keys(model_config, self.TransformerConfig)

        # If model_path changed, reload model
        if "model_path" in model_config and model_config[
            "model_path"
        ] != self.config.get("model_path"):
            self.config.update(model_config)
            self._load_model()
        else:
            self.config.update(model_config)

    @override
    def get_config(self) -> TransformerConfig:
        """Get the transformers model configuration.

        Returns:
            The transformers model configuration.
        """
        return self.config  # type: ignore[return-value]

    def _format_message_content(
        self, content: Union[ContentBlock, Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Format a content block for transformers.

        Args:
            content: Message content.

        Returns:
            Transformers compatible content block.

        Raises:
            TypeError: If the content block type cannot be converted to a compatible format.
        """
        if "text" in content:
            return {"type": "text", "text": content["text"]}

        if "image" in content:
            # For multimodal models, would need special handling
            return {"type": "image", "image": content["image"]["source"]["bytes"]}

        if "toolUse" in content:
            return {
                "type": "tool_use",
                "name": content["toolUse"]["name"],
                "input": content["toolUse"]["input"],
                "id": content["toolUse"]["toolUseId"],
            }

        if "toolResult" in content:
            return {
                "type": "tool_result",
                "tool_use_id": content["toolResult"]["toolUseId"],
                "content": [
                    {"text": json.dumps(c["json"])} if "json" in c else c
                    for c in content["toolResult"]["content"]
                ],
            }

        raise TypeError(f"content_type=<{next(iter(content))}> | unsupported type")

    def _format_messages(
        self,
        messages: Messages,
        system_prompt: Optional[str] = None,
        tool_specs: Optional[list[ToolSpec]] = None,
    ) -> str:
        """Format messages for transformers using chat template.

        Args:
            messages: List of message objects to be processed.
            system_prompt: System prompt to provide context to the model.
            tool_specs: List of tool specifications to make available to the model.

        Returns:
            Formatted prompt string ready for tokenization.
        """
        # Build chat messages
        chat_messages = []

        # Add system prompt with tool definitions if provided
        system_content = system_prompt or ""

        # Add tool specifications to system prompt
        if tool_specs:
            tools_description = (
                "\n\n# Available Tools\n\nYou have access to the following tools:\n\n"
            )
            for tool_spec in tool_specs:
                tools_description += f"## {tool_spec['name']}\n"
                tools_description += f"{tool_spec['description']}\n\n"
                tools_description += f"Parameters: {json.dumps(tool_spec['inputSchema']['json'], indent=2)}\n\n"

            tools_description += (
                "\nTo use a tool, output:\n"
                '<tool_call>{"name": "tool_name", "arguments": {"param": "value"}}</tool_call>\n\n'
                "You will receive the result in:\n"
                "<tool_response>result</tool_response>\n"
            )
            system_content += tools_description

        if system_content:
            chat_messages.append({"role": "system", "content": system_content})

        # Convert Strands messages to chat format
        for message in messages:
            role = message["role"]
            contents = message["content"]

            # Combine text contents
            text_parts = []
            tool_uses = []
            tool_results = []

            for content in contents:
                if "text" in content:
                    text_parts.append(content["text"])
                elif "toolUse" in content:
                    tool_uses.append(content["toolUse"])
                elif "toolResult" in content:
                    tool_results.append(content["toolResult"])

            # Create message
            if text_parts:
                chat_messages.append({"role": role, "content": " ".join(text_parts)})

            # Handle tool uses
            if tool_uses:
                # For now, convert tool uses to text (simplified)
                # Full implementation would require model-specific formatting
                for tool_use in tool_uses:
                    tool_text = f"<tool_call>{json.dumps({'name': tool_use['name'], 'arguments': tool_use['input']})}</tool_call>"
                    chat_messages.append({"role": "assistant", "content": tool_text})

            # Handle tool results
            if tool_results:
                for tool_result in tool_results:
                    result_content = " ".join(
                        [
                            (
                                json.dumps(c["json"])
                                if "json" in c
                                else c.get("text", str(c))
                            )
                            for c in tool_result["content"]
                        ]
                    )
                    result_text = f"<tool_response>{result_content}</tool_response>"
                    chat_messages.append({"role": "user", "content": result_text})

        # Apply chat template
        if self.is_qwen3 and self.config.get("enable_thinking", True):
            # Qwen3 with thinking mode
            formatted_prompt = self.tokenizer.apply_chat_template(
                chat_messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=True,
            )
        else:
            # Regular chat template
            formatted_prompt = self.tokenizer.apply_chat_template(
                chat_messages,
                tokenize=False,
                add_generation_prompt=True,
            )

        return formatted_prompt

    def _format_chunk(self, event: Dict[str, Any]) -> StreamEvent:
        """Format a generation event into a standardized message chunk.

        Args:
            event: A generation event.

        Returns:
            The formatted chunk.
        """
        match event["chunk_type"]:
            case "message_start":
                return {"messageStart": {"role": "assistant"}}

            case "content_start":
                if event.get("data_type") == "tool":
                    tool_name = event.get("tool_name", "unknown")
                    tool_use_id = event.get("tool_use_id", tool_name)
                    return {
                        "contentBlockStart": {
                            "start": {
                                "toolUse": {
                                    "name": tool_name,
                                    "toolUseId": tool_use_id,
                                }
                            }
                        }
                    }
                return {"contentBlockStart": {"start": {}}}

            case "content_delta":
                if event.get("data_type") == "thinking":
                    return {
                        "contentBlockDelta": {
                            "delta": {"reasoningContent": {"text": event["data"]}}
                        }
                    }
                if event.get("data_type") == "tool":
                    tool_arguments = event.get("tool_arguments", {})
                    return {
                        "contentBlockDelta": {
                            "delta": {"toolUse": {"input": json.dumps(tool_arguments)}}
                        }
                    }
                return {"contentBlockDelta": {"delta": {"text": event["data"]}}}

            case "content_stop":
                return {"contentBlockStop": {}}

            case "message_stop":
                return {"messageStop": {"stopReason": event.get("reason", "end_turn")}}

            case "metadata":
                return {
                    "metadata": {
                        "usage": {
                            "inputTokens": event.get("input_tokens", 0),
                            "outputTokens": event.get("output_tokens", 0),
                            "totalTokens": event.get("input_tokens", 0)
                            + event.get("output_tokens", 0),
                        },
                        "metrics": {
                            "latencyMs": event.get("latency_ms", 0),
                        },
                    },
                }

            case _:
                raise RuntimeError(f"chunk_type=<{event['chunk_type']}> | unknown type")

    @override
    async def stream(
        self,
        messages: Messages,
        tool_specs: Optional[list[ToolSpec]] = None,
        system_prompt: Optional[str] = None,
        *,
        tool_choice: ToolChoice | None = None,
        **kwargs: Any,
    ) -> AsyncGenerator[StreamEvent, None]:
        """Stream conversation with the transformers model.

        Args:
            messages: List of message objects to be processed by the model.
            tool_specs: List of tool specifications to make available to the model.
            system_prompt: System prompt to provide context to the model.
            tool_choice: Selection strategy for tool invocation. **Note: This parameter is accepted for
                interface consistency but is currently ignored for this model provider.**
            **kwargs: Additional keyword arguments for future extensibility.

        Yields:
            Formatted message chunks from the model.
        """
        warn_on_tool_choice_not_supported(tool_choice)

        # Track start time
        start_time = time.perf_counter()

        logger.debug("formatting messages")
        formatted_prompt = self._format_messages(messages, system_prompt, tool_specs)
        logger.debug(
            "prompt=<%s>",
            (
                formatted_prompt[:200] + "..."
                if len(formatted_prompt) > 200
                else formatted_prompt
            ),
        )

        # Tokenize
        inputs = self.tokenizer([formatted_prompt], return_tensors="pt")
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
        input_length = inputs["input_ids"].shape[1]

        # Get generation parameters
        params = self.config.get("params", {})
        max_tokens = params.get("max_tokens", 300)
        temperature = params.get("temperature", 1)
        top_p = params.get("top_p", 0.9)
        top_k = params.get("top_k", 20)
        do_sample = params.get("do_sample", True)
        repetition_penalty = params.get("repetition_penalty", 1.0)

        logger.debug("generating with streaming")

        # Create streamer
        # NOTE: skip_special_tokens=True removes chat template tokens like <|im_end|>
        # but preserves regular vocabulary tokens like <think> and </think>
        streamer = TextIteratorStreamer(
            self.tokenizer,
            skip_prompt=True,
            skip_special_tokens=True,  # Skip chat template tokens, keep <think> tags
        )

        # Generation kwargs
        generation_kwargs = dict(
            inputs=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_new_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            do_sample=do_sample,
            repetition_penalty=repetition_penalty,
            pad_token_id=self.tokenizer.eos_token_id,
            streamer=streamer,
        )

        # Start generation in thread
        thread = Thread(target=self.model.generate, kwargs=generation_kwargs)
        thread.start()

        # Yield events
        yield self._format_chunk({"chunk_type": "message_start"})
        yield self._format_chunk({"chunk_type": "content_start"})

        # Stream tokens
        thinking_mode = False
        tool_call_mode = False
        tool_requested = False
        thinking_buffer = ""
        tool_call_buffer = ""
        output_buffer = ""

        for new_text in streamer:
            # Check for tool call markers
            if "<tool_call>" in new_text:
                # Entering tool call mode
                tool_call_mode = True
                tool_requested = True
                parts = new_text.split("<tool_call>", 1)
                if parts[0]:
                    # Yield any text before tool call
                    yield self._format_chunk(
                        {"chunk_type": "content_delta", "data": parts[0]}
                    )
                    output_buffer += parts[0]
                # Close text content block
                yield self._format_chunk({"chunk_type": "content_stop"})
                if len(parts) > 1:
                    tool_call_buffer = parts[1]
                continue

            # Check if we're exiting tool call mode
            if tool_call_mode and "</tool_call>" in new_text:
                tool_call_mode = False
                parts = new_text.split("</tool_call>", 1)
                tool_call_buffer += parts[0]

                # Parse and emit tool call
                try:
                    tool_call_data = json.loads(tool_call_buffer)
                    tool_name = tool_call_data.get("name", "unknown")
                    tool_arguments = tool_call_data.get("arguments", {})

                    # Emit tool use events
                    yield self._format_chunk(
                        {
                            "chunk_type": "content_start",
                            "data_type": "tool",
                            "tool_name": tool_name,
                            "tool_use_id": tool_name,
                        }
                    )
                    yield self._format_chunk(
                        {
                            "chunk_type": "content_delta",
                            "data_type": "tool",
                            "tool_arguments": tool_arguments,
                        }
                    )
                    yield self._format_chunk(
                        {
                            "chunk_type": "content_stop",
                            "data_type": "tool",
                        }
                    )
                except json.JSONDecodeError as e:
                    logger.warning(
                        f"Failed to parse tool call: {e}, buffer: {tool_call_buffer}"
                    )

                tool_call_buffer = ""

                # Start new content block for remaining text
                if len(parts) > 1 and parts[1]:
                    yield self._format_chunk({"chunk_type": "content_start"})
                    yield self._format_chunk(
                        {"chunk_type": "content_delta", "data": parts[1]}
                    )
                    output_buffer += parts[1]
                continue

            # Accumulate in tool call buffer if in tool call mode
            if tool_call_mode:
                tool_call_buffer += new_text
                continue

            # Check for thinking mode markers (Qwen3)
            if self.is_qwen3 and self.config.get("enable_thinking", True):
                # Check if we're entering thinking mode
                if "<think>" in new_text:
                    thinking_mode = True
                    parts = new_text.split("<think>", 1)
                    if parts[0]:
                        yield self._format_chunk(
                            {"chunk_type": "content_delta", "data": parts[0]}
                        )
                        output_buffer += parts[0]
                    if len(parts) > 1:
                        thinking_buffer = parts[1]
                    continue

                # Check if we're exiting thinking mode
                if thinking_mode and "</think>" in new_text:
                    thinking_mode = False
                    parts = new_text.split("</think>", 1)
                    thinking_buffer += parts[0]
                    # Yield thinking content as reasoning
                    if thinking_buffer:
                        yield self._format_chunk(
                            {
                                "chunk_type": "content_delta",
                                "data_type": "thinking",
                                "data": thinking_buffer,
                            }
                        )
                    thinking_buffer = ""
                    if len(parts) > 1 and parts[1]:
                        yield self._format_chunk(
                            {"chunk_type": "content_delta", "data": parts[1]}
                        )
                        output_buffer += parts[1]
                    continue

                # Accumulate in appropriate buffer
                if thinking_mode:
                    thinking_buffer += new_text
                    yield self._format_chunk(
                        {
                            "chunk_type": "content_delta",
                            "data_type": "thinking",
                            "data": new_text,
                        }
                    )
                else:
                    output_buffer += new_text
                    yield self._format_chunk(
                        {"chunk_type": "content_delta", "data": new_text}
                    )
            else:
                # Regular streaming without thinking mode
                output_buffer += new_text
                yield self._format_chunk(
                    {"chunk_type": "content_delta", "data": new_text}
                )

        thread.join()

        yield self._format_chunk({"chunk_type": "content_stop"})
        yield self._format_chunk(
            {
                "chunk_type": "message_stop",
                "reason": "tool_use" if tool_requested else "end_turn",
            }
        )

        # Calculate metrics
        latency_ms = int((time.perf_counter() - start_time) * 1000)
        output_tokens = len(
            self.tokenizer.encode(
                output_buffer + thinking_buffer, add_special_tokens=False
            )
        )

        yield self._format_chunk(
            {
                "chunk_type": "metadata",
                "input_tokens": input_length,
                "output_tokens": output_tokens,
                "latency_ms": latency_ms,
            }
        )

        logger.debug("finished streaming response from model")

    @override
    async def structured_output(
        self,
        output_model: Type[T],
        prompt: Messages,
        system_prompt: Optional[str] = None,
        **kwargs: Any,
    ) -> AsyncGenerator[Dict[str, Union[T, Any]], None]:
        """Get structured output from the model.

        This implementation uses the model's generation with JSON schema guidance
        if supported, or falls back to prompt engineering.

        Args:
            output_model: The Pydantic model defining the expected output structure.
            prompt: The prompt messages to use for generation.
            system_prompt: System prompt to provide context to the model.
            **kwargs: Additional keyword arguments for future extensibility.

        Yields:
            Model events with the last being the structured output.
        """
        # Add JSON schema instruction to system prompt
        schema = output_model.model_json_schema()
        json_instruction = f"\n\nPlease respond with valid JSON matching this schema:\n{json.dumps(schema, indent=2)}"

        augmented_system_prompt = (system_prompt or "") + json_instruction

        # Collect the response
        response_text = ""
        async for event in self.stream(
            prompt, system_prompt=augmented_system_prompt, **kwargs
        ):
            if "contentBlockDelta" in event:
                delta = event["contentBlockDelta"]["delta"]
                if "text" in delta:
                    response_text += delta["text"]
            # Forward events to caller
            yield cast(Dict[str, Union[T, Any]], event)

        # Parse and validate the JSON response
        try:
            # Extract JSON from markdown code blocks if present
            if "```json" in response_text:
                response_text = (
                    response_text.split("```json")[1].split("```")[0].strip()
                )
            elif "```" in response_text:
                response_text = response_text.split("```")[1].split("```")[0].strip()

            data = json.loads(response_text.strip())
            output_instance = output_model(**data)
            yield {"output": output_instance}
        except Exception as e:
            raise ValueError(
                f"Failed to parse structured output: {e}\nResponse: {response_text}"
            ) from e
