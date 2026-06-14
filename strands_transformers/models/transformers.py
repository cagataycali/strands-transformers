"""HuggingFace Transformers model provider.

Direct integration with locally loaded transformers models, supporting
fine-tuned models with merged LoRA weights AND multimodal (vision-language)
models that consume image / video / document content blocks.

- Docs: https://huggingface.co/docs/transformers
"""

import io
import json
import re
import logging
import time
from typing import (
    Any,
    AsyncGenerator,
    Dict,
    List,
    Optional,
    Tuple,
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


def _extract_json(text: str) -> str:
    """Pull a JSON object/array out of a model response.

    Handles reasoning models that prefix output with <think>...</think>, and
    JSON wrapped in markdown code fences. Falls back to the first balanced
    {...} or [...] span found in the text.
    """
    # strip closed <think>...</think> reasoning blocks (Qwen3 etc.)
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    # for an unterminated <think> (no closing tag), drop the tag itself but keep
    # the trailing content (which usually contains the JSON answer)
    text = text.replace("<think>", "")

    # markdown code fences
    if "```json" in text:
        text = text.split("```json", 1)[1].split("```", 1)[0]
    elif "```" in text:
        parts = text.split("```")
        if len(parts) >= 3:
            text = parts[1]

    text = text.strip()

    # already valid?
    try:
        json.loads(text)
        return text
    except Exception:
        pass

    # find first balanced JSON object or array
    for open_ch, close_ch in (("{", "}"), ("[", "]")):
        start = text.find(open_ch)
        if start == -1:
            continue
        depth = 0
        for i in range(start, len(text)):
            if text[i] == open_ch:
                depth += 1
            elif text[i] == close_ch:
                depth -= 1
                if depth == 0:
                    return text[start : i + 1]
    return text


def _bytes_to_pil(data: Any) -> Optional[Any]:
    """Best-effort decode of image bytes (or a passthrough PIL/np image) to PIL.Image.

    Returns None if PIL is unavailable or the payload can't be decoded.
    """
    try:
        from PIL import Image
    except Exception:  # pragma: no cover - PIL should be present for VLMs
        return None

    # Already a PIL image
    if Image.isImageType(data):
        return data

    # numpy array -> PIL
    try:
        import numpy as np

        if isinstance(data, np.ndarray):
            return Image.fromarray(data)
    except Exception:
        pass

    # raw bytes -> PIL
    if isinstance(data, (bytes, bytearray)):
        try:
            return Image.open(io.BytesIO(bytes(data))).convert("RGB")
        except Exception as e:
            logger.warning("failed to decode image bytes: %s", e)
            return None

    # data URI / file path string
    if isinstance(data, str):
        try:
            if data.startswith("data:"):
                import base64

                b64 = data.split(",", 1)[1]
                return Image.open(io.BytesIO(base64.b64decode(b64))).convert("RGB")
            return Image.open(data).convert("RGB")
        except Exception as e:
            logger.warning("failed to open image path/uri: %s", e)
            return None

    return None


def _normalize_video(payload: Any) -> Optional[Any]:
    """Normalize a video payload into a (T, H, W, C) uint8 numpy array.

    Accepts: a list of frames (PIL/np), a 4D numpy array, or raw container
    bytes. Returns None if it cannot be decoded.
    """
    try:
        import numpy as np
    except Exception:
        return None

    # Already a 4D array (T, H, W, C)
    if isinstance(payload, np.ndarray):
        return payload if payload.ndim == 4 else None

    # A list/tuple of frames (PIL images or arrays)
    if isinstance(payload, (list, tuple)) and payload:
        frames = []
        for fr in payload:
            pil = _bytes_to_pil(fr)
            if pil is None:
                if isinstance(fr, np.ndarray):
                    frames.append(fr)
                continue
            frames.append(np.asarray(pil))
        if frames:
            try:
                return np.stack(frames, axis=0)
            except Exception:
                return None
        return None

    # Raw container bytes -> sample frames via torchvision/decord if available
    if isinstance(payload, (bytes, bytearray)):
        try:
            import io as _io
            import tempfile

            with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tf:
                tf.write(bytes(payload))
                path = tf.name
            try:
                from torchvision import io as tvio

                vframes, _, _ = tvio.read_video(path, pts_unit="sec")
                # (T, H, W, C) uint8
                return vframes.numpy()
            except Exception as e:
                logger.warning("could not decode video bytes: %s", e)
                return None
        except Exception as e:
            logger.warning("video temp write failed: %s", e)
            return None

    return None


def _document_to_text(doc: Dict[str, Any]) -> str:
    """Flatten a Strands DocumentContent block into plain text for the prompt."""
    name = doc.get("name", "document")
    fmt = doc.get("format", "txt")
    source = doc.get("source", {}) or {}
    raw = source.get("bytes") if isinstance(source, dict) else None

    text_body = ""
    if isinstance(raw, (bytes, bytearray)):
        try:
            text_body = bytes(raw).decode("utf-8", errors="replace")
        except Exception:
            text_body = f"<{len(raw)} bytes of binary {fmt} document>"
    elif isinstance(raw, str):
        text_body = raw
    elif isinstance(source, list):
        # DocumentSource may be a list of {text: ...} blocks
        text_body = "\n".join(
            b.get("text", "") for b in source if isinstance(b, dict)
        )

    header = f"[Document: {name} ({fmt})]"
    return f"{header}\n{text_body}".strip()


T = TypeVar("T", bound=BaseModel)


class TransformerModel(Model):
    """HuggingFace Transformers model provider implementation.

    Loads models directly from HuggingFace transformers without requiring
    external servers or format conversion. Ideal for fine-tuned models with
    merged LoRA weights.

    Supports BOTH:
    - text-only causal LMs (AutoModelForCausalLM + AutoTokenizer)
    - multimodal vision-language models (AutoModelForImageTextToText +
      AutoProcessor) that consume image / video / document content blocks.

    The implementation handles:
    - Local model loading with device management
    - Automatic multimodal detection (processor with an image_processor)
    - Streaming responses with TextIteratorStreamer
    - Tool/function calling
    - Qwen3 thinking mode
    - Chat template formatting (tokenizer OR processor)
    - image / video / document content blocks, including media returned
      inside a toolResult

    Example:
        Text-only:
        >>> model = TransformerModel(model_path="Qwen/Qwen3-0.6B")

        Vision-language (multimodal auto-detected):
        >>> model = TransformerModel(model_path="HuggingFaceTB/SmolVLM-256M-Instruct")
        >>> # now an Agent(model=model) can be passed image content blocks
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
            multimodal: Force-enable/disable multimodal processor path.
                Default: auto-detected from the loaded processor.
        """

        model_path: str
        device: Optional[str]
        params: Optional[Dict[str, Any]]
        enable_thinking: bool
        trust_remote_code: bool
        low_cpu_mem_usage: bool
        multimodal: Optional[bool]

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

        # Multimodal state (resolved during load)
        self.processor = None
        self.is_multimodal = False

        # Load model and tokenizer/processor
        self._load_model()

    def _load_model(self) -> None:
        """Load model and tokenizer (and processor, if multimodal) from path."""
        model_path = self.config["model_path"]
        trust = self.config["trust_remote_code"]
        logger.debug("model_path=<%s> | loading", model_path)

        force_mm = self.config.get("multimodal")

        # ── Try the multimodal (processor) path first, unless explicitly disabled ──
        processor = None
        if force_mm is not False:
            try:
                from transformers import AutoProcessor

                processor = AutoProcessor.from_pretrained(
                    model_path, trust_remote_code=trust
                )
                # A genuine multimodal processor exposes an image_processor.
                has_image = getattr(processor, "image_processor", None) is not None
                if force_mm or has_image:
                    self.is_multimodal = True
            except Exception as e:
                logger.debug("AutoProcessor unavailable for %s: %s", model_path, e)
                processor = None

        if self.is_multimodal and processor is not None:
            self.processor = processor
            # Tokenizer lives on the processor for chat templating / counting
            self.tokenizer = getattr(processor, "tokenizer", None)
            if self.tokenizer is None:
                self.tokenizer = AutoTokenizer.from_pretrained(
                    model_path, trust_remote_code=trust
                )

            model_kwargs: Dict[str, Any] = {
                "trust_remote_code": trust,
                "low_cpu_mem_usage": self.config["low_cpu_mem_usage"],
            }
            if self.device == "cuda":
                model_kwargs["torch_dtype"] = torch.bfloat16
                model_kwargs["device_map"] = "cuda"

            # Prefer the generic image-text-to-text class; fall back to Vision2Seq.
            self.model = None
            for cls_name in (
                "AutoModelForImageTextToText",
                "AutoModelForVision2Seq",
            ):
                try:
                    import transformers as _tf

                    ModelCls = getattr(_tf, cls_name, None)
                    if ModelCls is None:
                        continue
                    self.model = ModelCls.from_pretrained(model_path, **model_kwargs)
                    logger.debug("loaded multimodal model via %s", cls_name)
                    break
                except Exception as e:
                    logger.debug("%s failed for %s: %s", cls_name, model_path, e)
                    continue

            if self.model is None:
                # Last resort: AutoModel
                from transformers import AutoModel

                self.model = AutoModel.from_pretrained(model_path, **model_kwargs)

            if self.device == "cuda" and "device_map" not in model_kwargs:
                self.model = self.model.to(self.device)

            logger.debug("multimodal model loaded successfully")
        else:
            # ── Text-only path (unchanged behaviour) ──
            self.is_multimodal = False
            self.processor = None
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_path, trust_remote_code=trust
            )

            model_kwargs = {
                "trust_remote_code": trust,
                "low_cpu_mem_usage": self.config["low_cpu_mem_usage"],
            }
            if self.device == "cuda":
                model_kwargs["torch_dtype"] = torch.bfloat16

            self.model = AutoModelForCausalLM.from_pretrained(
                model_path, **model_kwargs
            )
            if self.device == "cuda":
                self.model = self.model.to(self.device)

            logger.debug("text model loaded successfully")

        # Set padding token
        if self.tokenizer is not None and self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Detect Qwen3 for thinking mode
        self.is_qwen3 = "qwen3" in model_path.lower() or (
            hasattr(self.model, "config")
            and getattr(self.model.config, "model_type", "") == "qwen3"
        )

    @override
    def update_config(self, **model_config: Unpack[TransformerConfig]) -> None:  # type: ignore[override]
        """Update the transformers model configuration with provided arguments."""
        validate_config_keys(model_config, self.TransformerConfig)

        if "model_path" in model_config and model_config[
            "model_path"
        ] != self.config.get("model_path"):
            self.config.update(model_config)
            self._load_model()
        else:
            self.config.update(model_config)

    @override
    def get_config(self) -> TransformerConfig:
        """Get the transformers model configuration."""
        return self.config  # type: ignore[return-value]

    # ──────────────────────────────────────────────────────────────────────
    # Multimodal content-block handling
    # ──────────────────────────────────────────────────────────────────────

    def _content_to_processor_parts(
        self, content: Union[ContentBlock, Dict[str, Any]]
    ) -> Tuple[List[Dict[str, Any]], List[Any], List[Any]]:
        """Convert a single Strands content block into processor chat parts.

        Returns a tuple of (parts, images, videos) where ``parts`` are the
        chat-template entries (e.g. {"type": "text"|"image"|"video"}) and
        ``images``/``videos`` are the decoded media objects collected in order.

        Supports the full multimodal taxonomy: text, image, video, document,
        toolUse, and toolResult (whose content may itself carry image/video).
        """
        parts: List[Dict[str, Any]] = []
        images: List[Any] = []
        videos: List[Any] = []

        # text
        if "text" in content:
            parts.append({"type": "text", "text": content["text"]})
            return parts, images, videos

        # image (Strands ImageContent: {"format","source":{"bytes"}}; also tolerate
        # bare PIL/bytes or the run-path {"image": <PIL>})
        if "image" in content:
            img = content["image"]
            payload = img
            if isinstance(img, dict):
                src = img.get("source", img)
                payload = src.get("bytes", src) if isinstance(src, dict) else src
            pil = _bytes_to_pil(payload)
            if pil is not None:
                images.append(pil)
                parts.append({"type": "image"})
            else:
                parts.append({"type": "text", "text": "[unrenderable image]"})
            return parts, images, videos

        # video
        if "video" in content:
            vid = content["video"]
            src = vid.get("source", vid) if isinstance(vid, dict) else vid
            payload = src.get("bytes", src) if isinstance(src, dict) else src
            norm = _normalize_video(payload)
            if norm is not None:
                videos.append(norm)
                parts.append({"type": "video"})
            else:
                parts.append({"type": "text", "text": "[unrenderable video]"})
            return parts, images, videos

        # document -> flatten to text
        if "document" in content:
            parts.append({"type": "text", "text": _document_to_text(content["document"])})
            return parts, images, videos

        # toolUse -> compact text marker
        if "toolUse" in content:
            tu = content["toolUse"]
            parts.append(
                {
                    "type": "text",
                    "text": "<tool_call>"
                    + json.dumps({"name": tu.get("name"), "arguments": tu.get("input")})
                    + "</tool_call>",
                }
            )
            return parts, images, videos

        # toolResult -> fold text/json as text, and pull media into the turn
        if "toolResult" in content:
            tr = content["toolResult"]
            for c in tr.get("content", []):
                if "json" in c:
                    parts.append({"type": "text", "text": json.dumps(c["json"])})
                elif "text" in c:
                    parts.append({"type": "text", "text": c["text"]})
                elif "image" in c:
                    img = c["image"]
                    src = img.get("source", img) if isinstance(img, dict) else img
                    payload = src.get("bytes", src) if isinstance(src, dict) else src
                    pil = _bytes_to_pil(payload)
                    if pil is not None:
                        images.append(pil)
                        parts.append({"type": "image"})
                elif "video" in c:
                    vid = c["video"]
                    src = vid.get("source", vid) if isinstance(vid, dict) else vid
                    vpayload = src.get("bytes", src) if isinstance(src, dict) else src
                    vnorm = _normalize_video(vpayload)
                    if vnorm is not None:
                        videos.append(vnorm)
                        parts.append({"type": "video"})
            if not parts:
                parts.append({"type": "text", "text": "[empty tool result]"})
            return parts, images, videos

        # unknown -> stringify
        parts.append({"type": "text", "text": str(content)})
        return parts, images, videos

    def _build_multimodal_chat(
        self,
        messages: Messages,
        system_prompt: Optional[str],
        tool_specs: Optional[list[ToolSpec]],
    ) -> Tuple[List[Dict[str, Any]], List[Any], List[Any]]:
        """Build processor chat messages + ordered media lists."""
        chat: List[Dict[str, Any]] = []
        images: List[Any] = []
        videos: List[Any] = []

        sys_text = system_prompt or ""
        if tool_specs:
            sys_text += self._tool_specs_to_text(tool_specs)
        if sys_text:
            chat.append({"role": "system", "content": [{"type": "text", "text": sys_text}]})

        for message in messages:
            role = message["role"]
            parts: List[Dict[str, Any]] = []
            for content in message["content"]:
                p, imgs, vids = self._content_to_processor_parts(content)
                parts.extend(p)
                images.extend(imgs)
                videos.extend(vids)
            if parts:
                chat.append({"role": role, "content": parts})

        return chat, images, videos

    @staticmethod
    def _tool_specs_to_text(tool_specs: list[ToolSpec]) -> str:
        desc = "\n\n# Available Tools\n\nYou have access to the following tools:\n\n"
        for spec in tool_specs:
            desc += f"## {spec['name']}\n{spec['description']}\n\n"
            desc += f"Parameters: {json.dumps(spec['inputSchema']['json'], indent=2)}\n\n"
        desc += (
            "\nTo use a tool, output:\n"
            '<tool_call>{"name": "tool_name", "arguments": {"param": "value"}}</tool_call>\n\n'
            "You will receive the result in:\n"
            "<tool_response>result</tool_response>\n"
        )
        return desc

    def _to_model_device(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Move tensors to the model device and cast float tensors to model dtype."""
        target_dtype = getattr(self.model, "dtype", None)
        out: Dict[str, Any] = {}
        for k, v in inputs.items():
            if hasattr(v, "to"):
                v = v.to(self.model.device)
                if target_dtype is not None and getattr(v, "is_floating_point", None) and v.is_floating_point():
                    v = v.to(target_dtype)
            out[k] = v
        return out

    def _prepare_multimodal_inputs(
        self,
        messages: Messages,
        system_prompt: Optional[str],
        tool_specs: Optional[list[ToolSpec]],
    ) -> Tuple[Dict[str, Any], int]:
        """Tokenize via the processor; returns (model_inputs, input_token_length)."""
        chat, images, videos = self._build_multimodal_chat(
            messages, system_prompt, tool_specs
        )

        prompt = self.processor.apply_chat_template(
            chat, tokenize=False, add_generation_prompt=True
        )

        proc_kwargs: Dict[str, Any] = {"text": prompt, "return_tensors": "pt"}
        if images:
            proc_kwargs["images"] = images
        if videos:
            # Processors expect videos nested per batch sample:
            # [sample][video] where each video is a (T, H, W, C) array.
            proc_kwargs["videos"] = [videos]

        inputs = self.processor(**proc_kwargs)
        inputs = dict(inputs)  # BatchFeature -> plain dict
        inputs = self._to_model_device(inputs)

        input_length = (
            inputs["input_ids"].shape[1] if "input_ids" in inputs else 0
        )
        return inputs, input_length

    # ──────────────────────────────────────────────────────────────────────
    # Text-only formatting (unchanged)
    # ──────────────────────────────────────────────────────────────────────

    def _format_message_content(
        self, content: Union[ContentBlock, Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Format a content block for transformers (legacy helper)."""
        if "text" in content:
            return {"type": "text", "text": content["text"]}
        if "image" in content:
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
        """Format messages for transformers using chat template (text path)."""
        chat_messages = []
        system_content = system_prompt or ""

        if tool_specs:
            system_content += self._tool_specs_to_text(tool_specs)

        if system_content:
            chat_messages.append({"role": "system", "content": system_content})

        for message in messages:
            role = message["role"]
            contents = message["content"]

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
                elif "document" in content:
                    text_parts.append(_document_to_text(content["document"]))

            if text_parts:
                chat_messages.append({"role": role, "content": " ".join(text_parts)})

            if tool_uses:
                for tool_use in tool_uses:
                    tool_text = f"<tool_call>{json.dumps({'name': tool_use['name'], 'arguments': tool_use['input']})}</tool_call>"
                    chat_messages.append({"role": "assistant", "content": tool_text})

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

        if self.is_qwen3 and self.config.get("enable_thinking", True):
            formatted_prompt = self.tokenizer.apply_chat_template(
                chat_messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=True,
            )
        else:
            formatted_prompt = self.tokenizer.apply_chat_template(
                chat_messages,
                tokenize=False,
                add_generation_prompt=True,
            )

        return formatted_prompt

    def _format_chunk(self, event: Dict[str, Any]) -> StreamEvent:
        """Format a generation event into a standardized message chunk."""
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

        Routes through the multimodal processor path when the model is a VLM
        and the conversation carries image/video content; otherwise uses the
        text-only tokenizer path.
        """
        warn_on_tool_choice_not_supported(tool_choice)

        start_time = time.perf_counter()

        # Decide path: multimodal model + any media present anywhere.
        use_mm = self.is_multimodal and self._has_media(messages)

        logger.debug("formatting messages (multimodal=%s)", use_mm)

        if use_mm:
            model_inputs, input_length = self._prepare_multimodal_inputs(
                messages, system_prompt, tool_specs
            )
        else:
            formatted_prompt = self._format_messages(
                messages, system_prompt, tool_specs
            )
            logger.debug(
                "prompt=<%s>",
                (
                    formatted_prompt[:200] + "..."
                    if len(formatted_prompt) > 200
                    else formatted_prompt
                ),
            )
            inputs = self.tokenizer([formatted_prompt], return_tensors="pt")
            inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
            model_inputs = inputs
            input_length = inputs["input_ids"].shape[1]

        # Generation parameters
        params = self.config.get("params", {})
        max_tokens = params.get("max_tokens", 300)
        temperature = params.get("temperature", 1)
        top_p = params.get("top_p", 0.9)
        top_k = params.get("top_k", 20)
        do_sample = params.get("do_sample", True)
        repetition_penalty = params.get("repetition_penalty", 1.0)

        logger.debug("generating with streaming")

        # Streamer uses the tokenizer (present in both paths)
        streamer = TextIteratorStreamer(
            self.tokenizer,
            skip_prompt=True,
            skip_special_tokens=True,
        )

        generation_kwargs = dict(
            max_new_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            do_sample=do_sample,
            repetition_penalty=repetition_penalty,
            pad_token_id=self.tokenizer.eos_token_id,
            streamer=streamer,
        )
        # Feed model inputs (input_ids/attention_mask [+ pixel_values, ...])
        generation_kwargs.update(model_inputs)

        thread = Thread(target=self.model.generate, kwargs=generation_kwargs)
        thread.start()

        yield self._format_chunk({"chunk_type": "message_start"})
        yield self._format_chunk({"chunk_type": "content_start"})

        thinking_mode = False
        tool_call_mode = False
        tool_requested = False
        thinking_buffer = ""
        tool_call_buffer = ""
        output_buffer = ""

        for new_text in streamer:
            if "<tool_call>" in new_text:
                tool_call_mode = True
                tool_requested = True
                parts = new_text.split("<tool_call>", 1)
                if parts[0]:
                    yield self._format_chunk(
                        {"chunk_type": "content_delta", "data": parts[0]}
                    )
                    output_buffer += parts[0]
                yield self._format_chunk({"chunk_type": "content_stop"})
                if len(parts) > 1:
                    tool_call_buffer = parts[1]
                continue

            if tool_call_mode and "</tool_call>" in new_text:
                tool_call_mode = False
                parts = new_text.split("</tool_call>", 1)
                tool_call_buffer += parts[0]
                try:
                    tool_call_data = json.loads(tool_call_buffer)
                    tool_name = tool_call_data.get("name", "unknown")
                    tool_arguments = tool_call_data.get("arguments", {})
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
                        {"chunk_type": "content_stop", "data_type": "tool"}
                    )
                except json.JSONDecodeError as e:
                    logger.warning(
                        f"Failed to parse tool call: {e}, buffer: {tool_call_buffer}"
                    )
                tool_call_buffer = ""
                if len(parts) > 1 and parts[1]:
                    yield self._format_chunk({"chunk_type": "content_start"})
                    yield self._format_chunk(
                        {"chunk_type": "content_delta", "data": parts[1]}
                    )
                    output_buffer += parts[1]
                continue

            if tool_call_mode:
                tool_call_buffer += new_text
                continue

            if self.is_qwen3 and self.config.get("enable_thinking", True):
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

                if thinking_mode and "</think>" in new_text:
                    thinking_mode = False
                    parts = new_text.split("</think>", 1)
                    thinking_buffer += parts[0]
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

    @staticmethod
    def _has_media(messages: Messages) -> bool:
        """Return True if any message carries image/video content (incl. tool results)."""
        for message in messages:
            for content in message.get("content", []):
                if "image" in content or "video" in content:
                    return True
                if "toolResult" in content:
                    for c in content["toolResult"].get("content", []):
                        if "image" in c or "video" in c:
                            return True
        return False

    @override
    async def structured_output(
        self,
        output_model: Type[T],
        prompt: Messages,
        system_prompt: Optional[str] = None,
        **kwargs: Any,
    ) -> AsyncGenerator[Dict[str, Union[T, Any]], None]:
        """Get structured output from the model (prompt-engineered JSON)."""
        schema = output_model.model_json_schema()
        json_instruction = f"\n\nPlease respond with valid JSON matching this schema:\n{json.dumps(schema, indent=2)}"
        augmented_system_prompt = (system_prompt or "") + json_instruction

        response_text = ""
        async for event in self.stream(
            prompt, system_prompt=augmented_system_prompt, **kwargs
        ):
            if "contentBlockDelta" in event:
                delta = event["contentBlockDelta"]["delta"]
                if "text" in delta:
                    response_text += delta["text"]
            yield cast(Dict[str, Union[T, Any]], event)

        try:
            data = json.loads(_extract_json(response_text))
            output_instance = output_model(**data)
            yield {"output": output_instance}
        except Exception as e:
            raise ValueError(
                f"Failed to parse structured output: {e}\nResponse: {response_text}"
            ) from e
