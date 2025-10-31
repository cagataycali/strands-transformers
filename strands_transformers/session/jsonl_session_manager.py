"""JSONL Session Manager - Store conversations in training-ready JSONL format.

Properly captures tool calls and formats conversations for model training.
"""

import json
import logging
import os
from pathlib import Path
from typing import TYPE_CHECKING, Any, Optional, Dict, List

from jinja2 import Template

if TYPE_CHECKING:
    from strands.agent.agent import Agent
    from strands.types.content import Message

from strands.session.session_manager import SessionManager

logger = logging.getLogger(__name__)


class JsonlSessionManager(SessionManager):
    """JSONL session manager - stores complete conversations with tool calls."""

    def __init__(
        self,
        session_id: str,
        template_name: str = "qwen3",
        storage_dir: Optional[str] = None,
        system_prompt: Optional[str] = None,
        **kwargs: Any,
    ):
        """Initialize JSONL session manager."""
        self.session_id = session_id
        self.template_name = template_name
        self.system_prompt = system_prompt

        # Storage directory
        self.storage_dir = storage_dir or os.path.expanduser("~/.strands/training_data")
        os.makedirs(self.storage_dir, exist_ok=True)

        # JSONL file path
        self.jsonl_path = os.path.join(self.storage_dir, f"{session_id}.jsonl")

        # Load template
        self.template = self._load_template()

        # Agent reference (captured during initialize)
        self.agent = None

        # Track last message count to detect new conversations
        self.last_message_count = 0

        logger.info(
            f"JsonlSessionManager: session_id={session_id}, output={self.jsonl_path}"
        )

    def _load_template(self) -> Template:
        """Load Jinja2 template from package templates directory."""
        # Get package directory (strands_transformers/)
        package_dir = Path(__file__).parent.parent
        template_path = package_dir / "templates" / f"{self.template_name}.j2"

        if not template_path.exists():
            raise FileNotFoundError(f"Template not found: {template_path}")

        with open(template_path, "r", encoding="utf-8") as f:
            return Template(f.read())

    def _serialize_tool_spec(self, tool) -> Dict[str, Any]:
        """Convert tool to simple JSON-serializable dict."""
        try:
            tool_spec = tool.to_strands_tool()
            if isinstance(tool_spec, dict) and "toolSpec" in tool_spec:
                spec = tool_spec["toolSpec"]
                # Convert to simple dict
                return {
                    "name": spec.get("name", "unknown"),
                    "description": spec.get("description", ""),
                    "inputSchema": spec.get("inputSchema", {}),
                }
        except Exception as e:
            logger.warning(f"Could not serialize tool: {e}")
        return None

    def _format_conversation(self, agent: "Agent") -> str:
        """Format agent's message history into Qwen3 training format."""
        # Build system prompt with tools
        system_parts = []
        if self.system_prompt:
            system_parts.append(self.system_prompt)

        # Add tool definitions if available
        if hasattr(agent, "tools") and agent.tools:
            system_parts.append("\n# Tools\n")
            system_parts.append(
                "You may call one or more functions to assist with the user query.\n\n"
            )
            system_parts.append(
                "You are provided with function signatures within <tools></tools> XML tags:\n"
            )
            system_parts.append("<tools>\n")

            for tool in agent.tools:
                tool_dict = self._serialize_tool_spec(tool)
                if tool_dict:
                    system_parts.append(
                        json.dumps(
                            {"type": "function", "function": tool_dict},
                            ensure_ascii=False,
                        )
                    )
                    system_parts.append("\n")

            system_parts.append("</tools>\n\n")
            system_parts.append(
                "For each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:\n"
            )
            system_parts.append("<tool_call>\n")
            system_parts.append(
                '{"name": <function-name>, "arguments": <args-json-object>}\n'
            )
            system_parts.append("</tool_call>")

        system_text = "".join(system_parts)

        # Start formatting conversation
        formatted = f"<|im_start|>system\n{system_text}<|im_end|>\n\n"

        # Process messages from agent.messages
        for msg in agent.messages:
            role = msg.get("role") if isinstance(msg, dict) else msg.role
            content = msg.get("content") if isinstance(msg, dict) else msg.content

            if role == "user":
                # User message
                formatted += f"<|im_start|>user\n"

                if isinstance(content, list):
                    for item in content:
                        if isinstance(item, dict):
                            if "text" in item:
                                formatted += item["text"]
                            elif "toolResult" in item:
                                # Tool result
                                tool_result = item["toolResult"]
                                result_content = tool_result.get("content", [])
                                formatted += "\n<tool_response>\n"
                                for result_item in result_content:
                                    if (
                                        isinstance(result_item, dict)
                                        and "text" in result_item
                                    ):
                                        formatted += result_item["text"]
                                formatted += "\n</tool_response>"

                formatted += "<|im_end|>\n"

            elif role == "assistant":
                # Assistant message
                formatted += f"<|im_start|>assistant\n"

                if isinstance(content, list):
                    for item in content:
                        if isinstance(item, dict):
                            if "text" in item:
                                formatted += item["text"]
                            elif "toolUse" in item:
                                # Tool call
                                tool_use = item["toolUse"]
                                tool_call = {
                                    "name": tool_use.get("name"),
                                    "arguments": tool_use.get("input", {}),
                                }
                                formatted += f"\n<tool_call>\n{json.dumps(tool_call, ensure_ascii=False)}\n</tool_call>"

                formatted += "<|im_end|>\n"

        return formatted

    def initialize(self, agent: "Agent", **kwargs: Any) -> None:
        """Initialize with agent reference."""
        self.agent = agent

        # Capture system prompt if not provided
        if not self.system_prompt and hasattr(agent, "system_prompt"):
            self.system_prompt = (
                agent.system_prompt or "You are a helpful AI assistant."
            )

        logger.info(f"Initialized JSONL session for agent {agent.agent_id}")

    def append_message(self, message: "Message", agent: "Agent", **kwargs: Any) -> None:
        """Hook called when message is added - we don't use this, we use sync_agent instead."""
        pass

    def sync_agent(self, agent: "Agent", **kwargs: Any) -> None:
        """Save complete conversation ONLY after final assistant response."""
        # Check if we have new messages
        current_message_count = len(agent.messages)

        if current_message_count <= self.last_message_count:
            return  # No new messages

        # Check if last message is from assistant with actual text content
        if current_message_count > 0:
            last_msg = agent.messages[-1]
            last_role = (
                last_msg.get("role") if isinstance(last_msg, dict) else last_msg.role
            )

            # Only save if last message is from assistant
            if last_role != "assistant":
                return  # Wait for assistant response

            # Check if assistant message has text content (not just tool call)
            last_content = (
                last_msg.get("content")
                if isinstance(last_msg, dict)
                else last_msg.content
            )

            has_text = False
            has_only_tool_use = True

            if isinstance(last_content, list):
                for item in last_content:
                    if isinstance(item, dict):
                        if "text" in item and item["text"].strip():
                            has_text = True
                            has_only_tool_use = False
                        elif "toolUse" not in item:
                            has_only_tool_use = False

            # Skip if no text or only tool calls (wait for final response)
            if not has_text or has_only_tool_use:
                logger.debug(
                    f"Skipping save - waiting for final assistant response (has_text={has_text}, only_tool={has_only_tool_use})"
                )
                return

        # We have complete exchange - format and save
        try:
            formatted_text = self._format_conversation(agent)

            # Create JSONL entry
            jsonl_entry = {"text": formatted_text}

            # Append to file
            with open(self.jsonl_path, "a", encoding="utf-8") as f:
                json.dump(jsonl_entry, f, ensure_ascii=False)
                f.write("\n")

            logger.info(
                f"✅ Saved complete conversation: {len(formatted_text)} chars, {current_message_count} messages"
            )

            # Update last message count
            self.last_message_count = current_message_count

        except Exception as e:
            logger.error(f"Error saving conversation: {e}", exc_info=True)

    def redact_latest_message(
        self, redact_message: "Message", agent: "Agent", **kwargs: Any
    ) -> None:
        """Redaction not supported for training data."""
        pass

    def get_jsonl_path(self) -> str:
        """Get path to JSONL file."""
        return self.jsonl_path

    def get_example_count(self) -> int:
        """Get number of examples saved."""
        if not os.path.exists(self.jsonl_path):
            return 0
        with open(self.jsonl_path, "r", encoding="utf-8") as f:
            return sum(1 for _ in f)

    def __repr__(self) -> str:
        """String representation."""
        return f"JsonlSessionManager(session_id='{self.session_id}', examples={self.get_example_count()})"
