"""Strands Transformers — the universal entrypoint to HuggingFace transformers.

100% transformers coverage with zero hardcoding: every task across every modality
(text, image, video, audio, robot-state) in and out, the same way `use_aws` wraps
boto3 and `use_lerobot` wraps lerobot.

Quick start:
    from strands import Agent
    from strands_transformers import use_transformers

    agent = Agent(tools=[use_transformers])
    agent("Transcribe recording.wav")            # ASR
    agent("Describe scene.jpg and plan a grasp") # image-text-to-text / VLA
    agent("Say 'hello' as audio")                # text-to-audio

    # Or use a local HF model as the agent's brain:
    from strands_transformers import TransformerModel
    model = TransformerModel(model_path="Qwen/Qwen3-1.7B")
    agent = Agent(model=model, tools=[use_transformers])
"""

__version__ = "0.2.0"

from strands_transformers.core import engine, io, registry
from strands_transformers.tools.use_transformers import use_transformers


def __getattr__(name):
    # Lazy import the model provider (pulls in torch) only when requested.
    if name == "TransformerModel":
        from strands_transformers.models.transformers import TransformerModel
        return TransformerModel
    raise AttributeError(f"module 'strands_transformers' has no attribute '{name}'")


__all__ = [
    "use_transformers",
    "TransformerModel",
    "registry",
    "engine",
    "io",
]
