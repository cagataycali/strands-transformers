"""Strands Transformers - Local HuggingFace models + fine-tuning infrastructure."""

__version__ = "0.1.0"

from strands_transformers.models.transformers import TransformerModel
from strands_transformers.session.jsonl_session_manager import JsonlSessionManager


# Lazy imports for tools (require optional dependencies)
def __getattr__(name):
    """Lazy load tools to avoid import errors if dependencies not installed."""
    if name == "dataset_generator":
        from strands_transformers.tools.dataset_generator import dataset_generator

        return dataset_generator
    elif name == "model_trainer":
        from strands_transformers.tools.model_trainer import model_trainer

        return model_trainer
    elif name == "template":
        from strands_transformers.tools.template import template

        return template
    elif name == "use_transformers":
        from strands_transformers.tools.use_transformers import use_transformers

        return use_transformers
    raise AttributeError(f"module 'strands_transformers' has no attribute '{name}'")


__all__ = [
    "TransformerModel",
    "JsonlSessionManager",
    "dataset_generator",
    "model_trainer",
    "template",
    "use_transformers",
]
