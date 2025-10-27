"""
Universal Transformers Tool - COMPLETE access to transformers library.

Built following the use_aws pattern - no hardcoded operations!
Universal abstraction layer over transformers, sentence-transformers, diffusers, and more.

This tool provides dynamic access to ANY class, function, or method in the transformers
ecosystem without hardcoding specific operations. Just like use_aws wraps boto3 universally,
this wraps the entire transformers library.

Key Features:
1. Universal Access - Call ANY transformers function/class/method dynamically
2. Module Discovery - List available modules, classes, functions, methods
3. Smart Caching - Cache loaded models and tokenizers for performance
4. Dynamic Imports - Load any transformers-related package on demand
5. Parameter Introspection - Generate schemas for functions/methods

Examples:
    # Load a model (returns cached reference)
    use_transformers(
        action="call",
        module="transformers",
        target="AutoModelForCausalLM.from_pretrained",
        parameters={"pretrained_model_name_or_path": "gpt2"},
        cache_key="gpt2_model"
    )

    # Call pipeline function
    use_transformers(
        action="call",
        module="transformers",
        target="pipeline",
        parameters={"task": "text-generation", "model": "gpt2"}
    )

    # Call method on cached model
    use_transformers(
        action="call",
        module="transformers",
        target="cached:gpt2_model.generate",
        parameters={"input_ids": [...], "max_new_tokens": 50}
    )

    # Use sentence-transformers
    use_transformers(
        action="call",
        module="sentence_transformers",
        target="SentenceTransformer",
        parameters={"model_name_or_path": "all-MiniLM-L6-v2"},
        cache_key="encoder"
    )

    # Call method on cached encoder
    use_transformers(
        action="call",
        module="sentence_transformers",
        target="cached:encoder.encode",
        parameters={"sentences": ["Hello", "World"]}
    )

    # Discovery actions
    use_transformers(action="list_modules")
    use_transformers(action="list_classes", module="transformers")
    use_transformers(action="list_functions", module="transformers")
    use_transformers(action="list_methods", module="transformers", target="AutoModel")
    use_transformers(action="inspect", module="transformers", target="pipeline")
"""

import os
import sys
import json
import subprocess
import importlib
import inspect
import logging
from typing import Dict, Any, List, Optional, Union
from pathlib import Path
from strands import tool

# Setup logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

# Global cache for loaded objects (models, tokenizers, etc.)
# Note: Cache persists within the same Python session
# Hot reload will reset the cache
_OBJECT_CACHE = {}

# Known transformers-related modules
KNOWN_MODULES = [
    "transformers",
    "sentence_transformers",
    "diffusers",
    "datasets",
    "tokenizers",
    "accelerate",
    "peft",
]


def _ensure_package(package: str) -> tuple[bool, str]:
    """Ensure package is installed."""
    try:
        importlib.import_module(package)
        return True, "Available"
    except ImportError:
        logger.info(f"📦 Installing {package}...")
        result = subprocess.run(
            [sys.executable, "-m", "pip", "install", package],
            capture_output=True,
            text=True,
            timeout=300,
        )
        if result.returncode == 0:
            return True, "Installed"
        return False, f"Failed: {result.stderr}"


def _get_device() -> str:
    """Auto-detect best available device."""
    try:
        import torch

        if torch.cuda.is_available():
            return "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
    except:
        pass
    return "cpu"


def _import_target(module_name: str, target: str) -> tuple[Any, str]:
    """
    Import and return the target object from module.

    Target can be:
    - "ClassName" - returns class
    - "function_name" - returns function
    - "ClassName.method_name" - returns unbound method (need instance)
    - "ClassName.class_method" - returns class method
    - "cached:key" - returns cached object
    - "cached:key.method" - returns method on cached object

    Returns:
        (target_object, object_type)
    """
    debug_logs = []

    # Handle cached objects
    if target.startswith("cached:"):
        cache_ref = target[7:]  # Remove "cached:" prefix

        if "." in cache_ref:
            # cached:key.method
            cache_key, method_name = cache_ref.split(".", 1)
            if cache_key not in _OBJECT_CACHE:
                raise ValueError(f"No cached object with key: {cache_key}")

            obj = _OBJECT_CACHE[cache_key]
            debug_logs.append(f"Retrieved cached object: {cache_key}")

            # Navigate nested attributes (e.g., "encode.something")
            for attr in method_name.split("."):
                obj = getattr(obj, attr)
                debug_logs.append(f"  → {attr}")

            return obj, "cached_method", debug_logs
        else:
            # cached:key
            if cache_ref not in _OBJECT_CACHE:
                raise ValueError(f"No cached object with key: {cache_ref}")
            return _OBJECT_CACHE[cache_ref], "cached_object", debug_logs

    # Import module
    try:
        mod = importlib.import_module(module_name)
        debug_logs.append(f"Imported module: {module_name}")
    except ImportError as e:
        raise ImportError(f"Failed to import {module_name}: {e}")

    # Handle nested target (e.g., "AutoModel.from_pretrained")
    parts = target.split(".")
    obj = mod

    for i, part in enumerate(parts):
        obj = getattr(obj, part)
        debug_logs.append(f"  → {part}")

        # Determine type at final step
        if i == len(parts) - 1:
            if inspect.isclass(obj):
                return obj, "class", debug_logs
            elif inspect.isfunction(obj) or inspect.ismethod(obj):
                return obj, "function", debug_logs
            elif callable(obj):
                return obj, "callable", debug_logs
            else:
                return obj, "object", debug_logs

    return obj, "unknown", debug_logs


def _list_classes(module_name: str) -> List[str]:
    """List all classes in a module."""
    mod = importlib.import_module(module_name)
    return [name for name, obj in inspect.getmembers(mod, inspect.isclass)]


def _list_functions(module_name: str) -> List[str]:
    """List all functions in a module."""
    mod = importlib.import_module(module_name)
    return [name for name, obj in inspect.getmembers(mod, inspect.isfunction)]


def _list_methods(module_name: str, class_name: str) -> List[str]:
    """List all methods in a class."""
    mod = importlib.import_module(module_name)
    cls = getattr(mod, class_name)
    return [
        name
        for name, obj in inspect.getmembers(cls, inspect.ismethod)
        if not name.startswith("_")
    ]


def _inspect_signature(module_name: str, target: str) -> Dict[str, Any]:
    """Get function/method signature and docstring."""
    obj, obj_type, _ = _import_target(module_name, target)

    try:
        sig = inspect.signature(obj)
        params = {}
        for name, param in sig.parameters.items():
            params[name] = {
                "type": (
                    str(param.annotation)
                    if param.annotation != inspect.Parameter.empty
                    else "Any"
                ),
                "default": (
                    str(param.default)
                    if param.default != inspect.Parameter.empty
                    else "required"
                ),
                "kind": str(param.kind),
            }

        return {
            "signature": str(sig),
            "parameters": params,
            "docstring": inspect.getdoc(obj) or "No documentation available",
            "type": obj_type,
        }
    except Exception as e:
        return {
            "error": str(e),
            "docstring": inspect.getdoc(obj) or "No documentation available",
            "type": obj_type,
        }


@tool
def use_transformers(
    action: str,
    module: str = None,
    target: str = None,
    parameters: Dict[str, Any] = None,
    cache_key: str = None,
    device: str = None,
) -> Dict[str, Any]:
    """
    Universal access to ALL transformers library functionality - no hardcoding!

    Like use_aws for boto3, this provides dynamic access to any transformers function/class/method.

    Args:
        action: Action to perform
            - call: Call any function/class/method dynamically
            - list_modules: Show available transformers modules
            - list_classes: List classes in a module
            - list_functions: List functions in a module
            - list_methods: List methods in a class
            - inspect: Get signature and docs for function/method
            - list_cache: Show cached objects
            - clear_cache: Clear cache or specific key

        module: Module name (e.g., "transformers", "sentence_transformers", "diffusers")

        target: What to call - flexible syntax:
            - "function_name" - Call module function
            - "ClassName" - Instantiate class
            - "ClassName.method" - Call class method
            - "cached:key" - Get cached object
            - "cached:key.method" - Call method on cached object

        parameters: Dict of parameters to pass to the function/method/class

        cache_key: Optional key to cache the result (for models, tokenizers, etc.)

        device: Device to use (cpu, cuda, mps) - auto-detected if not specified

    Returns:
        Dict with status and results including debug logs

    Examples:
        # List available modules
        use_transformers(action="list_modules")

        # List classes in transformers
        use_transformers(action="list_classes", module="transformers")

        # Inspect a function
        use_transformers(action="inspect", module="transformers", target="pipeline")

        # Call pipeline function
        use_transformers(
            action="call",
            module="transformers",
            target="pipeline",
            parameters={"task": "text-generation", "model": "gpt2"}
        )

        # Load and cache a model
        use_transformers(
            action="call",
            module="transformers",
            target="AutoModelForCausalLM.from_pretrained",
            parameters={"pretrained_model_name_or_path": "gpt2"},
            cache_key="my_model"
        )

        # Call method on cached model
        use_transformers(
            action="call",
            module="transformers",
            target="cached:my_model.generate",
            parameters={"input_ids": [[1, 2, 3]], "max_new_tokens": 20}
        )
    """

    debug_logs = []

    try:
        # List available modules
        if action == "list_modules":
            available = []
            for mod_name in KNOWN_MODULES:
                ok, status = _ensure_package(mod_name)
                available.append(f"{'✅' if ok else '❌'} {mod_name}: {status}")

            return {
                "status": "success",
                "content": [
                    {
                        "text": (
                            "📦 **Available Transformers Modules**\n\n"
                            + "\n".join(available)
                            + "\n\n**Usage:** Specify any of these in the 'module' parameter"
                        )
                    }
                ],
            }

        # List classes in module
        elif action == "list_classes":
            if not module:
                return {
                    "status": "error",
                    "content": [{"text": "❌ Provide 'module' parameter"}],
                }

            ok, status = _ensure_package(module)
            if not ok:
                return {"status": "error", "content": [{"text": f"❌ {status}"}]}

            classes = _list_classes(module)

            return {
                "status": "success",
                "content": [
                    {
                        "text": (
                            f"📋 **Classes in {module}** ({len(classes)} total)\n\n"
                            + "\n".join([f"  • {c}" for c in sorted(classes)[:50]])
                            + (
                                f"\n\n... and {len(classes) - 50} more"
                                if len(classes) > 50
                                else ""
                            )
                        )
                    }
                ],
            }

        # List functions in module
        elif action == "list_functions":
            if not module:
                return {
                    "status": "error",
                    "content": [{"text": "❌ Provide 'module' parameter"}],
                }

            ok, status = _ensure_package(module)
            if not ok:
                return {"status": "error", "content": [{"text": f"❌ {status}"}]}

            functions = _list_functions(module)

            return {
                "status": "success",
                "content": [
                    {
                        "text": (
                            f"🔧 **Functions in {module}** ({len(functions)} total)\n\n"
                            + "\n".join([f"  • {f}" for f in sorted(functions)[:50]])
                            + (
                                f"\n\n... and {len(functions) - 50} more"
                                if len(functions) > 50
                                else ""
                            )
                        )
                    }
                ],
            }

        # List methods in class
        elif action == "list_methods":
            if not module or not target:
                return {
                    "status": "error",
                    "content": [
                        {"text": "❌ Provide 'module' and 'target' (class name)"}
                    ],
                }

            ok, status = _ensure_package(module)
            if not ok:
                return {"status": "error", "content": [{"text": f"❌ {status}"}]}

            methods = _list_methods(module, target)

            return {
                "status": "success",
                "content": [
                    {
                        "text": (
                            f"⚙️ **Methods in {module}.{target}** ({len(methods)} total)\n\n"
                            + "\n".join([f"  • {m}" for m in sorted(methods)[:50]])
                            + (
                                f"\n\n... and {len(methods) - 50} more"
                                if len(methods) > 50
                                else ""
                            )
                        )
                    }
                ],
            }

        # Inspect function/method signature
        elif action == "inspect":
            if not module or not target:
                return {
                    "status": "error",
                    "content": [{"text": "❌ Provide 'module' and 'target'"}],
                }

            ok, status = _ensure_package(module)
            if not ok:
                return {"status": "error", "content": [{"text": f"❌ {status}"}]}

            info = _inspect_signature(module, target)

            params_text = "\n".join(
                [
                    f"  • **{name}**: {details['type']} (default: {details['default']})"
                    for name, details in info.get("parameters", {}).items()
                ]
            )

            return {
                "status": "success",
                "content": [
                    {
                        "text": (
                            f"🔍 **{module}.{target}**\n\n"
                            f"**Type:** {info.get('type', 'unknown')}\n\n"
                            f"**Signature:**\n```python\n{info.get('signature', 'N/A')}\n```\n\n"
                            f"**Parameters:**\n{params_text or 'None'}\n\n"
                            f"**Documentation:**\n{info.get('docstring', 'No docs available')[:500]}"
                        )
                    }
                ],
            }

        # List cached objects
        elif action == "list_cache":
            if not _OBJECT_CACHE:
                return {
                    "status": "success",
                    "content": [{"text": "📦 **Cache is empty**"}],
                }

            cache_info = []
            for key, obj in _OBJECT_CACHE.items():
                obj_type = type(obj).__name__
                obj_module = type(obj).__module__
                cache_info.append(f"  • **{key}**: {obj_module}.{obj_type}")

            return {
                "status": "success",
                "content": [
                    {
                        "text": (
                            f"📦 **Cached Objects** ({len(_OBJECT_CACHE)} total)\n\n"
                            + "\n".join(cache_info)
                            + "\n\n**Usage:** Use `cached:key` or `cached:key.method` to access"
                        )
                    }
                ],
            }

        # Clear cache
        elif action == "clear_cache":
            if cache_key:
                if cache_key in _OBJECT_CACHE:
                    del _OBJECT_CACHE[cache_key]
                    return {
                        "status": "success",
                        "content": [{"text": f"✅ Cleared cache key: {cache_key}"}],
                    }
                else:
                    return {
                        "status": "error",
                        "content": [{"text": f"❌ No cache key: {cache_key}"}],
                    }
            else:
                count = len(_OBJECT_CACHE)
                _OBJECT_CACHE.clear()
                return {
                    "status": "success",
                    "content": [{"text": f"✅ Cleared all cache ({count} objects)"}],
                }

        # Call function/method/class
        elif action == "call":
            if not module or not target:
                return {
                    "status": "error",
                    "content": [{"text": "❌ Provide 'module' and 'target'"}],
                }

            if parameters is None:
                parameters = {}

            debug_logs.append(f"🎯 Call: {module}.{target}")
            debug_logs.append(f"📊 Parameters: {list(parameters.keys())}")

            # Ensure module is available
            ok, status = _ensure_package(module)
            if not ok:
                return {"status": "error", "content": [{"text": f"❌ {status}"}]}

            # Import and get target
            obj, obj_type, import_logs = _import_target(module, target)
            debug_logs.extend(import_logs)
            debug_logs.append(f"✅ Target type: {obj_type}")

            # Add device to parameters if needed and not specified
            if device or (
                "device" not in parameters and obj_type in ["class", "function"]
            ):
                auto_device = device or _get_device()
                # Only add device if the target accepts it
                try:
                    sig = inspect.signature(obj)
                    if "device" in sig.parameters and "device" not in parameters:
                        parameters["device"] = auto_device
                        debug_logs.append(f"🖥️ Auto device: {auto_device}")
                except:
                    pass

            # Call the target
            debug_logs.append("⚙️ Executing...")
            result = obj(**parameters)
            debug_logs.append("✅ Execution complete")

            # Cache if requested
            if cache_key:
                _OBJECT_CACHE[cache_key] = result
                debug_logs.append(f"💾 Cached as: {cache_key}")

            # Format result
            result_type = type(result).__name__
            result_module = type(result).__module__

            # Try to get useful info about result
            result_info = f"**Type:** {result_module}.{result_type}\n"

            if hasattr(result, "__len__") and not isinstance(result, str):
                try:
                    result_info += f"**Length:** {len(result)}\n"
                except:
                    pass

            if hasattr(result, "shape"):
                result_info += f"**Shape:** {result.shape}\n"

            if isinstance(result, (list, tuple)) and len(result) > 0:
                result_info += f"**First item type:** {type(result[0]).__name__}\n"

            if isinstance(result, dict):
                result_info += f"**Keys:** {', '.join(list(result.keys())[:10])}\n"

            # Format result value
            if isinstance(result, (str, int, float, bool)):
                result_value = f"```\n{result}\n```"
            elif isinstance(result, (list, dict)) and len(str(result)) < 500:
                result_value = (
                    f"```json\n{json.dumps(result, indent=2, default=str)}\n```"
                )
            else:
                result_value = f"```\n{str(result)[:500]}{'...' if len(str(result)) > 500 else ''}\n```"

            return {
                "status": "success",
                "content": [
                    {
                        "text": (
                            f"✅ **Call Result**\n\n"
                            f"**Debug:**\n```\n" + "\n".join(debug_logs) + "\n```\n\n"
                            f"{result_info}\n"
                            f"**Value:**\n{result_value}"
                        )
                    }
                ],
            }

        else:
            return {
                "status": "error",
                "content": [
                    {
                        "text": (
                            f"❌ Unknown action: {action}\n\n"
                            f"**Available actions:**\n"
                            f"  • **call** - Call any function/class/method\n"
                            f"  • **list_modules** - Show available modules\n"
                            f"  • **list_classes** - List classes in module\n"
                            f"  • **list_functions** - List functions in module\n"
                            f"  • **list_methods** - List methods in class\n"
                            f"  • **inspect** - Get signature and docs\n"
                            f"  • **list_cache** - Show cached objects\n"
                            f"  • **clear_cache** - Clear cache"
                        )
                    }
                ],
            }

    except Exception as e:
        import traceback

        error_trace = traceback.format_exc()

        return {
            "status": "error",
            "content": [
                {
                    "text": (
                        f"❌ **Error: {str(e)}**\n\n"
                        f"**Debug:**\n```\n" + "\n".join(debug_logs) + "\n```\n\n"
                        f"**Traceback:**\n```\n{error_trace}\n```"
                    )
                }
            ],
        }


__all__ = ["use_transformers"]
