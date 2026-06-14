"""Native multimodal I/O — take images/video/audio/text/arrays in, get them out.

Inputs: file paths, URLs, base64 data URIs, raw bytes, numpy arrays, PIL images.
Outputs: any pipeline result serialized to JSON-safe form, with binary artifacts
(audio waveforms, generated images) written to disk and referenced by path.

The goal: an agent can pass "cat.jpg" or "https://.../clip.mp4" or a numpy robot
state array, and receive back text + paths to generated media, all natively.
"""

from __future__ import annotations

import base64
import io
import os
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

ARTIFACT_DIR = Path(os.getenv("STRANDS_TRANSFORMERS_ARTIFACTS", tempfile.gettempdir())) / "strands_transformers"
ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)


# ───────────────────────── INPUT COERCION ─────────────────────────

def coerce_input(value: Any) -> Any:
    """Best-effort coercion of an input spec into something a pipeline accepts.

    Pipelines already accept paths/URLs/PIL/arrays natively, so mostly we just
    decode base64 data URIs and pass everything else through untouched.
    """
    if isinstance(value, str):
        if value.startswith("data:"):
            return _decode_data_uri(value)
        return value  # path / URL / text — pipelines handle these
    if isinstance(value, list):
        return [coerce_input(v) for v in value]
    if isinstance(value, dict):
        return {k: coerce_input(v) for k, v in value.items()}
    return value


def _decode_data_uri(uri: str) -> Any:
    """Decode a data: URI into a PIL Image / bytes depending on mime."""
    header, _, b64 = uri.partition(",")
    raw = base64.b64decode(b64)
    mime = header.split(";")[0].removeprefix("data:")
    if mime.startswith("image/"):
        from PIL import Image
        return Image.open(io.BytesIO(raw)).convert("RGB")
    return raw


def load_array(spec: Any):
    """Load a numpy array from list, .npy path, or pass through ndarray.

    Useful for robot state / time-series inputs.
    """
    import numpy as np

    if isinstance(spec, np.ndarray):
        return spec
    if isinstance(spec, (list, tuple)):
        return np.asarray(spec)
    if isinstance(spec, str) and spec.endswith(".npy"):
        return np.load(spec)
    raise ValueError(f"Cannot load array from {type(spec).__name__}")


# ───────────────────────── OUTPUT SERIALIZATION ─────────────────────────

def serialize_output(result: Any, task: str = "", save_artifacts: bool = True) -> Dict[str, Any]:
    """Convert any pipeline/model output into a JSON-safe dict.

    Binary artifacts (audio, images) are written to ARTIFACT_DIR and referenced
    by path so the agent can hand them to the user or downstream tools.
    """
    artifacts: List[str] = []
    payload = _serialize(result, artifacts, save_artifacts)
    out: Dict[str, Any] = {"result": payload}
    if artifacts:
        out["artifacts"] = artifacts
    return out


def _serialize(obj: Any, artifacts: List[str], save: bool, depth: int = 0) -> Any:
    if depth > 6:
        return str(obj)[:200]

    # Primitives
    if obj is None or isinstance(obj, (str, int, float, bool)):
        return obj

    # Audio dict from TTS pipelines: {"audio": ndarray, "sampling_rate": int}
    if isinstance(obj, dict) and "audio" in obj and "sampling_rate" in obj:
        if save:
            path = _save_audio(obj["audio"], obj["sampling_rate"])
            artifacts.append(path)
            return {"type": "audio", "path": path, "sampling_rate": obj["sampling_rate"]}
        return {"type": "audio", "sampling_rate": obj["sampling_rate"]}

    # PIL image
    try:
        from PIL import Image
        if isinstance(obj, Image.Image):
            if save:
                path = _save_image(obj)
                artifacts.append(path)
                return {"type": "image", "path": path, "size": list(obj.size)}
            return {"type": "image", "size": list(obj.size)}
    except ImportError:
        pass

    # numpy / torch tensors
    arr = _maybe_array(obj)
    if arr is not None:
        return arr

    # Mappings & sequences
    if isinstance(obj, dict):
        return {str(k): _serialize(v, artifacts, save, depth + 1) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_serialize(v, artifacts, save, depth + 1) for v in obj[:200]]

    # Objects with __dict__ (e.g. ModelOutput) → try dict-like
    if hasattr(obj, "to_dict"):
        try:
            return _serialize(obj.to_dict(), artifacts, save, depth + 1)
        except Exception:
            pass
    if hasattr(obj, "keys"):
        try:
            return {str(k): _serialize(obj[k], artifacts, save, depth + 1) for k in obj.keys()}
        except Exception:
            pass

    return str(obj)[:50000]


def _maybe_array(obj: Any) -> Optional[Any]:
    """Convert numpy/torch arrays to nested lists, summarizing huge ones."""
    # torch tensor
    try:
        import torch
        if isinstance(obj, torch.Tensor):
            obj = obj.detach().cpu().numpy()
    except ImportError:
        pass
    try:
        import numpy as np
        if isinstance(obj, (np.ndarray, np.generic)):
            a = np.asarray(obj)
            if a.size <= 256:
                return a.tolist()
            return {
                "type": "ndarray",
                "shape": list(a.shape),
                "dtype": str(a.dtype),
                "preview": a.flatten()[:16].tolist(),
            }
    except ImportError:
        pass
    return None


def _save_audio(audio, sampling_rate: int) -> str:
    import numpy as np

    a = np.asarray(audio)
    if a.ndim > 1:
        a = a.squeeze()
    path = ARTIFACT_DIR / f"audio_{int(time.time()*1000)}.wav"
    try:
        import soundfile as sf
        sf.write(str(path), a, int(sampling_rate))
    except ImportError:
        _write_wav_stdlib(str(path), a, int(sampling_rate))
    return str(path)


def _write_wav_stdlib(path: str, audio, sampling_rate: int) -> None:
    """Write a WAV without soundfile, using the stdlib wave module."""
    import wave
    import numpy as np

    a = np.asarray(audio, dtype=np.float32)
    a = np.clip(a, -1.0, 1.0)
    pcm = (a * 32767).astype(np.int16)
    with wave.open(path, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sampling_rate)
        w.writeframes(pcm.tobytes())


def _save_image(image) -> str:
    path = ARTIFACT_DIR / f"image_{int(time.time()*1000)}.png"
    image.save(str(path))
    return str(path)
