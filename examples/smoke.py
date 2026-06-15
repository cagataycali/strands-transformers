"""Fast end-to-end smoke test - discovery + tiny pipelines, no large downloads.

A quick health check that exercises use_transformers against the real
implementation without pulling multi-GB models. Returns a non-zero exit code if
anything fails, so it doubles as a CI-free sanity gate:

    PYTHONPATH=. python examples/smoke.py
"""

import sys

from strands_transformers import use_transformers


def check(name, result, predicate=lambda r: r["status"] == "success"):
    ok = predicate(result)
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}")
    if not ok:
        print("        ->", result["content"][0]["text"][:200])
    return ok


def run() -> int:
    results = []

    # ── discovery (no downloads) ──
    results.append(check("tasks", use_transformers(action="tasks")))
    results.append(check("modalities", use_transformers(action="modalities")))
    results.append(check("classes", use_transformers(action="classes")))
    results.append(check("task_info", use_transformers(action="task_info", task="text-generation")))
    results.append(check("inspect pipeline", use_transformers(action="inspect", target="pipeline")))
    results.append(check("compat", use_transformers(action="compat")))
    results.append(
        check(
            "unknown action errors",
            use_transformers(action="does-not-exist"),
            predicate=lambda r: r["status"] == "error",
        )
    )

    # ── regression guards (no downloads) ──
    from strands_transformers.core import io, registry

    results.append(
        check(
            "task alias resolves (sentiment-analysis → text-classification)",
            {
                "status": "success"
                if registry.resolve_task("sentiment-analysis") == "text-classification"
                else "error",
                "content": [{"text": "alias resolution"}],
            },
        )
    )

    import base64

    _b = io.serialize_output(b"\x00\x01hi", save_artifacts=False)["result"]
    results.append(
        check(
            "bytes serialize as recoverable base64",
            {
                "status": "success"
                if (
                    isinstance(_b, dict)
                    and _b.get("encoding") == "base64"
                    and base64.b64decode(_b["data"]) == b"\x00\x01hi"
                )
                else "error",
                "content": [{"text": "bytes base64 roundtrip"}],
            },
        )
    )

    import os
    import tempfile
    import wave

    import numpy as _np

    _p = tempfile.mktemp(suffix=".wav")
    with wave.open(_p, "wb") as _w:
        _w.setnchannels(1)
        _w.setsampwidth(1)
        _w.setframerate(16000)
        _w.writeframes(_np.full(800, 128, dtype=_np.uint8).tobytes())  # 8-bit silence
    _dec = io.decode_wav(_p)
    os.remove(_p)
    results.append(
        check(
            "8-bit WAV silence decodes to ~0.0 (unsigned PCM)",
            {
                "status": "success"
                if (_dec is not None and abs(float(_dec[0][0])) < 0.01)
                else "error",
                "content": [{"text": "8-bit wav decode"}],
            },
        )
    )

    # ── tiny pipelines (small downloads, fast) ──
    results.append(
        check(
            "run text-generation",
            use_transformers(
                action="run",
                task="text-generation",
                model="sshleifer/tiny-gpt2",
                inputs="hello",
                parameters={"max_new_tokens": 4},
            ),
        )
    )
    results.append(
        check(
            "run text-classification",
            use_transformers(
                action="run",
                task="text-classification",
                model="hf-internal-testing/tiny-random-distilbert",
                inputs="good",
            ),
        )
    )

    # ── image task (structured output serialization) ──
    import numpy as np
    from PIL import Image

    img = Image.fromarray(np.random.default_rng(0).integers(0, 255, (64, 64, 3), dtype=np.uint8))
    results.append(
        check(
            "run image-classification",
            use_transformers(
                action="run",
                task="image-classification",
                model="hf-internal-testing/tiny-random-vit",
                inputs=img,
            ),
        )
    )

    # ── audio round-trip (TTS → wav → ASR), exercises io + torchcodec compat ──
    try:
        import torch

        torch.manual_seed(0)  # tiny-random VITS is numerically unstable on some seeds
    except ImportError:
        pass
    tts = use_transformers(
        action="run",
        task="text-to-audio",
        model="hf-internal-testing/tiny-random-VitsModel",
        inputs="smoke test",
    )
    results.append(
        check(
            "run text-to-audio (artifact)",
            tts,
            predicate=lambda r: r["status"] == "success" and bool(r.get("artifacts")),
        )
    )
    wav = tts.get("artifacts", [None])[0]
    if wav:
        results.append(
            check(
                "run automatic-speech-recognition (wav path)",
                use_transformers(
                    action="run",
                    task="automatic-speech-recognition",
                    model="hf-internal-testing/tiny-random-wav2vec2",
                    inputs=wav,
                ),
            )
        )

    # ── registry invariant: every task is bucketed in exactly one modality ──
    _mods = registry.tasks_by_modality()
    _all_tasks = set(registry.supported_tasks())
    _bucketed = set().union(*_mods.values()) if _mods else set()
    results.append(
        check(
            "every task bucketed into a modality",
            {
                "status": "success" if _bucketed == _all_tasks else "error",
                "content": [
                    {"text": f"missing={_all_tasks - _bucketed} phantom={_bucketed - _all_tasks}"}
                ],
            },
        )
    )

    # ── zero-shot image classification (CLIP, cached) ──
    import numpy as _np
    from PIL import Image as _Image

    _z = use_transformers(
        action="run",
        task="zero-shot-image-classification",
        model="openai/clip-vit-base-patch32",
        inputs=_Image.fromarray(_np.zeros((64, 64, 3), dtype=_np.uint8)),
        parameters={"candidate_labels": ["a cat", "random noise"]},
    )
    results.append(check("run zero-shot-image-classification", _z))

    # ── cache behavior guard ──
    from strands_transformers.core import engine as _eng

    _key = "pipe::text-classification::hf-internal-testing/tiny-random-distilbert"
    use_transformers(
        action="run",
        task="text-classification",
        model="hf-internal-testing/tiny-random-distilbert",
        inputs="x",
    )
    results.append(
        check(
            "pipeline is cached after run",
            {
                "status": "success" if _key in _eng._CACHE else "error",
                "content": [{"text": "pipeline cache"}],
            },
        )
    )

    passed = sum(results)
    total = len(results)
    print(f"\n{passed}/{total} checks passed")
    return 0 if passed == total else 1


if __name__ == "__main__":
    sys.exit(run())
