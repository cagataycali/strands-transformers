"""Fast end-to-end smoke test — discovery + tiny pipelines, no large downloads.

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
    results.append(
        check("task_info", use_transformers(action="task_info", task="text-generation"))
    )
    results.append(
        check("inspect pipeline", use_transformers(action="inspect", target="pipeline"))
    )
    results.append(check("compat", use_transformers(action="compat")))
    results.append(
        check(
            "unknown action errors",
            use_transformers(action="does-not-exist"),
            predicate=lambda r: r["status"] == "error",
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
    results.append(check("run text-to-audio (artifact)", tts,
                         predicate=lambda r: r["status"] == "success" and bool(r.get("artifacts"))))
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

    passed = sum(results)
    total = len(results)
    print(f"\n{passed}/{total} checks passed")
    return 0 if passed == total else 1


if __name__ == "__main__":
    sys.exit(run())
