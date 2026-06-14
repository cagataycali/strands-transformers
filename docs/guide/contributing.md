# Contributing

New transformers task? It already works — `use_transformers` reads the taxonomy
at runtime. Found a model that needs special handling, or want a new example?
PRs welcome.

## Ground rules

1. **Real, runnable examples.** Add to `examples/` and drive actual inference —
   no mocks. Show the verified output.
2. **Keep `smoke.py` green.** It's the fast E2E gate:
   ```bash
   PYTHONPATH=. python examples/smoke.py     # → "16/16 checks passed"
   ```
3. **Verify claims.** Any number/output in docs must come from a real run.

## Dev setup

=== "uv"
    ```bash
    uv pip install -e ".[all]"
    ```
=== "pip"
    ```bash
    pip install -e ".[all]"
    ```

## Building the docs

```bash
uv pip install -e ".[docs]"
mkdocs serve        # live preview at http://127.0.0.1:8000
mkdocs build        # static site → site/
```
