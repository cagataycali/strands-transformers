# Legacy model compatibility

Many `trust_remote_code` models were written for transformers 4.x and break on
5.x. The built-in `core/compat.py` shims patch the gaps automatically so the
model's own code runs unchanged — no version pinning.

| Breakage | Shim |
|----------|------|
| Moved tokenizer symbols (`PaddingStrategy`, …) | re-exposed on `transformers.tokenization_utils` |
| Removed `AutoModelForVision2Seq` | recreated as an alias of `AutoModelForImageTextToText`, registered for `auto_map` dispatch |
| `tie_weights()` signature drift | legacy overrides made kwarg-tolerant |
| Hard `timm` version pins | `compat.spoof_timm_version()` context manager |
| Broken `torchcodec` (audio decode) | disabled so stdlib/soundfile paths win |

`compat.apply()` is invoked automatically by the tool and provider. Trigger it
explicitly if you need to:

```python
use_transformers(action="compat", parameters={"timm_version": "0.9.16"})
```

These shims are generic — they help any 4.x-era custom-code model, not just
OpenVLA.
