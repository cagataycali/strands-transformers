# Architecture

The source of truth is transformers' own `SUPPORTED_TASKS` registry, read at
runtime — nothing is hardcoded per-task. A new transformers task is instantly
available.

```mermaid
flowchart TB
    AG["🤖 Strands Agent"]
    AG -->|tool| TOOL["🛠️ tools/use_transformers.py<br/>discover · run · call"]
    AG -->|model| PROV["🧠 models/transformers.py<br/>TransformerModel"]
    PROV --> TYPES["🔊 types/audio.py<br/>audio content block"]
    TOOL --> REG
    PROV --> ENG
    subgraph CORE["core/"]
        REG["📚 registry.py<br/>task taxonomy → modality → AutoModel"]
        ENG["⚙️ engine.py<br/>load/cache · device · dtype"]
        IO["🔁 io.py<br/>coerce in · serialize out · save media"]
        CMP["🩹 compat.py<br/>4.x → 5.x shims"]
    end
    REG --> ENG --> IO
    ENG --> CMP
    REG -->|reads| HF["🤗 transformers SUPPORTED_TASKS"]

    classDef a fill:#7C4DFF,stroke:#5b34d6,color:#fff;
    classDef t fill:#FFD21E,stroke:#E68A00,color:#3a2d00;
    classDef c fill:#00E5FF,stroke:#00b3cc,color:#003844;
    class AG a;
    class TOOL,PROV,TYPES t;
    class REG,ENG,IO,CMP,HF c;
```

| Layer | File | Responsibility |
|-------|------|----------------|
| **Registry** | `core/registry.py` | Reads transformers' task taxonomy → modality → AutoModel. Dynamic class/fn resolution. |
| **Engine** | `core/engine.py` | Loads & caches pipelines/models. Auto device (cuda/mps/cpu) + dtype. |
| **I/O** | `core/io.py` | Coerces multimodal inputs; serializes outputs; saves audio/images to disk. |
| **Provider** | `models/transformers.py` | `TransformerModel` — local model as the agent brain, multimodal content blocks. |
| **Types** | `types/audio.py` | Audio content-block extension to the Strands taxonomy. |
| **Tool** | `tools/use_transformers.py` | The single `@tool` agents call. Discovery + run + call. |

## Request flow (the brain)

1. `Agent` passes content blocks to `TransformerModel.stream(...)`.
2. The provider inspects the blocks and the model's processor to pick a path:
   text tokenizer · vision `AutoProcessor` · audio `feature_extractor` ·
   Omni Thinker+Talker.
3. Inputs are coerced (image bytes → PIL, video → frames + `VideoMetadata`,
   audio → resampled waveform), the model generates, and tokens stream back as
   standard Strands events (text, reasoning, tool calls). Omni additionally
   stashes a speech waveform retrievable via `get_last_audio()`.
