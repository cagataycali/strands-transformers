# Architecture

The source of truth is transformers' own `SUPPORTED_TASKS` registry, read at
runtime - nothing is hardcoded per-task. A new transformers task is instantly
available.

```mermaid
flowchart TB
    AG(["🤖 Strands Agent"])

    subgraph ENTRY [" "]
        direction LR
        TOOL["🛠️ use_transformers<br/><b>the tool</b><br/><span>discover · run · call</span>"]
        PROV["🧠 TransformerModel<br/><b>the brain</b><br/><span>multimodal content blocks</span>"]
    end

    subgraph CORE ["⚙️ core/"]
        direction LR
        REG["📚 registry<br/><span>tasks → modality → AutoModel</span>"]
        ENG["⚙️ engine<br/><span>load · cache · device · dtype</span>"]
        IO["🔁 io<br/><span>coerce in · serialize out</span>"]
        CMP["🩹 compat<br/><span>4.x → 5.x shims</span>"]
    end

    HF[("🤗 transformers<br/>SUPPORTED_TASKS")]

    AG -->|"as a tool"| TOOL
    AG -->|"as model="| PROV
    TOOL --> REG
    PROV --> ENG
    REG -.->|"reads taxonomy"| HF
    REG --> ENG
    ENG --> IO
    ENG --> CMP

    classDef agent fill:#7C5CFF26,stroke:#7C5CFF,stroke-width:1.5px,color:#7C5CFF;
    classDef entry fill:#22D3EE1f,stroke:#22D3EE,stroke-width:1.5px,color:#0F91A6;
    classDef core fill:#8B8B9414,stroke:#8B8B9466,stroke-width:1px,color:#6B6B76;
    classDef src fill:#FFB02E1a,stroke:#FFB02E,stroke-width:1.5px,color:#B5760A;
    class AG agent;
    class TOOL,PROV entry;
    class REG,ENG,IO,CMP core;
    class HF src;
    style ENTRY fill:none,stroke:none;
    style CORE fill:#8B8B9408,stroke:#8B8B9433,stroke-width:1px;
```

| Layer | File | Responsibility |
|-------|------|----------------|
| **Registry** | `core/registry.py` | Reads transformers' task taxonomy → modality → AutoModel. Dynamic class/fn resolution. |
| **Engine** | `core/engine.py` | Loads & caches pipelines/models. Auto device (cuda/mps/cpu) + dtype. |
| **I/O** | `core/io.py` | Coerces multimodal inputs; serializes outputs; saves audio/images to disk. |
| **Provider** | `models/transformers.py` | `TransformerModel` - local model as the agent brain, multimodal content blocks. |
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
