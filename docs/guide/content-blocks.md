# Content blocks & modalities

`TransformerModel` consumes the full Strands content-block taxonomy. Every output
below is a **real** model result (CUDA · transformers 5.12 · torch 2.10),
reproducible from the matching example.

```mermaid
flowchart TB
    subgraph CB["content blocks → handler"]
        direction LR
        T["📝 text"] --> HT["tokenizer fast-path"]
        I["🖼️ image"] --> HI["AutoProcessor 👁"]
        V["🎬 video"] --> HV["processor + VideoMetadata (fps)"]
        TR["🧰 toolResult(image)"] --> HI
        D["📄 document"] --> HD["flatten → text"]
        AU["🔊 audio*"] --> HA["feature_extractor 🔊"]
        OM["🔊 audio in/out"] --> HO["Omni Thinker+Talker"]
    end
    classDef blk fill:#7C5CFF22,stroke:#7C5CFF,stroke-width:1.5px,color:#7C5CFF;
    classDef h fill:#22D3EE1f,stroke:#22D3EE,stroke-width:1.5px,color:#0F91A6;
    class T,I,V,TR,D,AU,OM blk;
    class HT,HI,HV,HD,HA,HO h;
```
<sub>* `audio` is our extension to the Strands taxonomy - see [Audio](audio.md).</sub>

## Example responses

| Block | Input | Script | Real output |
|-------|-------|--------|-------------|
| `image` | <img src="../../assets/img/green.png" width="44"/> "Color? One word." | `multimodal_agent.py` | `"Green."` |
| `video` | 8 frames dark→bright (`fps=2.0`) | `multimodal_advanced.py` | `"BRIGHTER."` |
| `image` in `toolResult` | tool returns <img src="../../assets/img/blue.png" width="44"/> | `multimodal_advanced.py` | `"Blue."` |
| `document` | txt "…passphrase is BANANA-42…" | `document_and_audio.py` | recovers `BANANA-42` |
| `audio` | 440 Hz tone (Omni) | `omni_audio.py` | `"It's a pure tone."` |

## Media you can feed it

These are real artifacts - a TTS clip and an MP4 - that round-trip through the
library (the video decodes to 24 frames @ 12 fps; the audio re-transcribes
intelligibly):

<div align="center">
<table>
<tr><th>🎬 video (mp4 / gif)</th><th>🔊 audio (TTS, wav)</th></tr>
<tr>
<td><img src="../../assets/video/demo.gif" width="220"/></td>
<td><audio controls src="../../assets/audio/tts_demo.wav"></audio></td>
</tr>
</table>
</div>

```python
# /// script
# requires-python = ">=3.10"
# dependencies = ["strands-transformers[vision]", "imageio[ffmpeg]"]
# ///
from strands_transformers import use_transformers

# audio out - text-to-speech writes a .wav artifact
tts = use_transformers(action="run", task="text-to-audio",
                       model="facebook/mms-tts-eng", inputs="hello from strands")
print("audio:", tts["artifacts"][0])

# video in - classify a clip (frame list is auto-stacked to (T,H,W,C))
import numpy as np
frames = [np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8) for _ in range(16)]
vid = use_transformers(action="run", task="video-classification",
                       model="MCG-NJU/videomae-base-finetuned-kinetics", inputs=frames)
print("top label:", vid["data"][0]["label"])
```

## 🖼️ Image

```python
result = agent([
    {"image": {"format": "png", "source": {"bytes": png_bytes}}},
    {"text": "What color is this image? One word."},
])
```

```console
Green.
```

## 🎬 Video

A `video` block is a list of frames (or a `(T,H,W,C)` array / container bytes).
Provide `fps` so the model builds real frame timestamps.

```python
model.stream([{"role": "user", "content": [
    {"video": {"format": "mp4", "fps": 2.0, "source": {"bytes": frames}}},
    {"text": "Does this video get brighter or darker?"},
]}])
```

```console
BRIGHTER.
```

## 🧰 Tool-result images (the agentic-vision loop)

A tool returns an image *inside a `toolResult`*; the VLM reasons over it on the
next turn - exactly the loop you want for screen-watchers and camera agents.

```python
{"toolResult": {"toolUseId": "t1", "status": "success", "content": [
    {"text": "Here is the captured screen:"},
    {"image": {"format": "png", "source": {"bytes": blue_png}}},
]}}
```

```console
Blue.
```

## 📄 Document

```python
{"document": {"name": "secret", "format": "txt",
              "source": {"bytes": b"...the passphrase is BANANA-42..."}}}
# "What is the passphrase?"
```

```console
BANANA-42
```

## 🔊 Audio

See **[Audio (in & out)](audio.md)** - with playable real outputs.

## Supported transformers modalities (the tool)

| Modality | Example tasks |
|----------|---------------|
| **text** | text-generation, fill-mask, token/text-classification, feature-extraction, table-qa |
| **image** | image-classification, depth-estimation, image-feature-extraction, keypoint-matching |
| **audio** | automatic-speech-recognition, audio-classification, text-to-audio |
| **video** | video-classification |
| **multimodal** | image-text-to-text, visual/document-qa, object-detection, segmentation, zero-shot-*, any-to-any |

Run `use_transformers(action="tasks")` for the live, complete list on your install.
