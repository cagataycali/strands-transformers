"""Round 3: ASR roundtrip + VLM fix attempt + summarization investigation."""
import os, json, time, warnings, logging, glob
warnings.filterwarnings("ignore")
logging.getLogger("transformers").setLevel(logging.ERROR)
os.environ["TRANSFORMERS_VERBOSITY"] = "error"

from strands_transformers import use_transformers

# Find the TTS audio we just generated
audio_files = sorted(glob.glob("/tmp/strands_transformers/audio_*.wav"))
latest_audio = audio_files[-1] if audio_files else None
print(f"Latest TTS audio: {latest_audio}")

results = {}

def run_test(name, **kwargs):
    print(f"\n[{name}]", flush=True)
    t0 = time.time()
    try:
        r = use_transformers(**kwargs)
        elapsed = time.time() - t0
        status = r.get("status", "?")
        content = "".join(c.get("text", "") for c in r.get("content", []))
        print(f"  status={status} time={elapsed:.1f}s")
        print(f"  preview={content[:400]}")
        results[name] = {"status": status, "time": round(elapsed, 1), "preview": content[:400]}
    except Exception as e:
        print(f"  ❌ {type(e).__name__}: {str(e)[:200]}")
        results[name] = {"status": "exception", "error": str(e)[:200]}

# === Check what summarization is registered as ===
import json as j
r = use_transformers(action="tasks")
text = "".join(c.get("text", "") for c in r.get("content", []))
# Search for "summar" in tasks
for line in text.split("\n"):
    if "summar" in line.lower() or "translat" in line.lower():
        print(f"  TASK MATCH: {line}")

# === ASR with the TTS file ===
if latest_audio:
    run_test("asr_whisper_tiny",
             action="run", task="automatic-speech-recognition",
             model="openai/whisper-tiny",
             inputs=latest_audio)

# === VLM with proper chat-format prompt (idefics3 needs <image> token) ===
run_test("vlm_with_chat_format",
         action="run", task="image-text-to-text",
         model="HuggingFaceTB/SmolVLM-256M-Instruct",
         inputs={"text": [{"role": "user", "content": [
                    {"type": "image", "url": "http://images.cocodataset.org/val2017/000000039769.jpg"},
                    {"type": "text", "text": "What animals are in this picture? One sentence."}
                ]}]},
         parameters={"max_new_tokens": 50})

# === Translation (text2text-generation) ===
run_test("translation_en_to_fr",
         action="run", task="translation_en_to_fr",
         inputs="The cat sits on the mat.")

run_test("text2text_generation",
         action="run", task="text2text-generation",
         inputs="summarize: The Apollo program was a series of NASA missions that landed humans on the Moon between 1969 and 1972.",
         parameters={"max_length": 30})

print("\n" + json.dumps(results, indent=2))
with open("/tmp/strands_transformers_round3.json", "w") as f:
    json.dump(results, f, indent=2)
