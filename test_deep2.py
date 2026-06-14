"""Round 2: TTS, ASR, image-text-to-text, fixed text-gen."""
import os, sys, json, time, warnings, logging
warnings.filterwarnings("ignore")
logging.getLogger("transformers").setLevel(logging.ERROR)
os.environ["TRANSFORMERS_VERBOSITY"] = "error"

from strands_transformers import use_transformers

results = {}

def run_test(name, **kwargs):
    print(f"\n{'='*60}\n[{name}]\n{'='*60}", flush=True)
    t0 = time.time()
    try:
        r = use_transformers(**kwargs)
        elapsed = time.time() - t0
        status = r.get("status", "?")
        content = "".join(c.get("text", "") for c in r.get("content", []))
        artifacts = r.get("artifacts", [])
        print(f"  status={status} time={elapsed:.1f}s")
        print(f"  content[:300]={content[:300]}")
        if artifacts:
            print(f"  artifacts={artifacts}")
            for a in artifacts:
                if os.path.exists(a):
                    print(f"    -> {a}: {os.path.getsize(a)} bytes")
        results[name] = {"status": status, "time": round(elapsed, 1),
                         "preview": content[:250], "artifacts": artifacts}
    except Exception as e:
        elapsed = time.time() - t0
        print(f"  ❌ EXCEPTION: {type(e).__name__}: {str(e)[:300]}")
        results[name] = {"status": "exception", "time": round(elapsed, 1),
                         "error": f"{type(e).__name__}: {str(e)[:200]}"}

# Fixed text-gen with proper kwarg name
run_test("text_generation_FIXED",
         action="run", task="text-generation",
         model="distilbert/distilgpt2",
         inputs="The future of robotics is",
         parameters={"max_new_tokens": 30, "do_sample": False})

# Fixed summarization
run_test("summarization_FIXED",
         action="run", task="summarization",
         inputs="The Apollo program was a series of NASA missions that successfully landed astronauts on the Moon between 1969 and 1972. The program achieved its first landing with Apollo 11 on July 20, 1969, when Neil Armstrong became the first human to walk on the Moon. The program demonstrated technological capability and inspired generations of scientists.",
         parameters={"max_length": 50, "min_length": 15})

# Inspect — meta action
run_test("inspect_pipeline",
         action="inspect", target="pipeline")

# Cache listing
run_test("cache_list",
         action="cache")

# === TTS — produces audio artifact ===
run_test("text_to_speech",
         action="run", task="text-to-audio",
         model="facebook/mms-tts-eng",
         inputs="Hello from strands transformers running on Thor.")

# === ASR — round-trip the TTS output ===
# We'll get its path from the cache action

# === image-text-to-text ===
run_test("image_text_to_text_VLM",
         action="run", task="image-text-to-text",
         model="HuggingFaceTB/SmolVLM-256M-Instruct",
         inputs={"images": "http://images.cocodataset.org/val2017/000000039769.jpg",
                 "text": "What animals are in this picture? Answer in one sentence."})

# === call action — dynamic class instantiation ===
run_test("call_AutoTokenizer",
         action="call",
         target="AutoTokenizer.from_pretrained",
         parameters={"pretrained_model_name_or_path": "bert-base-uncased"},
         cache_key="bert_tok")

run_test("call_cached_method",
         action="call",
         target="cached:bert_tok.encode",
         parameters={"text": "Hello world"})

# === clear_cache ===
run_test("clear_cache",
         action="clear_cache", cache_key="bert_tok")

# === Save ===
print("\n" + "="*60 + "\nSUMMARY\n" + "="*60)
print(json.dumps(results, indent=2))
with open("/tmp/strands_transformers_round2.json", "w") as f:
    json.dump(results, f, indent=2)
