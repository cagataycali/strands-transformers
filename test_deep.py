"""Deep test of strands-transformers tool with diverse real inputs."""
import os, sys, json, time, warnings, logging
warnings.filterwarnings("ignore")
logging.getLogger("transformers").setLevel(logging.ERROR)
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
os.environ["TQDM_DISABLE"] = "1"

from strands_transformers import use_transformers

results = {}

def run_test(name, **kwargs):
    print(f"\n{'='*60}\n[{name}]\n{'='*60}", flush=True)
    t0 = time.time()
    try:
        r = use_transformers(**kwargs)
        elapsed = time.time() - t0
        status = r.get("status", "?")
        # Extract text content
        content = ""
        for c in r.get("content", []):
            content += c.get("text", "")
        data = r.get("data")
        artifacts = r.get("artifacts", [])
        print(f"  status={status} time={elapsed:.1f}s")
        print(f"  content[:300]={content[:300]}")
        if data is not None:
            print(f"  data={str(data)[:300]}")
        if artifacts:
            print(f"  artifacts={artifacts}")
        results[name] = {"status": status, "time": round(elapsed, 1),
                         "preview": content[:200], "artifacts": artifacts}
        return r
    except Exception as e:
        elapsed = time.time() - t0
        print(f"  ❌ EXCEPTION: {type(e).__name__}: {str(e)[:300]}")
        results[name] = {"status": "exception", "time": round(elapsed, 1),
                         "error": f"{type(e).__name__}: {str(e)[:200]}"}
        return None

# === Discovery actions ===
run_test("discover_modalities", action="modalities")
run_test("discover_task_info", action="task_info", task="image-text-to-text")
run_test("discover_classes", action="classes")

# === Text tasks ===
run_test("text_classification_negative",
         action="run", task="text-classification",
         inputs="This is the worst experience I've ever had.")

run_test("zero_shot_classification",
         action="run", task="zero-shot-classification",
         inputs={"sequences": "I want to book a hotel in Paris next week",
                 "candidate_labels": ["travel", "cooking", "sports", "politics"]})

run_test("token_classification_NER",
         action="run", task="token-classification",
         inputs="My name is Sarah and I work at Google in Mountain View.")

run_test("text_generation_short",
         action="run", task="text-generation",
         model="distilbert/distilgpt2",
         inputs="The future of robotics is",
         params={"max_new_tokens": 30, "do_sample": False})

run_test("fill_mask",
         action="run", task="fill-mask",
         inputs="The capital of France is <mask>.")

run_test("question_answering",
         action="run", task="question-answering",
         inputs={"question": "Who developed the theory of relativity?",
                 "context": "Albert Einstein was a German-born theoretical physicist who developed the theory of relativity, one of the two pillars of modern physics."})

run_test("summarization",
         action="run", task="summarization",
         inputs="The Apollo program was a series of NASA missions that successfully landed astronauts on the Moon between 1969 and 1972. The program achieved its first landing with Apollo 11 on July 20, 1969, when Neil Armstrong became the first human to walk on the Moon. The program demonstrated technological capability and inspired generations of scientists.",
         params={"max_length": 50, "min_length": 15})

# === Image tasks (use small public URL) ===
IMG_URL = "http://images.cocodataset.org/val2017/000000039769.jpg"  # cats

run_test("image_classification",
         action="run", task="image-classification",
         inputs=IMG_URL)

run_test("object_detection",
         action="run", task="object-detection",
         inputs=IMG_URL)

# === Save results ===
print("\n\n" + "="*60)
print("SUMMARY")
print("="*60)
print(json.dumps(results, indent=2))

with open("/tmp/strands_transformers_results.json", "w") as f:
    json.dump(results, f, indent=2)
print(f"\n💾 Saved to /tmp/strands_transformers_results.json")
