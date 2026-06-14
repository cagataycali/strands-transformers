"""Example: a fully multimodal Strands agent powered by transformers.

The agent uses a hosted/local LLM as its reasoning brain and `use_transformers`
as its hands — giving it 100% of HuggingFace transformers: any task, any modality.

    python agent.py
    # then: "transcribe sample.wav", "what's in cat.jpg?", "say hello as audio"

Swap the brain for a LOCAL HF model with TransformerModel (see bottom).
"""

from strands import Agent

from strands_transformers import use_transformers

SYSTEM_PROMPT = """You are a multimodal AI with full access to HuggingFace transformers
through the `use_transformers` tool. You can run ANY task across text, image, video,
audio, and robot-state — natively.

Workflow:
1. Don't guess task names — discover: use_transformers(action="tasks") or
   action="task_info" / action="modalities".
2. Run with: use_transformers(action="run", task="<task>", inputs=<path|url|text|dict>).
   For multimodal models pass a dict, e.g. {"images": "scene.jpg", "text": "..."}.
3. For low-level / robot-action (VLA) models, load components with action="call"
   (AutoProcessor / AutoModelForImageTextToText) and cache them with cache_key.
Generated audio/images are saved to disk; report their paths to the user."""

def build_agent():
    """Build the multimodal agent.

    Brain selection:
    - Set STRANDS_TRANSFORMERS_LOCAL=1 (or have no cloud creds) to run a LOCAL
      HuggingFace model as the brain via TransformerModel — fully offline.
    - Otherwise Strands' default provider (Bedrock/OpenAI/… by env) is used.
    """
    import os

    if os.getenv("STRANDS_TRANSFORMERS_LOCAL", "").lower() in ("1", "true", "yes"):
        from strands_transformers import TransformerModel

        brain = TransformerModel(
            model_path=os.getenv("STRANDS_TRANSFORMERS_MODEL", "Qwen/Qwen3-0.6B"),
            device="auto",
            params={"max_tokens": 256, "temperature": 0.7},
        )
        return Agent(model=brain, tools=[use_transformers], system_prompt=SYSTEM_PROMPT)

    return Agent(tools=[use_transformers], system_prompt=SYSTEM_PROMPT)


agent = build_agent()

if __name__ == "__main__":
    print("🤗 Multimodal transformers agent. Ctrl-C to exit.\n")
    while True:
        try:
            agent(input("\n# "))
        except (KeyboardInterrupt, EOFError):
            print("\nbye 👋")
            break


# ─── Alternative: use a LOCAL HuggingFace model as the agent's brain ───
#
# from strands_transformers import TransformerModel
# brain = TransformerModel(model_path="Qwen/Qwen3-1.7B", device="auto")
# agent = Agent(model=brain, tools=[use_transformers], system_prompt=SYSTEM_PROMPT)
