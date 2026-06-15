"""Use a LOCAL HuggingFace model as the agent's brain (TransformerModel provider).

Besides being a *tool* (`use_transformers`), this package ships a Strands model
provider, `TransformerModel`, that runs any HF causal-LM locally as the agent's
reasoning engine - with streaming, chat templates, Qwen3 `<think>` reasoning, and
XML tool-calling.

Pair it with `use_transformers` and the agent can both think locally AND reach the
entire transformers ecosystem as tools.

    PYTHONPATH=. python examples/local_model_agent.py

Note: small models (≤1B) reason but are unreliable at structured tool calls; use
a 7B+ instruct model for dependable tool use.
"""

from strands import Agent

from strands_transformers import TransformerModel, use_transformers

MODEL = "Qwen/Qwen3-0.6B"  # swap for a larger instruct model for real tool use


def build_agent(max_tokens: int = 64):
    brain = TransformerModel(
        model_path=MODEL,
        device="auto",
        enable_thinking=False,
        params={"max_tokens": max_tokens, "temperature": 0.7},
    )
    return Agent(
        model=brain,
        tools=[use_transformers],
        system_prompt="You are a concise local assistant with access to the "
        "use_transformers tool for any HuggingFace task.",
    )


if __name__ == "__main__":
    agent = build_agent(max_tokens=48)
    print(agent("Greet the user in one short sentence."))
