from strands import Agent  # pip3 install strands-agents
from models.transformers import TransformerModel  # Local transformers model provider
from strands_tools import shell  # pip3 install strands-agents-tools

# Create a TransformerModel instance with Qwen3-1.7B
# Option 1: Use fine-tuned merged model
transformer_model = TransformerModel(
    model_path="./qwen3_1.7b_strands_merged",  # Fine-tuned Strands-aware model
    # model_path="Qwen/Qwen3-1.7B",  # Or use base model
    device="auto",  # Automatically selects cuda/mps/cpu
    enable_thinking=True,  # Enable Qwen3 thinking mode
    params={
        "max_tokens": 500,
        "temperature": 1,
        "top_p": 0.9,
        "top_k": 20,
        "repetition_penalty": 1.2,  # Prevent repetition loops
    },
)

# Option 2: Use your fine-tuned merged model
# transformer_model = TransformerModel(
#     model_path="./maxs_merged",  # Your fine-tuned Strands-aware model
#     device="auto",
#     enable_thinking=True,
# )

# Create an agent using the TransformerModel
agent = Agent(model=transformer_model, tools=[shell])

# Use the agent
# agent("do you know strands agents sdk")  # Prints model output to stdout by default
# agent("list your tools")  # Prints model output to stdout by default
# agent("run ls -la with shell tool")  # Prints model output to stdout by default

# interactive mode
while True:
    agent(input("\n# "))
