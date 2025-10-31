from strands import Agent  # pip3 install strands-agents
from strands_tools import shell  # pip3 install strands-agents-tools

from models.transformers import TransformerModel  # Local transformers model provider
from jsonl_session_manager import JsonlSessionManager

# Create a TransformerModel instance with Qwen3-1.7B
# Use fine-tuned merged model
transformer_model = TransformerModel(
    model_path="./qwen3_session_merged",  # Fine-tuned Strands-aware model
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

session_manager = JsonlSessionManager(
    session_id="test_strands", template_name="qwen3", storage_dir="./test_training_data"
)

# Create an agent using the TransformerModel
agent = Agent(
    model=transformer_model,
    tools=[shell],
    # session_manager=session_manager
)

# Use the agent
agent("list your tools")  # Prints model output to stdout by default
agent("run ls -la with shell tool")  # Prints model output to stdout by default

# interactive mode
# while True:
#     agent(input("\n# "))
