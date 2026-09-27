# Use a pipeline as a high-level helper
from transformers import pipeline

pipe = pipeline("text-generation", model="AdaptLLM/finance-LLM", max_new_tokens=500)
print(
    pipe(
        "System: You are a helpful AI assistant.Please respond once to the user below. User: How can I get the greatest return on a 10k invest? I would prefer real/physical assets, especially those that can protect against currency devaluation and inflation, or even take advantage of inflation. Assistant:"
    )
)
