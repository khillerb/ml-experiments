import torch
from transformers import GPT2Tokenizer, GPT2Model
from datasets import load_dataset


# Load GPT2 tokenizer and model
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token
gpt2_model = GPT2Model.from_pretrained("gpt2")

# Sample text dataset
text_dataset = load_dataset("imdb")["train"]["text"]

# Tokenize text dataset
tokenized_dataset = tokenizer(
    text_dataset,
    return_tensors="pt",
    padding=True,
    truncation=True,
    max_length=512,
    add_special_tokens=True,
)

# Pass tokenized input through the model to get embeddings
with torch.no_grad():
    outputs = gpt2_model(**tokenized_dataset)

# Extract embeddings
embeddings = outputs.last_hidden_state

# Create a dictionary to map tokens to embeddings
token_to_embedding = {}
for i, text in enumerate(text_dataset):
    tokens = tokenizer.tokenize(text)
    for j, token in enumerate(tokens):
        if token not in token_to_embedding:
            token_to_embedding[token] = embeddings[i, j].tolist()

# Example: Access embedding for token 'ln'
print(token_to_embedding)
