# What are the best location opportunities for Product X for Supplier Z?
# What are the best product opportunities at Location Y for Supplier Z?
# Which stores have the highest opportunity for all products?
# What are the operator details for  Location Y?

import pandas as pd
import langchain as lc
import torch
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    pipeline,
)

model_name = "mistralai/Mistral-7B-Instruct-v0.1"

model_config = AutoConfig.from_pretrained(
    model_name,
)
model = AutoModelForCausalLM.from_pretrained(model_name, config=model_config)
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"
#################################################################
# bitsandbytes parameters
#################################################################

# Activate 4-bit precision base model loading
use_4bit = True
# Compute dtype for 4-bit base models
bnb_4bit_compute_dtype = "float16"
# Quantization type (fp4 or nf4)
bnb_4bit_quant_type = "nf4"
# Activate nested quantization for 4-bit base models (double quantization)
use_nested_quant = False
compute_dtype = getattr(torch, bnb_4bit_compute_dtype)

bnb_config = BitsAndBytesConfig(
    load_in_4bit=use_4bit,
    bnb_4bit_quant_type=bnb_4bit_quant_type,
    bnb_4bit_compute_dtype=compute_dtype,
    bnb_4bit_use_double_quant=use_nested_quant,
)

# Check GPU compatibility with bfloat16
if compute_dtype == torch.float16 and use_4bit:
    major, _ = torch.cuda.get_device_capability()
    if major >= 8:
        print("=" * 80)
        print("Your GPU supports bfloat16: accelerate training with bf16=True")
        print("=" * 80)

inputs_not_chat = tokenizer.encode_plus(
    "[INST] Tell me about fantasy football? [/INST]", return_tensors="pt"
)["input_ids"].to("cuda")

generated_ids = model.generate(inputs_not_chat, max_new_tokens=1000, do_sample=True)
decoded = tokenizer.batch_decode(generated_ids)


from langchain.llms import huggingface_pipeline
from langchain.chains.llm import LLMChain
from langchain.prompts import PromptTemplate

text_generation_pipeline = pipeline(
    model=model,
    tokenizer=tokenizer,
    task="text-generation",
    temperature=0.2,
    repetition_penalty=1.1,
    return_full_text=True,
    max_new_tokens=300,
)

prompt_template = """
### [INST] 
Instruction: Answer the question based on your 
fantasy football knowledge. Here is context to help:

{context}

### QUESTION:
{question} 

[/INST]
 """

mistral_llm = huggingface_pipeline(pipeline=text_generation_pipeline)

# Create prompt from prompt template
prompt = PromptTemplate(
    input_variables=["context", "question"],
    template=prompt_template,
)

# Create llm chain
llm_chain = LLMChain(llm=mistral_llm, prompt=prompt)

query = "Should I pick up Alvin Kamara for my fantasy team?"

retriever = db.as_retriever()

rag_chain = {"context": retriever, "question": RunnablePassthrough()} | llm_chain

rag_chain.invoke(query)
