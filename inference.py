from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import re

# Load model and tokenizer
tokenizer = AutoTokenizer.from_pretrained("legal-grpo/checkpoint-500")
model = AutoModelForCausalLM.from_pretrained(
    "legal-grpo/checkpoint-500",
    torch_dtype=torch.float16,
    device_map="auto"
)

# Define structured tags
reasoning_start = "<legal_analysis>"
reasoning_end = "</legal_analysis>"
solution_start = "<answer>"
solution_end = "</answer>"

user_question = "what are the core components of insider trading?"

system_prompt = f"""
<|begin_of_text|>
<|start_header_id|>system<|end_header_id|>
You are a legal assistant specialized in Indian law. Provide only factual and accurate legal information in a strictly structured format.

Follow these rules exactly:
1. Analyze the legal question between <legal_analysis> and </legal_analysis>
2. Provide the final answer between <answer> and </answer>
3. Use relevant Indian legal principles, statutes, or case law
4. Never invent laws, citations, or case references
5. If unsure or the information is unavailable, respond: “This requires professional consultation.”
6. Do NOT include any text outside the <legal_analysis>…</legal_analysis> and <answer>…</answer> tags

<|eot_id|>
<|start_header_id|>user<|end_header_id|>
{user_question}
<|eot_id|>
<|start_header_id|>assistant<|end_header_id|>
"""

# Tokenize input
inputs = tokenizer(system_prompt, return_tensors="pt").to(model.device)

# Generate response
outputs = model.generate(
    **inputs,
    max_new_tokens=516,
    temperature=0.2,
    top_p=0.95,
    repetition_penalty=1.15,
    do_sample=True,
    eos_token_id=tokenizer.eos_token_id
)

# Decode response
response = tokenizer.decode(outputs[0], skip_special_tokens=False)

print("Full Response:\n", response)
