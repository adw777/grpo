import torch
from unsloth import FastLanguageModel
from datasets import load_dataset
from trl import SFTTrainer
from transformers import TrainingArguments, DataCollatorForLanguageModeling
import os

# Configuration
max_seq_length = 2048
dtype = None  # Auto detection
load_in_4bit = True

# Load model and tokenizer
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="unsloth/Llama-3.1-1B-bnb-4bit",
    max_seq_length=max_seq_length,
    dtype=dtype,
    load_in_4bit=load_in_4bit,
)

# Add LoRA adapters
model = FastLanguageModel.get_peft_model(
    model,
    r=16,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    lora_alpha=16,
    lora_dropout=0,
    bias="none",
    use_gradient_checkpointing="unsloth",
    random_state=3407,
    use_rslora=False,
    loftq_config=None,
)

# Load dataset
dataset = load_dataset("axondendriteplus/context-relevance-classifier-dataset")

# Formatting function
def format_prompt(examples):
    """Format the data for classification task"""
    formatted_examples = []
    
    for i in range(len(examples['question'])):
        question = examples['question'][i]
        answer = examples['answer'][i]
        context = examples['context'][i]
        label = examples['label'][i]
        
        # Create classification prompt
        prompt = f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>
You are a context relevance classifier. Given a question, answer, and context, determine if the answer was generated from the given context. Respond with either "YES" if the answer is derived from the context, or "NO" if it is not.

<|eot_id|><|start_header_id|>user<|end_header_id|>
Question: {question}

Answer: {answer}

Context: {context}

Was this answer generated from the given context? Respond with YES or NO only.
<|eot_id|><|start_header_id|>assistant<|end_header_id|>
{"YES" if label == 1 else "NO"}<|eot_id|>"""
        
        formatted_examples.append(prompt)
    
    return {"text": formatted_examples}

# Apply formatting
train_dataset = dataset["train"].map(format_prompt, batched=True, remove_columns=dataset["train"].column_names)

# Training arguments
training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=3,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,
    optim="adamw_8bit",
    learning_rate=2e-4,
    weight_decay=0.01,
    lr_scheduler_type="linear",
    warmup_steps=50,
    logging_steps=10,
    save_steps=100,
    evaluation_strategy="no",
    save_strategy="steps",
    save_total_limit=2,
    dataloader_pin_memory=False,
    fp16=not torch.cuda.is_bf16_supported(),
    bf16=torch.cuda.is_bf16_supported(),
    group_by_length=True,
    seed=3407,
    report_to="none",
)

# Data collator
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False,
)

# Initialize trainer
trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=train_dataset,
    dataset_text_field="text",
    max_seq_length=max_seq_length,
    data_collator=data_collator,
    args=training_args,
    packing=False,
)

# Show model info
gpu_stats = torch.cuda.get_device_properties(0)
start_gpu_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
max_memory = round(gpu_stats.total_memory / 1024 / 1024 / 1024, 3)
print(f"GPU = {gpu_stats.name}. Max memory = {max_memory} GB.")
print(f"{start_gpu_memory} GB of memory reserved.")

# Train the model
trainer_stats = trainer.train()

# Show final memory and time stats
used_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
used_memory_for_lora = round(used_memory - start_gpu_memory, 3)
used_percentage = round(used_memory / max_memory * 100, 3)
lora_percentage = round(used_memory_for_lora / max_memory * 100, 3)
print(f"{trainer_stats.metrics['train_runtime']} seconds used for training.")
print(f"{round(trainer_stats.metrics['train_runtime']/60, 2)} minutes used for training.")
print(f"Peak reserved memory = {used_memory} GB.")
print(f"Peak reserved memory for training = {used_memory_for_lora} GB.")
print(f"Peak reserved memory % of max memory = {used_percentage} %.")
print(f"Peak reserved memory for training % of max memory = {lora_percentage} %.")

# Save model
model.save_pretrained("context-relevance-classifier")
tokenizer.save_pretrained("context-relevance-classifier")

# Save to Hugging Face Hub (optional)
# model.push_to_hub("your-username/context-relevance-classifier", token="your-token")
# tokenizer.push_to_hub("your-username/context-relevance-classifier", token="your-token")

print("Training completed!")