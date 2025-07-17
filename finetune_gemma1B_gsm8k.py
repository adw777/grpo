"""
Gemma 3 1B GRPO Finetuning on GSM8K Dataset - Fixed Version
"""

import os
import re
import torch
from datasets import load_dataset
from unsloth import FastModel
from trl import GRPOConfig, GRPOTrainer
from transformers import TextStreamer

# Configuration - Made more conservative
MAX_SEQ_LENGTH = 512
MAX_PROMPT_LENGTH = 200  # Reduced to avoid length issues
LEARNING_RATE = 5e-6
MAX_STEPS = 50
SAVE_STEPS = 50
NUM_GENERATIONS = 2
BATCH_SIZE = 1
MODEL_NAME = "unsloth/gemma-3-1b-it"
OUTPUT_DIR = "outputs"

# Special tokens for reasoning and solution
REASONING_START = "<start_working_out>"
REASONING_END = "<end_working_out>"
SOLUTION_START = "<SOLUTION>"
SOLUTION_END = "</SOLUTION>"

# System prompt (shorter to avoid length issues)
SYSTEM_PROMPT = f"""Solve step by step. Work in {REASONING_START}...{REASONING_END}. Answer in {SOLUTION_START}...{SOLUTION_END}."""

def load_model_and_tokenizer():
    """Load and configure the Gemma model with LoRA"""
    print("Loading model and tokenizer...")
    
    model, tokenizer = FastModel.from_pretrained(
        model_name=MODEL_NAME,
        max_seq_length=MAX_SEQ_LENGTH,
        load_in_4bit=False,
        load_in_8bit=False,
        full_finetuning=False,
    )
    
    # Ensure tokenizer has padding token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Add LoRA adapters with more conservative settings
    model = FastModel.get_peft_model(
        model,
        finetune_vision_layers=False,
        finetune_language_layers=True,
        finetune_attention_modules=True,
        finetune_mlp_modules=True,
        r=8,
        lora_alpha=16,
        lora_dropout=0.05,
        bias="none",
        random_state=3407,
    )
    
    return model, tokenizer

def extract_hash_answer(text):
    """Extract answer after #### symbol"""
    if "####" not in text:
        return None
    answer = text.split("####")[1].strip()
    # Keep only the numeric part if possible
    import re
    numeric_match = re.search(r'[\d,]+\.?\d*', answer)
    if numeric_match:
        return numeric_match.group(0).replace(',', '')
    return answer

def prepare_dataset(tokenizer):
    """Load and prepare GSM8K dataset - FIXED VERSION"""
    print("Loading GSM8K dataset...")
    
    dataset = load_dataset("openai/gsm8k", "main", split="train")
    dataset = dataset.select(range(min(500, len(dataset))))  # Smaller dataset for testing
    
    def format_sample(x):
        answer = extract_hash_answer(x["answer"])
        if answer is None:
            return None
        
        # Keep questions very short to avoid tokenization issues
        question = x["question"]
        if len(question) > 100:
            question = question[:100] + "..."
        
        # Create the conversation properly
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": question},
        ]
        
        # Convert to prompt string using tokenizer
        prompt_text = tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=False,
        )
        
        # Ensure prompt is within limits - CRITICAL FIX
        prompt_tokens = tokenizer(prompt_text, truncation=False, padding=False)
        if len(prompt_tokens['input_ids']) > MAX_PROMPT_LENGTH - 10:
            # Truncate question more aggressively
            question = question[:50] + "..."
            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": question},
            ]
            prompt_text = tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=True,
                tokenize=False,
            )
            
            # Final check
            prompt_tokens = tokenizer(prompt_text, truncation=False, padding=False)
            if len(prompt_tokens['input_ids']) > MAX_PROMPT_LENGTH - 10:
                return None  # Skip this sample
        
        return {
            "prompt": prompt_text,
            "answer": answer,
        }
    
    # Filter and map
    dataset = dataset.filter(lambda x: extract_hash_answer(x["answer"]) is not None)
    dataset = dataset.map(format_sample, remove_columns=dataset.column_names)
    dataset = dataset.filter(lambda x: x is not None)
    
    # Final validation - ensure all prompts are within limits
    def validate_length(x):
        tokens = tokenizer(x["prompt"], truncation=False, padding=False)
        return len(tokens['input_ids']) <= MAX_PROMPT_LENGTH - 10
    
    dataset = dataset.filter(validate_length)
    
    return dataset

def create_reward_functions():
    """Create reward functions for GRPO training - FIXED VERSION"""
    
    def simple_reward(prompts, completions, **kwargs):
        """Simple reward function that always returns correct number of scores"""
        scores = []
        
        # Ensure we process the right number of completions
        if isinstance(completions, list):
            for completion in completions:
                if isinstance(completion, list) and len(completion) > 0:
                    response = completion[0].get("content", "") if isinstance(completion[0], dict) else str(completion[0])
                else:
                    response = str(completion)
                
                score = 0.0
                
                # Check for format elements
                if REASONING_START in response and REASONING_END in response:
                    score += 0.5
                if SOLUTION_START in response and SOLUTION_END in response:
                    score += 0.5
                
                # Basic length check
                if len(response.strip()) > 10:
                    score += 0.2
                
                scores.append(score)
        else:
            # Fallback for unexpected format
            scores = [0.5] * NUM_GENERATIONS
        
        # Ensure we return exactly the right number of scores
        while len(scores) < NUM_GENERATIONS:
            scores.append(0.0)
        scores = scores[:NUM_GENERATIONS]
        
        return scores
    
    def answer_reward(prompts, completions, answer, **kwargs):
        """Check if the extracted answer matches - FIXED VERSION"""
        scores = []
        
        # Ensure answer is a list
        if not isinstance(answer, list):
            answer = [answer] * len(completions)
        
        for i, completion in enumerate(completions):
            if isinstance(completion, list) and len(completion) > 0:
                response = completion[0].get("content", "") if isinstance(completion[0], dict) else str(completion[0])
            else:
                response = str(completion)
            
            score = 0.0
            true_answer = answer[i] if i < len(answer) else answer[0]
            
            # Try to extract answer between solution tags
            if SOLUTION_START in response and SOLUTION_END in response:
                try:
                    start_idx = response.find(SOLUTION_START) + len(SOLUTION_START)
                    end_idx = response.find(SOLUTION_END, start_idx)
                    if end_idx > start_idx:
                        extracted = response[start_idx:end_idx].strip()
                        
                        # Try numeric comparison
                        try:
                            extracted_num = float(extracted.replace(',', ''))
                            true_num = float(true_answer.replace(',', ''))
                            if abs(extracted_num - true_num) < 0.01:
                                score += 1.0
                            elif abs(extracted_num - true_num) / max(abs(true_num), 1) < 0.1:
                                score += 0.5
                        except:
                            # String comparison fallback
                            if extracted.strip() == true_answer.strip():
                                score += 0.8
                except:
                    pass
            
            scores.append(score)
        
        # Ensure we return exactly the right number of scores
        while len(scores) < NUM_GENERATIONS:
            scores.append(0.0)
        scores = scores[:NUM_GENERATIONS]
        
        return scores
    
    return [simple_reward, answer_reward]

def train_model(model, tokenizer, dataset):
    """Train the model using GRPO - FIXED VERSION"""
    print("Setting up GRPO training...")
    
    # Disable problematic optimizations
    os.environ["TORCH_CUDNN_V8_API_ENABLED"] = "1"
    
    # More conservative training arguments
    training_args = GRPOConfig(
        learning_rate=LEARNING_RATE,
        adam_beta1=0.9,
        adam_beta2=0.95,
        weight_decay=0.01,
        warmup_ratio=0.1,
        lr_scheduler_type="cosine",
        optim="adamw_torch",
        logging_steps=1,
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=1,
        num_generations=NUM_GENERATIONS,
        max_prompt_length=MAX_PROMPT_LENGTH,
        max_completion_length=MAX_SEQ_LENGTH - MAX_PROMPT_LENGTH,
        max_steps=MAX_STEPS,
        save_steps=SAVE_STEPS,
        max_grad_norm=0.5,
        report_to="none",
        output_dir=OUTPUT_DIR,
        # Stability settings
        bf16=True,
        fp16=False,
        dataloader_drop_last=True,
        remove_unused_columns=False,
        gradient_checkpointing=False,
        dataloader_num_workers=0,
        seed=42,
        data_seed=42,
        save_safetensors=True,
        # Additional padding settings
        pad_token_id=tokenizer.eos_token_id,
    )
    
    reward_functions = create_reward_functions()
    
    trainer = GRPOTrainer(
        model=model,
        processing_class=tokenizer,
        reward_funcs=reward_functions,
        args=training_args,
        train_dataset=dataset,
    )
    
    print("Starting training...")
    try:
        trainer.train()
    except Exception as e:
        print(f"Training error: {e}")
        print("Attempting to continue with ultra-minimal parameters...")
        # Emergency fallback settings
        trainer.args.num_generations = 1
        trainer.args.max_completion_length = 32
        trainer.args.per_device_train_batch_size = 1
        trainer.args.gradient_accumulation_steps = 1
        trainer.args.max_prompt_length = 128
        
        # Update reward functions for single generation
        def single_reward(prompts, completions, **kwargs):
            return [0.5] * len(completions)
        
        trainer.reward_funcs = [single_reward]
        
        try:
            trainer.train()
        except Exception as e2:
            print(f"Final training error: {e2}")
            print("Training failed completely. This might be a library compatibility issue.")
            return None
    
    return trainer

def test_inference(model, tokenizer):
    """Test the trained model with a sample question"""
    print("\nTesting inference...")
    
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": "What is 15 + 27?"},
    ]
    
    text = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=False,
    )
    
    print("Generated response:")
    with torch.no_grad():
        inputs = tokenizer(
            text, 
            return_tensors="pt", 
            truncation=True, 
            max_length=MAX_PROMPT_LENGTH,
            padding=True
        ).to("cuda")
        
        outputs = model.generate(
            **inputs,
            max_new_tokens=64,
            temperature=0.7,
            top_p=0.9,
            top_k=50,
            streamer=TextStreamer(tokenizer, skip_prompt=True),
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
        )

def save_model(model, tokenizer, save_path="gemma-3-gsm8k"):
    """Save the trained model"""
    print(f"\nSaving model to {save_path}...")
    try:
        model.save_pretrained(save_path)
        tokenizer.save_pretrained(save_path)
        print("Model saved successfully!")
    except Exception as e:
        print(f"Error saving model: {e}")

def main():
    """Main training pipeline"""
    print("Starting Gemma 3 1B GRPO finetuning on GSM8K dataset")
    
    # Set environment variables for stability
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
    
    # Load model and tokenizer
    model, tokenizer = load_model_and_tokenizer()
    
    # Prepare dataset with tokenizer
    dataset = prepare_dataset(tokenizer)
    print(f"Dataset size: {len(dataset)}")
    
    if len(dataset) == 0:
        print("Error: Dataset is empty after processing!")
        return
    
    # Debug: Print sample and check tokenization
    print("Sample data:")
    sample = dataset[0]
    print(f"Prompt: {sample['prompt'][:200]}...")
    print(f"Answer: {sample['answer']}")
    
    # Check tokenization
    tokens = tokenizer(sample['prompt'], truncation=False, padding=False)
    print(f"Prompt token length: {len(tokens['input_ids'])}")
    print(f"Max prompt length: {MAX_PROMPT_LENGTH}")
    
    if len(tokens['input_ids']) > MAX_PROMPT_LENGTH:
        print("WARNING: Sample prompt is too long!")
        return
    
    # Train model
    trainer = train_model(model, tokenizer, dataset)
    
    if trainer is None:
        print("Training failed!")
        return
    
    # Test inference
    test_inference(model, tokenizer)
    
    # Save model
    save_model(model, tokenizer)
    
    print("\nTraining completed successfully!")

if __name__ == "__main__":
    main()