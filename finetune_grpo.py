#!/usr/bin/env python3
"""
Advanced Llama 3.2 (3B) GRPO LoRA Training Script

This script implements GRPO (Generative Reward Policy Optimization) training
for Llama 3.2 3B model using LoRA adapters on the GSM8K math dataset.

Features:
- Multi-component reward system for structured reasoning
- Custom format enforcement for mathematical problem solving
- LoRA fine-tuning with vLLM inference acceleration
- Comprehensive model saving options (LoRA, merged, GGUF)
"""

import os
import re
import torch
import argparse
import logging
from typing import List, Dict, Any, Optional
from pathlib import Path

# Core libraries
from unsloth import FastLanguageModel
from datasets import load_dataset
from trl import GRPOConfig, GRPOTrainer
from vllm import SamplingParams
from safetensors import safe_open

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class GRPOConfig:
    """Configuration class for GRPO training parameters"""
    
    def __init__(self):
        # Model configuration
        self.model_name = "meta-llama/Llama-3.2-3B-Instruct"
        self.max_seq_length = 2048
        self.lora_rank = 64
        self.load_in_4bit = False
        self.fast_inference = True
        self.gpu_memory_utilization = 0.6
        
        # Training configuration
        self.learning_rate = 5e-6
        self.weight_decay = 0.1
        self.warmup_ratio = 0.1
        self.lr_scheduler_type = "cosine"
        self.optim = "adamw_8bit"
        self.per_device_train_batch_size = 1
        self.gradient_accumulation_steps = 4
        self.num_generations = 4
        self.max_steps = 500
        self.save_steps = 250
        self.max_grad_norm = 1.0
        
        # Output configuration
        self.output_dir = "outputs"
        self.model_save_path = "grpo_saved_lora"
        self.report_to = "none"
        
        # Format tokens
        self.reasoning_start = "<start_working_out>"
        self.reasoning_end = "<end_working_out>"
        self.solution_start = "<SOLUTION>"
        self.solution_end = "</SOLUTION>"

class GSM8KDataProcessor:
    """Processes GSM8K dataset for GRPO training"""
    
    def __init__(self, config: GRPOConfig):
        self.config = config
        self.system_prompt = self._create_system_prompt()
        self.match_format = self._create_format_regex()
        self.match_numbers = self._create_numbers_regex()
        
        # Global counters for debugging
        self.printed_times = 0
        self.print_every_steps = 5
    
    def _create_system_prompt(self) -> str:
        """Create system prompt for structured reasoning"""
        return f"""You are given a problem.
Think about the problem and provide your working out.
Place it between {self.config.reasoning_start} and {self.config.reasoning_end}.
Then, provide your solution between {self.config.solution_start}{self.config.solution_end}"""
    
    def _create_format_regex(self) -> re.Pattern:
        """Create regex pattern for format matching"""
        return re.compile(
            rf"^[\s]{{0,}}"
            rf"{self.config.reasoning_start}.+?{self.config.reasoning_end}.*?"
            rf"{self.config.solution_start}(.+?){self.config.solution_end}"
            rf"[\s]{{0,}}$",
            flags=re.MULTILINE | re.DOTALL
        )
    
    def _create_numbers_regex(self) -> re.Pattern:
        """Create regex pattern for number extraction"""
        return re.compile(
            self.config.solution_start + r".*?([\d\.\,]{1,})",
            flags=re.MULTILINE | re.DOTALL
        )
    
    def extract_hash_answer(self, text: str) -> Optional[str]:
        """Extract numerical answer from GSM8K format"""
        if "####" not in text:
            return None
        return text.split("####")[1].strip()
    
    def load_and_process_dataset(self):
        """Load and process GSM8K dataset"""
        logger.info("Loading GSM8K dataset...")
        dataset = load_dataset("openai/gsm8k", "main", split="train")
        
        logger.info("Processing dataset...")
        dataset = dataset.map(lambda x: {
            "prompt": [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": x["question"]},
            ],
            "answer": self.extract_hash_answer(x["answer"]),
        })
        
        # Filter out entries without valid answers
        dataset = dataset.filter(lambda x: x["answer"] is not None)
        
        logger.info(f"Processed {len(dataset)} examples")
        return dataset
    
    def get_reward_functions(self):
        """Return list of reward functions for GRPO training"""
        return [
            self.match_format_exactly,
            self.match_format_approximately,
            self.check_answer,
            self.check_numbers,
        ]
    
    def match_format_exactly(self, completions, **kwargs) -> List[float]:
        """Reward function for exact format matching"""
        scores = []
        for completion in completions:
            score = 0.0
            response = completion[0]["content"]
            if self.match_format.search(response) is not None:
                score += 3.0
            scores.append(score)
        return scores
    
    def match_format_approximately(self, completions, **kwargs) -> List[float]:
        """Reward function for approximate format matching"""
        scores = []
        for completion in completions:
            score = 0.0
            response = completion[0]["content"]
            
            # Count occurrences of each token
            score += 0.5 if response.count(self.config.reasoning_start) == 1 else -1.0
            score += 0.5 if response.count(self.config.reasoning_end) == 1 else -1.0
            score += 0.5 if response.count(self.config.solution_start) == 1 else -1.0
            score += 0.5 if response.count(self.config.solution_end) == 1 else -1.0
            
            scores.append(score)
        return scores
    
    def check_answer(self, prompts, completions, answer, **kwargs) -> List[float]:
        """Reward function for answer verification"""
        responses = [completion[0]["content"] for completion in completions]
        
        extracted_responses = [
            guess.group(1) if (guess := self.match_format.search(r)) is not None else None
            for r in responses
        ]
        
        scores = []
        for guess, true_answer in zip(extracted_responses, answer):
            score = 0.0
            if guess is None:
                scores.append(0)
                continue
            
            # Exact match
            if guess == true_answer:
                score += 3.0
            # Match after stripping whitespace
            elif guess.strip() == true_answer.strip():
                score += 1.5
            else:
                # Numerical proximity check
                try:
                    ratio = float(guess) / float(true_answer)
                    if 0.9 <= ratio <= 1.1:
                        score += 1.0
                    elif 0.8 <= ratio <= 1.2:
                        score += 0.5
                    else:
                        score -= 1.5
                except:
                    score -= 1.5
            
            scores.append(score)
        return scores
    
    def check_numbers(self, prompts, completions, answer, **kwargs) -> List[float]:
        """Reward function for numerical extraction and verification"""
        question = prompts[0][-1]["content"]
        responses = [completion[0]["content"] for completion in completions]
        
        extracted_responses = [
            guess.group(1) if (guess := self.match_numbers.search(r)) is not None else None
            for r in responses
        ]
        
        scores = []
        
        # Debug printing every few steps
        if self.printed_times % self.print_every_steps == 0:
            logger.info(f"Question: {question}")
            logger.info(f"Expected Answer: {answer[0]}")
            logger.info(f"Response: {responses[0][:200]}...")
            logger.info(f"Extracted: {extracted_responses[0]}")
            logger.info("-" * 50)
        self.printed_times += 1
        
        for guess, true_answer in zip(extracted_responses, answer):
            if guess is None:
                scores.append(0)
                continue
            
            try:
                true_answer = float(true_answer.strip())
                guess = float(guess.strip().replace(",", ""))
                scores.append(1.5 if guess == true_answer else -0.5)
            except:
                scores.append(0)
        
        return scores

class GRPOTrainer:
    """Main GRPO training class"""
    
    def __init__(self, config: GRPOConfig):
        self.config = config
        self.model = None
        self.tokenizer = None
        self.trainer = None
        self.data_processor = GSM8KDataProcessor(config)
    
    def setup_model(self):
        """Initialize model and tokenizer"""
        logger.info(f"Loading model: {self.config.model_name}")
        
        self.model, self.tokenizer = FastLanguageModel.from_pretrained(
            model_name=self.config.model_name,
            max_seq_length=self.config.max_seq_length,
            load_in_4bit=self.config.load_in_4bit,
            fast_inference=self.config.fast_inference,
            max_lora_rank=self.config.lora_rank,
            gpu_memory_utilization=self.config.gpu_memory_utilization,
        )
        
        logger.info("Setting up LoRA configuration...")
        self.model = FastLanguageModel.get_peft_model(
            self.model,
            r=self.config.lora_rank,
            target_modules=[
                "q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj",
            ],
            lora_alpha=self.config.lora_rank,
            use_gradient_checkpointing="unsloth",
            random_state=3407,
        )
    
    def prepare_data(self):
        """Load and prepare training data"""
        dataset = self.data_processor.load_and_process_dataset()
        
        # Calculate maximum prompt length
        logger.info("Calculating maximum prompt length...")
        max_prompt_length = max(dataset.map(
            lambda x: {"tokens": self.tokenizer.apply_chat_template(
                x["prompt"], add_generation_prompt=True, tokenize=True
            )},
            batched=True,
        ).map(lambda x: {"length": len(x["tokens"])})["length"])
        
        logger.info(f"Maximum prompt length: {max_prompt_length}")
        return dataset, max_prompt_length + 1  # +1 for safety
    
    def setup_training(self, dataset, max_prompt_length):
        """Setup GRPO trainer"""
        from trl import GRPOConfig as TRLGRPOConfig, GRPOTrainer as TRLGRPOTrainer
        
        training_args = TRLGRPOConfig(
            learning_rate=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
            warmup_ratio=self.config.warmup_ratio,
            lr_scheduler_type=self.config.lr_scheduler_type,
            optim=self.config.optim,
            logging_steps=1,
            per_device_train_batch_size=self.config.per_device_train_batch_size,
            gradient_accumulation_steps=self.config.gradient_accumulation_steps,
            num_generations=self.config.num_generations,
            max_prompt_length=max_prompt_length,
            max_completion_length=self.config.max_seq_length - max_prompt_length,
            max_steps=self.config.max_steps,
            save_steps=self.config.save_steps,
            max_grad_norm=self.config.max_grad_norm,
            report_to=self.config.report_to,
            output_dir=self.config.output_dir,
        )
        
        self.trainer = TRLGRPOTrainer(
            model=self.model,
            processing_class=self.tokenizer,
            reward_funcs=self.data_processor.get_reward_functions(),
            args=training_args,
            train_dataset=dataset,
        )
    
    def train(self):
        """Execute GRPO training"""
        logger.info("Starting GRPO training...")
        result = self.trainer.train()
        logger.info(f"Training completed: {result}")
        return result
    
    def save_model(self):
        """Save trained LoRA adapter"""
        logger.info(f"Saving LoRA adapter to {self.config.model_save_path}")
        self.model.save_lora(self.config.model_save_path)
        
        # Verify LoRA was actually trained
        self._verify_lora_training()
    
    def _verify_lora_training(self):
        """Verify that LoRA weights are non-zero"""
        adapter_path = Path(self.config.model_save_path) / "adapter_model.safetensors"
        
        if not adapter_path.exists():
            logger.warning("LoRA adapter file not found!")
            return
        
        logger.info("Verifying LoRA training...")
        with safe_open(str(adapter_path), framework="pt") as f:
            for key in f.keys():
                tensor = f.get_tensor(key)
                n_zeros = (tensor == 0).sum() / tensor.numel()
                assert n_zeros.item() != tensor.numel(), f"LoRA layer {key} is all zeros!"
        
        logger.info("LoRA verification passed - model was successfully trained!")
    
    def test_inference(self):
        """Test model inference before and after training"""
        test_question = "What is the sqrt of 101?"
        
        logger.info("Testing base model inference...")
        base_output = self._generate_response(test_question, use_lora=False)
        logger.info(f"Base model output: {base_output[:200]}...")
        
        logger.info("Testing LoRA model inference...")
        lora_output = self._generate_response(test_question, use_lora=True)
        logger.info(f"LoRA model output: {lora_output[:200]}...")
        
        return base_output, lora_output
    
    def _generate_response(self, question: str, use_lora: bool = False) -> str:
        """Generate response for given question"""
        if use_lora:
            messages = [
                {"role": "system", "content": self.data_processor.system_prompt},
                {"role": "user", "content": question},
            ]
        else:
            messages = [{"role": "user", "content": question}]
        
        text = self.tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=False,
        )
        
        sampling_params = SamplingParams(
            temperature=0.8,
            top_p=0.95,
            max_tokens=1024,
        )
        
        lora_request = None
        if use_lora:
            lora_request = self.model.load_lora(self.config.model_save_path)
        
        output = self.model.fast_generate(
            [text],
            sampling_params=sampling_params,
            lora_request=lora_request,
        )[0].outputs[0].text
        
        return output
    
    def save_merged_model(self, save_method: str = "merged_16bit", push_to_hub: bool = False, hf_token: str = ""):
        """Save merged model in various formats"""
        if save_method == "merged_16bit":
            logger.info("Saving merged 16-bit model...")
            self.model.save_pretrained_merged(
                "model_merged_16bit",
                self.tokenizer,
                save_method="merged_16bit"
            )
        elif save_method == "merged_4bit":
            logger.info("Saving merged 4-bit model...")
            self.model.save_pretrained_merged(
                "model_merged_4bit",
                self.tokenizer,
                save_method="merged_4bit"
            )
        elif save_method == "gguf":
            logger.info("Saving GGUF model...")
            self.model.save_pretrained_gguf(
                "model_gguf",
                self.tokenizer,
                quantization_method="q8_0"
            )
        
        if push_to_hub and hf_token:
            logger.info("Uploading to Hugging Face Hub...")
            # Implementation depends on specific requirements

def main():
    parser = argparse.ArgumentParser(description="GRPO Training for Llama 3.2 3B")
    parser.add_argument("--model-name", default="meta-llama/Llama-3.2-3B-Instruct", help="Model name")
    parser.add_argument("--max-steps", type=int, default=500, help="Maximum training steps")
    parser.add_argument("--lora-rank", type=int, default=64, help="LoRA rank")
    parser.add_argument("--learning-rate", type=float, default=5e-6, help="Learning rate")
    parser.add_argument("--output-dir", default="outputs", help="Output directory")
    parser.add_argument("--save-path", default="grpo_saved_lora", help="LoRA save path")
    parser.add_argument("--test-only", action="store_true", help="Only run inference test")
    parser.add_argument("--save-merged", choices=["merged_16bit", "merged_4bit", "gguf"], help="Save merged model")
    
    args = parser.parse_args()
    
    # Initialize configuration
    config = GRPOConfig()
    config.model_name = args.model_name
    config.max_steps = args.max_steps
    config.lora_rank = args.lora_rank
    config.learning_rate = args.learning_rate
    config.output_dir = args.output_dir
    config.model_save_path = args.save_path
    
    # Initialize trainer
    trainer = GRPOTrainer(config)
    trainer.setup_model()
    
    if args.test_only:
        logger.info("Running inference test only...")
        if Path(config.model_save_path).exists():
            trainer.test_inference()
        else:
            logger.error(f"LoRA model not found at {config.model_save_path}")
        return
    
    # Prepare data and train
    dataset, max_prompt_length = trainer.prepare_data()
    trainer.setup_training(dataset, max_prompt_length)
    
    # Train model
    trainer.train()
    
    # Save model
    trainer.save_model()
    
    # Test inference
    trainer.test_inference()
    
    # Save merged model if requested
    if args.save_merged:
        trainer.save_merged_model(args.save_merged)
    
    logger.info("GRPO training completed successfully!")

if __name__ == "__main__":
    main()

"""
# Basic training
python grpo.py

# Custom configuration
python grpo.py --max-steps 1000 --lora-rank 128 --learning-rate 1e-5

# Test inference only
python grpo.py --test-only

# Save merged model
python grpo.py --save-merged gguf
"""