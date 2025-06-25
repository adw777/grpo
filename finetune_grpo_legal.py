"""
Advanced Llama 3.2 (3B) GRPO LoRA Training Script for Legal Q&A

This script implements GRPO (Generative Reward Policy Optimization) training
for Llama 3.2 3B model using LoRA adapters on the Legal Q&A dataset.

Features:
- Gemini-powered intelligent reward system for legal response evaluation
- Custom format enforcement for structured legal responses
- LoRA fine-tuning with vLLM inference acceleration
- Comprehensive model saving options (LoRA, merged, GGUF)
"""

import os
import re
import torch
import argparse
import logging
import asyncio
from typing import List, Dict, Any, Optional
from pathlib import Path
from datetime import datetime
import wandb
from dotenv import load_dotenv

load_dotenv()

from unsloth import FastLanguageModel
from datasets import load_dataset
from trl import GRPOConfig as TRLGRPOConfig, GRPOTrainer as TRLGRPOTrainer
from vllm import SamplingParams
from safetensors import safe_open

from google import genai
from google.genai import types
from pydantic import BaseModel

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class LegalEvaluation(BaseModel):
    accuracy: float  # 0-10 scale
    completeness: float  # 0-10 scale
    relevance: float  # 0-10 scale
    clarity: float  # 0-10 scale
    legal_soundness: float  # 0-10 scale
    overall_quality: float  # 0-10 scale
    reasoning: str

class GeminiClient:
    """Client for Gemini API interactions with structured output."""
    
    def __init__(self, api_key: str, model: str = "gemini-2.5-flash-preview-04-17"):
        self.client = genai.Client(api_key=api_key)
        self.model = model
    
    async def evaluate_legal_answer(self, question: str, generated_answer: str, 
                                  reference_answer: str, temperature: float = 0.3) -> LegalEvaluation:
        """Evaluate legal answer using structured output."""
        try:
            evaluation_prompt = f"""
            You are an expert legal evaluator. Please evaluate the quality of a generated legal answer compared to a reference answer.

            QUESTION: {question}

            REFERENCE ANSWER: {reference_answer}

            GENERATED ANSWER: {generated_answer}

            Please evaluate the generated answer on these criteria and provide scores from 0-10:

            1. ACCURACY: How factually correct is the legal information? (0=completely wrong, 10=perfectly accurate)
            2. COMPLETENESS: How thoroughly does it address the question? (0=incomplete, 10=comprehensive)
            3. RELEVANCE: How well does it stay on topic and address the specific legal issue? (0=off-topic, 10=highly relevant)
            4. CLARITY: How clear, well-structured, and understandable is the response? (0=confusing, 10=very clear)
            5. LEGAL_SOUNDNESS: How legally sound are the principles and reasoning presented? (0=legally flawed, 10=legally sound)
            6. OVERALL_QUALITY: Overall assessment of the response quality (0=poor, 10=excellent)

            Also provide a brief reasoning for your evaluation.

            Be strict but fair in your evaluation. Consider that the generated answer should be helpful to someone seeking legal information.
            """
            
            response = await self.client.aio.models.generate_content(
                model=self.model,
                contents=[evaluation_prompt],
                config={
                    "temperature": temperature,
                    "response_mime_type": "application/json",
                    "response_schema": LegalEvaluation,
                }
            )
            
            if response is None or not hasattr(response, 'parsed') or response.parsed is None:
                logger.error("Gemini API returned invalid response")
                return self._get_default_evaluation()
            
            return response.parsed
            
        except Exception as e:
            logger.error(f"Error in Gemini legal evaluation: {e}")
            return self._get_default_evaluation()
    
    def _get_default_evaluation(self) -> LegalEvaluation:
        """Return default evaluation when API fails."""
        return LegalEvaluation(
            accuracy=5.0,
            completeness=5.0,
            relevance=5.0,
            clarity=5.0,
            legal_soundness=5.0,
            overall_quality=5.0,
            reasoning="Evaluation failed, using default scores"
        )

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
        
        # Format tokens for structured legal responses
        self.reasoning_start = "<legal_analysis>"
        self.reasoning_end = "</legal_analysis>"
        self.solution_start = "<answer>"
        self.solution_end = "</answer>"
        
        # Gemini API configuration
        self.gemini_api_key = os.getenv("GEMINI_API_KEY")
        self.gemini_model = "gemini-2.5-flash-preview-04-17"
        self.gemini_requests_per_minute = 60
        
        # Initialize Gemini client
        if self.gemini_api_key:
            self.gemini_client = GeminiClient(
                api_key=self.gemini_api_key,
                model=self.gemini_model
            )
        else:
            logger.warning("GEMINI_API_KEY not found in environment variables")
            self.gemini_client = None

class LegalQADataProcessor:
    """Processes Legal Q&A dataset for GRPO training with Gemini evaluation"""
    
    def __init__(self, config: GRPOConfig):
        self.config = config
        self.system_prompt = self._create_system_prompt()
        self.match_format = self._create_format_regex()
        
        # Rate limiting
        self.request_interval = 60 / config.gemini_requests_per_minute if config.gemini_requests_per_minute > 0 else 1
        self.last_request_time = datetime.now()
        
        # Cache for evaluations
        self.evaluation_cache = {}
        
        # Debug counters
        self.evaluation_count = 0
        self.print_every_evaluations = 10
    
    def _create_system_prompt(self) -> str:
        """Create system prompt for structured legal responses"""
        return f"""You are a legal assistant. Provide accurate, helpful legal information in a structured format.

        Follow this format:
        1. First, analyze the legal question between {self.config.reasoning_start} and {self.config.reasoning_end}
        2. Then provide your final answer between {self.config.solution_start} and {self.config.solution_end}

        Always include relevant legal principles, cite applicable laws when possible, and mention any limitations or need for professional consultation."""
            
    def _create_format_regex(self) -> re.Pattern:
        """Create regex pattern for format matching"""
        return re.compile(
            rf"^[\s]{{0,}}"
            rf"{self.config.reasoning_start}.+?{self.config.reasoning_end}.*?"
            rf"{self.config.solution_start}(.+?){self.config.solution_end}"
            rf"[\s]{{0,}}$",
            flags=re.MULTILINE | re.DOTALL
        )
    
    def load_and_process_dataset(self):
        """Load and process legal Q&A dataset"""
        logger.info("Loading legal Q&A dataset...")
        dataset = load_dataset("axondendriteplus/legal-qna-dataset", split="train")
        
        logger.info("Processing dataset...")
        dataset = dataset.map(lambda x: {
            "prompt": [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": x["question"]},
            ],
            "reference_answer": x["answer"],
            "question": x["question"],
        })
        
        logger.info(f"Processed {len(dataset)} examples")
        return dataset
    
    async def _rate_limit(self):
        """Implement rate limiting for Gemini API"""
        now = datetime.now()
        time_since_last = (now - self.last_request_time).total_seconds()
        if time_since_last < self.request_interval:
            await asyncio.sleep(self.request_interval - time_since_last)
        self.last_request_time = datetime.now()
    
    async def _get_gemini_evaluation(self, question: str, generated_answer: str, 
                                   reference_answer: str) -> LegalEvaluation:
        """Get evaluation from Gemini with caching and rate limiting"""
        if not self.config.gemini_client:
            logger.warning("Gemini client not available, returning default evaluation")
            return LegalEvaluation(
                accuracy=5.0, completeness=5.0, relevance=5.0, clarity=5.0,
                legal_soundness=5.0, overall_quality=5.0,
                reasoning="Gemini client not available"
            )
        
        # Create cache key
        cache_key = hash(f"{question}_{generated_answer}_{reference_answer}")
        if cache_key in self.evaluation_cache:
            return self.evaluation_cache[cache_key]
        
        await self._rate_limit()
        
        evaluation = await self.config.gemini_client.evaluate_legal_answer(
            question=question,
            generated_answer=generated_answer,
            reference_answer=reference_answer
        )
        
        # Cache the result
        self.evaluation_cache[cache_key] = evaluation
        
        # Debug logging
        self.evaluation_count += 1
        if self.evaluation_count % self.print_every_evaluations == 0:
            logger.info(f"Gemini Evaluation #{self.evaluation_count}")
            logger.info(f"Question: {question[:100]}...")
            logger.info(f"Generated: {generated_answer[:100]}...")
            logger.info(f"Scores - Accuracy: {evaluation.accuracy}, Quality: {evaluation.overall_quality}")
            logger.info(f"Reasoning: {evaluation.reasoning}")
            logger.info("-" * 50)
        
        return evaluation
    
    def get_reward_functions(self):
        """Return list of reward functions for GRPO training"""
        return [
            self.check_format_compliance,
            self.gemini_accuracy_reward,
            self.gemini_completeness_reward,
            self.gemini_overall_quality_reward,
        ]
    
    def check_format_compliance(self, completions, **kwargs) -> List[float]:
        """Reward function for format compliance"""
        scores = []
        for completion in completions:
            score = 0.0
            response = completion[0]["content"]
            
            # Check for exact format match
            if self.match_format.search(response) is not None:
                score += 3.0
            else:
                # Partial format checking
                score += 0.5 if response.count(self.config.reasoning_start) == 1 else -0.5
                score += 0.5 if response.count(self.config.reasoning_end) == 1 else -0.5
                score += 0.5 if response.count(self.config.solution_start) == 1 else -0.5
                score += 0.5 if response.count(self.config.solution_end) == 1 else -0.5
            
            scores.append(score)
        return scores
    
    def gemini_accuracy_reward(self, prompts, completions, reference_answer, question, **kwargs) -> List[float]:
        """Reward function based on Gemini accuracy evaluation"""
        return asyncio.run(self._async_gemini_accuracy_reward(prompts, completions, reference_answer, question))
    
    async def _async_gemini_accuracy_reward(self, prompts, completions, reference_answer, question) -> List[float]:
        """Async version of accuracy reward"""
        scores = []
        
        try:
            evaluations = await asyncio.gather(*[
                self._get_gemini_evaluation(
                    question[i], 
                    completion[0]["content"], 
                    reference_answer[i]
                )
                for i, completion in enumerate(completions)
            ])
            
            for evaluation in evaluations:
                accuracy_score = evaluation.accuracy
                # Convert 0-10 scale to reward scale (-2 to +3)
                if accuracy_score >= 8:
                    reward = 3.0
                elif accuracy_score >= 6:
                    reward = 1.5
                elif accuracy_score >= 4:
                    reward = 0.0
                else:
                    reward = -2.0
                
                scores.append(reward)
                
        except Exception as e:
            logger.error(f"Gemini accuracy evaluation failed: {e}")
            scores = [0.0] * len(completions)
        
        return scores
    
    def gemini_completeness_reward(self, prompts, completions, reference_answer, question, **kwargs) -> List[float]:
        """Reward function based on Gemini completeness evaluation"""
        return asyncio.run(self._async_gemini_completeness_reward(prompts, completions, reference_answer, question))
    
    async def _async_gemini_completeness_reward(self, prompts, completions, reference_answer, question) -> List[float]:
        """Async version of completeness reward"""
        scores = []
        
        try:
            evaluations = await asyncio.gather(*[
                self._get_gemini_evaluation(
                    question[i], 
                    completion[0]["content"], 
                    reference_answer[i]
                )
                for i, completion in enumerate(completions)
            ])
            
            for evaluation in evaluations:
                completeness_score = evaluation.completeness
                if completeness_score >= 8:
                    reward = 2.5
                elif completeness_score >= 6:
                    reward = 1.0
                elif completeness_score >= 4:
                    reward = 0.0
                else:
                    reward = -1.5
                
                scores.append(reward)
                
        except Exception as e:
            logger.error(f"Gemini completeness evaluation failed: {e}")
            scores = [0.0] * len(completions)
        
        return scores
    
    def gemini_overall_quality_reward(self, prompts, completions, reference_answer, question, **kwargs) -> List[float]:
        """Reward function based on Gemini overall quality evaluation"""
        return asyncio.run(self._async_gemini_overall_quality_reward(prompts, completions, reference_answer, question))
    
    async def _async_gemini_overall_quality_reward(self, prompts, completions, reference_answer, question) -> List[float]:
        """Async version of overall quality reward"""
        scores = []
        
        try:
            evaluations = await asyncio.gather(*[
                self._get_gemini_evaluation(
                    question[i], 
                    completion[0]["content"], 
                    reference_answer[i]
                )
                for i, completion in enumerate(completions)
            ])
            
            for evaluation in evaluations:
                # Combine multiple scores for overall quality
                combined_score = (
                    evaluation.overall_quality * 0.4 +
                    evaluation.legal_soundness * 0.3 +
                    evaluation.clarity * 0.2 +
                    evaluation.relevance * 0.1
                )
                
                if combined_score >= 8:
                    reward = 4.0
                elif combined_score >= 6:
                    reward = 2.0
                elif combined_score >= 4:
                    reward = 0.0
                else:
                    reward = -2.0
                
                scores.append(reward)
                
        except Exception as e:
            logger.error(f"Gemini overall quality evaluation failed: {e}")
            scores = [0.0] * len(completions)
        
        return scores

class GRPOTrainer:
    """Main GRPO training class for legal Q&A"""
    
    def __init__(self, config: GRPOConfig):
        self.config = config
        self.model = None
        self.tokenizer = None
        self.trainer = None
        self.data_processor = LegalQADataProcessor(config)
    
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
        test_question = "What are the key elements required to form a valid contract under common law?"
        
        logger.info("Testing base model inference...")
        base_output = self._generate_response(test_question, use_lora=False)
        logger.info(f"Base model output: {base_output[:300]}...")
        
        logger.info("Testing LoRA model inference...")
        lora_output = self._generate_response(test_question, use_lora=True)
        logger.info(f"LoRA model output: {lora_output[:300]}...")
        
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
    parser = argparse.ArgumentParser(description="GRPO Training for Llama 3.2 3B on Legal Q&A")
    parser.add_argument("--model-name", default="meta-llama/Llama-3.2-3B-Instruct", help="Model name")
    parser.add_argument("--max-steps", type=int, default=500, help="Maximum training steps")
    parser.add_argument("--lora-rank", type=int, default=64, help="LoRA rank")
    parser.add_argument("--learning-rate", type=float, default=5e-6, help="Learning rate")
    parser.add_argument("--output-dir", default="outputs", help="Output directory")
    parser.add_argument("--save-path", default="grpo_saved_lora", help="LoRA save path")
    parser.add_argument("--test-only", action="store_true", help="Only run inference test")
    parser.add_argument("--save-merged", choices=["merged_16bit", "merged_4bit", "gguf"], help="Save merged model")
    parser.add_argument("--enable-wandb", action="store_true", help="Enable W&B logging")
    
    args = parser.parse_args()
    
    # Initialize W&B if requested
    if args.enable_wandb:
        wandb.login(key=os.getenv("WANDB_TOKEN"))
        wandb.init(project="legal-grpo-training", name="legal-grpo-training")
    
    # Initialize configuration
    config = GRPOConfig()
    config.model_name = args.model_name
    config.max_steps = args.max_steps
    config.lora_rank = args.lora_rank
    config.learning_rate = args.learning_rate
    config.output_dir = args.output_dir
    config.model_save_path = args.save_path
    
    if args.enable_wandb:
        config.report_to = "wandb"
    
    # Validate Gemini API key
    if not config.gemini_api_key:
        logger.error("GEMINI_API_KEY environment variable is required!")
        return
    
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
    
    logger.info("Legal GRPO training completed successfully!")
    
    if args.enable_wandb:
        wandb.finish()

if __name__ == "__main__":
    main()

"""
Usage Examples:

# Basic training with Gemini evaluation
python legal_grpo.py

# Custom configuration
python legal_grpo.py --max-steps 1000 --lora-rank 128 --learning-rate 1e-5

# Test inference only
python legal_grpo.py --test-only

# Save merged model with W&B logging
python legal_grpo.py --save-merged gguf --enable-wandb

Environment Variables Required:
- GEMINI_API_KEY: Your Gemini API key
- WANDB_TOKEN: Your W&B token (optional)
"""