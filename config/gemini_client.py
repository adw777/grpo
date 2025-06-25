from google import genai
from google.genai import types
from pydantic import BaseModel
from typing import List, Optional
import logging
from dotenv import load_dotenv
import os
load_dotenv()

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
        self.client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
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