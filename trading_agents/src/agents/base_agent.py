from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field
from datetime import datetime
from openai import OpenAI
from src.config import Config
import os
import time
import random
import re

@dataclass
class AgentResponse:
    """Standardized agent response format"""
    agent_name: str
    analysis: str
    recommendation: str  # BUY, SELL, HOLD
    confidence: float  # 0.0 to 1.0
    reasoning: List[str]
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "agent_name": self.agent_name,
            "analysis": self.analysis,
            "recommendation": self.recommendation,
            "confidence": self.confidence,
            "reasoning": self.reasoning,
            "timestamp": self.timestamp.isoformat(),
            "metadata": self.metadata
        }

class BaseAgent(ABC):
    """Base class for all trading agents"""
    
    def __init__(self, name: str, role: str, model: str = None):
        self.name = name
        self.role = role
        
        self.model = model or os.getenv("MODEL_NAME", "gemini-2.5-flash")
        
        self.client = OpenAI(
            api_key=Config.GEMINI_API_KEY,
            base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
        )
        
        self.conversation_history: List[Dict[str, str]] = []
   
    @abstractmethod
    def get_system_prompt(self) -> str:
        """Return the system prompt defining the agent's role"""
        pass
    
    @abstractmethod
    def analyze(self, data: Dict[str, Any]) -> AgentResponse:
        """Perform analysis and return structured response"""
        pass
    
    def _call_llm(self, user_message: str, system_prompt: Optional[str] = None) -> str:
        """Make a call to the LLM with exponential backoff for 503/429 errors"""
        messages = [{"role": "system", "content": system_prompt or self.get_system_prompt()}]
        messages.extend(self.conversation_history[-4:])
        messages.append({"role": "user", "content": user_message})
        
        max_retries = 5
        for attempt in range(max_retries):
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=Config.TEMPERATURE,
                    max_tokens=Config.MAX_TOKENS
                )
                assistant_message = response.choices[0].message.content
                
                # Update history only on success
                self.conversation_history.append({"role": "user", "content": user_message})
                self.conversation_history.append({"role": "assistant", "content": assistant_message})
                
                return assistant_message

            except Exception as e:
                # Check for 503 (Overloaded) or 429 (Rate Limit)
                error_msg = str(e).lower()
                if "503" in error_msg or "overloaded" in error_msg or "429" in error_msg:
                    # Exponential backoff: 2s, 4s, 8s, 16s... plus jitter
                    wait_time = (2 ** attempt) + random.uniform(0, 1)
                    print(f" {self.name}: Model overloaded or rate limited. Retrying in {wait_time:.1f}s...")
                    time.sleep(wait_time)
                else:
                    # If it's a different error, raise it immediately
                    raise e
        
        raise Exception(f" {self.name}: Failed after {max_retries} attempts due to model overload.")
    
    def parse_recommendation(self, text: str) -> tuple[str, float]:
        """Extract recommendation and confidence from LLM response with improved parsing"""
        text_upper = text.upper()
        
        # Extract recommendation
        if "STRONG BUY" in text_upper or "STRONGBUY" in text_upper:
            recommendation = "STRONG BUY"
        elif "STRONG SELL" in text_upper or "STRONGSELL" in text_upper:
            recommendation = "STRONG SELL"
        elif "BUY" in text_upper:
            recommendation = "BUY"
        elif "SELL" in text_upper:
            recommendation = "SELL"
        else:
            recommendation = "HOLD"
        
        # Extract confidence with multiple pattern matching
        confidence = None  # Start with None to detect if we found anything
        
        # Pattern 1: Just find any number followed by % near "confidence"
        pattern1 = re.findall(r'confidence[:\s\*]*(?:level)?[:\s\*]*(\d+(?:\.\d+)?)\s*%', text_upper)
        if pattern1:
            # Take the last occurrence (most specific)
            conf_value = float(pattern1[-1])
            confidence = conf_value / 100 if conf_value > 1 else conf_value
        else:
            # Pattern 2: "75% confidence" or "confidence of 75%"
            pattern2 = re.search(r'(\d+(?:\.\d+)?)\s*%\s*confidence', text_upper)
            if pattern2:
                conf_value = float(pattern2.group(1))
                confidence = conf_value / 100 if conf_value > 1 else conf_value
            else:
                # Pattern 3: "Confidence: 0.75" (decimal format)
                pattern3 = re.search(r'confidence[:\s]+(0?\.\d+)', text_upper)
                if pattern3:
                    confidence = float(pattern3.group(1))
                else:
                    # Pattern 4: Any percentage in the first 1000 chars (last resort)
                    pattern4 = re.findall(r'(\d+)\s*%', text_upper[:1000])
                    if pattern4:
                        # Find percentages that look like confidence (40-100 range)
                        valid_confidences = [float(p) for p in pattern4 if 30 <= float(p) <= 100]
                        if valid_confidences:
                            # Take the first valid one
                            conf_value = valid_confidences[0]
                            confidence = conf_value / 100
        
        # If no confidence found, use default
        if confidence is None:
            confidence = 0.5
        
        # Ensure confidence is in valid range
        confidence = max(0.0, min(1.0, confidence))
        
        return recommendation, confidence
    
    def debate_response(self, opposing_view: str, original_analysis: str) -> str:
        """Respond to an opposing viewpoint"""
        prompt = f"""
        Your original analysis was:
        {original_analysis}
        
        Another agent has presented the following opposing view:
        {opposing_view}
        
        Please provide a reasoned response. You may:
        1. Defend your position with additional evidence
        2. Acknowledge valid points and adjust your view
        3. Explain why you still maintain your original recommendation
        
        Be professional and data-driven.
        
        IMPORTANT: Include your confidence level (0-100%) in your response.
        """
        return self._call_llm(prompt)
    
    def reset_conversation(self):
        """Clear conversation history"""
        self.conversation_history = []