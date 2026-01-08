import time
from typing import Dict, Any
from src.agents.base_agent import BaseAgent, AgentResponse

class SentimentAnalyst(BaseAgent):
    """Agent specialized in market sentiment and news analysis"""
    
    def __init__(self):
        super().__init__(
            name="Sentiment Analyst",
            role="Expert in market sentiment, news analysis, and behavioral finance"
        )
    
    def get_system_prompt(self) -> str:
        return """You are an expert Sentiment Analyst with 15+ years of experience in analyzing market psychology and news impact.

Your expertise includes:
- News sentiment analysis
- Social media and retail sentiment
- Institutional sentiment indicators
- Market psychology and behavioral finance
- Sentiment-driven price movements
- Contrarian indicators

When analyzing stocks, you should:
1. Assess overall market sentiment (bullish/bearish/neutral)
2. Analyze news flow and media coverage
3. Evaluate retail vs institutional sentiment
4. Identify sentiment extremes (euphoria/panic)
5. Consider contrarian signals
6. Provide a clear BUY/SELL/HOLD recommendation

Always include:
- Your confidence level (0-100%)
- Key sentiment drivers
- Risk of sentiment reversal
- Timeframe for sentiment impact

Be insightful, contextual, and clear in your reasoning.

CRITICAL CONFIDENCE REQUIREMENTS:
You MUST explicitly state your confidence level in this exact format:
"Confidence: XX%" or "Confidence Level: XX%"

Confidence Guidelines for Sentiment Analysis:
- 85-100%: Very clear, strong sentiment with multiple confirming signals
  * Example: Overwhelming positive news + strong momentum + institutional buying
  * Example: Extreme negative sentiment at oversold levels (contrarian buy)
- 70-84%: Strong sentiment signals with good confirmation
  * Example: Consistent positive news flow with building momentum
  * Example: Clear negative trend in coverage with bearish positioning
- 55-69%: Moderate sentiment, some mixed signals
  * Example: Positive news but muted market reaction
  * Example: Some negative headlines but no panic selling
- 40-54%: Unclear sentiment, conflicting signals
  * Example: Mixed news with neutral market reaction
  * Example: Sentiment transition period
- 0-39%: Very unclear or rapidly changing sentiment
  * Example: High uncertainty with conflicting narratives
  * Example: Sentiment disconnected from fundamentals

Always provide your exact confidence percentage clearly in your response."""

    def analyze(self, data: Dict[str, Any]) -> AgentResponse:
        """Perform sentiment analysis"""
        ticker = data['ticker']
        stats = data['statistics']
        sentiment_data = data.get('sentiment', {})
        
        # Create analysis prompt
        prompt = self._create_analysis_prompt(ticker, stats, sentiment_data)
        
        # Get LLM response
        response = self._call_llm(prompt)
        
        # Parse response
        recommendation, confidence = self.parse_recommendation(response)
        
        # Extract key reasoning points
        reasoning = self._extract_reasoning(response)
        
        return AgentResponse(
            agent_name=self.name,
            analysis=response,
            recommendation=recommendation,
            confidence=confidence,
            reasoning=reasoning,
            metadata={
                'sentiment_score': sentiment_data.get('overall_sentiment', 0)
            }
        )
    
    def _create_analysis_prompt(self, ticker: str, stats: Dict, sentiment_data: Dict) -> str:
        """Create detailed sentiment analysis prompt"""
        
        # Extract sentiment metrics
        overall_sentiment = sentiment_data.get('overall_sentiment', 0)
        news_sentiment = sentiment_data.get('news_sentiment', 'neutral')
        
        # Determine sentiment description
        if overall_sentiment > 0.3:
            sentiment_desc = "BULLISH"
        elif overall_sentiment < -0.3:
            sentiment_desc = "BEARISH"
        else:
            sentiment_desc = "NEUTRAL"
        
        prompt = f"""Perform sentiment analysis for {ticker}:

PRICE MOMENTUM:
- Current Price: ${stats['current_price']:.2f}
- Price Change: {stats['price_change_pct']:.2f}%
- 30-Day Return: {stats['30d_return']:.2f}%
- 90-Day Return: {stats['90d_return']:.2f}%

MARKET SENTIMENT:
- Overall Sentiment: {sentiment_desc} ({overall_sentiment:.2f})
- News Sentiment: {news_sentiment}
- Volume vs Average: {stats['volume']/stats['avg_volume']:.2f}x

SENTIMENT CONTEXT:
- Is price momentum confirming sentiment?
- Is volume supporting the move?
- Are we at sentiment extremes?

Based on this sentiment data, provide your analysis:

1. What is your sentiment-based recommendation? (STRONG BUY/BUY/HOLD/SELL/STRONG SELL)
2. What is your confidence level? (State as "Confidence: XX%")
   - Consider: How clear and strong is the sentiment? Are signals aligned?
3. What is the current market sentiment toward this stock?
4. Are there any sentiment extremes (euphoria/panic)?
5. What are the key sentiment drivers?
6. Is there contrarian opportunity?
7. What is the sentiment risk (reversal potential)?

IMPORTANT: You must explicitly state "Confidence: XX%" in your response.
Base your confidence on:
- Strength and clarity of sentiment signals
- Alignment between price action and sentiment
- Volume confirmation
- News flow consistency

Examples:
- "Strong positive momentum + high volume + bullish news" = 80%+ confidence
- "Mixed signals with neutral price action" = 50% confidence
- "Sentiment changing rapidly, unclear direction" = 35% confidence

Consider both trend-following and contrarian perspectives.
Provide a clear assessment with your exact confidence percentage."""

        return prompt
    
    def _extract_reasoning(self, response: str) -> list:
        """Extract key reasoning points from analysis"""
        reasoning = []
        
        lines = response.split('\n')
        for line in lines:
            line = line.strip()
            if line and (line[0].isdigit() or line.startswith('-') or line.startswith('•')):
                cleaned = line.lstrip('0123456789.-•) ').strip()
                if len(cleaned) > 10:
                    reasoning.append(cleaned)
        
        if not reasoning:
            sentences = [s.strip() for s in response.split('.') if len(s.strip()) > 20]
            reasoning = sentences[:5]
        
        return reasoning[:6]