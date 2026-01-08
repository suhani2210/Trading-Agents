from typing import Dict, Any
from src.agents.base_agent import BaseAgent, AgentResponse
import json
import time

class TechnicalAnalyst(BaseAgent):
    """Agent specialized in technical analysis"""
    
    def __init__(self):
        super().__init__(
            name="Technical Analyst",
            role="Expert in chart patterns, technical indicators, and price action analysis"
        )
    
    def get_system_prompt(self) -> str:
        return """You are an expert Technical Analyst with 15+ years of experience in analyzing stock charts and technical indicators.

Your expertise includes:
- Chart patterns (head and shoulders, double tops/bottoms, triangles, etc.)
- Technical indicators (RSI, MACD, Moving Averages, Bollinger Bands)
- Support and resistance levels
- Volume analysis
- Trend identification
- Momentum analysis

When analyzing stocks, you should:
1. Identify current trend (uptrend, downtrend, sideways)
2. Analyze key technical indicators
3. Identify support and resistance levels
4. Assess momentum and volatility
5. Look for chart patterns
6. Provide a clear BUY/SELL/HOLD recommendation

Always include:
- Your confidence level (0-100%)
- Key technical levels to watch
- Risk assessment based on technical factors
- Entry and exit points if recommending action

Be precise, data-driven, and clear in your reasoning.

CRITICAL CONFIDENCE REQUIREMENTS:
You MUST explicitly state your confidence level in this exact format:
"Confidence: XX%" or "Confidence Level: XX%"

Confidence Guidelines for Technical Analysis:
- 85-100%: Multiple strong technical signals aligned (trend + momentum + volume all confirm)
- 70-84%: Strong technical setup with 2-3 aligned indicators
- 55-69%: Moderate signals, some indicators aligned but with mixed messages
- 40-54%: Weak signals, conflicting indicators, sideways action
- 0-39%: Very unclear technical picture, no clear pattern

Examples of high confidence scenarios:
- Strong uptrend + RSI not overbought + increasing volume + bullish MACD crossover = 85%+
- Price at key support + oversold RSI + bullish reversal pattern = 80%+

Examples of low confidence scenarios:
- Sideways price action + neutral RSI + low volume = 45%
- Mixed signals with some bullish and some bearish indicators = 50%

Always provide your exact confidence percentage in your response."""

    def analyze(self, data: Dict[str, Any]) -> AgentResponse:
        """Perform technical analysis"""
        ticker = data['ticker']
        stats = data['statistics']
        technical = data['technical_snapshot']
        price_data = data['price_data']
        
        # Create analysis prompt
        prompt = self._create_analysis_prompt(ticker, stats, technical, price_data)
        
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
                'technical_indicators': technical,
                'current_price': stats['current_price']
            }
        )
    
    def _create_analysis_prompt(self, ticker: str, stats: Dict, technical: Dict, price_data) -> str:
        """Create detailed analysis prompt with market data"""
        
        # Get trend direction
        latest_price = stats['current_price']
        sma_20 = technical.get('SMA_20')
        sma_50 = technical.get('SMA_50')
        
        trend = "NEUTRAL"
        if sma_20 and sma_50:
            if latest_price > sma_20 > sma_50:
                trend = "STRONG UPTREND"
            elif latest_price > sma_20:
                trend = "UPTREND"
            elif latest_price < sma_20 < sma_50:
                trend = "STRONG DOWNTREND"
            elif latest_price < sma_20:
                trend = "DOWNTREND"
        
        # Format SMA values safely
        sma_20_str = f"${sma_20:.2f}" if sma_20 else "N/A"
        sma_50_str = f"${sma_50:.2f}" if sma_50 else "N/A"
        
        # Format RSI safely
        rsi = technical.get('RSI')
        rsi_str = f"{rsi:.2f}" if rsi else "N/A"
        
        # Format MACD safely
        macd = technical.get('MACD')
        macd_str = f"{macd:.4f}" if macd else "N/A"
        
        macd_signal = technical.get('MACD_signal')
        macd_signal_str = f"{macd_signal:.4f}" if macd_signal else "N/A"
        
        # Format Bollinger Bands safely
        bb_upper = technical.get('BB_upper')
        bb_upper_str = f"${bb_upper:.2f}" if bb_upper else "N/A"
        
        bb_lower = technical.get('BB_lower')
        bb_lower_str = f"${bb_lower:.2f}" if bb_lower else "N/A"
        
        # Determine BB position
        bb_position = "Mid-range"
        if bb_upper and latest_price > bb_upper * 0.98:
            bb_position = "Near upper band"
        elif bb_lower and latest_price < bb_lower * 1.02:
            bb_position = "Near lower band"
        
        prompt = f"""Perform technical analysis for {ticker}:

PRICE ACTION:
- Current Price: ${stats['current_price']:.2f}
- Price Change: {stats['price_change_pct']:.2f}%
- 30-Day Return: {stats['30d_return']:.2f}%
- 90-Day Return: {stats['90d_return']:.2f}%
- Volatility (Annualized): {stats['volatility']:.2f}%

TREND ANALYSIS:
- Current Trend: {trend}
- Price vs SMA(20): ${latest_price:.2f} vs {sma_20_str}
- Price vs SMA(50): ${latest_price:.2f} vs {sma_50_str}

TECHNICAL INDICATORS:
- RSI(14): {rsi_str}
  * <30 = Oversold, >70 = Overbought
- MACD: {macd_str}
- MACD Signal: {macd_signal_str}
- Bollinger Bands:
  * Upper: {bb_upper_str}
  * Lower: {bb_lower_str}
  * Current position: {bb_position}

VOLUME:
- Current Volume: {stats['volume']:,}
- Average Volume: {stats['avg_volume']:,.0f}
- Volume Trend: {'Above average' if stats['volume'] > stats['avg_volume'] else 'Below average'}

Based on this technical data, provide your analysis:

1. What is your technical recommendation? (STRONG BUY/BUY/HOLD/SELL/STRONG SELL)
2. What is your confidence level? (State as "Confidence: XX%")
   - Consider: Are multiple indicators aligned? Is the trend clear? Is volume confirming?
3. What are the key support and resistance levels?
4. What technical patterns do you observe?
5. What are the entry/exit points?
6. What is the risk/reward ratio?

IMPORTANT: You must explicitly state "Confidence: XX%" in your response.
Base your confidence on:
- Signal alignment (do indicators agree?)
- Trend clarity (strong trend = higher confidence)
- Volume confirmation
- Pattern strength

Provide a clear, structured analysis with specific price targets and your exact confidence percentage."""

        return prompt
    
    def _extract_reasoning(self, response: str) -> list:
        """Extract key reasoning points from analysis"""
        reasoning = []
        
        # Look for numbered points or bullet points
        lines = response.split('\n')
        for line in lines:
            line = line.strip()
            if line and (line[0].isdigit() or line.startswith('-') or line.startswith('•')):
                # Clean up the line
                cleaned = line.lstrip('0123456789.-•) ').strip()
                if len(cleaned) > 10:  # Only substantial points
                    reasoning.append(cleaned)
        
        # If no structured points found, take key sentences
        if not reasoning:
            sentences = [s.strip() for s in response.split('.') if len(s.strip()) > 20]
            reasoning = sentences[:5]  # Top 5 sentences
        
        return reasoning[:6]  # Max 6 points