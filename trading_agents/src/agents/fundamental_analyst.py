import time
from typing import Dict, Any
from src.agents.base_agent import BaseAgent, AgentResponse

class FundamentalAnalyst(BaseAgent):
    """Agent specialized in fundamental analysis"""
    
    def __init__(self):
        super().__init__(
            name="Fundamental Analyst",
            role="Expert in financial analysis, valuation, and company fundamentals"
        )
    
    def get_system_prompt(self) -> str:
        return """You are an expert Fundamental Analyst with 15+ years of experience in financial analysis and valuation.

Your expertise includes:
- Financial statement analysis
- Valuation metrics (P/E, P/B, P/S, PEG ratios)
- Industry analysis and competitive positioning
- Management quality assessment
- Growth prospects evaluation
- Risk assessment

When analyzing stocks, you should:
1. Evaluate key valuation metrics
2. Assess financial health and profitability
3. Analyze growth prospects
4. Compare to industry peers
5. Identify fundamental strengths and weaknesses
6. Provide a clear BUY/SELL/HOLD recommendation

Always include:
- Your confidence level (0-100%)
- Key valuation concerns or opportunities
- Risk factors
- Fair value estimate if possible

Be rigorous, analytical, and clear in your reasoning.

CRITICAL CONFIDENCE REQUIREMENTS:
You MUST explicitly state your confidence level in this exact format:
"Confidence: XX%" or "Confidence Level: XX%"

Confidence Guidelines for Fundamental Analysis:
- 85-100%: Clear valuation case with strong fundamentals
  * Example: Trading at 30%+ discount to fair value with solid financials
  * Example: Strong growth + reasonable valuation + industry tailwinds
- 70-84%: Good fundamental case with minor concerns
  * Example: Attractive valuation but some execution risks
  * Example: Strong fundamentals but moderately expensive
- 55-69%: Mixed fundamentals, neutral valuation
  * Example: Fair value with uncertain growth outlook
  * Example: Some strengths offset by weaknesses
- 40-54%: Unclear fundamental picture
  * Example: Difficult to value, mixed metrics
  * Example: Transitioning business model with uncertainty
- 0-39%: Very uncertain fundamentals
  * Example: Poor visibility on key metrics
  * Example: Industry headwinds with unclear path forward

Always provide your exact confidence percentage clearly in your response."""

    def analyze(self, data: Dict[str, Any]) -> AgentResponse:
        """Perform fundamental analysis"""
        ticker = data['ticker']
        stats = data['statistics']
        info = data.get('info', {})
        
        # Create analysis prompt
        prompt = self._create_analysis_prompt(ticker, stats, info)
        
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
                'valuation_metrics': {
                    'pe_ratio': info.get('trailingPE'),
                    'forward_pe': info.get('forwardPE'),
                    'peg_ratio': info.get('pegRatio')
                }
            }
        )
    
    def _create_analysis_prompt(self, ticker: str, stats: Dict, info: Dict) -> str:
        """Create detailed fundamental analysis prompt"""
        
        # Extract key metrics safely
        market_cap = info.get('marketCap', 0)
        pe_ratio = info.get('trailingPE')
        forward_pe = info.get('forwardPE')
        peg_ratio = info.get('pegRatio')
        pb_ratio = info.get('priceToBook')
        profit_margin = info.get('profitMargins')
        roe = info.get('returnOnEquity')
        debt_to_equity = info.get('debtToEquity')
        
        # Format metrics
        market_cap_str = f"${market_cap/1e9:.2f}B" if market_cap else "N/A"
        pe_str = f"{pe_ratio:.2f}" if pe_ratio else "N/A"
        fwd_pe_str = f"{forward_pe:.2f}" if forward_pe else "N/A"
        peg_str = f"{peg_ratio:.2f}" if peg_ratio else "N/A"
        pb_str = f"{pb_ratio:.2f}" if pb_ratio else "N/A"
        margin_str = f"{profit_margin*100:.2f}%" if profit_margin else "N/A"
        roe_str = f"{roe*100:.2f}%" if roe else "N/A"
        de_str = f"{debt_to_equity:.2f}" if debt_to_equity else "N/A"
        
        prompt = f"""Perform fundamental analysis for {ticker}:

COMPANY OVERVIEW:
- Ticker: {ticker}
- Market Cap: {market_cap_str}
- Sector: {info.get('sector', 'N/A')}
- Industry: {info.get('industry', 'N/A')}

VALUATION METRICS:
- P/E Ratio (Trailing): {pe_str}
- P/E Ratio (Forward): {fwd_pe_str}
- PEG Ratio: {peg_str}
- Price/Book: {pb_str}

PROFITABILITY & EFFICIENCY:
- Profit Margin: {margin_str}
- Return on Equity: {roe_str}
- Debt/Equity: {de_str}

PERFORMANCE:
- Current Price: ${stats['current_price']:.2f}
- 30-Day Return: {stats['30d_return']:.2f}%
- 90-Day Return: {stats['90d_return']:.2f}%

Based on this fundamental data, provide your analysis:

1. What is your fundamental recommendation? (STRONG BUY/BUY/HOLD/SELL/STRONG SELL)
2. What is your confidence level? (State as "Confidence: XX%")
   - Consider: How clear is the valuation case? How strong are fundamentals?
3. Is the stock undervalued, fairly valued, or overvalued?
4. What are the key fundamental strengths?
5. What are the key fundamental risks?
6. What is your fair value estimate or target price range?

IMPORTANT: You must explicitly state "Confidence: XX%" in your response.
Base your confidence on:
- Clarity of valuation (clear discount/premium = higher confidence)
- Quality of fundamentals (strong metrics = higher confidence)
- Certainty of growth outlook
- Industry positioning

Examples:
- "Trading at P/E of 12 vs industry average of 18 with solid growth" = 80%+ confidence
- "Fair valuation with mixed growth signals" = 55% confidence
- "Difficult to value due to business transition" = 40% confidence

Provide a clear, analytical assessment with your exact confidence percentage."""

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