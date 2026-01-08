import time
from typing import Dict, Any, List
from src.agents.base_agent import BaseAgent, AgentResponse
from src.config import Config

class RiskManager(BaseAgent):
    """Agent specialized in risk assessment and portfolio management"""
    
    def __init__(self):
        super().__init__(
            name="Risk Manager",
            role="Expert in risk assessment, position sizing, and portfolio protection"
        )
    
    def get_system_prompt(self) -> str:
        return """You are an expert Risk Manager with expertise in:
- Portfolio risk assessment
- Position sizing and capital allocation
- Stop-loss and take-profit strategies
- Volatility analysis
- Correlation and diversification
- Maximum drawdown analysis
- Risk/reward ratio evaluation
- Market risk and systematic risk

Your responsibilities:
1. Assess the risk level of proposed trades
2. Recommend appropriate position sizes
3. Set stop-loss and take-profit levels
4. Evaluate overall portfolio risk
5. Identify potential tail risks
6. Ensure trades align with risk management principles

You provide:
- Risk rating (Low/Medium/High/Very High)
- Recommended position size (% of portfolio)
- Stop-loss level (specific price)
- Take-profit targets (specific prices)
- Risk/reward ratio
- Maximum acceptable loss

CRITICAL: Your stop-loss must ALWAYS be BELOW current price for longs and ABOVE for shorts.
Position size must reflect the actual trade direction and risk level.

Be conservative and prioritize capital preservation."""

    def analyze(self, data: Dict[str, Any]) -> AgentResponse:
        """Perform risk analysis"""
        ticker = data['ticker']
        stats = data['statistics']
        trading_decision = data.get('trading_decision', 'HOLD')
        other_recommendations = data.get('other_recommendations', [])
        
        prompt = self._create_analysis_prompt(ticker, stats, trading_decision, other_recommendations)
        response = self._call_llm(prompt)
        
        recommendation, confidence = self.parse_recommendation(response)
        reasoning = self._extract_reasoning(response)
        
        # Calculate risk metrics using BOTH LLM output and trading decision
        risk_metrics = self._calculate_risk_metrics(stats, trading_decision, response)
        
        return AgentResponse(
            agent_name=self.name,
            analysis=response,
            recommendation=recommendation,
            confidence=confidence,
            reasoning=reasoning,
            metadata={
                'risk_level': risk_metrics['risk_level'],
                'position_size': risk_metrics['position_size'],
                'stop_loss_price': risk_metrics['stop_loss_price'],
                'take_profit_price': risk_metrics['take_profit_price'],
                'risk_reward_ratio': risk_metrics['risk_reward_ratio'],
                'max_loss_dollars': risk_metrics['max_loss_dollars']
            }
        )
    
    def _create_analysis_prompt(self, ticker: str, stats: Dict, trading_decision: str, other_recs: List) -> str:
        """Create risk analysis prompt"""
        
        # Summarize other agents' recommendations
        agent_summary = "TRADING AGENTS' RECOMMENDATIONS:\n"
        if other_recs:
            for rec in other_recs:
                agent_summary += f"- {rec['agent_name']}: {rec['recommendation']} (Confidence: {rec['confidence']:.0%})\n"
        else:
            agent_summary += f"Head Trader Decision: {trading_decision}\n"
        
        prompt = f"""Perform risk assessment for proposed {ticker} trade:

TRADING DECISION: {trading_decision}

MARKET DATA:
- Current Price: ${stats['current_price']:.2f}
- Volatility (Annualized): {stats['volatility']:.2f}%
- 30-Day Return: {stats['30d_return']:.2f}%
- 90-Day Return: {stats['90d_return']:.2f}%
- Sharpe Ratio: {stats.get('sharpe_ratio', 0):.2f}

{agent_summary}

RISK MANAGEMENT PARAMETERS:
- Max Position Size: {Config.MAX_POSITION_SIZE * 100:.0f}% of portfolio
- Default Stop Loss: {Config.STOP_LOSS_PERCENT * 100:.0f}% from entry
- Default Take Profit: {Config.TAKE_PROFIT_PERCENT * 100:.0f}% from entry

RISK ASSESSMENT TASKS:
1. Evaluate overall risk level (Low/Medium/High/Very High) considering:
   - Current volatility ({stats['volatility']:.1f}%)
   - Recent performance trend
   - Market conditions

2. For {trading_decision} recommendation, provide:
   - Appropriate position size (adjust based on risk level)
   - SPECIFIC stop-loss price (must be BELOW ${stats['current_price']:.2f} for long positions)
   - SPECIFIC take-profit price (must be ABOVE ${stats['current_price']:.2f} for long positions)
   - Risk/reward ratio (take-profit distance / stop-loss distance)

3. Risk considerations:
   - What could go wrong?
   - Maximum acceptable loss
   - Exit strategy

CRITICAL REQUIREMENTS:
- If HOLD: position_size = 0%, no stop loss needed
- If BUY/STRONG BUY: position_size > 0%, stop_loss < current_price, take_profit > current_price
- If SELL/STRONG SELL (shorting): position_size > 0%, stop_loss > current_price, take_profit < current_price
- High volatility = reduce position size
- Low confidence = reduce position size

Provide specific numbers and clear reasoning."""

        return prompt
    
    def _calculate_risk_metrics(self, stats: Dict, trading_decision: str, llm_response: str) -> Dict:
        """Calculate risk metrics with proper logic"""
        current_price = stats['current_price']
        volatility = stats['volatility']
        
        # Parse LLM recommendations if available
        import re
        
        # Try to extract position size from LLM response
        position_match = re.search(r'position.*?(\d+(?:\.\d+)?)\s*%', llm_response.lower())
        llm_position_pct = float(position_match.group(1)) / 100 if position_match else None
        
        # Try to extract stop loss price
        stop_match = re.search(r'stop[-\s]?loss.*?\$?\s*(\d+(?:\.\d+)?)', llm_response.lower())
        llm_stop_price = float(stop_match.group(1)) if stop_match else None
        
        # Try to extract take profit price
        take_match = re.search(r'take[-\s]?profit.*?\$?\s*(\d+(?:\.\d+)?)', llm_response.lower())
        llm_take_price = float(take_match.group(1)) if take_match else None
        
        # Determine risk level based on volatility
        if volatility < 15:
            risk_level = "Low"
            volatility_adjustment = 1.0
        elif volatility < 25:
            risk_level = "Medium"
            volatility_adjustment = 0.75
        elif volatility < 40:
            risk_level = "High"
            volatility_adjustment = 0.5
        else:
            risk_level = "Very High"
            volatility_adjustment = 0.3
        
        # Calculate position size and stop loss based on trading decision
        if "BUY" in trading_decision.upper():
            # LONG POSITION
            base_position = Config.MAX_POSITION_SIZE * volatility_adjustment
            
            # Use LLM position if reasonable, otherwise use calculated
            if llm_position_pct and 0.01 <= llm_position_pct <= Config.MAX_POSITION_SIZE:
                position_size = llm_position_pct
            else:
                position_size = base_position
            
            # Stop loss BELOW current price
            if llm_stop_price and llm_stop_price < current_price * 0.99:
                stop_loss_price = llm_stop_price
            else:
                stop_loss_price = current_price * (1 - Config.STOP_LOSS_PERCENT)
            
            # Take profit ABOVE current price
            if llm_take_price and llm_take_price > current_price * 1.01:
                take_profit_price = llm_take_price
            else:
                take_profit_price = current_price * (1 + Config.TAKE_PROFIT_PERCENT)
                
        elif "SELL" in trading_decision.upper():
            # SHORT POSITION (if your system supports it)
            base_position = Config.MAX_POSITION_SIZE * volatility_adjustment * 0.7  # Reduced for shorts
            
            if llm_position_pct and 0.01 <= llm_position_pct <= Config.MAX_POSITION_SIZE:
                position_size = llm_position_pct
            else:
                position_size = base_position
            
            # Stop loss ABOVE current price (short position)
            if llm_stop_price and llm_stop_price > current_price * 1.01:
                stop_loss_price = llm_stop_price
            else:
                stop_loss_price = current_price * (1 + Config.STOP_LOSS_PERCENT)
            
            # Take profit BELOW current price (short position)
            if llm_take_price and llm_take_price < current_price * 0.99:
                take_profit_price = llm_take_price
            else:
                take_profit_price = current_price * (1 - Config.TAKE_PROFIT_PERCENT)
                
        else:  # HOLD
            position_size = 0.0
            stop_loss_price = None  # No position, no stop loss
            take_profit_price = None
        
        # Calculate risk/reward ratio
        if position_size > 0 and stop_loss_price:
            risk_distance = abs(current_price - stop_loss_price)
            reward_distance = abs(take_profit_price - current_price) if take_profit_price else 0
            risk_reward_ratio = reward_distance / risk_distance if risk_distance > 0 else 0
            
            # Calculate max loss in dollars (assuming $10,000 portfolio)
            portfolio_value = 10000
            max_loss_dollars = portfolio_value * position_size * (risk_distance / current_price)
        else:
            risk_reward_ratio = 0
            max_loss_dollars = 0
        
        return {
            'risk_level': risk_level,
            'position_size': f"{position_size * 100:.1f}%",
            'stop_loss_price': stop_loss_price,
            'take_profit_price': take_profit_price,
            'risk_reward_ratio': f"{risk_reward_ratio:.2f}" if risk_reward_ratio > 0 else "N/A",
            'max_loss_dollars': f"${max_loss_dollars:.2f}"
        }
        
    def _extract_reasoning(self, response: str) -> list:
        """Extract key reasoning points"""
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