from typing import Dict, Any, List
from src.agents.base_agent import BaseAgent, AgentResponse
import numpy as np
import time

class TraderAgent(BaseAgent):
    """Final decision-making agent with explicit weighting logic"""
    
    def __init__(self):
        super().__init__(
            name="Head Trader",
            role="Senior trader responsible for final trading decisions"
        )
        # Explicit weights for different agents
        self.agent_weights = {
            'Technical Analyst': 0.35,
            'Fundamental Analyst': 0.35,
            'Sentiment Analyst': 0.30
        }
    
    def get_system_prompt(self) -> str:
        return """You are a Senior Head Trader with 20+ years of experience.
Your role is to synthesize input from multiple specialist analysts and make the final trading decision.

CRITICAL: Your decision must be logically consistent with the weighted consensus of analysts.
Do not make extreme recommendations that contradict the majority without exceptional justification.

Provide:
- Final recommendation: STRONG BUY / BUY / HOLD / SELL / STRONG SELL
- Confidence level (0-100%) - this should reflect uncertainty, not just copy analyst confidence
- Clear reasoning that explains how you weighted different viewpoints
- Specific entry/exit strategy
- Key factors and risks

Be decisive but prudent. Always explain your reasoning."""

    def make_decision(
        self, 
        ticker: str,
        agent_responses: List[AgentResponse],
        market_data: Dict[str, Any],
        risk_assessment: AgentResponse = None
    ) -> AgentResponse:
        """Make final trading decision using quantitative + qualitative approach"""
        
        # Separate risk assessment from trading agents
        trading_agents = [r for r in agent_responses if r.agent_name != 'Risk Manager']
        
        # Calculate quantitative consensus FIRST
        quant_decision = self._calculate_quantitative_consensus(trading_agents)
        
        # Get LLM qualitative analysis
        prompt = self._create_decision_prompt(
            ticker, 
            trading_agents, 
            market_data, 
            quant_decision,
            risk_assessment
        )
        llm_response = self._call_llm(prompt)
        
        # Parse LLM recommendation
        llm_recommendation, llm_confidence = self.parse_recommendation(llm_response)
        
        # SANITY CHECK: If LLM contradicts strong quantitative consensus, override
        final_recommendation, final_confidence = self._reconcile_decisions(
            quant_decision,
            llm_recommendation,
            llm_confidence,
            trading_agents
        )
        
        reasoning = self._extract_reasoning(llm_response)
        
        # Add consensus info to reasoning
        reasoning.insert(0, f"Quantitative Consensus: {quant_decision['recommendation']} (Score: {quant_decision['consensus_score']:.2f})")
        reasoning.insert(1, f"Agreement Level: {quant_decision['agreement_level']:.0%}")
        
        return AgentResponse(
            agent_name=self.name,
            analysis=llm_response,
            recommendation=final_recommendation,
            confidence=final_confidence,
            reasoning=reasoning,
            metadata={
                'quantitative_decision': quant_decision,
                'llm_recommendation': llm_recommendation,
                'llm_confidence': llm_confidence,
                'was_overridden': final_recommendation != llm_recommendation,
                'agent_weights': self.agent_weights
            }
        )
    
    def _calculate_quantitative_consensus(self, responses: List[AgentResponse]) -> Dict:
        """Calculate weighted quantitative consensus"""
        
        # Map recommendations to numeric scores
        rec_to_score = {
            'STRONG BUY': 2,
            'BUY': 1,
            'HOLD': 0,
            'SELL': -1,
            'STRONG SELL': -2
        }
        
        score_to_rec = {
            2: 'STRONG BUY',
            1: 'BUY',
            0: 'HOLD',
            -1: 'SELL',
            -2: 'STRONG SELL'
        }
        
        # Calculate weighted consensus score
        weighted_scores = []
        total_weight = 0
        
        for response in responses:
            agent_name = response.agent_name
            weight = self.agent_weights.get(agent_name, 0.33)
            score = rec_to_score.get(response.recommendation, 0)
            
            # Adjust weight by confidence - low confidence means less weight
            adjusted_weight = weight * response.confidence
            weighted_scores.append(score * adjusted_weight)
            total_weight += adjusted_weight
        
        # Calculate consensus
        if total_weight > 0:
            consensus_score = sum(weighted_scores) / total_weight
        else:
            consensus_score = 0
        
        # Map score back to recommendation
        rounded_score = round(consensus_score)
        rounded_score = max(-2, min(2, rounded_score))  # Clamp to valid range
        consensus_rec = score_to_rec[rounded_score]
        
        # Calculate agreement level (how close are the agents?)
        scores = [rec_to_score.get(r.recommendation, 0) for r in responses]
        variance = np.var(scores) if len(scores) > 1 else 0
        max_variance = 4  # Maximum possible variance for [-2, 2] range
        agreement_level = max(0, 1 - (variance / max_variance))
        
        # Calculate consensus confidence (weighted average of agent confidences)
        avg_confidence = sum(r.confidence * self.agent_weights.get(r.agent_name, 0.33) 
                            for r in responses) / sum(self.agent_weights.values())
        
        return {
            'recommendation': consensus_rec,
            'consensus_score': consensus_score,
            'agreement_level': agreement_level,
            'consensus_confidence': avg_confidence,
            'individual_scores': {r.agent_name: rec_to_score.get(r.recommendation, 0) 
                                 for r in responses},
            'weights_used': self.agent_weights
        }
    
    def _reconcile_decisions(
        self,
        quant_decision: Dict,
        llm_recommendation: str,
        llm_confidence: float,
        agents: List[AgentResponse]
    ) -> tuple[str, float]:
        """Reconcile quantitative consensus with LLM decision"""
        
        agreement_level = quant_decision['agreement_level']
        quant_rec = quant_decision['recommendation']
        
        # If strong agreement among agents (>70%), enforce quantitative consensus
        if agreement_level > 0.7:
            if self._is_contradictory(llm_recommendation, quant_rec):
                print(f"⚠️  LLM decision ({llm_recommendation}) contradicts strong consensus ({quant_rec})")
                print(f"   Using quantitative consensus instead...")
                return quant_rec, quant_decision['consensus_confidence']
        
        # If moderate agreement (50-70%), blend the decisions
        elif agreement_level > 0.5:
            # If LLM is extreme but consensus is moderate, moderate the LLM
            if llm_recommendation in ['STRONG BUY', 'STRONG SELL'] and quant_rec in ['BUY', 'SELL', 'HOLD']:
                moderated = llm_recommendation.replace('STRONG ', '')
                print(f"   Moderating LLM decision from {llm_recommendation} to {moderated}")
                return moderated, (llm_confidence + quant_decision['consensus_confidence']) / 2
        
        # Otherwise trust the LLM but adjust confidence based on agreement
        adjusted_confidence = llm_confidence * (0.5 + 0.5 * agreement_level)
        return llm_recommendation, adjusted_confidence
    
    def _is_contradictory(self, decision: str, consensus: str) -> bool:
        """Check if decision contradicts consensus"""
        buy_recs = ['STRONG BUY', 'BUY']
        sell_recs = ['STRONG SELL', 'SELL']
        
        if consensus in buy_recs and decision in sell_recs:
            return True
        if consensus in sell_recs and decision in buy_recs:
            return True
        if consensus == 'HOLD' and decision in ['STRONG BUY', 'BUY', 'SELL', 'STRONG SELL']:
            return True
        
        return False
    
    def _create_decision_prompt(
        self, 
        ticker: str, 
        responses: List[AgentResponse],
        market_data: Dict,
        quant_decision: Dict,
        risk_assessment: AgentResponse = None
    ) -> str:
        """Create decision prompt with quantitative consensus"""
        
        stats = market_data['statistics']
        
        agent_inputs = "ANALYST RECOMMENDATIONS:\n\n"
        for resp in responses:
            agent_inputs += f"{'='*60}\n"
            agent_inputs += f"{resp.agent_name.upper()}\n"
            agent_inputs += f"Recommendation: {resp.recommendation}\n"
            agent_inputs += f"Confidence: {resp.confidence:.0%}\n"
            agent_inputs += f"Weight in Consensus: {self.agent_weights.get(resp.agent_name, 0.33):.0%}\n"
            agent_inputs += f"\nKey Points:\n"
            for i, reason in enumerate(resp.reasoning[:4], 1):
                agent_inputs += f"{i}. {reason}\n"
            agent_inputs += "\n"
        
        risk_info = ""
        if risk_assessment:
            risk_info = f"""
RISK ASSESSMENT:
{risk_assessment.analysis[:500]}
"""
        
        prompt = f"""Make final trading decision for {ticker}:

MARKET DATA:
- Current Price: ${stats['current_price']:.2f}
- 30D Return: {stats['30d_return']:.2f}%
- 90D Return: {stats['90d_return']:.2f}%
- Volatility: {stats['volatility']:.2f}%

QUANTITATIVE CONSENSUS (Calculated from weighted analyst scores):
- Consensus Recommendation: {quant_decision['recommendation']}
- Consensus Score: {quant_decision['consensus_score']:.2f} (range: -2 to +2)
- Agreement Level: {quant_decision['agreement_level']:.0%}
- Average Confidence: {quant_decision['consensus_confidence']:.0%}

Individual Agent Scores:
{chr(10).join(f"  - {name}: {score:+d}" for name, score in quant_decision['individual_scores'].items())}

{agent_inputs}

{risk_info}

YOUR TASK:
The quantitative consensus suggests {quant_decision['recommendation']}. 
Review the analyst reasoning and either:
1. AGREE with the consensus and explain why it makes sense
2. DISAGREE if you have compelling qualitative reasons (be very specific)

Your decision should generally align with the quantitative consensus unless you identify:
- Critical information the analysts missed
- Unusual market conditions requiring different approach
- Timing considerations that change the picture

Provide:
- Final Recommendation (must be justified given the consensus)
- Your Confidence Level (be honest about uncertainty)
- Entry/Exit Strategy
- Key Risks
- What would invalidate this thesis

Remember: If agreement level is >70%, you should have VERY strong reasons to deviate."""

        return prompt
    
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
        
        return reasoning[:7]
    
    def analyze(self, data: Dict[str, Any]) -> AgentResponse:
        """Required base class method"""
        raise NotImplementedError("Use make_decision() method for TraderAgent")