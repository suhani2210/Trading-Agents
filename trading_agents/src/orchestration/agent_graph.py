from typing import Dict, Any, List
from dataclasses import dataclass, field
from datetime import datetime
import asyncio

from src.agents.technical_analyst import TechnicalAnalyst
from src.agents.fundamental_analyst import FundamentalAnalyst
from src.agents.sentiment_analyst import SentimentAnalyst
from src.agents.risk_manager import RiskManager
from src.agents.trader import TraderAgent
from src.agents.base_agent import BaseAgent, AgentResponse
from src.data.market_data import MarketDataFetcher

@dataclass
class TradingDecision:
    ticker: str
    final_recommendation: str
    confidence: float
    trader_analysis: str
    agent_responses: List[AgentResponse]
    risk_assessment: AgentResponse
    quantitative_consensus: Dict[str, Any]
    market_data: Dict[str, Any]
    timestamp: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict[str, Any]:
        consensus_score = self.quantitative_consensus.get('consensus_score', 0)
        return {
            'ticker': self.ticker,
            'final_recommendation': self.final_recommendation,
            'confidence': self.confidence,
            'trader_analysis': self.trader_analysis,
            'quantitative_consensus': {
                'recommendation': self.quantitative_consensus.get('recommendation'),
                'consensus_score': round(float(consensus_score), 2),
                'agreement_level': self.quantitative_consensus.get('agreement_level'),
                'individual_scores': self.quantitative_consensus.get('individual_scores', {})
            },
            'risk_assessment': {
                'risk_level': self.risk_assessment.metadata.get('risk_level'),
                'position_size': self.risk_assessment.metadata.get('position_size'),
                'stop_loss': self.risk_assessment.metadata.get('stop_loss_price'),
                'take_profit': self.risk_assessment.metadata.get('take_profit_price'),
                'risk_reward_ratio': self.risk_assessment.metadata.get('risk_reward_ratio'),
            },
            'agent_responses': [
                {
                    'agent': r.agent_name, 
                    'recommendation': r.recommendation, 
                    'confidence': r.confidence, 
                    'reasoning': r.reasoning
                } for r in self.agent_responses
            ],
            'timestamp': self.timestamp.isoformat(),
            'market_data': self.market_data
        }

class TradingAgentOrchestrator:
    def __init__(self):
        self.technical_analyst = TechnicalAnalyst()
        self.fundamental_analyst = FundamentalAnalyst()
        self.sentiment_analyst = SentimentAnalyst()
        self.risk_manager = RiskManager()
        self.trader = TraderAgent()
        self.data_fetcher = MarketDataFetcher()

    async def _run_single_analyst(self, analyst, data) -> AgentResponse:
        return await asyncio.to_thread(analyst.analyze, data)

    async def analyze_stock_stream(self, ticker: str, enable_debate: bool = True):
        # Step 1: Gathering Data
        yield {"step": 1, "message": f"Gathering market data for {ticker}..."}
        market_data = await asyncio.to_thread(self.data_fetcher.prepare_analysis_data, ticker)
        yield {"step": 1, "message": f"✓ Data collected: {len(market_data['price_data'])} points"}

        # Step 2: Parallel Analyst Runs
        yield {"step": 2, "message": "Running specialist analyst analysis..."}
        analysts = [self.technical_analyst, self.fundamental_analyst, self.sentiment_analyst]
        tasks = [self._run_single_analyst(a, market_data) for a in analysts]
        
        agent_responses = []
        for task in asyncio.as_completed(tasks):
            res = await task
            agent_responses.append(res)
            yield {"step": 2, "status": "agent_done", "agent": res.agent_name, "message": f"✓ {res.agent_name}: {res.recommendation}"}

        # Step 3: Debate (if enabled)
        if enable_debate and len(agent_responses) >= 2:
            yield {"step": 3, "message": "Initiating agent debate and refinement..."}
            agent_responses = await asyncio.to_thread(self._run_real_debate, agent_responses, market_data)
        
        # Step 4: Final Decision
        yield {"step": 5, "message": "Head Trader synthesizing final decision..."}
        final_decision = await asyncio.to_thread(self.trader.make_decision, ticker, agent_responses, market_data)

        # Step 5: Risk Assessment
        yield {"step": 6, "message": "Risk Manager assessing position..."}
        risk_assessment = await asyncio.to_thread(self.risk_manager.analyze, {**market_data, 'trading_decision': final_decision.recommendation})

        # Compile Final Decision
        decision = TradingDecision(
            ticker=ticker,
            final_recommendation=final_decision.recommendation,
            confidence=final_decision.confidence,
            trader_analysis=final_decision.analysis,
            agent_responses=agent_responses + [final_decision],
            risk_assessment=risk_assessment,
            quantitative_consensus=final_decision.metadata.get('quantitative_decision', {}),
            market_data=market_data
        )

        yield {"step": 7, "status": "complete", "final_decision": decision}

    def _run_real_debate(
        self, 
        responses: List[AgentResponse], 
        market_data: Dict[str, Any]
    ) -> List[AgentResponse]:
        """
        Real debate mechanism where agents respond to each other's arguments
        """
        if len(responses) < 2:
            return responses
        
        # Find conflicting views
        recommendations = [r.recommendation for r in responses]
        buy_votes = sum(1 for r in recommendations if 'BUY' in r)
        sell_votes = sum(1 for r in recommendations if 'SELL' in r)
        
        if buy_votes > 0 and sell_votes > 0:
            print("   Conflicting views detected - agents will debate...")
            
            # Get strongest bull and bear
            bulls = [r for r in responses if 'BUY' in r.recommendation]
            bears = [r for r in responses if 'SELL' in r.recommendation]
            
            if bulls and bears:
                strongest_bull = max(bulls, key=lambda x: x.confidence)
                strongest_bear = max(bears, key=lambda x: x.confidence)
                
                print(f"   {strongest_bull.agent_name} (Bull) vs {strongest_bear.agent_name} (Bear)")
                
                # Have each respond to the other's argument
                updated_responses = []
                for response in responses:
                    if response.agent_name == strongest_bull.agent_name:
                        # Bull responds to bear's argument
                        updated = self._agent_rebut(
                            response, 
                            strongest_bear, 
                            market_data,
                            self._get_agent_by_name(response.agent_name)
                        )
                        updated_responses.append(updated)
                        print(f"  ↪  {response.agent_name} rebutted bear case")
                        if updated.recommendation != response.recommendation:
                            print(f"      Changed from {response.recommendation} to {updated.recommendation}")
                        
                    elif response.agent_name == strongest_bear.agent_name:
                        # Bear responds to bull's argument
                        updated = self._agent_rebut(
                            response, 
                            strongest_bull, 
                            market_data,
                            self._get_agent_by_name(response.agent_name)
                        )
                        updated_responses.append(updated)
                        print(f"  ↪  {response.agent_name} rebutted bull case")
                        if updated.recommendation != response.recommendation:
                            print(f"      Changed from {response.recommendation} to {updated.recommendation}")
                        
                    else:
                        # Neutral agents review both arguments
                        updated_responses.append(response)
                
                print("  ✓ Debate complete - positions refined")
                return updated_responses
        else:
            print("  ℹ  Strong consensus - no debate needed")
        
        return responses
    
    def _agent_rebut(
        self,
        agent_response: AgentResponse,
        opposing_response: AgentResponse,
        market_data: Dict[str, Any],
        agent: BaseAgent
    ) -> AgentResponse:
        """Have an agent respond to opposing viewpoint"""
        
        rebuttal_prompt = f"""You previously analyzed this stock and concluded: {agent_response.recommendation}

Your reasoning was:
{chr(10).join(f"- {r}" for r in agent_response.reasoning[:3])}

However, {opposing_response.agent_name} disagrees and says: {opposing_response.recommendation}

Their reasoning:
{chr(10).join(f"- {r}" for r in opposing_response.reasoning[:3])}

Review their argument and either:
1. MAINTAIN your position with stronger justification
2. MODERATE your position if they have valid points
3. CHANGE your position if they're correct

Provide:
- Updated Recommendation (can be same or different)
- Updated Confidence (adjust if uncertainty increased)
- Response to their key points
- Any new factors you now consider important

Be intellectually honest. It's okay to change your mind if evidence warrants it."""

        try:
            rebuttal = agent._call_llm(rebuttal_prompt)
            new_rec, new_conf = agent.parse_recommendation(rebuttal)
            new_reasoning = agent._extract_reasoning(rebuttal)
            
            # Update response
            return AgentResponse(
                agent_name=agent_response.agent_name,
                analysis=rebuttal,
                recommendation=new_rec,
                confidence=new_conf,
                reasoning=new_reasoning,
                metadata={
                    **agent_response.metadata,
                    'debated': True,
                    'original_recommendation': agent_response.recommendation,
                    'original_confidence': agent_response.confidence
                }
            )
        except Exception as e:
            print(f"    Warning: Debate failed for {agent_response.agent_name}: {e}")
            return agent_response
    
    def _get_agent_by_name(self, name: str) -> BaseAgent:
        """Get agent instance by name"""
        agents = {
            'Technical Analyst': self.technical_analyst,
            'Fundamental Analyst': self.fundamental_analyst,
            'Sentiment Analyst': self.sentiment_analyst
        }
        return agents.get(name, self.technical_analyst)
    
    def batch_analyze(self, tickers: List[str]) -> Dict[str, TradingDecision]:
        """Analyze multiple stocks"""
        results = {}
        
        for ticker in tickers:
            try:
                decision = self.analyze_stock(ticker, enable_debate=False)
                results[ticker] = decision
            except Exception as e:
                print(f"Error analyzing {ticker}: {str(e)}")
                
        return results
    
    def compare_stocks(self, tickers: List[str]) -> Dict[str, Any]:
        """Compare multiple stocks and rank them"""
        print(f"\n{'='*70}")
        print(f"📊 COMPARATIVE ANALYSIS: {', '.join(tickers)}")
        print(f"{'='*70}\n")
        
        results = self.batch_analyze(tickers)
        
        # Rank by confidence-weighted recommendation score
        def score_decision(decision: TradingDecision) -> float:
            rec_scores = {
                'STRONG BUY': 2,
                'BUY': 1,
                'HOLD': 0,
                'SELL': -1,
                'STRONG SELL': -2
            }
            base_score = rec_scores.get(decision.final_recommendation, 0)
            return base_score * decision.confidence
        
        ranked = sorted(results.items(), key=lambda x: score_decision(x[1]), reverse=True)
        
        print("\n🏆 RANKING:")
        for i, (ticker, decision) in enumerate(ranked, 1):
            print(f"{i}. {ticker}: {decision.final_recommendation} ({decision.confidence:.0%})")
            print(f"   Position: {decision.risk_assessment.metadata.get('position_size')}")
            print(f"   Risk/Reward: {decision.risk_assessment.metadata.get('risk_reward_ratio')}")
        
        return {
            'results': results,
            'ranking': [(t, d.final_recommendation, d.confidence) for t, d in ranked]
        }