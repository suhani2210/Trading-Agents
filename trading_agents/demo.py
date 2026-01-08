#!/usr/bin/env python3
"""
Demo script for AI Trading Agents Platform
Run this to test the system and see example outputs
"""

import sys
from datetime import datetime

from src.orchestration.agent_graph import TradingAgentOrchestrator
from src.config import Config

def print_header(text):
    """Print formatted header"""
    print("\n" + "="*80)
    print(f"  {text}")
    print("="*80 + "\n")

def print_decision_summary(decision):
    """Print formatted decision summary"""
    print_header(f"TRADING DECISION: {decision.ticker}")
    
    print(f" Final Recommendation: {decision.final_recommendation}")
    print(f" Confidence Level: {decision.confidence:.1%}")
    print(f" Analysis Time: {decision.timestamp.strftime('%Y-%m-%d %H:%M:%S')}")
    
    print(f"\n Market Data:")
    print(f"   Current Price: ${decision.market_data['statistics']['current_price']:.2f}")
    print(f"   Price Change: {decision.market_data['statistics']['price_change_pct']:.2f}%")
    print(f"   30-Day Return: {decision.market_data['statistics']['30d_return']:.2f}%")
    print(f"   Volatility: {decision.market_data['statistics']['volatility']:.1f}%")
    
    print(f"\n Agent Consensus:")
    print(f"   Agreement: {decision.consensus_metrics.get('agreement_score', 'N/A')}")
    print(f"   Avg Confidence: {decision.consensus_metrics.get('confidence_avg', 'N/A')}")
    
    print(f"\n📋 Agent Recommendations:")
    for response in decision.agent_responses[:-1]:  # Exclude final decision
        emoji = "🟢" if "BUY" in response.recommendation else "🔴" if "SELL" in response.recommendation else "🟡"
        print(f"   {emoji} {response.agent_name}: {response.recommendation} ({response.confidence:.0%})")
    
    print(f"\n🎯 Head Trader Analysis:")
    # Print first 500 chars of analysis
    analysis = decision.trader_analysis[:500]
    if len(decision.trader_analysis) > 500:
        analysis += "..."
    print(f"   {analysis}")
    
    print(f"\n Key Reasoning Points:")
    final_response = decision.agent_responses[-1]
    for i, reason in enumerate(final_response.reasoning[:5], 1):
        print(f"   {i}. {reason}")
    
    print("\n" + "="*80 + "\n")

def demo_single_stock():
    """Demo: Analyze a single stock"""
    print_header("DEMO 1: Single Stock Analysis")
    
    ticker = "AAPL"
    print(f"Analyzing {ticker} with all agents...\n")
    
    orchestrator = TradingAgentOrchestrator()
    
    try:
        decision = orchestrator.analyze_stock(ticker, enable_debate=True)
        print_decision_summary(decision)
        
        # Save results
        import json
        with open(f'results_{ticker}_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json', 'w') as f:
            json.dump(decision.to_dict(), f, indent=2, default=str)
        
        print(" Results saved to JSON file")
        
    except Exception as e:
        print(f" Error: {str(e)}")
        import traceback
        traceback.print_exc()

def demo_stock_comparison():
    """Demo: Compare multiple stocks"""
    print_header("DEMO 2: Stock Comparison")
    
    tickers = ['AAPL', 'GOOGL', 'MSFT']
    print(f"Comparing stocks: {', '.join(tickers)}\n")
    
    orchestrator = TradingAgentOrchestrator()
    
    try:
        comparison = orchestrator.compare_stocks(tickers)
        
        print("\n FINAL RANKINGS:")
        print("-" * 60)
        for i, (ticker, rec, conf) in enumerate(comparison['ranking'], 1):
            emoji = "🥇" if i == 1 else "🥈" if i == 2 else "🥉"
            print(f"{emoji} #{i}: {ticker:6} | {rec:12} | {conf:.0%} confidence")
        print("-" * 60)
        
    except Exception as e:
        print(f" Error: {str(e)}")
        import traceback
        traceback.print_exc()

def demo_individual_agent():
    """Demo: Test individual agent"""
    print_header("DEMO 3: Individual Agent Test")
    
    from src.agents.technical_analyst import TechnicalAnalyst
    from src.data.market_data import MarketDataFetcher
    
    ticker = "TSLA"
    print(f"Testing Technical Analyst on {ticker}...\n")
    
    try:
        # Fetch data
        data = MarketDataFetcher.prepare_analysis_data(ticker)
        
        # Run technical analysis
        analyst = TechnicalAnalyst()
        response = analyst.analyze(data)
        
        print(f" Technical Analyst Results:")
        print(f"   Recommendation: {response.recommendation}")
        print(f"   Confidence: {response.confidence:.1%}")
        print(f"\n   Key Technical Signals:")
        for i, reason in enumerate(response.reasoning[:5], 1):
            print(f"   {i}. {reason}")
        
    except Exception as e:
        print(f" Error: {str(e)}")
        import traceback
        traceback.print_exc()

def check_configuration():
    """Check if configuration is set up correctly"""
    print_header("Configuration Check")
    
    checks = {
        "OpenAI API Key": bool(Config.GEMINI_API_KEY),
        "News API Key": bool(Config.NEWS_API_KEY),
        "Model Name": Config.MODEL_NAME,
        "Temperature": Config.TEMPERATURE,
    }
    
    all_good = True
    for name, value in checks.items():
        if isinstance(value, bool):
            status = "✅" if value else "❌"
            all_good = all_good and value
        else:
            status = "ℹ️"
        print(f"{status} {name}: {value}")
    
    if not all_good:
        print("\n  Warning: Some API keys are missing!")
        print("Please set them in your .env file")
    else:
        print("\n Configuration looks good!")
    
    print()

def main():
    """Main demo function"""
    print()
    print("╔" + "="*78 + "╗")
    print("║" + " "*20 + "AI TRADING AGENTS PLATFORM" + " "*32 + "║")
    print("║" + " "*25 + "Demo & Test Suite" + " "*36 + "║")
    print("╚" + "="*78 + "╝")
    
    # Check configuration
    check_configuration()
    
    # Run demos
    demos = {
        '1': ('Single Stock Analysis', demo_single_stock),
        '2': ('Stock Comparison', demo_stock_comparison),
        '3': ('Individual Agent Test', demo_individual_agent),
        'a': ('Run All Demos', None),
    }
    
    print("\nAvailable Demos:")
    for key, (name, _) in demos.items():
        print(f"  [{key}] {name}")
    print("  [q] Quit")
    
    choice = input("\nSelect demo (default: 1): ").strip().lower() or '1'
    
    if choice == 'q':
        print("Goodbye! 👋")
        return
    elif choice == 'a':
        demo_single_stock()
        demo_stock_comparison()
        demo_individual_agent()
    elif choice in demos and demos[choice][1]:
        demos[choice][1]()
    else:
        print("Invalid choice. Running single stock analysis...")
        demo_single_stock()
    
    print("\n✨ Demo complete!")
    print(" Tip: Run 'streamlit run web/app.py' to launch the web interface")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n Interrupted by user. Goodbye!")
        sys.exit(0)
    except Exception as e:
        print(f"\n Unexpected error: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)