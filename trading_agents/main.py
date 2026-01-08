import sys
import os

# This adds the current folder to the list of places Python looks for modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Now your existing imports will work
from src.orchestration.agent_graph import TradingAgentOrchestrator
from src.backtesting.backtest_engine import BacktestEngine

import json

def run_single_analysis(ticker: str = "AAPL"):
    """Run analysis on a single stock"""
    print("\n" + "="*70)
    print("🚀 RUNNING SINGLE STOCK ANALYSIS")
    print("="*70)
    
    orchestrator = TradingAgentOrchestrator()
    decision = orchestrator.analyze_stock(ticker, enable_debate=True)
    
    # Display results
    print("\n" + "="*70)
    print(" FINAL DECISION SUMMARY")
    print("="*70)
    print(f"\n Recommendation: {decision.final_recommendation}")
    print(f" Confidence: {decision.confidence:.0%}")
    print(f"\n Risk Assessment:")
    print(f"   Risk Level: {decision.risk_assessment.metadata['risk_level']}")
    print(f"   Position Size: {decision.risk_assessment.metadata['position_size']}")
    print(f"   Stop Loss: ${decision.risk_assessment.metadata['stop_loss_price']:.2f}")
    print(f"   Take Profit: ${decision.risk_assessment.metadata['take_profit_price']:.2f}")
    
    print(f"\n Quantitative Consensus:")
    qc = decision.quantitative_consensus
    print(f"   Consensus: {qc['recommendation']}")
    print(f"   Agreement Level: {qc['agreement_level']:.0%}")
    print(f"   Consensus Score: {qc['consensus_score']:.2f}")
    
    print(f"\n👥 Individual Agent Votes:")
    for resp in decision.agent_responses[:-1]:  # Exclude final trader
        if resp.recommendation:  # Skip risk manager
            print(f"   {resp.agent_name}: {resp.recommendation} ({resp.confidence:.0%})")
    
    print(f"\n Key Reasoning:")
    for i, reason in enumerate(decision.agent_responses[-1].reasoning[:5], 1):
        print(f"   {i}. {reason[:100]}...")
    
    # Save to file
    with open(f'{ticker}_analysis.json', 'w') as f:
        json.dump(decision.to_dict(), f, indent=2)
    print(f"\n Full analysis saved to {ticker}_analysis.json")
    
    return decision

def run_backtest(ticker: str = "AAPL"):
    """Run backtest on historical data"""
    print("\n" + "="*70)
    print(" RUNNING BACKTEST")
    print("="*70)
    
    orchestrator = TradingAgentOrchestrator()
    backtest = BacktestEngine(orchestrator, initial_capital=10000)
    
    # Run 1-year backtest
    results = backtest.run_backtest(
        ticker=ticker,
        start_date="2023-01-01",
        end_date="2024-01-01",
        rebalance_days=30  # Monthly rebalancing
    )
    
    if results:
        backtest.save_results(results, f'{ticker}_backtest.json')
    
    return results

def compare_multiple_stocks(tickers: list = None):
    """Compare multiple stocks"""
    if tickers is None:
        tickers = ["AAPL", "MSFT", "GOOGL"]
    
    print("\n" + "="*70)
    print(" COMPARING MULTIPLE STOCKS")
    print("="*70)
    
    orchestrator = TradingAgentOrchestrator()
    comparison = orchestrator.compare_stocks(tickers)
    
    print("\n" + "="*70)
    print(" DETAILED COMPARISON")
    print("="*70)
    
    for ticker, decision in comparison['results'].items():
        print(f"\n{ticker}:")
        print(f"  Recommendation: {decision.final_recommendation} ({decision.confidence:.0%})")
        print(f"  Price: ${decision.market_data['statistics']['current_price']:.2f}")
        print(f"  30D Return: {decision.market_data['statistics']['30d_return']:+.2f}%")
        print(f"  Risk Level: {decision.risk_assessment.metadata['risk_level']}")
    
    return comparison

def main():
    """Main entry point"""
    
    print("\n" + "="*70)
    print(" IMPROVED MULTI-AGENT TRADING SYSTEM")
    print("="*70)
    print("\nSelect an option:")
    print("1. Analyze single stock (with debate)")
    print("2. Run backtest on historical data")
    print("3. Compare multiple stocks")
    print("4. Run all tests")
    
    choice = input("\nEnter choice (1-4): ").strip()
    
    ticker = input("Enter ticker (default AAPL): ").strip().upper() or "AAPL"
    
    if choice == "1":
        run_single_analysis(ticker)
    
    elif choice == "2":
        run_backtest(ticker)
    
    elif choice == "3":
        tickers_input = input("Enter tickers separated by commas (default AAPL,MSFT,GOOGL): ").strip()
        if tickers_input:
            tickers = [t.strip().upper() for t in tickers_input.split(",")]
        else:
            tickers = ["AAPL", "MSFT", "GOOGL"]
        compare_multiple_stocks(tickers)
    
    elif choice == "4":
        print("\n🔄 Running all tests...\n")
        run_single_analysis(ticker)
        input("\nPress Enter to continue to backtest...")
        run_backtest(ticker)
    
    else:
        print("Invalid choice. Running single analysis...")
        run_single_analysis(ticker)

if __name__ == "__main__":
    main()