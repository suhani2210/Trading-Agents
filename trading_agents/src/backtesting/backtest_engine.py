import pandas as pd
import numpy as np
from typing import Dict, List, Any
from datetime import datetime, timedelta
from dataclasses import dataclass
import json

@dataclass
class BacktestTrade:
    """Individual trade in backtest"""
    ticker: str
    entry_date: datetime
    entry_price: float
    exit_date: datetime
    exit_price: float
    recommendation: str
    confidence: float
    position_size: float
    return_pct: float
    profit_loss: float
    stop_hit: bool = False
    target_hit: bool = False

class BacktestEngine:
    """
    Backtest the trading agent system on historical data
    """
    
    def __init__(self, orchestrator, initial_capital: float = 10000):
        self.orchestrator = orchestrator
        self.initial_capital = initial_capital
        self.current_capital = initial_capital
        self.trades: List[BacktestTrade] = []
        
    def run_backtest(
        self,
        ticker: str,
        start_date: str,
        end_date: str,
        rebalance_days: int = 30
    ) -> Dict[str, Any]:
        """
        Run backtest on historical data
        
        Args:
            ticker: Stock to test
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            rebalance_days: Days between decisions
        
        Returns:
            Backtest results with metrics
        """
        print(f"\n{'='*70}")
        print(f"🔬 BACKTESTING: {ticker}")
        print(f"   Period: {start_date} to {end_date}")
        print(f"   Rebalance: Every {rebalance_days} days")
        print(f"   Initial Capital: ${self.initial_capital:,.2f}")
        print(f"{'='*70}\n")
        
        # Get historical data
        from src.data.market_data import MarketDataFetcher
        fetcher = MarketDataFetcher()
        
        # Parse dates
        start = pd.to_datetime(start_date)
        end = pd.to_datetime(end_date)
        
        # Get price data
        try:
            price_data = fetcher.get_stock_data(ticker, start_date=start_date, end_date=end_date)
            price_data['Date'] = pd.to_datetime(price_data.index)
           
        except Exception as e:
            print(f"Error fetching data: {e}")
            return {}
        
        if len(price_data) < rebalance_days:
            print("Not enough historical data for backtesting")
            return {}
        
        # Run backtest
        current_date = start
        self.current_capital = self.initial_capital
        self.trades = []
        
        current_position = None  # Track open position
        
        while current_date < end:
            # Get data up to current date for analysis
            analysis_data = price_data[price_data['Date'] <= current_date]
            
            if len(analysis_data) < 30:  # Need minimum data
                current_date += timedelta(days=rebalance_days)
                continue
            
            print(f"\n📅 Decision Point: {current_date.strftime('%Y-%m-%d')}")
            print(f"   Portfolio Value: ${self.current_capital:,.2f}")
            
            try:
                # Run agent analysis (with historical data only)
                decision = self._analyze_at_date(ticker, current_date, analysis_data)
                
                # Close existing position if needed
                if current_position and decision.final_recommendation in ['SELL', 'STRONG SELL', 'HOLD']:
                    current_position = self._close_position(
                        current_position,
                        current_date,
                        price_data
                    )
                
                # Open new position if signal is BUY
                if decision.final_recommendation in ['BUY', 'STRONG BUY']:
                    current_position = self._open_position(
                        ticker,
                        current_date,
                        decision,
                        price_data
                    )
                
            except Exception as e:
                print(f"   ✗ Analysis failed: {e}")
            
            # Move to next decision point
            current_date += timedelta(days=rebalance_days)
        
        # Close any remaining position
        if current_position:
            self._close_position(current_position, end, price_data)
        
        # Calculate results
        results = self._calculate_results(ticker, start_date, end_date, price_data)
        
        print(f"\n{'='*70}")
        print(f" BACKTEST COMPLETE")
        print(f"{'='*70}\n")
        
        return results
    
    def _analyze_at_date(self, ticker: str, date: datetime, historical_data: pd.DataFrame) -> Any:
        """Run analysis using only data available at given date"""
        
        # Create market data dict from historical data
        recent_data = historical_data.tail(90)  # Last 90 days
        
        market_data = {
            'ticker': ticker,
            'price_data': recent_data,
            'statistics': {
                'current_price': recent_data['Close'].iloc[-1],
                'price_change_pct': ((recent_data['Close'].iloc[-1] - recent_data['Close'].iloc[0]) / 
                                    recent_data['Close'].iloc[0] * 100),
                '30d_return': ((recent_data['Close'].iloc[-1] - recent_data['Close'].iloc[-30]) / 
                              recent_data['Close'].iloc[-30] * 100) if len(recent_data) >= 30 else 0,
                '90d_return': ((recent_data['Close'].iloc[-1] - recent_data['Close'].iloc[0]) / 
                              recent_data['Close'].iloc[0] * 100),
                'volatility': recent_data['Close'].pct_change().std() * np.sqrt(252) * 100,
                'sharpe_ratio': self._calculate_sharpe(recent_data['Close'])
            }
        }
        
        # Run analysis (disable debate for speed)
        decision = self.orchestrator.analyze_stock(ticker, enable_debate=False)
        
        print(f"   → {decision.final_recommendation} ({decision.confidence:.0%})")
        
        return decision
    
    def _open_position(self, ticker: str, date: datetime, decision: Any, price_data: pd.DataFrame) -> Dict:
        """Open a new position"""
        
        # Get price at this date
        date_data = price_data[price_data['Date'] == date]
        if len(date_data) == 0:
            date_data = price_data[price_data['Date'] >= date].iloc[0:1]
        
        entry_price = date_data['Close'].iloc[0]
        
        # Get position size from risk assessment
        position_size_pct = float(decision.risk_assessment.metadata['position_size'].rstrip('%')) / 100
        position_value = self.current_capital * position_size_pct
        shares = position_value / entry_price
        
        print(f"    OPEN LONG: {shares:.2f} shares @ ${entry_price:.2f}")
        print(f"      Position Size: {position_size_pct*100:.1f}% (${position_value:,.2f})")
        
        return {
            'ticker': ticker,
            'entry_date': date,
            'entry_price': entry_price,
            'shares': shares,
            'position_value': position_value,
            'decision': decision,
            'stop_loss': decision.risk_assessment.metadata['stop_loss_price'],
            'take_profit': decision.risk_assessment.metadata['take_profit_price']
        }
    
    def _close_position(self, position: Dict, date: datetime, price_data: pd.DataFrame) -> None:
        """Close an existing position"""
        
        # Get price at this date
        date_data = price_data[price_data['Date'] == date]
        if len(date_data) == 0:
            date_data = price_data[price_data['Date'] >= date].iloc[0:1]
        
        exit_price = date_data['Close'].iloc[0]
        
        # Check if stopped out between entry and exit
        between_data = price_data[
            (price_data['Date'] > position['entry_date']) & 
            (price_data['Date'] <= date)
        ]
        
        stop_hit = False
        target_hit = False
        actual_exit = exit_price
        
        if len(between_data) > 0:
            # Check for stop loss hit
            if (between_data['Low'] <= position['stop_loss']).any():
                stop_hit = True
                actual_exit = position['stop_loss']
                print(f"    STOPPED OUT @ ${actual_exit:.2f}")
            # Check for take profit hit
            elif (between_data['High'] >= position['take_profit']).any():
                target_hit = True
                actual_exit = position['take_profit']
                print(f"    TARGET HIT @ ${actual_exit:.2f}")
            else:
                print(f"    CLOSE POSITION @ ${actual_exit:.2f}")
        
        # Calculate P&L
        profit_loss = (actual_exit - position['entry_price']) * position['shares']
        return_pct = (actual_exit - position['entry_price']) / position['entry_price'] * 100
        
        self.current_capital += profit_loss
        
        print(f"      P&L: ${profit_loss:,.2f} ({return_pct:+.2f}%)")
        print(f"      New Capital: ${self.current_capital:,.2f}")
        
        # Record trade
        trade = BacktestTrade(
            ticker=position['ticker'],
            entry_date=position['entry_date'],
            entry_price=position['entry_price'],
            exit_date=date,
            exit_price=actual_exit,
            recommendation=position['decision'].final_recommendation,
            confidence=position['decision'].confidence,
            position_size=position['position_value'] / self.initial_capital,
            return_pct=return_pct,
            profit_loss=profit_loss,
            stop_hit=stop_hit,
            target_hit=target_hit
        )
        self.trades.append(trade)
        
        return None
    
    def _calculate_sharpe(self, prices: pd.Series) -> float:
        """Calculate Sharpe ratio"""
        returns = prices.pct_change().dropna()
        if len(returns) == 0:
            return 0
        return (returns.mean() * 252) / (returns.std() * np.sqrt(252))
    
    def _calculate_results(self, ticker: str, start_date: str, end_date: str, price_data: pd.DataFrame) -> Dict:
        """Calculate comprehensive backtest results"""
        
        if len(self.trades) == 0:
            print("No trades executed in backtest period")
            return {}
        
        # Performance metrics
        total_return = (self.current_capital - self.initial_capital) / self.initial_capital * 100
        num_trades = len(self.trades)
        winning_trades = [t for t in self.trades if t.profit_loss > 0]
        losing_trades = [t for t in self.trades if t.profit_loss < 0]
        
        win_rate = len(winning_trades) / num_trades * 100 if num_trades > 0 else 0
        
        avg_win = np.mean([t.return_pct for t in winning_trades]) if winning_trades else 0
        avg_loss = np.mean([t.return_pct for t in losing_trades]) if losing_trades else 0
        
        # Calculate buy-and-hold comparison
        bh_return = ((price_data['Close'].iloc[-1] - price_data['Close'].iloc[0]) / 
                    price_data['Close'].iloc[0] * 100)
        
        # Print results
        print("\n BACKTEST RESULTS:")
        print(f"\n💰 Performance:")
        print(f"   Initial Capital:     ${self.initial_capital:,.2f}")
        print(f"   Final Capital:       ${self.current_capital:,.2f}")
        print(f"   Total Return:        {total_return:+.2f}%")
        print(f"   Buy & Hold Return:   {bh_return:+.2f}%")
        print(f"   Alpha:               {total_return - bh_return:+.2f}%")
        
        print(f"\n Trade Statistics:")
        print(f"   Total Trades:        {num_trades}")
        print(f"   Winning Trades:      {len(winning_trades)}")
        print(f"   Losing Trades:       {len(losing_trades)}")
        print(f"   Win Rate:            {win_rate:.1f}%")
        print(f"   Average Win:         {avg_win:+.2f}%")
        print(f"   Average Loss:        {avg_loss:+.2f}%")
        
        if losing_trades:
            profit_factor = abs(sum(t.profit_loss for t in winning_trades) / 
                              sum(t.profit_loss for t in losing_trades))
            print(f"   Profit Factor:       {profit_factor:.2f}")
        
        print(f"\n Risk Management:")
        stop_hits = sum(1 for t in self.trades if t.stop_hit)
        target_hits = sum(1 for t in self.trades if t.target_hit)
        print(f"   Stop Losses Hit:     {stop_hits}/{num_trades}")
        print(f"   Targets Hit:         {target_hits}/{num_trades}")
        
        return {
            'initial_capital': self.initial_capital,
            'final_capital': self.current_capital,
            'total_return_pct': total_return,
            'buy_hold_return_pct': bh_return,
            'alpha': total_return - bh_return,
            'num_trades': num_trades,
            'winning_trades': len(winning_trades),
            'losing_trades': len(losing_trades),
            'win_rate': win_rate,
            'avg_win_pct': avg_win,
            'avg_loss_pct': avg_loss,
            'trades': [
                {
                    'entry_date': t.entry_date.strftime('%Y-%m-%d'),
                    'exit_date': t.exit_date.strftime('%Y-%m-%d'),
                    'entry_price': t.entry_price,
                    'exit_price': t.exit_price,
                    'return_pct': t.return_pct,
                    'profit_loss': t.profit_loss,
                    'recommendation': t.recommendation,
                    'confidence': t.confidence
                }
                for t in self.trades
            ]
        }
    
    def save_results(self, results: Dict, filename: str = 'backtest_results.json'):
        """Save backtest results to file"""
        with open(filename, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\n💾 Results saved to {filename}")