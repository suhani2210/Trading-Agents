import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, Any, Optional
import ta  # Technical analysis library

class MarketDataFetcher:
    """Fetch and process market data for analysis"""
    
    @staticmethod
    def get_stock_data(
        ticker: str, 
        period: str = "3mo",
        interval: str = "1d",
        start_date: Optional[str] = None, # Add this
        end_date: Optional[str] = None    # Add this
    ) -> pd.DataFrame:
        """Fetch stock data with support for specific date ranges"""
        stock = yf.Ticker(ticker)
        
        # Priority: use start/end if they are provided
        if start_date and end_date:
            df = stock.history(start=start_date, end=end_date, interval=interval)
        else:
            df = stock.history(period=period, interval=interval)
        
        if df.empty:
            raise ValueError(f"No data found for ticker {ticker}")
        
        # Crucial fix for the comparison error we saw earlier
        df.index = df.index.tz_localize(None)
        
        return df
    
    @staticmethod
    def calculate_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
        """Calculate technical indicators"""
        df = df.copy()
        
        # Moving Averages
        df['SMA_20'] = ta.trend.sma_indicator(df['Close'], window=20)
        df['SMA_50'] = ta.trend.sma_indicator(df['Close'], window=50)
        df['EMA_12'] = ta.trend.ema_indicator(df['Close'], window=12)
        df['EMA_26'] = ta.trend.ema_indicator(df['Close'], window=26)
        
        # RSI
        df['RSI'] = ta.momentum.rsi(df['Close'], window=14)
        
        # MACD
        macd = ta.trend.MACD(df['Close'])
        df['MACD'] = macd.macd()
        df['MACD_signal'] = macd.macd_signal()
        df['MACD_diff'] = macd.macd_diff()
        
        # Bollinger Bands
        bollinger = ta.volatility.BollingerBands(df['Close'])
        df['BB_upper'] = bollinger.bollinger_hband()
        df['BB_middle'] = bollinger.bollinger_mavg()
        df['BB_lower'] = bollinger.bollinger_lband()
        
        # Volume indicators
        df['Volume_SMA'] = df['Volume'].rolling(window=20).mean()
        
        # ATR (Average True Range) for volatility
        df['ATR'] = ta.volatility.average_true_range(df['High'], df['Low'], df['Close'])
        
        return df
    
    @staticmethod
    def get_fundamental_data(ticker: str) -> Dict[str, Any]:
        """Get fundamental data for the stock"""
        stock = yf.Ticker(ticker)
        info = stock.info
        
        fundamentals = {
            'market_cap': info.get('marketCap', 'N/A'),
            'pe_ratio': info.get('trailingPE', 'N/A'),
            'forward_pe': info.get('forwardPE', 'N/A'),
            'peg_ratio': info.get('pegRatio', 'N/A'),
            'price_to_book': info.get('priceToBook', 'N/A'),
            'debt_to_equity': info.get('debtToEquity', 'N/A'),
            'roe': info.get('returnOnEquity', 'N/A'),
            'profit_margins': info.get('profitMargins', 'N/A'),
            'revenue_growth': info.get('revenueGrowth', 'N/A'),
            'earnings_growth': info.get('earningsGrowth', 'N/A'),
            'current_price': info.get('currentPrice', 'N/A'),
            'target_price': info.get('targetMeanPrice', 'N/A'),
            '52week_high': info.get('fiftyTwoWeekHigh', 'N/A'),
            '52week_low': info.get('fiftyTwoWeekLow', 'N/A'),
            'sector': info.get('sector', 'N/A'),
            'industry': info.get('industry', 'N/A'),
        }
        
        return fundamentals
    
    @staticmethod
    def get_summary_statistics(df: pd.DataFrame) -> Dict[str, float]:
        """Calculate summary statistics"""
        latest = df.iloc[-1]
        previous = df.iloc[-2] if len(df) > 1 else latest
        
        # Calculate returns
        daily_returns = df['Close'].pct_change().dropna()
        
        stats = {
            'current_price': float(latest['Close']),
            'previous_close': float(previous['Close']),
            'price_change': float(latest['Close'] - previous['Close']),
            'price_change_pct': float((latest['Close'] - previous['Close']) / previous['Close'] * 100),
            'volume': int(latest['Volume']),
            'avg_volume': float(df['Volume'].mean()),
            'volatility': float(daily_returns.std() * np.sqrt(252) * 100),  # Annualized
            'sharpe_ratio': float(daily_returns.mean() / daily_returns.std() * np.sqrt(252)) if daily_returns.std() != 0 else 0,
            '30d_return': float((df['Close'].iloc[-1] / df['Close'].iloc[-30] - 1) * 100) if len(df) >= 30 else 0,
            '90d_return': float((df['Close'].iloc[-1] / df['Close'].iloc[0] - 1) * 100),
        }
        
        return stats
    
    @staticmethod
    def prepare_analysis_data(ticker: str) -> Dict[str, Any]:
        """Prepare comprehensive data package for agent analysis"""
        # Fetch price data
        df = MarketDataFetcher.get_stock_data(ticker)
        
        # Add technical indicators
        df = MarketDataFetcher.calculate_technical_indicators(df)
        
        # Get fundamentals
        fundamentals = MarketDataFetcher.get_fundamental_data(ticker)
        
        # Get statistics
        stats = MarketDataFetcher.get_summary_statistics(df)
        
        # Get latest technical values
        latest = df.iloc[-1]
        technical_snapshot = {
            'SMA_20': float(latest['SMA_20']) if not pd.isna(latest['SMA_20']) else None,
            'SMA_50': float(latest['SMA_50']) if not pd.isna(latest['SMA_50']) else None,
            'RSI': float(latest['RSI']) if not pd.isna(latest['RSI']) else None,
            'MACD': float(latest['MACD']) if not pd.isna(latest['MACD']) else None,
            'MACD_signal': float(latest['MACD_signal']) if not pd.isna(latest['MACD_signal']) else None,
            'BB_upper': float(latest['BB_upper']) if not pd.isna(latest['BB_upper']) else None,
            'BB_lower': float(latest['BB_lower']) if not pd.isna(latest['BB_lower']) else None,
        }
        
        return {
            'ticker': ticker,
            'price_data': df,
            'fundamentals': fundamentals,
            'statistics': stats,
            'technical_snapshot': technical_snapshot,
            'timestamp': datetime.now().isoformat()
        }