import yfinance as yf
import pandas as pd

def get_stock_data(ticker: str):
    """Fetches historical data and calculates technical indicators."""
    df = yf.download(ticker, period="6mo", interval="1d")
    
    # Technical Analysis (The "ML/Math" part for your resume)
    df['SMA_20'] = df['Close'].rolling(window=20).mean()
    df['SMA_50'] = df['Close'].rolling(window=50).mean()
    
    # Relative Strength Index (RSI)
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))
    
    latest_price = df['Close'].iloc[-1]
    latest_rsi = df['RSI'].iloc[-1]
    
    return {
        "price": float(latest_price),
        "rsi": float(latest_rsi),
        "trend": "Bullish" if df['SMA_20'].iloc[-1] > df['SMA_50'].iloc[-1] else "Bearish",
        "data_summary": df.tail(5).to_string()
    }