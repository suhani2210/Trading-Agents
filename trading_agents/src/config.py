import os
from dotenv import load_dotenv
from typing import Optional

load_dotenv()

class Config:
    """Central configuration for the trading agents system"""
    
    # API Keys
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
    NEWS_API_KEY: str = os.getenv("NEWS_API_KEY", "")
    ALPHA_VANTAGE_KEY: Optional[str] = os.getenv("ALPHA_VANTAGE_KEY")
    
    # Model Settings
    MODEL_NAME: str = os.getenv("MODEL_NAME", "gemini-2.5-flash")
    # CRITICAL: Temperature must be low for consistent trading decisions
    # 0.7 = creative/random, 0.1-0.2 = focused/deterministic
    TEMPERATURE: float = float(os.getenv("TEMPERATURE", "0.05"))
    MAX_TOKENS: int = 2000
    
    # Agent Settings
    MAX_DEBATE_ROUNDS: int = 2
    CONFIDENCE_THRESHOLD: float = 0.7
    
    # Data Settings
    DEFAULT_LOOKBACK_DAYS: int = 90
    TECHNICAL_INDICATORS = [
        "SMA_20", "SMA_50", "RSI", "MACD", "BB_UPPER", "BB_LOWER"
    ]
    
    import os
from dotenv import load_dotenv
from typing import Optional

load_dotenv()

class Config:
    """Central configuration for the trading agents system"""
    
    # API Keys
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
    NEWS_API_KEY: str = os.getenv("NEWS_API_KEY", "")
    ALPHA_VANTAGE_KEY: Optional[str] = os.getenv("ALPHA_VANTAGE_KEY")
    
    # Model Settings
    MODEL_NAME: str = os.getenv("MODEL_NAME", "gemini-2.5-flash")
    # CRITICAL: Temperature must be low for consistent trading decisions
    # 0.7 = creative/random, 0.1-0.2 = focused/deterministic
    TEMPERATURE: float = float(os.getenv("TEMPERATURE", "0.2"))
    MAX_TOKENS: int = 2000
    
    # Agent Settings
    MAX_DEBATE_ROUNDS: int = 2
    CONFIDENCE_THRESHOLD: float = 0.7
    
    # Data Settings
    DEFAULT_LOOKBACK_DAYS: int = 90
    TECHNICAL_INDICATORS = [
        "SMA_20", "SMA_50", "RSI", "MACD", "BB_UPPER", "BB_LOWER"
    ]
    
    # Risk Management
    MAX_POSITION_SIZE: float = 0.1  # 10% of portfolio
    STOP_LOSS_PERCENT: float = 0.05  # 5%
    TAKE_PROFIT_PERCENT: float = 0.15  # 15%
    
    @classmethod
    def validate(cls) -> bool:
        """Validate required configuration"""
        if not cls.GEMINI_API_KEY:
            raise ValueError("GEMINI_API_KEY is required")
        
        # Warn if temperature is too high for trading
        if cls.TEMPERATURE > 0.3:
            print(f"⚠️  WARNING: Temperature is {cls.TEMPERATURE}")
            print(f"   High temperature causes inconsistent trading decisions.")
            print(f"   Recommended: 0.1-0.2 for production trading systems.")
        
        return True

# Validate on import
Config.validate()
  