import requests
from datetime import datetime, timedelta
from typing import List, Dict, Optional
from src.config import Config

class NewsFetcher:
    """Fetch and process news articles for sentiment analysis"""
    
    def __init__(self):
        self.api_key = Config.NEWS_API_KEY
        self.base_url = "https://newsapi.org/v2/everything"
    
    def get_stock_news(
        self, 
        ticker: str, 
        days: int = 7,
        language: str = 'en'
    ) -> List[Dict]:
        """
        Fetch recent news articles about a stock
        
        Args:
            ticker: Stock symbol
            days: Number of days to look back
            language: News language
        
        Returns:
            List of news articles with title, description, source, url, published date
        """
        if not self.api_key:
            return self._get_fallback_news(ticker)
        
        # Calculate date range
        to_date = datetime.now()
        from_date = to_date - timedelta(days=days)
        
        # Build query (search for company name or ticker)
        query = f"{ticker} OR {self._get_company_name(ticker)}"
        
        params = {
            'q': query,
            'from': from_date.strftime('%Y-%m-%d'),
            'to': to_date.strftime('%Y-%m-%d'),
            'language': language,
            'sortBy': 'relevancy',
            'apiKey': self.api_key,
            'pageSize': 20
        }
        
        try:
            response = requests.get(self.base_url, params=params, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                articles = data.get('articles', [])
                return self._process_articles(articles)
            else:
                print(f"News API error: {response.status_code}")
                return self._get_fallback_news(ticker)
        
        except Exception as e:
            print(f"Error fetching news: {e}")
            return self._get_fallback_news(ticker)
    
    def _process_articles(self, articles: List[Dict]) -> List[Dict]:
        """Process and clean news articles"""
        processed = []
        
        for article in articles:
            processed.append({
                'title': article.get('title', 'No title'),
                'description': article.get('description', ''),
                'source': article.get('source', {}).get('name', 'Unknown'),
                'url': article.get('url', ''),
                'published_at': article.get('publishedAt', '')[:10],  # Just date
                'author': article.get('author', 'Unknown')
            })
        
        return processed
    
    def _get_company_name(self, ticker: str) -> str:
        """Map ticker to company name for better search results"""
        # Common mappings
        ticker_map = {
            'AAPL': 'Apple',
            'GOOGL': 'Google Alphabet',
            'MSFT': 'Microsoft',
            'AMZN': 'Amazon',
            'TSLA': 'Tesla',
            'META': 'Meta Facebook',
            'NVDA': 'NVIDIA',
            'NFLX': 'Netflix',
            'AMD': 'AMD',
            'INTC': 'Intel',
            'JPM': 'JPMorgan Chase',
            'BAC': 'Bank of America',
            'WMT': 'Walmart',
            'DIS': 'Disney',
            'BA': 'Boeing',
            'V': 'Visa',
            'MA': 'Mastercard',
            'PYPL': 'PayPal',
            'CRM': 'Salesforce',
            'ORCL': 'Oracle'
        }
        
        return ticker_map.get(ticker.upper(), ticker)
    
    def _get_fallback_news(self, ticker: str) -> List[Dict]:
        """Provide fallback news when API is unavailable"""
        return [
            {
                'title': f'No recent news available for {ticker}',
                'description': 'Unable to fetch news articles. Please check your News API key or try again later.',
                'source': 'System',
                'url': '',
                'published_at': datetime.now().strftime('%Y-%m-%d'),
                'author': 'System'
            }
        ]