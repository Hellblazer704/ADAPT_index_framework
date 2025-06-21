"""
ADAPT Smart Indexing Engine - NIFTY Data Loader

This module handles loading and processing of NIFTY stock data,
including historical prices and fundamental metrics.
"""

import pandas as pd
import numpy as np
import yfinance as yf
from typing import Dict, List, Optional, Tuple
import warnings
warnings.filterwarnings('ignore')

class NiftyLoader:
    """Handles loading and processing of NIFTY stock data."""
    
    def __init__(self):
        """Initialize the NIFTY data loader."""
        self.nifty_50_stocks = [
            'RELIANCE.NS', 'TCS.NS', 'HDFCBANK.NS', 'INFY.NS', 'ICICIBANK.NS',
            'HINDUNILVR.NS', 'ITC.NS', 'SBIN.NS', 'BHARTIARTL.NS', 'KOTAKBANK.NS',
            'AXISBANK.NS', 'ASIANPAINT.NS', 'MARUTI.NS', 'HCLTECH.NS', 'SUNPHARMA.NS',
            'TATAMOTORS.NS', 'WIPRO.NS', 'ULTRACEMCO.NS', 'TITAN.NS', 'BAJFINANCE.NS',
            'NESTLEIND.NS', 'POWERGRID.NS', 'BAJAJFINSV.NS', 'NTPC.NS', 'ONGC.NS',
            'ADANIENT.NS', 'JSWSTEEL.NS', 'COALINDIA.NS', 'TECHM.NS', 'ADANIPORTS.NS',
            'HINDALCO.NS', 'CIPLA.NS', 'DRREDDY.NS', 'SHREECEM.NS', 'BRITANNIA.NS',
            'EICHERMOT.NS', 'HEROMOTOCO.NS', 'DIVISLAB.NS', 'GRASIM.NS', 'TATACONSUM.NS',
            'APOLLOHOSP.NS', 'BAJAJ-AUTO.NS', 'UPL.NS', 'TATASTEEL.NS', 'VEDL.NS',
            'SBILIFE.NS', 'HDFCLIFE.NS', 'INDUSINDBK.NS', 'MM.NS', 'BPCL.NS'
        ]
        
        self.sector_mapping = {
            'RELIANCE.NS': 'Energy', 'TCS.NS': 'Technology', 'HDFCBANK.NS': 'Banking',
            'INFY.NS': 'Technology', 'ICICIBANK.NS': 'Banking', 'HINDUNILVR.NS': 'FMCG',
            'ITC.NS': 'FMCG', 'SBIN.NS': 'Banking', 'BHARTIARTL.NS': 'Telecom',
            'KOTAKBANK.NS': 'Banking', 'AXISBANK.NS': 'Banking', 'ASIANPAINT.NS': 'Materials',
            'MARUTI.NS': 'Automotive', 'HCLTECH.NS': 'Technology', 'SUNPHARMA.NS': 'Pharma',
            'TATAMOTORS.NS': 'Automotive', 'WIPRO.NS': 'Technology', 'ULTRACEMCO.NS': 'Materials',
            'TITAN.NS': 'Consumer Discretionary', 'BAJFINANCE.NS': 'Financial Services',
            'NESTLEIND.NS': 'FMCG', 'POWERGRID.NS': 'Utilities', 'BAJAJFINSV.NS': 'Financial Services',
            'NTPC.NS': 'Utilities', 'ONGC.NS': 'Energy', 'ADANIENT.NS': 'Conglomerate',
            'JSWSTEEL.NS': 'Metals', 'COALINDIA.NS': 'Energy', 'TECHM.NS': 'Technology',
            'ADANIPORTS.NS': 'Infrastructure', 'HINDALCO.NS': 'Metals', 'CIPLA.NS': 'Pharma',
            'DRREDDY.NS': 'Pharma', 'SHREECEM.NS': 'Materials', 'BRITANNIA.NS': 'FMCG',
            'EICHERMOT.NS': 'Automotive', 'HEROMOTOCO.NS': 'Automotive', 'DIVISLAB.NS': 'Pharma',
            'GRASIM.NS': 'Materials', 'TATACONSUM.NS': 'FMCG', 'APOLLOHOSP.NS': 'Healthcare',
            'BAJAJ-AUTO.NS': 'Automotive', 'UPL.NS': 'Chemicals', 'TATASTEEL.NS': 'Metals',
            'VEDL.NS': 'Metals', 'SBILIFE.NS': 'Insurance', 'HDFCLIFE.NS': 'Insurance',
            'INDUSINDBK.NS': 'Banking', 'MM.NS': 'Technology', 'BPCL.NS': 'Energy'
        }
        
        self.cache = {}
    
    def fetch_stock_data(self, symbol: str, start_date: str = "2020-01-01", 
                        end_date: Optional[str] = None) -> Optional[pd.DataFrame]:
        """
        Fetch historical stock data for a given symbol.
        
        Args:
            symbol: Stock symbol (e.g., 'RELIANCE.NS')
            start_date: Start date for data fetch
            end_date: End date for data fetch (defaults to today)
            
        Returns:
            DataFrame with OHLCV data or None if failed
        """
        try:
            cache_key = f"{symbol}_{start_date}_{end_date}"
            if cache_key in self.cache:
                return self.cache[cache_key]
            
            ticker = yf.Ticker(symbol)
            data = ticker.history(start=start_date, end=end_date)
            
            if data.empty:
                print(f"No data found for {symbol}")
                return None
            
            # Clean column names
            data.columns = data.columns.str.lower()
            
            # Cache the result
            self.cache[cache_key] = data
            
            return data
            
        except Exception as e:
            print(f"Error fetching data for {symbol}: {str(e)}")
            return None
    
    def fetch_multiple_stocks(self, symbols: List[str], start_date: str = "2020-01-01",
                            end_date: Optional[str] = None) -> Dict[str, pd.DataFrame]:
        """
        Fetch data for multiple stocks.
        
        Args:
            symbols: List of stock symbols
            start_date: Start date for data fetch
            end_date: End date for data fetch
            
        Returns:
            Dictionary mapping symbols to their data
        """
        stock_data = {}
        
        for symbol in symbols:
            data = self.fetch_stock_data(symbol, start_date, end_date)
            if data is not None:
                stock_data[symbol] = data
        
        return stock_data
    
    def get_fundamental_data(self, symbol: str) -> Dict[str, any]:
        """
        Get fundamental data for a stock.
        
        Args:
            symbol: Stock symbol
            
        Returns:
            Dictionary with fundamental metrics
        """
        try:
            ticker = yf.Ticker(symbol)
            info = ticker.info
            
            # Extract key fundamental metrics
            fundamental_data = {
                'symbol': symbol,
                'sector': self.sector_mapping.get(symbol, 'Unknown'),
                'market_cap': info.get('marketCap', 0),
                'pe_ratio': info.get('trailingPE', np.nan),
                'pb_ratio': info.get('priceToBook', np.nan),
                'dividend_yield': info.get('dividendYield', 0) * 100 if info.get('dividendYield') else 0,
                'roe': info.get('returnOnEquity', np.nan),
                'debt_to_equity': info.get('debtToEquity', np.nan),
                'current_ratio': info.get('currentRatio', np.nan),
                'revenue_growth': info.get('revenueGrowth', np.nan),
                'earnings_growth': info.get('earningsGrowth', np.nan),
                'book_value': info.get('bookValue', np.nan),
                'enterprise_value': info.get('enterpriseValue', np.nan),
                'price_to_sales': info.get('priceToSalesTrailing12Months', np.nan)
            }
            
            return fundamental_data
            
        except Exception as e:
            print(f"Error fetching fundamental data for {symbol}: {str(e)}")
            return {
                'symbol': symbol,
                'sector': self.sector_mapping.get(symbol, 'Unknown'),
                'market_cap': 0,
                'pe_ratio': np.nan,
                'pb_ratio': np.nan,
                'dividend_yield': 0,
                'roe': np.nan,
                'debt_to_equity': np.nan,
                'current_ratio': np.nan,
                'revenue_growth': np.nan,
                'earnings_growth': np.nan,
                'book_value': np.nan,
                'enterprise_value': np.nan,
                'price_to_sales': np.nan
            }
    
    def calculate_volatility(self, data: pd.DataFrame, window: int = 252) -> float:
        """
        Calculate annualized volatility for a stock.
        
        Args:
            data: Stock price data
            window: Rolling window for calculation
            
        Returns:
            Annualized volatility as percentage
        """
        if data.empty or 'close' not in data.columns:
            return 0.0
        
        returns = data['close'].pct_change().dropna()
        volatility = returns.std() * np.sqrt(252) * 100  # Annualized volatility in %
        
        return volatility
    
    def calculate_beta(self, stock_data: pd.DataFrame, market_data: pd.DataFrame) -> float:
        """
        Calculate beta relative to market.
        
        Args:
            stock_data: Individual stock price data
            market_data: Market/benchmark price data
            
        Returns:
            Beta coefficient
        """
        try:
            if stock_data.empty or market_data.empty:
                return 1.0
            
            # Calculate returns
            stock_returns = stock_data['close'].pct_change().dropna()
            market_returns = market_data['close'].pct_change().dropna()
            
            # Align the data
            aligned_data = pd.concat([stock_returns, market_returns], axis=1, join='inner')
            aligned_data.columns = ['stock', 'market']
            aligned_data = aligned_data.dropna()
            
            if len(aligned_data) < 10:  # Need minimum data points
                return 1.0
            
            # Calculate beta
            covariance = np.cov(aligned_data['stock'], aligned_data['market'])[0, 1]
            market_variance = np.var(aligned_data['market'])
            
            if market_variance == 0:
                return 1.0
            
            beta = covariance / market_variance
            return beta
            
        except Exception as e:
            print(f"Error calculating beta: {str(e)}")
            return 1.0
    
    def get_nifty_50_data(self, start_date: str = "2020-01-01", 
                         end_date: Optional[str] = None) -> Dict[str, pd.DataFrame]:
        """
        Fetch data for all NIFTY 50 stocks.
        
        Args:
            start_date: Start date for data fetch
            end_date: End date for data fetch
            
        Returns:
            Dictionary mapping symbols to their data
        """
        return self.fetch_multiple_stocks(self.nifty_50_stocks, start_date, end_date)
    
    def get_nifty_index_data(self, start_date: str = "2020-01-01",
                           end_date: Optional[str] = None) -> Optional[pd.DataFrame]:
        """
        Fetch NIFTY 50 index data.
        
        Args:
            start_date: Start date for data fetch
            end_date: End date for data fetch
            
        Returns:
            DataFrame with NIFTY 50 index data
        """
        return self.fetch_stock_data("^NSEI", start_date, end_date)
    
    def calculate_returns_matrix(self, stock_data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """
        Calculate returns matrix for multiple stocks.
        
        Args:
            stock_data: Dictionary of stock data
            
        Returns:
            DataFrame with returns for each stock
        """
        returns_dict = {}
        
        for symbol, data in stock_data.items():
            if not data.empty and 'close' in data.columns:
                returns = data['close'].pct_change().dropna()
                returns_dict[symbol] = returns
        
        if not returns_dict:
            return pd.DataFrame()
        
        returns_df = pd.DataFrame(returns_dict)
        return returns_df.dropna()
    
    def get_stock_metrics_summary(self, symbols: List[str]) -> pd.DataFrame:
        """
        Get summary metrics for a list of stocks.
        
        Args:
            symbols: List of stock symbols
            
        Returns:
            DataFrame with summary metrics
        """
        summary_data = []
        
        for symbol in symbols:
            # Get price data
            price_data = self.fetch_stock_data(symbol, start_date="2023-01-01")
            
            # Get fundamental data
            fundamental = self.get_fundamental_data(symbol)
            
            # Calculate metrics
            if price_data is not None and not price_data.empty:
                volatility = self.calculate_volatility(price_data)
                current_price = price_data['close'].iloc[-1]
                
                # Calculate returns
                returns = price_data['close'].pct_change().dropna()
                annual_return = (1 + returns.mean()) ** 252 - 1
            else:
                volatility = 0
                current_price = 0
                annual_return = 0
            
            summary_data.append({
                'Symbol': symbol,
                'Sector': fundamental['sector'],
                'Current_Price': current_price,
                'Market_Cap': fundamental['market_cap'],
                'PE_Ratio': fundamental['pe_ratio'],
                'PB_Ratio': fundamental['pb_ratio'],
                'Dividend_Yield': fundamental['dividend_yield'],
                'Volatility': volatility,
                'Annual_Return': annual_return * 100,
                'ROE': fundamental['roe']
            })
        
        return pd.DataFrame(summary_data)
