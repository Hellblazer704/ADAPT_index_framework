"""
ADAPT Smart Indexing Engine - Portfolio Builder

This module builds personalized portfolios based on user profiles,
factor allocations, and market conditions.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from .nifty_loader import NiftyLoader
from .profile_classifier import UserProfile
import warnings
warnings.filterwarnings('ignore')

class PortfolioBuilder:
    """Builds personalized portfolios based on user profiles and factor models."""
    
    def __init__(self):
        """Initialize the portfolio builder."""
        self.data_loader = NiftyLoader()
        
        # ESG scores (mock data for demonstration)
        self.esg_scores = {
            'TCS.NS': 85, 'INFOSYS.NS': 82, 'HCLTECH.NS': 78, 'WIPRO.NS': 75,
            'HINDUNILVR.NS': 88, 'NESTLEIND.NS': 85, 'ITC.NS': 70,
            'RELIANCE.NS': 65, 'ONGC.NS': 55, 'COALINDIA.NS': 50,
            'HDFCBANK.NS': 80, 'ICICIBANK.NS': 78, 'KOTAKBANK.NS': 82,
            'SUNPHARMA.NS': 75, 'CIPLA.NS': 78, 'DRREDDY.NS': 76,
            'TITAN.NS': 70, 'ASIANPAINT.NS': 72, 'BAJFINANCE.NS': 68
        }
    
    def calculate_factor_scores(self, stock_data: Dict[str, pd.DataFrame],
                               fundamental_data: Dict[str, Dict]) -> pd.DataFrame:
        """
        Calculate factor scores for all stocks.
        
        Args:
            stock_data: Dictionary of stock price data
            fundamental_data: Dictionary of fundamental metrics
            
        Returns:
            DataFrame with factor scores for each stock
        """
        scores = []
        
        for symbol in stock_data.keys():
            if symbol not in fundamental_data:
                continue
                
            fundamental = fundamental_data[symbol]
            price_data = stock_data[symbol]
            
            if price_data.empty:
                continue
            
            # Calculate factor scores
            factor_score = self._calculate_single_stock_factors(symbol, price_data, fundamental)
            scores.append(factor_score)
        
        if not scores:
            return pd.DataFrame()
        
        df = pd.DataFrame(scores)
        return df.set_index('symbol')
    
    def _calculate_single_stock_factors(self, symbol: str, price_data: pd.DataFrame,
                                      fundamental: Dict) -> Dict[str, Any]:
        """
        Calculate factor scores for a single stock.
        
        Args:
            symbol: Stock symbol
            price_data: Price history data
            fundamental: Fundamental metrics
            
        Returns:
            Dictionary with factor scores
        """
        # Initialize scores
        scores = {'symbol': symbol}
        
        # Quality factor (based on ROE, PE ratio, debt levels)
        roe = fundamental.get('roe', 0)
        pe_ratio = fundamental.get('pe_ratio', np.nan)
        debt_to_equity = fundamental.get('debt_to_equity', np.nan)
        
        quality_score = 0
        if not pd.isna(roe) and roe > 0:
            quality_score += min(roe * 5, 50)  # Cap at 50
        if not pd.isna(pe_ratio) and 10 <= pe_ratio <= 25:
            quality_score += 25
        if not pd.isna(debt_to_equity) and debt_to_equity < 0.5:
            quality_score += 25
        
        scores['quality'] = quality_score
        
        # Value factor (based on PE, PB ratios)
        pb_ratio = fundamental.get('pb_ratio', np.nan)
        
        value_score = 0
        if not pd.isna(pe_ratio) and pe_ratio < 20:
            value_score += (20 - pe_ratio) * 2.5
        if not pd.isna(pb_ratio) and pb_ratio < 3:
            value_score += (3 - pb_ratio) * 15
        
        scores['value'] = max(0, value_score)
        
        # Momentum factor (based on recent price performance)
        if len(price_data) >= 60:  # Need at least 60 days
            recent_return = (price_data['close'].iloc[-1] / price_data['close'].iloc[-60] - 1) * 100
            momentum_score = max(0, min(100, recent_return + 50))  # Normalize to 0-100
        else:
            momentum_score = 50  # Neutral score
        
        scores['momentum'] = momentum_score
        
        # Low volatility factor
        if len(price_data) >= 30:
            returns = price_data['close'].pct_change().dropna()
            volatility = returns.std() * np.sqrt(252) * 100  # Annualized volatility
            low_vol_score = max(0, 100 - volatility * 2)  # Higher score for lower volatility
        else:
            low_vol_score = 50
        
        scores['low_volatility'] = low_vol_score
        
        # Size factor (based on market cap)
        market_cap = fundamental.get('market_cap', 0)
        if market_cap > 0:
            # Larger companies get lower size factor scores (size premium for small caps)
            if market_cap > 1000000000000:  # 1 trillion
                size_score = 20
            elif market_cap > 500000000000:  # 500 billion
                size_score = 40
            elif market_cap > 100000000000:  # 100 billion
                size_score = 60
            else:
                size_score = 80
        else:
            size_score = 50
        
        scores['size'] = size_score
        
        # ESG factor
        esg_score = self.esg_scores.get(symbol, 60)  # Default to 60 if not found
        scores['esg_ecology'] = esg_score
        
        # Dividend yield (part of quality/income factor)
        dividend_yield = fundamental.get('dividend_yield', 0)
        scores['dividend_yield'] = min(dividend_yield * 20, 100)  # Scale and cap
        
        return scores
    
    def build_portfolio(self, user_profile: UserProfile, universe: str = 'nifty_50',
                       max_stocks: int = 30) -> Dict[str, Any]:
        """
        Build a personalized portfolio based on user profile.
        
        Args:
            user_profile: User's investment profile
            universe: Stock universe ('nifty_50' or 'nifty_500')
            max_stocks: Maximum number of stocks in portfolio
            
        Returns:
            Dictionary containing portfolio weights and metadata
        """
        try:
            # Get stock universe
            if universe == 'nifty_50':
                stock_symbols = self.data_loader.nifty_50_stocks
            else:
                # For simplicity, use NIFTY 50 as base universe
                stock_symbols = self.data_loader.nifty_50_stocks[:30]  # Sample subset
            
            # Fetch stock data
            stock_data = self.data_loader.fetch_multiple_stocks(
                stock_symbols, start_date="2023-01-01"
            )
            
            # Get fundamental data
            fundamental_data = {}
            for symbol in stock_symbols:
                fundamental_data[symbol] = self.data_loader.get_fundamental_data(symbol)
            
            # Calculate factor scores
            factor_scores = self.calculate_factor_scores(stock_data, fundamental_data)
            
            if factor_scores.empty:
                raise ValueError("No factor scores calculated")
            
            # Build portfolio weights based on profile
            portfolio_weights = self._optimize_portfolio_weights(
                factor_scores, user_profile, max_stocks
            )
            
            # Get benchmark data (NIFTY 50)
            benchmark_data = self.data_loader.get_nifty_index_data(start_date="2023-01-01")
            
            return {
                'portfolio_weights': portfolio_weights,
                'factor_scores': factor_scores,
                'fundamental_data': fundamental_data,
                'stock_data': stock_data,
                'benchmark_data': benchmark_data,
                'profile_type': user_profile.profile_type,
                'factor_allocations': user_profile.factor_allocations,
                'universe': universe,
                'max_stocks': max_stocks
            }
            
        except Exception as e:
            print(f"Error building portfolio: {str(e)}")
            # Return a simple equal-weight portfolio as fallback
            fallback_stocks = ['RELIANCE.NS', 'TCS.NS', 'HDFCBANK.NS', 'INFY.NS', 'ICICIBANK.NS']
            fallback_weights = {stock: 20.0 for stock in fallback_stocks}
            
            return {
                'portfolio_weights': fallback_weights,
                'factor_scores': pd.DataFrame(),
                'fundamental_data': {},
                'stock_data': {},
                'benchmark_data': None,
                'profile_type': user_profile.profile_type,
                'factor_allocations': user_profile.factor_allocations,
                'universe': universe,
                'max_stocks': max_stocks
            }
    
    def _optimize_portfolio_weights(self, factor_scores: pd.DataFrame,
                                  user_profile: UserProfile, max_stocks: int) -> Dict[str, float]:
        """
        Optimize portfolio weights based on factor scores and user profile.
        
        Args:
            factor_scores: DataFrame with factor scores
            user_profile: User investment profile
            max_stocks: Maximum number of stocks
            
        Returns:
            Dictionary of stock weights
        """
        if factor_scores.empty:
            return {}
        
        # Calculate composite scores based on factor allocations
        factor_allocations = user_profile.factor_allocations
        
        # Normalize factor scores to 0-1 range for each factor
        normalized_scores = factor_scores.copy()
        for factor in ['quality', 'value', 'momentum', 'low_volatility', 'size', 'esg_ecology']:
            if factor in normalized_scores.columns:
                min_val = normalized_scores[factor].min()
                max_val = normalized_scores[factor].max()
                if max_val > min_val:
                    normalized_scores[factor] = (normalized_scores[factor] - min_val) / (max_val - min_val)
                else:
                    normalized_scores[factor] = 0.5
        
        # Calculate composite score
        composite_scores = pd.Series(0.0, index=normalized_scores.index)
        
        for factor, allocation in factor_allocations.items():
            if factor in normalized_scores.columns:
                composite_scores += normalized_scores[factor] * (allocation / 100)
        
        # Apply behavioral adjustments
        if hasattr(user_profile, 'behavioral_adjustments'):
            for adjustment, value in user_profile.behavioral_adjustments.items():
                if 'increase_low_volatility' in adjustment and 'low_volatility' in normalized_scores.columns:
                    composite_scores += normalized_scores['low_volatility'] * (value / 100)
                elif 'increase_momentum' in adjustment and 'momentum' in normalized_scores.columns:
                    composite_scores += normalized_scores['momentum'] * (value / 100)
                elif 'increase_quality' in adjustment and 'quality' in normalized_scores.columns:
                    composite_scores += normalized_scores['quality'] * (value / 100)
        
        # Select top stocks
        top_stocks = composite_scores.nlargest(max_stocks)
        
        if len(top_stocks) == 0:
            return {}
        
        # Apply risk-based weighting
        if user_profile.profile_type == 'Conservative':
            # More equal weighting for conservative profiles
            weights = pd.Series(1.0, index=top_stocks.index)
        elif user_profile.profile_type == 'Moderate':
            # Moderate concentration
            weights = top_stocks ** 0.5
        else:  # Aggressive
            # Allow higher concentration in top picks
            weights = top_stocks ** 1.2
        
        # Normalize weights to sum to 100%
        weights = weights / weights.sum() * 100
        
        # Apply maximum single position limit based on profile
        max_position = {
            'Conservative': 8.0,  # Max 8% in any single stock
            'Moderate': 12.0,     # Max 12% in any single stock
            'Aggressive': 15.0    # Max 15% in any single stock
        }
        
        position_limit = max_position[user_profile.profile_type]
        
        # Cap positions and redistribute excess
        while weights.max() > position_limit:
            excess_stocks = weights[weights > position_limit]
            excess_weight = (excess_stocks - position_limit).sum()
            
            # Cap the excess stocks
            weights[weights > position_limit] = position_limit
            
            # Redistribute excess to other stocks
            eligible_stocks = weights[weights < position_limit]
            if len(eligible_stocks) > 0:
                redistribution = excess_weight / len(eligible_stocks)
                weights[weights < position_limit] += redistribution
            else:
                break
        
        # Final normalization
        weights = weights / weights.sum() * 100
        
        return weights.to_dict()
    
    def get_portfolio_summary(self, portfolio: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate a comprehensive portfolio summary.
        
        Args:
            portfolio: Portfolio dictionary from build_portfolio
            
        Returns:
            Summary dictionary
        """
        weights = portfolio.get('portfolio_weights', {})
        fundamental_data = portfolio.get('fundamental_data', {})
        
        if not weights:
            return {'error': 'No portfolio weights available'}
        
        # Basic statistics
        num_stocks = len(weights)
        
        # Calculate weighted averages
        total_weight = sum(weights.values())
        weighted_pe = 0
        weighted_pb = 0
        weighted_dividend_yield = 0
        
        sector_allocation = {}
        
        for stock, weight in weights.items():
            if stock in fundamental_data:
                fundamental = fundamental_data[stock]
                weight_fraction = weight / total_weight
                
                # PE ratio
                pe = fundamental.get('pe_ratio', 0)
                if not pd.isna(pe) and pe > 0:
                    weighted_pe += pe * weight_fraction
                
                # PB ratio
                pb = fundamental.get('pb_ratio', 0)
                if not pd.isna(pb) and pb > 0:
                    weighted_pb += pb * weight_fraction
                
                # Dividend yield
                div_yield = fundamental.get('dividend_yield', 0)
                weighted_dividend_yield += div_yield * weight_fraction
                
                # Sector allocation
                sector = fundamental.get('sector', 'Unknown')
                sector_allocation[sector] = sector_allocation.get(sector, 0) + weight
        
        # Top holdings
        sorted_weights = sorted(weights.items(), key=lambda x: x[1], reverse=True)
        top_holdings = dict(sorted_weights[:10])
        
        return {
            'profile_type': portfolio.get('profile_type', 'Unknown'),
            'num_stocks': num_stocks,
            'avg_pe_ratio': weighted_pe,
            'avg_pb_ratio': weighted_pb,
            'avg_dividend_yield': weighted_dividend_yield,
            'top_holdings': top_holdings,
            'sector_allocation': sector_allocation,
            'factor_allocations': portfolio.get('factor_allocations', {}),
            'total_weight': total_weight
        }
    
    def rebalance_portfolio(self, current_weights: Dict[str, float],
                          target_weights: Dict[str, float],
                          threshold: float = 5.0) -> Dict[str, float]:
        """
        Calculate rebalancing trades.
        
        Args:
            current_weights: Current portfolio weights
            target_weights: Target portfolio weights
            threshold: Rebalancing threshold (%)
            
        Returns:
            Dictionary of required trades (positive = buy, negative = sell)
        """
        trades = {}
        
        # Get all stocks (current and target)
        all_stocks = set(current_weights.keys()) | set(target_weights.keys())
        
        for stock in all_stocks:
            current_weight = current_weights.get(stock, 0)
            target_weight = target_weights.get(stock, 0)
            
            difference = target_weight - current_weight
            
            # Only trade if difference exceeds threshold
            if abs(difference) > threshold:
                trades[stock] = difference
        
        return trades
