"""
ADAPT Smart Indexing Engine - Helper Functions

This module contains utility functions for data processing, formatting,
and portfolio calculations.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional
import csv
import io

def normalize_weights(weights: Dict[str, float]) -> Dict[str, float]:
    """
    Normalize portfolio weights to sum to 100%.
    
    Args:
        weights: Dictionary of stock symbols and their weights
        
    Returns:
        Dictionary with normalized weights
    """
    if not weights:
        return {}
    
    total = sum(weights.values())
    if total == 0:
        return weights
    
    return {symbol: (weight / total) * 100 for symbol, weight in weights.items()}

def format_percentage(value: float, decimals: int = 2) -> str:
    """
    Format a decimal value as a percentage string.
    
    Args:
        value: Decimal value (e.g., 0.1567)
        decimals: Number of decimal places
        
    Returns:
        Formatted percentage string (e.g., "15.67%")
    """
    if pd.isna(value):
        return "N/A"
    return f"{value * 100:.{decimals}f}%"

def format_currency(value: float, currency: str = "₹") -> str:
    """
    Format a value as currency.
    
    Args:
        value: Numeric value
        currency: Currency symbol
        
    Returns:
        Formatted currency string
    """
    if pd.isna(value):
        return "N/A"
    
    if value >= 10000000:  # 1 crore
        return f"{currency}{value/10000000:.2f} Cr"
    elif value >= 100000:  # 1 lakh
        return f"{currency}{value/100000:.2f} L"
    else:
        return f"{currency}{value:,.2f}"

def calculate_portfolio_metrics(returns: pd.Series, benchmark_returns: pd.Series = None) -> Dict[str, float]:
    """
    Calculate comprehensive portfolio metrics.
    
    Args:
        returns: Portfolio returns series
        benchmark_returns: Benchmark returns series (optional)
        
    Returns:
        Dictionary of calculated metrics
    """
    if returns.empty:
        return {}
    
    metrics = {}
    
    # Basic return metrics
    metrics['total_return'] = (1 + returns).prod() - 1
    metrics['annualized_return'] = (1 + returns).prod() ** (252 / len(returns)) - 1
    metrics['volatility'] = returns.std() * np.sqrt(252)
    
    # Risk metrics
    metrics['sharpe_ratio'] = metrics['annualized_return'] / metrics['volatility'] if metrics['volatility'] > 0 else 0
    
    # Downside metrics
    negative_returns = returns[returns < 0]
    if len(negative_returns) > 0:
        downside_volatility = negative_returns.std() * np.sqrt(252)
        metrics['sortino_ratio'] = metrics['annualized_return'] / downside_volatility if downside_volatility > 0 else 0
    else:
        metrics['sortino_ratio'] = float('inf')
    
    # Drawdown analysis
    cumulative = (1 + returns).cumprod()
    running_max = cumulative.expanding().max()
    drawdown = (cumulative - running_max) / running_max
    metrics['max_drawdown'] = drawdown.min()
    
    # Value at Risk (95%)
    metrics['var_95'] = np.percentile(returns, 5)
    metrics['cvar_95'] = returns[returns <= metrics['var_95']].mean()
    
    # Benchmark comparison if provided
    if benchmark_returns is not None and not benchmark_returns.empty:
        # Align the series
        aligned_returns, aligned_benchmark = returns.align(benchmark_returns, join='inner')
        
        if len(aligned_returns) > 1 and len(aligned_benchmark) > 1:
            covariance = np.cov(aligned_returns, aligned_benchmark)[0, 1]
            benchmark_variance = aligned_benchmark.var()
            
            if benchmark_variance > 0:
                metrics['beta'] = covariance / benchmark_variance
                
                benchmark_annual_return = (1 + aligned_benchmark).prod() ** (252 / len(aligned_benchmark)) - 1
                metrics['alpha'] = metrics['annualized_return'] - benchmark_annual_return
                
                tracking_error = (aligned_returns - aligned_benchmark).std() * np.sqrt(252)
                metrics['tracking_error'] = tracking_error
                
                if tracking_error > 0:
                    metrics['information_ratio'] = metrics['alpha'] / tracking_error
                else:
                    metrics['information_ratio'] = 0
    
    return metrics

def generate_portfolio_report(portfolio_data: Dict[str, Any]) -> str:
    """
    Generate a comprehensive portfolio report.
    
    Args:
        portfolio_data: Portfolio information and metrics
        
    Returns:
        Formatted report string
    """
    report = []
    report.append("ADAPT SMART INDEXING ENGINE - PORTFOLIO REPORT")
    report.append("=" * 60)
    
    # Portfolio overview
    if 'profile_type' in portfolio_data:
        report.append(f"Risk Profile: {portfolio_data['profile_type']}")
    
    if 'num_stocks' in portfolio_data:
        report.append(f"Number of Holdings: {portfolio_data['num_stocks']}")
    
    # Performance metrics
    if 'performance_metrics' in portfolio_data:
        metrics = portfolio_data['performance_metrics']
        report.append("\nPERFORMANCE METRICS:")
        report.append("-" * 30)
        
        for key, value in metrics.items():
            if isinstance(value, float):
                if 'return' in key.lower() or 'ratio' in key.lower():
                    report.append(f"{key.replace('_', ' ').title()}: {format_percentage(value)}")
                else:
                    report.append(f"{key.replace('_', ' ').title()}: {value:.4f}")
            else:
                report.append(f"{key.replace('_', ' ').title()}: {value}")
    
    # Top holdings
    if 'top_holdings' in portfolio_data:
        report.append("\nTOP HOLDINGS:")
        report.append("-" * 30)
        for stock, weight in list(portfolio_data['top_holdings'].items())[:10]:
            report.append(f"{stock}: {weight:.2f}%")
    
    return "\n".join(report)

def export_to_csv(data: Dict[str, Any], filename: str = "portfolio_data.csv") -> str:
    """
    Export portfolio data to CSV format.
    
    Args:
        data: Data to export
        filename: Output filename
        
    Returns:
        CSV string content
    """
    output = io.StringIO()
    writer = csv.writer(output)
    
    # Write headers and data based on data structure
    if isinstance(data, dict):
        # Write as key-value pairs
        writer.writerow(['Metric', 'Value'])
        for key, value in data.items():
            writer.writerow([key, value])
    elif isinstance(data, pd.DataFrame):
        # Write dataframe
        data.to_csv(output, index=True)
    else:
        # Convert to string representation
        writer.writerow(['Data'])
        writer.writerow([str(data)])
    
    return output.getvalue()

def validate_user_input(user_data: Dict[str, Any]) -> List[str]:
    """
    Validate user input data.
    
    Args:
        user_data: User profile data
        
    Returns:
        List of validation errors (empty if valid)
    """
    errors = []
    
    # Age validation
    if 'age' in user_data:
        age = user_data['age']
        if not isinstance(age, (int, float)) or age < 18 or age > 100:
            errors.append("Age must be between 18 and 100")
    
    # Income validation
    if 'income' in user_data:
        income = user_data['income']
        if not isinstance(income, (int, float)) or income < 0:
            errors.append("Income must be a positive number")
    
    # Investment horizon validation
    if 'investment_horizon' in user_data:
        horizon = user_data['investment_horizon']
        if not isinstance(horizon, (int, float)) or horizon < 1 or horizon > 50:
            errors.append("Investment horizon must be between 1 and 50 years")
    
    # Behavioral scores validation (1-10 scale)
    behavioral_fields = ['loss_aversion', 'overconfidence', 'herding_tendency', 'anchoring_bias', 'disposition_effect']
    for field in behavioral_fields:
        if field in user_data:
            score = user_data[field]
            if not isinstance(score, (int, float)) or score < 1 or score > 10:
                errors.append(f"{field.replace('_', ' ').title()} must be between 1 and 10")
    
    return errors

def calculate_sector_allocation(holdings: Dict[str, float], sector_data: Dict[str, str]) -> Dict[str, float]:
    """
    Calculate sector allocation from portfolio holdings.
    
    Args:
        holdings: Stock symbols and their weights
        sector_data: Mapping of stock symbols to sectors
        
    Returns:
        Dictionary of sector allocations
    """
    sector_allocation = {}
    
    for stock, weight in holdings.items():
        sector = sector_data.get(stock, 'Unknown')
        sector_allocation[sector] = sector_allocation.get(sector, 0) + weight
    
    return sector_allocation

def rebalance_frequency_to_days(frequency: str) -> int:
    """
    Convert rebalance frequency string to number of days.
    
    Args:
        frequency: Frequency string ('daily', 'weekly', 'monthly', 'quarterly', 'yearly')
        
    Returns:
        Number of days
    """
    frequency_map = {
        'daily': 1,
        'weekly': 7,
        'monthly': 30,
        'quarterly': 90,
        'yearly': 365
    }
    
    return frequency_map.get(frequency.lower(), 90)  # Default to quarterly

def generate_synthetic_returns(num_periods: int, annual_return: float = 0.08, 
                             annual_volatility: float = 0.15, seed: int = None) -> pd.Series:
    """
    Generate synthetic return series for testing.
    
    Args:
        num_periods: Number of periods to generate
        annual_return: Expected annual return
        annual_volatility: Annual volatility
        seed: Random seed for reproducibility
        
    Returns:
        Pandas Series of returns
    """
    if seed is not None:
        np.random.seed(seed)
    
    # Convert annual metrics to daily
    daily_return = annual_return / 252
    daily_volatility = annual_volatility / np.sqrt(252)
    
    # Generate random returns
    returns = np.random.normal(daily_return, daily_volatility, num_periods)
    
    # Create date index
    dates = pd.date_range(start='2020-01-01', periods=num_periods, freq='D')
    
    return pd.Series(returns, index=dates)
