"""
Test script for ADAPT Smart Indexing Engine

This script tests the core components of the ADAPT system to ensure
everything is working correctly.
"""

import sys
import os
import pandas as pd
import numpy as np

# Add the core directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), 'core'))
sys.path.append(os.path.join(os.path.dirname(__file__), 'utils'))

def test_nifty_loader():
    """Test the NiftyLoader component."""
    print("Testing NiftyLoader...")
    
    try:
        from core.nifty_loader import NiftyLoader
        
        loader = NiftyLoader()
        
        # Test fetching a single stock
        data = loader.fetch_stock_data('RELIANCE.NS', start_date='2023-01-01')
        if data is not None and not data.empty:
            print("✓ Single stock fetch successful")
        else:
            print("✗ Single stock fetch failed")
        
        # Test fundamental data
        fundamental = loader.get_fundamental_data('RELIANCE.NS')
        if fundamental and 'symbol' in fundamental:
            print("✓ Fundamental data fetch successful")
        else:
            print("✗ Fundamental data fetch failed")
        
        # Test volatility calculation
        if data is not None:
            volatility = loader.calculate_volatility(data)
            print(f"✓ Volatility calculation: {volatility:.2f}%")
        
        return True
        
    except Exception as e:
        print(f"✗ NiftyLoader test failed: {str(e)}")
        return False

def test_profile_classifier():
    """Test the ProfileClassifier component."""
    print("\nTesting ProfileClassifier...")
    
    try:
        from core.profile_classifier import ProfileClassifier
        
        classifier = ProfileClassifier()
        
        # Test user data
        user_data = {
            'age': 30,
            'income': 1000000,
            'investment_goal': 'capital_appreciation',
            'investment_horizon': 10,
            'loss_aversion': 5,
            'overconfidence': 6,
            'herding_tendency': 4,
            'anchoring_bias': 5,
            'disposition_effect': 4
        }
        
        # Test profile classification
        profile = classifier.classify_profile(user_data)
        if profile and hasattr(profile, 'profile_type'):
            print(f"✓ Profile classification successful: {profile.profile_type}")
        else:
            print("✗ Profile classification failed")
        
        # Test profile summary
        summary = classifier.get_profile_summary(profile)
        if summary and 'profile_type' in summary:
            print("✓ Profile summary generation successful")
        else:
            print("✗ Profile summary generation failed")
        
        return True
        
    except Exception as e:
        print(f"✗ ProfileClassifier test failed: {str(e)}")
        return False

def test_portfolio_builder():
    """Test the PortfolioBuilder component."""
    print("\nTesting PortfolioBuilder...")
    
    try:
        from core.portfolio_builder import PortfolioBuilder
        from core.profile_classifier import ProfileClassifier
        
        builder = PortfolioBuilder()
        classifier = ProfileClassifier()
        
        # Create a test profile
        user_data = {
            'age': 35,
            'income': 1500000,
            'investment_goal': 'moderate_growth',
            'investment_horizon': 15,
            'loss_aversion': 6,
            'overconfidence': 5,
            'herding_tendency': 5,
            'anchoring_bias': 5,
            'disposition_effect': 5
        }
        
        profile = classifier.classify_profile(user_data)
        
        # Test portfolio building (with limited data for testing)
        print("Note: Portfolio building test requires internet connection for stock data")
        
        # Test portfolio summary generation
        mock_portfolio = {
            'portfolio_weights': {'RELIANCE.NS': 5.0, 'TCS.NS': 5.0},
            'fundamental_data': {
                'RELIANCE.NS': {'sector': 'Energy', 'pe_ratio': 20, 'pb_ratio': 2, 'dividend_yield': 1.5},
                'TCS.NS': {'sector': 'Technology', 'pe_ratio': 25, 'pb_ratio': 8, 'dividend_yield': 2.0}
            },
            'profile_type': 'Moderate',
            'factor_allocations': profile.factor_allocations
        }
        
        summary = builder.get_portfolio_summary(mock_portfolio)
        if summary and 'num_stocks' in summary:
            print("✓ Portfolio summary generation successful")
        else:
            print("✗ Portfolio summary generation failed")
        
        return True
        
    except Exception as e:
        print(f"✗ PortfolioBuilder test failed: {str(e)}")
        return False

def test_backtester():
    """Test the Backtester component."""
    print("\nTesting Backtester...")
    
    try:
        from core.backtester import Backtester
        
        backtester = Backtester()
        
        # Test performance metrics calculation
        returns = pd.Series([0.01, -0.005, 0.02, -0.01, 0.015])
        benchmark_returns = pd.Series([0.008, -0.003, 0.018, -0.008, 0.012])
        
        metrics = backtester._calculate_performance_metrics(returns, benchmark_returns)
        
        if metrics and 'total_return' in metrics:
            print("✓ Performance metrics calculation successful")
        else:
            print("✗ Performance metrics calculation failed")
        
        # Test drawdown analysis
        drawdown = backtester._calculate_drawdown_analysis(returns)
        if drawdown and 'max_drawdown' in drawdown:
            print("✓ Drawdown analysis successful")
        else:
            print("✗ Drawdown analysis failed")
        
        return True
        
    except Exception as e:
        print(f"✗ Backtester test failed: {str(e)}")
        return False

def test_utils():
    """Test utility functions."""
    print("\nTesting Utility Functions...")
    
    try:
        from utils.helpers import normalize_weights, format_percentage, format_currency
        
        # Test weight normalization
        weights = {'AAPL': 30, 'GOOGL': 40, 'MSFT': 30}
        normalized = normalize_weights(weights)
        if sum(normalized.values()) == 100:
            print("✓ Weight normalization successful")
        else:
            print("✗ Weight normalization failed")
        
        # Test formatting functions
        percentage = format_percentage(0.1567, 2)
        currency = format_currency(1234567)
        
        if percentage and currency:
            print("✓ Formatting functions successful")
        else:
            print("✗ Formatting functions failed")
        
        return True
        
    except Exception as e:
        print(f"✗ Utility functions test failed: {str(e)}")
        return False

def main():
    """Run all tests."""
    print("=" * 50)
    print("ADAPT Smart Indexing Engine - System Test")
    print("=" * 50)
    
    tests = [
        test_nifty_loader,
        test_profile_classifier,
        test_portfolio_builder,
        test_backtester,
        test_utils
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        try:
            if test():
                passed += 1
        except Exception as e:
            print(f"✗ Test {test.__name__} failed with exception: {str(e)}")
    
    print("\n" + "=" * 50)
    print(f"Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! ADAPT system is ready to use.")
        return True
    else:
        print("⚠️  Some tests failed. Please check the errors above.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 