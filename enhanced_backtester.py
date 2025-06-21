"""
Enhanced Backtesting Engine with Real Yahoo Finance Data and Advanced Trading Logic
"""

import pandas as pd
import numpy as np
import yfinance as yf
from typing import Dict, List, Optional, Tuple, Any
from backtesting import Backtest, Strategy
from backtesting.lib import crossover
import warnings
from datetime import datetime, timedelta
from database import DatabaseManager

warnings.filterwarnings('ignore')

class ADAPTStrategy(Strategy):
    """Enhanced trading strategy with proper position sizing and risk management"""
    
    # Strategy parameters
    sma_short = 20
    sma_long = 50
    rsi_period = 14
    rsi_oversold = 30
    rsi_overbought = 70
    position_size = 0.1  # 10% of capital per trade
    stop_loss = 0.02  # 2% stop loss
    take_profit = 0.06  # 6% take profit
    
    def init(self):
        """Initialize strategy indicators"""
        # Calculate technical indicators with proper error handling
        try:
            # Simple Moving Averages
            self.sma_short = self.I(self.SMA, self.data.Close, self.sma_short)
            self.sma_long = self.I(self.SMA, self.data.Close, self.sma_long)
            
            # RSI for momentum
            self.rsi = self.I(self.RSI, self.data.Close, self.rsi_period)
            
            # Bollinger Bands for volatility
            self.bb_upper, self.bb_middle, self.bb_lower = self.I(
                self.bollinger_bands, self.data.Close, 20, 2
            )
            
        except Exception as e:
            print(f"Error initializing indicators: {e}")
    
    def SMA(self, values, period):
        """Simple Moving Average"""
        return pd.Series(values).rolling(window=period).mean()
    
    def RSI(self, values, period):
        """Relative Strength Index"""
        delta = pd.Series(values).diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))
    
    def bollinger_bands(self, values, period, std_dev):
        """Bollinger Bands"""
        sma = pd.Series(values).rolling(window=period).mean()
        std = pd.Series(values).rolling(window=period).std()
        upper = sma + (std * std_dev)
        lower = sma - (std * std_dev)
        return upper, sma, lower
    
    def next(self):
        """Execute trading logic on each bar"""
        try:
            # Get current values
            current_price = self.data.Close[-1]
            
            # Ensure we have enough data
            if len(self.data) < max(self.sma_long, self.rsi_period):
                return
            
            # Entry conditions
            bullish_signal = (
                crossover(self.sma_short, self.sma_long) and  # Golden cross
                self.rsi[-1] < self.rsi_overbought and  # Not overbought
                current_price > self.bb_lower[-1]  # Above lower Bollinger Band
            )
            
            bearish_signal = (
                crossover(self.sma_long, self.sma_short) and  # Death cross
                self.rsi[-1] > self.rsi_oversold  # Not oversold
            )
            
            # Position management
            if not self.position:
                if bullish_signal:
                    # Calculate position size based on portfolio value
                    size = self.position_size
                    self.buy(size=size)
                    
            else:
                # Exit conditions
                if bearish_signal:
                    self.position.close()
                
                # Stop loss and take profit
                entry_price = self.position.entry_price
                if entry_price:
                    stop_price = entry_price * (1 - self.stop_loss)
                    profit_price = entry_price * (1 + self.take_profit)
                    
                    if current_price <= stop_price or current_price >= profit_price:
                        self.position.close()
                        
        except Exception as e:
            print(f"Error in trading logic: {e}")

class EnhancedBacktester:
    """Enhanced backtesting engine with real data and advanced analytics"""
    
    def __init__(self):
        self.db_manager = DatabaseManager()
        self.risk_free_rate = 0.06  # 6% annual risk-free rate
        
    def fetch_real_data(self, symbols: List[str], start_date: str, end_date: str) -> Dict[str, pd.DataFrame]:
        """Fetch real-time data from Yahoo Finance"""
        print(f"Fetching real market data for {len(symbols)} symbols...")
        
        stock_data = {}
        
        for symbol in symbols:
            try:
                # Check if data exists in database first
                db_data = self.db_manager.get_stock_data(symbol, start_date, end_date)
                
                if not db_data.empty and self._is_data_recent(db_data):
                    print(f"Using cached data for {symbol}")
                    stock_data[symbol] = db_data
                else:
                    print(f"Fetching fresh data for {symbol}")
                    # Fetch from Yahoo Finance
                    ticker = yf.Ticker(symbol)
                    data = ticker.history(start=start_date, end=end_date)
                    
                    if not data.empty:
                        # Remove timezone info to avoid alignment issues
                        if data.index.tz is not None:
                            data.index = data.index.tz_localize(None)
                        
                        # Ensure proper column names
                        data.columns = ['Open', 'High', 'Low', 'Close', 'Volume']
                        
                        # Calculate adjusted close if missing
                        if 'Adj Close' not in data.columns:
                            data['Adj Close'] = data['Close']
                        
                        stock_data[symbol] = data
                        
                        # Save to database
                        self.db_manager.save_stock_data(symbol, data)
                        
                    else:
                        print(f"No data available for {symbol}")
                        
            except Exception as e:
                print(f"Error fetching data for {symbol}: {e}")
                continue
        
        return stock_data
    
    def _is_data_recent(self, data: pd.DataFrame, hours_threshold: int = 24) -> bool:
        """Check if cached data is recent enough"""
        if data.empty:
            return False
        
        last_date = data.index.max()
        now = datetime.now()
        
        # If last data is from today or yesterday (for weekends), consider it recent
        time_diff = now - last_date.to_pydatetime()
        return time_diff.total_seconds() < (hours_threshold * 3600)
    
    def run_portfolio_backtest(self, portfolio_weights: Dict[str, float], 
                             start_date: str, end_date: str,
                             initial_capital: float = 1000000) -> Dict[str, Any]:
        """Run enhanced backtest on portfolio"""
        
        try:
            # Fetch real market data
            symbols = list(portfolio_weights.keys())
            stock_data = self.fetch_real_data(symbols, start_date, end_date)
            
            if not stock_data:
                raise ValueError("No market data available for backtesting")
            
            # Calculate portfolio returns using real data
            portfolio_returns = self._calculate_portfolio_returns(stock_data, portfolio_weights)
            
            if portfolio_returns.empty:
                raise ValueError("Unable to calculate portfolio returns")
            
            # Run individual stock backtests for detailed analysis
            individual_results = {}
            for symbol, weight in portfolio_weights.items():
                if symbol in stock_data and weight > 0:
                    result = self._run_single_stock_backtest(
                        stock_data[symbol], symbol, initial_capital * (weight / 100)
                    )
                    individual_results[symbol] = result
            
            # Calculate comprehensive metrics
            benchmark_data = self._get_benchmark_data(start_date, end_date)
            performance_metrics = self._calculate_enhanced_metrics(
                portfolio_returns, benchmark_data, initial_capital
            )
            
            # Generate detailed analysis
            analysis = self._generate_detailed_analysis(
                portfolio_returns, individual_results, performance_metrics
            )
            
            return {
                'portfolio_returns': portfolio_returns,
                'portfolio_values': initial_capital * (1 + portfolio_returns).cumprod(),
                'performance_metrics': performance_metrics,
                'individual_results': individual_results,
                'analysis': analysis,
                'backtest_config': {
                    'start_date': start_date,
                    'end_date': end_date,
                    'initial_capital': initial_capital,
                    'num_stocks': len(portfolio_weights),
                    'data_source': 'Yahoo Finance (Real-time)'
                }
            }
            
        except Exception as e:
            print(f"Error in portfolio backtest: {e}")
            return {
                'error': f"Real-time data unavailable: {str(e)}",
                'message': 'Unable to fetch authentic market data. Please try again later or check your internet connection.',
                'data_source': 'Failed - No authentic data available'
            }
    
    def _run_single_stock_backtest(self, data: pd.DataFrame, symbol: str, capital: float) -> Dict[str, Any]:
        """Run detailed backtest on individual stock"""
        try:
            if len(data) < 60:  # Need minimum data for indicators
                return {'error': 'Insufficient data'}
            
            # Prepare data for backtesting library
            bt_data = data.copy()
            bt_data.index.name = 'Date'
            
            # Run backtest with enhanced strategy
            bt = Backtest(bt_data, ADAPTStrategy, cash=capital, commission=0.001)
            result = bt.run()
            
            return {
                'return': result['Return [%]'],
                'buy_hold_return': result['Buy & Hold Return [%]'],
                'max_drawdown': result['Max. Drawdown [%]'],
                'sharpe_ratio': result['Sharpe Ratio'],
                'trades': result['# Trades'],
                'win_rate': result['Win Rate [%]'],
                'profit_factor': result.get('Profit Factor', 1.0),
                'exposure_time': result['Exposure Time [%]']
            }
            
        except Exception as e:
            print(f"Error backtesting {symbol}: {e}")
            return {'error': str(e)}
    
    def _calculate_portfolio_returns(self, stock_data: Dict[str, pd.DataFrame], 
                                   weights: Dict[str, float]) -> pd.Series:
        """Calculate portfolio returns from real stock data"""
        
        # Normalize weights
        total_weight = sum(weights.values())
        if total_weight == 0:
            return pd.Series(dtype=float)
        
        normalized_weights = {stock: weight/total_weight for stock, weight in weights.items()}
        
        # Calculate returns for each stock
        returns_data = {}
        for symbol, weight in normalized_weights.items():
            if symbol in stock_data and not stock_data[symbol].empty:
                stock_returns = stock_data[symbol]['Close'].pct_change().dropna()
                returns_data[symbol] = stock_returns
        
        if not returns_data:
            return pd.Series(dtype=float)
        
        # Combine returns into portfolio
        returns_df = pd.DataFrame(returns_data).fillna(0)
        
        # Calculate weighted portfolio returns
        portfolio_returns = pd.Series(0.0, index=returns_df.index)
        for symbol, weight in normalized_weights.items():
            if symbol in returns_df.columns:
                portfolio_returns += returns_df[symbol] * weight
        
        return portfolio_returns.dropna()
    
    def _get_benchmark_data(self, start_date: str, end_date: str) -> pd.Series:
        """Get benchmark (NIFTY 50) data"""
        try:
            benchmark_data = self.fetch_real_data(['^NSEI'], start_date, end_date)
            if '^NSEI' in benchmark_data and not benchmark_data['^NSEI'].empty:
                return benchmark_data['^NSEI']['Close'].pct_change().dropna()
            return pd.Series(dtype=float)
        except:
            return pd.Series(dtype=float)
    
    def _calculate_enhanced_metrics(self, portfolio_returns: pd.Series, 
                                  benchmark_returns: pd.Series, 
                                  initial_capital: float) -> Dict[str, float]:
        """Calculate comprehensive performance metrics"""
        
        if portfolio_returns.empty:
            return {}
        
        metrics = {}
        
        # Basic return metrics
        total_return = (1 + portfolio_returns).prod() - 1
        num_years = len(portfolio_returns) / 252
        annualized_return = (1 + total_return) ** (1/num_years) - 1 if num_years > 0 else 0
        
        metrics.update({
            'total_return': total_return,
            'annualized_return': annualized_return,
            'volatility': portfolio_returns.std() * np.sqrt(252),
            'final_value': initial_capital * (1 + total_return)
        })
        
        # Risk-adjusted metrics
        excess_returns = portfolio_returns - self.risk_free_rate/252
        if portfolio_returns.std() > 0:
            metrics['sharpe_ratio'] = excess_returns.mean() / portfolio_returns.std() * np.sqrt(252)
        else:
            metrics['sharpe_ratio'] = 0
        
        # Downside metrics
        negative_returns = portfolio_returns[portfolio_returns < 0]
        if len(negative_returns) > 0:
            downside_volatility = negative_returns.std() * np.sqrt(252)
            metrics['sortino_ratio'] = excess_returns.mean() / downside_volatility * np.sqrt(252) if downside_volatility > 0 else 0
        else:
            metrics['sortino_ratio'] = float('inf')
        
        # Drawdown analysis
        cumulative = (1 + portfolio_returns).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        metrics['max_drawdown'] = drawdown.min()
        
        # Value at Risk
        metrics['var_95'] = np.percentile(portfolio_returns, 5)
        metrics['cvar_95'] = portfolio_returns[portfolio_returns <= metrics['var_95']].mean()
        
        # Benchmark comparison
        if not benchmark_returns.empty:
            aligned_portfolio, aligned_benchmark = portfolio_returns.align(benchmark_returns, join='inner')
            
            if len(aligned_portfolio) > 1:
                covariance = np.cov(aligned_portfolio, aligned_benchmark)[0, 1]
                benchmark_variance = aligned_benchmark.var()
                
                if benchmark_variance > 0:
                    metrics['beta'] = covariance / benchmark_variance
                    benchmark_return = (1 + aligned_benchmark).prod() ** (1/num_years) - 1 if num_years > 0 else 0
                    metrics['alpha'] = annualized_return - (self.risk_free_rate + metrics['beta'] * (benchmark_return - self.risk_free_rate))
                    
                    # Information ratio
                    active_returns = aligned_portfolio - aligned_benchmark
                    tracking_error = active_returns.std() * np.sqrt(252)
                    metrics['information_ratio'] = active_returns.mean() / active_returns.std() * np.sqrt(252) if active_returns.std() > 0 else 0
                    metrics['tracking_error'] = tracking_error
        
        return metrics
    
    def _generate_detailed_analysis(self, portfolio_returns: pd.Series, 
                                  individual_results: Dict, 
                                  performance_metrics: Dict) -> Dict[str, Any]:
        """Generate detailed portfolio analysis"""
        
        analysis = {
            'summary': f"Portfolio achieved {performance_metrics.get('annualized_return', 0)*100:.2f}% annualized return with {performance_metrics.get('volatility', 0)*100:.2f}% volatility.",
            'risk_analysis': {},
            'performance_drivers': {},
            'recommendations': []
        }
        
        # Risk analysis
        sharpe = performance_metrics.get('sharpe_ratio', 0)
        max_dd = performance_metrics.get('max_drawdown', 0)
        
        if sharpe > 1.0:
            analysis['risk_analysis']['sharpe_assessment'] = "Excellent risk-adjusted returns"
        elif sharpe > 0.5:
            analysis['risk_analysis']['sharpe_assessment'] = "Good risk-adjusted returns"
        else:
            analysis['risk_analysis']['sharpe_assessment'] = "Poor risk-adjusted returns"
        
        if abs(max_dd) < 0.1:
            analysis['risk_analysis']['drawdown_assessment'] = "Low drawdown risk"
        elif abs(max_dd) < 0.2:
            analysis['risk_analysis']['drawdown_assessment'] = "Moderate drawdown risk"
        else:
            analysis['risk_analysis']['drawdown_assessment'] = "High drawdown risk"
        
        # Performance drivers
        best_performers = []
        worst_performers = []
        
        for symbol, result in individual_results.items():
            if 'return' in result and isinstance(result['return'], (int, float)):
                if result['return'] > 10:
                    best_performers.append((symbol, result['return']))
                elif result['return'] < -5:
                    worst_performers.append((symbol, result['return']))
        
        analysis['performance_drivers']['best_performers'] = sorted(best_performers, key=lambda x: x[1], reverse=True)[:3]
        analysis['performance_drivers']['worst_performers'] = sorted(worst_performers, key=lambda x: x[1])[:3]
        
        # Recommendations
        if performance_metrics.get('volatility', 0) > 0.25:
            analysis['recommendations'].append("Consider reducing portfolio volatility through better diversification")
        
        if performance_metrics.get('beta', 1) > 1.5:
            analysis['recommendations'].append("Portfolio is highly sensitive to market movements - consider defensive stocks")
        
        if len(best_performers) == 0:
            analysis['recommendations'].append("No strong performers identified - review stock selection criteria")
        
        return analysis
    
    def _validate_data_availability(self, portfolio_weights: Dict[str, float]) -> Dict[str, Any]:
        """Validate that authentic market data is available for all portfolio stocks"""
        
        unavailable_stocks = []
        
        for symbol in portfolio_weights.keys():
            try:
                # Quick test to see if data is available
                test_data = yf.download(symbol, period="5d", progress=False)
                if test_data is None or test_data.empty:
                    unavailable_stocks.append(symbol)
            except Exception:
                unavailable_stocks.append(symbol)
        
        if unavailable_stocks:
            return {
                'error': f"Real-time data unavailable for {len(unavailable_stocks)} stocks: {', '.join(unavailable_stocks[:5])}{'...' if len(unavailable_stocks) > 5 else ''}",
                'message': 'Cannot proceed without authentic market data for all portfolio stocks.',
                'data_source': 'Failed - Authentication or API limits exceeded'
            }
        
        return {'success': True}