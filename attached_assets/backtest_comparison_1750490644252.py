import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import yfinance as yf
from core.profile_classifier import ProfileClassifier
from core.portfolio_builder import PortfolioBuilder
from core.backtester import Backtester
import warnings
warnings.filterwarnings('ignore')

class ADAPTIndexComparison:
    def __init__(self):
        self.profile_classifier = ProfileClassifier()
        self.portfolio_builder = PortfolioBuilder()
        self.backtester = Backtester()
        
        # Define date range
        self.start_date = '2024-06-20'
        self.end_date = '2025-06-20'
        
        # NIFTY 50 stocks for synthetic data generation
        self.nifty_stocks = [
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
        
    def generate_synthetic_data(self):
        """Generate synthetic market data for the specified period"""
        print("Generating synthetic market data...")
        
        # Create date range
        date_range = pd.date_range(start=self.start_date, end=self.end_date, freq='D')
        
        # Generate synthetic data for each stock
        synthetic_data = {}
        
        for stock in self.nifty_stocks:
            # Generate realistic price movements
            np.random.seed(hash(stock) % 1000)  # Consistent seed for reproducibility
            
            # Initial price between 100-5000
            initial_price = np.random.uniform(100, 5000)
            
            # Generate daily returns with realistic volatility
            daily_returns = np.random.normal(0.0008, 0.02, len(date_range))  # 0.08% daily return, 2% volatility
            
            # Add some market trends and cycles
            trend = np.linspace(0, 0.15, len(date_range))  # 15% annual trend
            cycle = 0.05 * np.sin(2 * np.pi * np.arange(len(date_range)) / 252)  # Annual cycle
            
            # Combine all factors
            cumulative_returns = np.cumprod(1 + daily_returns + trend/252 + cycle/252)
            prices = initial_price * cumulative_returns
            
            # Create OHLC data
            high_multiplier = 1 + np.random.uniform(0.01, 0.03, len(date_range))
            low_multiplier = 1 - np.random.uniform(0.01, 0.03, len(date_range))
            
            synthetic_data[stock] = pd.DataFrame({
                'Date': date_range,
                'Open': prices * np.random.uniform(0.995, 1.005, len(date_range)),
                'High': prices * high_multiplier,
                'Low': prices * low_multiplier,
                'Close': prices,
                'Volume': np.random.uniform(1000000, 10000000, len(date_range))
            }).set_index('Date')
        
        return synthetic_data
    
    def create_profile_portfolios(self):
        """Create portfolios for each profile type"""
        print("Creating ADAPT portfolios for each profile...")
        
        # Define profile parameters
        profiles = {
            'Conservative': {
                'risk_tolerance': 2,
                'investment_horizon': 3,
                'financial_goals': 2,
                'market_knowledge': 2,
                'emotional_stability': 4,
                'age': 55,
                'income': 800000,
                'existing_investments': 2000000
            },
            'Moderate': {
                'risk_tolerance': 3,
                'investment_horizon': 5,
                'financial_goals': 3,
                'market_knowledge': 3,
                'emotional_stability': 3,
                'age': 40,
                'income': 1200000,
                'existing_investments': 1500000
            },
            'Aggressive': {
                'risk_tolerance': 4,
                'investment_horizon': 8,
                'financial_goals': 4,
                'market_knowledge': 4,
                'emotional_stability': 2,
                'age': 30,
                'income': 1500000,
                'existing_investments': 800000
            }
        }
        
        portfolio_results = {}
        
        for profile_name, profile_data in profiles.items():
            print(f"\nCreating {profile_name} portfolio...")
            
            # Classify profile
            profile = self.profile_classifier.classify_profile(profile_data)
            print(f"Profile classification: {profile.profile_type}")
            print(f"Risk score: {profile.risk_tolerance_score:.2f}")
            print(f"Factor allocations: {profile.factor_allocations}")
            
            # Build portfolio
            portfolio = self.portfolio_builder.build_portfolio(
                user_profile=profile,
                universe='nifty_50',
                max_stocks=30
            )
            
            portfolio_results[profile_name] = {
                'profile': profile,
                'portfolio': portfolio,
                'weights': portfolio['portfolio_weights']
            }
            
            print(f"Portfolio created with {len(portfolio['portfolio_weights'])} stocks")
        
        return portfolio_results
    
    def run_backtest_comparison(self):
        """Run backtest comparison for all profiles"""
        print("Running ADAPT Index comparison backtest...")
        
        # Create portfolios
        portfolios = self.create_profile_portfolios()
        
        # Run backtests
        backtest_results = {}
        
        # Generate market data
        market_data = self.generate_synthetic_data()
        
        for profile_name, portfolio_info in portfolios.items():
            print(f"\nRunning backtest for {profile_name} profile...")
            
            # Run backtest
            backtest = self.backtester.backtest_portfolio(
                portfolio_weights=portfolio_info['weights'],
                stock_data=market_data,
                benchmark_data=portfolio_info['portfolio']['benchmark_data'],
                start_date=self.start_date,
                end_date=self.end_date,
                rebalance_frequency='quarterly'
            )
            
            backtest_results[profile_name] = backtest
            
            print(f"Backtest completed for {profile_name} profile.")
        
        return backtest_results
    
    def create_comparison_analysis(self, backtest_results):
        """Create comprehensive comparison analysis"""
        print("\nCreating comparison analysis...")
        
        # Create comparison DataFrame
        comparison_data = []
        
        for profile_name, results in backtest_results.items():
            metrics = results.get('performance_metrics', {})
            comparison_data.append({
                'Profile': profile_name,
                'Final Value (₹)': metrics.get('final_value', None),
                'Total Return (%)': metrics.get('total_return', None),
                'Annualized Return (%)': metrics.get('annualized_return', None),
                'Sharpe Ratio': metrics.get('sharpe_ratio', None),
                'Max Drawdown (%)': metrics.get('max_drawdown', None),
                'Volatility (%)': metrics.get('volatility', None),
                'Beta': metrics.get('beta', None)
            })
        
        comparison_df = pd.DataFrame(comparison_data)
        
        # Create performance comparison chart
        self.create_performance_charts(backtest_results, comparison_df)
        
        return comparison_df
    
    def create_performance_charts(self, backtest_results, comparison_df):
        """Create performance visualization charts"""
        print("Creating performance charts...")
        
        # Set up the plotting style
        plt.style.use('seaborn-v0_8')
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('ADAPT Index Performance Comparison (June 2024 - June 2025)', fontsize=16, fontweight='bold')
        
        # 1. Cumulative Returns Comparison
        ax1 = axes[0, 0]
        for profile_name, results in backtest_results.items():
            portfolio_returns = results.get('portfolio_returns', pd.Series(dtype=float))
            if not portfolio_returns.empty:
                cumulative_returns = (1 + portfolio_returns).cumprod()
                ax1.plot(cumulative_returns.index, cumulative_returns.values, 
                        label=profile_name, linewidth=2)
        
        ax1.set_title('Cumulative Returns Comparison')
        ax1.set_xlabel('Date')
        ax1.set_ylabel('Cumulative Return')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. Risk-Return Scatter Plot
        ax2 = axes[0, 1]
        for profile_name, results in backtest_results.items():
            ax2.scatter(results['volatility'] * 100, results['annualized_return'] * 100, 
                       s=100, label=profile_name, alpha=0.7)
        
        ax2.set_title('Risk-Return Profile')
        ax2.set_xlabel('Volatility (%)')
        ax2.set_ylabel('Annualized Return (%)')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 3. Performance Metrics Bar Chart
        ax3 = axes[1, 0]
        x = np.arange(len(comparison_df))
        width = 0.25
        
        ax3.bar(x - width, comparison_df['Total Return (%)'], width, label='Total Return', alpha=0.8)
        ax3.bar(x, comparison_df['Sharpe Ratio'], width, label='Sharpe Ratio', alpha=0.8)
        ax3.bar(x + width, comparison_df['Max Drawdown (%)'], width, label='Max Drawdown', alpha=0.8)
        
        ax3.set_title('Performance Metrics Comparison')
        ax3.set_xlabel('Profile Type')
        ax3.set_ylabel('Value')
        ax3.set_xticks(x)
        ax3.set_xticklabels(comparison_df['Profile'])
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # 4. Portfolio Value Evolution
        ax4 = axes[1, 1]
        for profile_name, results in backtest_results.items():
            portfolio_returns = results.get('portfolio_returns', pd.Series(dtype=float))
            if not portfolio_returns.empty:
                portfolio_values = 1000000 * (1 + portfolio_returns).cumprod()
                ax4.plot(portfolio_values.index, portfolio_values.values, 
                        label=profile_name, linewidth=2)
        
        ax4.set_title('Portfolio Value Evolution')
        ax4.set_xlabel('Date')
        ax4.set_ylabel('Portfolio Value (₹)')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('adapt_index_comparison.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # Save detailed results
        comparison_df.to_csv('adapt_index_comparison_results.csv', index=False)
        print("Results saved to 'adapt_index_comparison_results.csv'")
        print("Charts saved to 'adapt_index_comparison.png'")
    
    def print_summary_report(self, comparison_df):
        """Print comprehensive summary report"""
        print("\n" + "="*80)
        print("ADAPT INDEX COMPARISON SUMMARY REPORT")
        print("="*80)
        print(f"Analysis Period: {self.start_date} to {self.end_date}")
        print(f"Initial Investment: ₹10,00,000 per profile")
        print("="*80)
        
        # Overall performance summary
        print("\nPERFORMANCE SUMMARY:")
        print("-" * 50)
        for _, row in comparison_df.iterrows():
            print(f"\n{row['Profile']} Profile:")
            print(f"  Final Portfolio Value: ₹{row['Final Value (₹)']:,.2f}")
            print(f"  Total Return: {row['Total Return (%)']:.2f}%")
            print(f"  Annualized Return: {row['Annualized Return (%)']:.2f}%")
            print(f"  Sharpe Ratio: {row['Sharpe Ratio']:.2f}")
            print(f"  Maximum Drawdown: {row['Max Drawdown (%)']:.2f}%")
            print(f"  Volatility: {row['Volatility (%)']:.2f}%")
            print(f"  Beta: {row['Beta']:.2f}")
        
        # Rankings
        print("\n" + "="*80)
        print("RANKINGS:")
        print("="*80)
        
        # Best performers
        best_return = comparison_df.loc[comparison_df['Total Return (%)'].idxmax()]
        best_sharpe = comparison_df.loc[comparison_df['Sharpe Ratio'].idxmax()]
        lowest_drawdown = comparison_df.loc[comparison_df['Max Drawdown (%)'].idxmin()]
        
        print(f"\n🏆 Best Total Return: {best_return['Profile']} ({best_return['Total Return (%)']:.2f}%)")
        print(f"📈 Best Risk-Adjusted Return (Sharpe): {best_sharpe['Profile']} ({best_sharpe['Sharpe Ratio']:.2f})")
        print(f"🛡️  Lowest Maximum Drawdown: {lowest_drawdown['Profile']} ({lowest_drawdown['Max Drawdown (%)']:.2f}%)")
        
        # Key insights
        print("\n" + "="*80)
        print("KEY INSIGHTS:")
        print("="*80)
        
        conservative = comparison_df[comparison_df['Profile'] == 'Conservative'].iloc[0]
        moderate = comparison_df[comparison_df['Profile'] == 'Moderate'].iloc[0]
        aggressive = comparison_df[comparison_df['Profile'] == 'Aggressive'].iloc[0]
        
        print(f"\n• Conservative Profile: Focuses on capital preservation with lower volatility ({conservative['Volatility (%)']:.2f}%)")
        print(f"• Moderate Profile: Balanced approach with moderate risk-return trade-off")
        print(f"• Aggressive Profile: Higher potential returns but with increased volatility ({aggressive['Volatility (%)']:.2f}%)")
        
        print(f"\n• Risk-Return Trade-off: Higher returns generally come with higher volatility")
        print(f"• Sharpe Ratio Analysis: {best_sharpe['Profile']} profile provides the best risk-adjusted returns")
        print(f"• Drawdown Analysis: {lowest_drawdown['Profile']} profile shows the most resilience during market downturns")
        
        print("\n" + "="*80)
        print("RECOMMENDATIONS:")
        print("="*80)
        print("• Conservative: Suitable for retirees and risk-averse investors")
        print("• Moderate: Ideal for balanced growth with moderate risk tolerance")
        print("• Aggressive: Best for young investors with high risk tolerance and long investment horizon")
        print("="*80)

def main():
    """Main function to run the ADAPT index comparison"""
    print("🚀 Starting ADAPT Index Comparison Analysis")
    print("="*60)
    
    # Initialize the comparison engine
    comparison_engine = ADAPTIndexComparison()
    
    try:
        # Run the complete analysis
        backtest_results = comparison_engine.run_backtest_comparison()
        
        # Create comparison analysis
        comparison_df = comparison_engine.create_comparison_analysis(backtest_results)
        
        # Print summary report
        comparison_engine.print_summary_report(comparison_df)
        
        print("\n✅ ADAPT Index Comparison completed successfully!")
        print("📊 Check the generated files for detailed results and visualizations.")
        
    except Exception as e:
        print(f"❌ Error during analysis: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 