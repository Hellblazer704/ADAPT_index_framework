"""
ADAPT Smart Indexing Engine - Streamlit App

This is the main Streamlit application that provides a web interface for
the ADAPT smart indexing engine. Users can input their profile information,
view portfolio recommendations, and analyze backtest results.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import sys
import os

# Add the core directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), 'core'))

from core.nifty_loader import NiftyLoader
from core.profile_classifier import ProfileClassifier, UserProfile
from core.portfolio_builder import PortfolioBuilder
from core.backtester import Backtester

# Configure page
st.set_page_config(
    page_title="ADAPT Smart Indexing Engine",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .profile-card {
        background-color: #e8f4fd;
        padding: 1.5rem;
        border-radius: 0.5rem;
        border: 2px solid #1f77b4;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_components():
    """Load and cache the core components."""
    return {
        'loader': NiftyLoader(),
        'classifier': ProfileClassifier(),
        'builder': PortfolioBuilder(),
        'backtester': Backtester()
    }

def main():
    """Main application function."""
    
    # Header
    st.markdown('<h1 class="main-header">ADAPT Smart Indexing Engine</h1>', unsafe_allow_html=True)
    st.markdown("### Personalized Smart Indexing Based on Behavioral Finance")
    
    # Load components
    components = load_components()
    
    # Sidebar for navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.selectbox(
        "Choose a page:",
        ["User Profile", "Portfolio Analysis", "Backtest Results", "About"]
    )
    
    if page == "User Profile":
        show_user_profile_page(components)
    elif page == "Portfolio Analysis":
        show_portfolio_analysis_page(components)
    elif page == "Backtest Results":
        show_backtest_results_page(components)
    elif page == "About":
        show_about_page()

def show_user_profile_page(components):
    """Show the user profile input page."""
    st.header("📊 User Profile & Risk Assessment")
    
    # Create two columns
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("Personal Information")
        
        # Basic information
        age = st.slider("Age", 18, 80, 35)
        income = st.selectbox(
            "Annual Income",
            ["< ₹5 Lakhs", "₹5-10 Lakhs", "₹10-20 Lakhs", "₹20-50 Lakhs", "> ₹50 Lakhs"],
            index=2
        )
        
        # Convert income to numeric
        income_map = {
            "< ₹5 Lakhs": 300000,
            "₹5-10 Lakhs": 750000,
            "₹10-20 Lakhs": 1500000,
            "₹20-50 Lakhs": 3500000,
            "> ₹50 Lakhs": 7500000
        }
        income_value = income_map[income]
        
        investment_goal = st.selectbox(
            "Investment Goal",
            ["Wealth Preservation", "Income Generation", "Capital Appreciation", "Aggressive Growth"],
            index=2
        )
        
        investment_horizon = st.slider("Investment Horizon (years)", 1, 30, 10)
        
        st.subheader("Behavioral Finance Assessment")
        st.markdown("Rate yourself on a scale of 1-10 for each behavioral trait:")
        
        loss_aversion = st.slider(
            "Loss Aversion (How much do you dislike losses compared to gains?)",
            1, 10, 5,
            help="1 = Not bothered by losses, 10 = Extremely loss averse"
        )
        
        overconfidence = st.slider(
            "Overconfidence (How confident are you in your investment decisions?)",
            1, 10, 5,
            help="1 = Very cautious, 10 = Extremely confident"
        )
        
        herding_tendency = st.slider(
            "Herding Tendency (Do you follow the crowd?)",
            1, 10, 5,
            help="1 = Independent thinker, 10 = Always follow trends"
        )
        
        anchoring_bias = st.slider(
            "Anchoring Bias (Do you fixate on initial prices/values?)",
            1, 10, 5,
            help="1 = Flexible thinking, 10 = Stuck on initial reference points"
        )
        
        disposition_effect = st.slider(
            "Disposition Effect (Do you hold losers too long and sell winners too early?)",
            1, 10, 5,
            help="1 = Rational selling, 10 = Emotional selling patterns"
        )
    
    with col2:
        st.subheader("Profile Preview")
        
        # Create user data dictionary
        user_data = {
            'age': age,
            'income': income_value,
            'investment_goal': investment_goal.lower().replace(' ', '_'),
            'investment_horizon': investment_horizon,
            'loss_aversion': loss_aversion,
            'overconfidence': overconfidence,
            'herding_tendency': herding_tendency,
            'anchoring_bias': anchoring_bias,
            'disposition_effect': disposition_effect
        }
        
        # Classify profile
        profile = components['classifier'].classify_profile(user_data)
        summary = components['classifier'].get_profile_summary(profile)
        
        # Display profile information
        st.markdown(f"""
        <div class="profile-card">
            <h4>Risk Profile: {summary['profile_type']}</h4>
            <p><strong>Volatility Target:</strong> {summary['volatility_target']}</p>
            <p><strong>Risk Tolerance:</strong> {summary['risk_tolerance_score']}</p>
            <p><strong>Investment Horizon:</strong> {summary['investment_horizon']}</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Factor allocations chart
        st.subheader("Factor Allocations")
        factor_data = pd.DataFrame(
            list(summary['factor_allocations'].items()),
            columns=['Factor', 'Allocation (%)']
        )
        
        fig = px.bar(
            factor_data,
            x='Factor',
            y='Allocation (%)',
            title="Portfolio Factor Allocations",
            color='Allocation (%)',
            color_continuous_scale='Blues'
        )
        fig.update_layout(height=300)
        st.plotly_chart(fig, use_container_width=True)
    
    # Store profile in session state
    if st.button("Generate Portfolio", type="primary"):
        st.session_state['user_profile'] = profile
        st.session_state['user_data'] = user_data
        st.success("Profile created successfully! Navigate to 'Portfolio Analysis' to view your personalized portfolio.")

def show_portfolio_analysis_page(components):
    """Show the portfolio analysis page."""
    st.header("📈 Portfolio Analysis")
    
    if 'user_profile' not in st.session_state:
        st.warning("Please complete your user profile first!")
        return
    
    profile = st.session_state['user_profile']
    
    # Portfolio configuration
    st.subheader("Portfolio Configuration")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        universe = st.selectbox("Stock Universe", ["NIFTY 50", "NIFTY 500"], index=0)
    with col2:
        max_stocks = st.slider("Maximum Stocks", 10, 50, 25)
    with col3:
        rebalance_freq = st.selectbox("Rebalance Frequency", ["Monthly", "Quarterly", "Yearly"], index=1)
    
    # Build portfolio
    if st.button("Build Portfolio", type="primary"):
        with st.spinner("Building your personalized portfolio..."):
            try:
                # Build portfolio
                portfolio = components['builder'].build_portfolio(
                    profile,
                    universe=universe.lower().replace(' ', '_'),
                    max_stocks=max_stocks
                )
                
                # Store in session state
                st.session_state['portfolio'] = portfolio
                st.session_state['rebalance_freq'] = rebalance_freq.lower()
                
                st.success("Portfolio built successfully!")
                
            except Exception as e:
                st.error(f"Error building portfolio: {str(e)}")
    
    # Display portfolio if available
    if 'portfolio' in st.session_state:
        portfolio = st.session_state['portfolio']
        summary = components['builder'].get_portfolio_summary(portfolio)
        
        # Portfolio overview
        st.subheader("Portfolio Overview")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Profile Type", summary['profile_type'])
        with col2:
            st.metric("Number of Stocks", summary['num_stocks'])
        with col3:
            st.metric("Avg PE Ratio", f"{summary['avg_pe_ratio']:.2f}")
        with col4:
            st.metric("Avg Dividend Yield", f"{summary['avg_dividend_yield']:.2f}%")
        
        # Top holdings
        st.subheader("Top Holdings")
        holdings_df = pd.DataFrame(
            list(summary['top_holdings'].items()),
            columns=['Stock', 'Weight (%)']
        )
        
        fig = px.bar(
            holdings_df.head(10),
            x='Stock',
            y='Weight (%)',
            title="Top 10 Holdings",
            color='Weight (%)',
            color_continuous_scale='Greens'
        )
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
        
        # Sector allocation
        st.subheader("Sector Allocation")
        sector_df = pd.DataFrame(
            list(summary['sector_allocation'].items()),
            columns=['Sector', 'Weight (%)']
        )
        
        fig = px.pie(
            sector_df,
            values='Weight (%)',
            names='Sector',
            title="Sector Allocation"
        )
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
        
        # Factor allocations
        st.subheader("Factor Allocations")
        factor_df = pd.DataFrame(
            list(summary['factor_allocations'].items()),
            columns=['Factor', 'Allocation (%)']
        )
        
        fig = px.bar(
            factor_df,
            x='Factor',
            y='Allocation (%)',
            title="Factor Allocations",
            color='Allocation (%)',
            color_continuous_scale='Blues'
        )
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)

def show_backtest_results_page(components):
    """Show the backtest results page."""
    st.header("📊 Backtest Results")
    
    if 'portfolio' not in st.session_state:
        st.warning("Please build a portfolio first!")
        return
    
    portfolio = st.session_state['portfolio']
    rebalance_freq = st.session_state.get('rebalance_freq', 'quarterly')
    
    # Backtest configuration
    st.subheader("Backtest Configuration")
    col1, col2 = st.columns(2)
    
    with col1:
        start_date = st.date_input("Start Date", value=pd.to_datetime('2010-01-01'))
    with col2:
        end_date = st.date_input("End Date", value=pd.to_datetime('2024-01-01'))
    
    # Run backtest
    if st.button("Run Backtest", type="primary"):
        with st.spinner("Running backtest..."):
            try:
                # Run backtest
                backtest_results = components['backtester'].backtest_portfolio(
                    portfolio['portfolio_weights'],
                    portfolio['fundamental_data'],  # This should be stock_data
                    portfolio['benchmark_data'],
                    start_date=str(start_date),
                    end_date=str(end_date),
                    rebalance_frequency=rebalance_freq
                )
                
                # Store results
                st.session_state['backtest_results'] = backtest_results
                
                st.success("Backtest completed successfully!")
                
            except Exception as e:
                st.error(f"Error running backtest: {str(e)}")
    
    # Display results if available
    if 'backtest_results' in st.session_state:
        results = st.session_state['backtest_results']
        metrics = results['performance_metrics']
        
        # Performance metrics
        st.subheader("Performance Metrics")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Return", f"{metrics.get('total_return', 0):.2f}%")
        with col2:
            st.metric("Annualized Return", f"{metrics.get('annualized_return', 0):.2f}%")
        with col3:
            st.metric("Volatility", f"{metrics.get('volatility', 0):.2f}%")
        with col4:
            st.metric("Sharpe Ratio", f"{metrics.get('sharpe_ratio', 0):.3f}")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Max Drawdown", f"{metrics.get('max_drawdown', 0):.2f}%")
        with col2:
            st.metric("Sortino Ratio", f"{metrics.get('sortino_ratio', 0):.3f}")
        with col3:
            st.metric("Beta", f"{metrics.get('beta', 0):.3f}")
        with col4:
            st.metric("Alpha", f"{metrics.get('alpha', 0):.2f}%")
        
        # Performance comparison chart
        if 'portfolio_returns' in results and 'benchmark_returns' in results:
            st.subheader("Performance Comparison")
            
            # Calculate cumulative returns
            portfolio_cumulative = (1 + results['portfolio_returns']).cumprod()
            benchmark_cumulative = (1 + results['benchmark_returns']).cumprod()
            
            # Create comparison chart
            fig = go.Figure()
            
            fig.add_trace(go.Scatter(
                x=portfolio_cumulative.index,
                y=portfolio_cumulative.values * 100,
                mode='lines',
                name='ADAPT Portfolio',
                line=dict(color='blue', width=2)
            ))
            
            fig.add_trace(go.Scatter(
                x=benchmark_cumulative.index,
                y=benchmark_cumulative.values * 100,
                mode='lines',
                name='NIFTY 50 Benchmark',
                line=dict(color='red', width=2)
            ))
            
            fig.update_layout(
                title="Portfolio vs Benchmark Performance",
                xaxis_title="Date",
                yaxis_title="Cumulative Return (%)",
                height=500
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        # Drawdown analysis
        if 'drawdown_analysis' in results:
            st.subheader("Drawdown Analysis")
            
            drawdown_data = results['drawdown_analysis']
            
            if 'drawdown_series' in drawdown_data:
                fig = go.Figure()
                
                fig.add_trace(go.Scatter(
                    x=drawdown_data['drawdown_series'].index,
                    y=drawdown_data['drawdown_series'].values,
                    mode='lines',
                    name='Drawdown',
                    fill='tonexty',
                    line=dict(color='red', width=1)
                ))
                
                fig.update_layout(
                    title="Portfolio Drawdown",
                    xaxis_title="Date",
                    yaxis_title="Drawdown (%)",
                    height=400
                )
                
                st.plotly_chart(fig, use_container_width=True)
        
        # Download results
        st.subheader("Download Results")
        
        if st.button("Download Performance Report"):
            report = components['backtester'].generate_performance_report(results)
            
            # Create download button
            st.download_button(
                label="Download Report as TXT",
                data=report,
                file_name="adapt_performance_report.txt",
                mime="text/plain"
            )

def show_about_page():
    """Show the about page."""
    st.header("ℹ️ About ADAPT Smart Indexing Engine")
    
    st.markdown("""
    ## What is ADAPT?
    
    ADAPT (Adaptive Dynamic Allocation Portfolio Technology) is a personalized smart indexing engine 
    designed for retail investors based on behavioral finance principles. It dynamically builds and 
    rebalances index portfolios based on:
    
    - **Psychometric Data**: Your behavioral traits and risk preferences
    - **Macroeconomic Indicators**: Current market conditions and economic trends
    - **Sector Health Signals**: Industry-specific performance indicators
    
    ## Key Features
    
    ### 1. Behavioral Finance Integration
    - Loss aversion assessment
    - Overconfidence evaluation
    - Herding tendency analysis
    - Anchoring bias detection
    - Disposition effect measurement
    
    ### 2. Smart Factor Allocations
    - **ESG/Ecology**: Environmental, Social, and Governance factors
    - **Low Volatility**: Defensive positioning during market stress
    - **Quality**: Companies with strong fundamentals
    - **Value**: Undervalued stocks with growth potential
    - **Momentum**: Trending stocks with positive price action
    - **Size**: Market capitalization considerations
    - **Behavioral Adjustments**: Custom tilts based on your profile
    
    ### 3. Risk Profiles
    
    | Profile | Volatility Target | Characteristics |
    |---------|------------------|-----------------|
    | Conservative | <10% | Capital preservation, low risk |
    | Moderate | 10-15% | Balanced growth and stability |
    | Aggressive | 15-25% | High growth, higher risk tolerance |
    
    ## Technology Stack
    
    - **Backend**: Python, FastAPI, PostgreSQL
    - **Data**: yFinance, NSE data feeds
    - **Analytics**: Pandas, NumPy, scikit-learn
    - **Frontend**: Streamlit, Plotly
    - **Deployment**: Docker, GitHub Actions
    
    ## Methodology
    
    ADAPT uses a multi-factor approach combining:
    
    1. **User Profiling**: Behavioral assessment and risk tolerance
    2. **Stock Screening**: Fundamental and technical filters
    3. **Factor Weighting**: Dynamic allocation based on market conditions
    4. **Portfolio Construction**: Optimal stock selection and weighting
    5. **Performance Analysis**: Comprehensive backtesting and metrics
    
    ## Disclaimer
    
    This tool is for educational and research purposes. Past performance does not guarantee future results. 
    Always consult with a financial advisor before making investment decisions.
    """)

if __name__ == "__main__":
    main() 