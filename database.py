"""
Database Schema and Operations for ADAPT Smart Indexing Engine
"""

import os
import pandas as pd
from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime, Text, Boolean
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from datetime import datetime
import json

Base = declarative_base()

class UserProfile(Base):
    __tablename__ = 'user_profiles'
    
    id = Column(Integer, primary_key=True)
    user_id = Column(String(50), unique=True, nullable=False)
    age = Column(Integer)
    income = Column(Float)
    investment_goal = Column(String(100))
    investment_horizon = Column(Integer)
    loss_aversion = Column(Integer)
    overconfidence = Column(Integer)
    herding_tendency = Column(Integer)
    anchoring_bias = Column(Integer)
    disposition_effect = Column(Integer)
    profile_type = Column(String(50))
    risk_tolerance_score = Column(Float)
    volatility_target = Column(Float)
    created_at = Column(DateTime, default=datetime.utcnow)

class Portfolio(Base):
    __tablename__ = 'portfolios'
    
    id = Column(Integer, primary_key=True)
    user_id = Column(String(50), nullable=False)
    portfolio_name = Column(String(100))
    weights = Column(Text)  # JSON string of portfolio weights
    universe = Column(String(50))
    max_stocks = Column(Integer)
    profile_type = Column(String(50))
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow)

class BacktestResult(Base):
    __tablename__ = 'backtest_results'
    
    id = Column(Integer, primary_key=True)
    portfolio_id = Column(Integer)
    user_id = Column(String(50), nullable=False)
    start_date = Column(DateTime)
    end_date = Column(DateTime)
    initial_capital = Column(Float)
    final_value = Column(Float)
    total_return = Column(Float)
    annualized_return = Column(Float)
    sharpe_ratio = Column(Float)
    max_drawdown = Column(Float)
    volatility = Column(Float)
    alpha = Column(Float)
    beta = Column(Float)
    performance_data = Column(Text)  # JSON string of detailed results
    created_at = Column(DateTime, default=datetime.utcnow)

class StockData(Base):
    __tablename__ = 'stock_data'
    
    id = Column(Integer, primary_key=True)
    symbol = Column(String(20), nullable=False)
    date = Column(DateTime, nullable=False)
    open_price = Column(Float)
    high_price = Column(Float)
    low_price = Column(Float)
    close_price = Column(Float)
    volume = Column(Float)
    adj_close = Column(Float)
    last_updated = Column(DateTime, default=datetime.utcnow)

class DatabaseManager:
    def __init__(self):
        self.database_url = os.getenv('DATABASE_URL')
        if not self.database_url:
            raise ValueError("DATABASE_URL environment variable is not set")
        self.engine = create_engine(self.database_url)
        Base.metadata.create_all(self.engine)
        Session = sessionmaker(bind=self.engine)
        self.session = Session()
    
    def save_user_profile(self, user_data, profile_result):
        """Save user profile to database"""
        try:
            user_profile = UserProfile(
                user_id=user_data.get('user_id', f"user_{datetime.now().timestamp()}"),
                age=user_data.get('age'),
                income=user_data.get('income'),
                investment_goal=user_data.get('investment_goal'),
                investment_horizon=user_data.get('investment_horizon'),
                loss_aversion=user_data.get('loss_aversion'),
                overconfidence=user_data.get('overconfidence'),
                herding_tendency=user_data.get('herding_tendency'),
                anchoring_bias=user_data.get('anchoring_bias'),
                disposition_effect=user_data.get('disposition_effect'),
                profile_type=profile_result.profile_type,
                risk_tolerance_score=profile_result.risk_tolerance_score,
                volatility_target=profile_result.volatility_target
            )
            
            # Check if user exists, update if so
            existing = self.session.query(UserProfile).filter_by(user_id=user_profile.user_id).first()
            if existing:
                for key, value in user_profile.__dict__.items():
                    if key != '_sa_instance_state' and value is not None:
                        setattr(existing, key, value)
                self.session.commit()
                return existing.id
            else:
                self.session.add(user_profile)
                self.session.commit()
                return user_profile.id
                
        except Exception as e:
            self.session.rollback()
            print(f"Error saving user profile: {e}")
            return None
    
    def save_portfolio(self, user_id, portfolio_data):
        """Save portfolio to database"""
        try:
            portfolio = Portfolio(
                user_id=user_id,
                portfolio_name=f"{portfolio_data.get('profile_type', 'Unknown')} Portfolio",
                weights=json.dumps(portfolio_data.get('portfolio_weights', {})),
                universe=portfolio_data.get('universe', 'nifty_50'),
                max_stocks=portfolio_data.get('max_stocks', 25),
                profile_type=portfolio_data.get('profile_type', 'Unknown')
            )
            
            self.session.add(portfolio)
            self.session.commit()
            return portfolio.id
            
        except Exception as e:
            self.session.rollback()
            print(f"Error saving portfolio: {e}")
            return None
    
    def save_backtest_result(self, user_id, portfolio_id, backtest_data):
        """Save backtest results to database"""
        try:
            config = backtest_data.get('backtest_config', {})
            metrics = backtest_data.get('performance_metrics', {})
            
            result = BacktestResult(
                portfolio_id=portfolio_id,
                user_id=user_id,
                start_date=pd.to_datetime(config.get('start_date')),
                end_date=pd.to_datetime(config.get('end_date')),
                initial_capital=config.get('initial_capital', 1000000),
                final_value=metrics.get('final_value', 0),
                total_return=metrics.get('total_return', 0),
                annualized_return=metrics.get('annualized_return', 0),
                sharpe_ratio=metrics.get('sharpe_ratio', 0),
                max_drawdown=metrics.get('max_drawdown', 0),
                volatility=metrics.get('volatility', 0),
                alpha=metrics.get('alpha', 0),
                beta=metrics.get('beta', 1),
                performance_data=json.dumps({
                    'drawdown_analysis': backtest_data.get('drawdown_analysis', {}),
                    'rolling_metrics': str(backtest_data.get('rolling_metrics', {}))
                })
            )
            
            self.session.add(result)
            self.session.commit()
            return result.id
            
        except Exception as e:
            self.session.rollback()
            print(f"Error saving backtest result: {e}")
            return None
    
    def save_stock_data(self, symbol, data_df):
        """Save stock data to database"""
        try:
            # Clear existing data for this symbol
            self.session.query(StockData).filter_by(symbol=symbol).delete()
            
            # Insert new data
            for date, row in data_df.iterrows():
                stock_data = StockData(
                    symbol=symbol,
                    date=date,
                    open_price=row.get('Open', 0),
                    high_price=row.get('High', 0),
                    low_price=row.get('Low', 0),
                    close_price=row.get('Close', 0),
                    volume=row.get('Volume', 0),
                    adj_close=row.get('Adj Close', row.get('Close', 0))
                )
                self.session.add(stock_data)
            
            self.session.commit()
            
        except Exception as e:
            self.session.rollback()
            print(f"Error saving stock data for {symbol}: {e}")
    
    def get_user_portfolios(self, user_id):
        """Get all portfolios for a user"""
        try:
            portfolios = self.session.query(Portfolio).filter_by(user_id=user_id).all()
            return portfolios
        except Exception as e:
            print(f"Error getting user portfolios: {e}")
            return []
    
    def get_backtest_history(self, user_id):
        """Get backtest history for a user"""
        try:
            results = self.session.query(BacktestResult).filter_by(user_id=user_id).order_by(BacktestResult.created_at.desc()).all()
            return results
        except Exception as e:
            print(f"Error getting backtest history: {e}")
            return []
    
    def get_stock_data(self, symbol, start_date=None, end_date=None):
        """Get stock data from database"""
        try:
            query = self.session.query(StockData).filter_by(symbol=symbol)
            
            if start_date:
                query = query.filter(StockData.date >= start_date)
            if end_date:
                query = query.filter(StockData.date <= end_date)
            
            data = query.order_by(StockData.date).all()
            
            if data:
                df = pd.DataFrame([{
                    'Date': d.date,
                    'Open': d.open_price,
                    'High': d.high_price,
                    'Low': d.low_price,
                    'Close': d.close_price,
                    'Volume': d.volume,
                    'Adj Close': d.adj_close
                } for d in data])
                df.set_index('Date', inplace=True)
                return df
            
            return pd.DataFrame()
            
        except Exception as e:
            print(f"Error getting stock data for {symbol}: {e}")
            return pd.DataFrame()
    
    def close(self):
        """Close database session"""
        self.session.close()