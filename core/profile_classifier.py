"""
ADAPT Smart Indexing Engine - Profile Classifier

This module handles user profiling and risk assessment based on
behavioral finance principles and traditional risk factors.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any
from dataclasses import dataclass

@dataclass
class UserProfile:
    """Data class representing a user's investment profile."""
    profile_type: str
    risk_tolerance_score: float
    volatility_target: float
    investment_horizon: int
    factor_allocations: Dict[str, float]
    behavioral_adjustments: Dict[str, float]
    fragility_signals: Dict[str, float]

class ProfileClassifier:
    """Classifies users into investment profiles based on behavioral finance principles."""
    
    def __init__(self):
        """Initialize the profile classifier."""
        self.factor_base_allocations = {
            'Conservative': {
                'esg_ecology': 30,
                'low_volatility': 33,
                'quality': 12,
                'value': 10,
                'momentum': 8,
                'size': 4,
                'behavioral_adj': 3
            },
            'Moderate': {
                'esg_ecology': 22,
                'low_volatility': 22,
                'quality': 18,
                'value': 18,
                'momentum': 20,
                'size': 8,
                'behavioral_adj': 6
            },
            'Aggressive': {
                'esg_ecology': 10,
                'low_volatility': 10,
                'quality': 15,
                'value': 20,
                'momentum': 20,
                'size': 12,
                'behavioral_adj': 13
            }
        }
        
        self.fragility_base_signals = {
            'Conservative': {
                'behavioral_fragility': 27.5,
                'innovation_decay': 20,
                'labor_market_contraction': 20,
                'valuation_excess': 15,
                'debt_fragility': 10,
                'supply_chain_stress': 10,
                'macro_vulnerability': 10
            },
            'Moderate': {
                'behavioral_fragility': 30,
                'innovation_decay': 20,
                'labor_market_contraction': 17,
                'valuation_excess': 12,
                'debt_fragility': 10,
                'supply_chain_stress': 10,
                'macro_vulnerability': 10
            },
            'Aggressive': {
                'behavioral_fragility': 32.5,
                'innovation_decay': 20,
                'labor_market_contraction': 15,
                'valuation_excess': 10,
                'debt_fragility': 10,
                'supply_chain_stress': 10,
                'macro_vulnerability': 10
            }
        }
    
    def calculate_age_risk_score(self, age: int) -> float:
        """
        Calculate risk score based on age.
        
        Args:
            age: User's age
            
        Returns:
            Risk score (0-1, higher = more risk tolerant)
        """
        if age <= 25:
            return 0.9
        elif age <= 35:
            return 0.8
        elif age <= 45:
            return 0.6
        elif age <= 55:
            return 0.4
        elif age <= 65:
            return 0.2
        else:
            return 0.1
    
    def calculate_income_risk_score(self, income: float) -> float:
        """
        Calculate risk score based on income.
        
        Args:
            income: Annual income in rupees
            
        Returns:
            Risk score (0-1, higher = more risk tolerant)
        """
        if income >= 5000000:  # 50L+
            return 1.0
        elif income >= 2000000:  # 20L+
            return 0.8
        elif income >= 1000000:  # 10L+
            return 0.6
        elif income >= 500000:  # 5L+
            return 0.4
        else:
            return 0.2
    
    def calculate_goal_risk_score(self, investment_goal: str) -> float:
        """
        Calculate risk score based on investment goal.
        
        Args:
            investment_goal: Investment objective
            
        Returns:
            Risk score (0-1, higher = more risk tolerant)
        """
        goal_scores = {
            'wealth_preservation': 0.2,
            'income_generation': 0.4,
            'moderate_growth': 0.6,
            'capital_appreciation': 0.8,
            'aggressive_growth': 1.0
        }
        
        return goal_scores.get(investment_goal.lower().replace(' ', '_'), 0.5)
    
    def calculate_horizon_risk_score(self, investment_horizon: int) -> float:
        """
        Calculate risk score based on investment horizon.
        
        Args:
            investment_horizon: Investment horizon in years
            
        Returns:
            Risk score (0-1, higher = more risk tolerant)
        """
        if investment_horizon >= 20:
            return 1.0
        elif investment_horizon >= 15:
            return 0.8
        elif investment_horizon >= 10:
            return 0.6
        elif investment_horizon >= 5:
            return 0.4
        else:
            return 0.2
    
    def calculate_behavioral_score(self, behavioral_data: Dict[str, float]) -> float:
        """
        Calculate behavioral risk score from psychometric data.
        
        Args:
            behavioral_data: Dictionary with behavioral scores (1-10 scale)
            
        Returns:
            Behavioral risk score (0-1)
        """
        # Extract behavioral scores (default to 5 if not provided)
        loss_aversion = behavioral_data.get('loss_aversion', 5)
        overconfidence = behavioral_data.get('overconfidence', 5)
        herding_tendency = behavioral_data.get('herding_tendency', 5)
        anchoring_bias = behavioral_data.get('anchoring_bias', 5)
        disposition_effect = behavioral_data.get('disposition_effect', 5)
        
        # Calculate weighted behavioral score
        # Higher loss aversion and herding = lower risk tolerance
        # Higher overconfidence = higher risk tolerance
        # Anchoring and disposition effects reduce effective risk management
        
        risk_reducing_factors = (loss_aversion + herding_tendency + anchoring_bias + disposition_effect) / 4
        risk_increasing_factors = overconfidence
        
        # Normalize to 0-1 scale where higher = more risk tolerant
        behavioral_score = (risk_increasing_factors + (10 - risk_reducing_factors)) / 20
        
        return max(0, min(1, behavioral_score))
    
    def classify_profile(self, user_data: Dict[str, Any]) -> UserProfile:
        """
        Classify user into a risk profile.
        
        Args:
            user_data: Dictionary containing user information
            
        Returns:
            UserProfile object with classification results
        """
        # Extract user data
        age = user_data.get('age', 35)
        income = user_data.get('income', 1000000)
        investment_goal = user_data.get('investment_goal', 'moderate_growth')
        investment_horizon = user_data.get('investment_horizon', 10)
        
        # Calculate component risk scores
        age_score = self.calculate_age_risk_score(age)
        income_score = self.calculate_income_risk_score(income)
        goal_score = self.calculate_goal_risk_score(investment_goal)
        horizon_score = self.calculate_horizon_risk_score(investment_horizon)
        behavioral_score = self.calculate_behavioral_score(user_data)
        
        # Calculate weighted overall risk tolerance score
        weights = {
            'age': 0.25,
            'income': 0.15,
            'goal': 0.25,
            'horizon': 0.20,
            'behavioral': 0.15
        }
        
        risk_tolerance_score = (
            weights['age'] * age_score +
            weights['income'] * income_score +
            weights['goal'] * goal_score +
            weights['horizon'] * horizon_score +
            weights['behavioral'] * behavioral_score
        )
        
        # Classify into profile types
        if risk_tolerance_score < 0.33:
            profile_type = 'Conservative'
            volatility_target = 8.0  # <10%
        elif risk_tolerance_score < 0.67:
            profile_type = 'Moderate'
            volatility_target = 12.5  # 10-15%
        else:
            profile_type = 'Aggressive'
            volatility_target = 20.0  # 15-25%
        
        # Get base factor allocations
        factor_allocations = self.factor_base_allocations[profile_type].copy()
        
        # Apply behavioral adjustments
        behavioral_adjustments = self._calculate_behavioral_adjustments(user_data, profile_type)
        
        # Get fragility signals
        fragility_signals = self.fragility_base_signals[profile_type].copy()
        
        return UserProfile(
            profile_type=profile_type,
            risk_tolerance_score=risk_tolerance_score,
            volatility_target=volatility_target,
            investment_horizon=investment_horizon,
            factor_allocations=factor_allocations,
            behavioral_adjustments=behavioral_adjustments,
            fragility_signals=fragility_signals
        )
    
    def _calculate_behavioral_adjustments(self, user_data: Dict[str, Any], 
                                        profile_type: str) -> Dict[str, float]:
        """
        Calculate behavioral adjustments to portfolio allocation.
        
        Args:
            user_data: User behavioral data
            profile_type: Base profile type
            
        Returns:
            Dictionary of behavioral adjustments
        """
        adjustments = {}
        
        # Loss aversion adjustment
        loss_aversion = user_data.get('loss_aversion', 5)
        if loss_aversion > 7:
            adjustments['increase_low_volatility'] = 5.0
            adjustments['decrease_momentum'] = 3.0
        elif loss_aversion < 3:
            adjustments['decrease_low_volatility'] = 3.0
            adjustments['increase_momentum'] = 5.0
        
        # Overconfidence adjustment
        overconfidence = user_data.get('overconfidence', 5)
        if overconfidence > 7:
            adjustments['increase_momentum'] = 5.0
            adjustments['decrease_quality'] = 2.0
        elif overconfidence < 3:
            adjustments['increase_quality'] = 3.0
            adjustments['decrease_momentum'] = 2.0
        
        # Herding tendency adjustment
        herding_tendency = user_data.get('herding_tendency', 5)
        if herding_tendency > 7:
            adjustments['increase_momentum'] = 3.0
            adjustments['decrease_value'] = 2.0
        elif herding_tendency < 3:
            adjustments['increase_value'] = 3.0
            adjustments['decrease_momentum'] = 2.0
        
        return adjustments
    
    def get_profile_summary(self, profile: UserProfile) -> Dict[str, Any]:
        """
        Get a comprehensive summary of the user profile.
        
        Args:
            profile: UserProfile object
            
        Returns:
            Dictionary with profile summary
        """
        return {
            'profile_type': profile.profile_type,
            'risk_tolerance_score': f"{profile.risk_tolerance_score:.2f}",
            'volatility_target': f"{profile.volatility_target:.1f}%",
            'investment_horizon': f"{profile.investment_horizon} years",
            'factor_allocations': {
                key.replace('_', ' ').title(): f"{value:.1f}%"
                for key, value in profile.factor_allocations.items()
            },
            'behavioral_adjustments': profile.behavioral_adjustments,
            'fragility_signals': {
                key.replace('_', ' ').title(): f"{value:.1f}%"
                for key, value in profile.fragility_signals.items()
            }
        }
    
    def suggest_portfolio_adjustments(self, profile: UserProfile, 
                                    market_conditions: Dict[str, float] = None) -> Dict[str, str]:
        """
        Suggest portfolio adjustments based on profile and market conditions.
        
        Args:
            profile: User profile
            market_conditions: Current market condition indicators
            
        Returns:
            Dictionary of suggested adjustments
        """
        suggestions = []
        
        # Profile-based suggestions
        if profile.profile_type == 'Conservative':
            suggestions.append("Focus on dividend-paying stocks and defensive sectors")
            suggestions.append("Maintain high allocation to quality and low-volatility factors")
        elif profile.profile_type == 'Moderate':
            suggestions.append("Balance growth and defensive characteristics")
            suggestions.append("Consider value opportunities in quality companies")
        else:  # Aggressive
            suggestions.append("Emphasize growth and momentum factors")
            suggestions.append("Consider small-cap and emerging opportunities")
        
        # Behavioral adjustments
        if 'increase_low_volatility' in profile.behavioral_adjustments:
            suggestions.append("Consider increasing allocation to low-volatility stocks due to loss aversion")
        
        if 'increase_momentum' in profile.behavioral_adjustments:
            suggestions.append("Your confidence level suggests momentum strategies may be suitable")
        
        # Market condition adjustments (if provided)
        if market_conditions:
            vix_level = market_conditions.get('vix', 20)
            if vix_level > 30:
                suggestions.append("High market volatility detected - consider defensive positioning")
            elif vix_level < 15:
                suggestions.append("Low volatility environment - consider growth positioning")
        
        return {f"suggestion_{i+1}": suggestion for i, suggestion in enumerate(suggestions)}
