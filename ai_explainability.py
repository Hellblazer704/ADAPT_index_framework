"""
AI Explainability Module for ADAPT Smart Indexing Engine

This module provides interpretable machine learning features to explain
portfolio recommendations, risk assessments, and investment decisions
using custom SHAP-like functionality and feature importance analysis.
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import mean_squared_error, accuracy_score
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from typing import Dict, List, Tuple, Any
import warnings

warnings.filterwarnings('ignore')

class ADAPTExplainer:
    """AI Explainability engine for ADAPT portfolio recommendations."""
    
    def __init__(self):
        self.portfolio_model = RandomForestRegressor(n_estimators=100, random_state=42)
        self.risk_model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.feature_names = []
        self.is_trained = False
        
    def prepare_training_data(self, user_profiles: List[Dict], portfolio_results: List[Dict]) -> Tuple[pd.DataFrame, np.ndarray]:
        """Prepare training data from user profiles and portfolio performance."""
        
        # Create feature matrix from user profiles
        features_data = []
        targets = []
        
        for i, (profile, result) in enumerate(zip(user_profiles, portfolio_results)):
            feature_row = {
                'age': profile.get('age', 35),
                'income': profile.get('income', 1500000),
                'investment_horizon': profile.get('investment_horizon', 10),
                'loss_aversion': profile.get('loss_aversion', 5),
                'overconfidence': profile.get('overconfidence', 5),
                'herding_tendency': profile.get('herding_tendency', 5),
                'anchoring_bias': profile.get('anchoring_bias', 5),
                'disposition_effect': profile.get('disposition_effect', 5),
                'risk_tolerance_score': result.get('risk_tolerance_score', 0.5),
                'volatility_target': result.get('volatility_target', 15.0),
                'portfolio_size': len(result.get('portfolio_weights', {})),
                'max_weight': max(result.get('portfolio_weights', {}).values()) if result.get('portfolio_weights') else 0.05,
                'diversification_ratio': self._calculate_diversification_ratio(result.get('portfolio_weights', {})),
                'sector_concentration': self._calculate_sector_concentration(result.get('portfolio_weights', {}))
            }
            
            # Add investment goal encoding
            goal_mapping = {'wealth_preservation': 1, 'wealth_accumulation': 2, 'aggressive_growth': 3}
            feature_row['investment_goal_encoded'] = goal_mapping.get(profile.get('investment_goal', 'wealth_accumulation'), 2)
            
            features_data.append(feature_row)
            
            # Target is portfolio performance (annualized return)
            performance = result.get('backtest_metrics', {})
            target_return = performance.get('annualized_return', 0.12)  # Default 12%
            targets.append(target_return)
        
        features_df = pd.DataFrame(features_data)
        self.feature_names = list(features_df.columns)
        
        return features_df, np.array(targets)
    
    def _calculate_diversification_ratio(self, weights: Dict[str, float]) -> float:
        """Calculate portfolio diversification ratio."""
        if not weights:
            return 0.0
        
        values = list(weights.values())
        if len(values) <= 1:
            return 0.0
        
        # Herfindahl-Hirschman Index for concentration
        hhi = sum(w**2 for w in values)
        # Diversification ratio (inverse of concentration)
        return 1 - hhi
    
    def _calculate_sector_concentration(self, weights: Dict[str, float]) -> float:
        """Calculate sector concentration (simplified)."""
        if not weights:
            return 0.0
        
        # Simplified sector mapping based on stock symbols
        sector_weights = {}
        for symbol, weight in weights.items():
            # Basic sector classification based on common patterns
            sector = self._classify_sector(symbol)
            sector_weights[sector] = sector_weights.get(sector, 0) + weight
        
        if len(sector_weights) <= 1:
            return 1.0  # High concentration
        
        # Calculate concentration using Gini coefficient approach
        sorted_weights = sorted(sector_weights.values(), reverse=True)
        total_weight = sum(sorted_weights)
        if total_weight == 0:
            return 0.0
        
        # Top 3 sectors concentration
        top3_concentration = sum(sorted_weights[:3]) / total_weight
        return top3_concentration
    
    def _classify_sector(self, symbol: str) -> str:
        """Basic sector classification based on symbol patterns."""
        # Simplified sector mapping for demonstration
        tech_symbols = ['INFY', 'TCS', 'WIPRO', 'TECHM', 'HCLTECH']
        banking_symbols = ['HDFCBANK', 'ICICIBANK', 'KOTAKBANK', 'AXISBANK', 'SBIN']
        energy_symbols = ['RELIANCE', 'ONGC', 'IOC', 'BPCL', 'GAIL']
        pharma_symbols = ['SUNPHARMA', 'DRREDDY', 'CIPLA', 'LUPIN', 'BIOCON']
        
        symbol_upper = symbol.upper()
        
        if any(tech in symbol_upper for tech in tech_symbols):
            return 'Technology'
        elif any(bank in symbol_upper for bank in banking_symbols):
            return 'Banking'
        elif any(energy in symbol_upper for energy in energy_symbols):
            return 'Energy'
        elif any(pharma in symbol_upper for pharma in pharma_symbols):
            return 'Pharmaceuticals'
        else:
            return 'Others'
    
    def train_models(self, user_profiles: List[Dict], portfolio_results: List[Dict]) -> Dict[str, Any]:
        """Train explainable models for portfolio recommendation."""
        
        # Prepare training data
        features_df, targets = self.prepare_training_data(user_profiles, portfolio_results)
        
        # Scale features
        features_scaled = self.scaler.fit_transform(features_df)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            features_scaled, targets, test_size=0.2, random_state=42
        )
        
        # Train portfolio performance model
        self.portfolio_model.fit(X_train, y_train)
        portfolio_predictions = self.portfolio_model.predict(X_test)
        portfolio_mse = mean_squared_error(y_test, portfolio_predictions)
        
        # Train risk classification model
        risk_labels = ['Low' if t < 0.10 else 'Medium' if t < 0.15 else 'High' for t in targets]
        risk_encoded = self.label_encoder.fit_transform(risk_labels)
        
        X_train_risk, X_test_risk, y_train_risk, y_test_risk = train_test_split(
            features_scaled, risk_encoded, test_size=0.2, random_state=42
        )
        
        self.risk_model.fit(X_train_risk, y_train_risk)
        risk_predictions = self.risk_model.predict(X_test_risk)
        risk_accuracy = accuracy_score(y_test_risk, risk_predictions)
        
        self.is_trained = True
        
        return {
            'portfolio_mse': portfolio_mse,
            'risk_accuracy': risk_accuracy,
            'portfolio_feature_importance': self.portfolio_model.feature_importances_,
            'risk_feature_importance': self.risk_model.feature_importances_,
            'feature_names': self.feature_names
        }
    
    def explain_portfolio_recommendation(self, user_profile: Dict, portfolio_result: Dict) -> Dict[str, Any]:
        """Provide detailed explanation for portfolio recommendation."""
        
        if not self.is_trained:
            # Generate synthetic training data for demonstration
            self._train_with_synthetic_data()
        
        # Prepare user features
        user_features = self._prepare_user_features(user_profile, portfolio_result)
        user_features_scaled = self.scaler.transform([user_features])
        
        # Get predictions
        expected_return = self.portfolio_model.predict(user_features_scaled)[0]
        risk_probability = self.risk_model.predict_proba(user_features_scaled)[0]
        
        # Calculate feature contributions (SHAP-like)
        feature_contributions = self._calculate_feature_contributions(user_features_scaled[0])
        
        # Generate explanations
        explanations = self._generate_explanations(user_features, feature_contributions, expected_return)
        
        return {
            'expected_return': expected_return,
            'risk_probabilities': {
                'Low': risk_probability[0] if len(risk_probability) > 0 else 0.33,
                'Medium': risk_probability[1] if len(risk_probability) > 1 else 0.33,
                'High': risk_probability[2] if len(risk_probability) > 2 else 0.34
            },
            'feature_contributions': feature_contributions,
            'explanations': explanations,
            'confidence_score': self._calculate_confidence_score(user_features_scaled[0])
        }
    
    def _prepare_user_features(self, user_profile: Dict, portfolio_result: Dict) -> List[float]:
        """Prepare user features for prediction."""
        features = []
        
        # Match the training feature order
        features.append(user_profile.get('age', 35))
        features.append(user_profile.get('income', 1500000))
        features.append(user_profile.get('investment_horizon', 10))
        features.append(user_profile.get('loss_aversion', 5))
        features.append(user_profile.get('overconfidence', 5))
        features.append(user_profile.get('herding_tendency', 5))
        features.append(user_profile.get('anchoring_bias', 5))
        features.append(user_profile.get('disposition_effect', 5))
        features.append(portfolio_result.get('risk_tolerance_score', 0.5))
        features.append(portfolio_result.get('volatility_target', 15.0))
        features.append(len(portfolio_result.get('portfolio_weights', {})))
        features.append(max(portfolio_result.get('portfolio_weights', {}).values()) if portfolio_result.get('portfolio_weights') else 0.05)
        features.append(self._calculate_diversification_ratio(portfolio_result.get('portfolio_weights', {})))
        features.append(self._calculate_sector_concentration(portfolio_result.get('portfolio_weights', {})))
        
        # Investment goal encoding
        goal_mapping = {'wealth_preservation': 1, 'wealth_accumulation': 2, 'aggressive_growth': 3}
        features.append(goal_mapping.get(user_profile.get('investment_goal', 'wealth_accumulation'), 2))
        
        return features
    
    def _calculate_feature_contributions(self, user_features_scaled: np.ndarray) -> Dict[str, float]:
        """Calculate SHAP-like feature contributions."""
        
        # Use tree-based feature importance as proxy for SHAP values
        portfolio_importance = self.portfolio_model.feature_importances_
        
        # Calculate contributions based on feature values and importance
        contributions = {}
        baseline_prediction = 0.12  # Baseline expected return
        
        for i, (feature_name, importance) in enumerate(zip(self.feature_names, portfolio_importance)):
            # Normalize feature value impact
            feature_value = user_features_scaled[i]
            contribution = feature_value * importance * 0.05  # Scale factor
            contributions[feature_name] = contribution
        
        return contributions
    
    def _generate_explanations(self, user_features: List[float], contributions: Dict[str, float], expected_return: float) -> Dict[str, Any]:
        """Generate human-readable explanations."""
        
        explanations = {
            'summary': f"Expected portfolio return: {expected_return*100:.2f}% annually",
            'key_factors': [],
            'recommendations': [],
            'risk_factors': []
        }
        
        # Sort contributions by absolute value
        sorted_contributions = sorted(contributions.items(), key=lambda x: abs(x[1]), reverse=True)
        
        # Top positive factors
        positive_factors = [(k, v) for k, v in sorted_contributions if v > 0][:3]
        negative_factors = [(k, v) for k, v in sorted_contributions if v < 0][:3]
        
        # Generate explanations for top factors
        for factor, contribution in positive_factors:
            explanation = self._explain_factor(factor, contribution, user_features, True)
            explanations['key_factors'].append(explanation)
        
        for factor, contribution in negative_factors:
            explanation = self._explain_factor(factor, contribution, user_features, False)
            explanations['risk_factors'].append(explanation)
        
        # Generate recommendations
        explanations['recommendations'] = self._generate_recommendations(user_features, contributions)
        
        return explanations
    
    def _explain_factor(self, factor_name: str, contribution: float, user_features: List[float], is_positive: bool) -> str:
        """Generate explanation for a specific factor."""
        
        factor_explanations = {
            'age': lambda x: f"Age ({int(user_features[0])}) suggests {'higher' if is_positive else 'lower'} risk capacity",
            'income': lambda x: f"Income level (₹{user_features[1]:,.0f}) indicates {'strong' if is_positive else 'moderate'} investment capacity",
            'investment_horizon': lambda x: f"Investment horizon ({int(user_features[2])} years) allows for {'aggressive' if is_positive else 'conservative'} strategy",
            'loss_aversion': lambda x: f"Loss aversion level ({int(user_features[3])}/10) {'reduces' if not is_positive else 'supports'} risk taking",
            'overconfidence': lambda x: f"Confidence level ({int(user_features[4])}/10) {'increases' if is_positive else 'moderates'} expected returns",
            'risk_tolerance_score': lambda x: f"Risk tolerance ({user_features[8]:.2f}) {'enables' if is_positive else 'limits'} growth potential",
            'diversification_ratio': lambda x: f"Portfolio diversification ({user_features[12]:.2f}) {'enhances' if is_positive else 'reduces'} risk-adjusted returns"
        }
        
        if factor_name in factor_explanations:
            return factor_explanations[factor_name](contribution)
        else:
            return f"{factor_name.replace('_', ' ').title()} {'positively' if is_positive else 'negatively'} impacts portfolio performance"
    
    def _generate_recommendations(self, user_features: List[float], contributions: Dict[str, float]) -> List[str]:
        """Generate actionable recommendations."""
        
        recommendations = []
        
        # Analyze key metrics
        age = user_features[0]
        investment_horizon = user_features[2]
        loss_aversion = user_features[3]
        risk_tolerance = user_features[8]
        diversification = user_features[12]
        
        if age < 30 and investment_horizon > 10:
            recommendations.append("Consider increasing equity allocation for long-term growth")
        
        if loss_aversion > 7:
            recommendations.append("Focus on downside protection and defensive sectors")
        
        if diversification < 0.7:
            recommendations.append("Improve diversification across sectors and market caps")
        
        if risk_tolerance > 0.7:
            recommendations.append("Consider growth-oriented stocks and emerging sectors")
        
        return recommendations
    
    def _calculate_confidence_score(self, user_features_scaled: np.ndarray) -> float:
        """Calculate confidence score for the prediction."""
        
        # Use ensemble variance as confidence proxy
        predictions = []
        for estimator in self.portfolio_model.estimators_[:10]:  # Use first 10 trees
            pred = estimator.predict([user_features_scaled])[0]
            predictions.append(pred)
        
        variance = np.var(predictions)
        # Convert variance to confidence (lower variance = higher confidence)
        confidence = max(0.1, 1.0 - min(variance * 10, 0.9))
        
        return confidence
    
    def _train_with_synthetic_data(self):
        """Train models with synthetic data for demonstration."""
        
        # Generate synthetic training data
        np.random.seed(42)
        n_samples = 1000
        
        synthetic_profiles = []
        synthetic_results = []
        
        for _ in range(n_samples):
            # Generate diverse user profiles
            age = np.random.normal(40, 10)
            income = np.random.normal(1500000, 500000)
            horizon = np.random.choice([3, 5, 7, 10, 15, 20])
            
            profile = {
                'age': max(25, min(65, age)),
                'income': max(500000, income),
                'investment_goal': np.random.choice(['wealth_preservation', 'wealth_accumulation', 'aggressive_growth']),
                'investment_horizon': horizon,
                'loss_aversion': np.random.randint(1, 11),
                'overconfidence': np.random.randint(1, 11),
                'herding_tendency': np.random.randint(1, 11),
                'anchoring_bias': np.random.randint(1, 11),
                'disposition_effect': np.random.randint(1, 11)
            }
            
            # Generate corresponding portfolio results
            risk_score = np.random.uniform(0.2, 0.8)
            volatility = np.random.uniform(8, 25)
            portfolio_size = np.random.randint(15, 35)
            
            # Simulate portfolio weights
            weights = {}
            for i in range(portfolio_size):
                weights[f'STOCK_{i}'] = np.random.uniform(0.01, 0.08)
            
            # Normalize weights
            total_weight = sum(weights.values())
            weights = {k: v/total_weight for k, v in weights.items()}
            
            result = {
                'risk_tolerance_score': risk_score,
                'volatility_target': volatility,
                'portfolio_weights': weights,
                'backtest_metrics': {
                    'annualized_return': max(0.05, np.random.normal(0.12 + risk_score * 0.08, 0.03))
                }
            }
            
            synthetic_profiles.append(profile)
            synthetic_results.append(result)
        
        # Train models
        self.train_models(synthetic_profiles, synthetic_results)
    
    def create_explanation_visualizations(self, explanation_result: Dict[str, Any]) -> Dict[str, go.Figure]:
        """Create visualization charts for explanations."""
        
        visualizations = {}
        
        # Feature contributions chart
        contributions = explanation_result['feature_contributions']
        feature_names = list(contributions.keys())
        values = list(contributions.values())
        
        colors = ['green' if v > 0 else 'red' for v in values]
        
        fig_contributions = go.Figure(data=[
            go.Bar(
                x=values,
                y=feature_names,
                orientation='h',
                marker_color=colors,
                text=[f"{v:.4f}" for v in values],
                textposition='auto'
            )
        ])
        
        fig_contributions.update_layout(
            title="Feature Contributions to Portfolio Performance",
            xaxis_title="Contribution to Expected Return",
            yaxis_title="Features",
            height=400
        )
        
        visualizations['feature_contributions'] = fig_contributions
        
        # Risk probability chart
        risk_probs = explanation_result['risk_probabilities']
        
        fig_risk = go.Figure(data=[
            go.Bar(
                x=list(risk_probs.keys()),
                y=list(risk_probs.values()),
                marker_color=['green', 'orange', 'red'],
                text=[f"{v:.2%}" for v in risk_probs.values()],
                textposition='auto'
            )
        ])
        
        fig_risk.update_layout(
            title="Risk Assessment Probabilities",
            xaxis_title="Risk Level",
            yaxis_title="Probability",
            height=300
        )
        
        visualizations['risk_probabilities'] = fig_risk
        
        # Confidence gauge
        confidence = explanation_result['confidence_score']
        
        fig_confidence = go.Figure(go.Indicator(
            mode="gauge+number",
            value=confidence * 100,
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': "Prediction Confidence"},
            gauge={
                'axis': {'range': [None, 100]},
                'bar': {'color': "darkblue"},
                'steps': [
                    {'range': [0, 50], 'color': "lightgray"},
                    {'range': [50, 80], 'color': "yellow"},
                    {'range': [80, 100], 'color': "green"}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': 90
                }
            }
        ))
        
        fig_confidence.update_layout(height=300)
        visualizations['confidence'] = fig_confidence
        
        return visualizations