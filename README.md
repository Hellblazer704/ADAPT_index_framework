# 📊 ADAPT Framework: Regime-Aware, Personalized Indexing System

Welcome to the official repository for the **Enhanced ADAPT Framework** — a next-generation indexing solution designed to outperform traditional benchmarks like NIFTY 50 by integrating real-time regime detection, factor-driven alpha sleeves, and investor personalization.

---

## 🧠 What is ADAPT?

**ADAPT** stands for:

> **A**lpha-aware  
> **D**efensive  
> **A**ctively-tilted  
> **P**ersonalized  
> **T**actical Indexing

Unlike static indices, ADAPT dynamically reallocates based on:
- Market regime (Bull / Bear / Sideways)
- Behavioral personas (Conservative, Moderate, Aggressive)
- Factor scores (Quality, Momentum, Value, Growth)
- Real-world frictions (slippage, commissions, turnover constraints)

---

## 🔧 Features

- ✅ **Regime Detection Engine** using price/MAs and volatility signals  
- 🧮 **Mathematical Portfolio Construction**: Core, Tactical, Defensive sleeves  
- 🏗️ **Optimization Techniques**: Mean-variance, Risk parity, Turnover control  
- 🔄 **Rebalancing Framework**: Monthly, Bi-weekly, or Quarterly as per profile  
- 📈 **Comprehensive Backtesting**: Alpha, beta, Sharpe, Sortino, drawdowns  
- 🧾 **Realistic Costs**: Transaction cost modeling (slippage + commission)  
- 📊 **Benchmarking vs NIFTY 50 TRI**

---

## 📁 Folder Structure

```plaintext
adapt-framework/
├── data/                    # Raw price & rebalance inputs
├── src/                     # Regime detection, optimizers, backtester
│   ├── regime/              # Bull/Bear logic
│   ├── portfolio/           # Factor scoring & weighting
│   └── utils/               # Helpers, logger, constants
├── results/                 # Output CSVs, plots, summary tables
├── notebooks/               # Exploratory analysis & validation
├── README.md
└── requirements.txt
