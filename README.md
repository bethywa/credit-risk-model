# 💳 Credit Scoring Business Understanding
This project aims to develop a credit scoring model for Bati Bank's new "buy-now-pay-later" (BNPL) service. The following sections address the core business and regulatory considerations that shape this technical task.

## 1. Basel II Accord, Risk Measurement, and Model Interpretability
The Basel II Capital Accord is a foundational international banking regulation. Its primary goal is to ensure banks hold enough capital to withstand unexpected losses. A key innovation of Basel II was allowing banks to develop their own Advanced Internal Ratings-Based (A-IRB) models to estimate crucial risk parameters like the Probability of Default (PD).

This privilege comes with significant responsibility, creating a direct need for interpretable and well-documented models:

Regulatory Scrutiny & Pillar 3: Basel II's third pillar mandates increased public disclosure about how banks quantify and manage their risks. Regulators must be able to audit and validate our model's logic and assumptions. An opaque "black-box" model would fail this requirement.

Justifying Decisions: Banking regulations often protect consumers from adverse decisions based on uninterpretable models. We must be able to explain why a customer was assigned a high-risk score, both for internal governance and potential customer inquiries.

Managing Model Risk: Inaccurate or unstable models can lead to poor lending decisions and significant financial losses. A well-documented, interpretable model is easier to monitor, validate, and debug over time, which is a core part of sound model risk management.

## 2. The Necessity and Risks of a Proxy Variable
Our project faces a fundamental challenge: we have e-commerce transaction data but no historical loan performance data. A direct "default" label does not exist.

Why a Proxy is Necessary: To train any predictive model, we need a target variable. Creating a proxy variable from available behavioral data (like Recency, Frequency, and Monetary values) is the only way to formulate a supervised learning problem. We assume that past shopping behavior is indicative of future financial responsibility, a common hypothesis in alternative credit scoring.

Potential Business Risks: Predictions based on a proxy are inherently risky:

Misaligned Assumptions: The core assumption—that shopping engagement directly correlates with creditworthiness—may be flawed. For example, a frugal saver might be a low-risk borrower but appear "inactive" on our platform.

Fairness and Bias: If the proxy is inadvertently correlated with sensitive attributes (e.g., through product categories popular in certain demographics), the model could perpetuate or amplify biases, leading to unfair lending practices and regulatory issues.

Performance Uncertainty: The model's accuracy in predicting true loan default is unknown until real BNPL data is collected, creating a "blind spot" in our initial risk assessment.

## 3.📊 Model Selection Trade-offs: Interpretability vs. Performance
When choosing a credit scoring model in a regulated environment, we must balance interpretability and performance.

1. Simple, Interpretable Model (e.g., Logistic Regression with WoE)
🔍 Interpretability: High ("White-Box"). The logic is clear because each feature's relationship to the outcome is direct and can be expressed in a simple scorecard. This makes it easy to explain why a decision was made.

📈 Performance: Potentially Lower. It might miss complex, non-linear patterns in the data, which can limit its predictive power.

⚖️ Regulatory Fit: Excellent. This method is well-established and trusted by regulators. It naturally creates a clear audit trail, making it straightforward to justify decisions for Basel II compliance.

🛠️ Implementation: Straightforward. It is simpler to build, monitor, and deploy as a standard scorecard.

2. Complex, High-Performance Model (e.g., Gradient Boosting)
🔍 Interpretability: Low ("Black-Box"). While tools like SHAP can help explain its decisions after the fact, the model's internal decision-making process is inherently complex and not natively transparent.

📈 Performance: Generally Higher. This type of model excels at finding intricate patterns and interactions in data, often leading to superior predictive accuracy.

⚖️ Regulatory Fit: Challenging. Its lack of transparency makes regulatory approval more difficult. It requires building additional layers of Explainable AI (XAI) to meet compliance requirements.

🛠️ Implementation: More Complex. It needs a sophisticated MLOps pipeline to manage training, deployment, and the consistent generation of explanations.

### In a regulated financial context, interpretability and compliance often outweigh pure predictive performance, so careful consideration is required when choosing the model.

    

# Feature Engineering for Credit Risk Model

## Overview
This task focuses on transforming raw transaction data into meaningful features for credit risk modeling, using a modular and reproducible pipeline approach.

## Features Created

### 1. Time-Based Features
- **Transaction Timing**: Time of day, day of week, month, and year
- **Time Since First/Last Transaction**
- **Transaction Intervals**: Time between consecutive transactions

### 2. Customer Behavioral Features
- **Transaction Frequency**: Count of transactions per time period
- **Spending Patterns**: 
  - Average, minimum, maximum transaction amounts
  - Total spending per period
  - Transaction amount variability
- **Transaction Channel Analysis**: Usage patterns across different channels

### 3. Categorical Features
- **One-Hot Encoded**:
  - Currency Code
  - Product Category
  - Channel ID
  - Provider ID
- **Rare Label Encoding** for low-frequency categories

### 4. Advanced Features
- **Weight of Evidence (WoE)** transformations for selected numerical features
- **Monotonic Binning** of continuous variables
- **Interaction Terms** between key variables


# Target Engineering


# Proxy Target Variable Engineering

## Overview
This task creates a binary target variable for credit risk prediction using RFM (Recency, Frequency, Monetary) analysis and K-means clustering.

## Methodology

### 1. RFM Metrics Calculation
- **Recency (R)**: Days since last transaction
- **Frequency (F)**: Number of transactions per month
- **Monetary (M)**: Median transaction value

### 2. Data Preparation
- Log transformation of Frequency and Monetary values
- Standard scaling of features
- Handling of edge cases and outliers

### 3. Clustering
- K-means clustering (k=3) to identify customer segments
- Silhouette analysis for optimal cluster validation
- Cluster interpretation and risk scoring

### 4. Target Definition
- High-risk cluster: Customers with high recency (inactive) and moderate spending
- Binary target variable: `is_high_risk` (1 for high-risk, 0 otherwise)

## Results

### Cluster Distribution
- **Cluster 0 (Low Risk)**: 59,262 customers (62%)
  - Recent (3 days), frequent (80x/month), moderate spenders ($1,000)
- **Cluster 1 (Low Risk)**: 23,377 customers (24%)
  - Recent (6 days), regular (30x/month), high spenders ($6,000)
- **Cluster 2 (High Risk)**: 13,023 customers (14%)
  - Inactive (54 days), moderate frequency (38x/month), moderate spenders ($1,000)

### Target Variable
- **High-risk customers**: 13,023 (13.6%)
- **Low-risk customers**: 82,639 (86.4%)



