# Sephora Product Success Predictor

## Overview
This project is an end-to-end machine learning application that predicts whether a Sephora product review is likely to be high-rated. The model uses both natural language processing (NLP) and structured product features to generate predictions and provide interpretable insights.

The application is deployed using Streamlit and designed to simulate a real-world ML product used in retail decision-making.

---

## Problem Statement
Beauty retailers like Sephora rely heavily on customer reviews to evaluate product performance. However, manually analyzing large volumes of reviews is inefficient.

This project aims to:
- Predict whether a product review will be high-rated
- Identify key features that influence product success
- Provide explainable insights for business decision-making

---

## Features
- Text + structured data modeling
- Real-time prediction interface (Streamlit)
- Probability-based predictions
- Explainable AI using SHAP
- Clean UI with user-friendly inputs
- Business-focused insights section

---

## Dataset
The model is built using a Sephora product dataset containing:
- Product information (price, ingredients, categories)
- Customer reviews and ratings
- Product highlights

Key engineered features include:
- `combined_text` (review + ingredients + highlights)
- `price_usd`
- `ingredient_count`
- `review_length`

---

## Machine Learning Pipeline

### 1. Data Preprocessing
- Text cleaning (lowercasing, removing special characters)
- Feature engineering (ingredient count, review length)
- TF-IDF vectorization for text features
- Scaling and encoding for structured data

### 2. Model Development
- Baseline models: Logistic Regression, Random Forest
- Final model: XGBoost classifier
- Class imbalance handled using SMOTE / class weighting
- Hyperparameter tuning for performance improvement

### 3. Evaluation Metrics
- Accuracy
- F1 Score
- ROC-AUC
- Confusion Matrix

---

## Model Performance
*(Replace with your actual scores)*

- Accuracy: ~0.80  
- F1 Score: ~0.78  
- ROC-AUC: ~0.82  

The model performs well overall, though predicting high-rated reviews remains challenging due to class imbalance.

---

## Explainable AI (XAI)
The application integrates SHAP (SHapley Additive exPlanations) to provide:

- Global feature importance
- Local explanations for individual predictions
- Insight into how features like price, text sentiment, and ingredients impact predictions

---

## Deployment
The model is deployed as an interactive dashboard using Streamlit.

### Run locally:
```bash
pip install -r requirements.txt
streamlit run app.py
