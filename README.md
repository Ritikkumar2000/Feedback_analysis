# Sentiment Analysis - 4-Class Feedback Classifier

**Accuracy:** 70.7% | **Model:** Logistic Regression | **Data:** 1,022 train / 256 test

## Setup
pip install -r requirements.txt
python modeltrain.py
streamlit run app.py

## What It Does
Classifies feedback into 4 categories: **Positive**, **Negative**, **Positive+Suggestion**, **Negative+Suggestion** using TF-IDF features and SMOTE for class balancing.

## Key Features
- TF-IDF vectorization (5,000 features) | SMOTE for imbalanced data | GridSearchCV tuning | Confusion matrix visualization

## Files
`modeltrain.py` (training) | `app.py` (web UI) | `train_data.csv` (1,022 samples) | `test_data.csv` (256 samples) | `sentiment_model.pkl` (model)

## Performance
| Model | Accuracy |
|-------|----------|
| **Logistic Regression** | **70.7%** ✓ |   main base model
| Random Forest | 67.2% |
| SVM | 69.5% |