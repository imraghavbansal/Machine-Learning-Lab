# Sonar Classification

A supervised machine learning project that uses sonar signal measurements to classify underwater objects as either a Rock or a Mine.

This project is part of my Machine Learning Lab, where I am building and studying practical machine learning projects with an emphasis on understanding the complete workflow, experimentation and model evaluation rather than only obtaining a final prediction.

---

## Overview

Sonar systems use sound waves to detect objects by analyzing the signals that return after interacting with them.

This project uses a dataset containing sonar signal measurements to build a binary classification system that predicts whether a detected object is:

- `R` → Rock
- `M` → Mine

The project starts with a Logistic Regression baseline and progressively evaluates preprocessing, cross-validation, alternative models and hyperparameter tuning before performing a final evaluation on an untouched test set.

The complete workflow is:

Dataset
   ↓
Data exploration
   ↓
Data quality checks
   ↓
Feature / target separation
   ↓
Train-test split
   ↓
Logistic Regression baseline
   ↓
Feature scaling
   ↓
Cross-validation
   ↓
SVM
   ↓
Random Forest
   ↓
Model comparison
   ↓
SVM hyperparameter tuning
   ↓
Final evaluation
   ↓
Error analysis
   ↓
Prediction on new data

---

### What all i learned from this project

Core ML Concepts
Supervised learning
Binary classification
Features and targets
Training data and test data
Model fitting
Prediction
Generalization
Data leakage
Reproducibility
Data Preparation
Loading datasets with Pandas
Inspecting dataset structure
Checking missing values
Checking duplicate observations
Understanding class distribution
Separating features from targets
Splitting data using stratification
Model Development
Logistic Regression
Feature scaling with StandardScaler
Support Vector Machines
Random Forest
Cross-validation
Hyperparameter tuning with GridSearchCV
Model Evaluation
Accuracy
Confusion matrix
Precision
Recall
F1 score
Error analysis
Held-out test evaluation
Experimental Thinking
