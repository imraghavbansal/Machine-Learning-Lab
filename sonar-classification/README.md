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

```text
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
