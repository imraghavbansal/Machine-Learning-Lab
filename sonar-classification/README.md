# Sonar Classification

A supervised machine learning project that uses sonar signal measurements to classify underwater objects as either a **Rock** or a **Mine**.

This project is part of my **Machine Learning Lab**, where I am building and studying practical machine learning projects with an emphasis on understanding the complete workflow, experimentation and model evaluation rather than simply obtaining a final prediction.

---

## Overview

Sonar systems use sound waves to detect objects by analyzing the signals that return after interacting with them.

This project uses a dataset containing **60 sonar signal measurements** for each observation to build a binary classification system that predicts whether a detected object is:

* `R` → Rock
* `M` → Mine

The project begins with a **Logistic Regression baseline** and progressively explores data preparation, feature scaling, cross-validation, alternative models and hyperparameter tuning before evaluating the final model on an untouched test set.

### Machine Learning Workflow

```text
Dataset
   ↓
Data exploration
   ↓
Data quality checks
   ↓
Feature / target separation
   ↓
Stratified train-test split
   ↓
Logistic Regression baseline
   ↓
Feature scaling
   ↓
Cross-validation
   ↓
Alternative model evaluation
   ↓
SVM + Random Forest
   ↓
Model comparison
   ↓
Hyperparameter tuning
   ↓
Final evaluation
   ↓
Error analysis
   ↓
Prediction on new data
```

The main focus was understanding **why each step exists, what problem it solves and how to evaluate whether an experiment actually improved the model**.

---

## Dataset

The dataset contains:

* **208 observations**
* **60 numerical sonar features**
* **1 target variable**
* Two classes:

  * `R` → Rock
  * `M` → Mine

Because the dataset is relatively small, careful evaluation is important. A single train-test split can produce unstable results, which is why cross-validation was also used during model development.

---

## Project Structure

```text
sonar-classification/
├── sonar.csv
├── sonar_classification.ipynb
└── README.md
```

### Files

| File                         | Description                       |
| ---------------------------- | --------------------------------- |
| `sonar.csv`                  | Sonar signal dataset              |
| `sonar_classification.ipynb` | Complete experimentation notebook |
| `README.md`                  | Project documentation             |

---

## Data Preparation

Before training models, the dataset was inspected and prepared for experimentation.

The preprocessing workflow included:

* Loading the dataset with Pandas
* Inspecting dataset structure and dimensions
* Checking for missing values
* Checking for duplicate observations
* Understanding class distribution
* Separating features from the target
* Splitting the data into training and test sets
* Using **stratification** to preserve class proportions between the splits

The test set was kept separate throughout model development so that it could provide a more honest final evaluation.

---

## Models

### 1. Logistic Regression

Logistic Regression was used as the initial baseline.

The purpose of the baseline was not to immediately find the best model, but to establish a reference point against which later experiments could be compared.

---

### 2. Feature Scaling

`StandardScaler` was introduced to standardize the numerical features.

This was particularly relevant for distance and margin-based algorithms such as **Support Vector Machines**, where differences in feature scale can affect model behavior.

The effect of scaling was experimentally evaluated rather than assumed to automatically improve performance.

---

### 3. Support Vector Machine

SVM was evaluated as an alternative classification approach.

Different configurations were explored to understand how model behavior changes with different hyperparameters.

---

### 4. Random Forest

Random Forest was evaluated as a tree-based alternative to the linear and margin-based models.

This provided another perspective for comparing different model families on the same dataset.

---

## Cross-Validation

Instead of relying only on a single validation split, **5-fold cross-validation** was used during model development.

The training data was divided into five folds.

For each experiment:

```text
Fold 1 → Validation
Fold 2 → Training
Fold 3 → Training
Fold 4 → Training
Fold 5 → Training
```

This process was repeated so that every fold served as the validation set once.

The resulting scores were then compared to obtain a more reliable estimate of how the model performs across different subsets of the training data.

Cross-validation was used for **model development and comparison**, while the final test set remained untouched.

---

## Hyperparameter Tuning

After comparing the candidate models, **GridSearchCV** was used to search through selected SVM hyperparameter combinations.

The goal was not simply to maximize a single score, but to systematically test whether different configurations could improve the model's cross-validation performance.

The tuning process allowed the experiment to move from:

```text
Choose a model
      ↓
Test reasonable configurations
      ↓
Compare cross-validation performance
      ↓
Select the best configuration
      ↓
Evaluate once on untouched test data
```

This helped reinforce the difference between **model training, validation and final testing**.

---

## Model Evaluation

The models were evaluated using multiple classification metrics rather than accuracy alone.

### Accuracy

The proportion of predictions that were correct.

```text
Accuracy = Correct Predictions / Total Predictions
```

Useful when the class distribution is reasonably balanced, but it does not show which type of mistake the model is making.

### Confusion Matrix

The confusion matrix shows how predictions are distributed across the actual classes.

It helps identify:

* True positives
* True negatives
* False positives
* False negatives

### Precision

Of the samples predicted as a particular class, precision measures how many were actually that class.

```text
Precision = TP / (TP + FP)
```

### Recall

Of the samples that actually belong to a particular class, recall measures how many the model successfully identified.

```text
Recall = TP / (TP + FN)
```

### F1 Score

The F1 score combines precision and recall using their harmonic mean.

```text
F1 = 2 × (Precision × Recall) / (Precision + Recall)
```

Using multiple metrics provided a more complete picture of model behavior than relying on accuracy alone.

---

## Experimental Thinking

One of the main lessons from this project was that machine learning is not simply:

```text
data → model → accuracy
```

A more realistic workflow is:

```text
Understand the data
        ↓
Establish a baseline
        ↓
Evaluate
        ↓
Form a hypothesis
        ↓
Run a controlled experiment
        ↓
Compare results
        ↓
Tune when justified
        ↓
Evaluate on untouched data
        ↓
Analyze errors
```

Each experiment was performed to answer a specific question.

For example:

* Does feature scaling affect the model?
* Does another model family perform differently?
* Is the observed improvement consistent across cross-validation folds?
* Can hyperparameter tuning improve the selected model?
* Does the improvement remain when evaluated on completely unseen test data?

This shifted the project from simply implementing algorithms to thinking about **why an experiment should be performed and what its result actually means**.

---

## What I Learned

This project helped me understand the complete workflow of a **supervised classification problem**, from raw data preparation through final evaluation.

### Core Machine Learning Concepts

* Supervised learning
* Binary classification
* Features and targets
* Training data and test data
* Model fitting
* Prediction
* Generalization
* Data leakage
* Reproducibility

### Data Preparation

* Loading datasets with Pandas
* Inspecting dataset structure
* Checking missing values
* Checking duplicate observations
* Understanding class distribution
* Separating features from targets
* Splitting data using stratification

### Model Development

* Logistic Regression
* Feature scaling with `StandardScaler`
* Support Vector Machines
* Random Forest
* Cross-validation
* Hyperparameter tuning with `GridSearchCV`

### Model Evaluation

* Accuracy
* Confusion matrix
* Precision
* Recall
* F1 score
* Error analysis
* Held-out test evaluation

### Experimental Thinking

* Establishing a baseline before changing the model
* Forming hypotheses before experiments
* Comparing models under controlled conditions
* Understanding why cross-validation is useful
* Distinguishing validation performance from final test performance
* Avoiding data leakage
* Tuning only when there is a reason to tune
* Evaluating the final model on previously unseen data
* Understanding that a single test score does not fully describe real-world performance

---

## Results

The final tuned model achieved:

**95.24% accuracy on the held-out test set.**

However, this result should be interpreted carefully.

The dataset contains only **208 observations**, and the final test set contains only **21 samples**. With such a small test set, even a small number of different predictions can noticeably change the reported accuracy.

Therefore, the 95.24% test accuracy should **not be treated as a definitive estimate of real-world performance**.

The result is best understood as the outcome of this particular experiment and test split.

---

## Limitations

This project is primarily a **learning and experimentation project**.

The main limitations are:

* The dataset is relatively small with only 208 observations.
* The final test set contains only 21 samples.
* A single held-out test set can produce a noisy estimate of generalization performance.
* The dataset may not represent the full range of real-world sonar conditions.
* The project does not evaluate the model against an independent external dataset.
* The model has not been deployed into a production sonar system.

A stronger evaluation would require additional unseen data and potentially repeated experiments using different splits or an independent external validation dataset.

---

## Possible Future Improvements

Potential extensions include:

* Testing additional classification algorithms
* Performing more extensive feature analysis
* Exploring feature selection
* Evaluating additional hyperparameter configurations
* Using repeated cross-validation
* Testing the final model on an independent external dataset
* Building a small interface around the prediction system
* Packaging preprocessing and the trained model into a reusable inference pipeline

These are possible extensions rather than requirements for the current project.

---

## Learning Context

This project was built as part of my **Machine Learning Lab** to move beyond simply following a model-training tutorial and understand the reasoning behind a complete machine learning workflow.

The project started with a simple **Logistic Regression implementation** and was progressively extended through:

```text
Data quality checks
        ↓
Baseline model
        ↓
Evaluation
        ↓
Feature scaling
        ↓
Cross-validation
        ↓
Model comparison
        ↓
Hyperparameter tuning
        ↓
Final evaluation
        ↓
Error analysis
```

The focus throughout the project was on understanding **why each step exists, how the results change and how to determine whether those changes are meaningful**.

---

## Technologies Used

* **Python**
* **NumPy**
* **Pandas**
* **Matplotlib**
* **Seaborn**
* **Scikit-learn**
* **Jupyter Notebook**

---

## Running the Project

Clone the repository:

```bash
git clone https://github.com/imraghavbansal/Machine-Learning-Lab.git
cd Machine-Learning-Lab
```

Open the notebook:

```text
sonar-classification/sonar_classification.ipynb
```

Make sure the notebook is using the project's Python environment with the required dependencies installed.

The dataset should remain in the same directory as the notebook:

```text
sonar-classification/
├── sonar.csv
└── sonar_classification.ipynb
```

---

## Status

**Completed**

The final notebook contains the complete experimentation workflow, including:

* Data exploration
* Data quality checks
* Logistic Regression baseline
* Feature scaling
* Cross-validation
* SVM
* Random Forest
* Model comparison
* Hyperparameter tuning
* Final held-out test evaluation
* Error analysis
* Prediction workflow

The project is considered complete as a learning-focused classification project, with the future improvements above serving as optional extensions.

---

## Key Takeaway

The most important lesson from this project was not a particular algorithm or accuracy score.

It was learning to treat machine learning as an **experimental process**:

> Understand the data → establish a baseline → form a hypothesis → experiment → evaluate → compare → tune when justified → test on unseen data → analyze the errors.

That workflow is more important than simply getting a high score on one dataset.
