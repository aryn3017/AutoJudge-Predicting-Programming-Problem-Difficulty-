## Automated Difficulty Prediction for Competitive Programming Problems
```
This project builds an intelligent system that automatically predicts the difficulty of competitive programming problems using only their textual descriptions. The system performs both:

Classification → Easy / Medium / Hard

Regression → Numerical difficulty score

No deep learning is used; the solution relies on classical machine learning and NLP techniques.
```
### Problem Motivation
```
Online platforms like Codeforces, CodeChef, and Kattis assign difficulty labels and scores based on human judgment and user feedback. This process is:

Subjective

Time-consuming

Inconsistent across platforms

This project demonstrates that problem difficulty can be reasonably inferred from text alone, enabling automated difficulty estimation.
```
### Dataset Description
```
Each data sample contains:

Field	Description
title	Problem title
description	Problem statement
input_description	Input format
output_description	Output format
sample_io	Sample input/output
problem_class	Easy / Medium / Hard
problem_score	Numerical difficulty score
Dataset Statistics

Total samples: 4112

Difficulty score range: 1.1 – 9.7

Class distribution:

Hard: 1941

Medium: 1405

Easy: 766
```
#### Project Pipeline
```
Raw Text Data
     ↓
Text Preprocessing
     ↓
Feature Engineering
     ↓
Train/Test Split
     ↓
Classification Model
Regression Model
     ↓
Evaluation
```
#### Data Preprocessing
```
Combined all text fields into a single full_text

Filled missing text values with empty strings

Applied light normalization:

Lowercasing

Whitespace normalization

Preserved:

Numbers

Mathematical symbols

Algorithmic keywords

This ensures that difficulty-related cues are not lost.
```
#### Exploratory Data Analysis (EDA)
```
EDA revealed several important insights:

Hard problems tend to be longer and more structured

Difficulty score correlates positively with:

Word count

Line count

Constraint density

Significant overlap between difficulty classes explains classification ambiguity

Algorithmic keywords (dp, graph, union find) strongly correlate with harder problems

(See figures referenced in the report.)
```
#### Feature Engineering
```
Three types of features were used:

1. Semantic Features

TF-IDF (unigrams + bigrams)

Vocabulary size limited to reduce noise

2. Structural Features

Character count

Word count

Line count

Digit count

3. Algorithmic Signals

Keyword frequency (dp, graph, greedy, etc.)

Symbol counts (<=, %, ^, ==)

All numeric features were standardized and combined with TF-IDF vectors.
```
#### Models Used
```
Baseline Model

Always predicts “Medium”

Accuracy ≈ 0.34

Serves as a lower bound

Classification Model

Random Forest Classifier

Captures non-linear relationships

Leverages structural and keyword features

Provides feature importance

Evaluation Metrics

Accuracy

Confusion Matrix

Result

Accuracy ≈ 0.53

This significantly outperforms the baseline but is limited by the high-dimensional sparse nature of text features.

Regression Model

Random Forest Regressor

Predicts continuous difficulty score

Robust to non-linear feature interactions

Evaluation Metrics

Mean Absolute Error (MAE)

Root Mean Squared Error (RMSE)
```
### Result
```
Reasonable prediction accuracy

Increased error around medium difficulty due to class overlap

📊 Evaluation Summary
Task	Metric	Result
Classification	Accuracy	~0.53
Classification	Baseline Accuracy	~0.34
Regression	MAE	1.6696534769403284
Regression	RMSE	0.5540704738760632
📌 Key Observations

Textual complexity is a strong proxy for difficulty

Keyword and symbol features significantly improve performance

Difficulty classes overlap heavily, limiting maximum accuracy

Random Forest models are interpretable but less optimal for sparse text compared to linear models
```
#### Conclusion

This project demonstrates that:

Competitive programming problem difficulty can be inferred using text alone

Classical machine learning models are effective when paired with good feature engineering

Combining classification and regression provides richer insights

Proper EDA is essential for understanding model limitations