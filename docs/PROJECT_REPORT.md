# Project Report Guide: Phishing Website Detection Using Machine Learning

## Abstract

Phishing attacks use deceptive websites to obtain sensitive user information. This project develops a supervised machine-learning classifier that distinguishes phishing websites from legitimate ones using URL, domain, security, and popularity-related attributes. A scaled Logistic Regression model is trained on labelled examples with a stratified hold-out evaluation. The experiment reports accuracy, precision, recall, F1-score, ROC-AUC, and a confusion matrix. The work demonstrates an interpretable and reproducible baseline for automated phishing detection.

## Objectives

1. Prepare a labelled phishing-website dataset for supervised learning.
2. Train a reproducible binary classification baseline.
3. Evaluate the model using security-relevant metrics, not accuracy alone.
4. Save experiment outputs and the trained model for repeatable demonstrations.

## Methodology

```text
CSV dataset → validation & identifier removal → stratified split →
StandardScaler + Logistic Regression → test-set evaluation → metrics/model artifacts
```

The `index` field is removed because it identifies a record rather than describing a website. Retaining it can create data leakage and overstate performance. In the included UCI-style data, `-1` is phishing and `1` is legitimate; labels are normalized before training. Stratification preserves the legitimate/phishing class balance in both training and test data.

## Software requirements

- Python 3.10+
- pandas
- scikit-learn
- joblib

## Experimental procedure

1. Create and activate a virtual environment.
2. Install packages using `python -m pip install -r requirements.txt`.
3. Run `python code.py --data phishing.csv --save-model artifacts/phishing_model.joblib`.
4. Copy the output metrics from `artifacts/metrics.json` into the Results chapter.
5. Interpret false positives and false negatives from the confusion matrix.

## Results table template

| Metric | Value |
|---|---:|
| Accuracy | Fill from `metrics.json` |
| Precision | Fill from `metrics.json` |
| Recall | Fill from `metrics.json` |
| F1-score | Fill from `metrics.json` |
| ROC-AUC | Fill from `metrics.json` |

Do not claim a fixed result in the report before running the exact version of the code and dataset used for submission.

## Limitations and ethics

Dataset-based performance may not generalize to newly created or adversarial phishing pages. The classifier should support—not replace—browser protections and analyst judgment. Only inspect URLs in an isolated, safe environment; do not automatically visit untrusted links.

## Suggested viva questions

1. Why is recall important in phishing detection?
2. What is data leakage, and why is the index column removed?
3. Why use a stratified split?
4. How do precision and recall trade off?
5. Why should a real phishing detector be retrained over time?
