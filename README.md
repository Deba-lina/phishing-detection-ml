# Phishing Website Detection Using Machine Learning

A reproducible machine-learning project that classifies websites as **legitimate** or **phishing** from URL- and domain-level features. It is designed as a clear, defensible baseline for a B.Tech CSE final-semester project.

## Problem statement

Phishing pages imitate trusted websites to steal credentials and financial information. Manual checking does not scale, so this project uses supervised machine learning to identify suspicious websites from measurable indicators such as URL structure, SSL state, domain age, redirects, and traffic signals.

## Highlights

- Leakage-safe preprocessing: removes row identifiers before training.
- Stratified train/test split for fair class representation.
- Scaled Logistic Regression baseline with balanced class weights.
- Accuracy, precision, recall, F1-score, ROC-AUC, confusion matrix, and full classification report.
- JSON experiment record and optional reusable model artifact.
- Deterministic experiments using a configurable random seed.

## Project structure

```text
.
├── code.py                 # Training and evaluation command-line program
├── phishing.csv            # Dataset (kept local; not committed)
├── requirements.txt        # Python dependencies
├── docs/PROJECT_REPORT.md  # Submission/report outline
└── artifacts/              # Generated metrics and models (ignored by Git)
```

## Setup

Requires Python 3.10 or later.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

## Run an experiment

```powershell
python code.py --data phishing.csv --save-model artifacts/phishing_model.joblib
```

The terminal displays evaluation results. A detailed, machine-readable report is stored in `artifacts/metrics.json`; the optional `.joblib` file can be loaded later for inference.

### Options

```text
--data PATH           Dataset location (default: phishing.csv)
--target COLUMN       Target column (default: Result)
--test-size FLOAT     Held-out proportion (default: 0.20)
--random-state INT    Reproducibility seed (default: 42)
--save-model PATH     Save trained model pipeline
--metrics-out PATH    Metrics report location
```

## Dataset and labels

The script expects a CSV with numeric feature columns and a `Result` target. For the included UCI-style dataset, `-1` means phishing and `1` means legitimate; the program normalizes this to `0` = legitimate and `1` = phishing. It also accepts an already normalized `0`/`1` target. The leading `index`/`id` column is discarded automatically because it is not a genuine website feature.

## Evaluation notes

Accuracy alone can hide costly phishing false negatives. For the project demonstration, emphasize **recall** (how many phishing pages are detected), **precision** (how many alerts are trustworthy), and the confusion matrix alongside accuracy. Always report results from the held-out test set, not the training set.

## Future work

- Compare Logistic Regression with Random Forest, XGBoost, and SVM using cross-validation.
- Add feature-importance analysis and SHAP explanations.
- Build a Flask/Streamlit interface that accepts a URL and extracts features securely.
- Add drift monitoring and periodically retrain on recent phishing feeds.

## Academic use

This is an educational classifier, not a complete security product. Real deployments require safe URL feature extraction, privacy controls, adversarial testing, continuous data updates, and human review for high-risk decisions.
