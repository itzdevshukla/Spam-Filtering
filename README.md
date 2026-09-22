# 🛡️ SpamShield — Sentinel Shield Edition (Architect: Harsh)
### Academic Machine Learning & NLP System using Logistic Regression | Author: Harsh

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![Django 5.2](https://img.shields.io/badge/Django-5.2-092E20.svg)](https://www.djangoproject.com/)
[![Scikit-Learn](https://img.shields.io/badge/scikit--learn-1.7%2B-orange.svg)](https://scikit-learn.org/)
[![Accuracy 98.8%](https://img.shields.io/badge/Accuracy-98.84%25-brightgreen.svg)]()
[![ROC-AUC 0.998](https://img.shields.io/badge/ROC--AUC-0.9983-blueviolet.svg)]()
[![Tests 15/15](https://img.shields.io/badge/Tests-15%20Passed-success.svg)]()

---

## 📌 Project Overview
**SpamShield** is an academic-grade machine learning and natural language processing system for binary classification of text messages into **HAM (0, Legitimate)** and **SPAM (1, Unsolicited / Malicious)**. 

The primary algorithm is **Logistic Regression**, implemented both **from first principles (Scratch)** using vectorized Batch Gradient Descent and via a tuned **Scikit-Learn pipeline**. The system features real-time feature attribution explainability, batch CSV processing, a modern Django web interface, and full REST API support.

---

## 🔬 Architecture & ML Pipeline

```
                                      +------------------------------------+
                                      |      UCI SMS Spam Collection       |
                                      |        (5,574 raw messages)        |
                                      +-----------------+------------------+
                                                        |
                                                        v
                                      +------------------------------------+
                                      |     Data Cleaning & Ingestion      |
                                      |  - Deduplication (5,160 clean)     |
                                      |  - Target Encoding (ham=0, spam=1) |
                                      +-----------------+------------------+
                                                        |
                                                        v
                                      +------------------------------------+
                                      |    Linguistic Preprocessing &      |
                                      |      Spam Signal Extraction        |
                                      |  - Preserve URLs, Currency, Phones |
                                      |  - Exclamation & Caps Ratios       |
                                      +-----------------+------------------+
                                                        |
                                                        v
                                      +------------------------------------+
                                      |      Feature Engineering           |
                                      |  - TF-IDF N-grams (1, 2)           |
                                      |  - Scaled Meta-Features Union      |
                                      +--------+------------------+--------+
                                               |                  |
                       +-----------------------+                  +----------------------+
                       v                                                                 v
+---------------------------------------+                      +---------------------------------------+
|    Scratch Logistic Regression        |                      |    Scikit-Learn Pipeline              |
|  - Sigmoid Activation Function        |                      |  - L-BFGS Solver                      |
|  - Vectorized Binary Cross-Entropy    |                      |  - Class Weight: Balanced             |
|  - L2 Ridge Regularization            |                      |  - Threshold Optimization (&tau;=0.60)|
|  - Batch Gradient Descent Optimizer   |                      |  - Calibrated Probability Output      |
+----------------------+----------------+                      +-----------------+---------------------+
                       |                                                         |
                       +-----------------------+---------------------------------+
                                               |
                                               v
                               +-------------------------------+
                               |  Holdout Evaluation (N=1,032) |
                               |  - Acc: 98.8% | ROC-AUC: 0.998|
                               |  - False Positive Rate: 0.44% |
                               +---------------+---------------+
                                               |
                                               v
                               +-------------------------------+
                               |     Django 5 Web Application  |
                               |  - Real-time Message Analyzer |
                               |  - Log-Odds Feature Explainer |
                               |  - Batch CSV Classifier       |
                               |  - REST API & Audit Database  |
                               +-------------------------------+
```

---

## 📊 Empirical Evaluation Benchmark (Unseen Holdout: N = 1,032)

| Evaluation Metric | Scratch LR ($\tau=0.50$) | Scikit-Learn LR ($\tau=0.50$) | Scikit-Learn (Tuned $\tau=0.60$) |
| :--- | :---: | :---: | :---: |
| **Accuracy** | 98.26% | 98.55% | **98.84%** |
| **Precision (PPV)** | **98.25%** | 94.49% | 96.77% |
| **Recall (Sensitivity)** | 87.50% | **93.75%** | **93.75%** |
| **F1-Score** | 92.56% | 94.12% | **95.24%** |
| **Specificity (TNR)** | **99.78%** | 99.23% | 99.56% |
| **False Positive Rate (FPR)** | **0.22%** | 0.77% | 0.44% |
| **ROC - AUC** | 0.9949 | **0.9983** | **0.9983** |
| **PR - AUC** | 0.9768 | **0.9899** | **0.9899** |
| **True Positives (TP)** | 112 | 120 | 120 |
| **False Positives (FP)** | **2** | 7 | 4 |
| **True Negatives (TN)** | 902 | 897 | 900 |
| **False Negatives (FN)** | 16 | 8 | 8 |

---

## 📁 Repository Directory Structure

```
Spam Filtering/
├── data/
│   ├── raw/                   # Raw downloaded datasets (UCI SMS, SpamAssassin)
│   ├── processed/             # Cleaned, standardized dataset (sms_spam_clean.csv)
│   └── external/              # Generalization test dataset (SpamAssassin)
├── notebooks/
│   └── 01_exploratory_data_analysis.ipynb   # Interactive Jupyter EDA notebook
├── outputs/
│   ├── eda/                   # 8 generated publication plots & wordclouds
│   └── evaluation/            # Confusion matrices, ROC, PR, threshold, loss curves
├── models/
│   ├── spam_classifier_pipeline.joblib      # Serialized Scikit-learn pipeline
│   ├── scratch_logistic_regression.json    # Scratch model weights & parameters
│   └── model_metadata.json                 # Comprehensive training metadata
├── src/
│   ├── __init__.py
│   ├── data_loader.py         # Automated dataset downloaders & parsers
│   ├── preprocessor.py        # Text cleaning, regex normalization, meta-feature extractor
│   ├── feature_engineering.py # TF-IDF + meta-feature union pipeline
│   ├── eda.py                 # Automated EDA visualizer and report generator
│   └── models/
│       ├── __init__.py
│       ├── scratch_logistic_regression.py   # Vectorized Logistic Regression from scratch
│       ├── sklearn_logistic_regression.py   # Scikit-learn model with GridSearch & tuning
│       └── evaluator.py                     # Metric calculation & diagnostic plotting
├── scripts/
│   ├── download_data.py       # Standalone downloader script
│   ├── run_eda.py             # Script to generate all 8 EDA figures
│   ├── train_and_evaluate.py  # End-to-end training, benchmarking, & export
│   └── test_generalization.py # External validation on SpamAssassin dataset
├── spamshield_web/            # Django 5 web application
│   ├── manage.py
│   ├── db.sqlite3             # SQLite database
│   ├── spamshield_project/    # Project configuration (settings, urls, wsgi)
│   └── detector/              # Detector app
│       ├── models.py          # PredictionLog model
│       ├── services.py        # Inference & log-odds explainability engine
│       ├── views.py           # Real-time, batch, dashboard, and API views
│       ├── urls.py            # URL routing
│       ├── templates/detector/# Modern glassmorphic Bootstrap 5 templates
│       └── static/detector/   # CSS design system & interactive JavaScript
├── tests/
│   ├── test_preprocessor.py   # Unit tests for preprocessing & signal extraction
│   ├── test_scratch_model.py  # Unit tests for scratch model math & convergence
│   └── test_api.py            # Integration tests for Django endpoints
├── VIVA_GUIDE.md              # 25+ academic viva defense questions & full math derivations
├── requirements.txt           # Project dependencies
└── README.md                  # This file
```

---

## 🚀 Quickstart Guide

### 1. Prerequisites & Environment Setup
Clone the repository and install requirements:
```bash
git clone <repo-url>
cd "Spam Filtering"
pip install -r requirements.txt
```

### 2. Dataset Acquisition
Download the UCI SMS Spam Collection and SpamAssassin dataset:
```bash
python scripts/download_data.py
```

### 3. Generate Exploratory Data Analysis (EDA) Visualizations
Generate 8 high-resolution charts saved to `outputs/eda/`:
```bash
python scripts/run_eda.py
```

### 4. Train Models & Evaluate Benchmark
Train the scratch model, fit the Scikit-Learn pipeline, and evaluate on unseen holdout:
```bash
python scripts/train_and_evaluate.py
```

### 5. Run Generalization Test (SpamAssassin Corpus)
Evaluate cross-domain robustness:
```bash
python scripts/test_generalization.py
```

### 6. Run Unit & Integration Tests
Run the automated pytest test suite:
```bash
pytest tests/ -v
```

### 7. Launch the Django Web Application
Apply migrations and start the development server:
```bash
cd spamshield_web
python manage.py migrate
python manage.py runserver
```
Open your browser at **`http://127.0.0.1:8000/`**.

---

## 🌐 Web Interface Features

1. **Real-Time Message Analyzer (`/`)**:
   - Live text input with instant character and word counters.
   - Quick-load test presets (Authentic Ham, Lottery Scam, Bank Phishing, Casual Chat).
   - Animated **Spam Risk Gauge** with three risk tiers: Safe (<35%), Suspicious (35-70%), Critical Spam (>70%).
   - **Log-Odds Explainability Engine**: Displays exact token-by-token contributions ($w_i \cdot x_i$) showing which words triggered the spam decision.
2. **Batch Message Classifier (`/batch/`)**:
   - Supports uploading `.csv` files or pasting up to 500 messages at once.
   - Computes bulk metrics (Total, Spam %, Ham count) and filterable tabular preview.
   - One-click export to `spamshield_classified_batch.csv`.
3. **Model Evaluation & Academic Viva Dashboard (`/dashboard/`)**:
   - Live KPI cards, side-by-side benchmark tables, and embedded high-resolution diagnostic charts (Confusion Matrix, ROC Curve, PR Curve, Threshold Optimization, Loss Trajectory).
4. **Prediction Audit Log (`/history/`)**:
   - Persistent SQLite audit history tracking timestamps, classifications, probabilities, and latencies.

---

## 🔌 REST API Integration

### Single Message Classification
**Endpoint:** `POST /api/predict/`  
**Content-Type:** `application/json`

```bash
curl -X POST http://127.0.0.1:8000/api/predict/ \
  -H "Content-Type: application/json" \
  -d '{"text": "URGENT! You have won £1000 cash prize. Call 0800123456 now to claim."}'
```

**Response:**
```json
{
  "success": true,
  "label": "SPAM",
  "is_spam": true,
  "spam_probability": 0.9998,
  "ham_probability": 0.0002,
  "risk_level": "CRITICAL",
  "inference_time_ms": 3.42,
  "top_signals": [
    {"feature": "[meta]_has_urgent_keyword", "contribution": 3.12, "pushes_toward": "SPAM"},
    {"feature": "__currency__", "contribution": 2.45, "pushes_toward": "SPAM"}
  ]
}
```

---

## 🎓 Academic Defense & Viva Resources
For examiners and students preparing for project defense, consult **[VIVA_GUIDE.md](file:///C:/Users/Dev%20Shukla/Desktop/Spam%20Filtering/VIVA_GUIDE.md)** for:
- Complete mathematical derivations (Sigmoid inversion, Maximum Likelihood Estimation, Vectorized Gradient Descent).
- Deep-dive into trade-offs (False Positives vs False Negatives, Threshold Tuning).
- 25 comprehensive viva defense questions and model answers.
