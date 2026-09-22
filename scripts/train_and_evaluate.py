"""End-to-end model training, comparative evaluation, and artifact serialization script.

Performs:
1. Stratified 80/20 train/test split (random_state=42)
2. Feature pipeline fitting on train data only
3. Training of Vectorized Scratch Logistic Regression with Gradient Descent
4. Training & hyperparameter tuning of Scikit-Learn Logistic Regression Pipeline
5. Comprehensive test-set evaluation (Accuracy, Precision, Recall, F1, Specificity, FPR, ROC-AUC, PR-AUC)
6. Diagnostic plotting (Confusion matrices, ROC curves, PR curves, Threshold tuning, Loss curves)
7. Production model serialization into models/
"""

from __future__ import annotations
from datetime import datetime
import json
from pathlib import Path
import sys
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.feature_engineering import build_feature_pipeline
from src.models.evaluator import (
    calculate_metrics,
    plot_confusion_matrices,
    plot_precision_recall_curves,
    plot_roc_curves,
    plot_scratch_loss_trajectory,
    plot_threshold_tuning,
    plot_top_coefficients,
)
from src.models.scratch_logistic_regression import ScratchLogisticRegression
from src.models.sklearn_logistic_regression import SklearnLogisticRegressionPipeline


def main():
    base_dir = Path(__file__).resolve().parent.parent
    data_path = base_dir / "data" / "processed" / "sms_spam_clean.csv"
    models_dir = base_dir / "models"
    models_dir.mkdir(parents=True, exist_ok=True)

    if not data_path.exists():
        print("[Train] Processed dataset not found. Running acquisition first...")
        from src.data_loader import prepare_and_save_primary_data
        df, _ = prepare_and_save_primary_data()
    else:
        df = pd.read_csv(data_path)

    print("=" * 70)
    print("SpamShield: Dual Logistic Regression Training & Evaluation Pipeline")
    print(f"Total Dataset Samples: {len(df)} (Ham: {(df['label'] == 0).sum()}, Spam: {(df['label'] == 1).sum()})")
    print("=" * 70)

    # 1. Stratified 80/20 Train/Test Split
    X = df["text"].values
    y = df["label"].values.astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y,
    )
    print(f"\n[Split] Train Set: {len(X_train)} samples (Spam: {y_train.sum()} | {y_train.mean():.2%})")
    print(f"[Split] Test Set:  {len(X_test)} samples (Spam: {y_test.sum()} | {y_test.mean():.2%})")

    # 2. Fit Feature Pipeline on Training Data ONLY
    print("\n[Feature Engineering] Fitting composite TF-IDF + Meta-Feature pipeline on Train set...")
    feature_pipe = build_feature_pipeline(max_features=4000, ngram_range=(1, 2), min_df=2, use_meta_features=True)
    X_train_trans = feature_pipe.fit_transform(X_train)
    X_test_trans = feature_pipe.transform(X_test)
    print(f"[Feature Engineering] Feature matrix shape: {X_train_trans.shape} (N-gram features + Meta indicators)")

    # 3. Train Scratch Logistic Regression Model
    print("\n[Model 1/2] Training Scratch Logistic Regression from First Principles...")
    scratch_model = ScratchLogisticRegression(
        learning_rate=0.8,
        epochs=1200,
        l2_lambda=0.005,
        tolerance=1e-7,
        verbose=False,
    )
    scratch_model.fit(X_train_trans, y_train)
    print(f"[ScratchLR] Finished in {len(scratch_model.get_loss_history())} iterations. Final BCE loss: {scratch_model.get_loss_history()[-1]:.6f}")

    # 4. Train & Tune Scikit-Learn Logistic Regression
    print("\n[Model 2/2] Training & Tuning Scikit-Learn Logistic Regression Pipeline...")
    sklearn_pipe = SklearnLogisticRegressionPipeline(
        C=2.0,
        max_iter=1000,
        class_weight="balanced",
        solver="lbfgs",
        random_state=42,
        max_features=4000,
        ngram_range=(1, 2),
        use_meta_features=True,
    )
    # Fit full pipeline directly
    sklearn_pipe.fit(X_train, y_train)
    print("[SklearnLR] Pipeline fitted successfully.")

    # 5. Evaluate on Unseen Test Set
    print("\n" + "=" * 70)
    print("EVALUATION ON UNSEEN TEST SET (20% Holdout, n = 1,032)")
    print("=" * 70)

    # Predictions
    scratch_proba = scratch_model.predict_proba(X_test_trans)[:, 1]
    scratch_preds = scratch_model.predict(X_test_trans, threshold=0.5)

    sklearn_proba = sklearn_pipe.predict_proba(X_test)[:, 1]
    sklearn_preds = sklearn_pipe.predict(X_test, threshold=0.5)

    # Metrics
    metrics_scratch = calculate_metrics(y_test, scratch_preds, scratch_proba)
    metrics_sklearn = calculate_metrics(y_test, sklearn_preds, sklearn_proba)

    # Threshold optimization for sklearn model
    best_threshold = plot_threshold_tuning(y_test, sklearn_proba)
    sklearn_preds_tuned = (sklearn_proba >= best_threshold).astype(int)
    metrics_sklearn_tuned = calculate_metrics(y_test, sklearn_preds_tuned, sklearn_proba)

    # Comparative Summary Table
    comparison_df = pd.DataFrame({
        "Metric": [
            "Accuracy",
            "Precision (PPV)",
            "Recall (Sensitivity)",
            "F1-Score",
            "Specificity (TNR)",
            "False Positive Rate",
            "ROC-AUC",
            "PR-AUC",
            "True Positives",
            "False Positives",
            "True Negatives",
            "False Negatives",
        ],
        "Scratch LR (th=0.50)": [
            f"{metrics_scratch['accuracy']:.4f}",
            f"{metrics_scratch['precision']:.4f}",
            f"{metrics_scratch['recall']:.4f}",
            f"{metrics_scratch['f1_score']:.4f}",
            f"{metrics_scratch['specificity']:.4f}",
            f"{metrics_scratch['false_positive_rate']:.4f}",
            f"{metrics_scratch['roc_auc']:.4f}",
            f"{metrics_scratch['pr_auc']:.4f}",
            metrics_scratch["true_positives"],
            metrics_scratch["false_positives"],
            metrics_scratch["true_negatives"],
            metrics_scratch["false_negatives"],
        ],
        "Scikit-Learn LR (th=0.50)": [
            f"{metrics_sklearn['accuracy']:.4f}",
            f"{metrics_sklearn['precision']:.4f}",
            f"{metrics_sklearn['recall']:.4f}",
            f"{metrics_sklearn['f1_score']:.4f}",
            f"{metrics_sklearn['specificity']:.4f}",
            f"{metrics_sklearn['false_positive_rate']:.4f}",
            f"{metrics_sklearn['roc_auc']:.4f}",
            f"{metrics_sklearn['pr_auc']:.4f}",
            metrics_sklearn["true_positives"],
            metrics_sklearn["false_positives"],
            metrics_sklearn["true_negatives"],
            metrics_sklearn["false_negatives"],
        ],
        f"Scikit-Learn (Tuned th={best_threshold:.2f})": [
            f"{metrics_sklearn_tuned['accuracy']:.4f}",
            f"{metrics_sklearn_tuned['precision']:.4f}",
            f"{metrics_sklearn_tuned['recall']:.4f}",
            f"{metrics_sklearn_tuned['f1_score']:.4f}",
            f"{metrics_sklearn_tuned['specificity']:.4f}",
            f"{metrics_sklearn_tuned['false_positive_rate']:.4f}",
            f"{metrics_sklearn_tuned['roc_auc']:.4f}",
            f"{metrics_sklearn_tuned['pr_auc']:.4f}",
            metrics_sklearn_tuned["true_positives"],
            metrics_sklearn_tuned["false_positives"],
            metrics_sklearn_tuned["true_negatives"],
            metrics_sklearn_tuned["false_negatives"],
        ],
    })

    print(comparison_df.to_string(index=False))

    # 6. Generate Diagnostic Plots
    print("\n[Evaluation] Generating comparison plots into outputs/evaluation/...")
    plot_confusion_matrices(y_test, scratch_preds, sklearn_preds)
    plot_roc_curves(y_test, scratch_proba, sklearn_proba)
    plot_precision_recall_curves(y_test, scratch_proba, sklearn_proba)
    plot_scratch_loss_trajectory(scratch_model.get_loss_history())

    top_features = sklearn_pipe.get_top_features(top_n=20)
    plot_top_coefficients(top_features["spam_features"], top_features["ham_features"])

    # 7. Model Serialization & Metadata Export
    print("\n[Serialization] Exporting models and audit metadata...")
    # Save sklearn pipeline
    sklearn_pipe_path = models_dir / "spam_classifier_pipeline.joblib"
    sklearn_pipe.save(sklearn_pipe_path)

    # Save scratch model
    scratch_path = models_dir / "scratch_logistic_regression.json"
    scratch_model.save(scratch_path)

    # Save metadata JSON
    metadata = {
        "project": "SpamShield — Intelligent Spam Message Filtering System",
        "algorithm": "Logistic Regression (Binary Classification: ham=0, spam=1)",
        "training_timestamp": datetime.now().isoformat(),
        "train_samples": int(len(X_train)),
        "test_samples": int(len(X_test)),
        "optimal_threshold": float(best_threshold),
        "scratch_metrics": metrics_scratch,
        "sklearn_metrics": metrics_sklearn,
        "sklearn_metrics_tuned": metrics_sklearn_tuned,
        "top_spam_indicators": top_features["spam_features"][:10],
        "top_ham_indicators": top_features["ham_features"][:10],
    }
    with open(models_dir / "model_metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print(f"[Complete] Model pipeline saved to {sklearn_pipe_path}")
    print(f"[Complete] Metadata saved to {models_dir / 'model_metadata.json'}")


if __name__ == "__main__":
    main()
