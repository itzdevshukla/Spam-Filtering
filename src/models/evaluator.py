"""Comprehensive evaluation, visualization, and threshold tuning module for SpamShield.

Computes academic metrics (Accuracy, Precision, Recall, F1, Specificity, FPR, ROC-AUC, PR-AUC),
plots high-resolution comparison curves, and generates diagnostic threshold tuning plots.
"""

from __future__ import annotations
from pathlib import Path
from typing import Any
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

BASE_DIR = Path(__file__).resolve().parent.parent.parent
EVAL_OUT_DIR = BASE_DIR / "outputs" / "evaluation"


def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_proba: np.ndarray) -> dict[str, float]:
    """Calculate complete suite of binary classification metrics."""
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()

    accuracy = float(accuracy_score(y_true, y_pred))
    precision = float(precision_score(y_true, y_pred, zero_division=0))
    recall = float(recall_score(y_true, y_pred, zero_division=0))
    f1 = float(f1_score(y_true, y_pred, zero_division=0))
    specificity = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0
    fpr = float(fp / (fp + tn)) if (fp + tn) > 0 else 0.0
    roc_auc = float(roc_auc_score(y_true, y_proba)) if len(np.unique(y_true)) > 1 else 0.0
    pr_auc = float(average_precision_score(y_true, y_proba)) if len(np.unique(y_true)) > 1 else 0.0

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "specificity": specificity,
        "false_positive_rate": fpr,
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
        "true_positives": int(tp),
        "false_positives": int(fp),
        "true_negatives": int(tn),
        "false_negatives": int(fn),
    }


def plot_confusion_matrices(
    y_true: np.ndarray,
    scratch_preds: np.ndarray,
    sklearn_preds: np.ndarray,
    out_path: Path | None = None,
):
    """Plot side-by-side heatmaps of confusion matrices for Scratch vs Sklearn."""
    if out_path is None:
        out_path = EVAL_OUT_DIR / "01_confusion_matrices.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5), dpi=300)
    labels = ["Ham (0)", "Spam (1)"]

    cm_scratch = confusion_matrix(y_true, scratch_preds)
    cm_sklearn = confusion_matrix(y_true, sklearn_preds)

    sns.heatmap(
        cm_scratch,
        annot=True,
        fmt="d",
        cmap="Blues",
        ax=axes[0],
        xticklabels=labels,
        yticklabels=labels,
        cbar=False,
        annot_kws={"size": 14, "weight": "bold"},
    )
    axes[0].set_title("Scratch Logistic Regression\nConfusion Matrix", fontsize=13, fontweight="bold", pad=10)
    axes[0].set_xlabel("Predicted Class", fontsize=11)
    axes[0].set_ylabel("True Ground-Truth Class", fontsize=11)

    sns.heatmap(
        cm_sklearn,
        annot=True,
        fmt="d",
        cmap="Greens",
        ax=axes[1],
        xticklabels=labels,
        yticklabels=labels,
        cbar=False,
        annot_kws={"size": 14, "weight": "bold"},
    )
    axes[1].set_title("Scikit-Learn Logistic Regression\nConfusion Matrix", fontsize=13, fontweight="bold", pad=10)
    axes[1].set_xlabel("Predicted Class", fontsize=11)
    axes[1].set_ylabel("True Ground-Truth Class", fontsize=11)

    plt.tight_layout()
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()


def plot_roc_curves(
    y_true: np.ndarray,
    scratch_proba: np.ndarray,
    sklearn_proba: np.ndarray,
    out_path: Path | None = None,
):
    """Plot overlay ROC curves with AUC scores for both models."""
    if out_path is None:
        out_path = EVAL_OUT_DIR / "02_roc_curves.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    fpr_scratch, tpr_scratch, _ = roc_curve(y_true, scratch_proba)
    auc_scratch = roc_auc_score(y_true, scratch_proba)

    fpr_sklearn, tpr_sklearn, _ = roc_curve(y_true, sklearn_proba)
    auc_sklearn = roc_auc_score(y_true, sklearn_proba)

    plt.figure(figsize=(8, 6), dpi=300)
    plt.plot(fpr_scratch, tpr_scratch, color="#2b6cb0", lw=2, label=f"Scratch Model (AUC = {auc_scratch:.4f})")
    plt.plot(fpr_sklearn, tpr_sklearn, color="#2f855a", lw=2, linestyle="--", label=f"Scikit-Learn Model (AUC = {auc_sklearn:.4f})")
    plt.plot([0, 1], [0, 1], color="#a0aec0", lw=1.5, linestyle=":", label="Chance (AUC = 0.5000)")

    plt.xlim([-0.01, 1.0])
    plt.ylim([0.0, 1.02])
    plt.xlabel("False Positive Rate (1 - Specificity)", fontsize=11)
    plt.ylabel("True Positive Rate (Recall / Sensitivity)", fontsize=11)
    plt.title("Receiver Operating Characteristic (ROC) Curve Comparison", fontsize=13, fontweight="bold", pad=12)
    plt.legend(loc="lower right", frameon=True, fontsize=10)
    plt.tight_layout()
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()


def plot_precision_recall_curves(
    y_true: np.ndarray,
    scratch_proba: np.ndarray,
    sklearn_proba: np.ndarray,
    out_path: Path | None = None,
):
    """Plot Precision-Recall curves (critical for imbalanced classification)."""
    if out_path is None:
        out_path = EVAL_OUT_DIR / "03_precision_recall_curves.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    prec_sc, rec_sc, _ = precision_recall_curve(y_true, scratch_proba)
    ap_sc = average_precision_score(y_true, scratch_proba)

    prec_sk, rec_sk, _ = precision_recall_curve(y_true, sklearn_proba)
    ap_sk = average_precision_score(y_true, sklearn_proba)

    plt.figure(figsize=(8, 6), dpi=300)
    plt.plot(rec_sc, prec_sc, color="#2b6cb0", lw=2, label=f"Scratch Model (PR-AUC = {ap_sc:.4f})")
    plt.plot(rec_sk, prec_sk, color="#2f855a", lw=2, linestyle="--", label=f"Scikit-Learn Model (PR-AUC = {ap_sk:.4f})")

    baseline = float(np.mean(y_true))
    plt.axhline(baseline, color="#e53e3e", linestyle=":", lw=1.5, label=f"Class Prevalence ({baseline:.2%})")

    plt.xlim([0.0, 1.02])
    plt.ylim([0.0, 1.02])
    plt.xlabel("Recall", fontsize=11)
    plt.ylabel("Precision", fontsize=11)
    plt.title("Precision-Recall Curve (Evaluation on Imbalanced SMS Data)", fontsize=13, fontweight="bold", pad=12)
    plt.legend(loc="lower left", frameon=True, fontsize=10)
    plt.tight_layout()
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()


def plot_threshold_tuning(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    out_path: Path | None = None,
) -> float:
    """Analyze Precision, Recall, and F1 across decision thresholds and return optimal threshold."""
    if out_path is None:
        out_path = EVAL_OUT_DIR / "04_threshold_tuning.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    thresholds = np.linspace(0.05, 0.95, 91)
    precisions = []
    recalls = []
    f1s = []

    for t in thresholds:
        preds = (y_proba >= t).astype(int)
        precisions.append(precision_score(y_true, preds, zero_division=0))
        recalls.append(recall_score(y_true, preds, zero_division=0))
        f1s.append(f1_score(y_true, preds, zero_division=0))

    best_idx = int(np.argmax(f1s))
    best_t = float(thresholds[best_idx])
    best_f1 = float(f1s[best_idx])

    plt.figure(figsize=(9, 5.5), dpi=300)
    plt.plot(thresholds, precisions, label="Precision", color="#3182ce", lw=2)
    plt.plot(thresholds, recalls, label="Recall", color="#e53e3e", lw=2)
    plt.plot(thresholds, f1s, label=f"F1-Score (Peak: {best_f1:.4f} at {best_t:.2f})", color="#38a169", lw=2.5)

    plt.axvline(best_t, color="#38a169", linestyle="--", alpha=0.7, label=f"Optimal F1 Threshold (th = {best_t:.2f})")
    plt.axvline(0.5, color="#718096", linestyle=":", alpha=0.7, label="Default Threshold (th = 0.50)")

    plt.title("Decision Threshold Optimization vs Evaluation Metrics", fontsize=13, fontweight="bold", pad=12)
    plt.xlabel("Classification Probability Threshold (τ)", fontsize=11)
    plt.ylabel("Metric Score", fontsize=11)
    plt.ylim([0.0, 1.05])
    plt.legend(frameon=True, fontsize=10)
    plt.tight_layout()
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()

    return best_t


def plot_scratch_loss_trajectory(loss_history: list[float], out_path: Path | None = None):
    """Plot iteration-by-iteration Binary Cross-Entropy loss curve for Scratch model."""
    if out_path is None:
        out_path = EVAL_OUT_DIR / "05_scratch_loss_trajectory.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(8, 5), dpi=300)
    plt.plot(loss_history, color="#805ad5", lw=2, label="BCE Cost with L2 Penalty")
    plt.title("Scratch Logistic Regression: Gradient Descent Convergence", fontsize=13, fontweight="bold", pad=12)
    plt.xlabel("Iteration / Epoch", fontsize=11)
    plt.ylabel("Objective Cost J(w, b)", fontsize=11)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend(frameon=True, fontsize=10)
    plt.tight_layout()
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()


def plot_top_coefficients(
    top_spam: list[tuple[str, float]],
    top_ham: list[tuple[str, float]],
    out_path: Path | None = None,
):
    """Plot horizontal bar charts of most influential Logistic Regression weights."""
    if out_path is None:
        out_path = EVAL_OUT_DIR / "06_top_coefficients.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6), dpi=300)

    # Top spam features (positive weights)
    spam_names, spam_weights = zip(*reversed(top_spam[:15]))
    y_pos = np.arange(len(spam_names))
    axes[0].barh(y_pos, spam_weights, color="#e53e3e", alpha=0.85)
    axes[0].set_yticks(y_pos)
    axes[0].set_yticklabels(spam_names, fontsize=10)
    axes[0].set_xlabel("Positive Coefficient (Log-Odds Impact for Spam)", fontsize=11)
    axes[0].set_title("Top 15 Spam Indicators (w > 0)", fontsize=12, fontweight="bold")

    # Top ham features (negative weights)
    ham_names, ham_weights = zip(*top_ham[:15])
    y_pos_ham = np.arange(len(ham_names))
    axes[1].barh(y_pos_ham, [abs(w) for w in ham_weights], color="#2b6cb0", alpha=0.85)
    axes[1].set_yticks(y_pos_ham)
    axes[1].set_yticklabels(ham_names, fontsize=10)
    axes[1].set_xlabel("|Negative Coefficient| (Log-Odds Impact for Ham)", fontsize=11)
    axes[1].set_title("Top 15 Legitimate/Ham Indicators (w < 0)", fontsize=12, fontweight="bold")

    plt.suptitle("Logistic Regression Interpretability: Feature Importance via Log-Odds Weights", fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()
