"""Exploratory Data Analysis (EDA) module for SpamShield.

Generates academic-quality, publication-ready visualizations analyzing the
empirical distributions, linguistic patterns, and discriminative markers
of the UCI SMS Spam Collection dataset. All figures are saved to outputs/eda/.
"""

from __future__ import annotations
from collections import Counter
import os
from pathlib import Path
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from wordcloud import WordCloud

from src.preprocessor import clean_and_normalize_text, extract_meta_features

BASE_DIR = Path(__file__).resolve().parent.parent
EDA_OUT_DIR = BASE_DIR / "outputs" / "eda"

# Set academic aesthetic styling
plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
PALETTE = {"Ham (0)": "#2b6cb0", "Spam (1)": "#e53e3e"}
COLORS = ["#2b6cb0", "#e53e3e"]


def run_full_eda(df: pd.DataFrame, out_dir: Path | None = None) -> dict:
    """Execute complete EDA pipeline and save charts.
    
    Expects df with columns ['label', 'text'].
    Returns summary statistics dictionary.
    """
    if out_dir is None:
        out_dir = EDA_OUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    print("[EDA] Computing metadata and linguistic features...")
    # Extract meta-features for all rows
    meta_records = [extract_meta_features(str(t)) for t in df["text"]]
    meta_df = pd.DataFrame(meta_records)
    analysis_df = pd.concat([df.reset_index(drop=True), meta_df], axis=1)
    analysis_df["class_name"] = analysis_df["label"].map({0: "Ham (0)", 1: "Spam (1)"})

    stats = {
        "total_messages": len(df),
        "ham_count": int((df["label"] == 0).sum()),
        "spam_count": int((df["label"] == 1).sum()),
        "spam_percentage": float(df["label"].mean() * 100),
        "ham_mean_length": float(analysis_df[analysis_df["label"] == 0]["char_count"].mean()),
        "spam_mean_length": float(analysis_df[analysis_df["label"] == 1]["char_count"].mean()),
        "ham_mean_words": float(analysis_df[analysis_df["label"] == 0]["word_count"].mean()),
        "spam_mean_words": float(analysis_df[analysis_df["label"] == 1]["word_count"].mean()),
    }

    # 1. Class Distribution (Bar + Donut)
    _plot_class_distribution(analysis_df, out_dir / "01_class_distribution.png")

    # 2. Message Length Distribution (Chars)
    _plot_length_distribution(analysis_df, out_dir / "02_message_length_distribution.png")

    # 3. Word Count Distribution
    _plot_word_count_distribution(analysis_df, out_dir / "03_word_count_distribution.png")

    # 4. Top Spam Words
    _plot_top_words(analysis_df[analysis_df["label"] == 1]["text"], "Spam (1)", "#c53030", out_dir / "04_top_spam_words.png")

    # 5. Top Ham Words
    _plot_top_words(analysis_df[analysis_df["label"] == 0]["text"], "Ham (0)", "#2b6cb0", out_dir / "05_top_ham_words.png")

    # 6. Meta Features Comparison (Boxplots)
    _plot_meta_features(analysis_df, out_dir / "06_meta_features_comparison.png")

    # 7. WordClouds
    _plot_wordclouds(analysis_df, out_dir / "07_wordclouds_spam_vs_ham.png")

    # 8. Feature Correlation Heatmap
    _plot_correlation_heatmap(analysis_df, out_dir / "08_feature_correlation_heatmap.png")

    print(f"[EDA] All 8 publication-ready charts successfully saved to {out_dir}")
    return stats


def _plot_class_distribution(df: pd.DataFrame, out_path: Path):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), dpi=300)
    counts = df["class_name"].value_counts()

    # Bar plot
    sns.barplot(x=counts.index, y=counts.values, ax=axes[0], hue=counts.index, palette=["#2b6cb0", "#e53e3e"], legend=False)
    axes[0].set_title("Class Frequency Distribution", fontsize=13, fontweight="bold", pad=12)
    axes[0].set_ylabel("Message Count", fontsize=11)
    axes[0].set_xlabel("Class", fontsize=11)
    for i, v in enumerate(counts.values):
        pct = (v / len(df)) * 100
        axes[0].text(i, v + 40, f"{v:,}\n({pct:.1f}%)", ha="center", fontsize=10, fontweight="semibold")
    axes[0].set_ylim(0, max(counts.values) * 1.15)

    # Donut plot
    axes[1].pie(
        counts.values,
        labels=counts.index,
        autopct="%1.1f%%",
        startangle=140,
        colors=["#2b6cb0", "#e53e3e"],
        wedgeprops={"edgecolor": "white", "linewidth": 2, "width": 0.5},
        textprops={"fontsize": 11, "fontweight": "semibold"},
    )
    axes[1].set_title("Class Imbalance Ratio", fontsize=13, fontweight="bold", pad=12)

    plt.tight_layout()
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()


def _plot_length_distribution(df: pd.DataFrame, out_path: Path):
    plt.figure(figsize=(10, 5), dpi=300)
    sns.histplot(
        data=df[df["char_count"] <= 350],
        x="char_count",
        hue="class_name",
        palette=PALETTE,
        kde=True,
        bins=50,
        element="step",
        common_norm=False,
        alpha=0.4,
    )
    plt.axvline(160, color="#d69e2e", linestyle="--", linewidth=1.5, label="Standard SMS 160-Char Limit")
    plt.title("Character Length Distribution: Ham vs Spam", fontsize=13, fontweight="bold", pad=12)
    plt.xlabel("Character Count (Truncated at 350 for visualization)", fontsize=11)
    plt.ylabel("Density / Count", fontsize=11)
    plt.legend(title="Class", frameon=True)
    plt.tight_layout()
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()


def _plot_word_count_distribution(df: pd.DataFrame, out_path: Path):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), dpi=300)

    # Boxplot
    sns.boxplot(data=df, x="class_name", y="word_count", ax=axes[0], hue="class_name", palette=PALETTE, width=0.4, showfliers=False, legend=False)
    axes[0].set_title("Word Count Boxplot (Excl. Outliers)", fontsize=13, fontweight="bold")
    axes[0].set_ylabel("Word Count", fontsize=11)
    axes[0].set_xlabel("Class", fontsize=11)

    # KDE plot
    sns.kdeplot(
        data=df[df["word_count"] <= 60],
        x="word_count",
        hue="class_name",
        palette=PALETTE,
        ax=axes[1],
        fill=True,
        common_norm=False,
        alpha=0.3,
    )
    axes[1].set_title("Word Count Kernel Density Estimate", fontsize=13, fontweight="bold")
    axes[1].set_xlabel("Word Count", fontsize=11)

    plt.tight_layout()
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()


def _plot_top_words(text_series: pd.Series, class_title: str, bar_color: str, out_path: Path):
    plt.figure(figsize=(10, 6), dpi=300)
    stop_words = {
        "to", "you", "a", "i", "the", "u", "and", "is", "in", "me", "my", "it",
        "for", "your", "of", "that", "on", "have", "are", "at", "with", "so",
        "be", "or", "can", "if", "not", "do", "we", "this", "from", "will", "get", "go",
    }
    all_tokens = []
    for text in text_series:
        cleaned = clean_and_normalize_text(text, preserve_signals=True)
        tokens = [w for w in cleaned.split() if w not in stop_words and len(w) > 1]
        all_tokens.extend(tokens)

    counter = Counter(all_tokens)
    top_20 = counter.most_common(20)
    words, counts = zip(*reversed(top_20))

    y_pos = np.arange(len(words))
    plt.barh(y_pos, counts, color=bar_color, edgecolor="none", alpha=0.85)
    plt.yticks(y_pos, words, fontsize=10)
    plt.xlabel("Occurrence Frequency", fontsize=11)
    plt.title(f"Top 20 Distinctive Tokens in {class_title} Messages", fontsize=13, fontweight="bold", pad=12)

    for i, v in enumerate(counts):
        plt.text(v + (max(counts) * 0.01), i, f" {v:,}", va="center", fontsize=9, color="#2d3748")

    plt.xlim(0, max(counts) * 1.12)
    plt.tight_layout()
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()


def _plot_meta_features(df: pd.DataFrame, out_path: Path):
    fig, axes = axes_grid = plt.subplots(2, 2, figsize=(12, 9), dpi=300)
    features = [
        ("uppercase_ratio", "Uppercase Character Ratio", axes[0, 0]),
        ("digit_ratio", "Digit Character Ratio", axes[0, 1]),
        ("currency_count", "Currency Symbol Count", axes[1, 0]),
        ("exclamation_count", "Exclamation Mark Count", axes[1, 1]),
    ]
    for feat, title, ax in features:
        sns.barplot(data=df, x="class_name", y=feat, ax=ax, hue="class_name", palette=PALETTE, capsize=0.1, err_kws={"linewidth": 1.2}, legend=False)
        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.set_ylabel(f"Mean {feat}", fontsize=10)
        ax.set_xlabel("")

    plt.suptitle("Syntactic & Structural Spam Indicators Comparison", fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()


def _plot_wordclouds(df: pd.DataFrame, out_path: Path):
    fig, axes = plt.subplots(1, 2, figsize=(16, 7), dpi=300)

    # Ham WordCloud
    ham_text = " ".join(df[df["label"] == 0]["text"])
    wc_ham = WordCloud(
        width=800,
        height=400,
        background_color="white",
        colormap="Blues",
        max_words=100,
        random_state=42,
    ).generate(ham_text)
    axes[0].imshow(wc_ham, interpolation="bilinear")
    axes[0].axis("off")
    axes[0].set_title("Ham WordCloud (Legitimate Messages)", fontsize=14, fontweight="bold", pad=12)

    # Spam WordCloud
    spam_text = " ".join(df[df["label"] == 1]["text"])
    wc_spam = WordCloud(
        width=800,
        height=400,
        background_color="white",
        colormap="Reds",
        max_words=100,
        random_state=42,
    ).generate(spam_text)
    axes[1].imshow(wc_spam, interpolation="bilinear")
    axes[1].axis("off")
    axes[1].set_title("Spam WordCloud (Unsolicited Messages)", fontsize=14, fontweight="bold", pad=12)

    plt.tight_layout()
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()


def _plot_correlation_heatmap(df: pd.DataFrame, out_path: Path):
    plt.figure(figsize=(10, 8), dpi=300)
    cols = [
        "label",
        "char_count",
        "word_count",
        "uppercase_ratio",
        "digit_ratio",
        "currency_count",
        "url_count",
        "phone_count",
        "exclamation_count",
        "has_urgent_keyword",
    ]
    corr = df[cols].corr(method="spearman")
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(
        corr,
        mask=mask,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        center=0,
        square=True,
        linewidths=0.5,
        cbar_kws={"shrink": 0.8},
    )
    plt.title("Spearman Correlation Matrix of Engineered Meta-Features", fontsize=13, fontweight="bold", pad=12)
    plt.tight_layout()
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()
