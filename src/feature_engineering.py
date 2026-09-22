"""Feature engineering pipeline for SpamShield.

Combines N-gram TF-IDF vectorization with scaled syntactic and behavioral meta-features
via a unified scikit-learn compatible ColumnTransformer / Pipeline.
"""

from __future__ import annotations
import numpy as np
import scipy.sparse as sp
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.preprocessing import StandardScaler

from src.preprocessor import (
    MetaFeatureExtractor,
    TextCleanerTransformer,
    clean_and_normalize_text,
)


class TextSelector(BaseEstimator, TransformerMixin):
    """Transformer that passes raw text as an array or Series."""

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        if hasattr(X, "values"):
            return X.values
        return np.asarray(X)


def build_tfidf_vectorizer(
    max_features: int = 4000,
    ngram_range: tuple[int, int] = (1, 2),
    min_df: int = 2,
    sublinear_tf: bool = True,
) -> TfidfVectorizer:
    """Instantiate a tuned TF-IDF vectorizer for text classification."""
    return TfidfVectorizer(
        ngram_range=ngram_range,
        max_features=max_features,
        min_df=min_df,
        sublinear_tf=sublinear_tf,
        stop_words="english",
        token_pattern=r"(?u)\b\w+\b|__\w+__",  # Ensures __url__, __phone__, etc. are captured as distinct tokens
    )


def build_feature_pipeline(
    max_features: int = 4000,
    ngram_range: tuple[int, int] = (1, 2),
    min_df: int = 2,
    use_meta_features: bool = True,
) -> Pipeline:
    """Build a complete feature extraction pipeline.
    
    If use_meta_features is True, returns a FeatureUnion combining:
    1. Text cleaner -> TF-IDF vectorizer (sparse)
    2. Meta-feature extractor -> StandardScaler (dense/sparse)
    """
    if not use_meta_features:
        return Pipeline([
            ("cleaner", TextCleanerTransformer(preserve_signals=True)),
            ("tfidf", build_tfidf_vectorizer(max_features=max_features, ngram_range=ngram_range, min_df=min_df)),
        ])

    tfidf_branch = Pipeline([
        ("cleaner", TextCleanerTransformer(preserve_signals=True)),
        ("tfidf", build_tfidf_vectorizer(max_features=max_features, ngram_range=ngram_range, min_df=min_df)),
    ])

    meta_branch = Pipeline([
        ("extractor", MetaFeatureExtractor()),
        ("scaler", StandardScaler(with_mean=False)),  # with_mean=False keeps sparse compatibility
    ])

    union = FeatureUnion(
        transformer_list=[
            ("tfidf_features", tfidf_branch),
            ("meta_features", meta_branch),
        ]
    )

    return Pipeline([("features", union)])


def get_all_feature_names(feature_pipeline: Pipeline) -> list[str]:
    """Extract ordered feature names from the fitted feature pipeline."""
    try:
        # Check if pipeline has 'features' (FeatureUnion)
        if "features" in feature_pipeline.named_steps:
            union = feature_pipeline.named_steps["features"]
            tfidf_step = union.transformer_list[0][1].named_steps["tfidf"]
            meta_step = union.transformer_list[1][1].named_steps["extractor"]

            tfidf_names = list(tfidf_step.get_feature_names_out())
            meta_names = [f"[meta]_{name}" for name in meta_step.feature_names_]
            return tfidf_names + meta_names
        elif "tfidf" in feature_pipeline.named_steps:
            return list(feature_pipeline.named_steps["tfidf"].get_feature_names_out())
    except Exception as e:
        print(f"Warning extracting feature names: {e}")
    return []
