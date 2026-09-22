"""Production-grade Scikit-learn Logistic Regression Pipeline for SpamShield.

Combines text cleaning, N-gram TF-IDF vectorization, syntactic meta-feature extraction,
and L2-regularized Logistic Regression with cross-validated hyperparameter optimization.
"""

from __future__ import annotations
from pathlib import Path
from typing import Any
import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline

from src.feature_engineering import build_feature_pipeline, get_all_feature_names


class SklearnLogisticRegressionPipeline:
    """End-to-end NLP classification pipeline using Scikit-Learn Logistic Regression."""

    def __init__(
        self,
        C: float = 2.0,
        max_iter: int = 1000,
        class_weight: str | dict | None = "balanced",
        solver: str = "lbfgs",
        random_state: int = 42,
        max_features: int = 4000,
        ngram_range: tuple[int, int] = (1, 2),
        use_meta_features: bool = True,
    ):
        self.C = C
        self.max_iter = max_iter
        self.class_weight = class_weight
        self.solver = solver
        self.random_state = random_state
        self.max_features = max_features
        self.ngram_range = ngram_range
        self.use_meta_features = use_meta_features

        self.pipeline: Pipeline | None = None
        self.is_fitted: bool = False
        self.best_params_: dict[str, Any] = {}

    def _build_pipeline(self, C: float | None = None, class_weight: Any = None) -> Pipeline:
        feature_pipe = build_feature_pipeline(
            max_features=self.max_features,
            ngram_range=self.ngram_range,
            min_df=2,
            use_meta_features=self.use_meta_features,
        )
        classifier = LogisticRegression(
            C=C if C is not None else self.C,
            max_iter=self.max_iter,
            class_weight=class_weight if class_weight is not None else self.class_weight,
            solver=self.solver,
            random_state=self.random_state,
        )
        return Pipeline([
            ("feature_engineering", feature_pipe),
            ("classifier", classifier),
        ])

    def fit(self, X_train: pd.Series | list[str] | np.ndarray, y_train: pd.Series | np.ndarray) -> SklearnLogisticRegressionPipeline:
        """Fit the full pipeline directly on raw text messages."""
        self.pipeline = self._build_pipeline()
        self.pipeline.fit(X_train, y_train)
        self.is_fitted = True
        return self

    def tune_and_fit(
        self,
        X_train: pd.Series | list[str] | np.ndarray,
        y_train: pd.Series | np.ndarray,
        param_grid: dict[str, list[Any]] | None = None,
        cv: int = 5,
        scoring: str = "f1",
    ) -> SklearnLogisticRegressionPipeline:
        """Tune hyperparameters using Stratified 5-Fold Cross-Validation and refit."""
        if param_grid is None:
            param_grid = {
                "classifier__C": [0.5, 1.0, 2.0, 5.0, 10.0],
                "classifier__class_weight": [None, "balanced"],
            }

        base_pipe = self._build_pipeline()
        cv_strategy = StratifiedKFold(n_splits=cv, shuffle=True, random_state=self.random_state)

        grid = GridSearchCV(
            estimator=base_pipe,
            param_grid=param_grid,
            cv=cv_strategy,
            scoring=scoring,
            n_jobs=-1,
            refit=True,
            verbose=1,
        )
        print(f"[SklearnLR] Starting Grid Search across {len(param_grid.get('classifier__C', [])) * len(param_grid.get('classifier__class_weight', []))} candidates...")
        grid.fit(X_train, y_train)

        self.pipeline = grid.best_estimator_
        self.best_params_ = grid.best_params_
        self.is_fitted = True

        print(f"[SklearnLR] Best parameters: {self.best_params_} (Best CV F1: {grid.best_score_:.4f})")
        return self

    def predict_proba(self, X: pd.Series | list[str] | np.ndarray) -> np.ndarray:
        """Compute posterior class probabilities."""
        if not self.is_fitted or self.pipeline is None:
            raise RuntimeError("Pipeline is not fitted yet.")
        return self.pipeline.predict_proba(X)

    def predict(self, X: pd.Series | list[str] | np.ndarray, threshold: float = 0.5) -> np.ndarray:
        """Predict binary class label given threshold."""
        probabilities = self.predict_proba(X)[:, 1]
        return (probabilities >= threshold).astype(int)

    def get_top_features(self, top_n: int = 25) -> dict[str, list[tuple[str, float]]]:
        """Extract top positive (spam-predictive) and top negative (ham-predictive) features.
        
        Returns:
            dict with 'spam_features' and 'ham_features' pairs (feature_name, coefficient).
        """
        if not self.is_fitted or self.pipeline is None:
            raise RuntimeError("Pipeline is not fitted yet.")

        feature_pipe = self.pipeline.named_steps["feature_engineering"]
        classifier = self.pipeline.named_steps["classifier"]

        feature_names = get_all_feature_names(feature_pipe)
        coefs = classifier.coef_[0]

        if len(feature_names) != len(coefs):
            # Fallback if mismatch
            return {"spam_features": [], "ham_features": []}

        pairs = list(zip(feature_names, coefs))
        sorted_pairs = sorted(pairs, key=lambda x: x[1], reverse=True)

        top_spam = [(name, float(val)) for name, val in sorted_pairs[:top_n]]
        top_ham = [(name, float(val)) for name, val in sorted_pairs[-top_n:]]

        return {
            "spam_features": top_spam,
            "ham_features": top_ham,
        }

    def save(self, filepath: str | Path) -> None:
        """Serialize fitted pipeline to disk."""
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.pipeline, path)
        print(f"[SklearnLR] Successfully saved model pipeline to {path}")

    @classmethod
    def load(cls, filepath: str | Path) -> SklearnLogisticRegressionPipeline:
        """Load serialized pipeline from disk."""
        path = Path(filepath)
        loaded_pipe = joblib.load(path)
        instance = cls()
        instance.pipeline = loaded_pipe
        instance.is_fitted = True
        return instance
