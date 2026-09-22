"""Unit tests for Scratch Logistic Regression model."""

from pathlib import Path
import tempfile
import numpy as np
import pytest
from src.models.scratch_logistic_regression import ScratchLogisticRegression


def test_scratch_sigmoid_limits():
    z_large_pos = np.array([1000.0])
    z_large_neg = np.array([-1000.0])
    sig_pos = ScratchLogisticRegression._sigmoid(z_large_pos)
    sig_neg = ScratchLogisticRegression._sigmoid(z_large_neg)

    assert np.isclose(sig_pos[0], 1.0)
    assert np.isclose(sig_neg[0], 0.0)


def test_scratch_fit_and_convergence():
    np.random.seed(42)
    # Linearly separable 2D data
    X = np.vstack([
        np.random.randn(50, 2) + np.array([2.0, 2.0]),
        np.random.randn(50, 2) - np.array([2.0, 2.0]),
    ])
    y = np.array([1] * 50 + [0] * 50)

    model = ScratchLogisticRegression(learning_rate=0.5, epochs=200, l2_lambda=0.01)
    model.fit(X, y)

    assert model.is_fitted
    assert len(model.loss_history) > 0
    # Loss should decrease
    assert model.loss_history[-1] < model.loss_history[0]

    preds = model.predict(X)
    accuracy = np.mean(preds == y)
    assert accuracy >= 0.95


def test_scratch_save_and_load():
    np.random.seed(42)
    X = np.random.randn(30, 3)
    y = (X[:, 0] > 0).astype(int)

    model = ScratchLogisticRegression(learning_rate=0.1, epochs=50)
    model.fit(X, y)

    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_file = Path(tmp_dir) / "scratch_test.json"
        model.save(tmp_file)

        loaded_model = ScratchLogisticRegression.load(tmp_file)
        assert loaded_model.is_fitted
        np.testing.assert_allclose(model.weights, loaded_model.weights)
        assert np.isclose(model.bias, loaded_model.bias)

        orig_preds = model.predict(X)
        loaded_preds = loaded_model.predict(X)
        np.testing.assert_array_equal(orig_preds, loaded_preds)
