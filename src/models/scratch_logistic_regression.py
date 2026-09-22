"""Logistic Regression implemented from first principles (scratch).

Includes:
- Vectorized computation compatible with both dense and scipy sparse matrices
- Sigmoid activation with numerical overflow safeguards
- Binary Cross-Entropy (Log-Loss) with L2 Ridge regularization
- Vectorized analytic gradient computation
- Batch Gradient Descent with loss trajectory tracking & early stopping
- Full probability calibration and adjustable classification thresholding
"""

from __future__ import annotations
import json
from pathlib import Path
from typing import Any
import numpy as np
import scipy.sparse as sp


class ScratchLogisticRegression:
    """Binary Logistic Regression classifier built from mathematical first principles.
    
    Attributes:
        learning_rate (float): Step size for gradient descent optimization (alpha).
        epochs (int): Maximum number of gradient descent iterations.
        l2_lambda (float): L2 regularization hyperparameter (weight decay penalty).
        tolerance (float): Convergence threshold on absolute loss decrease.
        weights (np.ndarray): Learned parameter coefficients (w).
        bias (float): Learned intercept/bias term (b).
        loss_history (list[float]): Trajectory of cost function values over iterations.
    """

    def __init__(
        self,
        learning_rate: float = 0.5,
        epochs: int = 1000,
        l2_lambda: float = 0.01,
        tolerance: float = 1e-6,
        verbose: bool = False,
    ):
        self.learning_rate = float(learning_rate)
        self.epochs = int(epochs)
        self.l2_lambda = float(l2_lambda)
        self.tolerance = float(tolerance)
        self.verbose = bool(verbose)

        self.weights: np.ndarray | None = None
        self.bias: float = 0.0
        self.loss_history: list[float] = []
        self.is_fitted: bool = False
        self.n_features_: int = 0

    @staticmethod
    def _sigmoid(z: np.ndarray) -> np.ndarray:
        """Numerically stable Sigmoid activation function.
        
        Computes sigma(z) = 1 / (1 + exp(-z)).
        Clips z into [-500, 500] to prevent floating point overflow in exp(-z).
        """
        z_clipped = np.clip(z, -500.0, 500.0)
        return 1.0 / (1.0 + np.exp(-z_clipped))

    def _compute_loss(self, y_true: np.ndarray, y_pred_prob: np.ndarray) -> float:
        """Compute Binary Cross-Entropy loss with L2 regularization penalty.
        
        Cost J(w, b) = -1/m * sum(y*ln(y_hat) + (1-y)*ln(1-y_hat)) + (lambda / (2*m)) * ||w||^2
        """
        m = len(y_true)
        eps = 1e-15  # Avoid log(0)
        y_prob_safe = np.clip(y_pred_prob, eps, 1.0 - eps)

        bce_loss = - (1.0 / m) * np.sum(
            y_true * np.log(y_prob_safe) + (1.0 - y_true) * np.log(1.0 - y_prob_safe)
        )
        l2_penalty = (self.l2_lambda / (2.0 * m)) * np.sum(self.weights ** 2) if self.weights is not None else 0.0

        return float(bce_loss + l2_penalty)

    def fit(self, X: np.ndarray | sp.spmatrix, y: np.ndarray | list[int]) -> ScratchLogisticRegression:
        """Train model parameters (weights and bias) using vectorized Batch Gradient Descent.
        
        Args:
            X: Feature matrix of shape (m_samples, n_features), either dense or sparse.
            y: Target binary labels (0 or 1) of shape (m_samples,).
        """
        m_samples = X.shape[0]
        n_features = X.shape[1]
        self.n_features_ = n_features

        y_vec = np.asarray(y, dtype=np.float64).reshape(-1)

        # Initialize weights with small Gaussian random values / zeros
        # Using zero initialization is standard and valid for convex logistic regression
        self.weights = np.zeros(n_features, dtype=np.float64)
        self.bias = 0.0
        self.loss_history = []

        is_sparse = sp.issparse(X)

        for epoch in range(1, self.epochs + 1):
            # Forward pass: Linear combination z = X*w + b
            if is_sparse:
                z = X.dot(self.weights) + self.bias
            else:
                z = np.dot(X, self.weights) + self.bias

            # Activation: probabilities
            y_hat = self._sigmoid(z)

            # Compute and record cost
            loss = self._compute_loss(y_vec, y_hat)
            self.loss_history.append(loss)

            # Check convergence
            if epoch > 1 and abs(self.loss_history[-2] - loss) < self.tolerance:
                if self.verbose:
                    print(f"[ScratchLR] Converged at epoch {epoch} with loss {loss:.6f}")
                break

            # Analytic Gradients:
            # error = y_hat - y
            # dJ/dw = 1/m * X^T * error + (lambda/m) * w
            # dJ/db = 1/m * sum(error)
            error = y_hat - y_vec
            if is_sparse:
                grad_w = (1.0 / m_samples) * X.T.dot(error) + (self.l2_lambda / m_samples) * self.weights
            else:
                grad_w = (1.0 / m_samples) * np.dot(X.T, error) + (self.l2_lambda / m_samples) * self.weights

            grad_b = (1.0 / m_samples) * np.sum(error)

            # Parameter update
            self.weights -= self.learning_rate * grad_w
            self.bias -= self.learning_rate * grad_b

            if self.verbose and (epoch % 100 == 0 or epoch == self.epochs):
                print(f"[ScratchLR] Epoch {epoch:4d}/{self.epochs:4d} - Loss: {loss:.6f}")

        self.is_fitted = True
        return self

    def predict_proba(self, X: np.ndarray | sp.spmatrix) -> np.ndarray:
        """Predict posterior class probabilities: P(Y=0|X) and P(Y=1|X).
        
        Returns:
            np.ndarray of shape (m_samples, 2) where col 0 is P(Ham) and col 1 is P(Spam).
        """
        if not self.is_fitted or self.weights is None:
            raise RuntimeError("Model is not fitted yet. Call fit() first.")

        if sp.issparse(X):
            z = X.dot(self.weights) + self.bias
        else:
            z = np.dot(X, self.weights) + self.bias

        p_spam = self._sigmoid(z)
        p_ham = 1.0 - p_spam
        return np.column_stack([p_ham, p_spam])

    def predict(self, X: np.ndarray | sp.spmatrix, threshold: float = 0.5) -> np.ndarray:
        """Predict binary discrete labels (0=Ham, 1=Spam) given decision threshold."""
        proba = self.predict_proba(X)[:, 1]
        return (proba >= threshold).astype(int)

    def get_loss_history(self) -> list[float]:
        """Return the loss trajectory recorded during gradient descent."""
        return self.loss_history

    def to_dict(self) -> dict[str, Any]:
        """Serialize model weights and hyperparameters to python dictionary."""
        return {
            "learning_rate": self.learning_rate,
            "epochs": self.epochs,
            "l2_lambda": self.l2_lambda,
            "tolerance": self.tolerance,
            "n_features": self.n_features_,
            "weights": self.weights.tolist() if self.weights is not None else [],
            "bias": float(self.bias),
            "final_loss": float(self.loss_history[-1]) if self.loss_history else None,
            "epochs_run": len(self.loss_history),
        }

    def save(self, filepath: str | Path) -> None:
        """Save model configuration and weights to a JSON file."""
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, filepath: str | Path) -> ScratchLogisticRegression:
        """Load model configuration and weights from a JSON file."""
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)

        model = cls(
            learning_rate=data["learning_rate"],
            epochs=data["epochs"],
            l2_lambda=data["l2_lambda"],
            tolerance=data["tolerance"],
        )
        model.n_features_ = data["n_features"]
        model.weights = np.array(data["weights"], dtype=np.float64)
        model.bias = float(data["bias"])
        model.is_fitted = True
        return model
