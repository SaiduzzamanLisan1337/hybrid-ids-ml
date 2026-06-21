"""
models/autoencoder_model.py
============================
Autoencoder-based semi-supervised Anomaly Detector.
Sklearn MLPRegressor version — no TensorFlow required.
"""
from __future__ import annotations

import os
import time
from typing import Optional

import numpy as np

import config


class AutoencoderIDS:
    """
    Semi-supervised anomaly detector (anomaly gate only).
    Uses sklearn MLPRegressor instead of TensorFlow/Keras.
    """
    name = "Autoencoder"

    def __init__(
            self,
            input_dim: int,
            threshold_percentile: float = None,
            encoding_dim: int = 32,
    ):
        self.input_dim = input_dim
        self.threshold_percentile = threshold_percentile or config.AE_THRESHOLD_PERCENTILE
        self.encoding_dim = encoding_dim
        self.threshold_ = None
        self.model = None
        self.history = None
        self.train_time: float = 0.0
        self.scaler = None

    def _build(self):
        from sklearn.neural_network import MLPRegressor
        from sklearn.preprocessing import StandardScaler

        self.scaler = StandardScaler()

        # Architecture mirrors the TF version: 128 → 64 → bottleneck → 64 → 128
        return MLPRegressor(
            hidden_layer_sizes=(128, 64, self.encoding_dim, 64, 128),
            activation='relu',
            solver='adam',
            alpha=config.DL_PARAMS.get("l2_reg", 1e-4),
            batch_size=config.DL_PARAMS.get("batch_size", 512),
            learning_rate_init=config.DL_PARAMS.get("learning_rate", 0.001),
            max_iter=1000,
            early_stopping=True,
            validation_fraction=0.10,
            n_iter_no_change=config.DL_PARAMS.get("patience", 5),
            random_state=config.RANDOM_STATE,
            verbose=False,
        )

    def fit(self, X_benign: np.ndarray, **kwargs) -> None:
        self.model = self._build()
        print(f"\n  ↳ Training {self.name} on {len(X_benign):,} BENIGN samples only…")

        # Scale features (critical for MLP convergence)
        X_scaled = self.scaler.fit_transform(X_benign)

        t0 = time.time()
        self.model.fit(X_scaled, X_scaled)
        self.train_time = time.time() - t0
        n_iter = self.model.n_iter_
        print(f"    Done in {self.train_time:.1f}s  ({n_iter} iterations)")

        # Threshold = Nth percentile of BENIGN reconstruction errors
        errors = self._mse(X_benign)
        self.threshold_ = np.percentile(errors, self.threshold_percentile)
        print(f"    Anomaly threshold  : {self.threshold_:.6f}  "
              f"(p{self.threshold_percentile:.0f} of training errors)")

    def _mse(self, X: np.ndarray) -> np.ndarray:
        X_scaled = self.scaler.transform(X)
        X_rec = self.model.predict(X_scaled)
        return np.mean((X_scaled - X_rec) ** 2, axis=1)

    def reconstruction_errors(self, X: np.ndarray) -> np.ndarray:
        return self._mse(X)

    def anomaly_scores(self, X: np.ndarray) -> np.ndarray:
        errors = self._mse(X)
        max_val = max(self.threshold_ * 3, errors.max() + 1e-9)
        return np.clip(errors / max_val, 0.0, 1.0)

    def is_anomalous(self, X: np.ndarray) -> np.ndarray:
        assert self.threshold_ is not None, "Call fit() first."
        return self._mse(X) > self.threshold_

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.is_anomalous(X).astype(int)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        attack_prob = self.anomaly_scores(X)
        return np.column_stack([1 - attack_prob, attack_prob])

    def tune_threshold(
            self, X_val: np.ndarray, y_val: np.ndarray, metric: str = "f1"
    ) -> float:
        from sklearn.metrics import f1_score, precision_score, recall_score
        errors = self._mse(X_val)
        best_thr = self.threshold_
        best_val = 0.0
        fn = {"f1": f1_score, "precision": precision_score, "recall": recall_score}[metric]

        for p in range(50, 100):
            thr = np.percentile(errors, p)
            pred = (errors > thr).astype(int)
            score = fn(y_val, pred, zero_division=0)
            if score > best_val:
                best_val, best_thr = score, thr

        self.threshold_ = best_thr
        print(f"    Tuned threshold → {best_thr:.6f}  ({metric}={best_val:.4f})")
        return best_thr

    def save(self, directory: str) -> str:
        import joblib
        os.makedirs(directory, exist_ok=True)
        path = os.path.join(directory, "Autoencoder.pkl")
        meta = {
            "input_dim": self.input_dim,
            "threshold_percentile": self.threshold_percentile,
            "encoding_dim": self.encoding_dim,
            "threshold_": self.threshold_,
            "train_time": self.train_time,
        }
        joblib.dump({"model": self.model, "scaler": self.scaler, "meta": meta}, path)
        print(f"  Model saved → {path}")
        return path

    @classmethod
    def load(cls, directory: str) -> "AutoencoderIDS":
        import joblib
        path = os.path.join(directory, "Autoencoder.pkl")
        data = joblib.load(path)
        meta = data["meta"]
        ae = cls(
            input_dim=meta["input_dim"],
            threshold_percentile=meta["threshold_percentile"],
            encoding_dim=meta["encoding_dim"],
        )
        ae.threshold_ = meta["threshold_"]
        ae.train_time = meta["train_time"]
        ae.model = data["model"]
        ae.scaler = data["scaler"]
        return ae