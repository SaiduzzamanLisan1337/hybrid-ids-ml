"""
models/random_forest_model.py
Random Forest classifier — excellent baseline with built-in feature importance.
"""
from __future__ import annotations
import time
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import config
from models.base_model import BaseIDSModel


class RandomForestIDS(BaseIDSModel):
    """Random Forest with balanced class weights."""
    name = "RandomForest"

    def __init__(self):
        self.model = RandomForestClassifier(**config.RF_PARAMS)
        self.train_time: float = 0.0

    def fit(self, X_train: np.ndarray, y_train: np.ndarray, **kwargs) -> None:
        t0 = time.time()
        print(f"  ↳ Fitting {self.name}…")
        self.model.fit(X_train, y_train)
        self.train_time = time.time() - t0
        print(f"    Done in {self.train_time:.1f}s")

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict_proba(X)

    @property
    def feature_importances_(self) -> np.ndarray:
        return self.model.feature_importances_
