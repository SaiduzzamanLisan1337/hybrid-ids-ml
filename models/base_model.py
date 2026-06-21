"""
models/base_model.py
=====================
Abstract base class — every IDS model must implement these methods.
"""
from __future__ import annotations

import os
from abc import ABC, abstractmethod

import numpy as np


class BaseIDSModel(ABC):
    """
    Abstract interface for all Anomaly IDS classifiers.

    Sub-classes provide:
      fit(X_train, y_train, **kwargs)
      predict(X)          → 1-D label array
      predict_proba(X)    → 2-D probability matrix  (n_samples × n_classes)
      save(directory) / load(directory)
    """

    name: str = "BaseModel"

    @abstractmethod
    def fit(self, X_train: np.ndarray, y_train: np.ndarray, **kwargs) -> None: ...

    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray: ...

    @abstractmethod
    def predict_proba(self, X: np.ndarray) -> np.ndarray: ...

    def save(self, directory: str) -> str:
        import joblib
        os.makedirs(directory, exist_ok=True)
        path = os.path.join(directory, f"{self.name}.pkl")
        joblib.dump(self, path)
        print(f"  Model saved → {path}")
        return path

    @classmethod
    def load(cls, path: str) -> "BaseIDSModel":
        import joblib
        return joblib.load(path)

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__}>"
