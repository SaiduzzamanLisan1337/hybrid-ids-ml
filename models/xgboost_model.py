"""
models/xgboost_model.py
XGBoost classifier — typically fastest tree ensemble with highest accuracy.
"""
from __future__ import annotations
import time
import numpy as np
import config
from models.base_model import BaseIDSModel


class XGBoostIDS(BaseIDSModel):
    name = "XGBoost"

    def __init__(self):
        import xgboost as xgb
        self.model = xgb.XGBClassifier(**config.XGB_PARAMS)
        self.train_time: float = 0.0

    def fit(self, X_train: np.ndarray, y_train: np.ndarray, **kwargs) -> None:
        t0 = time.time()
        print(f"  ↳ Fitting {self.name}…")
        self.model.fit(
            X_train, y_train,
            eval_set=[(X_train, y_train)],
            verbose=False,
        )
        self.train_time = time.time() - t0
        print(f"    Done in {self.train_time:.1f}s")

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict_proba(X)
