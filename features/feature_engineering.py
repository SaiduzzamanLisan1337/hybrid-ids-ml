"""
features/feature_engineering.py
=================================
Feature engineering pipeline — unchanged logic, now also stores the
training medians so real-time inference can impute consistently.
"""
from __future__ import annotations

from typing import List, Tuple

import numpy as np
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import VarianceThreshold
from sklearn.preprocessing import RobustScaler

import config


class FeatureEngineer:
    """
    Stateful feature pipeline. Call fit_transform() on training data once,
    then transform() on any subsequent split (test / real-time stream).
    """

    def __init__(self):
        self.scaler = RobustScaler()
        self._variance_mask: np.ndarray | None = None
        self._corr_mask: np.ndarray | None     = None
        self._selected_idx: np.ndarray | None  = None
        self._feature_importances: np.ndarray | None = None
        self._selected_feature_names: List[str] = []
        # Stored so real-time detector can replicate the exact same imputation
        self.train_medians: np.ndarray | None  = None
        self.n_input_features: int             = 0

    # ── Public API ────────────────────────────────────────────────────────

    def fit_transform(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        feature_names: List[str],
        train_medians: np.ndarray | None = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Fit the full pipeline on training data and return (X_transformed, y).
        y may be SMOTE-augmented if config.APPLY_SMOTE is True.

        Parameters
        ----------
        train_medians : Pre-computed training-set medians from preprocess()
                        (used for storing in the pkl for real-time use).
        """
        print("\n[Feature Engineering]")
        self.n_input_features = X_train.shape[1]
        self.train_medians    = train_medians

        X, names = self._variance_filter(X_train, feature_names)
        X, names = self._correlation_filter(X, names)
        X        = self.scaler.fit_transform(X)
        X, names = self._importance_selection(X, y_train, names)

        self._selected_feature_names = names
        print(f"  Final feature count : {X.shape[1]}")
        print(f"  Top features        : {', '.join(names[:5])} …")

        if config.APPLY_SMOTE:
            X, y_train = self._smote(X, y_train)

        return X, y_train

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Apply fitted transformations to a new array."""
        assert self._variance_mask is not None, "Call fit_transform first."
        X = X[:, self._variance_mask]
        X = X[:, self._corr_mask]
        X = self.scaler.transform(X)
        X = X[:, self._selected_idx]
        return X

    @property
    def selected_feature_names(self) -> List[str]:
        return self._selected_feature_names

    @property
    def feature_importances(self) -> np.ndarray:
        return self._feature_importances

    # ── Private helpers ───────────────────────────────────────────────────

    def _variance_filter(
        self, X: np.ndarray, names: List[str]
    ) -> Tuple[np.ndarray, List[str]]:
        sel   = VarianceThreshold(threshold=config.VARIANCE_THRESHOLD)
        X_out = sel.fit_transform(X)
        mask  = sel.get_support()
        self._variance_mask = mask
        names_out = [n for n, m in zip(names, mask) if m]
        print(f"  After variance filter     : {X_out.shape[1]:>3} features")
        return X_out, names_out

    def _correlation_filter(
        self, X: np.ndarray, names: List[str]
    ) -> Tuple[np.ndarray, List[str]]:
        if X.shape[1] > 1:
            corr  = np.corrcoef(X.T)
            np.fill_diagonal(corr, 0)
            upper = np.triu(np.abs(corr), k=1)
            to_drop = set()
            for i in range(upper.shape[0]):
                for j in range(i + 1, upper.shape[1]):
                    if upper[i, j] > config.CORRELATION_THRESHOLD:
                        to_drop.add(j)
            mask = np.array([i not in to_drop for i in range(X.shape[1])])
        else:
            mask = np.ones(X.shape[1], dtype=bool)

        self._corr_mask = mask
        X_out     = X[:, mask]
        names_out = [n for n, m in zip(names, mask) if m]
        print(f"  After correlation filter  : {X_out.shape[1]:>3} features")
        return X_out, names_out

    def _importance_selection(
        self, X: np.ndarray, y: np.ndarray, names: List[str]
    ) -> Tuple[np.ndarray, List[str]]:
        k  = min(config.TOP_N_FEATURES, X.shape[1])
        et = ExtraTreesClassifier(
            n_estimators = 100,
            random_state = config.RANDOM_STATE,
            n_jobs       = -1,
        )
        et.fit(X, y)
        importances = et.feature_importances_
        idx         = np.argsort(importances)[::-1][:k]
        idx_sorted  = np.sort(idx)

        self._selected_idx        = idx_sorted
        self._feature_importances = importances[idx_sorted]
        names_out = [names[i] for i in idx_sorted]
        X_out     = X[:, idx_sorted]
        print(f"  After top-{k} ET selection : {X_out.shape[1]:>3} features")
        return X_out, names_out

    @staticmethod
    def _smote(X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        try:
            from imblearn.over_sampling import SMOTE
            unique, counts = np.unique(y, return_counts=True)
            if len(unique) < 2 or min(counts) < 6:
                print("  SMOTE skipped (not enough minority samples)")
                return X, y
            sm = SMOTE(random_state=config.RANDOM_STATE, k_neighbors=5)
            X_res, y_res = sm.fit_resample(X, y)
            added = len(X_res) - len(X)
            print(f"  After SMOTE               : {len(X_res):,} samples (+{added:,})")
            return X_res, y_res
        except ImportError:
            print("  SMOTE skipped (pip install imbalanced-learn)")
            return X, y
