"""
models/svm_model.py
"""
import time
import numpy as np
from collections import Counter
from sklearn.svm import SVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.preprocessing import StandardScaler
import config
from models.base_model import BaseIDSModel


class SVMIDS(BaseIDSModel):
    name = "SVM"

    def __init__(self):
        # Use linear kernel by default — much faster and more stable
        # on high-dimensional tabular data with many classes than RBF
        svm_params = {
            'kernel': 'linear',  # ← CHANGED: linear is faster & more stable
            'C': 1.0,  # ← CHANGED: lower C for regularization
            'class_weight': 'balanced',
            'max_iter': 20000,
            'random_state': config.RANDOM_STATE,
        }
        # Allow config override if kernel/C/gamma are specified
        for k, v in config.SVM_PARAMS.items():
            if k not in ('probability',):
                svm_params[k] = v

        base_svc = SVC(**svm_params)
        # cv=2 prevents the "least populated class has only 1 member" warning
        self.model = CalibratedClassifierCV(
            base_svc, ensemble=False, method='sigmoid', cv=2
        )
        self.scaler = StandardScaler()
        self.train_time = 0.0
        self._subsampled = False

    def _stratified_subsample(self, X, y, max_rows):
        """
        Smart subsampling: keep ALL minority class samples intact.
        Only downsample the majority class (BENIGN) to fit within max_rows.
        This preserves rare attack types so SVM can actually learn them.
        """
        classes, counts = np.unique(y, return_counts=True)
        n_classes = len(classes)

        # Always keep at least 2 samples per class (minimum for learning)
        min_per_class = 2
        reserved = min_per_class * n_classes
        remaining_budget = max_rows - reserved

        if remaining_budget < 0:
            # Edge case: max_rows smaller than 2×n_classes
            # Just do simple stratified sampling
            from sklearn.model_selection import train_test_split
            _, idx = train_test_split(
                np.arange(len(y)), test_size=max_rows / len(y),
                stratify=y, random_state=config.RANDOM_STATE
            )
            return X[idx], y[idx], True

        # Calculate how many samples to take from each class
        # Strategy: give rare classes their full count, cap majority class
        total_rare = sum(c for c in counts if c <= 100)
        majority_budget = max_rows - total_rare

        selected_idx = []
        for cls, cnt in zip(classes, counts):
            cls_idx = np.where(y == cls)[0]
            if cnt <= 100:
                # Keep all rare class samples
                n_take = cnt
            else:
                # Majority class: take proportional share of budget
                n_take = min(cnt, majority_budget)

            if n_take < cnt:
                rng = np.random.default_rng(config.RANDOM_STATE + int(cls))
                chosen = rng.choice(cls_idx, n_take, replace=False)
            else:
                chosen = cls_idx

            selected_idx.extend(chosen)

        selected_idx = np.array(selected_idx)
        rng = np.random.default_rng(config.RANDOM_STATE)
        rng.shuffle(selected_idx)  # shuffle to avoid class ordering bias

        return X[selected_idx], y[selected_idx], True

    def fit(self, X_train, y_train, **kwargs):
        t0 = time.time()
        max_rows = config.SVM_MAX_TRAIN_ROWS

        # Smart subsampling: preserve minority classes
        if len(X_train) > max_rows:
            X_train, y_train, self._subsampled = self._stratified_subsample(
                X_train, y_train, max_rows
            )
            print(f"  Fitting SVM (stratified subsample to {len(X_train):,})...")
            # Show what we kept
            counts = Counter(y_train)
            print(f"    Class distribution after subsampling:")
            for cls, cnt in sorted(counts.items(), key=lambda x: x[1]):
                pct = 100 * cnt / len(y_train)
                print(f"      Class {cls}: {cnt:,} samples ({pct:.1f}%)")
        else:
            print(f"  Fitting SVM...")

        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)

        self.model.fit(X_train_scaled, y_train)
        self.train_time = time.time() - t0
        print(f"    Done in {self.train_time:.1f}s")

    def predict(self, X):
        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)

    def predict_proba(self, X):
        X_scaled = self.scaler.transform(X)
        return self.model.predict_proba(X_scaled)