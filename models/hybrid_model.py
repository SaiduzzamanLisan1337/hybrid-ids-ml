"""
models/hybrid_model.py
=======================
HybridIDS — fuses the semi-supervised Autoencoder gate with a supervised
multiclass classifier to produce final per-attack-type predictions.

Decision Logic
--------------
For each incoming flow:

  1.  Supervised classifier  →  predicted class  (e.g. "DDoS", "BENIGN")
                              +  confidence score

  2.  Autoencoder gate       →  is_anomalous flag
                              +  anomaly score  [0, 1]

  3.  Fusion rule:

      ┌─────────────────────────────┬──────────────────────────────────────┐
      │ Supervised says BENIGN      │ AE says NOT anomalous  → BENIGN      │
      │                             │ AE says     anomalous  → "Unknown    │
      │                             │   Attack (Anomaly)"  [zero-day flag] │
      ├─────────────────────────────┼──────────────────────────────────────┤
      │ Supervised says <attack>    │ AE says NOT anomalous  → <attack>    │
      │                             │   (supervised wins — likely a known  │
      │                             │   attack that doesn't reconstruct    │
      │                             │   badly by coincidence)              │
      │                             │ AE says     anomalous  → <attack>    │
      │                             │   (both agree it's an attack;        │
      │                             │   keep the specific label)           │
      └─────────────────────────────┴──────────────────────────────────────┘

  In other words: the AE overrides BENIGN predictions when it detects
  anomalies (catches zero-day attacks), but never overrides specific
  attack-type labels (the supervised model has more information there).

predict_detail() returns a richer dict per flow for real-time use.
"""
from __future__ import annotations

import os
from typing import List, Optional

import numpy as np

import config
from models.base_model import BaseIDSModel
from models.autoencoder_model import AutoencoderIDS


class HybridIDS(BaseIDSModel):
    """
    Semi-supervised + supervised hybrid intrusion detector.

    Parameters
    ----------
    supervised_model : Any trained BaseIDSModel (LightGBM recommended)
    autoencoder      : Trained AutoencoderIDS (BENIGN-only)
    class_names      : List of class name strings from LabelEncoder
    benign_idx       : Integer index of BENIGN in class_names
    """
    name = "HybridIDS"

    def __init__(
        self,
        supervised_model: BaseIDSModel,
        autoencoder: AutoencoderIDS,
        class_names: List[str],
        benign_idx: int,
    ):
        self.supervised   = supervised_model
        self.ae           = autoencoder
        self.class_names  = class_names
        self.benign_idx   = benign_idx
        self.n_classes    = len(class_names)
        # Append the unknown-attack label if not already present
        if config.UNKNOWN_ATTACK_LABEL not in self.class_names:
            self._extended_names = self.class_names + [config.UNKNOWN_ATTACK_LABEL]
        else:
            self._extended_names = self.class_names

    # ── BaseIDSModel interface ─────────────────────────────────────────────

    def fit(self, X_train: np.ndarray, y_train: np.ndarray, **kwargs) -> None:
        """
        HybridIDS components are trained separately before assembly.
        This method is a no-op (training already done externally).
        """
        pass

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Return integer class indices using the hybrid fusion rule.
        Index for UNKNOWN_ATTACK_LABEL = len(self.class_names).
        """
        unknown_idx = len(self.class_names)   # index beyond original classes
        sup_pred    = self.supervised.predict(X)
        is_anomaly  = self.ae.is_anomalous(X)

        hybrid = sup_pred.copy()
        for i in range(len(X)):
            if sup_pred[i] == self.benign_idx and is_anomaly[i]:
                # Supervised says BENIGN but AE flagged it — zero-day
                hybrid[i] = unknown_idx
        return hybrid

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Return extended probability matrix with an extra column for the
        Unknown Attack class.  Shape: (n_samples, n_classes + 1).
        """
        sup_proba   = self.supervised.predict_proba(X)   # (N, C)
        ae_scores   = self.ae.anomaly_scores(X)          # (N,) in [0,1]

        # Add a "Unknown Attack" column: probability = AE score × P(BENIGN by sup)
        p_benign    = sup_proba[:, self.benign_idx]
        p_unknown   = ae_scores * p_benign               # high when AE says anomalous AND sup says benign

        # Renormalise: reduce BENIGN probability by what we assigned to unknown
        extended    = np.column_stack([sup_proba, p_unknown])
        extended[:, self.benign_idx] = np.maximum(p_benign - p_unknown, 0.0)
        row_sums    = extended.sum(axis=1, keepdims=True)
        extended    = extended / np.maximum(row_sums, 1e-9)
        return extended

    def predict_detail(self, X: np.ndarray) -> List[dict]:
        """
        Rich per-flow prediction dict for real-time / logging use.

        Returns list of dicts with keys:
          label          : str   — final predicted class name
          label_idx      : int   — integer index (len=unknown)
          confidence     : float — probability of predicted class
          ae_score       : float — anomaly score [0, 1]
          is_anomaly     : bool  — True if AE flagged it
          sup_label      : str   — what the supervised model predicted
          sup_confidence : float — supervised model confidence
          source         : str   — "supervised" | "autoencoder" | "hybrid"
        """
        unknown_idx  = len(self.class_names)
        sup_pred     = self.supervised.predict(X)
        sup_proba    = self.supervised.predict_proba(X)
        ae_scores    = self.ae.anomaly_scores(X)
        is_anomaly   = self.ae.is_anomalous(X)

        results = []
        for i in range(len(X)):
            s_idx   = int(sup_pred[i])
            s_label = self.class_names[s_idx] if s_idx < len(self.class_names) else "?"
            s_conf  = float(sup_proba[i, s_idx]) if sup_proba.shape[1] > s_idx else 0.0
            ae_s    = float(ae_scores[i])
            is_att  = bool(is_anomaly[i])

            if s_idx == self.benign_idx and is_att:
                # Zero-day / novel attack
                label  = config.UNKNOWN_ATTACK_LABEL
                l_idx  = unknown_idx
                conf   = ae_s
                source = "autoencoder"
            elif s_idx != self.benign_idx:
                # Known attack type — supervised wins
                label  = s_label
                l_idx  = s_idx
                conf   = s_conf
                source = "supervised" if not is_att else "hybrid"
            else:
                # Both say BENIGN
                label  = s_label
                l_idx  = s_idx
                conf   = s_conf
                source = "supervised"

            results.append({
                "label":          label,
                "label_idx":      l_idx,
                "confidence":     round(conf, 5),
                "ae_score":       round(ae_s, 5),
                "is_anomaly":     is_att,
                "sup_label":      s_label,
                "sup_confidence": round(s_conf, 5),
                "source":         source,
            })
        return results

    def extended_class_names(self) -> List[str]:
        """Class names including the Unknown Attack label at the end."""
        return self._extended_names

    # ── Persistence ───────────────────────────────────────────────────────

    def save(self, directory: str) -> str:
        import joblib
        os.makedirs(directory, exist_ok=True)
        # Save the supervised sub-model via its own save()
        self.supervised.save(directory)
        # Save the autoencoder via its own save()
        self.ae.save(directory)
        # Save metadata for reconstruction
        meta = {
            "supervised_name": self.supervised.name,
            "class_names":     self.class_names,
            "benign_idx":      self.benign_idx,
        }
        path = os.path.join(directory, "HybridIDS_meta.pkl")
        joblib.dump(meta, path)
        print(f"  HybridIDS meta saved → {path}")
        return path

    @classmethod
    def load(
        cls,
        directory: str,
        supervised_model: Optional[BaseIDSModel] = None,
    ) -> "HybridIDS":
        """
        Load a saved HybridIDS.  If supervised_model is not provided, it
        tries to load LightGBM (the default hybrid backbone) from directory.
        """
        import joblib
        meta = joblib.load(os.path.join(directory, "HybridIDS_meta.pkl"))

        if supervised_model is None:
            # Default: load LightGBM
            from models.lightgbm_model import LightGBMIDS
            sup = joblib.load(os.path.join(directory, "LightGBM.pkl"))
        else:
            sup = supervised_model

        ae = AutoencoderIDS.load(directory)
        return cls(
            supervised_model = sup,
            autoencoder      = ae,
            class_names      = meta["class_names"],
            benign_idx       = meta["benign_idx"],
        )
