"""
evaluation/evaluator.py
========================
Metrics for all IDS models, including the new HybridIDS.

New in this version
-------------------
- evaluate_hybrid() evaluates the HybridIDS separately: it reports
  standard multiclass metrics PLUS a dedicated section showing:
    • True-positive rate per attack type (detection rate)
    • False-positive rate on BENIGN traffic
    • Unknown Attack (zero-day) recall — how many novel flows were caught
- All evaluation is multiclass — no binary-mode special-casing.
"""
from __future__ import annotations

import json
import os
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
    auc,
)

import config


class Evaluator:
    """
    Evaluate and compare IDS models.

    Usage
    -----
    ev = Evaluator(class_names=["BENIGN","DDoS","DoS Hulk",...])
    metrics = ev.evaluate("LightGBM", y_test, y_pred, y_proba)
    df      = ev.comparison_dataframe()
    ev.save("outputs/")
    """

    def __init__(self, class_names: List[str]):
        self.class_names = class_names
        self.n_classes   = len(class_names)
        self.results: Dict[str, dict] = {}

    # ── Single model evaluation ───────────────────────────────────────────

    def evaluate(
        self,
        name: str,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_proba: Optional[np.ndarray] = None,
        training_time: float = 0.0,
    ) -> dict:
        avg = "weighted"

        m: dict = {
            "accuracy":  accuracy_score(y_true, y_pred),
            "precision": precision_score(y_true, y_pred, average=avg, zero_division=0),
            "recall":    recall_score(y_true, y_pred, average=avg, zero_division=0),
            "f1":        f1_score(y_true, y_pred, average=avg, zero_division=0),
            "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
            "training_time_s":  round(training_time, 2),
            "classification_report": classification_report(
                y_true, y_pred,
                labels=list(range(self.n_classes)),
                target_names=self.class_names,
                zero_division=0,
            ),
        }

        # ROC-AUC (weighted OvR for multiclass)
        if y_proba is not None:
            try:
                n_col = y_proba.shape[1]
                if self.n_classes <= n_col:
                    proba_subset = y_proba[:, :self.n_classes]
                    if n_col > self.n_classes:
                        # Extra columns (e.g. HybridIDS "Unknown Attack")
                        # were sliced off — renormalise so rows sum to 1,
                        # otherwise sklearn rejects them as probabilities.
                        row_sums = proba_subset.sum(axis=1, keepdims=True)
                        proba_subset = proba_subset / np.maximum(row_sums, 1e-9)
                    m["roc_auc"] = roc_auc_score(
                        y_true, proba_subset,
                        multi_class="ovr", average="weighted",
                        labels=list(range(self.n_classes)),
                    )
                    m["avg_precision"] = float(
                        np.mean([
                            average_precision_score(
                                (y_true == c).astype(int),
                                proba_subset[:, c],
                            )
                            for c in range(self.n_classes)
                            if (y_true == c).any()
                        ])
                    )
            except Exception as exc:
                print(f"  ⚠  AUC skipped for {name}: {exc}")

        # Per-class detection rate
        cm  = np.array(m["confusion_matrix"])
        with np.errstate(divide="ignore", invalid="ignore"):
            per_class_recall = np.where(
                cm.sum(axis=1) > 0,
                np.diag(cm) / cm.sum(axis=1),
                0.0,
            )
        m["per_class_detection_rate"] = {
            cls: float(r)
            for cls, r in zip(self.class_names, per_class_recall)
        }

        self.results[name] = m
        self._print(name, m)
        return m

    # ── Hybrid-specific evaluation ─────────────────────────────────────────

    def evaluate_hybrid(
        self,
        hybrid_model,               # HybridIDS instance
        X_test: np.ndarray,
        y_test: np.ndarray,         # integer labels in [0, n_classes)
        holdout_mask: Optional[np.ndarray] = None,
    ) -> dict:
        """
        Evaluate the HybridIDS including unknown-attack detection analysis.

        Parameters
        ----------
        holdout_mask : Optional bool array marking samples that were
                       NEVER seen during supervised training.  When provided,
                       computes a zero-day detection rate separately.
        """
        print("\n[Hybrid IDS Evaluation]")

        extended_names  = hybrid_model.extended_class_names()
        unknown_idx     = len(self.class_names)   # index of Unknown Attack

        # Run prediction
        details  = hybrid_model.predict_detail(X_test)
        y_pred_h = np.array([d["label_idx"] for d in details])

        # ── Standard multiclass metrics (excluding Unknown Attack column) ─
        # Map unknown-attack predictions back to the nearest supervised class
        # for standard metrics (so label space matches y_test)
        y_pred_std = np.where(y_pred_h == unknown_idx,
                              hybrid_model.benign_idx,   # count as "missed BENIGN"
                              y_pred_h)
        # Use the hybrid's own probability matrix (already includes the AE
        # fusion) so ROC-AUC / avg-precision are computed too, not left NaN.
        y_proba_h = hybrid_model.predict_proba(X_test)
        m = self.evaluate("HybridIDS", y_test, y_pred_std, y_proba_h, training_time=0.0)

        # ── AE gate stats ─────────────────────────────────────────────────
        ae_scores  = np.array([d["ae_score"] for d in details])
        is_anomaly = np.array([d["is_anomaly"] for d in details])
        benign_idx = hybrid_model.benign_idx

        n_benign_true  = (y_test == benign_idx).sum()
        n_attack_true  = (y_test != benign_idx).sum()
        ae_fp          = is_anomaly[y_test == benign_idx].sum()   # false alarms on BENIGN
        ae_tp          = is_anomaly[y_test != benign_idx].sum()   # correctly flagged attacks

        m["ae_stats"] = {
            "attack_recall_pct":  round(100 * ae_tp / max(n_attack_true, 1), 2),
            "benign_fpr_pct":     round(100 * ae_fp / max(n_benign_true, 1), 2),
            "n_unknown_flagged":  int((y_pred_h == unknown_idx).sum()),
        }

        # ── Zero-day / holdout analysis ────────────────────────────────────
        if holdout_mask is not None and holdout_mask.any():
            X_ho      = X_test[holdout_mask]
            y_ho      = y_test[holdout_mask]
            det_ho    = hybrid_model.predict_detail(X_ho)
            # A holdout attack is "detected" if predicted as anything except BENIGN
            detected  = sum(1 for d in det_ho if d["label_idx"] != benign_idx)
            m["ae_stats"]["zero_day_detection_pct"] = round(
                100 * detected / max(len(det_ho), 1), 2
            )
            m["ae_stats"]["n_holdout_samples"]      = int(holdout_mask.sum())

        self.results["HybridIDS"] = m
        self._print_hybrid_summary(m)
        return m

    # ── ROC data ──────────────────────────────────────────────────────────

    def roc_data(
        self, name: str, y_true: np.ndarray, y_proba: np.ndarray
    ) -> Optional[Tuple]:
        """
        Binary BENIGN-vs-ATTACK ROC for a single model.
        Works by binarising: BENIGN=0, anything else=1.
        """
        if y_proba is None:
            return None
        benign_idx = self.class_names.index(config.BENIGN_LABEL) \
            if config.BENIGN_LABEL in self.class_names else 0
        try:
            y_bin   = (y_true != benign_idx).astype(int)
            # Attack probability = 1 - P(BENIGN)
            p_attack = 1 - y_proba[:, benign_idx]
            fpr, tpr, _ = roc_curve(y_bin, p_attack)
            auc_val     = auc(fpr, tpr)
            return fpr, tpr, auc_val
        except Exception:
            return None

    # ── Comparison table ──────────────────────────────────────────────────

    def comparison_dataframe(self) -> pd.DataFrame:
        rows = []
        for name, m in self.results.items():
            row = {
                "Model":          name,
                "Accuracy (%)":   round(m["accuracy"]  * 100, 2),
                "Precision (%)":  round(m["precision"] * 100, 2),
                "Recall (%)":     round(m["recall"]    * 100, 2),
                "F1-Score (%)":   round(m["f1"]        * 100, 2),
                "ROC-AUC (%)":    round(m.get("roc_auc", float("nan")) * 100, 2),
                "Train Time (s)": m.get("training_time_s", 0),
            }
            if "ae_stats" in m:
                row["AE Attack Recall (%)"]  = m["ae_stats"].get("attack_recall_pct", "—")
                row["AE BENIGN FPR (%)"]     = m["ae_stats"].get("benign_fpr_pct", "—")
                row["Zero-Day Detected (%)"] = m["ae_stats"].get("zero_day_detection_pct", "—")
            rows.append(row)
        return pd.DataFrame(rows).set_index("Model")

    # ── Persistence ───────────────────────────────────────────────────────

    def save(self, output_dir: str = None) -> None:
        output_dir = output_dir or config.OUTPUT_DIR
        os.makedirs(output_dir, exist_ok=True)
        serialisable = {}
        for name, m in self.results.items():
            serialisable[name] = {
                k: v for k, v in m.items() if k != "classification_report"
            }
        json_path = os.path.join(output_dir, "evaluation_results.json")
        with open(json_path, "w") as f:
            json.dump(serialisable, f, indent=2)
        csv_path = os.path.join(output_dir, "model_comparison.csv")
        self.comparison_dataframe().to_csv(csv_path)
        print(f"\n  ✓ Metrics saved  → {json_path}")
        print(f"  ✓ Comparison CSV → {csv_path}")

    # ── Private ───────────────────────────────────────────────────────────

    @staticmethod
    def _print(name: str, m: dict) -> None:
        bar = "─" * 60
        print(f"\n{bar}")
        print(f"  {name}")
        print(bar)
        for k in ("accuracy", "precision", "recall", "f1", "roc_auc", "avg_precision"):
            if k in m:
                print(f"  {k.upper().replace('_', ' '):<22}: {m[k]*100:>7.3f}%")
        print(f"\n{m['classification_report']}")
        if "per_class_detection_rate" in m:
            print("  Detection Rate per Attack Type:")
            for cls, dr in m["per_class_detection_rate"].items():
                bar_filled = "█" * int(dr * 20)
                bar_empty  = "░" * (20 - int(dr * 20))
                print(f"    {cls:<35} [{bar_filled}{bar_empty}]  {dr*100:>6.2f}%")

    @staticmethod
    def _print_hybrid_summary(m: dict) -> None:
        if "ae_stats" not in m:
            return
        s = m["ae_stats"]
        print("\n  ── Autoencoder Gate Stats ──────────────────────────")
        print(f"    Attack recall (AE flagged as anomaly)  : {s.get('attack_recall_pct','?'):>6}%")
        print(f"    BENIGN false positive rate             : {s.get('benign_fpr_pct','?'):>6}%")
        print(f"    Flows labelled 'Unknown Attack'        : {s.get('n_unknown_flagged','?'):>6}")
        if "zero_day_detection_pct" in s:
            print(f"    Zero-day (holdout) detection rate      : {s['zero_day_detection_pct']:>6}%")
            print(f"    (n={s['n_holdout_samples']} holdout flows)")
