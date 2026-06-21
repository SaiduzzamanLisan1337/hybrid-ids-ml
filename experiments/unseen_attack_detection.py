"""
experiments/unseen_attack_detection.py
========================================
Zero-day detection experiment using the HybridIDS.

Design
------
1. Pick N attack types to hold out completely from training (e.g. "Bot",
   "Infiltration", "Heartbleed").
2. Train a supervised multiclass classifier (LightGBM) + Autoencoder on
   the REMAINING data only — the held-out attacks are never seen.
3. Assemble a HybridIDS from those two components.
4. Test on the held-out attack flows:
     - Supervised-only: almost always predicts BENIGN (never learned the
       pattern) or misattributes it to a different known attack class.
     - HybridIDS: the Autoencoder gate flags the unusual reconstruction
       error even though the supervised model says BENIGN, surfacing it
       as "Unknown Attack (Anomaly)" — a successful zero-day catch.

Fix from original implementation
---------------------------------
Column alignment between train/holdout splits now uses an explicit
pd.DataFrame.reindex(columns=..., fill_value=0) instead of manual
zero-padding, preventing silent feature misalignment.
"""
from __future__ import annotations

import os
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

import config
from features.feature_engineering import FeatureEngineer
from models.lightgbm_model import LightGBMIDS
from models.autoencoder_model import AutoencoderIDS
from models.hybrid_model import HybridIDS
from visualization.visualizer import plot_unseen_attack_results

DEFAULT_HOLDOUT_ATTACKS = ["Bot", "Infiltration", "Heartbleed"]


def run_experiment(
    data: pd.DataFrame,
    feature_engineer: Optional[FeatureEngineer] = None,
    class_names: Optional[List[str]] = None,
    benign_idx: Optional[int] = None,
    holdout_attacks: Optional[List[str]] = None,
) -> Dict[str, Dict[str, float]]:
    """
    Run the zero-day detection experiment and return:
        {"Supervised-Only": {attack: detection_rate}, "HybridIDS": {...}}
    """
    lbl = config.LABEL_COLUMN
    holdout_attacks = holdout_attacks or DEFAULT_HOLDOUT_ATTACKS
    available       = [a for a in holdout_attacks if a in data[lbl].unique()]

    if not available:
        print(f"  ⚠  None of {holdout_attacks} found in dataset. "
              f"Available labels: {sorted(data[lbl].unique())[:15]}")
        return {}

    print(f"\n  Holding out attack types: {available}")

    seen_data     = data[~data[lbl].isin(available)].copy()
    holdout_data  = data[data[lbl].isin(available)].copy()
    print(f"  Seen data    : {len(seen_data):,} rows")
    print(f"  Holdout data : {len(holdout_data):,} rows")

    # ── Preprocess the "seen" data (everything except holdout attacks) ────
    from data.data_loader import preprocess
    X_train, X_test, y_train, y_test, le, feat_names, seen_class_names, medians = preprocess(seen_data)
    b_idx = seen_class_names.index(config.BENIGN_LABEL)

    # ── Feature engineering on seen data ───────────────────────────────────
    fe = FeatureEngineer()
    X_train_fe, y_train_fe = fe.fit_transform(X_train, y_train, feat_names, train_medians=medians)
    X_test_fe              = fe.transform(X_test)

    # ── Train supervised model (LightGBM) on seen classes only ────────────
    print("\n  Training LightGBM on SEEN classes only…")
    sup = LightGBMIDS()
    sup.fit(X_train_fe, y_train_fe)

    # ── Train Autoencoder on BENIGN-only from seen data ────────────────────
    print("\n  Training Autoencoder on BENIGN-only (seen data)…")
    ae = AutoencoderIDS(input_dim=X_train_fe.shape[1])
    ae.fit(X_train_fe[y_train_fe == b_idx])

    hybrid = HybridIDS(sup, ae, seen_class_names, b_idx)

    # ── Prepare holdout data using the SAME numeric columns/order ─────────
    holdout_y_raw = holdout_data[lbl].copy()
    holdout_X_raw = holdout_data.drop(columns=[lbl]).select_dtypes(include=[np.number])

    # Robust column alignment — explicit reindex, not manual zero-padding.
    # Any column present in training but missing here is filled with 0;
    # any extra column in holdout but absent from training is dropped.
    holdout_X_aligned = holdout_X_raw.reindex(columns=feat_names, fill_value=0)
    holdout_X_aligned = holdout_X_aligned.replace([np.inf, -np.inf], np.nan)
    holdout_X_aligned = holdout_X_aligned.fillna(
        pd.Series(medians, index=feat_names)
    )

    X_holdout_fe = fe.transform(holdout_X_aligned.values)

    # ── Evaluate: Supervised-only vs HybridIDS on holdout attacks ─────────
    sup_pred = sup.predict(X_holdout_fe)
    hyb_det  = hybrid.predict_detail(X_holdout_fe)

    results: Dict[str, Dict[str, float]] = {
        "Supervised-Only": {},
        "HybridIDS":        {},
    }

    for attack in available:
        mask = (holdout_y_raw.values == attack)
        if mask.sum() == 0:
            continue

        # Supervised-only: "detected" means predicted anything but BENIGN
        sup_detected = (sup_pred[mask] != b_idx).mean()
        results["Supervised-Only"][attack] = float(sup_detected)

        # HybridIDS: "detected" means label != BENIGN
        # (includes both correct-attack-type matches it might fluke into,
        #  AND "Unknown Attack (Anomaly)" catches from the AE gate)
        hyb_labels   = np.array([hyb_det[i]["label"] for i in np.where(mask)[0]])
        hyb_detected = (hyb_labels != config.BENIGN_LABEL).mean()
        results["HybridIDS"][attack] = float(hyb_detected)

        print(f"\n  Attack type: {attack}  (n={mask.sum()})")
        print(f"    Supervised-only detection rate : {sup_detected*100:6.2f}%  "
              f"(never saw this attack — mostly misses)")
        print(f"    HybridIDS detection rate       : {hyb_detected*100:6.2f}%  "
              f"(AE gate catches anomalous patterns)")

        n_unknown = sum(1 for i in np.where(mask)[0]
                         if hyb_det[i]["label"] == config.UNKNOWN_ATTACK_LABEL)
        print(f"    → of which flagged 'Unknown Attack' : {n_unknown}/{mask.sum()}")

    # ── Plot ────────────────────────────────────────────────────────────────
    plot_unseen_attack_results(results)

    # ── Save raw results ──────────────────────────────────────────────────
    import json
    out_path = os.path.join(config.OUTPUT_DIR, "unseen_attack_results.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  ✓ Results saved → {out_path}")

    return results


if __name__ == "__main__":
    from data.data_loader import load_dataset
    data = load_dataset()
    run_experiment(data)
