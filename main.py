"""
main.py  —  Anomaly IDS  —  Hybrid Pipeline
=============================================
Runs supervised multiclass classifiers AND the semi-supervised Autoencoder
gate together, then assembles a HybridIDS that identifies specific attack types
while also catching zero-day / unseen attacks.

Quick start
-----------
# Demo (no dataset needed):
    python main.py --demo

# Full run (CICIDS CSVs in data/raw/):
    python main.py --data_dir data/raw

# Specific models only:
    python main.py --data_dir data/raw --models LightGBM DeepLearning

# Skip SMOTE (faster):
    python main.py --demo --no_smote

# Run unseen-attack experiment after training:
    python main.py --data_dir data/raw --experiment

Pipeline steps
--------------
[1] Load data
[2] Preprocess  (multiclass labels, leakage-free imputation)
[3] Feature engineering
[4] Train supervised classifiers  (per-attack-type labels)
[5] Train Autoencoder  (BENIGN-only, anomaly gate)
[6] Assemble HybridIDS  (AE gate + best supervised model)
[7] Evaluate all models + Hybrid
[8] Comparison plots + export
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import joblib
import numpy as np
from sklearn.model_selection import train_test_split

import config

from data.data_loader import load_dataset, preprocess, generate_demo_data
from features.feature_engineering import FeatureEngineer
from models.random_forest_model import RandomForestIDS
from models.xgboost_model import XGBoostIDS
from models.lightgbm_model import LightGBMIDS
from models.svm_model import SVMIDS
from models.deep_learning_model import DeepLearningIDS
from models.autoencoder_model import AutoencoderIDS
from models.hybrid_model import HybridIDS
from evaluation.evaluator import Evaluator
from visualization.visualizer import (
    plot_class_distribution,
    plot_feature_importance,
    plot_confusion_matrix,
    plot_roc_curves,
    plot_model_comparison,
    plot_dl_training_history,
    plot_per_class_f1_heatmap,
    plot_per_attack_detection_rates,
    plot_hybrid_attack_breakdown,
    plot_reconstruction_error_dist,
)


# ─── CLI ─────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Anomaly IDS — Hybrid Semi-supervised + Supervised Pipeline",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--data_dir",    default=config.DATA_DIR)
    p.add_argument("--sample",      type=float, default=config.SAMPLE_FRACTION)
    p.add_argument("--models", nargs="+",
                   default=config.MODELS_TO_RUN,
                   choices=["RandomForest","XGBoost","LightGBM","SVM","DeepLearning"])
    p.add_argument("--no_smote",    action="store_true")
    p.add_argument("--no_hybrid",   action="store_true",
                   help="Skip Autoencoder training and HybridIDS assembly")
    p.add_argument("--hybrid_backbone", default="LightGBM",
                   choices=["RandomForest","XGBoost","LightGBM","SVM","DeepLearning"],
                   help="Which trained supervised model to use as the HybridIDS backbone")
    p.add_argument("--demo",        action="store_true")
    p.add_argument("--experiment",  action="store_true")
    p.add_argument("--output_dir",  default=config.OUTPUT_DIR)
    return p.parse_args()


# ─── Model factory ────────────────────────────────────────────────────────────

def build_model(name: str, input_dim: int, n_classes: int):
    registry = {
        "RandomForest": RandomForestIDS,
        "XGBoost":      XGBoostIDS,
        "LightGBM":     LightGBMIDS,
        "SVM":          SVMIDS,
        "DeepLearning": lambda: DeepLearningIDS(input_dim=input_dim, n_classes=n_classes),
    }
    if name in registry:
        return registry[name]()
    raise ValueError(f"Unknown model: {name}")


# ─── Banner ───────────────────────────────────────────────────────────────────

def _banner(msg: str, width: int = 65) -> None:
    print(f"\n{'='*width}")
    print(f"  {msg}")
    print(f"{'='*width}")


# ─── Main pipeline ────────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()

    config.APPLY_SMOTE     = not args.no_smote
    config.SAMPLE_FRACTION = args.sample
    config.DATA_DIR        = args.data_dir
    config.OUTPUT_DIR      = args.output_dir
    config.PLOTS_DIR       = os.path.join(args.output_dir, "plots")
    config.MODELS_DIR      = os.path.join(args.output_dir, "saved_models")
    config.HYBRID_MODE     = not args.no_hybrid

    for d in (config.OUTPUT_DIR, config.PLOTS_DIR, config.MODELS_DIR):
        os.makedirs(d, exist_ok=True)

    _banner("Anomaly-Based IDS  —  Hybrid ML Pipeline")
    print(f"  Supervised models : {', '.join(args.models)}")
    print(f"  Semi-supervised   : {'Autoencoder (AE gate)' if config.HYBRID_MODE else 'disabled'}")
    print(f"  Hybrid backbone   : {args.hybrid_backbone}")
    print(f"  SMOTE             : {'enabled' if config.APPLY_SMOTE else 'disabled'}")
    print(f"  Sample fraction   : {config.SAMPLE_FRACTION*100:.0f}%")

    # ── [1] Load data ─────────────────────────────────────────────────────
    _banner("[1] Loading Data")
    if args.demo:
        print("  ⚡  Demo mode — generating synthetic CICIDS-like data …")
        data = generate_demo_data(n_samples=60_000)
    else:
        data = load_dataset()

    # ── [2] Preprocess ────────────────────────────────────────────────────
    _banner("[2] Preprocessing  (multiclass — per-attack-type labels)")
    X_train, X_test, y_train, y_test, le, feature_names, class_names, train_medians = preprocess(data)
    n_classes  = len(class_names)
    benign_idx = class_names.index(config.BENIGN_LABEL)

    plot_class_distribution(y_train, class_names)

    # ── [3] Feature engineering ───────────────────────────────────────────
    _banner("[3] Feature Engineering")
    fe = FeatureEngineer()
    X_train_fe, y_train_fe = fe.fit_transform(
        X_train, y_train, feature_names, train_medians=train_medians
    )
    X_test_fe  = fe.transform(X_test)
    input_dim  = X_train_fe.shape[1]

    plot_feature_importance(fe.feature_importances, fe.selected_feature_names)

    fe_path = os.path.join(config.MODELS_DIR, "feature_engineer.pkl")
    joblib.dump(fe, fe_path)
    print(f"\n  ✓ FeatureEngineer saved → {fe_path}")

    # ── [4] Validation split (for DeepLearning) ───────────────────────────
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_train_fe, y_train_fe,
        test_size    = config.VALIDATION_SIZE,
        random_state = config.RANDOM_STATE,
        stratify     = y_train_fe,
    )

    # ── [5] Train & evaluate supervised models ────────────────────────────
    _banner("[4] Training Supervised Classifiers  (multiclass — identifies attack type)")
    evaluator    = Evaluator(class_names)
    roc_data     = {}
    trained_sup  = {}   # name → model instance (used to pick hybrid backbone)

    for model_name in args.models:
        _banner(f"  {model_name}")
        try:
            model = build_model(model_name, input_dim, n_classes)
        except Exception as exc:
            print(f"  ⚠  Could not build {model_name}: {exc} — skipping")
            continue

        t0 = time.time()
        try:
            if model_name == "DeepLearning":
                model.fit(X_tr, y_tr, X_val=X_val, y_val=y_val)
            else:
                model.fit(X_train_fe, y_train_fe)
        except Exception as exc:
            print(f"  ✗  Training failed: {exc}")
            continue

        train_time = time.time() - t0
        y_pred     = model.predict(X_test_fe)
        y_proba    = model.predict_proba(X_test_fe)

        metrics = evaluator.evaluate(model_name, y_test, y_pred, y_proba, training_time=train_time)

        cm = np.array(metrics["confusion_matrix"])
        plot_confusion_matrix(cm, class_names, model_name)

        roc = evaluator.roc_data(model_name, y_test, y_proba)
        if roc:
            roc_data[model_name] = roc

        if model_name == "DeepLearning" and model.history:
            plot_dl_training_history(model.history)

        model.save(config.MODELS_DIR)
        trained_sup[model_name] = model

    # ── [6] Train Autoencoder + assemble HybridIDS ────────────────────────
    hybrid_model = None
    ae           = None

    if config.HYBRID_MODE:
        _banner("[5] Training Autoencoder (BENIGN-only, anomaly gate)")
        try:
            X_benign = X_train_fe[y_train_fe == benign_idx]
            ae       = AutoencoderIDS(input_dim=input_dim)
            t0       = time.time()
            ae.fit(X_benign)
            ae_train_time = time.time() - t0

            # Plot reconstruction error distributions
            errors_b = ae.reconstruction_errors(X_test_fe[y_test == benign_idx][:5000])
            errors_a = ae.reconstruction_errors(X_test_fe[y_test != benign_idx][:5000])
            plot_reconstruction_error_dist(errors_b, errors_a, ae.threshold_)

            if ae.history:
                plot_dl_training_history(ae.history, filename="06b_autoencoder_training.png")
            ae.save(config.MODELS_DIR)

            # Pick backbone
            backbone_name = args.hybrid_backbone
            if backbone_name not in trained_sup:
                # Fall back to the first available model
                backbone_name = next(iter(trained_sup), None)

            if backbone_name and backbone_name in trained_sup:
                _banner(f"[6] Assembling HybridIDS  (AE gate + {backbone_name})")
                hybrid_model = HybridIDS(
                    supervised_model = trained_sup[backbone_name],
                    autoencoder      = ae,
                    class_names      = class_names,
                    benign_idx       = benign_idx,
                )
                hybrid_model.save(config.MODELS_DIR)

                # Evaluate HybridIDS
                _banner("[6a] Evaluating HybridIDS")
                evaluator.evaluate_hybrid(hybrid_model, X_test_fe, y_test)

                # Hybrid ROC (binary: BENIGN vs any attack)
                h_proba = hybrid_model.predict_proba(X_test_fe)
                roc_h   = evaluator.roc_data("HybridIDS", y_test, h_proba)
                if roc_h:
                    roc_data["HybridIDS"] = roc_h

                # Per-attack breakdown plot
                details = hybrid_model.predict_detail(X_test_fe)
                plot_hybrid_attack_breakdown(details, class_names, y_test)
            else:
                print("  ⚠  No backbone model available — skipping HybridIDS assembly.")

        except Exception as exc:
            print(f"  ⚠  Autoencoder / HybridIDS failed: {exc}")
            import traceback; traceback.print_exc()

    # ── [7] Comparison plots ──────────────────────────────────────────────
    _banner("[7] Generating Comparison Plots")
    comp_df = evaluator.comparison_dataframe()
    print("\n", comp_df.to_string())

    plot_model_comparison(comp_df)
    if roc_data:
        plot_roc_curves(roc_data)
    plot_per_class_f1_heatmap(evaluator.results, class_names)
    plot_per_attack_detection_rates(evaluator.results)

    # ── [8] Save results ──────────────────────────────────────────────────
    _banner("[8] Saving Results")
    evaluator.save(config.OUTPUT_DIR)
    comp_df.to_csv(os.path.join(config.OUTPUT_DIR, "model_comparison.csv"))
    print(f"\n  All outputs saved to: {os.path.abspath(config.OUTPUT_DIR)}/")

    # ── [9] Optional experiment ───────────────────────────────────────────
    if args.experiment and not args.demo:
        _banner("[9] Unseen Attack Detection Experiment")
        try:
            from experiments.unseen_attack_detection import run_experiment
            exp_data = load_dataset()
            run_experiment(
                exp_data,
                feature_engineer = fe,
                class_names      = class_names,
                benign_idx       = benign_idx,
            )
        except Exception as exc:
            print(f"  ⚠  Experiment failed: {exc}")
            import traceback; traceback.print_exc()

    _banner("Pipeline Complete!")
    print(f"  Plots   → {os.path.abspath(config.PLOTS_DIR)}/")
    print(f"  Results → {os.path.abspath(config.OUTPUT_DIR)}/")
    print(f"  Models  → {os.path.abspath(config.MODELS_DIR)}/\n")


if __name__ == "__main__":
    main()
