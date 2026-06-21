"""
visualization/visualizer.py
============================
Publication-quality plots for the Hybrid Anomaly IDS pipeline.

Plots generated
---------------
01. Class distribution (bar + pie)               — per attack type
02. Feature importance (top-N horizontal bar)
03. Confusion matrices per model (raw + normalised)
04. ROC curves – all models on one axes
05. Model comparison grouped bar chart
06. Deep Learning / Autoencoder training history (loss + accuracy)
07. Per-class F1 heatmap by model
08. Autoencoder reconstruction error distribution
09. Per-attack-type detection rate (multiclass)
10. Hybrid attack-type breakdown (NEW) — what HybridIDS actually labelled
    each true class as, including Unknown-Attack (zero-day) catches
11. Unseen attack detection summary (experiment mode)
"""
from __future__ import annotations

import os
import re
from typing import Dict, List, Optional, Tuple

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

import config

sns.set_theme(style="whitegrid", palette="deep", font_scale=1.05)
PALETTE = ["#1565C0", "#2E7D32", "#C62828", "#6A1B9A", "#E65100", "#00695C", "#AD1457"]


# ─── Shared helpers ──────────────────────────────────────────────────────────

def _save(fig: plt.Figure, filename: str) -> str:
    os.makedirs(config.PLOTS_DIR, exist_ok=True)
    path = os.path.join(config.PLOTS_DIR, filename)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  📊  Saved → {path}")
    return path


# ─── Plot 1 – Class distribution ─────────────────────────────────────────────

def plot_class_distribution(
    y: np.ndarray, class_names: List[str],
    filename: str = "01_class_distribution.png"
) -> str:
    unique, counts = np.unique(y, return_counts=True)
    labels  = [class_names[i] for i in unique]
    colours = [PALETTE[i % len(PALETTE)] for i in range(len(labels))]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    bars = ax1.bar(labels, counts, color=colours, edgecolor="black", linewidth=0.6)
    ax1.set_title("Sample Count per Attack Type", fontsize=13, fontweight="bold")
    ax1.set_ylabel("Count")
    ax1.tick_params(axis="x", rotation=30, labelsize=8)
    for bar, c in zip(bars, counts):
        ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() * 1.01,
                 f"{c:,}", ha="center", va="bottom", fontsize=7.5, fontweight="bold")

    wedges, texts, autotexts = ax2.pie(
        counts, labels=labels, colors=colours,
        autopct="%1.1f%%", startangle=90, pctdistance=0.80,
        wedgeprops={"edgecolor": "white", "linewidth": 1.2},
        textprops={"fontsize": 8},
    )
    for at in autotexts:
        at.set_fontsize(7.5)
    ax2.set_title("Attack Type Proportion", fontsize=13, fontweight="bold")

    fig.suptitle("Dataset Class Distribution (Multiclass)", fontsize=15, fontweight="bold", y=1.02)
    plt.tight_layout()
    return _save(fig, filename)


# ─── Plot 2 – Feature importance ─────────────────────────────────────────────

def plot_feature_importance(
    importances: np.ndarray,
    feature_names: List[str],
    top_n: int = 20,
    filename: str = "02_feature_importance.png",
) -> str:
    idx  = np.argsort(importances)[::-1][:top_n]
    imp  = importances[idx]
    nms  = [feature_names[i] for i in idx]
    cmap = sns.color_palette("Blues_r", top_n)

    fig, ax = plt.subplots(figsize=(10, 8))
    bars = ax.barh(nms[::-1], imp[::-1], color=cmap, edgecolor="black", linewidth=0.4)
    for bar, val in zip(bars, imp[::-1]):
        ax.text(val + imp.max() * 0.01, bar.get_y() + bar.get_height() / 2,
                f"{val:.4f}", va="center", fontsize=7.5)
    ax.set_xlabel("Importance Score (ExtraTrees)", fontsize=11)
    ax.set_title(f"Top {top_n} Feature Importances", fontsize=13, fontweight="bold")
    plt.tight_layout()
    return _save(fig, filename)


# ─── Plot 3 – Confusion matrix ───────────────────────────────────────────────

def plot_confusion_matrix(
    cm: np.ndarray,
    class_names: List[str],
    model_name: str,
    filename: Optional[str] = None,
) -> str:
    filename = filename or f"03_confusion_{model_name}.png"
    cm_norm  = cm.astype(float) / np.maximum(cm.sum(axis=1, keepdims=True), 1)
    n        = len(class_names)
    annot_kw = {"fontsize": max(5.5, 9 - n // 2)}

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(max(13, n * 1.3), 5 + n // 3))
    for ax, data, fmt, title in [
        (ax1, cm,      "d",    "Raw Counts"),
        (ax2, cm_norm, ".2%",  "Row-Normalised"),
    ]:
        sns.heatmap(data, annot=True, fmt=fmt, cmap="Blues", ax=ax,
                    xticklabels=class_names, yticklabels=class_names,
                    linewidths=0.4, linecolor="lightgray",
                    annot_kws=annot_kw, square=n < 6)
        ax.set_xlabel("Predicted Attack Type", fontsize=11)
        ax.set_ylabel("True Attack Type", fontsize=11)
        ax.set_title(f"{model_name} — {title}", fontsize=12, fontweight="bold")
        ax.tick_params(axis="x", rotation=35, labelsize=7.5)
        ax.tick_params(axis="y", rotation=0,  labelsize=7.5)

    plt.tight_layout()
    return _save(fig, filename)


# ─── Plot 4 – ROC curves ────────────────────────────────────────────────────

def plot_roc_curves(
    roc_data: Dict[str, Tuple],
    filename: str = "04_roc_curves.png",
) -> str:
    fig, ax = plt.subplots(figsize=(9, 7))
    for i, (name, (fpr, tpr, auc_val)) in enumerate(roc_data.items()):
        style = "--" if name == "HybridIDS" else "-"
        lw    = 2.6 if name == "HybridIDS" else 2.0
        ax.plot(fpr, tpr, lw=lw, ls=style, color=PALETTE[i % len(PALETTE)],
                label=f"{name}  (AUC = {auc_val:.4f})")
    ax.plot([0, 1], [0, 1], "k--", lw=1.2, label="Random Classifier")
    ax.fill_between([0, 1], [0, 1], alpha=0.05, color="gray")
    ax.set_xlabel("False Positive Rate", fontsize=12)
    ax.set_ylabel("True Positive Rate", fontsize=12)
    ax.set_title("ROC Curves — BENIGN vs ATTACK (all models)", fontsize=14, fontweight="bold")
    ax.legend(loc="lower right", fontsize=9.5, framealpha=0.9)
    ax.grid(True, alpha=0.35)
    ax.set_xlim([-0.01, 1.01])
    ax.set_ylim([-0.01, 1.01])
    plt.tight_layout()
    return _save(fig, filename)


# ─── Plot 5 – Model comparison bar chart ─────────────────────────────────────

def plot_model_comparison(
    comparison_df,
    filename: str = "05_model_comparison.png",
) -> str:
    metrics  = [c for c in ("Accuracy (%)", "Precision (%)", "Recall (%)",
                             "F1-Score (%)", "ROC-AUC (%)") if c in comparison_df.columns]
    n_models = len(comparison_df)
    x        = np.arange(len(metrics))
    width    = 0.80 / max(n_models, 1)
    offsets  = np.linspace(-(n_models - 1) / 2 * width,
                            (n_models - 1) / 2 * width, n_models)

    fig, ax = plt.subplots(figsize=(14, 7))
    for i, (model, row) in enumerate(comparison_df.iterrows()):
        vals = [row.get(m, 0) for m in metrics]
        bars = ax.bar(x + offsets[i], vals, width, label=model,
                      color=PALETTE[i % len(PALETTE)], edgecolor="black",
                      linewidth=0.5, alpha=0.88)
        for bar, v in zip(bars, vals):
            if pd_isna(v):
                continue
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.4,
                    f"{v:.1f}", ha="center", va="bottom", fontsize=6.5, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels([m.replace(" (%)", "") for m in metrics], fontsize=11)
    ax.set_ylabel("Score (%)", fontsize=12)
    ax.set_ylim(0, 112)
    ax.set_title("Model Performance Comparison", fontsize=14, fontweight="bold")
    ax.legend(fontsize=9.5, loc="lower right")
    ax.grid(axis="y", alpha=0.35)
    plt.tight_layout()
    return _save(fig, filename)


def pd_isna(v) -> bool:
    try:
        return v != v  # NaN check without importing pandas here
    except Exception:
        return False


# ─── Plot 6 – Training history (DL or Autoencoder) ───────────────────────────

def plot_dl_training_history(
    history,
    filename: str = "06_dl_training_history.png",
) -> str:
    hist   = history.history
    epochs = range(1, len(hist["loss"]) + 1)
    has_acc = "accuracy" in hist

    if has_acc:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
        panels = [(ax1, "loss", "Loss"), (ax2, "accuracy", "Accuracy")]
    else:
        fig, ax1 = plt.subplots(figsize=(7, 5))
        panels = [(ax1, "loss", "Reconstruction Loss (MSE)")]

    for ax, key, title in panels:
        ax.plot(epochs, hist[key], "b-o", markersize=4, lw=1.8, label="Train")
        if f"val_{key}" in hist:
            ax.plot(epochs, hist[f"val_{key}"], "r-s", markersize=4, lw=1.8, label="Validation")
        ax.set_xlabel("Epoch", fontsize=11)
        ax.set_ylabel(title, fontsize=11)
        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.legend()
        ax.grid(True, alpha=0.35)
        ax.set_xlim([0, len(epochs) + 1])

    plt.tight_layout()
    return _save(fig, filename)


# ─── Plot 7 – Per-class F1 heatmap ───────────────────────────────────────────

def plot_per_class_f1_heatmap(
    results: dict,
    class_names: List[str],
    filename: str = "07_per_class_f1_heatmap.png",
) -> Optional[str]:
    model_f1s: Dict[str, Dict[str, float]] = {}
    for model_name, m in results.items():
        report = m.get("classification_report", "")
        f1s: Dict[str, float] = {}
        for cls in class_names:
            pat = rf"(?:^|\n)\s*{re.escape(cls)}\s+[\d.]+\s+[\d.]+\s+([\d.]+)"
            hit = re.search(pat, report)
            if hit:
                f1s[cls] = float(hit.group(1))
        if f1s:
            model_f1s[model_name] = f1s

    if not model_f1s:
        return None

    models  = list(model_f1s.keys())
    matrix  = np.array([[model_f1s[m].get(c, 0.0) for c in class_names] for m in models])

    fig, ax = plt.subplots(figsize=(max(10, len(class_names) * 1.4),
                                     max(5, len(models) * 0.9)))
    sns.heatmap(matrix, annot=True, fmt=".2f", cmap="YlGnBu",
                xticklabels=class_names, yticklabels=models,
                linewidths=0.4, linecolor="white", ax=ax,
                vmin=0, vmax=1, annot_kws={"size": 8.5})
    ax.set_xlabel("Attack Type", fontsize=12)
    ax.set_title("Per-Attack-Type F1-Score by Model", fontsize=14, fontweight="bold")
    ax.tick_params(axis="x", rotation=35, labelsize=8.5)
    plt.tight_layout()
    return _save(fig, filename)


# ─── Plot 8 – Autoencoder reconstruction error distribution ──────────────────

def plot_reconstruction_error_dist(
    errors_benign: np.ndarray,
    errors_attack: np.ndarray,
    threshold: float,
    filename: str = "08_autoencoder_error_dist.png",
) -> str:
    fig, ax = plt.subplots(figsize=(11, 5))
    cap = np.percentile(np.concatenate([errors_benign, errors_attack]), 99)

    ax.hist(np.clip(errors_benign, 0, cap), bins=100,
            alpha=0.65, color="#1565C0", density=True, label="BENIGN")
    ax.hist(np.clip(errors_attack, 0, cap), bins=100,
            alpha=0.65, color="#C62828", density=True, label="ATTACK (any type)")
    ax.axvline(threshold, color="#E65100", lw=2.0, ls="--",
               label=f"Decision Threshold = {threshold:.5f}")
    ax.set_xlabel("Reconstruction Error (MSE)", fontsize=12)
    ax.set_ylabel("Density", fontsize=12)
    ax.set_title("Autoencoder Anomaly Gate: Reconstruction Error Distributions",
                 fontsize=13, fontweight="bold")
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.35)
    x_max = min(cap, ax.get_xlim()[1])
    ax.axvspan(threshold, x_max, alpha=0.08, color="#C62828")
    plt.tight_layout()
    return _save(fig, filename)


# ─── Plot 9 – Per-attack detection rate (multiclass) ─────────────────────────

def plot_per_attack_detection_rates(
    all_model_results: dict,
    filename: str = "09_per_attack_detection_rates.png",
) -> Optional[str]:
    models_with_dr = {
        name: m["per_class_detection_rate"]
        for name, m in all_model_results.items()
        if "per_class_detection_rate" in m
    }
    if not models_with_dr:
        return None

    all_classes = sorted({c for m in models_with_dr.values() for c in m})
    n_models    = len(models_with_dr)
    x           = np.arange(len(all_classes))
    width       = 0.75 / max(n_models, 1)
    offsets     = np.linspace(-(n_models - 1) / 2 * width,
                               (n_models - 1) / 2 * width, n_models)

    fig, ax = plt.subplots(figsize=(max(12, len(all_classes) * 1.2), 7))
    for i, (model, dr_dict) in enumerate(models_with_dr.items()):
        vals = [dr_dict.get(c, 0.0) * 100 for c in all_classes]
        ax.bar(x + offsets[i], vals, width, label=model,
               color=PALETTE[i % len(PALETTE)], edgecolor="black", linewidth=0.4, alpha=0.88)

    ax.set_xticks(x)
    ax.set_xticklabels(all_classes, rotation=35, ha="right", fontsize=8.5)
    ax.set_ylabel("Detection Rate (%)", fontsize=12)
    ax.set_ylim(0, 115)
    ax.set_title("Per-Attack-Type Detection Rate", fontsize=13, fontweight="bold")
    ax.legend(fontsize=9.5)
    ax.grid(axis="y", alpha=0.35)
    plt.tight_layout()
    return _save(fig, filename)


# ─── Plot 10 – Hybrid attack-type breakdown (NEW) ────────────────────────────

def plot_hybrid_attack_breakdown(
    details: List[dict],
    class_names: List[str],
    y_test: np.ndarray,
    filename: str = "10_hybrid_attack_breakdown.png",
) -> str:
    """
    For each TRUE attack type, show what the HybridIDS actually predicted —
    correct label, wrong label, or flagged as 'Unknown Attack (Anomaly)'.
    This highlights the value the Autoencoder gate adds on top of the
    supervised classifier.
    """
    true_labels = [class_names[i] for i in y_test]
    pred_labels = [d["label"] for d in details]

    categories = sorted(set(true_labels))
    outcome_counts = {cat: {"Correct": 0, "Misclassified": 0, "Flagged Unknown": 0} for cat in categories}

    for t, p in zip(true_labels, pred_labels):
        if p == t:
            outcome_counts[t]["Correct"] += 1
        elif p == config.UNKNOWN_ATTACK_LABEL:
            outcome_counts[t]["Flagged Unknown"] += 1
        else:
            outcome_counts[t]["Misclassified"] += 1

    outcomes = ["Correct", "Misclassified", "Flagged Unknown"]
    colours  = ["#2E7D32", "#C62828", "#6A1B9A"]
    matrix   = np.array([[outcome_counts[c][o] for o in outcomes] for c in categories])
    totals   = matrix.sum(axis=1, keepdims=True)
    pct      = matrix / np.maximum(totals, 1) * 100

    fig, ax = plt.subplots(figsize=(max(11, len(categories) * 1.3), 6.5))
    bottom = np.zeros(len(categories))
    for i, (outcome, colour) in enumerate(zip(outcomes, colours)):
        vals = pct[:, i]
        ax.bar(categories, vals, bottom=bottom, label=outcome,
               color=colour, edgecolor="black", linewidth=0.4, alpha=0.88)
        for j, v in enumerate(vals):
            if v > 3:
                ax.text(j, bottom[j] + v / 2, f"{v:.0f}%", ha="center", va="center",
                        fontsize=8, fontweight="bold", color="white")
        bottom += vals

    ax.set_ylabel("Percentage of True Class (%)", fontsize=12)
    ax.set_title(
        "HybridIDS Outcome by True Attack Type\n"
        "(Correct ID · Misclassified · Caught only as 'Unknown Attack')",
        fontsize=13, fontweight="bold",
    )
    ax.tick_params(axis="x", rotation=35, labelsize=9)
    ax.legend(fontsize=10, loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=3)
    ax.set_ylim(0, 105)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    return _save(fig, filename)


# ─── Plot 11 – Unseen attack detection bar chart (experiment mode) ───────────

def plot_unseen_attack_results(
    results: Dict[str, Dict[str, float]],
    filename: str = "11_unseen_attack_detection.png",
) -> str:
    models  = list(results.keys())
    attacks = sorted({a for m in results.values() for a in m})
    matrix  = np.array([[results[m].get(a, 0.0) * 100 for a in attacks] for m in models])

    fig, ax = plt.subplots(figsize=(max(10, len(attacks) * 1.4),
                                    max(5, len(models) * 0.8 + 2)))
    sns.heatmap(matrix, annot=True, fmt=".1f", cmap="RdYlGn",
                xticklabels=attacks, yticklabels=models,
                vmin=0, vmax=100, linewidths=0.4, linecolor="white",
                annot_kws={"size": 9}, ax=ax)
    ax.set_xlabel("Unseen (Holdout) Attack Type", fontsize=12)
    ax.set_title(
        "Zero-Day Detection Rate (%)  —  Models never trained on these attacks",
        fontsize=12, fontweight="bold",
    )
    ax.tick_params(axis="x", rotation=30, labelsize=9)
    plt.tight_layout()
    return _save(fig, filename)
