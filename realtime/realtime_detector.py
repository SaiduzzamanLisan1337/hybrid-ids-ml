"""
realtime/realtime_detector.py
==============================
Real-time network flow detection with attack-type identification.

Changes from original
---------------------
- Uses HybridIDS by default: the Autoencoder gate + a supervised classifier
  run on every chunk.  Each flow gets a specific attack-type label, not just
  BENIGN / ATTACK.
- LiveDashboard now shows a breakdown of detected attack types in a rolling
  window, including "Unknown Attack (Anomaly)" for zero-day detections.
- Falls back to a supervised-only model if HybridIDS is not available.
- train_medians are stored in the FeatureEngineer and applied to chunks
  before the FE transform (fixes the leakage bug from the original).

Usage
-----
# After training with main.py:
    python realtime/realtime_detector.py \\
        --csv  data/raw/Friday-WorkingHours.csv \\
        --model outputs/saved_models/LightGBM.pkl \\
        --ae    outputs/saved_models   \\
        --fe    outputs/saved_models/feature_engineer.pkl \\
        --chunk 200 --delay 0.3

# Demo mode (no files needed):
    python realtime/realtime_detector.py --demo
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from collections import Counter, deque
from datetime import datetime
from typing import List, Optional

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import config

# ─── ANSI colour codes ────────────────────────────────────────────────────────
RED    = "\033[91m"
GREEN  = "\033[92m"
YELLOW = "\033[93m"
CYAN   = "\033[96m"
BOLD   = "\033[1m"
RESET  = "\033[0m"
DIM    = "\033[2m"
MAGENTA = "\033[95m"

ATTACK_COLOUR = {
    "BENIGN":                          GREEN,
    config.UNKNOWN_ATTACK_LABEL:       MAGENTA,
}


def _col(text: str, colour: str) -> str:
    return f"{colour}{text}{RESET}"


def _attack_col(label: str) -> str:
    return ATTACK_COLOUR.get(label, RED)


# ─── Live Dashboard ───────────────────────────────────────────────────────────

class LiveDashboard:
    """
    ANSI terminal dashboard.  Shows:
      - Total flows / BENIGN / ATTACK counts
      - Rolling attack rate bar
      - Breakdown of the top-5 detected attack types
      - Recent flow log with attack labels + AE scores
    """
    WINDOW = 500

    def __init__(self, class_names: List[str]):
        self.class_names   = class_names
        self.total         = 0
        self.n_benign      = 0
        self.n_attack      = 0
        self.history       = deque(maxlen=self.WINDOW)
        self.attack_counts = Counter()
        self.recent_log    = deque(maxlen=6)
        self._start        = time.time()
        self._alert_thresh = 0.30

    def update(self, details: List[dict]) -> None:
        benign_label = config.BENIGN_LABEL
        now = time.time()
        for d in details:
            self.total += 1
            label = d["label"]
            is_att = label != benign_label
            self.history.append((now, is_att))
            if is_att:
                self.n_attack += 1
                self.attack_counts[label] += 1
                self.recent_log.append(
                    f"  {datetime.now():%H:%M:%S}  "
                    f"{_col(f'{label:<38}', _attack_col(label))}  "
                    f"conf={d['confidence']:.3f}  ae={d['ae_score']:.3f}  [{d['source']}]"
                )
            else:
                self.n_benign += 1

    @property
    def _recent_attack_rate(self) -> float:
        if not self.history:
            return 0.0
        return sum(is_att for _, is_att in self.history) / len(self.history)

    def render(self) -> None:
        elapsed = time.time() - self._start
        fps     = self.total / max(elapsed, 1)
        rate    = self._recent_attack_rate
        alert   = rate > self._alert_thresh

        if self.total > 0:
            print("\033[28A\033[J", end="")

        sep = "─" * 66
        print(_col(f"\n{'═'*66}", CYAN))
        print(_col(f"  🛡   ANOMALY IDS  —  HYBRID REAL-TIME MONITOR", BOLD + CYAN))
        print(_col(sep, CYAN))

        print(f"  Flows processed  : {_col(f'{self.total:>8,}', BOLD)}")
        print(f"  BENIGN           : {_col(f'{self.n_benign:>8,}', GREEN)}")
        print(f"  ATTACK (total)   : {_col(f'{self.n_attack:>8,}', RED if self.n_attack else GREEN)}")
        print(f"  Throughput       : {fps:>8.1f} flows/s")
        print(f"  Elapsed          : {elapsed:>8.1f} s")

        # Rolling attack rate bar
        bar_len  = 32
        filled   = int(rate * bar_len)
        bar_col  = RED if rate > self._alert_thresh else (YELLOW if rate > 0.1 else GREEN)
        bar_str  = "█" * filled + "░" * (bar_len - filled)
        print(f"\n  {_col(f'Attack rate (last {self.WINDOW} flows)', BOLD)}")
        print(f"  [{_col(bar_str, bar_col)}]  {rate*100:5.1f}%")

        if alert:
            print(f"\n  {_col('⚠  HIGH ATTACK RATE — POSSIBLE INCIDENT', RED + BOLD)}")
        else:
            print(f"\n  {_col('✓  Traffic appears normal', GREEN)}")

        # Attack type breakdown (top 5)
        print(f"\n  {_col('Attack type breakdown (all time):', BOLD)}")
        top5 = self.attack_counts.most_common(5)
        if top5:
            for label, cnt in top5:
                pct = cnt / max(self.n_attack, 1) * 100
                col = _attack_col(label)
                print(f"    {_col(f'{label:<40}', col)} {cnt:>6,}  ({pct:5.1f}%)")
        else:
            print("    No attacks detected yet")
        # Pad to 5 lines
        for _ in range(5 - len(top5)):
            print()

        # Recent attack log
        print(f"\n  {_col('Recent detections:', BOLD)}")
        recent = list(self.recent_log)
        for entry in recent[-6:]:
            print(entry)
        for _ in range(6 - len(recent)):
            print()

        print(_col(f"{'═'*66}", CYAN))
        sys.stdout.flush()


# ─── Detector ────────────────────────────────────────────────────────────────

class RealTimeDetector:
    """
    Wraps a HybridIDS (or any BaseIDSModel) + FeatureEngineer and
    processes network flows from a CSV file or a generator in chunks.
    """

    def __init__(self, model, feature_engineer, class_names: List[str]):
        self.model = model
        self.fe    = feature_engineer
        self.class_names = class_names
        self._is_hybrid = hasattr(model, "predict_detail")

    def _prepare_chunk(self, chunk: pd.DataFrame) -> Optional[np.ndarray]:
        lbl = config.LABEL_COLUMN
        if lbl in chunk.columns:
            chunk = chunk.drop(columns=[lbl])

        chunk.columns = [c.strip() for c in chunk.columns]
        chunk = chunk.select_dtypes(include=[np.number])
        chunk.replace([float("inf"), float("-inf")], float("nan"), inplace=True)

        # Use training medians for imputation (no leakage)
        if self.fe.train_medians is not None and len(self.fe.train_medians) == chunk.shape[1]:
            for j in range(chunk.shape[1]):
                mask = chunk.iloc[:, j].isna()
                if mask.any():
                    chunk.iloc[mask, j] = self.fe.train_medians[j]
        else:
            chunk.fillna(chunk.median(), inplace=True)

        if chunk.empty:
            return None
        try:
            return self.fe.transform(chunk.values)
        except Exception as exc:
            print(f"  ⚠  FE transform failed: {exc}")
            return None

    def _predict_details(self, X: np.ndarray) -> List[dict]:
        """Unified interface: return list of detail dicts regardless of model type."""
        if self._is_hybrid:
            return self.model.predict_detail(X)
        else:
            # Wrap a plain supervised model in the same dict format
            preds  = self.model.predict(X)
            probes = self.model.predict_proba(X)
            results = []
            for i, pred in enumerate(preds):
                label = self.class_names[pred] if pred < len(self.class_names) else "?"
                conf  = float(probes[i, pred]) if probes.shape[1] > pred else 0.0
                results.append({
                    "label":          label,
                    "label_idx":      int(pred),
                    "confidence":     round(conf, 5),
                    "ae_score":       0.0,
                    "is_anomaly":     label != config.BENIGN_LABEL,
                    "sup_label":      label,
                    "sup_confidence": round(conf, 5),
                    "source":         "supervised",
                })
            return results

    def stream_csv(
        self,
        csv_path: str,
        chunk_size: int = 200,
        delay: float = 0.3,
        log_path: Optional[str] = None,
        max_chunks: Optional[int] = None,
    ) -> None:
        print(f"  Streaming: {csv_path}")
        print(f"  Chunk size: {chunk_size}  |  Delay: {delay}s\n")

        model_name = getattr(self.model, "name", type(self.model).__name__)
        dash       = LiveDashboard(self.class_names)
        log_rows   = []

        reader = pd.read_csv(csv_path, chunksize=chunk_size, low_memory=False)

        for i, chunk in enumerate(reader):
            if max_chunks and i >= max_chunks:
                break
            X = self._prepare_chunk(chunk)
            if X is None:
                continue

            details = self._predict_details(X)
            dash.update(details)
            dash.render()

            for j, d in enumerate(details):
                log_rows.append({
                    "chunk":        i,
                    "flow_index":   i * chunk_size + j,
                    "timestamp":    datetime.now().isoformat(),
                    "label":        d["label"],
                    "confidence":   d["confidence"],
                    "ae_score":     d["ae_score"],
                    "is_anomaly":   d["is_anomaly"],
                    "source":       d["source"],
                })
            time.sleep(delay)

        print("\n\n  ✓  Stream complete.")
        if log_path and log_rows:
            pd.DataFrame(log_rows).to_csv(log_path, index=False)
            print(f"  📄  Detection log → {log_path}")

    def stream_generator(
        self,
        row_generator,
        feature_names: list,
        chunk_size: int = 100,
        delay: float = 0.5,
    ) -> None:
        """Accept a generator that yields individual flow dicts."""
        dash = LiveDashboard(self.class_names)
        buf  = []

        for row in row_generator:
            buf.append(row)
            if len(buf) < chunk_size:
                continue

            chunk = pd.DataFrame(buf, columns=feature_names)
            buf   = []
            X     = self._prepare_chunk(chunk)
            if X is None:
                continue

            details = self._predict_details(X)
            dash.update(details)
            dash.render()
            time.sleep(delay)


# ─── Demo mode ────────────────────────────────────────────────────────────────

def _synthetic_generator(
    n_features: int,
    n_flows: int = 600,
    attack_rate: float = 0.20,
    attack_types: List[str] = None,
):
    """Yield synthetic flows with random attack type assignment."""
    rng          = np.random.default_rng(42)
    attack_types = attack_types or ["DDoS", "DoS Hulk", "Port Scan", "Bot"]
    for _ in range(n_flows):
        is_attack = rng.random() < attack_rate
        row       = rng.exponential(300 if is_attack else 1000, size=n_features).tolist()
        yield row, (rng.choice(attack_types) if is_attack else config.BENIGN_LABEL)
        time.sleep(0.002)


def _demo_run():
    from sklearn.ensemble import IsolationForest, RandomForestClassifier
    from features.feature_engineering import FeatureEngineer
    from data.data_loader import generate_demo_data, preprocess
    from models.autoencoder_model import AutoencoderIDS
    from models.hybrid_model import HybridIDS

    print(_col("\n  [Demo] Generating synthetic training data…\n", CYAN))
    demo_data = generate_demo_data(n_samples=20_000)

    X_tr, X_te, y_tr, y_te, le, feat_names, class_names, medians = preprocess(demo_data)
    benign_idx = class_names.index(config.BENIGN_LABEL)

    fe = FeatureEngineer()
    X_tr_fe, y_tr_fe = fe.fit_transform(X_tr, y_tr, feat_names, train_medians=medians)
    X_te_fe          = fe.transform(X_te)

    # Lightweight supervised model for demo
    rf = RandomForestClassifier(n_estimators=50, random_state=42, n_jobs=-1, class_weight="balanced")
    rf.fit(X_tr_fe, y_tr_fe)

    # Wrap RF in a compatible object
    class RFWrapper:
        name = "RandomForest (Demo)"
        def predict(self, X): return rf.predict(X)
        def predict_proba(self, X): return rf.predict_proba(X)

    # Simple AE for demo (no TF needed — use IsolationForest proxy)
    class IFAutoencoder:
        name = "Autoencoder (Demo)"
        threshold_ = 0.5
        def __init__(self):
            self._m = IsolationForest(contamination=0.15, random_state=42)
        def fit_benign(self, X):
            self._m.fit(X)
        def is_anomalous(self, X):
            return self._m.predict(X) == -1
        def anomaly_scores(self, X):
            s = -self._m.score_samples(X)
            s = (s - s.min()) / (s.max() - s.min() + 1e-9)
            return s

    ae_proxy = IFAutoencoder()
    X_benign = X_tr_fe[y_tr_fe == benign_idx]
    ae_proxy.fit_benign(X_benign)

    # Build a demo HybridIDS-compatible object
    class DemoHybrid:
        name = "HybridIDS (Demo)"
        def __init__(self, rf_w, ae_p, cls_names, b_idx):
            self.rf  = rf_w
            self.ae  = ae_p
            self.class_names = cls_names
            self.benign_idx  = b_idx
        def predict_detail(self, X):
            preds     = self.rf.predict(X)
            probes    = self.rf.predict_proba(X)
            ae_scores = self.ae.anomaly_scores(X)
            is_anom   = self.ae.is_anomalous(X)
            results   = []
            unknown_i = len(self.class_names)
            for i, pred in enumerate(preds):
                ae_s = float(ae_scores[i])
                is_a = bool(is_anom[i])
                if pred == self.benign_idx and is_a:
                    label  = config.UNKNOWN_ATTACK_LABEL
                    l_idx  = unknown_i
                    conf   = ae_s
                    source = "autoencoder"
                else:
                    label  = self.class_names[pred]
                    l_idx  = int(pred)
                    conf   = float(probes[i, pred])
                    source = "supervised" if not is_a else "hybrid"
                results.append({
                    "label": label, "label_idx": l_idx, "confidence": round(conf,5),
                    "ae_score": round(ae_s,5), "is_anomaly": is_a,
                    "sup_label": self.class_names[pred], "sup_confidence": round(float(probes[i,pred]),5),
                    "source": source,
                })
            return results
        def extended_class_names(self):
            return self.class_names + [config.UNKNOWN_ATTACK_LABEL]

    hybrid = DemoHybrid(RFWrapper(), ae_proxy, class_names, benign_idx)
    all_names = class_names + [config.UNKNOWN_ATTACK_LABEL]

    detector = RealTimeDetector(hybrid, fe, all_names)
    print(_col("\n  [Demo] Starting synthetic stream…\n", CYAN))

    n_feat = X_tr_fe.shape[1]
    feat_cols = [f"f{i}" for i in range(n_feat)]
    raw_gen = _synthetic_generator(n_feat, n_flows=700, attack_types=class_names[1:])
    row_gen = (row for row, _ in raw_gen)

    detector.stream_generator(row_gen, feature_names=feat_cols, chunk_size=50, delay=0.15)


# ─── CLI ─────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Anomaly IDS — Hybrid Real-Time Detection")
    p.add_argument("--csv",    default=None)
    p.add_argument("--model",  default="outputs/saved_models/LightGBM.pkl")
    p.add_argument("--ae",     default="outputs/saved_models",
                   help="Directory containing Autoencoder.keras + Autoencoder_meta.pkl")
    p.add_argument("--fe",     default="outputs/saved_models/feature_engineer.pkl")
    p.add_argument("--chunk",  type=int,   default=200)
    p.add_argument("--delay",  type=float, default=0.3)
    p.add_argument("--max_chunks", type=int, default=None)
    p.add_argument("--log",    default="outputs/realtime_log.csv")
    p.add_argument("--no_hybrid", action="store_true",
                   help="Use supervised model only (skip Autoencoder gate)")
    p.add_argument("--demo",   action="store_true")
    return p.parse_args()


def main():
    args = parse_args()

    if args.demo:
        _demo_run()
        return

    if args.csv is None:
        print("  ✗  Provide --csv or use --demo")
        sys.exit(1)

    import joblib
    print(f"  Loading model          : {args.model}")
    sup_model = joblib.load(args.model)
    print(f"  Loading FeatureEngineer: {args.fe}")
    fe        = joblib.load(args.fe)

    # Load class names from meta
    meta_path = os.path.join(os.path.dirname(args.fe), "HybridIDS_meta.pkl")
    class_names = None
    if os.path.exists(meta_path):
        meta        = joblib.load(meta_path)
        class_names = meta["class_names"]

    if not args.no_hybrid and os.path.exists(os.path.join(args.ae, "Autoencoder.keras")):
        from models.autoencoder_model import AutoencoderIDS
        from models.hybrid_model import HybridIDS
        ae = AutoencoderIDS.load(args.ae)
        benign_idx = class_names.index(config.BENIGN_LABEL) if class_names else 0
        model = HybridIDS(sup_model, ae, class_names or [], benign_idx)
        print("  Hybrid IDS (AE gate + supervised) loaded.")
    else:
        model = sup_model
        print("  Supervised-only model loaded.")

    all_names = (class_names or []) + ([] if args.no_hybrid else [config.UNKNOWN_ATTACK_LABEL])
    os.makedirs(os.path.dirname(args.log), exist_ok=True)

    detector = RealTimeDetector(model, fe, all_names)
    detector.stream_csv(
        args.csv,
        chunk_size = args.chunk,
        delay      = args.delay,
        log_path   = args.log,
        max_chunks = args.max_chunks,
    )


if __name__ == "__main__":
    main()
