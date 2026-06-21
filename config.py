"""
config.py — Central configuration for the Anomaly IDS Pipeline.

New in this version
-------------------
- CLASSIFICATION_MODE is now always "multiclass" (per-attack-type labels).
- HYBRID_MODE enables the semi-supervised + supervised pipeline together.
- The Autoencoder acts as an anomaly GATE; a multiclass classifier then
  identifies WHICH attack type was detected.
- Binary shorthand (BENIGN vs ATTACK) is only used internally by the
  Autoencoder gate; all final outputs carry per-attack-type labels.
"""
import os

# =============================================================================
# Paths
# =============================================================================
DATA_DIR    = "data/MachineLearningCSV"
OUTPUT_DIR  = "outputs"
PLOTS_DIR   = os.path.join(OUTPUT_DIR, "plots")
MODELS_DIR  = os.path.join(OUTPUT_DIR, "saved_models")

# =============================================================================
# Dataset
# =============================================================================
LABEL_COLUMN        = "Label"
BENIGN_LABEL        = "BENIGN"

# Always multiclass so we can identify the attack type.
CLASSIFICATION_MODE = "multiclass"

SAMPLE_FRACTION     = 1.0
RANDOM_STATE        = 42

# =============================================================================
# Hybrid semi-supervised + supervised pipeline
# =============================================================================
# When True, the pipeline runs BOTH:
#   • Supervised multiclass classifiers  (identify specific attack type)
#   • Autoencoder anomaly detector       (catch unseen/zero-day attacks)
# and fuses them into a HybridIDS that combines both signals.
HYBRID_MODE = True

# Autoencoder anomaly-gate threshold percentile.
# Flows whose reconstruction error exceeds the Nth percentile of BENIGN
# training errors are flagged as anomalous before the multiclass step.
AE_THRESHOLD_PERCENTILE = 95.0

# When the Autoencoder flags a flow as anomalous but the supervised model
# says BENIGN, the hybrid system uses the AE label.  This trades a small
# false-positive increase for much better zero-day recall.
HYBRID_AE_OVERRIDES_BENIGN = True

# =============================================================================
# Feature Engineering
# =============================================================================
VARIANCE_THRESHOLD    = 0.01
CORRELATION_THRESHOLD = 0.98
TOP_N_FEATURES        = 30
APPLY_SMOTE           = True

# =============================================================================
# Train / Test / Validation Split
# =============================================================================
TEST_SIZE       = 0.20
VALIDATION_SIZE = 0.10

# =============================================================================
# Model hyperparameters
# =============================================================================
RF_PARAMS = dict(
    n_estimators      = 200,
    max_depth         = 20,
    min_samples_split = 5,
    min_samples_leaf  = 2,
    n_jobs            = -1,
    random_state      = RANDOM_STATE,
    class_weight      = "balanced",
)

XGB_PARAMS = dict(
    n_estimators      = 200,
    max_depth         = 7,
    learning_rate     = 0.1,
    subsample         = 0.8,
    colsample_bytree  = 0.8,
    eval_metric       = "mlogloss",
    random_state      = RANDOM_STATE,
    n_jobs            = -1,
    tree_method       = "hist",
)

LGBM_PARAMS = dict(
    n_estimators     = 200,
    max_depth        = 7,
    learning_rate    = 0.1,
    num_leaves       = 63,
    subsample        = 0.8,
    colsample_bytree = 0.8,
    class_weight     = "balanced",
    random_state     = RANDOM_STATE,
    n_jobs           = -1,
    verbose          = -1,
)

SVM_PARAMS = dict(
    kernel       = "rbf",
    C            = 10.0,
    gamma        = "scale",
    probability  = True,
    class_weight = "balanced",
    max_iter     = 20000,   # ← CHANGED from 5000
)

DL_PARAMS = dict(
    epochs        = 30,
    batch_size    = 512,
    learning_rate = 0.001,
    dropout_rate  = 0.30,
    patience      = 5,
    l2_reg        = 1e-4,
)

# =============================================================================
# Which models to run
# =============================================================================
# "Autoencoder" is always included when HYBRID_MODE=True.
MODELS_TO_RUN = ["RandomForest", "XGBoost", "LightGBM", "SVM"]

SVM_MAX_TRAIN_ROWS = 80_000

# Label used when the Autoencoder detects an anomaly but the supervised
# classifier has no confident prediction (e.g. reconstruction error very high
# but the attack type was never seen in training).
UNKNOWN_ATTACK_LABEL = "Unknown Attack (Anomaly)"