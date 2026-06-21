"""
models/deep_learning_model.py
==============================
Multi-Layer Perceptron (MLP) built with TensorFlow / Keras.
Architecture: Input → 256 → BN → Dropout → 128 → BN → Dropout → 64 → Output

Always outputs per-class probabilities (softmax) for multiclass mode.
"""
from __future__ import annotations

import os
import time

import numpy as np

import config
from models.base_model import BaseIDSModel


class DeepLearningIDS(BaseIDSModel):
    name = "DeepLearning"

    def __init__(self, input_dim: int = None, n_classes: int = 2):
        self.input_dim  = input_dim
        self.n_classes  = n_classes
        self.model      = None
        self.history    = None
        self.train_time: float = 0.0

    def _build(self):
        import tensorflow as tf
        from tensorflow.keras import layers, regularizers

        lr     = config.DL_PARAMS["learning_rate"]
        drop   = config.DL_PARAMS["dropout_rate"]
        l2_reg = config.DL_PARAMS["l2_reg"]
        reg    = regularizers.l2(l2_reg)

        inp = tf.keras.Input(shape=(self.input_dim,), name="input")
        x   = layers.Dense(256, activation="relu", kernel_regularizer=reg)(inp)
        x   = layers.BatchNormalization()(x)
        x   = layers.Dropout(drop)(x)
        x   = layers.Dense(128, activation="relu", kernel_regularizer=reg)(x)
        x   = layers.BatchNormalization()(x)
        x   = layers.Dropout(drop)(x)
        x   = layers.Dense(64, activation="relu")(x)
        x   = layers.Dropout(drop / 2)(x)

        # Always softmax — we need per-attack-type probabilities
        out  = layers.Dense(self.n_classes, activation="softmax", name="output")(x)
        loss = "sparse_categorical_crossentropy"

        model = tf.keras.Model(inp, out)
        model.compile(
            optimizer = tf.keras.optimizers.Adam(learning_rate=lr),
            loss      = loss,
            metrics   = ["accuracy"],
        )
        return model

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray = None,
        y_val: np.ndarray = None,
        **kwargs,
    ) -> None:
        import tensorflow as tf

        self.input_dim = X_train.shape[1]
        self.model     = self._build()
        print(f"  ↳ Fitting {self.name} "
              f"({self.model.count_params():,} parameters)…")

        callbacks = [
            tf.keras.callbacks.EarlyStopping(
                monitor="val_loss", patience=config.DL_PARAMS["patience"],
                restore_best_weights=True, verbose=0,
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor="val_loss", factor=0.5, patience=3, verbose=0,
            ),
        ]

        val_data = (X_val, y_val) if X_val is not None else None
        t0       = time.time()
        self.history = self.model.fit(
            X_train, y_train,
            validation_data = val_data,
            epochs          = config.DL_PARAMS["epochs"],
            batch_size      = config.DL_PARAMS["batch_size"],
            callbacks       = callbacks,
            verbose         = 1,
        )
        self.train_time = time.time() - t0
        epochs_run = len(self.history.history["loss"])
        print(f"    Done in {self.train_time:.1f}s  ({epochs_run} epochs)")

    def predict(self, X: np.ndarray) -> np.ndarray:
        proba = self.model.predict(X, verbose=0)
        return np.argmax(proba, axis=1)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(X, verbose=0)

    def save(self, directory: str) -> str:
        import joblib
        os.makedirs(directory, exist_ok=True)
        keras_path = os.path.join(directory, f"{self.name}.keras")
        self.model.save(keras_path)
        meta = {
            "input_dim": self.input_dim,
            "n_classes": self.n_classes,
            "train_time": self.train_time,
        }
        joblib.dump(meta, os.path.join(directory, f"{self.name}_meta.pkl"))
        print(f"  Model saved → {keras_path}")
        return keras_path

    @classmethod
    def load(cls, directory: str) -> "DeepLearningIDS":
        import joblib
        import tensorflow as tf
        meta    = joblib.load(os.path.join(directory, "DeepLearning_meta.pkl"))
        wrapper = cls(input_dim=meta["input_dim"], n_classes=meta["n_classes"])
        wrapper.train_time = meta["train_time"]
        wrapper.model = tf.keras.models.load_model(
            os.path.join(directory, "DeepLearning.keras")
        )
        return wrapper
