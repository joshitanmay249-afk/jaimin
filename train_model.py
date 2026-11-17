"""
train_model.py
--------------
Downloads the MNIST dataset and trains a neural-network
(MLPClassifier) on 28x28 pixel images.

Run this ONCE to generate model_mnist_28x28.joblib.
The file is already pre-built for you, but you can re-run
this script to retrain from scratch.
"""

import urllib.request
import gzip
import os
import time
import numpy as np
import joblib
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report

# ── URLs for the raw MNIST binary files (Google mirror) ──────────────────────
BASE_URL = "https://storage.googleapis.com/cvdf-datasets/mnist/"
FILES = {
    "train_images": "train-images-idx3-ubyte.gz",
    "train_labels": "train-labels-idx1-ubyte.gz",
    "test_images":  "t10k-images-idx3-ubyte.gz",
    "test_labels":  "t10k-labels-idx1-ubyte.gz",
}
MODEL_OUT = "model_mnist_28x28.joblib"


def download_if_missing():
    for key, fname in FILES.items():
        if not os.path.exists(fname):
            print(f"Downloading {fname} …")
            urllib.request.urlretrieve(BASE_URL + fname, fname)
            print(f"  Saved → {fname}")
        else:
            print(f"  {fname} already present, skipping download.")


def load_images(path: str) -> np.ndarray:
    with gzip.open(path, "rb") as f:
        data = np.frombuffer(f.read(), dtype=np.uint8, offset=16)
    return (data.reshape(-1, 28 * 28) / 255.0).astype(np.float32)


def load_labels(path: str) -> np.ndarray:
    with gzip.open(path, "rb") as f:
        return np.frombuffer(f.read(), dtype=np.uint8, offset=8)


def train():
    download_if_missing()

    print("\nLoading dataset …")
    X_train = load_images(FILES["train_images"])
    y_train = load_labels(FILES["train_labels"])
    X_test  = load_images(FILES["test_images"])
    y_test  = load_labels(FILES["test_labels"])
    print(f"  Train: {X_train.shape}  |  Test: {X_test.shape}")

    print("\nTraining MLPClassifier (hidden layers: 128 → 64) …")
    t0 = time.time()
    model = MLPClassifier(
        hidden_layer_sizes=(128, 64),
        activation="relu",
        solver="adam",
        alpha=1e-4,
        max_iter=25,
        random_state=42,
        verbose=True,
    )
    model.fit(X_train, y_train)
    elapsed = time.time() - t0

    y_pred = model.predict(X_test)
    acc    = accuracy_score(y_test, y_pred)
    print(f"\nTraining time : {elapsed:.1f} s")
    print(f"Test accuracy : {acc * 100:.2f} %\n")
    print(classification_report(y_test, y_pred))

    joblib.dump(model, MODEL_OUT, compress=3)
    print(f"Model saved → {MODEL_OUT}")


if __name__ == "__main__":
    train()
