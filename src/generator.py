"""Data generator: tiga skenario gangguan kualitas data (replikasi ADBench Angle III)."""
import numpy as np


def duplicate_anomalies(X, y, times):
    """Duplikasi anomali sehingga tiap anomali muncul `times` kali (times=1 -> tanpa perubahan)."""
    if times < 1:
        raise ValueError("times harus >= 1")
    idx = np.where(y == 1)[0]
    if times == 1 or len(idx) == 0:
        return X.copy(), y.copy()
    X_new = np.concatenate([X] + [X[idx]] * (times - 1), axis=0)
    y_new = np.concatenate([y] + [y[idx]] * (times - 1), axis=0)
    return X_new, y_new


def add_irrelevant_features(X_train, X_test, ratio, rng):
    """Tambah fitur noise sebanyak round(ratio * d) kolom.

    Setiap kolom noise ~ Uniform(min(x_k), max(x_k)) dari kolom acak k pada data latih.
    Kolom k dan rentangnya sama untuk data latih dan data uji agar dimensi konsisten.
    """
    if ratio <= 0:
        return X_train.copy(), X_test.copy()
    d = X_train.shape[1]
    n_noise = max(1, int(round(d * ratio)))
    cols = rng.integers(0, d, size=n_noise)
    lo = X_train[:, cols].min(axis=0)
    hi = X_train[:, cols].max(axis=0)

    def make(X):
        noise = rng.uniform(lo, hi, size=(X.shape[0], n_noise))
        return np.hstack([X, noise])

    return make(X_train), make(X_test)


def flip_labels(y_train, rate, rng):
    """Balik label (0<->1) pada proporsi `rate` dari data latih. Label uji tidak diubah."""
    y_new = y_train.copy()
    n_flip = int(round(rate * len(y_train)))
    if n_flip == 0:
        return y_new
    idx = rng.choice(len(y_train), size=n_flip, replace=False)
    y_new[idx] = 1 - y_new[idx]
    return y_new


def mask_labels(y_train, label_ratio, rng):
    """Semi-supervised: hanya `label_ratio` anomali yang tetap berlabel 1, sisanya jadi 0 (tak berlabel)."""
    y_new = y_train.copy()
    idx = np.where(y_train == 1)[0]
    if len(idx) == 0:
        return y_new
    n_keep = max(1, int(round(label_ratio * len(idx))))
    keep = rng.choice(idx, size=min(n_keep, len(idx)), replace=False)
    y_new[idx] = 0
    y_new[keep] = 1
    return y_new
