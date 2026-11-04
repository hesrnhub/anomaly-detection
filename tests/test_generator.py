import os, sys
import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from generator import duplicate_anomalies, add_irrelevant_features, flip_labels, mask_labels


@pytest.fixture
def data():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(200, 10))
    y = np.zeros(200, dtype=int)
    y[:20] = 1
    return X, y, rng


def test_duplicate_count(data):
    X, y, _ = data
    X2, y2 = duplicate_anomalies(X, y, 3)
    assert y2.sum() == 3 * y.sum()
    assert (y2 == 0).sum() == (y == 0).sum()
    assert X2.shape[0] == len(y2)


def test_duplicate_times_one_unchanged(data):
    X, y, _ = data
    X2, y2 = duplicate_anomalies(X, y, 1)
    assert np.array_equal(X, X2) and np.array_equal(y, y2)


def test_duplicate_no_side_effect(data):
    X, y, _ = data
    X0, y0 = X.copy(), y.copy()
    duplicate_anomalies(X, y, 6)
    assert np.array_equal(X, X0) and np.array_equal(y, y0)


def test_noise_columns(data):
    X, y, rng = data
    Xtr, Xte = add_irrelevant_features(X[:140], X[140:], 0.5, rng)
    assert Xtr.shape[1] == 10 + 5 and Xte.shape[1] == 15
    assert np.array_equal(Xtr[:, :10], X[:140])


def test_noise_within_range(data):
    X, y, rng = data
    Xtr, Xte = add_irrelevant_features(X[:140], X[140:], 0.5, rng)
    noise = Xtr[:, 10:]
    assert noise.min() >= X[:140].min() and noise.max() <= X[:140].max()


def test_flip_rate(data):
    X, y, rng = data
    y2 = flip_labels(y, 0.10, rng)
    assert (y2 != y).sum() == 20


def test_flip_no_side_effect(data):
    X, y, rng = data
    y0 = y.copy()
    flip_labels(y, 0.5, rng)
    assert np.array_equal(y, y0)


def test_mask_labels(data):
    X, y, rng = data
    y2 = mask_labels(y, 0.10, rng)
    assert y2.sum() == 2
    assert np.all(y2 <= y)
