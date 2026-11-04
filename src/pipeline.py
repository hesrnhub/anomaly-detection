"""Pipeline: muat -> split -> injeksi gangguan -> latih -> evaluasi -> catat CSV."""
import os
import time
import warnings
warnings.filterwarnings("ignore")
import itertools
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, average_precision_score

from generator import duplicate_anomalies, add_irrelevant_features, flip_labels, mask_labels
from models import MODELS, score

LABEL_RATIO = 0.10  # proporsi anomali berlabel pada semi-supervised (samakan dengan paper/repo)

# level gangguan per skenario (cek ulang dengan data_generator.py ADBench)
SCENARIOS = {
    "clean": [0],
    "duplicated_anomalies": [2, 3, 4, 5, 6],
    "irrelevant_features": [0.01, 0.05, 0.10, 0.25, 0.50],
    "annotation_errors": [0.05, 0.10, 0.25, 0.40, 0.50],
}


def load_dataset(path):
    d = np.load(path, allow_pickle=True)
    return d["X"].astype(float), d["y"].astype(int)


def run_one(X, y, algo, scenario, level, seed):
    rng = np.random.default_rng(seed)
    supervision, make = MODELS[algo]

    # annotation errors hanya relevan untuk metode berlabel
    if scenario == "annotation_errors" and supervision == "unsupervised":
        return None

    # 1) split stratified 70/30 lebih dulu (mencegah kebocoran data)
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3, stratify=y, random_state=seed)

    # 2) injeksi gangguan
    if scenario == "duplicated_anomalies":
        X_tr, y_tr = duplicate_anomalies(X_tr, y_tr, level)
        X_te, y_te = duplicate_anomalies(X_te, y_te, level)
    elif scenario == "irrelevant_features":
        X_tr, X_te = add_irrelevant_features(X_tr, X_te, level, rng)
    elif scenario == "annotation_errors":
        y_tr = flip_labels(y_tr, level, rng)

    # 3) susun label latih sesuai tingkat supervisi
    if supervision == "semi-supervised":
        y_fit = mask_labels(y_tr, LABEL_RATIO, rng)
    else:
        y_fit = y_tr

    # 4) latih
    model = make(seed)
    t0 = time.time()
    if supervision == "unsupervised":
        model.fit(X_tr)
    else:
        model.fit(X_tr, y_fit)
    t_train = time.time() - t0

    # 5) evaluasi di data uji
    s = score(model, algo, X_te)
    return {
        "algoritma": algo, "supervisi": supervision, "skenario": scenario, "level": level,
        "seed": seed, "auc_roc": roc_auc_score(y_te, s), "auc_pr": average_precision_score(y_te, s),
        "waktu_latih_detik": round(t_train, 4),
    }


def run_all(data_dir, out_csv, algos=None, seeds=(0, 1, 2), scenarios=None):
    algos = algos or list(MODELS)
    scenarios = scenarios or SCENARIOS
    rows = []
    for path in sorted(os.listdir(data_dir)):
        if not path.endswith(".npz"):
            continue
        name = path[:-4]
        X, y = load_dataset(os.path.join(data_dir, path))
        for (sc, levels), algo, seed in itertools.product(scenarios.items(), algos, seeds):
            for lv in levels:
                try:
                    r = run_one(X, y, algo, sc, lv, seed)
                except Exception as e:  # catat dan lanjut
                    r = {"algoritma": algo, "skenario": sc, "level": lv, "seed": seed, "error": str(e)}
                if r is not None:
                    r["dataset"] = name
                    rows.append(r)
        pd.DataFrame(rows).to_csv(out_csv, index=False)  # simpan inkremental
    return pd.DataFrame(rows)


if __name__ == "__main__":
    import sys
    run_all(sys.argv[1], sys.argv[2])
