import os
import glob
import time
import tempfile
import argparse
import numpy as np
import pandas as pd

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["MPLCONFIGDIR"] = os.path.join(tempfile.gettempdir(), "matplotlib_tmp")

# Models are trained one at a time, never concurrently with each other, so
# letting each estimator's own joblib-based parallelism use multiple cores
# is safe here and speeds up fitting. (The OMP/OPENBLAS/MKL thread caps
# above only limit the BLAS layer, which is a separate thread pool, so
# there's no oversubscription conflict with this.)
N_JOBS = max(1, (os.cpu_count() or 2) - 1)

import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.neighbors import NearestNeighbors
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA
from scipy.spatial.distance import cdist

def load_npz_dataset(filepath):
    data = np.load(filepath)
    X = data['X'].astype(np.float64)
    y = data['y'].astype(np.int32)
    return X, y


def split_data(X, y, test_size=0.30, random_state=42):
    return train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state
    )

class BaseDetector:
    def fit(self, X_train, y_train=None):
        raise NotImplementedError

    def predict_score(self, X_test):
        raise NotImplementedError


class ECODDetector(BaseDetector):
    def __init__(self):
        self.sorted_train = None

    def fit(self, X_train, y_train=None):
        # Sort each feature column once here, instead of re-sorting it on
        # every predict_score() call (sorting is O(n log n) per column).
        self.sorted_train = np.sort(X_train, axis=0)
        return self

    def predict_score(self, X_test):
        n_train, n_feats = self.sorted_train.shape
        scores = np.zeros(X_test.shape[0], dtype=np.float64)
        
        eps = 1e-10
        for j in range(n_feats):
            feat_train = self.sorted_train[:, j]
            feat_test = X_test[:, j]
            
            # Left tail ECDF
            left_counts = np.searchsorted(feat_train, feat_test, side='right')
            left_prob = np.clip(left_counts / n_train, eps, 1.0)
            left_score = -np.log(left_prob)
            
            # Right tail ECDF
            right_counts = n_train - np.searchsorted(feat_train, feat_test, side='left')
            right_prob = np.clip(right_counts / n_train, eps, 1.0)
            right_score = -np.log(right_prob)
            
            # Skoring ECOD: akumulasi deviasi ekor
            scores += np.maximum(left_score, right_score)
            
        return scores


class HBOSDetector(BaseDetector):
    def __init__(self, n_bins=10):
        self.n_bins = n_bins
        self.histograms = []
        self.bin_edges = []

    def fit(self, X_train, y_train=None):
        self.histograms = []
        self.bin_edges = []
        n_samples, n_feats = X_train.shape
        eps = 1e-10
        
        for j in range(n_feats):
            col = X_train[:, j]
            hist, edges = np.histogram(col, bins=self.n_bins, density=True)
            hist = np.clip(hist, eps, None)
            self.histograms.append(hist)
            self.bin_edges.append(edges)
        return self

    def predict_score(self, X_test):
        n_samples, n_feats = X_test.shape
        scores = np.zeros(n_samples, dtype=np.float64)
        eps = 1e-10
        
        for j in range(n_feats):
            hist = self.histograms[j]
            edges = self.bin_edges[j]
            col = X_test[:, j]
            
            bin_idx = np.digitize(col, edges) - 1
            bin_idx = np.clip(bin_idx, 0, len(hist) - 1)
            densities = hist[bin_idx]
            densities = np.clip(densities, eps, None)
            scores += -np.log(densities)
        return scores


class KNNDetector(BaseDetector):
    def __init__(self, k=5):
        self.k = k
        self.model = None

    def fit(self, X_train, y_train=None):
        k_val = min(self.k, len(X_train))
        self.model = NearestNeighbors(n_neighbors=k_val, algorithm='auto', n_jobs=N_JOBS)
        self.model.fit(X_train)
        return self

    def predict_score(self, X_test):
        distances, _ = self.model.kneighbors(X_test)
        # Jarak ke tetangga ke-k
        return distances[:, -1]


class PCADetector(BaseDetector):
    def __init__(self, n_components=0.8):
        self.n_components = n_components
        self.pca = None

    def fit(self, X_train, y_train=None):
        n_comp = min(X_train.shape[1], X_train.shape[0])
        self.pca = PCA(n_components=self.n_components if self.n_components < 1.0 else min(int(self.n_components), n_comp))
        self.pca.fit(X_train)
        return self

    def predict_score(self, X_test):
        X_projected = self.pca.inverse_transform(self.pca.transform(X_test))
        recon_error = np.sum((X_test - X_projected) ** 2, axis=1)
        return recon_error


class IForestDetector(BaseDetector):
    def __init__(self, n_estimators=100, random_state=42):
        self.model = IsolationForest(n_estimators=n_estimators, random_state=random_state, n_jobs=N_JOBS)

    def fit(self, X_train, y_train=None):
        self.model.fit(X_train)
        return self

    def predict_score(self, X_test):
        # IsolationForest decision_function bernilai tinggi untuk normal, rendah untuk anomali
        # Dibalik negasinya agar nilai tinggi = lebih anomali
        return -self.model.decision_function(X_test)


class SupervisedRFDetector(BaseDetector):
    def __init__(self, n_estimators=100, random_state=42):
        self.model = RandomForestClassifier(n_estimators=n_estimators, random_state=random_state, class_weight='balanced', n_jobs=N_JOBS)

    def fit(self, X_train, y_train=None):
        if y_train is None or len(np.unique(y_train)) < 2:
            raise ValueError("Supervised detector memerlukan label biner (y_train).")
        self.model.fit(X_train, y_train)
        return self

    def predict_score(self, X_test):
        proba = self.model.predict_proba(X_test)
        if proba.shape[1] == 2:
            return proba[:, 1]
        return proba[:, 0]


class SemiSupervisedDetector(BaseDetector):
    def __init__(self, base_unsup=None, gamma_l=0.05, random_state=42):
        self.gamma_l = gamma_l
        self.random_state = random_state
        self.base_unsup = base_unsup if base_unsup is not None else ECODDetector()
        self.known_anomalies = None

    def fit(self, X_train, y_train):
        rng = np.random.default_rng(self.random_state)
        # Ambil sampel partial labeled anomalies
        anomaly_idx = np.where(y_train == 1)[0]
        n_labeled = max(1, int(len(anomaly_idx) * self.gamma_l))
        labeled_idx = rng.choice(anomaly_idx, size=n_labeled, replace=False)
        self.known_anomalies = X_train[labeled_idx]

        # Fit model unsupervised pada data training
        self.base_unsup.fit(X_train)
        return self

    def predict_score(self, X_test):
        unsup_scores = self.base_unsup.predict_score(X_test)
        # Normalisasi skor unsupervised ke [0, 1]
        unsup_norm = (unsup_scores - np.min(unsup_scores)) / (np.ptp(unsup_scores) + 1e-10)
        
        # Jarak ke anomali yang diketahui (makin dekat = makin tinggi kemiripan)
        # cdist uses an optimized BLAS routine instead of materializing a full
        # (n_test, n_labeled, n_feats) temporary array via broadcasting.
        dists = np.min(cdist(X_test, self.known_anomalies), axis=1)
        sim_scores = 1.0 / (1.0 + dists)
        sim_norm = (sim_scores - np.min(sim_scores)) / (np.ptp(sim_scores) + 1e-10)
        
        # Kombinasi bobot (Angle I ADBench)
        return 0.5 * unsup_norm + 0.5 * sim_norm

class DataQualityStressTest:
    @staticmethod
    def inject_duplicated_anomalies(X, y, repeat_factor=2, random_state=42):
        if repeat_factor <= 1:
            return np.copy(X), np.copy(y)
        
        anomaly_mask = (y == 1)
        X_anom = X[anomaly_mask]
        y_anom = y[anomaly_mask]
        
        # Duplikasi anomali sebanyak (repeat_factor - 1) kali
        reps = repeat_factor - 1
        X_dups = np.repeat(X_anom, reps, axis=0)
        y_dups = np.repeat(y_anom, reps, axis=0)
        
        X_corrupted = np.vstack([X, X_dups])
        y_corrupted = np.concatenate([y, y_dups])
        
        # Acak urutan
        rng = np.random.default_rng(random_state)
        perm = rng.permutation(len(y_corrupted))
        return X_corrupted[perm], y_corrupted[perm]

    @staticmethod
    def inject_irrelevant_features(X, noise_ratio=0.10, random_state=42):
        if noise_ratio <= 0.0:
            return np.copy(X)
        
        rng = np.random.default_rng(random_state)
        n_samples, n_feats = X.shape
        n_noisy_feats = max(1, int(round(n_feats * noise_ratio)))

        # Pilih fitur acak sebagai referensi batas min/max untuk tiap kolom noise
        ref_cols = rng.integers(0, n_feats, size=n_noisy_feats)
        col_min = np.min(X[:, ref_cols], axis=0)
        col_max = np.max(X[:, ref_cols], axis=0)
        degenerate = col_min == col_max

        # Bangkitkan semua kolom noise sekaligus (vectorized) alih-alih
        # satu per satu dalam loop Python.
        noise = rng.uniform(0.0, 1.0, size=(n_samples, n_noisy_feats))
        uniform_cols = col_min + noise * (col_max - col_min)
        if np.any(degenerate):
            normal_cols = rng.normal(0.0, 1.0, size=(n_samples, n_noisy_feats)) + col_min
            uniform_cols = np.where(degenerate, normal_cols, uniform_cols)

        X_corrupted = np.hstack([X, uniform_cols])
        return X_corrupted

    @staticmethod
    def inject_annotation_errors(y, flip_ratio=0.10, random_state=42):
        if flip_ratio <= 0.0:
            return np.copy(y)
        
        rng = np.random.default_rng(random_state)
        y_corrupted = np.copy(y)
        n_samples = len(y)
        n_flips = max(1, int(round(n_samples * flip_ratio)))
        flip_indices = rng.choice(n_samples, size=n_flips, replace=False)
        y_corrupted[flip_indices] = 1 - y_corrupted[flip_indices]
        return y_corrupted

class SyntheticAnomalyGenerator:
    @staticmethod
    def generate_synthetic_benchmark(X_normal, anomaly_type="local", anomaly_ratio=0.05, random_state=42):
        rng = np.random.default_rng(random_state)
        n_normal, n_feats = X_normal.shape
        n_anomalies = max(2, int(round(n_normal * anomaly_ratio)))
        
        mean_vec = np.mean(X_normal, axis=0)
        cov_matrix = np.cov(X_normal, rowvar=False) + np.eye(n_feats) * 1e-4

        if anomaly_type == "local":
            # Local: skala matriks kovarians alpha = 5
            alpha = 5.0
            anomalies = rng.multivariate_normal(mean_vec, cov_matrix * alpha, size=n_anomalies)

        elif anomaly_type == "global":
            # Global: Unif(alpha * min(X_k), alpha * max(X_k)) dengan alpha = 1.1
            alpha = 1.1
            mins = np.min(X_normal, axis=0)
            maxs = np.max(X_normal, axis=0)
            lows = np.where(mins < 0, mins * alpha, mins / alpha)
            highs = np.where(maxs > 0, maxs * alpha, maxs / alpha)
            anomalies = rng.uniform(lows, highs, size=(n_anomalies, n_feats))

        elif anomaly_type == "dependency":
            # Dependency: merusak korelasi antar fitur (permutasi independen per kolom)
            sample_idx = rng.choice(n_normal, size=n_anomalies, replace=True)
            subset = X_normal[sample_idx]
            # Independent per-column permutation, vectorized: argsort a matrix
            # of random keys (one random ordering per column) then gather,
            # instead of calling rng.permutation() in a Python loop per column.
            perm_idx = np.argsort(rng.random(subset.shape), axis=0)
            anomalies = np.take_along_axis(subset, perm_idx, axis=0)

        elif anomaly_type == "clustered":
            # Clustered / Group anomalies: pergeseran cluster mean alpha = 5
            alpha = 5.0
            shift_mean = mean_vec * alpha
            # Jika mean mendekati 0, tambahkan offset standar deviasi
            stds = np.std(X_normal, axis=0) + 1e-4
            shift_mean = np.where(np.abs(shift_mean) < 1e-3, mean_vec + alpha * stds, shift_mean)
            anomalies = rng.multivariate_normal(shift_mean, cov_matrix * 0.5, size=n_anomalies)
        else:
            raise ValueError(f"Tipe anomali tidak dikenal: {anomaly_type}")

        # Gabungkan data normal dan sintetis
        X_all = np.vstack([X_normal, anomalies])
        y_all = np.concatenate([np.zeros(n_normal, dtype=np.int32), np.ones(n_anomalies, dtype=np.int32)])
        
        perm = rng.permutation(len(y_all))
        return X_all[perm], y_all[perm]

def evaluate_predictions(y_true, scores):
    try:
        auc_roc = roc_auc_score(y_true, scores)
    except Exception:
        auc_roc = 0.5
    try:
        auc_pr = average_precision_score(y_true, scores)
    except Exception:
        auc_pr = np.mean(y_true)
    return auc_roc, auc_pr


def get_default_models():
    return {
        # Unsupervised
        "IForest": IForestDetector(n_estimators=100, random_state=42),
        "KNN": KNNDetector(k=5),
        "ECOD": ECODDetector(),
        "HBOS": HBOSDetector(n_bins=10),
        "PCA": PCADetector(n_components=0.8),
        # Semi-Supervised (5% labeled anomalies)
        "SemiSup_5%": SemiSupervisedDetector(gamma_l=0.05, random_state=42),
        # Supervised (Binary classification)
        "RandomForest": SupervisedRFDetector(n_estimators=100, random_state=42)
    }


def run_clean_baseline(dataset_paths, models=None):
    if models is None:
        models = get_default_models()

    results = []
    print("=" * 80)
    print("1. EVALUASI BASELINE PADA DATA BERSIH (CLEAN DATA)")
    print("=" * 80)

    for path in dataset_paths:
        name = os.path.basename(path).replace(".npz", "")
        X, y = load_npz_dataset(path)
        X_train, X_test, y_train, y_test = split_data(X, y)

        # Standardisasi fitur
        scaler = StandardScaler()
        X_train_sc = scaler.fit_transform(X_train)
        X_test_sc = scaler.transform(X_test)

        print(f"\n[Dataset: {name}] N_train={len(X_train)}, N_test={len(X_test)}, D={X.shape[1]}, Anomali={np.sum(y==1)} ({np.mean(y==1)*100:.1f}%)")
        
        for m_name, model in models.items():
            t0 = time.time()
            if "SemiSup" in m_name or "RandomForest" in m_name:
                model.fit(X_train_sc, y_train)
            else:
                model.fit(X_train_sc)
            train_time = time.time() - t0

            scores = model.predict_score(X_test_sc)
            auc_roc, auc_pr = evaluate_predictions(y_test, scores)
            
            results.append({
                "Dataset": name,
                "Model": m_name,
                "AUC-ROC": round(auc_roc * 100, 2),
                "AUC-PR": round(auc_pr * 100, 2),
                "TrainTime_s": round(train_time, 4)
            })
            print(f"  - {m_name:<14} -> AUC-ROC: {auc_roc*100:6.2f}% | AUC-PR: {auc_pr*100:6.2f}% | Time: {train_time:.3f}s")

    df_res = pd.DataFrame(results)
    return df_res


def run_data_quality_stress_test(dataset_path, models=None):
    if models is None:
        models = get_default_models()

    name = os.path.basename(dataset_path).replace(".npz", "")
    X, y = load_npz_dataset(dataset_path)
    X_train, X_test, y_train, y_test = split_data(X, y)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    print("\n" + "=" * 80)
    print(f"2. UJI KUALITAS DATA & ROBUSTNESS (ANGLE III) PADA DATASET: {name}")
    print("=" * 80)

    records = []

    print("\n--- [A] Uji Duplikasi Anomali (Anomaly Masking k=1..6) ---")
    for k in [1, 2, 4, 6]:
        X_tr_c, y_tr_c = DataQualityStressTest.inject_duplicated_anomalies(X_train, y_train, repeat_factor=k)
        X_te_c, y_te_c = DataQualityStressTest.inject_duplicated_anomalies(X_test, y_test, repeat_factor=k)

        for m_name, model in models.items():
            if "SemiSup" in m_name or "RandomForest" in m_name:
                model.fit(X_tr_c, y_tr_c)
            else:
                model.fit(X_tr_c)
            scores = model.predict_score(X_te_c)
            auc_roc, _ = evaluate_predictions(y_te_c, scores)
            records.append({
                "Test_Type": "Duplicated Anomalies",
                "Corruption_Level": f"x{k}",
                "Model": m_name,
                "AUC-ROC": auc_roc * 100
            })

    print("--- [B] Uji Fitur Irrelevan / Noise Dimensi (0%..50%) ---")
    for ratio in [0.0, 0.10, 0.25, 0.50]:
        X_tr_c = DataQualityStressTest.inject_irrelevant_features(X_train, noise_ratio=ratio)
        X_te_c = DataQualityStressTest.inject_irrelevant_features(X_test, noise_ratio=ratio)

        for m_name, model in models.items():
            if "SemiSup" in m_name or "RandomForest" in m_name:
                model.fit(X_tr_c, y_train)
            else:
                model.fit(X_tr_c)
            scores = model.predict_score(X_te_c)
            auc_roc, _ = evaluate_predictions(y_test, scores)
            records.append({
                "Test_Type": "Irrelevant Features",
                "Corruption_Level": f"{int(ratio*100)}%",
                "Model": m_name,
                "AUC-ROC": auc_roc * 100
            })

    print("--- [C] Uji Kesalahan Anotasi / Label Noise (0%..20%) ---")
    for flip_r in [0.0, 0.05, 0.10, 0.20]:
        y_tr_c = DataQualityStressTest.inject_annotation_errors(y_train, flip_ratio=flip_r)

        for m_name, model in models.items():
            if "SemiSup" in m_name or "RandomForest" in m_name:
                model.fit(X_train, y_tr_c)
            else:
                model.fit(X_train)
            scores = model.predict_score(X_test)
            auc_roc, _ = evaluate_predictions(y_test, scores)
            records.append({
                "Test_Type": "Annotation Errors",
                "Corruption_Level": f"{int(flip_r*100)}%",
                "Model": m_name,
                "AUC-ROC": auc_roc * 100
            })

    df_stress = pd.DataFrame(records)
    return df_stress


def plot_and_save_stress_results(df_stress, output_png="adbench_robustness_results.png"):
    test_types = df_stress['Test_Type'].unique()
    fig, axes = plt.subplots(1, len(test_types), figsize=(16, 4.5), sharey=True)

    colors = {
        "IForest": "#1f77b4",
        "KNN": "#ff7f0e",
        "ECOD": "#2ca02c",
        "HBOS": "#d62728",
        "PCA": "#9467bd",
        "SemiSup_5%": "#8c564b",
        "RandomForest": "#e377c2"
    }

    for i, t_type in enumerate(test_types):
        ax = axes[i]
        sub_df = df_stress[df_stress['Test_Type'] == t_type]
        for model_name in sub_df['Model'].unique():
            m_data = sub_df[sub_df['Model'] == model_name]
            ax.plot(
                m_data['Corruption_Level'],
                m_data['AUC-ROC'],
                marker='o',
                linewidth=2,
                label=model_name,
                color=colors.get(model_name, None)
            )
        ax.set_title(f"Robustness: {t_type}", fontsize=12, fontweight='bold')
        ax.set_xlabel("Tingkat Gangguan (Corruption Level)", fontsize=10)
        if i == 0:
            ax.set_ylabel("AUC-ROC (%)", fontsize=10)
        ax.grid(True, linestyle="--", alpha=0.6)

    axes[-1].legend(bbox_to_anchor=(1.05, 1), loc='upper left', frameon=True)
    plt.tight_layout()
    plt.savefig(output_png, dpi=300)
    plt.close()
    print(f"\n[Visualisasi disimpan ke: {output_png}]")


def run_synthetic_anomaly_types(dataset_path, models=None):
    if models is None:
        models = get_default_models()

    name = os.path.basename(dataset_path).replace(".npz", "")
    X, y = load_npz_dataset(dataset_path)
    X_normal = X[y == 0]

    print("\n" + "=" * 80)
    print(f"3. UJI TIPE ANOMALI SINTETIS (ANGLE II) DARI DATA NORMAL DATASET: {name}")
    print("=" * 80)

    anomaly_types = ["local", "global", "dependency", "clustered"]
    results = []

    for a_type in anomaly_types:
        X_synth, y_synth = SyntheticAnomalyGenerator.generate_synthetic_benchmark(
            X_normal, anomaly_type=a_type, anomaly_ratio=0.10, random_state=42
        )
        X_tr, X_te, y_tr, y_te = split_data(X_synth, y_synth)

        scaler = StandardScaler()
        X_tr = scaler.fit_transform(X_tr)
        X_te = scaler.transform(X_te)

        for m_name, model in models.items():
            if "SemiSup" in m_name or "RandomForest" in m_name:
                model.fit(X_tr, y_tr)
            else:
                model.fit(X_tr)
            scores = model.predict_score(X_te)
            auc_roc, auc_pr = evaluate_predictions(y_te, scores)
            results.append({
                "Anomaly_Type": a_type.capitalize(),
                "Model": m_name,
                "AUC-ROC": round(auc_roc * 100, 2),
                "AUC-PR": round(auc_pr * 100, 2)
            })

    df_synth = pd.DataFrame(results)
    return df_synth


def plot_and_save_synthetic_results(df_synth, output_png="benchmark_anomaly_types_results.png"):
    pivot = df_synth.pivot(index="Model", columns="Anomaly_Type", values="AUC-ROC")
    ax = pivot.plot(kind="bar", figsize=(12, 5), colormap="viridis", width=0.8)
    plt.title("Performa Deteksi Berdasarkan Tipe Anomali (Angle II ADBench)", fontsize=13, fontweight='bold')
    plt.ylabel("AUC-ROC (%)", fontsize=11)
    plt.xlabel("Model Deteksi", fontsize=11)
    plt.xticks(rotation=30, ha="right")
    plt.ylim(0, 105)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.legend(title="Tipe Anomali", bbox_to_anchor=(1.02, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(output_png, dpi=300)
    plt.close()
    print(f"\n[Visualisasi Anomaly Types disimpan ke: {output_png}]")


def resolve_data_directory(custom_dir=None):
    if custom_dir:
        if os.path.exists(custom_dir):
            return os.path.abspath(custom_dir)
        else:
            print(f"Peringatan: Folder custom '{custom_dir}' tidak ditemukan. Mencoba auto-detect...")

    script_dir = os.path.dirname(os.path.abspath(__file__))
    candidate_paths = [
        os.path.join(script_dir, "dataset"),
        "dataset",
        os.path.join(script_dir, "..", "dataset"),
        os.path.join(os.getcwd(), "dataset"),
        os.path.join(os.getcwd(), "anomaly-detection", "dataset"),
        script_dir,
        os.getcwd()
    ]
    for path in candidate_paths:
        if os.path.exists(path) and len(glob.glob(os.path.join(path, "*.npz"))) > 0:
            return os.path.abspath(path)
            
    # Default fallback
    return os.path.join(script_dir, "dataset")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ADBench: Tabular Anomaly Detection & Data Quality Stress Test")
    parser.add_argument("--data-dir", type=str, default=None, help="Lokasi folder dataset (default: auto-detect folder 'dataset').")
    parser.add_argument("--output-dir", type=str, default=None, help="Folder untuk menyimpan file output CSV & gambar.")
    parser.add_argument("--datasets", nargs="*", default=None, help="Daftar nama file .npz yang ingin diuji.")
    parser.add_argument("--all", action="store_true", help="Jalankan pada semua file .npz di folder dataset.")
    parser.add_argument("--skip-stress", action="store_true", help="Lewati uji stress kualitas data.")
    parser.add_argument("--skip-synthetic", action="store_true", help="Lewati uji tipe anomali sintetis.")
    args = parser.parse_args()

    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.abspath(args.output_dir) if args.output_dir else script_dir
    os.makedirs(output_dir, exist_ok=True)

    data_dir = resolve_data_directory(args.data_dir)
    print(f"Direktori dataset yang digunakan: {data_dir}")

    all_npz = sorted(glob.glob(os.path.join(data_dir, "*.npz")))
    if not all_npz:
        print(f"Error: Tidak ada file .npz yang ditemukan di '{data_dir}'!")
        exit(1)

    print(f"Ditemukan {len(all_npz)} file dataset .npz di dalam folder.")
    
    if args.all:
        selected_datasets = all_npz
    elif args.datasets:
        selected_datasets = []
        for d in args.datasets:
            fname = d if d.endswith(".npz") else f"{d}.npz"
            path_in_data_dir = os.path.join(data_dir, fname)
            if os.path.exists(path_in_data_dir):
                selected_datasets.append(path_in_data_dir)
            elif os.path.exists(d):
                selected_datasets.append(os.path.abspath(d))
            else:
                print(f"Peringatan: File dataset '{d}' tidak ditemukan di '{data_dir}' atau path saat ini.")
        if not selected_datasets:
            print("Tidak ada dataset terpilih yang valid. Menggunakan default...")
            selected_datasets = all_npz[:3]
    else:
        defaults = ["18_Ionosphere.npz", "29_Pima.npz", "4_breastw.npz"]
        selected_datasets = [
            os.path.join(data_dir, d) for d in defaults if os.path.exists(os.path.join(data_dir, d))
        ]
        if not selected_datasets:
            selected_datasets = all_npz[:3]

    print(f"Dataset yang dievaluasi ({len(selected_datasets)}): {[os.path.basename(p) for p in selected_datasets]}")

    # 1. Jalankan Baseline pada data bersih
    df_baseline = run_clean_baseline(selected_datasets)
    baseline_csv = os.path.join(output_dir, "benchmark_baseline_clean_results.csv")
    df_baseline.to_csv(baseline_csv, index=False)
    print(f"\nRingkasan Baseline tersimpan ke: '{baseline_csv}'.")

    # 2. Jalankan Uji Stress Kualitas Data pada dataset pertama
    test_target = selected_datasets[0]
    if not args.skip_stress:
        df_stress = run_data_quality_stress_test(test_target)
        stress_csv = os.path.join(output_dir, "benchmark_data_quality_stress_results.csv")
        df_stress.to_csv(stress_csv, index=False)
        print(f"Ringkasan Uji Stress tersimpan ke: '{stress_csv}'.")
        stress_png = os.path.join(output_dir, "benchmark_robustness_results.png")
        plot_and_save_stress_results(df_stress, stress_png)

    # 3. Jalankan Uji Tipe Anomali Sintetis (Angle II)
    if not args.skip_synthetic:
        df_synth = run_synthetic_anomaly_types(test_target)
        synth_csv = os.path.join(output_dir, "benchmark_synthetic_anomaly_types_results.csv")
        df_synth.to_csv(synth_csv, index=False)
        print(f"Ringkasan Tipe Anomali Sintetis tersimpan ke: '{synth_csv}'.")
        synth_png = os.path.join(output_dir, "benchmark_anomaly_types_results.png")
        plot_and_save_synthetic_results(df_synth, synth_png)
    
    print("\n" + "=" * 80)
    print("SELURUH RANGKAIAN BENCHMARK TELAH SELESAI DENGAN SUKSES!")
    print("=" * 80)
