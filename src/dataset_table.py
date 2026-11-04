"""Hasilkan tabel ringkasan dataset (untuk Tabel di Bab 5.2)."""
import glob, os, sys
import numpy as np
import pandas as pd

data_dir = sys.argv[1] if len(sys.argv) > 1 else "data"
rows = []
for path in sorted(glob.glob(os.path.join(data_dir, "*.npz"))):
    d = np.load(path, allow_pickle=True)
    X, y = d["X"], d["y"]
    rows.append({"Dataset": os.path.basename(path)[:-4], "Sampel": X.shape[0], "Fitur": X.shape[1],
                 "Anomali": int(y.sum()), "Rasio anomali (%)": round(100 * float(np.mean(y)), 2)})
df = pd.DataFrame(rows)
print(df.to_string(index=False))
out_dir = os.path.join(os.path.dirname(os.path.abspath(data_dir)), "results")
os.makedirs(out_dir, exist_ok=True)
df.to_csv(os.path.join(out_dir, "dataset_summary.csv"), index=False)
