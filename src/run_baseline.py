"""Baseline: semua algoritma pada data bersih. Pakai: python run_baseline.py ../data ../results/baseline.csv"""
import sys
from pipeline import run_all

df = run_all(sys.argv[1], sys.argv[2], scenarios={"clean": [0]}, seeds=(0, 1, 2))
print(df.groupby(["supervisi", "algoritma"])[["auc_roc", "auc_pr"]].mean().round(3))
