"""
Evaluate the 5 Gemini decision tree functions on the Bankruptcy dataset.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'Models with all 5 functions'))

import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
from collections import Counter
from .bankruptcy import dt_function_0, dt_function_1, dt_function_2, dt_function_3, dt_function_4

# Output label mapping → binary target
def to_binary(pred):
    if isinstance(pred, (int, float)):
        return int(pred)
    s = str(pred).lower()
    if "non" in s or "not" in s:
        return 0
    return 1  # "Bankrupt", "BANKRUPT", etc.


def run_model_0(row):
    pred, _ = dt_function_0(row["WC/TA"], row["RE/TA"], row["EBIT/TA"], row["S/TA"], row["BVE/BVL"])
    return to_binary(pred)

def run_model_1(row):
    features = {
        'Working Capital/Total Assets': row["WC/TA"],
        'Retained Earnings/Total Assets': row["RE/TA"],
        'Earnings Before Interest and Tax/Total Assets': row["EBIT/TA"],
        'Sales/Total Assets': row["S/TA"],
        'Book Value of Equity/Book Value of Liabilities': row["BVE/BVL"],
    }
    pred, _ = dt_function_1(features)
    return to_binary(pred)

def run_model_2(row):
    pred, _ = dt_function_2(row["WC/TA"], row["RE/TA"], row["EBIT/TA"], row["S/TA"], row["BVE/BVL"])
    return to_binary(pred)

def run_model_3(row):
    pred, _ = dt_function_3(row["WC/TA"], row["RE/TA"], row["EBIT/TA"], row["S/TA"], row["BVE/BVL"])
    return to_binary(pred)

def run_model_4(row):
    pred, _ = dt_function_4(row["WC/TA"], row["RE/TA"], row["EBIT/TA"], row["S/TA"], row["BVE/BVL"])
    return to_binary(pred)


def evaluate():
    X = pd.read_csv("./data/data_sets/bankruptcy/X.csv")
    y = pd.read_csv("./data/data_sets/bankruptcy/y.csv")["target"].values

    runners = [run_model_0, run_model_1, run_model_2, run_model_3, run_model_4]
    all_preds = {}

    for i, runner in enumerate(runners):
        preds = [runner(row) for _, row in X.iterrows()]
        all_preds[f"Model_{i}"] = preds

    print("=" * 70)
    print("BANKRUPTCY DATASET - Zero-Shot Decision Tree Evaluation")
    print("=" * 70)
    print(f"Dataset size: {len(y)} | Bankrupt(1)={sum(y)}, Non-Bankrupt(0)={len(y)-sum(y)}\n")

    for name, preds in all_preds.items():
        acc = accuracy_score(y, preds)
        prec = precision_score(y, preds, zero_division=0)
        rec = recall_score(y, preds, zero_division=0)
        f1 = f1_score(y, preds, zero_division=0)
        print(f"  {name}: Accuracy={acc:.4f}  Precision={prec:.4f}  Recall={rec:.4f}  F1={f1:.4f}")

    # Ensemble
    ens = [Counter([all_preds[f"Model_{i}"][j] for i in range(5)]).most_common(1)[0][0] for j in range(len(y))]
    acc = accuracy_score(y, ens)
    prec = precision_score(y, ens, zero_division=0)
    rec = recall_score(y, ens, zero_division=0)
    f1 = f1_score(y, ens, zero_division=0)
    print(f"\n  Ensemble (Majority Voting): Accuracy={acc:.4f}  Precision={prec:.4f}  Recall={rec:.4f}  F1={f1:.4f}")
    print("\n" + classification_report(y, ens, target_names=["Non-Bankrupt", "Bankrupt"]))


if __name__ == "__main__":
    evaluate()
