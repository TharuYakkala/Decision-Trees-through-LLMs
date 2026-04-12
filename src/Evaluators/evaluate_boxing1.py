"""
Evaluate the 5 Gemini decision tree functions on the Boxing1 (Lewis vs Holyfield) dataset.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'Models with all 5 functions'))

import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
from collections import Counter
from .boxing1 import dt_function_0, dt_function_1, dt_function_2, dt_function_3, dt_function_4

JUDGE_MAP = {
    0: "E. Williams", 1: "L. O'Connell", 2: "S. Christodoulu", 3: "HBO-Lederman",
    4: "Sportsticker", 5: "Boxing Times", 6: "Sportsline", 7: "Associated Press",
    8: "ESPN", 9: "Boxing Monthly-Leach"
}
OFFICIAL_MAP = {1: "yes", 0: "no"}
TARGET_MAP = {"Lewis": 1, "Holyfield": 0}


def prep(row):
    return JUDGE_MAP[int(row["Judge"])], OFFICIAL_MAP[int(row["Official"])], int(row["Round"])


def run_model_0(row):
    j, o, r = prep(row)
    pred, _ = dt_function_0(j, o, r)
    return TARGET_MAP[pred]

def run_model_1(row):
    j, o, r = prep(row)
    pred, _ = dt_function_1(j, o, r)
    return TARGET_MAP[pred]

def run_model_2(row):
    j, o, r = prep(row)
    pred, _ = dt_function_2(j, o, r)
    return TARGET_MAP[pred]

def run_model_3(row):
    j, o, r = prep(row)
    pred, _ = dt_function_3(j, o, r)
    return TARGET_MAP[pred]

def run_model_4(row):
    j, o, r = prep(row)
    pred, _ = dt_function_4(j, o, r)
    return TARGET_MAP[pred]


def evaluate():
    X = pd.read_csv("./data/data_sets/boxing1/X.csv")
    y = pd.read_csv("./data/data_sets/boxing1/y.csv")["target"].values

    runners = [run_model_0, run_model_1, run_model_2, run_model_3, run_model_4]
    all_preds = {}

    for i, runner in enumerate(runners):
        preds = [runner(row) for _, row in X.iterrows()]
        all_preds[f"Model_{i}"] = preds

    print("=" * 70)
    print("BOXING1 (Lewis vs Holyfield) - Zero-Shot Decision Tree Evaluation")
    print("=" * 70)
    print(f"Dataset size: {len(y)} | Lewis(1)={sum(y==1)}, Holyfield(0)={sum(y==0)}\n")

    for name, preds in all_preds.items():
        acc = accuracy_score(y, preds)
        prec = precision_score(y, preds, zero_division=0)
        rec = recall_score(y, preds, zero_division=0)
        f1 = f1_score(y, preds, zero_division=0)
        print(f"  {name}: Accuracy={acc:.4f}  Precision={prec:.4f}  Recall={rec:.4f}  F1={f1:.4f}")

    ens = [Counter([all_preds[f"Model_{i}"][j] for i in range(5)]).most_common(1)[0][0] for j in range(len(y))]
    acc = accuracy_score(y, ens)
    prec = precision_score(y, ens, zero_division=0)
    rec = recall_score(y, ens, zero_division=0)
    f1 = f1_score(y, ens, zero_division=0)
    print(f"\n  Ensemble (Majority Voting): Accuracy={acc:.4f}  Precision={prec:.4f}  Recall={rec:.4f}  F1={f1:.4f}")
    print("\n" + classification_report(y, ens, target_names=["Holyfield", "Lewis"], zero_division=0))


if __name__ == "__main__":
    evaluate()
