"""
Evaluate the 5 Gemini decision tree functions on the Boxing2 (Trinidad vs de la Hoya) dataset.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'Models with all 5 functions'))

import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
from collections import Counter
from .boxing2 import dt_function_0, dt_function_1, dt_function_2, dt_function_3, dt_function_4

JUDGE_MAP = {
    0: "G. Hamada", 1: "B. Logist", 2: "J. Roth", 3: "HBO-Lederman",
    4: "Sportsticker", 5: "Los Angeles Times-Kawakami", 6: "USA Today",
    7: "Associated Press", 8: "Las Vegas Review-Journal",
    9: "Los Angeles Times-Springer", 10: "van de Wiele"
}
OFFICIAL_MAP = {1: "yes", 0: "no"}
TARGET_MAP = {"de la Hoya": 1, "Trinidad": 0}


def prep(row):
    return JUDGE_MAP[int(row["Judge"])], OFFICIAL_MAP[int(row["Official"])], int(row["Round"])


def run_model_0(row):
    j, o, r = prep(row)
    pred, _ = dt_function_0(j, o, r)
    return TARGET_MAP[pred]

def run_model_1(row):
    j, o, r = prep(row)
    features = {'Judge': j, 'Offical judge': o, 'Round': r}
    pred, _ = dt_function_1(features)
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
    X = pd.read_csv("./data/data_sets/boxing2/X.csv")
    y = pd.read_csv("./data/data_sets/boxing2/y.csv")["target"].values

    runners = [run_model_0, run_model_1, run_model_2, run_model_3, run_model_4]
    all_preds = {}

    for i, runner in enumerate(runners):
        preds = [runner(row) for _, row in X.iterrows()]
        all_preds[f"Model_{i}"] = preds

    print("=" * 70)
    print("BOXING2 (Trinidad vs de la Hoya) - Zero-Shot Decision Tree Evaluation")
    print("=" * 70)
    print(f"Dataset size: {len(y)} | de la Hoya(1)={sum(y==1)}, Trinidad(0)={sum(y==0)}\n")

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
    print("\n" + classification_report(y, ens, target_names=["Trinidad", "de la Hoya"]))


if __name__ == "__main__":
    evaluate()
