"""
Evaluate the 5 Gemini decision tree functions on the Credit Score dataset.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'Models with all 5 functions'))

import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
from collections import Counter
from .creditscore import dt_function_0, dt_function_1, dt_function_2, dt_function_3, dt_function_4

OWN_HOME_MAP = {1: "yes", 0: "no"}
SELF_EMP_MAP = {1: "yes", 0: "no"}

# Output → binary
ACCEPT_MAP = {"Accepted": 1, "accepted": 1, "Rejected": 0, "rejected": 0, "Denied": 0}


def run_model_0(row):
    pred, _ = dt_function_0(
        row["Age"], row["Income.per.dependent"], row["Monthly.credit.card.exp"],
        OWN_HOME_MAP[int(row["Own.home"])], SELF_EMP_MAP[int(row["Self.employed"])],
        int(row["Derogatory.reports"])
    )
    return ACCEPT_MAP[pred]

def run_model_1(row):
    features = {
        'age (years)': row["Age"],
        'income per dependent (1.5 to 10)': row["Income.per.dependent"],
        'monthly credit card expenses ($)': row["Monthly.credit.card.exp"],
        'owning a home (yes / no)': OWN_HOME_MAP[int(row["Own.home"])],
        'self employed (yes / no)': SELF_EMP_MAP[int(row["Self.employed"])],
        'number of derogatory reports': int(row["Derogatory.reports"]),
    }
    pred, _ = dt_function_1(features)
    return ACCEPT_MAP[pred]

def run_model_2(row):
    pred, _ = dt_function_2(
        row["Age"], row["Income.per.dependent"], row["Monthly.credit.card.exp"],
        OWN_HOME_MAP[int(row["Own.home"])], SELF_EMP_MAP[int(row["Self.employed"])],
        int(row["Derogatory.reports"])
    )
    return ACCEPT_MAP[pred]

def run_model_3(row):
    pred, _ = dt_function_3(
        row["Age"], row["Income.per.dependent"], row["Monthly.credit.card.exp"],
        OWN_HOME_MAP[int(row["Own.home"])], SELF_EMP_MAP[int(row["Self.employed"])],
        int(row["Derogatory.reports"])
    )
    return ACCEPT_MAP[pred]

def run_model_4(row):
    pred, _ = dt_function_4(
        row["Age"], row["Income.per.dependent"], row["Monthly.credit.card.exp"],
        OWN_HOME_MAP[int(row["Own.home"])], SELF_EMP_MAP[int(row["Self.employed"])],
        int(row["Derogatory.reports"])
    )
    return ACCEPT_MAP[pred]


def evaluate():
    X = pd.read_csv("./data/data_sets/creditscore/X.csv")
    y = pd.read_csv("./data/data_sets/creditscore/y.csv")["target"].values

    runners = [run_model_0, run_model_1, run_model_2, run_model_3, run_model_4]
    all_preds = {}

    for i, runner in enumerate(runners):
        preds = [runner(row) for _, row in X.iterrows()]
        all_preds[f"Model_{i}"] = preds

    print("=" * 70)
    print("CREDIT SCORE DATASET - Zero-Shot Decision Tree Evaluation")
    print("=" * 70)
    print(f"Dataset size: {len(y)} | Accepted(1)={sum(y==1)}, Rejected(0)={sum(y==0)}\n")

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
    print("\n" + classification_report(y, ens, target_names=["Rejected", "Accepted"]))


if __name__ == "__main__":
    evaluate()
