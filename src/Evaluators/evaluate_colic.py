"""
Evaluate the 5 Gemini decision tree functions on the Colic (Horse) dataset.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'Models with all 5 functions'))

import pandas as pd
import numpy as np
from sklearn.metrics import classification_report
from collections import Counter
from sklearn.model_selection import train_test_split
from .Embeddings.emb_generator_colic import get_emb_features_colic
from .Embeddings.embedding_eval import rt_embeddings, et_embeddings, rf_embeddings, xgb_embeddings
from .model_extractors.colic_models import run_model_0, run_model_1, run_model_2, run_model_3, run_model_4
from .emb_eval_looper import get_metrics, ml_looper
from sklearn.impute import SimpleImputer

import warnings
warnings.filterwarnings("ignore")

def evaluate():
    X = pd.read_csv("./data/data_sets/colic/X.csv")
    y = pd.read_csv("./data/data_sets/colic/y.csv")["target"].values
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, test_size=0.3)
    runners = [run_model_0, run_model_1, run_model_2, run_model_3, run_model_4]
    all_preds = {}
    
    # Colic dataset has missing data in X, so we need to use an imputer
    imputer = SimpleImputer(strategy="most_frequent")
    X_train = pd.DataFrame(imputer.fit_transform(X_train), columns=X.columns)
    X_test = pd.DataFrame(imputer.transform(X_test), columns=X.columns)
    
    for i, runner in enumerate(runners):
        preds = [runner(row)[0] for _, row in X_test.iterrows()]
        all_preds[f"Model_{i}"] = preds

    print("=" * 70)
    print("COLIC (Horse) DATASET - Zero-Shot Decision Tree Evaluation")
    print("=" * 70)
    print(f"Dataset size: {len(y)} | Surgical(1)={sum(y==1)}, Non-Surgical(0)={sum(y==0)}\n")

    for name, preds in all_preds.items():
        acc, prec, rec, f1 = get_metrics(y_test, preds)
        print(f"  {name}: Accuracy={acc:.4f}  Precision={prec:.4f}  Recall={rec:.4f}  F1={f1:.4f}")

    ens = [Counter([all_preds[f"Model_{i}"][j] for i in range(5)]).most_common(1)[0][0] for j in range(len(y_test))]
    acc, prec, rec, f1 = get_metrics(ens, y_test)
    print(f"\n  Ensemble (Majority Voting): Accuracy={acc:.4f}  Precision={prec:.4f}  Recall={rec:.4f}  F1={f1:.4f}")
    print("\n" + classification_report(y_test, ens, target_names=["Non-Surgical", "Surgical"]))

    ### Get all embeddings
    llm_train_emb = pd.DataFrame(get_emb_features_colic(X_train))
    llm_train_emb['target'] = y_train
    llm_test_emb = pd.DataFrame(get_emb_features_colic(X_test))
    baseline_train = X_train.copy()
    baseline_train['target'] = y_train
    
    all_embeddings = {
        "baseline": (baseline_train, X_test),
        "llm_emb": (llm_train_emb,llm_test_emb),
        "rt_emb": rt_embeddings(X_train, y_train, X_test),
        'et_emb_ss': et_embeddings(X_train, y_train, X_test, self_supervised=True),
        'et_emb_s': et_embeddings(X_train, y_train, X_test, self_supervised=False),
        'rf_emb_ss': rf_embeddings(X_train, y_train, X_test, self_supervied=True),
        'rf_emb_s': rf_embeddings(X_train, y_train, X_test, self_supervied=False),
        'xgb_emb_ss': xgb_embeddings(X_train, y_train, X_test, self_supervised=True),
        'xgb_emb_s': xgb_embeddings(X_train, y_train, X_test, self_supervised=False)
    }
    
    emb_df = ml_looper(embeddings=all_embeddings, y_test=y_test)
    return emb_df

if __name__ == "__main__":
    evaluate()
