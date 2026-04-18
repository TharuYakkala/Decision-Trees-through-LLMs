"""
Evaluate the 5 Gemini decision tree functions on the Credit Score dataset.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'Models with all 5 functions'))

import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
from collections import Counter
from sklearn.model_selection import train_test_split
from autogluon.tabular import TabularPredictor
from .Embeddings.emb_generator_credit import get_emb_features_credit
from .Embeddings.embedding_eval import rt_embeddings, et_embeddings, rf_embeddings, xgb_embeddings
from .model_extractors.credit_models import run_model_0, run_model_1, run_model_2, run_model_3, run_model_4, SEED
from tqdm import tqdm
import warnings

warnings.filterwarnings("ignore")

def evaluate():
    X = pd.read_csv("./data/data_sets/creditscore/X.csv")
    y = pd.read_csv("./data/data_sets/creditscore/y.csv")["target"].values
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=SEED, test_size=0.3)
    runners = [run_model_0, run_model_1, run_model_2, run_model_3, run_model_4]
    all_preds = {}

    for i, runner in enumerate(runners):
        preds = [runner(row)[0] for _, row in X_test.iterrows()]
        all_preds[f"Model_{i}"] = preds

    print("=" * 70)
    print("CREDIT SCORE DATASET - Zero-Shot Decision Tree Evaluation")
    print("=" * 70)
    print(f"Dataset size: {len(y)} | Accepted(1)={sum(y==1)}, Rejected(0)={sum(y==0)}\n")

    for name, preds in all_preds.items():
        acc, prec, rec, f1 = get_metrics(y_test, preds)
        print(f"  {name}: Accuracy={acc:.4f}  Precision={prec:.4f}  Recall={rec:.4f}  F1={f1:.4f}")

    ens = [Counter([all_preds[f"Model_{i}"][j] for i in range(5)]).most_common(1)[0][0] for j in range(len(y_test))]
    acc = accuracy_score(y_test, ens)
    prec = precision_score(y_test, ens, zero_division=0)
    rec = recall_score(y_test, ens, zero_division=0)
    f1 = f1_score(y_test, ens, zero_division=0)
    print(f"\n  Ensemble (Majority Voting): Accuracy={acc:.4f}  Precision={prec:.4f}  Recall={rec:.4f}  F1={f1:.4f}")
    print("\n" + classification_report(y_test, ens, target_names=["Rejected", "Accepted"]))
    
    ### Get two versions of embeddings, one from the LLM trees, and another for RandomTreeEmbeddings
    llm_train_emb = pd.DataFrame(get_emb_features_credit(X_train))
    llm_train_emb['target'] = y_train
    llm_test_emb = pd.DataFrame(get_emb_features_credit(X_test))
    
    all_embeddings = {
        "llm_emb": (llm_train_emb,llm_test_emb),
        "rt_emb": rt_embeddings(X_train, y_train, X_test),
        'et_emb_ss': et_embeddings(X_train, y_train, X_test, self_supervised=True),
        'et_emb_s': et_embeddings(X_train, y_train, X_test, self_supervised=False),
        'rf_emb_ss': rf_embeddings(X_train, y_train, X_test, self_supervied=True),
        'rf_emb_s': rf_embeddings(X_train, y_train, X_test, self_supervied=False),
        'xgb_emb_ss': xgb_embeddings(X_train, y_train, X_test, self_supervised=True),
        'xgb_emb_s': xgb_embeddings(X_train, y_train, X_test, self_supervised=False)
    }
    
    emb_results = []
    for emb_name, (train_emb, test_emb) in tqdm(all_embeddings.items(), desc="Testing embeddings"):
        # Autogluon
        model = TabularPredictor(label='target', verbosity=0).fit(train_emb)
        pred = model.predict(pd.DataFrame(test_emb))
        acc, prec, rec, f1 = get_metrics(y_test, pred)
        emb_results.append(
            {
                'model': 'autogluon',
                'emb_type': emb_name,
                'acc': acc,
                'prec': prec,
                'rec': rec,
                'f1': f1 
            }
        )
    emb_df = pd.DataFrame(emb_results)
    print(emb_df)
    # 

def get_metrics(y_true, y_pred):
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0, pos_label=1, average='macro')
    rec = recall_score(y_true, y_pred, zero_division=0, pos_label=1, average='macro')
    f1 = f1_score(y_true, y_pred, zero_division=0, pos_label=1, average='macro')
    return acc, prec, rec, f1

if __name__ == "__main__":
    evaluate()
