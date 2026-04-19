import re
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.metrics import f1_score
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.impute import KNNImputer
from xgboost import XGBClassifier
import warnings
from sklearn.exceptions import ConvergenceWarning

warnings.filterwarnings("ignore", category=ConvergenceWarning)
MLP_HIDDEN_STATES = [(10,), (25,), (50,), (75,), (100,)]
MLP_ALPHAS = [0.0001, 0.001, 0.01, 0.1]

MODEL_FOLDERS = {
    'gpt-oss': 'gpt_oss',
    'gemma3': 'gemma3',
    'mistral': 'mistral_small3',
    'qwen3': 'qwen3'
}


def inpute_k_neighbours(X_train, X_test, n_neighbors=10):
    imputer = KNNImputer(n_neighbors=n_neighbors)
    X_train = imputer.fit_transform(X_train)
    X_test = imputer.transform(X_test)
    return X_train, X_test


def load_tree_function(filepath):
    with open(filepath, 'r') as f:
        code = f.read()
    code = re.sub(r'```python\s*\n?', '', code)
    code = re.sub(r'```\s*\n?', '', code)
    code = code.strip()
    match = re.search(r'^def\s+(\w+)\s*\(', code, re.MULTILINE)
    func_name = match.group(1)
    local_ns = {}
    exec(code, local_ns)
    return local_ns[func_name]


def train_mlp(X_train, y_train, X_test, y_test, random_state):
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=random_state)

    mlp = MLPClassifier(random_state=random_state, max_iter=1000)

    param_grid = {
        "hidden_layer_sizes": MLP_HIDDEN_STATES,
        "alpha": MLP_ALPHAS
    }

    grid = GridSearchCV(
        estimator=mlp,
        param_grid=param_grid,
        scoring='f1_macro',
        cv=cv,
        n_jobs=-1
    )

    grid.fit(X_train, y_train)
    y_pred = grid.best_estimator_.predict(X_test)
    score = f1_score(y_test, y_pred, average="macro")
    return score


def evaluate_ind_dataset(dataset: str):
    valid_datasets = ('bankruptcy', 'boxing1', 'boxing2', 'colic', 'creditscore')
    if dataset not in valid_datasets:
        raise ValueError(f"Dataset must be one of {valid_datasets}")
    X = pd.read_csv(f"./data/data_sets/{dataset}/X.csv")
    y = pd.read_csv(f"./data/data_sets/{dataset}/y.csv").squeeze()

    results = []

    for i_split in range(5):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=i_split)
        X_train, X_test = inpute_k_neighbours(X_train, X_test, n_neighbors=10)
        X_train = pd.DataFrame(X_train, columns=X.columns)
        X_test = pd.DataFrame(X_test, columns=X.columns)

        # Get LLM Induction Predictions
        for model_name, model_folder in MODEL_FOLDERS.items():
            functions = [load_tree_function(f"./data/llm_induction/{model_folder}/{dataset}/dt_func_{i}.txt") for i in range(5)]
            preds = []
            for _, row in X_test.iterrows():
                votes = []
                for func in functions:
                    pred, _ = func(row)
                    votes.append(int(pred))
                preds.append(int(sum(votes) > len(votes) / 2))
            score = f1_score(y_test.values, preds, average='macro')
            results.append({
                'model': model_name,
                'f1-score': score
            })

        # Decision Tree
        dt = DecisionTreeClassifier(max_depth=2, random_state=i_split)
        dt.fit(X_train, y_train)
        score = f1_score(y_test.values, dt.predict(X_test), average='macro')
        results.append({'model': 'dt', 'f1-score': score})

        # Random Forest
        rf = RandomForestClassifier(n_estimators=5, max_depth=2, random_state=42)
        rf.fit(X_train, y_train)
        score = f1_score(y_test.values, rf.predict(X_test), average='macro')
        results.append({'model': 'rf', 'f1-score': score})

        # Extra Trees
        et = ExtraTreesClassifier(n_estimators=5, max_depth=2, random_state=42)
        et.fit(X_train, y_train)
        score = f1_score(y_test.values, et.predict(X_test), average='macro')
        results.append({'model': 'et', 'f1-score': score})

        # XGBoost
        xgb = XGBClassifier(n_estimators=5, max_depth=2, random_state=42, eval_metric='logloss', verbosity=0)
        xgb.fit(X_train, y_train)
        score = f1_score(y_test.values, xgb.predict(X_test), average='macro')
        results.append({'model': 'xgb', 'f1-score': score})

        # baseline
        score = train_mlp(
            X_train,
            y_train,
            X_test,
            y_test,
            random_state=i_split
        )

        results.append({
            'model': 'baseline',
            'f1-score': score
        })
    return pd.DataFrame(results)
