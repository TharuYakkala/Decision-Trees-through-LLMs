from tqdm import tqdm
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from autogluon.tabular import TabularPredictor
from tabpfn import TabPFNClassifier
from autoprognosis.studies.classifiers import ClassifierStudy
import pandas as pd
import dotenv
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold
dotenv.load_dotenv()

def get_metrics(y_true, y_pred):
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0, pos_label=1, average='macro')
    rec = recall_score(y_true, y_pred, zero_division=0, pos_label=1, average='macro')
    f1 = f1_score(y_true, y_pred, zero_division=0, pos_label=1, average='macro')
    return acc, prec, rec, f1

def ml_looper(embeddings: dict, y_test):
    emb_results = []
    for emb_name, (train_emb, test_emb) in tqdm(embeddings.items(), desc="Testing embeddings", position=0):
        models = ["autogluon", "mlp", "tabpfn"]
        
        for model_name in tqdm(models, desc="Training each model", position=1, leave=False):
            match model_name:
                case 'autogluon':
                    # Autogluon
                    model = TabularPredictor(label='target', verbosity=0, eval_metric='f1_macro').fit(train_emb)
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
                case "autoprognosis": # This has been removed since it was too slow
                    # autoprognosis
                    study = ClassifierStudy(
                        train_emb,
                        target='target',
                        num_study_iter=2,
                        score_threshold=0.3,
                        metric="f1_score_macro"
                    )
                    model = study.fit()
                    pred = model.predict(test_emb)
                    acc, prec, rec, f1 = get_metrics(y_test, pred)
                    emb_results.append(
                        {
                            'model': 'autoprognosis',
                            'emb_type': emb_name,
                            'acc': acc,
                            'prec': prec,
                            'rec': rec,
                            'f1': f1 
                        }
                    )
                case "mlp":
                    grid = GridSearchCV(
                        estimator=MLPClassifier(),
                        param_grid={
                            "hidden_layer_sizes": [(10,), (25,), (50,), (75,), (100,)],
                            "alpha": [0.001, 0.001, 0.01, 0.1, 1.0]
                        },
                        scoring="f1_macro",
                        cv=StratifiedKFold(n_splits=3, shuffle=True, random_state=42),
                        n_jobs=-1
                    )
                    grid.fit(train_emb.drop(columns='target'), train_emb['target'])
                    pred = grid.predict(test_emb)
                    acc, prec, rec, f1 = get_metrics(y_test, pred)
                    emb_results.append(
                        {
                            'model': 'mlp',
                            'emb_type': emb_name,
                            'acc': acc,
                            'prec': prec,
                            'rec': rec,
                            'f1': f1 
                        }
                    )
                case "tabpfn":
                    # tabpfn
                    model = TabPFNClassifier(n_estimators=32)
                    model.fit(train_emb.drop(columns='target'), train_emb['target'])
                    pred = model.predict(test_emb)
                    acc, prec, rec, f1 = get_metrics(y_test, pred)
                    emb_results.append(
                        {
                            'model': 'tabpfn',
                            'emb_type': emb_name,
                            'acc': acc,
                            'prec': prec,
                            'rec': rec,
                            'f1': f1 
                        }
                    )
    return pd.DataFrame(emb_results)