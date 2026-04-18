
from ..creditscore import dt_function_0, dt_function_1, dt_function_2, dt_function_3, dt_function_4

OWN_HOME_MAP = {1: "yes", 0: "no"}
SELF_EMP_MAP = {1: "yes", 0: "no"}

# Output → binary
ACCEPT_MAP = {"Accepted": 1, "accepted": 1, "Rejected": 0, "rejected": 0, "Denied": 0}
SEED = 42

def run_model_0(row):
    pred, emb = dt_function_0(
        row["Age"], row["Income.per.dependent"], row["Monthly.credit.card.exp"],
        OWN_HOME_MAP[int(row["Own.home"])], SELF_EMP_MAP[int(row["Self.employed"])],
        int(row["Derogatory.reports"])
    )
    return ACCEPT_MAP[pred], emb

def run_model_1(row):
    features = {
        'age (years)': row["Age"],
        'income per dependent (1.5 to 10)': row["Income.per.dependent"],
        'monthly credit card expenses ($)': row["Monthly.credit.card.exp"],
        'owning a home (yes / no)': OWN_HOME_MAP[int(row["Own.home"])],
        'self employed (yes / no)': SELF_EMP_MAP[int(row["Self.employed"])],
        'number of derogatory reports': int(row["Derogatory.reports"]),
    }
    pred, emb = dt_function_1(features)
    return ACCEPT_MAP[pred], emb

def run_model_2(row):
    pred, emb = dt_function_2(
        row["Age"], row["Income.per.dependent"], row["Monthly.credit.card.exp"],
        OWN_HOME_MAP[int(row["Own.home"])], SELF_EMP_MAP[int(row["Self.employed"])],
        int(row["Derogatory.reports"])
    )
    return ACCEPT_MAP[pred], emb

def run_model_3(row):
    pred, emb = dt_function_3(
        row["Age"], row["Income.per.dependent"], row["Monthly.credit.card.exp"],
        OWN_HOME_MAP[int(row["Own.home"])], SELF_EMP_MAP[int(row["Self.employed"])],
        int(row["Derogatory.reports"])
    )
    return ACCEPT_MAP[pred], emb

def run_model_4(row):
    pred, emb = dt_function_4(
        row["Age"], row["Income.per.dependent"], row["Monthly.credit.card.exp"],
        OWN_HOME_MAP[int(row["Own.home"])], SELF_EMP_MAP[int(row["Self.employed"])],
        int(row["Derogatory.reports"])
    )
    return ACCEPT_MAP[pred], emb