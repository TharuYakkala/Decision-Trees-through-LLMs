from ..boxing2 import dt_function_0, dt_function_1, dt_function_2, dt_function_3, dt_function_4

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
    pred, emb = dt_function_0(j, o, r)
    return TARGET_MAP[pred], emb

def run_model_1(row):
    j, o, r = prep(row)
    features = {'Judge': j, 'Offical judge': o, 'Round': r}
    pred, emb = dt_function_1(features)
    return TARGET_MAP[pred], emb

def run_model_2(row):
    j, o, r = prep(row)
    pred, emb = dt_function_2(j, o, r)
    return TARGET_MAP[pred], emb

def run_model_3(row):
    j, o, r = prep(row)
    pred, emb = dt_function_3(j, o, r)
    return TARGET_MAP[pred], emb

def run_model_4(row):
    j, o, r = prep(row)
    pred, emb = dt_function_4(j, o, r)
    return TARGET_MAP[pred], emb