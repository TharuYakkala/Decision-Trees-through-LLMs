from ..boxing1 import dt_function_0, dt_function_1, dt_function_2, dt_function_3, dt_function_4


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
    pred, emb = dt_function_0(j, o, r)
    return TARGET_MAP[pred], emb

def run_model_1(row):
    j, o, r = prep(row)
    pred, emb = dt_function_1(j, o, r)
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