from ..bankruptcy import dt_function_0, dt_function_1, dt_function_2, dt_function_3, dt_function_4


# Output label mapping → binary target
def to_binary(pred):
    if isinstance(pred, (int, float)):
        return int(pred)
    s = str(pred).lower()
    if "non" in s or "not" in s:
        return 0
    return 1  # "Bankrupt", "BANKRUPT", etc.

def run_model_0(row):
    pred, emb = dt_function_0(row["WC/TA"], row["RE/TA"], row["EBIT/TA"], row["S/TA"], row["BVE/BVL"])
    return to_binary(pred), emb

def run_model_1(row):
    features = {
        'Working Capital/Total Assets': row["WC/TA"],
        'Retained Earnings/Total Assets': row["RE/TA"],
        'Earnings Before Interest and Tax/Total Assets': row["EBIT/TA"],
        'Sales/Total Assets': row["S/TA"],
        'Book Value of Equity/Book Value of Liabilities': row["BVE/BVL"],
    }
    pred, emb = dt_function_1(features)
    return to_binary(pred), emb

def run_model_2(row):
    pred, emb = dt_function_2(row["WC/TA"], row["RE/TA"], row["EBIT/TA"], row["S/TA"], row["BVE/BVL"])
    return to_binary(pred), emb

def run_model_3(row):
    pred, emb = dt_function_3(row["WC/TA"], row["RE/TA"], row["EBIT/TA"], row["S/TA"], row["BVE/BVL"])
    return to_binary(pred), emb

def run_model_4(row):
    pred, emb = dt_function_4(row["WC/TA"], row["RE/TA"], row["EBIT/TA"], row["S/TA"], row["BVE/BVL"])
    return to_binary(pred), emb