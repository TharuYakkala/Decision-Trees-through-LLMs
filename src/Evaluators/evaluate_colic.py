"""
Evaluate the 5 Gemini decision tree functions on the Colic (Horse) dataset.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'Models with all 5 functions'))

import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
from collections import Counter
from .colic import dt_function_0, dt_function_1, dt_function_2, dt_function_3, dt_function_4

# Feature encodings
PAIN_MAP = {1: "alert", 2: "continuous severe pain", 3: "depressed",
            4: "intermittent mild pain", 5: "intermittent severe pain"}
ABDOMEN_MAP = {1: "distended large intestine", 2: "distended small intestine",
               3: "firm feces in large intestine", 4: "normal", 5: "other"}
REFLUX_MAP = {0: ">1 liter", 1: "<1 liter", 2: "missing", 3: "none"}
ABDOM_APPEAR_MAP = {1: "clear", 2: "cloudy", 3: "serosanguinous"}
SURGERY_MAP = {1: "no", 2: "yes"}
AGE_MAP = {0: "adult", 1: "young"}
TEMP_EXT_MAP = {1: "cold", 2: "cool", 3: "normal", 4: "warm"}
PERIPH_PULSE_MAP = {1: "absent", 2: "increased", 3: "normal", 4: "reduced"}
MUCOUS_MAP = {1: "bright pink", 2: "bright red", 3: "dark cyanotic", 4: "normal pink", 5: "pale cyanotic", 6: "pale pink"}
CAP_REFILL_MAP = {0: "more than 3 seconds", 1: "less than 3 seconds", 2: "missing", 3: "meaning unknown"}
PERISTALSIS_MAP = {1: "absent", 2: "hypermotile", 3: "hypomotile", 4: "normal"}
ABDOM_DIST_MAP = {1: "moderate", 2: "none", 3: "severe", 4: "slight"}
NASO_TUBE_MAP = {1: "none", 2: "significant", 3: "slight"}
RECTAL_EXAM_MAP = {1: "absent", 2: "decreased", 3: "increased", 4: "normal"}
OUTCOME_MAP = {1: "died", 2: "euthanized", 3: "lived"}

# Output → binary
SURGICAL_MAP = {"surgical": 1, "yes": 1, "not surgical": 0, "non-surgical": 0, "no": 0}

def safe_map(val, mapping, default=None):
    if pd.isna(val):
        return default
    return mapping.get(int(val), default)


# Model 0: takes (Abdomen_Appearance, Degree_of_Pain)
def run_model_0(row):
    abdomen = safe_map(row["abdomen"], ABDOMEN_MAP, "normal")
    pain = safe_map(row["pain"], PAIN_MAP, "alert")
    # The function expects 'severe' but data has 'continuous severe pain' / 'intermittent severe pain'
    # Map to what the function checks
    pain_simple = "severe" if "severe" in str(pain) else pain
    pred, _ = dt_function_0(abdomen, pain_simple)
    return SURGICAL_MAP[pred]

# Model 1: takes dict with 'Abdominocentesis Appearance' and 'Nasogastric Reflux'
def run_model_1(row):
    features_dict = {
        'Abdominocentesis Appearance': safe_map(row["abdominocentesis_appearance"], ABDOM_APPEAR_MAP, "clear"),
        'Nasogastric Reflux': safe_map(row["nasogastric_reflux"], REFLUX_MAP, "none"),
    }
    pred, _ = dt_function_1(features_dict)
    return SURGICAL_MAP[pred]

# Model 2: takes all 22 features as individual args
def run_model_2(row):
    pain = safe_map(row["pain"], PAIN_MAP, "alert")
    pain_simple = "severe" if "severe" in str(pain) else pain
    args = (
        safe_map(row["surgery"], SURGERY_MAP, "no"),
        safe_map(row["Age"], AGE_MAP, "adult"),
        row["rectal_temperature"] if pd.notna(row["rectal_temperature"]) else 38.0,
        row["pulse"] if pd.notna(row["pulse"]) else 40,
        row["respiratory_rate"] if pd.notna(row["respiratory_rate"]) else 20,
        safe_map(row["temp_extremities"], TEMP_EXT_MAP, "normal"),
        safe_map(row["peripheral_pulse"], PERIPH_PULSE_MAP, "normal"),
        safe_map(row["mucous_membranes"], MUCOUS_MAP, "normal pink"),
        safe_map(row["capillary_refill_time"], CAP_REFILL_MAP, "less than 3 seconds"),
        pain_simple,
        safe_map(row["peristalsis"], PERISTALSIS_MAP, "normal"),
        safe_map(row["abdominal_distension"], ABDOM_DIST_MAP, "none"),
        safe_map(row["nasogastric_tube"], NASO_TUBE_MAP, "none"),
        safe_map(row["nasogastric_reflux"], REFLUX_MAP, "none"),
        row["nasogastric_reflux_PH"] if pd.notna(row["nasogastric_reflux_PH"]) else 7.0,
        safe_map(row["rectal_examination"], RECTAL_EXAM_MAP, "normal"),
        safe_map(row["abdomen"], ABDOMEN_MAP, "normal"),
        row["packed_cell_volume"] if pd.notna(row["packed_cell_volume"]) else 40.0,
        row["total_protein"] if pd.notna(row["total_protein"]) else 6.5,
        safe_map(row["abdominocentesis_appearance"], ABDOM_APPEAR_MAP, "clear"),
        row["abdomcentesis_total_protein"] if pd.notna(row["abdomcentesis_total_protein"]) else 2.0,
        safe_map(row["outcome"], OUTCOME_MAP, "lived"),
    )
    pred, _ = dt_function_2(*args)
    return SURGICAL_MAP[pred]

# Model 3: takes dict with 'Degree of Pain' and 'Abdomen Appearance'
def run_model_3(row):
    pain = safe_map(row["pain"], PAIN_MAP, "alert")
    pain_simple = "severe" if "severe" in str(pain) else pain
    features = {
        'Degree of Pain': pain_simple,
        'Abdomen Appearance': safe_map(row["abdomen"], ABDOMEN_MAP, "normal"),
    }
    pred, _ = dt_function_3(features)
    return SURGICAL_MAP[pred]

# Model 4: takes all 22 features as individual args
def run_model_4(row):
    pain = safe_map(row["pain"], PAIN_MAP, "alert")
    pain_simple = "severe" if "severe" in str(pain) else pain
    args = (
        safe_map(row["surgery"], SURGERY_MAP, "no"),
        safe_map(row["Age"], AGE_MAP, "adult"),
        row["rectal_temperature"] if pd.notna(row["rectal_temperature"]) else 38.0,
        row["pulse"] if pd.notna(row["pulse"]) else 40,
        row["respiratory_rate"] if pd.notna(row["respiratory_rate"]) else 20,
        safe_map(row["temp_extremities"], TEMP_EXT_MAP, "normal"),
        safe_map(row["peripheral_pulse"], PERIPH_PULSE_MAP, "normal"),
        safe_map(row["mucous_membranes"], MUCOUS_MAP, "normal pink"),
        safe_map(row["capillary_refill_time"], CAP_REFILL_MAP, "less than 3 seconds"),
        pain_simple,
        safe_map(row["peristalsis"], PERISTALSIS_MAP, "normal"),
        safe_map(row["abdominal_distension"], ABDOM_DIST_MAP, "none"),
        safe_map(row["nasogastric_tube"], NASO_TUBE_MAP, "none"),
        safe_map(row["nasogastric_reflux"], REFLUX_MAP, "none"),
        row["nasogastric_reflux_PH"] if pd.notna(row["nasogastric_reflux_PH"]) else 7.0,
        safe_map(row["rectal_examination"], RECTAL_EXAM_MAP, "normal"),
        safe_map(row["abdomen"], ABDOMEN_MAP, "normal"),
        row["packed_cell_volume"] if pd.notna(row["packed_cell_volume"]) else 40.0,
        row["total_protein"] if pd.notna(row["total_protein"]) else 6.5,
        safe_map(row["abdominocentesis_appearance"], ABDOM_APPEAR_MAP, "clear"),
        row["abdomcentesis_total_protein"] if pd.notna(row["abdomcentesis_total_protein"]) else 2.0,
        safe_map(row["outcome"], OUTCOME_MAP, "lived"),
    )
    pred, _ = dt_function_4(*args)
    return SURGICAL_MAP[pred]


def evaluate():
    X = pd.read_csv("./data/data_sets/colic/X.csv")
    y = pd.read_csv("./data/data_sets/colic/y.csv")["target"].values

    runners = [run_model_0, run_model_1, run_model_2, run_model_3, run_model_4]
    all_preds = {}

    for i, runner in enumerate(runners):
        preds = [runner(row) for _, row in X.iterrows()]
        all_preds[f"Model_{i}"] = preds

    print("=" * 70)
    print("COLIC (Horse) DATASET - Zero-Shot Decision Tree Evaluation")
    print("=" * 70)
    print(f"Dataset size: {len(y)} | Surgical(1)={sum(y==1)}, Non-Surgical(0)={sum(y==0)}\n")

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
    print("\n" + classification_report(y, ens, target_names=["Non-Surgical", "Surgical"]))


if __name__ == "__main__":
    evaluate()
