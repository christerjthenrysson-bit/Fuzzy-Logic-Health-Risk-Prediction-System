"""
fuzzy_model.py
COMP 3106 – Fuzzy Logic Health Risk Prediction System

Author: Christer Henrysson (Data Lead, AI Logic Designer), Tirth Raval (AI Logic Designer),

Backend engine for:
- Fuzzy membership system
- Rule base
- Fuzzy inference
- Dataset scoring
- UI-safe wrapper for Streamlit
"""

# =========================
# Imports
# =========================

import numpy as np
import pandas as pd
import skfuzzy as fuzz
from skfuzzy import control as ctrl
from sklearn.metrics import f1_score

# Optional sklearn evaluation
try:
    from sklearn.metrics import (
        r2_score,
        mean_absolute_error,
        accuracy_score,
        confusion_matrix,
    )
    SKLEARN_AVAILABLE = True
except:
    SKLEARN_AVAILABLE = False


# =========================
# Dataset configuration
# =========================

DATA_PATH = "cleaned_risk_data.csv"

BMI_COL       = "BMI"
BP_COL        = "BLOOD PRESSURE"
EXERCISE_COL  = "ABDOMINAL CIRCUMFERENCE"
SMOKING_COL   = "SMOKING"
DIET_COL      = "TOTAL CHOLESTEROL"
TARGET_COL    = "CVD RISK"


# =========================
# Fuzzy Antecedents & Consequent
# =========================

bmi        = ctrl.Antecedent(np.arange(15, 40.1, 0.1), "bmi")
bp         = ctrl.Antecedent(np.arange(90, 200.1, 1), "bp")
waist      = ctrl.Antecedent(np.arange(70, 120.1, 1), "waist")
smoking    = ctrl.Antecedent(np.arange(0, 1.1, 0.1), "smoking")
chol       = ctrl.Antecedent(np.arange(120, 340.1, 1), "cholesterol")

risk       = ctrl.Consequent(np.arange(0, 100.1, 1), "risk")


# =========================
# Membership Functions
# =========================

# BMI
bmi["under"]  = fuzz.trimf(bmi.universe, [15, 15, 18.5])
bmi["normal"] = fuzz.trimf(bmi.universe, [18, 23, 27])
bmi["over"]   = fuzz.trimf(bmi.universe, [25, 34, 40])

# Blood pressure
bp["low"]     = fuzz.trimf(bp.universe, [90, 90, 110])
bp["normal"]  = fuzz.trimf(bp.universe, [105, 120, 135])
bp["high"]    = fuzz.trimf(bp.universe, [130, 170, 200])

# Waist
waist["small"]  = fuzz.trimf(waist.universe, [70, 70, 85])
waist["medium"] = fuzz.trimf(waist.universe, [80, 95, 105])
waist["large"]  = fuzz.trimf(waist.universe, [100, 120, 120])

# Smoking
smoking["non"] = fuzz.trimf(smoking.universe, [0, 0, 0.3])
smoking["yes"] = fuzz.trimf(smoking.universe, [0.7, 1, 1])

# Cholesterol
chol["good"] = fuzz.trimf(chol.universe, [120, 120, 190])
chol["avg"]  = fuzz.trimf(chol.universe, [180, 220, 260])
chol["poor"] = fuzz.trimf(chol.universe, [240, 340, 340])

# Risk
risk["low"]  = fuzz.trimf(risk.universe, [0, 0, 40])
risk["med"]  = fuzz.trimf(risk.universe, [30, 50, 70])
risk["high"] = fuzz.trimf(risk.universe, [60, 100, 100])


# =========================
# Rules
# =========================

rules: list[ctrl.Rule] = []

# High risk rules
rules += [
    ctrl.Rule(bmi["over"] & bp["high"], risk["high"]),
    ctrl.Rule(bmi["over"] & chol["poor"], risk["high"]),
    ctrl.Rule(bmi["over"] & waist["large"], risk["high"]),
    ctrl.Rule(chol["poor"] & smoking["yes"], risk["high"]),
    ctrl.Rule(bp["high"] & waist["large"], risk["high"]),
    ctrl.Rule(bp["high"] & chol["poor"], risk["high"]),
    ctrl.Rule(bp["high"] & smoking["yes"], risk["high"]),
    ctrl.Rule(waist["large"] & smoking["yes"], risk["high"]),
    ctrl.Rule(waist["large"] & chol["poor"], risk["high"]),
    ctrl.Rule(bmi["over"] & bp["high"] & smoking["yes"], risk["high"]),
    ctrl.Rule(bmi["over"] & bp["high"] & chol["poor"], risk["high"]),
    ctrl.Rule(chol["poor"] & waist["medium"] & smoking["yes"], risk["high"]),
]

# Medium risk
rules += [
    ctrl.Rule(bmi["normal"] & bp["high"], risk["med"]),
    ctrl.Rule(bmi["over"] & bp["normal"] & waist["medium"], risk["med"]),
    ctrl.Rule(bmi["over"] & chol["avg"] & waist["medium"], risk["med"]),
    ctrl.Rule(bp["normal"] & smoking["yes"], risk["med"]),
    ctrl.Rule(chol["avg"] & waist["large"], risk["med"]),
    ctrl.Rule(chol["avg"] & waist["medium"], risk["med"]),
    ctrl.Rule(bmi["under"] & chol["poor"], risk["med"]),
    ctrl.Rule(bmi["normal"] & waist["medium"] & chol["avg"], risk["med"]),
    ctrl.Rule(waist["medium"] & smoking["yes"], risk["med"]),
    ctrl.Rule(bp["low"] & bmi["over"], risk["med"]),
    ctrl.Rule(waist["small"] & chol["poor"], risk["med"]),
    ctrl.Rule(waist["large"] & chol["good"], risk["med"]),
]

# Low risk
rules += [
    ctrl.Rule(bmi["normal"] & bp["normal"] & waist["small"] & smoking["non"] & chol["good"], risk["low"]),
    ctrl.Rule(waist["small"] & chol["good"] & smoking["non"], risk["low"]),
    ctrl.Rule(waist["small"] & bp["normal"], risk["low"]),
    ctrl.Rule(bmi["normal"] & chol["good"] & smoking["non"], risk["low"]),
    ctrl.Rule(bmi["under"] & waist["small"] & chol["good"], risk["low"]),
    ctrl.Rule(bmi["under"] & bp["normal"] & smoking["non"], risk["low"]),
    ctrl.Rule(bp["low"] & waist["small"], risk["low"]),
    ctrl.Rule(chol["good"] & bp["normal"], risk["low"]),
]

risk_ctrl = ctrl.ControlSystem(rules)


# =========================
# SAFE FUZZY COMPUTATION 
# =========================

def predict_risk(bmi_val, bp_val, waist_val, smoking_val, chol_val, verbose=False):
    """
    Safe Mamdani fuzzy inference.
    Prevents KeyError('risk') and fallback issues.
    """

    sim = ctrl.ControlSystemSimulation(risk_ctrl)

    sim.input["bmi"]         = bmi_val
    sim.input["bp"]          = bp_val
    sim.input["waist"]       = waist_val
    sim.input["smoking"]     = smoking_val
    sim.input["cholesterol"] = chol_val

    try:
        sim.compute()

        # FIX 1: Prevent KeyError('risk')
        if "risk" not in sim.output:
            raise KeyError("risk missing")

        score = float(sim.output["risk"])

    except Exception as e:
        print(f"[WARNING] Fuzzy computation fallback: {e}")
        score = 50.0  # Safe midpoint fallback

    # Categorization
    if score < 49:   category = "Low"
    elif score < 68: category = "Medium"
    else:            category = "High"

    return score, category


# =========================
# Apply model to entire dataset
# =========================

def apply_model_to_dataset(save_path="cleaned_risk_with_fuzzy.csv"):
    df = pd.read_csv(DATA_PATH)

    scores = []
    cats   = []

    for _, r in df.iterrows():
        s, c = predict_risk(
            float(r[BMI_COL]),
            float(r[BP_COL]),
            float(r[EXERCISE_COL]),
            float(r[SMOKING_COL]),
            float(r[DIET_COL])
        )
        scores.append(s)
        cats.append(c)

    df["FuzzyRiskScore"] = scores
    df["FuzzyRiskLevel"] = cats

    df.to_csv(save_path, index=False)
    return df


# =========================
# UI wrapper for Streamlit
# =========================

def predict_for_ui(input_dict: dict) -> dict:
    score, category = predict_risk(
        float(input_dict[BMI_COL]),
        float(input_dict[BP_COL]),
        float(input_dict[EXERCISE_COL]),
        float(input_dict[SMOKING_COL]),
        float(input_dict[DIET_COL]),
    )

    return {"score": score, "category": category}


# =========================
# Console pipeline tester
# =========================

def run_full_pipeline():
    df_raw = pd.read_csv(DATA_PATH)
    df_scored = apply_model_to_dataset()
    print(df_scored.head())

    # =========================
    # Evaluation Metrics (Console Only)
    # =========================
    if SKLEARN_AVAILABLE and TARGET_COL in df_scored.columns:
        y_true = df_scored[TARGET_COL].astype(float).values
        y_pred = df_scored["FuzzyRiskScore"].astype(float).values


        # --- Pearson correlation ---
        corr = np.corrcoef(y_true, y_pred)[0, 1]
        print(f"Pearson correlation: {corr:.3f}")

        # --- Convert fuzzy score --> class (0=Low,1=Med,2=High) ---
        def score_to_class(s):
            if s < 49:
                return 0
            elif s < 68:
                return 1
            return 2

        y_pred_class = np.array([score_to_class(s) for s in y_pred], dtype=int)
        y_true_class = y_true.astype(int)

        # --- Accuracy ---
        acc = accuracy_score(y_true_class, y_pred_class)
        print(f"Accuracy (3-class): {acc:.3f}")

        # --- F1 Scores ---
        macro_f1 = f1_score(y_true_class, y_pred_class, average="macro")
        weighted_f1 = f1_score(y_true_class, y_pred_class, average="weighted")
        per_class_f1 = f1_score(y_true_class, y_pred_class, average=None, labels=[0,1,2])

        print(f"F1 Score (macro): {macro_f1:.3f}")
        print(f"F1 Score (weighted): {weighted_f1:.3f}")
        print(f"F1 per class [Low, Medium, High]: {per_class_f1}")

        # --- Confusion matrix ---
        cm = confusion_matrix(y_true_class, y_pred_class, labels=[0,1,2])
        print("\nConfusion Matrix (rows=true, cols=pred):")
        print("Labels: 0=Low, 1=Medium, 2=High")
        print(cm)



    else:
        print("\n[INFO] sklearn not available or target column missing. Metrics skipped.")



if __name__ == "__main__":
    run_full_pipeline()
