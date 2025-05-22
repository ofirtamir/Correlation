import os
import json
import numpy as np
import pandas as pd
from statsmodels.stats.outliers_influence import variance_inflation_factor

DATA_DIR = "patient_contributions_DataSet1"
methods = ["SHAP", "Lime", "Inherent"]

def compute_vif(explanations):
    df = pd.DataFrame(explanations)

    # סינון משתנים עם שונות אפס (כל הערכים זהים)
    df = df.loc[:, df.apply(lambda x: x.nunique() > 1)]

    # חייבים לפחות שני משתנים שונים לחישוב VIF
    if df.shape[1] < 2:
        return None

    vif_data = pd.DataFrame()
    vif_data["feature"] = df.columns
    vif_data["VIF"] = [variance_inflation_factor(df.values, i) for i in range(df.shape[1])]
    return vif_data.sort_values(by="VIF", ascending=False)

def process_vif_file(filepath):
    with open(filepath, 'r') as f:
        data = json.load(f)

    vif_results = {}
    for model, methods_data in data.items():
        available_methods = [m for m in methods if m in methods_data]
        if len(available_methods) < 2:
            continue

        all_features = set()
        for m in available_methods:
            all_features.update(methods_data[m].keys())

        valid_features = []
        for f in all_features:
            values = [methods_data[m].get(f, 0) for m in available_methods]
            if all(v != 0 for v in values):
                valid_features.append(f)

        if not valid_features:
            continue

        explanations = {
            f: [methods_data[m].get(f, 0) for m in available_methods]
            for f in valid_features
        }

        vif_df = compute_vif(explanations)
        if vif_df is not None:
            vif_results[model] = vif_df.set_index("feature")["VIF"].to_dict()

    return vif_results

def run_vif_analysis():
    all_vif_results = {}
    for filename in os.listdir(DATA_DIR):
        if filename.endswith(".json"):
            filepath = os.path.join(DATA_DIR, filename)
            vif_result = process_vif_file(filepath)
            all_vif_results[filename] = vif_result

    with open("vif_analysis_results.json", "w") as f:
        json.dump(all_vif_results, f, indent=2)

    print("✅ VIF analysis completed. Results saved to 'vif_analysis_results.json'.")

if __name__ == "__main__":
    run_vif_analysis()
