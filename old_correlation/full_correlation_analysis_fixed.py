
import os
import json
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, kendalltau

DATA_DIR = "../patient_contributions_DataSet1"
methods = ["SHAP", "Lime", "Inherent"]

def load_explanations(path):
    with open(path, "r") as f:
        return json.load(f)

def extract_feature_matrix(data):
    feature_dict = {}
    sign_dict = {}

    for model in data:
        for method in methods:
            if method not in data[model]:
                continue
            for feature, value in data[model][method].items():
                key = (model, feature)
                feature_dict.setdefault(key, {}).setdefault(method, []).append(value)
                sign = 1 if value > 0 else (-1 if value < 0 else 0)
                sign_dict.setdefault(key, {}).setdefault(method, []).append(sign)

    return feature_dict, sign_dict

def compute_feature_level_correlation(feature_dict, sign_dict):
    result = {}
    for (model, feature), method_values in feature_dict.items():
        entry = {}
        for m1 in methods:
            for m2 in methods:
                if m1 < m2 and m1 in method_values and m2 in method_values:
                    v1 = np.array(method_values[m1])
                    v2 = np.array(method_values[m2])
                    if len(v1) >= 2 and len(v2) >= 2:
                        r, _ = pearsonr(v1, v2)
                        entry[f"{m1}_vs_{m2}_pearson"] = r

        # sign agreement
        sign_entry = sign_dict.get((model, feature), {})
        signs = []
        for method in methods:
            if method in sign_entry:
                signs.append(np.sign(sign_entry[method][0]))  # use first example
        if len(signs) >= 2:
            agree = sum(s == signs[0] for s in signs[1:])
            entry["sign_agreement"] = agree / (len(signs) - 1) if len(signs) > 1 else 1.0

        result.setdefault(model, {})[feature] = entry
    return result

def compute_global_metrics(feature_dict):
    global_results = {
        "pearson": {},
        "kendall_w": {},
        "average_pairwise": {}
    }
    model_feature_scores = {}

    for (model, feature), method_values in feature_dict.items():
        model_feature_scores.setdefault(model, {}).setdefault(feature, {})
        for method in methods:
            values = method_values.get(method, [0])
            model_feature_scores[model][feature][method] = np.mean(values)

    for model, features in model_feature_scores.items():
        df = pd.DataFrame(features).T
        df = df.fillna(0)

        # Pearson
        pearson_corrs = {}
        for m1 in methods:
            for m2 in methods:
                if m1 < m2:
                    r, _ = pearsonr(df[m1], df[m2])
                    pearson_corrs[f"{m1}_vs_{m2}"] = r
        global_results["pearson"][model] = pearson_corrs

        # Kendall's W approximation
        kendall_scores = []
        for i in range(len(methods)):
            for j in range(i + 1, len(methods)):
                tau, _ = kendalltau(df[methods[i]], df[methods[j]])
                kendall_scores.append(tau)
        global_results["kendall_w"][model] = np.mean(kendall_scores)

        # Average pairwise correlation
        pairwise_corrs = []
        for i in range(len(methods)):
            for j in range(i + 1, len(methods)):
                r, _ = pearsonr(df[methods[i]], df[methods[j]])
                pairwise_corrs.append(r)
        global_results["average_pairwise"][model] = np.mean(pairwise_corrs)

    return global_results

def main():
    files = [f for f in os.listdir(DATA_DIR) if f.endswith(".json")]
    feature_dict = {}
    sign_dict = {}

    for file in files:
        path = os.path.join(DATA_DIR, file)
        data = load_explanations(path)
        f_dict, s_dict = extract_feature_matrix(data)
        for k, v in f_dict.items():
            feature_dict.setdefault(k, {}).update(v)
        for k, v in s_dict.items():
            sign_dict.setdefault(k, {}).update(v)

    feature_level = compute_feature_level_correlation(feature_dict, sign_dict)
    global_level = compute_global_metrics(feature_dict)

    result = {
        "feature_level": feature_level,
        "global": global_level
    }

    with open("correlation_results.json", "w") as f:
        json.dump(result, f, indent=2)

    print("Correlation analysis completed. Results saved to 'correlation_results.json'.")

if __name__ == "__main__":
    main()
